"""Evaluate the threat-intelligence stage on labelled headlines.

`ml/nlp_headlines.csv` holds 288 synthetic headlines written for this project:
about half describe a disruption (with its type, severity 1-3 and the transport
modes it concerns), the rest are routine or positive logistics news, including
hard cases such as "Union and port employers strike a deal". Its three splits
(dev, test, holdout) were used in turn to build the engine and now all train
it, so their scores are in-sample.

`ml/nlp_real_headlines.csv` (split `real`) is the honest test: 188 real
headlines fetched from Google News on 29 Sep 2026, labelled before the engine
was run on them, and never read by the engine. Where the cause and the effect
differ ("typhoon deepens port congestion") both types are listed
("congestion|weather") and either counts.

Reports, per split:
- detection: ROC AUC of the threat margin, and recall, false-alarm rate and
  precision at the engine's operating point (score > 0)
- threat type: accuracy on the disruptions
- CARF: for every disruption and mode, whether the filter keeps the threat
  exactly when the news concerns that mode (or names no mode at all)

Usage: python ml/evaluate_nlp.py   (writes Execution/nlp_evaluation.json)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.engine.threat_intelligence import CARFFilter, ContrastiveNLPEngine  # noqa: E402

DATA = ROOT / "ml" / "nlp_headlines.csv"
REAL = ROOT / "ml" / "nlp_real_headlines.csv"
OUT = ROOT / "Execution" / "nlp_evaluation.json"
MODES = ("sea", "air", "rail", "road")
SPLITS = ("dev", "test", "holdout", "real")
# Real-news scores of the engine before any of this work (all-MiniLM-L6-v2 and
# the original anchors, at commit e550c5f), kept for comparison.
BASELINE_REAL = {"auc": 0.859, "recall": 0.681, "false_alarm_rate": 0.165, "precision": 0.795,
                 "type_accuracy": 0.758, "severity_rank_correlation": 0.242, "carf_accuracy": 0.882}


def load():
    with open(DATA, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(REAL, newline="", encoding="utf-8") as f:
        rows += [{**r, "split": "real"} for r in csv.DictReader(f)]
    return rows


def evaluate(rows, nlp, carf):
    scored = [(r, nlp.assess(r["headline"])) for r in rows]
    disrupted = [(r, a) for r, a in scored if r["label"] == "disrupted"]
    safe = [(r, a) for r, a in scored if r["label"] == "safe"]
    detected = [r for r, a in disrupted if a["score"] > 0]
    false_alarms = [r["headline"] for r, a in safe if a["score"] > 0]
    missed = [r["headline"] for r, a in disrupted if a["score"] == 0]

    per_type = defaultdict(lambda: [0, 0])
    wrong_type = []
    for r, a in disrupted:
        per_type[r["threat_type"]][1] += 1
        if a["type"] in r["threat_type"].split("|"):
            per_type[r["threat_type"]][0] += 1
        else:
            wrong_type.append(f'{r["headline"]} ({r["threat_type"]} read as {a["type"]})')

    carf_right, wrongly_dropped, wrongly_kept = 0, [], []
    for r, _ in disrupted:
        concerns = set(MODES) if r["modes"] == "any" else set(r["modes"].split("+"))
        for mode in MODES:
            kept = carf.apply_filter(1.0, r["headline"], mode) > 0
            if kept == (mode in concerns):
                carf_right += 1
            elif kept:
                wrongly_kept.append(f'{r["headline"]} ({mode})')
            else:
                wrongly_dropped.append(f'{r["headline"]} ({mode})')

    labels = [r["label"] == "disrupted" for r, _ in scored]
    tp = len(detected)
    return {
        "headlines": len(scored), "disrupted": len(disrupted), "safe": len(safe),
        "detection": {
            "auc": round(roc_auc_score(labels, [a["margin"] for _, a in scored]), 3),
            "recall": round(tp / len(disrupted), 3),
            "false_alarm_rate": round(len(false_alarms) / len(safe), 3),
            "precision": round(tp / (tp + len(false_alarms)), 3) if tp + len(false_alarms) else None,
        },
        "type_accuracy": round(sum(c for c, _ in per_type.values()) / len(disrupted), 3),
        # How well the predicted severity ranks the labelled 1-3 severities.
        "severity_rank_correlation": round(float(spearmanr([a["severity"] for _, a in disrupted],
                                                           [int(r["severity"]) for r, _ in disrupted])[0]), 3),
        "type_accuracy_by_type": {t: round(c / n, 2) for t, (c, n) in sorted(per_type.items())},
        "carf_accuracy": round(carf_right / (len(disrupted) * len(MODES)), 3),
        "errors": {"missed": missed, "false_alarms": false_alarms, "wrong_type": wrong_type,
                   "carf_wrongly_dropped": wrongly_dropped, "carf_wrongly_kept": wrongly_kept},
    }


def main():
    nlp, carf = ContrastiveNLPEngine(), CARFFilter()
    if not nlp.ready:
        sys.exit("The sentence-transformer model could not be loaded.")
    rows = load()
    report = {split: evaluate([r for r in rows if r["split"] == split], nlp, carf) for split in SPLITS}
    report["baseline_real"] = BASELINE_REAL
    for split in SPLITS:
        result = report[split]
        d = result["detection"]
        print(f"{split}: AUC {d['auc']}, recall {d['recall']}, false alarms {d['false_alarm_rate']}, "
              f"precision {d['precision']}, type accuracy {result['type_accuracy']}, "
              f"severity rank {result['severity_rank_correlation']}, CARF {result['carf_accuracy']}")
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    main()
