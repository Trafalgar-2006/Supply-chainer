"""How close is the delay model to the best possible one?

The training data comes from a known generator (ml/generate_delay_dataset.py),
so the Bayes-optimal quantile of every leg can be computed exactly: the delay is
a mixture over the hidden incident state (none, false alarm, missed incident, or
a reported incident of severity s), weighted by how well each state explains the
leg's observed news score. No model can beat its pinball loss on average.

On a fresh sample this compares the pinball loss of the naive per-group
baseline, the pinned model and that optimum, and reports the share of the
achievable improvement the model captures.

Usage: python ml/delay_ceiling.py   (writes Execution/delay_ceiling.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ml.generate_delay_dataset as g  # noqa: E402
from backend.engine.delay_features import QUANTILES, encode  # noqa: E402
from backend.engine.delay_model import DelayQuantileModel  # noqa: E402

OUT = ROOT / "Execution" / "delay_ceiling.json"
N_LEGS = 20_000
SAMPLE_SEED = 2026                             # a different draw from the training data
SEVERITY_GRID = np.linspace(0.3, 1.0, 141)     # incident severity is U(0.3, 1)
CLIPPED = 0.9999                               # news scores are clipped at 1


def optimal_quantiles(legs, quantiles):
    params = legs["mode"].map(g.MODE_PARAMS)
    base = (params.map(lambda p: p["dwell"]) * legs["arrival"].map(g.ARRIVAL_DWELL_SHARE)
            + g.TRANSIT_VARIABILITY * legs["distance_km"] / params.map(lambda p: p["speed"])).to_numpy()
    condition = legs["condition"].to_numpy()
    weather = np.where(condition == "rainy", g.RAIN_MULT, 1.0)
    weather = np.where(condition == "stormy", legs["mode"].map(g.STORM_MULT).to_numpy(), weather)
    sigma = params.map(lambda p: p["sigma"]).to_numpy()[:, None]
    news = legs["nlp_score"].to_numpy()

    # Likelihood of the observed news score under each hidden state.
    quiet = np.where(news < CLIPPED, 2 * norm.pdf(news / 0.05) / 0.05, 0.0)      # |N(0, 0.05)|
    false_alarm = np.where((news >= 0.2) & (news <= 0.6), 2.5, 0.0)               # U(0.2, 0.6)
    s = SEVERITY_GRID[None, :]
    reported = np.where(news[:, None] >= CLIPPED, 1 - norm.cdf((1 - s) / 0.1),
                        norm.pdf((news[:, None] - s) / 0.1) / 0.1)                # N(s, 0.1)
    no_incident = (1 - g.INCIDENT_PROB) * ((1 - g.FALSE_ALARM_PROB) * quiet + g.FALSE_ALARM_PROB * false_alarm)
    incident = g.INCIDENT_PROB / len(SEVERITY_GRID) * (
        g.MISSED_INCIDENT_PROB * quiet[:, None] + (1 - g.MISSED_INCIDENT_PROB) * reported)
    weights = np.column_stack([no_incident, incident])
    weights /= weights.sum(axis=1, keepdims=True)
    log_mean = (np.log(base * weather)[:, None]
                + np.log(np.concatenate([[1.0], 1 + g.INCIDENT_IMPACT * SEVERITY_GRID]))[None, :])

    result = {}
    for q in quantiles:
        lo, hi = np.full(len(legs), -5.0), np.full(len(legs), 12.0)   # log hours
        for _ in range(60):                                          # bisection on the mixture CDF
            mid = (lo + hi) / 2
            below = (weights * norm.cdf((mid[:, None] - log_mean) / sigma)).sum(axis=1) < q
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        result[q] = np.exp((lo + hi) / 2)
    return result


def main():
    train = g.generate(60_000, seed=g.SEED)
    legs = g.generate(N_LEGS, seed=SAMPLE_SEED)
    y = legs["delay_hours"].to_numpy()
    model = DelayQuantileModel.load()
    predicted = model._predict_matrix(encode(legs["mode"], legs["distance_km"], legs["arrival"],
                                             legs["condition"], legs["nlp_score"]))
    optimum = optimal_quantiles(legs, QUANTILES)
    report = {"legs": N_LEGS, "quantiles": {}}
    for i, q in enumerate(QUANTILES):
        naive = train.groupby(["mode", "arrival"])["delay_hours"].quantile(q)
        naive_pred = legs.set_index(["mode", "arrival"]).index.map(naive).to_numpy(dtype=float)
        losses = {name: mean_pinball_loss(y, p, alpha=q)
                  for name, p in (("naive", naive_pred), ("model", predicted[:, i]), ("optimal", optimum[q]))}
        report["quantiles"][f"p{int(q * 100)}"] = {
            **{k: round(v, 3) for k, v in losses.items()},
            "share_of_achievable_gain": round((losses["naive"] - losses["model"]) / (losses["naive"] - losses["optimal"]), 3),
        }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
