import pytest

from backend.engine.threat_intelligence import MINOR_SCORE

CATASTROPHIC = [
    "Container ship ran aground in the Suez Canal, blocking all traffic in both directions; hundreds of vessels delayed.",
    "Missile attacks on commercial vessels in the Red Sea force shipping lines to reroute around the Cape of Good Hope.",
]
OPERATIONAL = "Dock workers strike shuts down the port; container backlog grows for a second week."
MINOR = "Slight traffic near downtown due to a local festival; drivers delayed ten minutes."
SAFE = [
    "Operations at the terminal are proceeding normally with no reported disruptions.",
    "Shipment cleared customs in four hours; traffic flowing at full capacity.",
]


def test_a_full_canal_closure_is_at_least_a_significant_disruption(nlp):
    # Severity 2 (significant) maps to 0.6, severity 3 (severe) to 1.0.
    assert nlp.get_semantic_score(CATASTROPHIC[0]) >= 0.6


ROUTINE_CONGESTION = "Maritime congestion reported at major transshipment hubs. Berthing delays expected."


@pytest.mark.parametrize("text", CATASTROPHIC)
def test_catastrophic_disruptions_score_above_routine_congestion(nlp, text):
    # Every sea leg carries the standing report as its baseline, capped at minor
    # (see RouteRecommender.run_background_warmup). A catastrophe must clearly
    # exceed that baseline.
    baseline = min(nlp.get_semantic_score(ROUTINE_CONGESTION), MINOR_SCORE)
    assert nlp.get_semantic_score(text) > max(0.3, baseline)


def test_minor_nuisance_scores_far_below_a_port_shutdown(nlp):
    assert nlp.get_semantic_score(MINOR) < 0.3 < nlp.get_semantic_score(OPERATIONAL)


def test_short_fragments_are_ignored(nlp):
    assert nlp.get_semantic_score("Rotterdam port reports record volumes | Light rain forecast") == 0.0


def test_threat_type_comes_from_the_headline_that_set_the_score(nlp):
    feed = "Port of Rotterdam reports record quarterly throughput | Typhoon Haikui forces closure of Kaohsiung and Xiamen ports"
    result = nlp.assess(feed)
    assert result["headline"].startswith("Typhoon Haikui") and result["type"] == "weather"


@pytest.mark.parametrize("text", SAFE)
def test_normal_operations_score_zero(nlp, text):
    assert nlp.get_semantic_score(text) == 0.0


@pytest.mark.parametrize("text", CATASTROPHIC + SAFE + [OPERATIONAL, MINOR, "", "ok"])
def test_score_is_always_between_zero_and_one(nlp, text):
    assert 0.0 <= nlp.get_semantic_score(text) <= 1.0


def test_every_disruption_outscores_a_minor_nuisance(nlp):
    # Severity is learned from labelled examples; its ranking is checked on real
    # news in test_the_real_headlines_meet_the_reported_scores. Here, clear cases:
    s = nlp.get_semantic_score
    assert min(s(t) for t in CATASTROPHIC + [OPERATIONAL]) > s(MINOR) >= s(SAFE[0]) == 0.0


# Held-out headlines, written after the anchors were designed and never used to tune them.
HELD_OUT_DISRUPTED = [
    "Cyclone Biparjoy halts operations at Kandla and Mundra ports",
    "Canadian rail workers strike, freezing freight across the country",
    "Drone attack damages tanker near Strait of Hormuz",
    "Fire at Felixstowe terminal suspends container handling",
    "Panama Canal cuts daily transits as drought lowers water levels",
    "Heavy snowfall shuts major highways, stranding thousands of trucks",
    "Baltimore bridge collapse closes port shipping channel",
    "Truckers blockade border crossing, halting freight for days",
]


def test_held_out_disruptions_are_detected(nlp):
    # All 8 at zero false alarms (the MiniLM engine caught 6).
    missed = [t for t in HELD_OUT_DISRUPTED if nlp.get_semantic_score(t) == 0]
    assert missed == []


# Measured on the real headlines: AUC 0.967, recall 0.824, false alarms 0.062,
# type 0.89, severity rank 0.553, CARF 0.915. Floors sit a little below. The
# false-alarm floor, over 97 real routine headlines, replaced a check that six
# hand-picked safe headlines score zero.
REAL_FLOORS = {"auc": 0.95, "recall": 0.8, "false_alarm_rate": 0.08, "type_accuracy": 0.87,
               "severity_rank_correlation": 0.5, "carf_accuracy": 0.9}


def test_the_real_headlines_meet_the_reported_scores(nlp):
    # ml/nlp_real_headlines.csv: 188 real headlines, labelled before the engine
    # saw them and never used to fit anything. The figures are in
    # docs/MODEL_CARD.md; these floors catch a regression.
    from backend.engine.threat_intelligence import CARFFilter
    from ml.evaluate_nlp import evaluate, load
    result = evaluate([r for r in load() if r["split"] == "real"], nlp, CARFFilter())
    detection = result["detection"]
    assert detection["auc"] >= REAL_FLOORS["auc"] and detection["recall"] >= REAL_FLOORS["recall"]
    assert detection["false_alarm_rate"] <= REAL_FLOORS["false_alarm_rate"]
    assert result["type_accuracy"] >= REAL_FLOORS["type_accuracy"]
    assert result["severity_rank_correlation"] >= REAL_FLOORS["severity_rank_correlation"]
    assert result["carf_accuracy"] >= REAL_FLOORS["carf_accuracy"]


def test_the_engine_never_learns_from_the_real_headlines(nlp):
    from ml.evaluate_nlp import load
    real = {r["headline"] for r in load() if r["split"] == "real"}
    disrupted, routine = nlp._load_labelled()
    assert disrupted and routine and not real & {r["headline"] for r in disrupted + routine}


def test_a_threat_in_a_multi_headline_feed_is_not_diluted(nlp):
    alone = nlp.get_semantic_score("Typhoon Haikui forces closure of Kaohsiung and Xiamen ports")
    feed = nlp.get_semantic_score("Port of Rotterdam reports record quarterly throughput | "
                                  "Typhoon Haikui forces closure of Kaohsiung and Xiamen ports | Maersk expands fleet")
    assert feed == pytest.approx(alone, abs=1e-6) and feed > 0.3


@pytest.mark.parametrize("text, threat_type", [
    ("Typhoon Haikui forces closure of Kaohsiung and Xiamen ports", "weather"),
    ("Dense fog closes Delhi airport, dozens of cargo flights diverted", "weather"),
    ("Longshoremen at Los Angeles and Long Beach walk off the job in contract dispute", "labour"),
    ("Houthi missile strikes on container ships in the Red Sea", "geopolitical"),
    ("Ever Given grounding blocks Suez Canal for six days", "infrastructure"),
    ("Ransomware attack cripples DP World terminal systems in Australia", "cyber"),
    ("Vessel queues at Shanghai reach record as yard congestion worsens", "congestion"),
])
def test_threat_type_classification(nlp, text, threat_type):
    assert nlp.classify_threat(text)["type"] == threat_type


def test_a_flaky_first_download_is_retried(monkeypatch):
    import sentence_transformers
    import backend.engine.threat_intelligence as ti
    real = sentence_transformers.SentenceTransformer
    calls = []

    def flaky(name, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only or len(calls) <= 3:
            raise OSError("connection reset")  # not cached yet, then two dropped downloads
        return real(name, local_files_only=True)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", flaky)
    monkeypatch.setattr(ti.time, "sleep", lambda s: None)
    engine = ti.ContrastiveNLPEngine()
    if not engine.ready:
        pytest.skip("Sentence-transformer model unavailable")
    assert calls == [True, False, False, False]
