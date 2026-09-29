import pytest

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


def test_a_full_canal_closure_scores_near_maximum(nlp):
    assert nlp.get_semantic_score(CATASTROPHIC[0]) >= 0.9


ROUTINE_CONGESTION = "Maritime congestion reported at major transshipment hubs. Berthing delays expected."


@pytest.mark.parametrize("text", CATASTROPHIC)
def test_catastrophic_disruptions_score_above_routine_congestion(nlp, text):
    # The margin measures how much a headline reads like a disruption, not how
    # large it is, so an attack need not outscore a strike. It must still clearly
    # exceed the standing report every sea leg carries.
    assert nlp.get_semantic_score(text) > max(0.3, nlp.get_semantic_score(ROUTINE_CONGESTION))


def test_a_strike_stays_well_below_a_canal_closure(nlp):
    # The score feeds the delay model as incident severity, so a local strike
    # must not read like a full closure.
    strike = nlp.get_semantic_score(OPERATIONAL)
    assert 0.3 <= strike <= 0.8
    assert nlp.get_semantic_score(CATASTROPHIC[0]) - strike >= 0.2


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


def test_severity_ordering(nlp):
    s = nlp.get_semantic_score
    assert s(CATASTROPHIC[0]) >= s(OPERATIONAL) > s(MINOR) >= s(SAFE[0]) == 0.0


# Held-out headlines, written after the anchors were designed and never used to tune them.
HELD_OUT_SAFE = [
    "DHL opens automated hub in Leipzig to speed parcel sorting",
    "Singapore port throughput rises 5% year on year",
    "CMA CGM orders twelve LNG-powered container ships",
    "Freight rates stabilise as schedule reliability improves",
    "Indian Railways commissions new dedicated freight corridor section",
    "Airline cargo unit posts best quarter since 2021",
]
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


def test_no_false_alarms_on_held_out_safe_news(nlp):
    assert [nlp.get_semantic_score(t) for t in HELD_OUT_SAFE] == [0.0] * len(HELD_OUT_SAFE)


def test_held_out_disruptions_are_detected(nlp):
    # All 8 at zero false alarms (the MiniLM engine caught 6).
    missed = [t for t in HELD_OUT_DISRUPTED if nlp.get_semantic_score(t) == 0]
    assert missed == []


def test_the_holdout_split_meets_the_reported_scores(nlp):
    # ml/nlp_headlines.csv: the holdout split was written after all tuning and is
    # never used to fit anything. The figures are in docs/MODEL_CARD.md; these
    # floors catch a regression.
    from backend.engine.threat_intelligence import CARFFilter
    from ml.evaluate_nlp import evaluate, load
    result = evaluate([r for r in load() if r["split"] == "holdout"], nlp, CARFFilter())
    detection = result["detection"]
    assert detection["auc"] >= 0.95 and detection["recall"] >= 0.8 and detection["false_alarm_rate"] <= 0.1
    assert result["type_accuracy"] >= 0.9 and result["carf_accuracy"] >= 0.9


def test_the_type_classifier_never_learns_from_the_holdout(nlp):
    import backend.engine.threat_intelligence as ti
    from ml.evaluate_nlp import load
    holdout = {r["headline"] for r in load() if r["split"] == "holdout"}
    used = []
    real_encode = nlp.model.encode
    nlp.model.encode = lambda texts, **kw: used.extend(texts) or real_encode(texts, **kw)
    try:
        nlp._fit_type_model([])
    finally:
        nlp.model.encode = real_encode
    assert used and not holdout & set(used)


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
