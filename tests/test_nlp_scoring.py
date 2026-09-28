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


@pytest.mark.parametrize("text", CATASTROPHIC)
def test_catastrophic_disruptions_saturate(nlp, text):
    assert nlp.get_semantic_score(text) >= 0.9


def test_operational_disruption_is_a_moderate_threat(nlp):
    assert 0.2 <= nlp.get_semantic_score(OPERATIONAL) <= 0.8


@pytest.mark.parametrize("text", SAFE)
def test_normal_operations_score_zero(nlp, text):
    assert nlp.get_semantic_score(text) == 0.0


@pytest.mark.parametrize("text", CATASTROPHIC + SAFE + [OPERATIONAL, MINOR, "", "ok"])
def test_score_is_always_between_zero_and_one(nlp, text):
    assert 0.0 <= nlp.get_semantic_score(text) <= 1.0


def test_severity_ordering(nlp):
    s = nlp.get_semantic_score
    assert s(CATASTROPHIC[0]) > s(OPERATIONAL) > s(MINOR) >= s(SAFE[0]) == 0.0


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


def test_every_held_out_disruption_outranks_every_safe_headline(nlp):
    worst_safe = max(nlp.get_semantic_score(t) for t in HELD_OUT_SAFE)
    assert worst_safe < 0.1
    for text in HELD_OUT_DISRUPTED:
        assert nlp.get_semantic_score(text) > worst_safe, text


def test_a_threat_in_a_multi_headline_feed_is_not_diluted(nlp):
    alone = nlp.get_semantic_score("Typhoon Haikui forces closure of Kaohsiung and Xiamen ports")
    feed = nlp.get_semantic_score("Port of Rotterdam reports record quarterly throughput | "
                                  "Typhoon Haikui forces closure of Kaohsiung and Xiamen ports | Maersk expands fleet")
    assert feed == pytest.approx(alone, abs=1e-6) and feed > 0.5


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
