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
