import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.engine.scenario_manager import ScenarioManager


def route_hubs(rec):
    return [leg["to"] for leg in rec["legs"]]


def recommend(recommender, src="Shanghai", dst="Rotterdam", **kw):
    result = recommender.recommend(source=src, destination=dst, **kw)
    assert "error" not in result, result
    return result


def test_normal_shanghai_rotterdam_uses_suez(recommender):
    recs = recommend(recommender)["recommendations"]
    assert any("CHOKE-SUEZ" in route_hubs(r) for r in recs)


def test_suez_block_reroutes_risk_aware_personas(recommender):
    recs = recommend(recommender, scenario="SUEZ_BLOCK")["recommendations"]
    by_persona = {r["persona"]: r for r in recs}
    # Deduplication may merge personas that picked the same path; every returned
    # risk-aware route must avoid the blocked canal.
    for persona in ("SAFEST", "BALANCED"):
        if persona in by_persona:
            assert "CHOKE-SUEZ" not in route_hubs(by_persona[persona]), persona
    assert any("CHOKE-SUEZ" not in route_hubs(r) for r in recs)


def test_route_through_blocked_hub_reports_the_scenario(recommender):
    recs = recommend(recommender, scenario="SUEZ_BLOCK", transport_preference="sea")["recommendations"]
    for rec in recs:
        if "CHOKE-SUEZ" in route_hubs(rec):
            assert rec["threat_level"] == 1.0
            assert rec["audit_trace"]["eta"]["scenario"] >= 240


@pytest.mark.parametrize("scenario_id", list(ScenarioManager.SCENARIOS))
def test_scenario_delay_is_charged_once_per_disrupted_hub(recommender, scenario_id, hub_index):
    scenario = ScenarioManager.SCENARIOS[scenario_id]
    affected = set(scenario["affected_nodes"])
    pairs = {
        "SUEZ_BLOCK": ("Shanghai", "Rotterdam"),
        "RED_SEA_CONFLICT": ("Shanghai", "Rotterdam"),
        "LA_PORT_STRIKE": ("Shanghai", "Los Angeles"),
        "CHENNAI_FLOOD": ("Chennai", "Singapore"),
        "DUBAI_AIR_CONGESTION": ("Dubai", "Mumbai"),
        "HORMUZ_CLOSURE": ("Mumbai", "Rotterdam"),
    }
    src, dst = pairs[scenario_id]
    for rec in recommend(recommender, src=src, dst=dst, scenario=scenario_id)["recommendations"]:
        visited = {rec["legs"][0]["from"], *route_hubs(rec)} & affected
        expected = scenario["delay_hours"] * len(visited)
        assert rec["audit_trace"]["eta"]["scenario"] == pytest.approx(expected), (rec["persona"], visited)


@pytest.mark.parametrize("scenario", [None, "SUEZ_BLOCK", "RED_SEA_CONFLICT"])
def test_audit_trace_adds_up_to_the_totals(recommender, scenario):
    for rec in recommend(recommender, scenario=scenario, transport_preference="sea")["recommendations"]:
        eta, cost = rec["audit_trace"]["eta"], rec["audit_trace"]["cost"]
        assert eta["transit"] + eta["transfer"] + eta["scenario"] == pytest.approx(rec["adjusted_eta"], abs=0.1)
        assert cost["transit"] + cost["transfer"] + cost["scenario"] == pytest.approx(rec["total_cost"], abs=0.1)
        # Each leg ETA is rounded to 0.1h, so allow half a rounding step per leg.
        assert sum(l["eta"] for l in rec["legs"]) == pytest.approx(rec["adjusted_eta"], abs=0.05 * len(rec["legs"]) + 0.05)


def test_disrupted_origin_is_reported(recommender):
    for rec in recommend(recommender, src="Chennai", dst="Singapore", scenario="CHENNAI_FLOOD")["recommendations"]:
        assert rec["threat_level"] >= 0.75
        assert rec["audit_trace"]["eta"]["scenario"] >= 48


def test_explanation_percentages_are_valid(recommender):
    for scenario in (None, "SUEZ_BLOCK"):
        for rec in recommend(recommender, scenario=scenario)["recommendations"]:
            for pct in re.findall(r"(-?\d+(?:\.\d+)?)%", rec["explanation"]):
                assert 0 <= float(pct) <= 100, rec["explanation"]


def test_preferred_policy_biases_towards_the_preferred_mode(recommender):
    def hours_in(result, mode):
        return sum(l["eta"] for r in result["recommendations"] for l in r["legs"]
                   if l["type"] == "transit" and l["mode"] == mode.upper())

    unbiased = recommend(recommender, src="Shenzhen", dst="Mundra")
    preferred = recommend(recommender, src="Shenzhen", dst="Mundra", transport_preference="rail", routing_policy="PREFERRED")
    assert hours_in(preferred, "rail") > hours_in(unbiased, "rail")


def test_strict_policy_uses_only_the_chosen_mode_plus_road_access(recommender):
    for rec in recommend(recommender, transport_preference="sea", routing_policy="STRICT")["recommendations"]:
        assert {l["mode"] for l in rec["legs"] if l["type"] == "transit"} <= {"SEA", "ROAD"}


def test_concurrent_requests_do_not_leak_scenarios(recommender):
    jobs = [None, "SUEZ_BLOCK"] * 10
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda s: (s, recommender.recommend("Shanghai", "Rotterdam", scenario=s)), jobs))
    for scenario, result in results:
        expected = ScenarioManager.SCENARIOS[scenario]["name"] if scenario else None
        assert result["active_scenario"] == expected
        if scenario is None:
            assert all(r["audit_trace"]["eta"]["scenario"] == 0 for r in result["recommendations"])
