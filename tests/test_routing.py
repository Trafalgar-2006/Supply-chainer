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
        assert eta["transit"] + eta["transfer"] + eta["delay"] + eta["scenario"] == pytest.approx(rec["adjusted_eta"], abs=0.1)
        assert rec["eta_band"]["p50"] <= rec["eta_band"]["p85"] <= rec["eta_band"]["p95"]
        assert rec["eta_band"]["p85"] >= rec["adjusted_eta"]
        assert cost["transit"] + cost["transfer"] + cost["scenario"] == pytest.approx(rec["total_cost"], abs=0.1)
        # Each leg ETA is rounded to 0.1h, so allow half a rounding step per leg.
        assert sum(l["eta"] for l in rec["legs"]) == pytest.approx(rec["adjusted_eta"], abs=0.05 * len(rec["legs"]) + 0.05)


def test_la_port_strike_affects_ships_bound_for_los_angeles(recommender):
    normal = recommend(recommender, src="Shanghai", dst="Los Angeles", transport_preference="sea")["recommendations"]
    assert any({"PORT-LOSANGELES", "PORT-LONGBEACH"} & set(route_hubs(r)) for r in normal)
    for rec in recommend(recommender, src="Shanghai", dst="Los Angeles", transport_preference="sea",
                         scenario="LA_PORT_STRIKE")["recommendations"]:
        through_la = {"PORT-LOSANGELES", "PORT-LONGBEACH"} & set(route_hubs(rec))
        assert not through_la or rec["audit_trace"]["eta"]["scenario"] >= 120


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


def test_preferred_policy_never_trades_a_route_for_a_worse_one_without_the_mode(recommender):
    # Splitting a long road trip into short hops must not dodge the preference penalty.
    unbiased = recommend(recommender, src="Hamburg", dst="Rotterdam")["recommendations"]
    unbiased_paths = {tuple(l["to"] for l in r["legs"]) for r in unbiased}
    for rec in recommend(recommender, src="Hamburg", dst="Rotterdam", transport_preference="sea",
                         routing_policy="PREFERRED")["recommendations"]:
        uses_sea = any(l["mode"] == "SEA" for l in rec["legs"] if l["type"] == "transit")
        assert uses_sea or tuple(l["to"] for l in rec["legs"]) in unbiased_paths


@pytest.mark.parametrize("src, dst, scenario, pref, policy", [
    ("Shanghai", "Rotterdam", None, "any", "STRICT"),
    ("Shanghai", "Rotterdam", "SUEZ_BLOCK", "any", "STRICT"),
    ("Chennai", "Singapore", "CHENNAI_FLOOD", "any", "STRICT"),
    ("Hamburg", "Frankfurt", None, "rail", "PREFERRED"),
    ("Hamburg", "Frankfurt", None, "sea", "PREFERRED"),
])
def test_a_route_is_called_fastest_only_if_it_is(recommender, src, dst, scenario, pref, policy):
    recs = recommend(recommender, src=src, dst=dst, scenario=scenario, transport_preference=pref,
                     routing_policy=policy)["recommendations"]
    for rec in recs:
        if rec["explanation"].startswith("Fastest option:"):
            assert rec["adjusted_eta"] <= min(r["adjusted_eta"] for r in recs)


@pytest.mark.parametrize("scenario", [None] + list(ScenarioManager.SCENARIOS))
def test_fastest_persona_minimises_the_reported_eta(recommender, scenario):
    # The optimiser and the reported trace use the same scenario accounting, so with
    # no preference the FASTEST persona's reported ETA is the lowest of all personas.
    for src, dst in (("Shanghai", "Rotterdam"), ("Chennai", "Singapore"), ("Mumbai", "Dubai")):
        recs = recommend(recommender, src=src, dst=dst, scenario=scenario)["recommendations"]
        fastest = next(r for r in recs if "FASTEST" in r["personas"])
        assert fastest["adjusted_eta"] <= min(r["adjusted_eta"] for r in recs)


def test_near_total_savings_are_not_rounded_to_100_percent(recommender):
    for rec in recommend(recommender, src="Dubai", dst="Mumbai", scenario="DUBAI_AIR_CONGESTION")["recommendations"]:
        assert "100%" not in rec["explanation"]


def test_returned_routes_are_distinct_paths(recommender):
    recs = recommend(recommender, transport_preference="sea")["recommendations"]
    sigs = [tuple((l["to"], l["mode"], l["type"]) for l in r["legs"]) for r in recs]
    assert len(sigs) == len(set(sigs))
    assert sorted(p for r in recs for p in r["personas"]) == ["BALANCED", "FASTEST", "SAFEST"]


def test_routing_does_not_modify_the_shared_graph(recommender):
    before = recommender.unified_graph.number_of_edges()
    recommend(recommender, transport_preference="sea", routing_policy="STRICT",
              overrides={"avoid_chokepoints": ["CHOKE-SUEZ"]})
    assert recommender.unified_graph.number_of_edges() == before


def test_avoided_chokepoint_is_never_used(recommender):
    for rec in recommend(recommender, transport_preference="sea",
                         overrides={"avoid_chokepoints": ["CHOKE-SUEZ"]})["recommendations"]:
        assert "CHOKE-SUEZ" not in route_hubs(rec)
        assert rec["override_applied"]


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


def test_explanations_do_not_claim_meaningless_differences(recommender):
    for scenario in (None, "SUEZ_BLOCK"):
        for rec in recommend(recommender, scenario=scenario, transport_preference="sea")["recommendations"]:
            text = rec["explanation"]
            assert not re.search(r"(?<!\d)[01]% cheaper", text), text
            assert not re.search(r"(?<!\d)1\.0x its cost", text), text
            assert not re.search(r"(?<!\d)[+-]?0h (sooner|on its ETA)", text), text
