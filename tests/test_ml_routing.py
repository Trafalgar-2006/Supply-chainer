import pytest

from backend.engine.multimodal_network import create_multimodal_network
from backend.engine.route_recommender import RouteRecommender
from backend.engine.scenario_manager import ScenarioManager
from backend.engine.threat_intelligence import ThreatIntelligencePredictor


def recommend(recommender, src="Shanghai", dst="Rotterdam", **kw):
    result = recommender.recommend(source=src, destination=dst, **kw)
    assert "error" not in result, result
    return result


def transit_modes(rec):
    return {l["mode"] for l in rec["legs"] if l["type"] == "transit"}


def test_every_route_carries_a_delay_band_and_its_drivers(recommender):
    result = recommend(recommender, transport_preference="sea")
    assert result["delay_model"] is True
    for rec in result["recommendations"]:
        drivers = rec["delay_drivers"]
        assert drivers["quantile"] == "p85" and drivers["drivers"]
        # Shapley drivers plus the per-leg reference reconstruct the route's p85 delay.
        rebuilt = drivers["reference_hours"] + sum(d["hours"] for d in drivers["drivers"])
        assert rebuilt == pytest.approx(drivers["total_hours"], rel=0.05, abs=2)
        for l in rec["legs"]:
            assert l["delay"]["p50"] <= l["delay"]["p85"] <= l["delay"]["p95"]


def test_personas_plan_on_their_own_quantile(recommender):
    recs = recommend(recommender, transport_preference="sea")["recommendations"]
    safest = next(r for r in recs if "SAFEST" in r["personas"])
    fastest = next(r for r in recs if "FASTEST" in r["personas"])
    # FASTEST optimises the typical (p50) ETA, SAFEST the worst case (p95).
    assert fastest["eta_band"]["p50"] <= safest["eta_band"]["p50"]
    assert safest["eta_band"]["p95"] <= fastest["eta_band"]["p95"] + 1e-6 or safest is fastest


@pytest.mark.parametrize("cargo, banned", [("hazardous_waste", "AIR"), ("perishable_urgent", "SEA"), ("oversize_heavy", "ROAD")])
def test_cargo_restrictions_exclude_the_banned_mode(recommender, cargo, banned):
    result = recommender.recommend("Shanghai", "Rotterdam", cargo_type=cargo)
    if "error" in result:
        return  # no compliant route at all is an acceptable answer
    for rec in result["recommendations"]:
        assert banned not in transit_modes(rec)


def test_urgent_priority_trades_cost_for_time(recommender):
    def balanced(priority):
        recs = recommend(recommender, src="Mumbai", dst="Delhi", priority=priority)["recommendations"]
        return next(r for r in recs if "BALANCED" in r["personas"])
    low, urgent = balanced("low"), balanced("urgent")
    assert urgent["adjusted_eta"] <= low["adjusted_eta"]
    assert low["total_cost"] <= urgent["total_cost"]


@pytest.fixture(scope="module")
def live_recommender(nlp):
    rec = RouteRecommender(create_multimodal_network(), ThreatIntelligencePredictor(lazy_load=True), None,
                           ScenarioManager(), demo_mode=True)
    rec.nlp = nlp
    return rec


STORM = {"Rotterdam": "Severe storm and flooding force closure of the port of Rotterdam"}
ROTTERDAM = {"PORT-ROTTERDAM", "HUB-ROTTERDAM"}  # both have quays


def sea_legs_into_rotterdam(result):
    return [l for x in result["recommendations"] for l in x["legs"]
            if l["to"] in ROTTERDAM and l["mode"] == "SEA"]


def test_live_storm_news_is_reported_on_the_affected_legs(live_recommender, monkeypatch):
    monkeypatch.setattr(live_recommender.news_ingestor, "fetch_headlines", lambda place: STORM.get(place))
    stormy = recommend(live_recommender, transport_preference="sea", live_intel=True)

    (report,) = stormy["live_intel"]
    assert report["place"] == "Rotterdam" and "PORT-ROTTERDAM" in report["hubs"]
    assert report["score"] > 0.3 and report["threat_type"] == "weather" and report["condition"] != "clear"
    # Routes may divert around the storm-hit port; any that still sail in carry the live report.
    assert ROTTERDAM <= set(report["hubs"])
    assert all(l["intel_source"] == "LIVE" and l["threat"] > 0.05 for l in sea_legs_into_rotterdam(stormy))


def test_ships_divert_around_a_storm_closed_port(live_recommender, monkeypatch):
    monkeypatch.setattr(live_recommender.news_ingestor, "fetch_headlines", lambda place: STORM.get(place))
    assert sea_legs_into_rotterdam(recommend(live_recommender, transport_preference="sea"))
    assert not sea_legs_into_rotterdam(recommend(live_recommender, transport_preference="sea", live_intel=True))


def test_live_storm_news_raises_the_delay_of_lanes_into_the_port(live_recommender, monkeypatch):
    monkeypatch.setattr(live_recommender.news_ingestor, "fetch_headlines", lambda place: STORM.get(place))
    G = live_recommender.unified_graph
    stormy = live_recommender._intel_delays(live_recommender._live_intel(["Rotterdam"]))
    lanes = [(u, v) for hub in ROTTERDAM for u, v in G.in_edges(f"{hub}:sea") if G[u][v]["type"] == "transit"]
    assert lanes
    for u, v in lanes:
        calm = G[u][v]["delay_q"]
        assert all(s > c for s, c in zip(stormy[(u, v)], calm))


def test_no_live_news_means_no_live_signal(live_recommender, monkeypatch):
    monkeypatch.setattr(live_recommender.news_ingestor, "fetch_headlines", lambda place: None)
    result = recommend(live_recommender, live_intel=True)
    assert result["live_intel"] == []
    assert all(l["intel_source"] != "LIVE" for x in result["recommendations"] for l in x["legs"])


def test_live_intel_is_off_unless_requested(live_recommender, monkeypatch):
    def fail(place):
        raise AssertionError("news fetched without live_intel")
    monkeypatch.setattr(live_recommender.news_ingestor, "fetch_headlines", fail)
    recommend(live_recommender)


def test_api_reports_model_evaluation(client):
    body = client.get("/api/model").json()
    assert body["available"] is True
    assert set(body["quantiles"]) == {"p50", "p85", "p95"}
    status = client.get("/api/status").json()
    assert status["ml_trained"] is True and status["delay_model_error"] is None


def test_api_route_includes_bands(client):
    body = client.post("/api/recommend", json={"source": "Shanghai", "destination": "Rotterdam",
                                               "live_intel": False}).json()
    assert all({"eta_band", "delay_drivers", "personas"} <= set(r) for r in body["recommendations"])
