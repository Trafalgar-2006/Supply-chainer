import pytest

from backend.engine.multimodal_network import create_multimodal_network
from backend.engine.route_recommender import RouteRecommender
from backend.engine.scenario_manager import ScenarioManager


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
        # Exact Shapley: the reference plus every driver rebuilds the total (up to rounding).
        rebuilt = drivers["reference_hours"] + sum(d["hours"] for d in drivers["drivers"])
        assert rebuilt == pytest.approx(drivers["total_hours"], abs=0.1 * (len(drivers["drivers"]) + 2))
        for l in rec["legs"]:
            assert l["delay"]["p50"] <= l["delay"]["p85"] <= l["delay"]["p95"]


def test_eta_band_is_the_quantile_of_the_route_not_a_sum_of_leg_quantiles(recommender):
    legs = [(10.0, 30.0, 60.0)] * 6
    band = recommender._eta_band(100.0, legs)
    assert band == recommender._eta_band(100.0, legs)  # seeded: identical requests agree
    comonotonic_p95 = 100.0 + 6 * 60.0
    assert band["p50"] < band["p85"] < band["p95"] < comonotonic_p95
    assert band["p50"] > 100.0 + 6 * 10.0 * 0.9
    assert recommender._eta_band(42.0, []) == {"p50": 42.0, "p85": 42.0, "p95": 42.0}


def test_splitting_a_leg_does_not_inflate_the_band_like_summed_quantiles(recommender):
    one = recommender._eta_band(0.0, [(20.0, 50.0, 90.0)])
    split = recommender._eta_band(0.0, [(10.0, 25.0, 45.0)] * 2)
    # Summing quantiles would give identical bands; the simulated band of two
    # partly independent halves is tighter in the tail, never looser.
    assert split["p95"] <= one["p95"] + 1e-6


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
    rec = RouteRecommender(create_multimodal_network(), ScenarioManager(), demo_mode=True)
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
    assert report["score"] > 0.2 and report["threat_type"] == "weather" and report["condition"] != "clear"
    assert report["headline"] == STORM["Rotterdam"]
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


def test_live_news_is_not_fetched_before_the_nlp_engine_is_ready(recommender, monkeypatch):
    def fail(place):
        raise AssertionError("fetched news that cannot be scored")
    assert not recommender.nlp.ready
    monkeypatch.setattr(recommender.news_ingestor, "fetch_headlines", fail)
    assert recommend(recommender, live_intel=True)["live_intel"] == []


def test_a_slow_feed_cannot_stall_a_request(live_recommender, monkeypatch):
    import time
    from backend.engine import route_recommender as rr
    monkeypatch.setattr(rr, "LIVE_INTEL_TIMEOUT_S", 0.5)
    monkeypatch.setattr(live_recommender.news_ingestor, "fetch_headlines", lambda place: time.sleep(3) or None)
    started = time.perf_counter()
    recommend(live_recommender, live_intel=True)
    assert time.perf_counter() - started < 2.5


def test_a_leg_shows_the_report_that_actually_raised_its_threat(live_recommender, monkeypatch):
    feed = {"Shanghai": "Dock workers strike shuts down the port; container backlog grows for a second week.",
            "Rotterdam": "Airport cargo handlers report minor delays on evening flights"}
    monkeypatch.setattr(live_recommender.news_ingestor, "fetch_headlines", lambda place: feed.get(place))
    result = recommend(live_recommender, transport_preference="sea", live_intel=True)
    live_legs = [l for x in result["recommendations"] for l in x["legs"] if l["intel_source"] == "LIVE"]
    assert live_legs
    # Sea, road and sea-road transfer legs are only exposed to the port strike;
    # the airport report is filtered out by CARF for all of them.
    for leg in live_legs:
        assert "strike" in leg["reason"].lower(), leg


def test_live_intel_is_off_unless_requested(live_recommender, monkeypatch):
    def fail(place):
        raise AssertionError("news fetched without live_intel")
    monkeypatch.setattr(live_recommender.news_ingestor, "fetch_headlines", fail)
    recommend(live_recommender)


def test_api_reports_model_evaluation(client):
    body = client.get("/api/model").json()
    assert body["available"] is True
    assert set(body["quantiles"]) == {"p50", "p85", "p95"}
    assert body["nlp"]["holdout"]["detection"]["auc"] > 0.9
    ceiling = body["delay_ceiling"]["quantiles"]
    assert all(q["optimal"] <= q["model"] <= q["naive"] for q in ceiling.values())
    status = client.get("/api/status").json()
    assert status["ml_trained"] is True and status["delay_model_error"] is None


def test_api_route_includes_bands(client):
    body = client.post("/api/recommend", json={"source": "Shanghai", "destination": "Rotterdam",
                                               "live_intel": False}).json()
    assert all({"eta_band", "delay_drivers", "personas"} <= set(r) for r in body["recommendations"])
