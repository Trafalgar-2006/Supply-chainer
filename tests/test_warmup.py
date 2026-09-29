import time

from backend.engine.multimodal_network import create_multimodal_network
from backend.engine.route_recommender import RouteRecommender
from backend.engine.scenario_manager import ScenarioManager
from backend.engine.threat_intelligence import MINOR_SCORE


def test_warmup_scores_every_edge_quickly(nlp):
    rec = RouteRecommender(create_multimodal_network(), ScenarioManager())
    started = time.perf_counter()
    rec.run_background_warmup()
    elapsed = time.perf_counter() - started

    assert rec.is_warmed_up and not rec.warmup_failed
    assert elapsed < 60, f"warm-up took {elapsed:.0f}s"
    edges = [d for _, _, d in rec.unified_graph.edges(data=True) if d["transport_mode"] != "transfer"]
    assert all("base_threat" in d and "base_news" in d for d in edges)
    # The sea baseline report describes berthing congestion: routine conditions,
    # so a non-zero threat capped at minor.
    sea = {d["base_threat"] for d in edges if d["transport_mode"] == "sea"}
    assert len(sea) == 1 and 0 < sea.pop() <= MINOR_SCORE
