import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Skip the background NLP warm-up when the API app is imported in tests; routing
# tests drive threat through scenarios, and NLP tests build their own engine.
os.environ.setdefault("DEMO_MODE", "true")


@pytest.fixture(scope="session")
def hubs():
    from backend.engine.multimodal_network import load_canonical_hubs
    return load_canonical_hubs()


@pytest.fixture(scope="session")
def hub_index(hubs):
    return {h["id"]: h for h in hubs}


@pytest.fixture(scope="session")
def recommender():
    from backend.engine.multimodal_network import create_multimodal_network
    from backend.engine.route_recommender import RouteRecommender
    from backend.engine.scenario_manager import ScenarioManager
    from backend.engine.threat_intelligence import ThreatIntelligencePredictor

    return RouteRecommender(
        create_multimodal_network(),
        ThreatIntelligencePredictor(lazy_load=True),
        None,
        ScenarioManager(),
        demo_mode=True,
    )


@pytest.fixture(scope="session")
def nlp():
    from backend.engine.threat_intelligence import ContrastiveNLPEngine
    engine = ContrastiveNLPEngine()
    if not engine._ready:
        pytest.skip("Sentence-transformer model unavailable")
    return engine


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c
