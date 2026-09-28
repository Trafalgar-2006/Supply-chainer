from fastapi import Depends, FastAPI, Query, Request, WebSocket
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Literal, Optional
import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from . import security
from .engine.multimodal_network import create_multimodal_network, load_canonical_hubs
from .engine.route_recommender import RouteRecommender
from .engine.scenario_manager import ScenarioManager
from .engine.supplier_scorer import SupplierScorer

# Global engine state. The legacy US-only prototype in engine/ (simulator,
# baseline router, risk_model.pkl predictor) is not loaded by the API.
multimodal_net = create_multimodal_network()
scenario_mgr = ScenarioManager()
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
recommender = RouteRecommender(multimodal_net, scenario_mgr, demo_mode=DEMO_MODE)
canonical_hubs = load_canonical_hubs()
supplier_scorer = SupplierScorer(os.path.join(os.path.dirname(__file__), 'data', 'suppliers.json'))


HUB_IDS = {hub["id"] for hub in canonical_hubs}
MAX_UNITS = 10**9

def _known_scenario(value):
    if value is not None and value not in ScenarioManager.SCENARIOS:
        raise ValueError(f"unknown scenario; expected one of {sorted(ScenarioManager.SCENARIOS)}")
    return value

class Overrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    avoid_chokepoints: List[str] = Field(default_factory=list, max_length=20)
    cost_ceiling: Optional[float] = Field(None, gt=0, le=1e9)
    max_delay: Optional[float] = Field(None, gt=0, le=365) # days

    @field_validator("avoid_chokepoints")
    @classmethod
    def known_hubs(cls, value):
        unknown = [hub for hub in value if hub not in HUB_IDS]
        if unknown:
            raise ValueError(f"unknown hub ids: {unknown[:3]}")
        return value

class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(..., min_length=1, max_length=100) # Canonical Hub ID or City Name
    destination: str = Field(..., min_length=1, max_length=100)
    cargo_type: Literal["general", "perishable_urgent", "hazardous_waste", "oversize_heavy"] = "general"
    priority: Literal["low", "normal", "urgent"] = "normal"
    transport_preference: Literal["any", "sea", "air", "rail", "road"] = "any"
    routing_policy: Literal["STRICT", "PREFERRED"] = "STRICT"
    scenario: Optional[str] = Field(None, max_length=50)
    overrides: Optional[Overrides] = None
    live_intel: bool = True # fetch live news for the origin and destination

    _scenario = field_validator("scenario")(_known_scenario)

class SourcingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field("Electronics", min_length=1, max_length=50)
    current_inventory: int = Field(1000, ge=0, le=MAX_UNITS)
    safety_stock: int = Field(1500, ge=0, le=MAX_UNITS)
    demand_forecast: int = Field(800, ge=0, le=MAX_UNITS)
    scenario: Optional[str] = Field(None, max_length=50)

    _scenario = field_validator("scenario")(_known_scenario)

# Compute endpoints: optional API key, then the per-client rate limit.
PROTECTED = [Depends(security.require_api_key), Depends(security.rate_limit)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Supplychainer Engine Active: Canonical Global Registry Loaded.")
    if not DEMO_MODE:
        asyncio.create_task(asyncio.to_thread(recommender.run_background_warmup))
    yield

app = FastAPI(title="Supplychainer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=security.ALLOWED_ORIGINS,
    allow_credentials=False, # no cookies or auth headers are shared cross-origin
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    headers = {**security.SECURITY_HEADERS, **(security.API_HEADERS if request.url.path.startswith("/api/") else {})}
    for name, value in headers.items():
        response.headers.setdefault(name, value)
    return response

@app.get("/api/scenarios")
def get_scenarios():
    """Returns available disruption scenarios."""
    return scenario_mgr.get_all_scenarios()

@app.get("/api/hubs")
def get_hubs():
    """Returns the full canonical hub registry."""
    return canonical_hubs

SEARCH_LIMIT = 12

@app.get("/api/hubs/search")
def search_hubs(q: str = Query(..., min_length=1, max_length=100)):
    """Hubs a shipment can start or end at, matching a name, alias, ID or country.

    Best matches first: an exact name, then a word starting with the query, then
    any substring, then the country; busier hubs first within each group.
    Chokepoints are waypoints, so they are never offered.
    """
    q = q.lower().strip()

    def rank(hub):
        names = [hub["display_name"].lower(), hub["id"].lower(), *(a.lower() for a in hub["aliases"])]
        if q in names:
            return 0
        if any(word.startswith(q) for name in names for word in name.replace("-", " ").split()):
            return 1
        if any(q in name for name in names):
            return 2
        if q in hub["country"].lower():
            return 3
        return None

    ranked = [(r, -hub["importance"], hub["display_name"], hub) for hub in canonical_hubs
              if hub["type"] != "choke_point" and (r := rank(hub)) is not None]
    return [hub for *_, hub in sorted(ranked, key=lambda x: x[:3])[:SEARCH_LIMIT]]

@app.get("/api/network")
def get_network():
    nodes = []
    for n, data in multimodal_net.nodes(data=True):
        nodes.append({"id": n, "display_name": data.get("display_name"), "type": data.get("type")})
    
    edges = []
    seen = set()
    for u, v, data in multimodal_net.edges(data=True):
        edge_key = tuple(sorted((u, v)))
        if edge_key not in seen:
            edges.append({"source": u, "target": v, "baseline_time": data.get("baseline_time")})
            seen.add(edge_key)
        
    return {"nodes": nodes, "edges": edges}

@app.get("/api/status")
def get_status():
    return {
        "ml_trained": recommender.delay_model is not None,
        "delay_model_error": recommender.delay_model_error,
        "is_supplychainer": True,
        "geo_scope": "Global (Canonical)",
        "hub_count": len(canonical_hubs)
    }

@app.get("/api/model")
def get_model_report():
    """Held-out evaluation of the delay quantile model (coverage, pinball loss, importance)."""
    if recommender.delay_model is None:
        return {"available": False, "error": recommender.delay_model_error}
    return {"available": True, **recommender.delay_model.report}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Refuse cross-site WebSocket handshakes from pages on other origins.
    if not security.origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            if recommender.warmup_failed:
                status_msg = "WARM-UP FAILED"
            elif not recommender.is_warmed_up:
                status_msg = "WARMING RISK ENGINE"
            else:
                status_msg = "FULLY OPERATIONAL"
                
            state = {
                "ml_trained": recommender.delay_model is not None,
                "engine_status": status_msg,
                "hub_registry": "Synchronized"
            }
            await websocket.send_text(json.dumps(state))
            await asyncio.sleep(2.0)
    except Exception as e:
        print(f"WebSocket closed: {e}")

@app.get("/api/cities")
def get_cities():
    """Returns the city-to-hub mapping for multimodal resolution."""
    path = os.path.join(os.path.dirname(__file__), 'data', 'canonical_locations.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@app.post("/api/recommend", dependencies=PROTECTED)
def recommend_routes(req: RecommendRequest):
    result = recommender.recommend(
        source=req.source,
        destination=req.destination,
        cargo_type=req.cargo_type,
        priority=req.priority,
        transport_preference=req.transport_preference,
        routing_policy=req.routing_policy,
        scenario=req.scenario,
        overrides=req.overrides.model_dump(exclude_none=True) if req.overrides else None,
        live_intel=req.live_intel
    )
    return result

@app.post("/api/suppliers", dependencies=PROTECTED)
def get_suppliers(req: SourcingRequest):
    # Per-request lookup: never mutate the scenario shared by concurrent requests
    active_disruptions = scenario_mgr.get_disruptions(req.scenario)
    
    ranked_suppliers = supplier_scorer.get_ranked_suppliers(req.category, active_disruptions)
    advice = supplier_scorer.get_procurement_advice(req.current_inventory, req.safety_stock, req.demand_forecast)
    
    return {
        "suppliers": ranked_suppliers,
        "advice": advice,
        "active_disruptions": active_disruptions
    }
