from fastapi import Depends, FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Literal, Optional
import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from . import security
from .engine.delay_model import EXECUTION_DIR
from .engine.multimodal_network import create_multimodal_network, load_canonical_hubs
from .engine.route_recommender import RouteRecommender
from .engine.scenario_manager import ScenarioManager
from .engine.supplier_scorer import SupplierScorer

# Global engine state, built once at start-up and shared by all requests.
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
        # Warm up, then keep the chokepoint news fresh. A daemon thread, so an
        # endless watch never holds up shutdown.
        def warm_up_and_watch():
            recommender.run_background_warmup()
            recommender.watch_chokepoints()
        threading.Thread(target=warm_up_and_watch, daemon=True, name="warmup-and-watch").start()
    yield

app = FastAPI(title="Supplychainer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=security.ALLOWED_ORIGINS,
    allow_credentials=False, # no cookies or auth headers are shared cross-origin
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

MAX_BODY_BYTES = 16 * 1024  # a route request is well under 1 KB

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    # Declared before the headers middleware, so its refusals get the headers too.
    if request.method == "POST":
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return JSONResponse({"detail": "A request body needs a Content-Length header."}, status_code=411)
        if int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": f"Request body over {MAX_BODY_BYTES // 1024} KB."}, status_code=413)
    return await call_next(request)

@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Say where each problem is, without echoing the input back.
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]})

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    extra = (security.API_HEADERS if path.startswith("/api/") else
             {} if path.startswith(security.DOCS_PATHS) else security.DASHBOARD_HEADERS)
    headers = {**security.SECURITY_HEADERS, **extra}
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

    Best matches first: an exact name or city, then an exact country, then a
    word starting with the query, then any substring, then part of a country
    name; busier hubs first within each group. A blank query matches nothing.
    Chokepoints are waypoints, so they are never offered.
    """
    q = q.lower().strip()
    if not q:
        return []

    def rank(hub):
        names = [hub["display_name"].lower(), hub["id"].lower(), *(a.lower() for a in hub["aliases"]),
                 hub.get("parent_city", "").lower()]
        if q in names:
            return 0
        if q == hub["country"].lower():
            return 1  # "India" lists India's hubs before Indianapolis
        if any(word.startswith(q) for name in names for word in name.replace("-", " ").split()):
            return 2
        if any(q in name for name in names):
            return 3
        if q in hub["country"].lower():
            return 4
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

@app.get("/api/currencies")
def get_currencies():
    """US dollar exchange rates for showing cost estimates in other currencies; none while the source is unreachable."""
    return {"base": "USD", **(recommender.news_ingestor.fetch_rates() or {"date": None, "rates": {}})}

def load_evaluation(name):
    """An offline evaluation report from Execution/ (see ml/), or None if not generated."""
    path = os.path.join(EXECUTION_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Written by ml/evaluate_nlp.py and ml/delay_ceiling.py.
EVALUATIONS = {"nlp": load_evaluation("nlp_evaluation.json"),
               "delay_ceiling": load_evaluation("delay_ceiling.json")}

@app.get("/api/model")
def get_model_report():
    """Held-out evaluation of the delay model and its ceiling, and of the threat-intelligence stage."""
    if recommender.delay_model is None:
        return {"available": False, "error": recommender.delay_model_error, **EVALUATIONS}
    return {"available": True, **recommender.delay_model.report, **EVALUATIONS}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Refuse cross-site WebSocket handshakes from pages on other origins.
    if not security.origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
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
    except WebSocketDisconnect:
        pass  # the page was closed or reloaded
    except Exception as e:
        print(f"[WS] Status socket failed: {type(e).__name__}: {e}")

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
    # A well-formed request the engine can't serve (an unknown place, the same
    # hub twice, no route under the constraints) is a 422 with the reason.
    return JSONResponse(result, status_code=422) if "error" in result else result

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

# The built dashboard, when present (run.py builds it), is served from the same
# port as the API: its page at / and its files under /assets. There is no
# catch-all, so a wrong method on an API route still answers 405.
DASHBOARD = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isfile(os.path.join(DASHBOARD, "index.html")) and os.path.isdir(os.path.join(DASHBOARD, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DASHBOARD, "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(os.path.join(DASHBOARD, "index.html"))
