import pytest
from starlette.websockets import WebSocketDisconnect

from backend import security

ROUTE = {"source": "Shanghai", "destination": "Rotterdam", "live_intel": False}


@pytest.fixture(autouse=True)
def fresh_limiter():
    security.limiter.reset()
    yield
    security.limiter.reset()


@pytest.mark.parametrize("patch", [
    {"transport_preference": "teleport"},
    {"routing_policy": "WHATEVER"},
    {"cargo_type": "plutonium"},
    {"priority": "yesterday"},
    {"scenario": "ZOMBIE_OUTBREAK"},
    {"source": "x" * 101},
    {"source": ""},
    {"budget_sensitivity": "medium"},                        # unknown field
    {"overrides": {"avoid_chokepoints": ["CHOKE-NOWHERE"]}},
    {"overrides": {"avoid_chokepoints": ["CHOKE-SUEZ"] * 21}},
    {"overrides": {"cost_ceiling": -5}},
    {"overrides": {"max_delay": 10_000}},
    {"overrides": {"__class__": "x"}},
])
def test_invalid_route_requests_are_rejected(client, patch):
    response = client.post("/api/recommend", json={**ROUTE, **patch})
    assert response.status_code == 422


@pytest.mark.parametrize("patch", [
    {"current_inventory": -1},
    {"safety_stock": 10**12},
    {"demand_forecast": "lots"},
    {"scenario": "ZOMBIE_OUTBREAK"},
    {"category": "x" * 51},
    {"extra": 1},
])
def test_invalid_supplier_requests_are_rejected(client, patch):
    assert client.post("/api/suppliers", json=patch).status_code == 422


def test_valid_overrides_still_work(client):
    body = client.post("/api/recommend", json={**ROUTE, "overrides": {"avoid_chokepoints": ["CHOKE-SUEZ"]}}).json()
    assert all("CHOKE-SUEZ" not in [l["to"] for l in r["legs"]] for r in body["recommendations"])


def test_hub_search_query_is_bounded(client):
    assert client.get("/api/hubs/search", params={"q": "a" * 101}).status_code == 422


def test_compute_endpoints_are_rate_limited(client, monkeypatch):
    monkeypatch.setattr(security.limiter, "limit", 3)
    codes = [client.post("/api/suppliers", json={}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    blocked = client.post("/api/recommend", json=ROUTE)
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) >= 1


def test_rate_limit_window_slides(monkeypatch):
    limiter = security.RateLimiter(limit=2, window_s=10)
    clock = iter([0.0, 1.0, 2.0, 11.5])
    monkeypatch.setattr(security.time, "monotonic", lambda: next(clock))
    assert [limiter.check("a") > 0 for _ in range(4)] == [False, False, True, False]


def test_rate_limiter_memory_is_bounded(monkeypatch):
    limiter = security.RateLimiter(limit=5, window_s=1, max_clients=100)
    now = [0.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    for i in range(100):
        limiter.check(f"client-{i}")
    now[0] = 5.0  # every earlier window has expired
    limiter.check("newcomer")
    assert len(limiter._hits) == 1


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(security, "API_KEY", "s3cret-key")
    assert client.post("/api/suppliers", json={}).status_code == 401
    assert client.post("/api/suppliers", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/suppliers", json={}, headers={"X-API-Key": "s3cret-key"}).status_code == 200


def test_api_is_open_when_no_key_is_configured(client, monkeypatch):
    monkeypatch.setattr(security, "API_KEY", None)
    assert client.post("/api/suppliers", json={}).status_code == 200


def test_security_headers(client):
    response = client.get("/api/scenarios")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert response.headers["cache-control"] == "no-store"


def test_cors_allows_only_the_dashboard_origin(client):
    allowed = client.get("/api/scenarios", headers={"Origin": "http://localhost:5173"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-credentials" not in allowed.headers
    denied = client.get("/api/scenarios", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in denied.headers


def test_websocket_refuses_foreign_origins(client):
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_text()
    assert closed.value.code == 1008


def test_websocket_serves_the_dashboard_origin(client):
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5173"}) as ws:
        assert "engine_status" in ws.receive_json()
