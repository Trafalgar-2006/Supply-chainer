"""HTTP hardening for the Supplychainer API.

- Optional API key: set SUPPLYCHAINER_API_KEY to require an X-API-Key header on
  the compute endpoints (compared in constant time). Unset, the API stays open
  for local development and demos.
- Per-client rate limit on the compute endpoints (RATE_LIMIT_PER_MINUTE, default
  60), answered with 429 and Retry-After.
- CORS and WebSocket origins limited to CORS_ORIGINS.
- Security headers on every response, and a no-content CSP on the JSON API.

See SECURITY.md for the threat model.
"""
import hmac
import os
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

API_KEY = os.getenv("SUPPLYCHAINER_API_KEY") or None
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
# The API only returns JSON: nothing in a response may load or be framed.
API_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
}
# The dashboard, when this server serves it: its own scripts, Google Fonts,
# OpenStreetMap tiles, and the API and status socket on the same origin.
# Inline styles are allowed because Leaflet and the charts position with them.
DASHBOARD_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data: https://tile.openstreetmap.org; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"),
    # OpenStreetMap's tile servers answer requests without a Referer with an
    # "Access blocked" tile; this sends them the page's origin, never its path.
    "Referrer-Policy": "strict-origin-when-cross-origin",
}
DOCS_PATHS = ("/docs", "/redoc")  # FastAPI's interactive docs load their own scripts from a CDN


class RateLimiter:
    """Sliding-window request limit per client address, held in memory.

    Suits a single API process. Behind a reverse proxy every request shares the
    proxy's address; keying on X-Forwarded-For would need a trusted-proxy
    setting, since clients can forge that header.
    """

    def __init__(self, limit: int, window_s: float = 60.0, max_clients: int = 10_000):
        self.limit = limit
        self.window_s = window_s
        self.max_clients = max_clients
        self._hits = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> float:
        """Record a request; return 0 if allowed, else seconds until one is."""
        now = time.monotonic()
        with self._lock:
            if client not in self._hits and len(self._hits) >= self.max_clients:
                self._prune(now)
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window_s - (now - hits[0])
            hits.append(now)
            return 0.0

    def _prune(self, now: float):
        for client in [c for c, hits in self._hits.items() if not hits or now - hits[-1] >= self.window_s]:
            del self._hits[client]

    def reset(self):
        with self._lock:
            self._hits.clear()


limiter = RateLimiter(int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))


def require_api_key(request: Request):
    if API_KEY is None:
        return
    supplied = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(supplied.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key",
                            headers={"WWW-Authenticate": "API-Key"})


def rate_limit(request: Request):
    # Each endpoint has its own budget, so the supplier page refreshing as you
    # type never uses up route planning.
    client = request.client.host if request.client else "unknown"
    retry_after = limiter.check(f"{client} {request.url.path}")
    if retry_after:
        raise HTTPException(status_code=429, detail="Too many requests",
                            headers={"Retry-After": str(int(retry_after) + 1)})


def origin_allowed(origin, host=None) -> bool:
    """Whether a WebSocket handshake from this Origin is allowed.

    Browsers always send Origin on WebSocket handshakes; non-browser clients may
    omit it. Besides CORS_ORIGINS, a page this server served itself (Origin
    matching the Host header) is allowed, as when run.py serves the dashboard.
    """
    if origin is None or origin in ALLOWED_ORIGINS:
        return True
    return host is not None and origin in (f"http://{host}", f"https://{host}")
