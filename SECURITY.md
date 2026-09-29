# Security

Supplychainer is a decision-support API and dashboard. This document covers:

- what the system exposes
- what could go wrong, and how each risk is handled
- the tests that prove each protection works
- what is left for a production deployment

All tests referenced here are in `tests/test_security.py` unless stated otherwise.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `SUPPLYCHAINER_API_KEY` | unset (API open) | When set, `POST /api/recommend` and `POST /api/suppliers` require an `X-API-Key` header. The dev proxy injects the key on the server side, so it never reaches the browser. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Maximum compute requests per client address per minute. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Browser origins allowed to call the API and open the status WebSocket. |

## Threat model

| Entry point | Threat | Mitigation | Test |
|---|---|---|---|
| **Model artifact** (`Execution/delay_quantile_model.joblib`) | Unpickling a swapped or tampered file runs arbitrary code when the server starts | SHA-256 pinned **in code** (`EXPECTED_SHA256`). The bytes that are hashed are the bytes that are loaded, so there is no check-then-use gap. A mismatch is refused before deserialising, and the router falls back to nominal times. | `test_delay_model.py::test_tampered_artifact_is_refused_before_unpickling`, `test_a_rewritten_report_hash_does_not_bypass_the_pin` |
| **Legacy artifacts** (`risk_model.pkl`, `label_encoders.pkl`, `nlp_anchors.pt`) | Same unpickling risk, with no integrity data | Removed from the repository: the router uses the pinned delay model, and the NLP anchors are built from text in code | — |
| `POST /api/recommend`, `POST /api/suppliers` | Malformed or hostile input, parameter confusion, oversized payloads | Strict pydantic models:<br>• enumerated values for mode, policy, cargo and priority<br>• length limits on strings<br>• only known scenario and hub IDs accepted<br>• numbers bounded<br>• **unknown fields rejected** | `test_invalid_route_requests_are_rejected`, `test_invalid_supplier_requests_are_rejected` |
| Same endpoints | Resource exhaustion: each request runs three Dijkstra searches and possibly two news fetches | Per-client sliding-window rate limit, answered with **429 + Retry-After**. The limiter's memory is capped. | `test_compute_endpoints_are_rate_limited`, `test_rate_limit_window_slides`, `test_rate_limiter_memory_is_bounded` |
| Same endpoints | Unauthorised use of a deployed instance | Optional API key, compared in constant time (`hmac.compare_digest`) so the comparison time doesn't reveal the key | `test_api_key_is_enforced_when_configured` |
| `GET /api/hubs/search` | Oversized queries | `q` is limited to 100 characters | `test_hub_search_query_is_bounded` |
| **Browser clients** | A malicious site calling the API from a user's browser | CORS allows only the dashboard's origins, only `GET`/`POST`, and only the `Content-Type` and `X-API-Key` headers. No credentials are shared. | `test_cors_allows_only_the_dashboard_origin` |
| `WS /ws` | Cross-site WebSocket hijacking | The handshake's `Origin` is checked against `CORS_ORIGINS`; foreign origins are closed with code 1008 | `test_websocket_refuses_foreign_origins` |
| **Responses** | Clickjacking, MIME sniffing, sensitive data cached or leaked through the Referer header | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`. On `/api/*` also: `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'` and `Cache-Control: no-store`. | `test_security_headers` |
| **Live news** (Google News RSS) | SSRF, slow or hostile feeds | The URL host is fixed and the query is URL-encoded. The place name comes from the hub registry, never raw user input. Each fetch has its own timeout, and the whole fetch step has a 3 s budget. Feed text is only scored, never executed or rendered as HTML. | `test_ml_routing.py::test_a_slow_feed_cannot_stall_a_request` |
| **Live weather** (Open-Meteo) | SSRF, slow or malformed replies | The URL host is fixed; the only parameters are the hub's rounded coordinates from the registry. Each fetch has its own timeout inside the same 3 s budget. The reply is parsed defensively (typed fields only), and any error means no weather, cached for five minutes. | `test_weather.py::test_a_malformed_reply_means_no_weather_and_is_cached` |
| **Concurrency** | One request's scenario leaking into another's results | Scenarios are looked up per request; the shared-state API was removed | `test_routing.py::test_concurrent_requests_do_not_leak_scenarios` |
| **Secrets** | Keys committed to the repository | `.env` is git-ignored (a broken ignore rule was repaired). The API key is read from the environment only. | — |

## Dependency audit (29 Sep 2026)

**Tools:** `pip-audit` 2.10.1 against `requirements.txt`, and `npm audit` for the frontend.

### Upgraded

| Package | From → to | Why |
|---|---|---|
| starlette | 1.0.0 → 1.3.1 | Host-header and URL reconstruction flaws in the request path |
| urllib3 | 2.6.3 → 2.7.0 | Used by the news fetcher |
| idna | 3.13 → 3.15 | Used by the news fetcher |
| anyio | 4.13.0 → 4.14.2 | Security fixes |
| pillow | 12.2.0 → 12.3.0 | Security fixes |
| transformers | 5.6.2 → 5.10.2 | Path traversal in `save_pretrained`. pip-audit suggested 5.10.0, but that release was later yanked, so we use its patch release 5.10.2. |
| **vite** | 5 → 6.4.3 | The esbuild dev server let any website read dev-server responses |

The full test suite passes on the upgraded versions.

**Later the same day:**

| Package | From → to | Why |
|---|---|---|
| torch | 2.11.0 → 2.13.0 | PYSEC-2025-194, memory corruption in `torch.jit.script`. The app never used TorchScript, but 2.13 removes the question. The threat-intelligence scores are identical on it. |
| setuptools | 81.0.0 → 83.0.0 | PYSEC-2026-3447, `MANIFEST.in` handling at build time. torch 2.11 had required `setuptools<82`; 2.13 doesn't. |

**Result:** `pip-audit` finds no known vulnerabilities, and `npm audit` reports 0.

Licences: Leaflet (BSD-2-Clause) was chosen over react-leaflet, whose Hippocratic licence is not OSI-approved. recharts is MIT.

To re-run the audits:

```
python -m pip install -r requirements-dev.txt
python -m pip_audit -r requirements.txt
cd frontend && npm audit
```

## Before a production deployment

- Terminate TLS in front of the API.
- Replace the single shared API key with per-user authentication.
- Key the rate limiter on the real client address, using a trusted-proxy setting behind a load balancer. The limiter is in memory, which is fine for one process; several workers need a shared store such as Redis.
- Add request logging and monitoring for 401 and 429 spikes.
- Serve the built frontend (`npm run build`) from a static host with its own CSP, instead of the Vite dev server.

## Reporting

Please report security issues privately to the maintainers instead of opening a public issue.
