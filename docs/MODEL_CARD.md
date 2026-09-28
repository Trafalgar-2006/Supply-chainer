# Model card: leg-delay quantiles and threat intelligence

Supplychainer uses two learned components:

1. **Delay quantile model.** Predicts the p50, p85 and p95 of the extra hours a leg takes beyond its nominal transit time.
2. **Contrastive NLP stage.** Scores news for disruption, classifies the threat type, and filters it by transport mode (CARF).

Retrain with:

```
python ml/train_delay_model.py
```

Training is deterministic: seed 42, about 20 s on a laptop. The run rewrites `Execution/delay_quantile_model.joblib` and its report, `delay_quantile_model.json`.

---

## 1. Delay quantile model

### Why it was replaced

The shipped `risk_model.pkl` was a p85 regressor keyed on 16 hub *names* ("Mumbai Port", "Atlanta Air Hub", …). None of those names exist in the routing graph, whose IDs look like `PORT-SHANGHAI`. So every live hub fell back to the same encoding, and distance was not a feature at all.

| Leg | Old model p85 |
|---|---|
| PORT-SHANGHAI → PORT-ROTTERDAM (sea) | 24.16 h |
| PORT-MUMBAI → PORT-JEBEL (sea) | 24.16 h |

The model was loaded at start-up but never called by the router.

### Data

**There is no public dataset of per-leg freight delays, so the target is synthetic.** The README's earlier claim of "50,000+ real-world historical incidents" did not match the code: the original data came from generators in `Code/`.

`ml/generate_delay_dataset.py` keeps the original generator's physics priors. It changes one thing: legs are **sampled from the live routing graph**, so training and serving see the same distribution. Each sample draws a real edge (mode, distance, what it arrives at), then weather and an incident.

| Parameter | Value | Source |
|---|---|---|
| Dwell at an arrival terminal | sea 48 h, air 6 h, rail 12 h, road 2 h | original generator ("ground friction") |
| Dwell at a canal or strait | 25% / 2% of the terminal dwell | ~12 h canal convoy queue, ~1 h strait passage |
| Transit variability | 10% of transit time | original generator |
| Weather (clear / rainy / stormy) | probability 0.6 / 0.3 / 0.1; ×1.3 rain; ×4 storm for sea and air, ×2 for land | original generator |
| Incident | probability 8%; severity U(0.3, 1); delay × (1 + 4·severity) | strikes, closures, groundings |
| News signal | tracks the incident severity with noise; misses 15% of incidents; 3% false alarms | so the model learns how far to trust the NLP stage |
| Noise | log-normal, σ 0.6 (land) / 0.9 (sea, air) | original generator (fat tails) |

The dataset has 60,000 legs, split 80/20 for train and test.

### Features

The features exist for every edge, so the model generalises to all 444 hubs:

- `mode` (categorical)
- `distance_km`
- `arrival` (terminal / canal / strait, categorical)
- `condition` (clear / rainy / stormy)
- `nlp_score` (0–1)

### Model

There is one `HistGradientBoostingRegressor(loss="quantile")` per quantile. Settings:

- native categorical handling
- up to 400 iterations with early stopping
- learning rate 0.05
- 31 leaves per tree
- **Monotonic constraints:** more distance, worse weather or a stronger news signal can never lower a predicted delay.

At prediction time the model enforces `p50 ≤ p85 ≤ p95` and clips delays at zero.

### Evaluation (12,000 held-out legs)

| Quantile | Coverage (target) | Coverage by mode (air / rail / road / sea) | Pinball loss | Naive baseline* | Improvement |
|---|---|---|---|---|---|
| p50 | **49.2%** (50%) | 51.0 / 50.7 / 48.9 / 47.1 | 8.01 | 9.10 | **12%** |
| p85 | **84.5%** (85%) | 86.5 / 84.7 / 84.3 / 82.4 | 7.87 | 10.08 | **22%** |
| p95 | **95.1%** (95%) | 95.4 / 94.5 / 95.6 / 93.6 | 5.17 | 7.24 | **29%** |

\*The naive baseline is the empirical quantile of each (mode, arrival) group in the training set.

- Before the ordering is enforced, the quantiles cross on 0.06% of legs.
- Permutation importance for p85 (increase in pinball loss when the feature is shuffled): mode 11.0 > weather 2.2 > news 1.4 > arrival 1.0 > distance 0.6.

### How the router uses it

| Persona | Plans on | Why |
|---|---|---|
| FASTEST | p50 delay | typical outcome |
| BALANCED | p85 delay | buffered commitment |
| SAFEST | p95 delay | worst case, times the risk penalty |

Every route reports an ETA band: p50 / p85 / p95. The band adds up leg quantiles, which assumes delays on one route move together. That makes p85 and p95 a **conservative upper bound**, which suits a planning buffer. The audit trace separates nominal transit, transfers, typical delay (p50) and scenario delay, and these add up exactly to the ETA.

### Explainability

Each route has a `delay_drivers` breakdown of its p85 delay, made of **exact Shapley values**. There are only five features, so all 32 coalitions are evaluated; there's no sampling and no `shap` dependency.

- Each leg's contributions are measured against a reference leg: a short road hop into a terminal, in clear weather, with no news.
- Contributions add up exactly: `prediction(leg) − prediction(reference)`.
- A feature equal to its reference value contributes exactly 0 (tested).
- At route level, contributions are summed by factor. Example: "SEA leg +120 h, Weather: rainy +30 h".

### Integrity

The report stores the artifact's SHA-256. `DelayQuantileModel.load` checks it **before** unpickling, because unpickling a tampered file can run arbitrary code. If the file is missing or tampered with, the router falls back to nominal transit times and `/api/status` reports why.

---

## 2. Threat intelligence (NLP)

- **Score:** each headline is embedded with `all-MiniLM-L6-v2`. Its margin is the best cosine similarity to a *disaster* anchor minus the best similarity to a *safe* anchor, and the worst headline in a feed decides the score. The margin maps linearly to 0–1, starting at 0.10 and reaching 1 at 0.35.
- **Anchors:** the original historical corpus with **place and company names removed**, plus category archetypes. The named originals leaked location: every Rotterdam report resembled "Port of Rotterdam operating normally", so "Strike halts Rotterdam port operations" scored 0. The safe anchors also gained positive business news, such as new capacity and earnings.
- **Measured performance:**
  - AUC 0.995 separating safe from disrupted headlines, on a labelled set of 33.
  - On 14 held-out headlines never used for tuning: **no false alarms, 6 of 8 disruptions detected.** The misses were a bridge collapse and a cyclone, both just under the floor.
- **Threat type:** zero-shot, classified by nearest category archetype: weather, labour, geopolitical, infrastructure, cyber or congestion. Weather reports set the delay model's `condition` feature: rainy, or stormy when the score is 0.5 or higher.
- **CARF:** a report is dropped for a leg when it names another mode's infrastructure and none of the leg's own. Mode-neutral news (weather, conflict) applies to every mode.
- **Live news:** Google News RSS for the origin and destination cities. Fetches use a 2 s timeout, results are cached for 15 minutes, and a failure is retried after a minute. A report applies to every hub in its city. With no network, there's no live signal; the static fallback texts are never passed off as news.

### Limitations

- **The delay target is simulated.** The model learns the generator's physics priors, not real carrier data. The pipeline (features, constraints, evaluation) is built so it can be retrained on real AIS or port-call data without code changes.
- **Sea legs are slightly under-covered at p85** (82.4% vs 85%), from the fat sea tail.
- **The NLP score says how clearly a report describes a disruption, not how severe it is.** It doesn't reliably rank a strike below a full closure (pairwise AUC ≈ 0.6).
- **Live news is fetched only for the origin and destination.** Intermediate chokepoints are covered by the scripted scenarios.
- **Within one ocean basin, sea distances are straight lines.** Crossings between basins go through their real straits.
