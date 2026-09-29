# Model card: leg-delay quantiles and threat intelligence

Supplychainer uses two learned components:

1. **Delay quantile model.** Predicts the p50, p85 and p95 of the extra hours a leg takes beyond its nominal transit time.
2. **Contrastive NLP stage.** Scores news for disruption, classifies the threat type, and filters it by transport mode (CARF). Evaluate it with `python ml/evaluate_nlp.py`.

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

**There is no public dataset of per-leg freight delays, so the target is synthetic.** The README's earlier claim of "50,000+ real-world historical incidents" did not match the code: the original data came from generators in `Code/` (starter commit `8f15416`).

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

### Distance from the best possible model

The data comes from a known generator, so the best possible quantile of every leg can be computed exactly. `ml/delay_ceiling.py` does this: the delay is a mixture over the hidden incident state, weighted by how well each state explains the leg's news score. On 20,000 fresh legs:

| Quantile | Naive baseline | This model | Best possible | Share of the achievable gain |
|---|---|---|---|---|
| p50 | 8.70 | 7.73 | 7.63 | **90%** |
| p85 | 9.36 | 7.29 | 7.05 | **90%** |
| p95 | 6.45 | 4.42 | 4.29 | **94%** |

The rest of the loss is noise that no model can predict: whether an unreported incident is hiding behind a quiet news score, and the log-normal spread. Retraining on 200,000 legs, or with larger trees, moved these shares by 1–3 points in both directions, which is within sampling noise, so the pinned model was kept.

### How the router uses it

| Persona | Plans on | Why |
|---|---|---|
| FASTEST | p50 delay | typical outcome |
| BALANCED | p85 delay | buffered commitment |
| SAFEST | p95 delay | worst case, times the risk penalty |

Every route reports an ETA band (p50 / p85 / p95) for the **whole route**, estimated by Monte Carlo with 4,000 seeded samples:

- Each leg's delay is modelled as log-normal, fitted to that leg's p50 and p95.
- Legs are correlated through a Gaussian copula (ρ = 0.5), because delays on one route share weather, congestion and carrier performance.

Adding leg quantiles instead would assume every leg hits its bad case at once, and would make a route look worse just because it has more legs.

The typical ETA (`adjusted_eta`) adds each leg's median delay. The audit trace splits it into nominal transit, transfers, typical delay and scenario delay, and these parts add up exactly.

Planning is additive: for Dijkstra, SAFEST adds each leg's p95. That's a risk-averse heuristic. The reported band is the simulated one.

### Explainability

Each route has a `delay_drivers` breakdown of its p85 delay, made of **exact Shapley values**. There are only five features, so all 32 coalitions are evaluated; there's no sampling and no `shap` dependency.

- Each leg's contributions are measured against a reference leg: a short road hop into a terminal, in clear weather, with no news.
- Contributions add up exactly: `prediction(leg) − prediction(reference)`.
- A feature equal to its reference value contributes exactly 0 (tested).
- At route level, contributions are summed by factor. The response lists the five largest factors and pools the rest as "Other factors". Example: "SEA leg +120 h, Weather: rainy +30 h".
- `reference_hours` plus the drivers equals `total_hours` (tested).

### Integrity

The artifact's SHA-256 is **pinned in code** (`EXPECTED_SHA256` in `backend/engine/delay_model.py`), so write access to `Execution/` alone cannot swap the model. Retraining prints the new digest, and updating the pin is a reviewed code change.

`DelayQuantileModel.load` hashes the file's bytes and unpickles **those same bytes**. It refuses a mismatch before deserialising anything, because unpickling a tampered file can run arbitrary code.

If the file is missing or tampered with, the router falls back to nominal transit times and `/api/status` reports why. The legacy `risk_model.pkl` has been removed from the repository.

---

## 2. Threat intelligence (NLP)

- **Score:** each headline of four words or more is embedded with `BAAI/bge-small-en-v1.5` (MIT licence, about 130 MB). Its margin is its best cosine similarity to a *disaster* anchor minus the mean of its two best similarities to *safe* anchors, and the worst headline in a feed decides the score. Averaging two safe anchors stops a single topically close one ("Congestion eases...") from cancelling a real threat. The margin maps linearly to 0–1, starting at 0.08 and reaching 1 at 0.35. The resulting scale: Suez closure 0.96, dock strike 0.34, Red Sea attacks 0.32, routine berthing congestion 0.20. The standing air, rail and road reports score 0. The score feeds the delay model as incident severity.
- **Anchors:**
  - the original historical corpus, with **place and company names removed** (the named originals leaked location: every Rotterdam report resembled "Port of Rotterdam operating normally", so "Strike halts Rotterdam port operations" scored 0)
  - 37 short archetypes across the six threat types
  - 20 safe anchors: routine operations, business news, and news that a disruption has *ended*, which shares its words ("strike called off", "congestion eases")
- **Threat type:** learned by a logistic regression on the embeddings of the threat-type archetypes and the labelled dev and test disruptions: weather, labour, geopolitical, infrastructure, cyber or congestion. It never sees the holdout. The type is taken from **the same headline that set the score**. Weather reports set the delay model's `condition` feature: rainy from a score of 0.25, stormy from 0.6.
- **CARF:** a report is dropped for a leg when it names another mode's infrastructure or workforce and none of the leg's own ("dockworkers", "barge", "runway", "haulier"...). Mode-neutral news (weather, conflict, cyberattacks) applies to every mode. Ambiguous words stay out: "terminal" and "container" (every mode), "freighter" (ship or cargo plane), "docker" (also software), "anchorage" (also an air-cargo city).
- **Live news:** Google News RSS for the origin and destination cities, fetched only once the NLP engine is ready. The whole fetch has a 3 s budget. Results are cached for 15 minutes, and a failure is retried after five minutes. A report applies to every hub in its city. A leg shows the report that actually raised its threat, and a transfer is filtered for both modes it joins. With no network, there's no live signal; the static fallback texts are never passed off as news.

### Evaluation (288 labelled headlines)

`ml/nlp_headlines.csv` holds 288 headlines written for this evaluation. There are 138 disruptions across the six types, each tagged with the transport modes it concerns, and 150 pieces of routine or positive news. That includes hard cases like "Union and port employers strike a deal, averting walkout" and "Congestion eases at Los Angeles as vessel queue clears". There are three splits of 96:

- **dev:** used to choose the embedding model, the anchors and the threshold
- **test:** scored after that tuning; one keyword was then removed after seeing its errors
- **holdout:** written afterwards, and scored once, after the threat-type classifier had learned from the dev and test disruptions. Nothing ever fits on it; a test checks that.

`python ml/evaluate_nlp.py` reproduces the table, and a test fails if the holdout scores drop.

| | Before: original engine, holdout | **After: holdout** | After: test | After: dev |
|---|---|---|---|---|
| Detection AUC | 0.86 | **0.97** | 0.99 | 1.00 |
| Disruptions detected (recall) | 70% | **83%** | 91% | 98% |
| False alarms on routine news | 10% | **8%** | 2% | 0% |
| Precision | 0.87 | **0.91** | 0.98 | 1.00 |
| Threat-type accuracy | 76% | **94%** | 98%* | 100%* |
| CARF: right decision per headline and mode | 93% | **95%** | 94% | 100% |

\*In-sample: the type classifier learns from these splits.

**The holdout is the figure to quote.** The test split looked better (91% recall, 2% false alarms) because the same person wrote dev and test in a similar style, and one change followed its errors. The fresh holdout is harder.

How the gains came:

1. Broader anchors lifted dev AUC from 0.82 to 0.95 with MiniLM.
2. On dev, `bge-small-en-v1.5` (0.987) beat `all-MiniLM-L6-v2` (0.954), `all-MiniLM-L12-v2` (0.971) and `gte-small` (0.987, with a narrower margin spread) using the same anchors.
3. Averaging the two closest safe anchors took dev AUC to 0.992.
4. Threat type is learned: a logistic regression on the embeddings of the archetypes and the 92 labelled dev and test disruptions. Trained on dev and scored on test, then the reverse, it averaged 95% against 90% for the nearest archetype. The regularisation was set the same way. On the holdout, type accuracy went from 87% (nearest archetype) to 94%.

**Holdout misses:** 8 of 46. Four are congestion reports ("vessel queue climbs past 200 ships", "air freight backlog stretches to ten days"), which sit near the "congestion eases" safe anchor. The others are an ice storm at Memphis, a dockers' blockade, airspace avoidance and a sinkhole. **False alarms:** 4 of 50, mostly news that a disruption has ended ("Typhoon passes without damage as Hong Kong port reopens", "Container dwell times at Los Angeles fall to record lows"). Detection was not tuned again after this run.

### Limitations

- **The delay target is simulated.** The model learns the generator's physics priors, not real carrier data. The pipeline (features, constraints, evaluation) is built so it can be retrained on real AIS or port-call data without code changes.
- **Sea legs are slightly under-covered at p85** (82.4% vs 85%), from the fat sea tail.
- **The NLP score says how clearly a report describes a disruption, not how severe it is.** A full closure still scores near 1, but a Red Sea missile attack (0.32) and a dock strike (0.34) come out close together.
- **The evaluation headlines were written by the team, with AI assistance,** in the style of real logistics news, not sampled from a live feed. The holdout was written after all tuning, but by the same authors, so real news may score somewhat lower still.
- **Live news is fetched only for the origin and destination.** Intermediate chokepoints are covered by the scripted scenarios.
- **Within one ocean basin, sea distances are straight lines.** Crossings between basins go through their real straits.
