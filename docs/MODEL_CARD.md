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

Each headline of four words or more is embedded with `BAAI/bge-small-en-v1.5` (MIT licence, about 130 MB). The most threatening headline in a feed decides the report's score, severity and type. Three parts are learned from 288 labelled synthetic headlines in `ml/nlp_headlines.csv`.

- **Detection:** the margin is the best cosine similarity to a *disaster* anchor minus the mean of the two best similarities to *safe* anchors. Averaging two stops a single topically close anchor ("Congestion eases...") from cancelling a real threat. A headline counts as a disruption when its margin exceeds 0.047, the threshold with the best balanced accuracy on out-of-fold margins.
  - **Disaster anchors:** the starter's historical incidents with place names removed, 37 threat archetypes, and the 138 labelled disruptions.
  - **Safe anchors:** routine operations, 15 business and recovery archetypes ("strike called off", "congestion eases"), and the 150 labelled routine headlines.
  - **Proper nouns are stripped from the labelled examples,** as they were from the original anchors. Otherwise "Singapore port throughput rises" resembles "Vessel queue outside Singapore anchorage grows" and reads as a threat.
- **Severity:** a ridge regression predicts severity 1 (minor), 2 (significant) or 3 (severe) from the embedding. A disruption scores 0.2, 0.6 or 1.0, interpolated, and that score feeds the delay model as incident severity.
- **Threat type:** a logistic regression on the embeddings of the archetypes and the labelled disruptions: weather, labour, geopolitical, infrastructure, cyber or congestion. Weather reports set the delay model's `condition` feature: rainy from a score of 0.25, stormy from 0.6.
- **Standing reports** (the per-mode fallback texts every leg carries when there's no live news) are capped at minor (0.2). They describe routine conditions, not incidents.
- **CARF:** a report is dropped for a leg when it names another mode's infrastructure or workforce and none of the leg's own ("dockworkers", "barge", "runway", "haulier"...). Mode-neutral news (weather, conflict, cyberattacks) applies to every mode. Ambiguous words stay out: "terminal" and "container" (every mode), "freighter" (ship or cargo plane), "docker" (also software), "anchorage" (also an air-cargo city).
- **Live news and weather:** Google News RSS for the origin and destination cities on each request, and for all 15 chokepoints through a background watch refreshed every 15 minutes; current weather for the same places from Open-Meteo (thunderstorms, heavy snow or gale-force wind count as stormy; fog, rain or snow as rainy). Both are fetched only once the NLP engine is ready. A headline counts only if it was published in the last three days and names the place or one of its hubs (a search for Los Angeles can return Middle East news), and at most the three newest count. The whole fetch has a 3 s budget. Results are cached for 15 minutes, and a failure is retried after five minutes. A report applies to every hub in its city. A leg shows the report that actually raised its threat, and a transfer is filtered for both modes it joins. With no network, there's no live signal; the static fallback texts are never passed off as news.

### Evaluation on real news

`ml/nlp_real_headlines.csv` holds **188 real headlines** fetched from Google News on 29 Sep 2026 across 30 disruption and routine logistics topics. Each carries its publisher and date. They were labelled against written rules before the engine was run on them, and the engine never reads this file (a test checks it):

- **Disruption:** a current or scheduled disruption to freight, meaning a strike under way or on a set date, a closure, an attack on shipping, congestion, a blocking accident, a cyberattack stopping operations, or weather closing ports or lines.
- **Routine:** business, markets, openings, recoveries, averted strikes, and court cases about past events.
- **Dropped:** anything not about freight (house fires, war news on land), speculation ("could truckers strike?"), opinion pieces, and near-duplicate stories.

That leaves 91 disruptions (with type, severity 1–3 and modes) and 97 routine headlines. Where cause and effect differ ("typhoon deepens port congestion"), both types count.

`python ml/evaluate_nlp.py` reproduces the table, and a test fails if the real-news scores drop.

| Real news (188 headlines) | Original engine | After round 2 | **Now** |
|---|---|---|---|
| Detection AUC | 0.86 | 0.94 | **0.97** |
| Disruptions detected (recall) | 68% | 78% | **82%** |
| False alarms on routine news | 17% | 8% | **6%** |
| Precision | 0.80 | 0.90 | **0.93** |
| Severity ranking (Spearman vs labels) | 0.24 | 0.14 | **0.55** |
| Threat type correct | 76% | 91% | 89% |
| CARF: right decision per headline and mode | 88% | 92% | 92% |

The original engine is the starter's all-MiniLM-L6-v2 engine; round 2 is bge-small with the broader anchors. Type accuracy moved by two headlines, within noise.

**The real set was scored twice, and we say so.** The first run of the learned engine showed false alarms on court cases about the Baltimore bridge and on Los Angeles and Singapore port news, all matching labelled examples that named those places. That's the location leakage this project had already fixed once in the anchors. Stripping proper nouns from the examples was re-validated by cross-validation on the synthetic data before the second run. No threshold or setting was changed to fit the real set.

**Remaining errors:** 6 false alarms, mostly negations and court cases ("No national truckers' strike Thursday", "Houthis promise not to target European ships"), which sentence embeddings struggle to read. 16 missed disruptions, including multi-mode strike roundups and flooding that closed rail lines.

**How each part was chosen**, all by 3-fold cross-validation over the synthetic splits (train on two, score the third):

- **Model:** `bge-small-en-v1.5` beat `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2` and `gte-small`.
- **Detection:** anchors plus place-stripped examples beat anchors alone, raw examples, a logistic regression and nearest-neighbour voting. It scored AUC 0.994, 95.7% recall and 1.3% false alarms.
- **Severity:** ridge regression (α = 0.1, flat from 0.03 to 0.5) reached a rank correlation of 0.63, against 0.27 for the old margin ramp.
- **Type:** logistic regression reached 0.94, against 0.90 for the nearest archetype.
- **CARF:** a learned fallback for headlines that name no mode raised accuracy only from 96.2% to 97.1%, and dropped more genuine threats, so the keyword rule stayed.

The synthetic splits (dev, test, holdout) were used in turn while the engine was built, and now all train it, so their scores are in-sample.

### Limitations

- **The delay target is simulated.** The model learns the generator's physics priors, not real carrier data. The pipeline (features, constraints, evaluation) is built so it can be retrained on real AIS or port-call data without code changes.
- **Sea legs are slightly under-covered at p85** (82.4% vs 85%), from the fat sea tail.
- **Severity is only moderately reliable** (rank correlation 0.55 on real news). The model reads a port-shutting strike as severe, and can rate a routine congestion headline as significant; standing reports are capped at minor for that reason.
- **Labels were written by the team, with AI assistance.** The 288 synthetic training headlines were written in the style of logistics news; the 188 real test headlines are genuine but were labelled by the same authors. Another labeller would disagree on some borderline cases, especially severity.
- **Live news covers the origin, the destination and the 15 chokepoints,** not every port or depot on a route; those rely on the scripted scenarios and standing reports.
- **Within one ocean basin, sea distances are straight lines.** Crossings between basins go through their real straits.
