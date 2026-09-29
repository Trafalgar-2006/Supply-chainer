# Bug fixes

Each bug below was confirmed against the original code, then fixed. Every fix has
a regression test in `tests/`. "Before" numbers come from the original commit
(`8f15416`); "After" numbers come from the fixed code. Run `pytest` to check them.

## Headline results

| Check | Before | After |
|---|---|---|
| NLP score for *"Container ship ran aground in the Suez Canal, blocking all traffic…"* | 0.00 | **0.96** |
| CARF: ship news on a sea leg / airport news on a sea leg | 0.0 / 0.8 (inverted) | **0.8 / 0.0** |
| Ports that can reach each other by sea | 32 of 175 | **every seaport except the 2 landlocked Caspian ports** |
| `SUEZ_BLOCK`, Shanghai → Rotterdam (balanced route) | Waits at the blocked canal: +240 h | **Sails around the Cape of Good Hope: no blockage delay, about 170 h of extra sailing** |
| `RED_SEA_CONFLICT`, same route | 0 h delay (the lane skipped Bab el-Mandeb) | **+72 h and threat 0.85 via the Red Sea; the SAFEST route goes around the Cape** |
| `HORMUZ_CLOSURE`, Mumbai → Rotterdam | Sailed into Jebel Ali, inside the Gulf, with 0 h delay | **Stays out of the Gulf** |
| `LA_PORT_STRIKE`, Shanghai → Los Angeles by sea | No effect possible: LA and Long Beach had no shipping lanes | **Normally lands at Long Beach; during the strike every option diverts via Oakland and trucks south** |
| `CHENNAI_FLOOD`, Chennai → Singapore | 0 h delay, threat 0.05 | **+48 h, threat 0.75** |
| Explanation text | "reduces total landed cost by **396%**" | Real comparison against the other routes returned, e.g. "N% cheaper than the fastest route, +X h on its ETA" |
| Supplier lead time under `SUEZ_BLOCK` (SUP-GLOBAL-01) | 14 → 19 days (50%) | 14 → **15 days** (the documented 10%) |
| Start-up warm-up | over 5 min (hidden while NLP was off) | **~7 s** |

## Threat intelligence

| # | Bug | Root cause | Fix | Test |
|---|---|---|---|---|
| 1 | Every NLP score was 0 on machines without a GPU | `nlp_anchors.pt` holds CUDA tensors and `torch.load` had no `map_location`. The load failure was caught and the engine switched itself off. | Load with `map_location="cpu", weights_only=True` | `test_nlp_scoring.py` |
| 2 | Real threats scored 0; safe text scored negative | The noise-floor check was inverted (`margin >= floor → 0`) | Flipped the check and clamped the score to 0–1 | `test_nlp_scoring.py` |
| 3 | Even a full Suez closure capped at a score of 0.2 | A 0.35 multiplier (the author's prototype used 3.5) | Linear scale from the noise floor (0.10) to saturation (0.50), set from measured margins on labelled headlines (table below) | `test_severity_ordering` |
| 4 | CARF dropped relevant news and kept irrelevant news | Both mode checks were inverted, and rail and road were never checked | A threat is dropped only when the news names another mode's infrastructure and none of the leg's own. All 4 modes are covered, with whole-word matching (so "airport" no longer counts as "port"). | `test_carf.py` |
| 5 | Model files didn't load unless the server was started from the repo root | Paths were relative to the working directory | Paths are now resolved from the package location | `test_model_loading.py` |
| 6 | Start-up warm-up took over 5 minutes | The same 4 fallback reports were run through the transformer once per edge (about 5,600 times) | Score each mode's report once | `test_warmup.py` |

NLP calibration after the fixes, including the later removal of place names from
the anchors (see `docs/MODEL_CARD.md`). Margin = best disaster-anchor similarity
minus best safe-anchor similarity. Scores start at a margin of 0.10 and reach 1 at 0.50.

| Text | Margin | Old score (×0.35) | New score |
|---|---|---|---|
| Suez grounding, all traffic blocked | +0.49 | 0.20 | 0.96 |
| Red Sea missile attacks, rerouting | +0.39 | 0.17 | 0.72 |
| Dock workers strike | +0.34 | 0.09 | 0.61 |
| Sea fallback: "Berthing delays expected" | +0.15 | 0.09 | 0.11 |
| Minor festival traffic | +0.14 | 0.05 | 0.09 |
| "Operations proceeding normally" | −0.01 | 0.00 | 0.00 |

## Network and data

| # | Bug | Fix | Test |
|---|---|---|---|
| 7 | Transit lanes were built in one direction only. Most lanes are listed on just one of their two hubs, so 133 of 175 ports had no way in by sea, and Hormuz, Bab el-Mandeb, Malacca and the Cape could never be entered. | Every listed lane is built in both directions | `test_every_*_lane_is_navigable_both_ways` |
| 8 | Sea lanes crossed land. Jebel Ali → Haifa sailed across Arabia, skipping Hormuz, Bab el-Mandeb and Suez, so those scenarios had no effect. | Lanes between basins are routed through their real straits, taking the shortest real sequence. Lanes out of the landlocked Caspian are dropped. | `test_enclosed_seas_are_only_reachable_through_their_straits` |
| 9 | Road and rail lanes crossed open sea (e.g. rail Riyadh → Port Sudan across the Red Sea, which appeared in a Shanghai → Rotterdam route) | Removed 18 impossible rail lanes. Road and rail must stay on one landmass; Britain connects to the continent only via the Channel Tunnel. `tools/audit_land_lanes.py` checks every land lane against a 1 km land mask. | `test_impossible_land_lanes_are_absent` |
| 10 | 13 major ports had no shipping lanes at all: Los Angeles, Long Beach, New York, Houston, Vancouver, Santos, Callao, Arica, Abidjan, Sydney, Melbourne, Fremantle and Tauranga. No ship could reach them, so `LA_PORT_STRIKE` could never apply. | Added 51 liner lanes that match real services (trans-Pacific, transatlantic, intra-regional) | `test_every_seaport_outside_the_caspian_is_on_one_navigable_ocean`, `test_la_port_strike_affects_ships_bound_for_los_angeles` |
| 11 | `HUB-CHICAGO` was defined twice; the Elk Grove entry overwrote the Chicago DC | Renamed the second entry to `HUB-ELKGROVE` | `test_hub_ids_are_unique` |
| 12 | 30 air lanes pointed at `AIR-CHENNAI`, which didn't exist | Added Chennai International Airport | `test_every_connection_targets_an_existing_hub` |
| 13 | 4 links pointed at rail hubs that don't exist; 3 hubs had self-loops | Added the missing junctions and removed the self-loops | `test_data_integrity.py` |

## Routing and API

| # | Bug | Fix | Test |
|---|---|---|---|
| 14 | A disruption at the origin was ignored; a transfer inside a disrupted hub was charged a second time | Each disrupted hub is charged once per route, on the first transit leg that touches it | `test_scenario_delay_is_charged_once_per_disrupted_hub`, `test_disrupted_origin_is_reported` |
| 15 | Concurrent requests could see each other's scenario: `/api/suppliers` changed the shared scenario, and routing read it | Scenarios are looked up per request | `test_concurrent_requests_do_not_leak_scenarios`, `test_supplier_endpoint_does_not_change_global_scenario` |
| 16 | The audit trace didn't add up: scenario delay was counted twice, the risk premium was shown but not billed, and baseline risk included the scenario | The trace parts now add up exactly to the ETA and cost | `test_audit_trace_adds_up_to_the_totals` |
| 17 | Explanations used made-up figures ("396%") | Real trade-offs computed against the other routes returned | `test_explanation_percentages_are_valid` |
| 18 | The PREFERRED ("soft bias") policy behaved exactly like "any" | Non-preferred modes are weighted 1.5×; first- and last-mile road hops are exempt | `test_preferred_policy_biases_towards_the_preferred_mode` |
| 19 | The supplier penalty used 50% of the delay; the documented rule is 10% | Applied the documented 10% | `test_suez_block_adds_documented_lead_time_penalty` |
| 20 | `ml_trained` was hard-coded to `true` | The real model load state is reported | — |
| 21 | Hub search broke on names containing `&`, `#` or `?`, and slow responses could overwrite newer ones | The query is URL-encoded and stale responses are ignored | `tools/ui_smoke.py` |
| 22 | A route whose origin and destination were the same hub came back with no legs, which the dashboard could not draw | Rejected with a clear error | `test_same_origin_and_destination_is_an_error_not_an_empty_route` |
| 23 | Solapur Freight Terminal had no connections, so no route could start or end there (found by a 1,500-request randomised sweep) | Connected by rail and road to Pune and Hyderabad, as on the real Central Railway line and NH65 | `test_every_origin_can_reach_and_be_reached_from_every_other` |
| 24 | Hub search returned every match unranked (about 150 for "po") and offered canals and straits as origins | Best matches first, at most 12, no chokepoints; routing rejects a chokepoint as an origin or destination | `test_hub_search_offers_the_best_matches_first_and_no_chokepoints`, `test_a_chokepoint_is_not_an_origin_or_destination` |
| 25 | A supplier's cost score was `1 - price / $1,000`, so every Raw Materials supplier (over $1,000 a unit) scored below zero | Price relative to the cheapest supplier in the category, always 0-1; the ranking order is unchanged | `test_scores_stay_between_zero_and_one_at_any_price_level` |
| 26 | 8 "flights" were shorter than a truck ride, e.g. Memphis hub to Memphis airport (14 km) and Al Maktoum to Dubai International (45 km); under the Dubai surge the fastest route from Dubai to Mumbai flew via Frankfurt and Doha | Lanes under 200 km are never flown (the road link already exists), and Al Maktoum, Dubai's cargo airport, gained its main routes to India and Asia | `test_no_flight_is_shorter_than_a_truck_ride` |
| 27 | The Dubai surge said "48h clearance backlog" but charged 24 h, and named both Dubai airports while disrupting one | 48 h, and the description names the one airport it disrupts | `test_scenario_delay_is_charged_once_per_disrupted_hub` |

## Repository

- `frontend/node_modules` (7,863 files) and a stale `frontend/dist` build were committed. Both are now untracked and ignored.
- A missing trailing newline merged `.env` and `node_modules/` into one `.gitignore` rule. That rule is repaired.
- `uvloop` has no Windows build and broke `pip install` there. It is now installed only on other platforms.
