from concurrent.futures import ThreadPoolExecutor, wait

import networkx as nx
import numpy as np

from .delay_features import CONDITIONS, FEATURES, arrival_kind
from .delay_model import DelayQuantileModel, label as feature_label
from .multimodal_network import MODE_PROFILES, create_multimodal_network
from .threat_intelligence import ContrastiveNLPEngine, CARFFilter
from .news_ingestion import DynamicNewsIngestor
from .node_resolver import NodeResolver

PREFERENCE_PENALTY = 1.5  # weight multiplier on non-preferred modes under PREFERRED policy
# Delay quantile each persona plans on: typical (p50), buffered (p85), worst case (p95).
PERSONA_QUANTILE = {"FASTEST": 0, "BALANCED": 1, "SAFEST": 2}
QUANTILE_NAMES = ("p50", "p85", "p95")
# BALANCED time/cost weights by shipment priority; the risk weight stays 0.2.
BALANCED_WEIGHTS = {"low": (0.2, 0.6), "normal": (0.3, 0.5), "urgent": (0.5, 0.3)}
NO_DELAY = (0.0, 0.0, 0.0)
# Explanations call two routes equal when they differ by less than this.
SAME_TIME_H = 1.0
SAME_COST_RATIO = 0.02
# Route ETA band: legs are log-normal (fitted to each leg's p50 and p95) and share
# a Gaussian-copula correlation, since delays on one route share weather,
# congestion and carrier performance.
LEG_CORRELATION = 0.5
BAND_SAMPLES = 4000
Z95 = 1.6448536
# Live news: whole-fetch time budget, and the threat scores at which a weather
# report sets the delay model's weather feature to rainy / stormy.
LIVE_INTEL_TIMEOUT_S = 3.0
RAIN_SCORE, STORM_SCORE = 0.25, 0.6

class RouteRecommender:
    """
    Supplychainer Unified Multimodal Optimization Engine.
    V8: Virtual-Node Forensic Edition.
    """

    def __init__(self, network, scenario_mgr, demo_mode=False):
        self.scenario_mgr = scenario_mgr
        self.demo_mode = demo_mode
        self.is_warmed_up = False
        self.warmup_failed = False

        self.nlp = ContrastiveNLPEngine(lazy_load=True)
        self.carf = CARFFilter()
        self.news_ingestor = DynamicNewsIngestor()
        self.resolver = NodeResolver()

        print(f"[STARTUP] Initializing Split-Node Global Topology...")
        self.unified_graph = network if network is not None else create_multimodal_network()
        self._hub_nodes = {}
        self._city_hubs = {}
        for n, data in self.unified_graph.nodes(data=True):
            self._hub_nodes.setdefault(data["physical_id"], []).append(n)
            city = data.get("parent_city") or data["display_name"]
            self._city_hubs.setdefault(city, set()).add(data["physical_id"])

        self.delay_model_error = None
        self.delay_model = self._load_delay_model()
        self._refresh_edge_delays()

        if self.demo_mode:
            self.is_warmed_up = True

        print(f"[STARTUP] Unified Engine Ready.")

    def run_background_warmup(self):
        if self.is_warmed_up: return
        print("[WARMUP] Calibrating global threat floor...")
        try:
            self.nlp.warmup()

            # Enrich unified graph with baseline intelligence. Every edge of a mode
            # shares that mode's baseline report, so score each report once rather
            # than running the transformer for all ~5,600 edges.
            baseline = {}
            for mode, news in self.news_ingestor.fallback_news.items():
                score = self.nlp.get_semantic_score(news)
                baseline[mode] = (self.carf.apply_filter(score, news, mode), news)
            for u, v, d in self.unified_graph.edges(data=True):
                mode = d.get("transport_mode", "road")
                if mode == "transfer": continue
                threat, news = baseline.get(mode, (0.0, "Normal conditions."))
                d["base_threat"] = threat
                d["base_news"] = news

            self.is_warmed_up = True
            print("[WARMUP] Unified Calibration Complete.")
        except Exception as e:
            print(f"[WARMUP] Error during warmup: {e}")
            self.warmup_failed = True

    # ---- Delay model --------------------------------------------------------

    def _load_delay_model(self):
        try:
            return DelayQuantileModel.load()
        except Exception as e:
            # A missing or tampered artifact must not take routing down: route on
            # nominal transit times and report the reason in /api/status.
            self.delay_model_error = str(e)
            print(f"[DELAY MODEL] Unavailable, routing on nominal transit times: {e}")
            return None

    def _leg_features(self, u, v, d, intel=None):
        """Delay-model features of a transit edge, with live intel for either end."""
        G = self.unified_graph
        nlp_score, condition = 0.0, "clear"
        for h in dict.fromkeys((G.nodes[u]["physical_id"], G.nodes[v]["physical_id"])):
            if intel and h in intel:
                report = intel[h]
                nlp_score = max(nlp_score, self.carf.apply_filter(report["score"], report["headlines"], d["transport_mode"]))
                condition = max(condition, report["condition"], key=CONDITIONS.index)
        return {"mode": d["transport_mode"], "distance_km": d["distance"],
                "arrival": arrival_kind(G.nodes[v]["physical_id"]),
                "condition": condition, "nlp_score": nlp_score}

    def _predict_delays(self, edges, intel=None):
        """Delay quantiles for (u, v, data) transit edges as {(u, v): (p50, p85, p95)}."""
        if self.delay_model is None or not edges:
            return {(u, v): NO_DELAY for u, v, _ in edges}
        features = [self._leg_features(u, v, d, intel) for u, v, d in edges]
        q = self.delay_model.predict(*[[f[k] for f in features] for k in FEATURES])
        return {(u, v): tuple(float(x) for x in row) for (u, v, _), row in zip(edges, q)}

    def _refresh_edge_delays(self):
        # Baseline for every lane: clear weather and no live news. Request-time
        # intel overrides the lanes around the hubs it concerns.
        G = self.unified_graph
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "transit"]
        for (u, v), q in self._predict_delays(edges).items():
            G[u][v]["delay_q"] = q

    def _delay_drivers(self, leg_features):
        """Exact Shapley breakdown of the p85 delays of a route's legs, summed by factor.

        reference_hours + the drivers equals total_hours exactly (up to rounding):
        the five largest factors are listed and the rest are pooled.
        """
        if self.delay_model is None or not leg_features:
            return None
        reference, phi = self.delay_model.explain(leg_features, quantile_index=1)
        by_label = {}
        for leg, contributions in zip(leg_features, phi):
            for feature, hours in zip(FEATURES, contributions):
                name = feature_label(feature, leg[feature])
                by_label[name] = by_label.get(name, 0.0) + float(hours)
        ranked = sorted(by_label.items(), key=lambda kv: -abs(kv[1]))
        drivers = [{"factor": k, "hours": round(v, 1)} for k, v in ranked[:5]]
        rest = sum(v for _, v in ranked[5:])
        if abs(rest) >= 0.05:
            drivers.append({"factor": "Other factors", "hours": round(rest, 1)})
        return {
            "quantile": "p85",
            "reference_hours": round(reference * len(leg_features), 1),
            "reference": "one short road hop per leg, clear weather, no news",
            "drivers": drivers,
            "total_hours": round(reference * len(leg_features) + float(phi.sum()), 1),
        }

    @staticmethod
    def _eta_band(fixed_hours, leg_quantiles):
        """p50/p85/p95 of a route's door-to-door hours, by Monte Carlo.

        Adding each leg's p85 or p95 would assume every leg hits its bad case at
        once and would make routes with more legs look worse; sampling the sum
        gives the route's own quantiles. Seeded, so identical requests agree.
        """
        q = np.asarray([lq for lq in leg_quantiles if lq[2] > 1e-6], dtype=float).reshape(-1, 3)
        if not len(q):
            return {name: round(fixed_hours, 1) for name in QUANTILE_NAMES}
        p50 = np.maximum(q[:, 0], 1e-3)
        mu = np.log(p50)
        sigma = np.maximum(np.log(np.maximum(q[:, 2], p50) / p50) / Z95, 1e-6)
        rng = np.random.default_rng(0)
        shared = rng.standard_normal((BAND_SAMPLES, 1))
        z = np.sqrt(LEG_CORRELATION) * shared + np.sqrt(1 - LEG_CORRELATION) * rng.standard_normal((BAND_SAMPLES, len(q)))
        totals = fixed_hours + np.exp(mu + sigma * z).sum(axis=1)
        return dict(zip(QUANTILE_NAMES, (round(float(v), 1) for v in np.percentile(totals, [50, 85, 95]))))

    # ---- Live intelligence --------------------------------------------------

    def _live_intel(self, places):
        """Live news per hub for the given place names.

        A report about a city applies to every hub in it (port, rail yard, airport,
        distribution centre), so a port-closure story reaches the ships even when
        the route enters the city through a road depot. News is only fetched once
        the NLP engine can score it, and the whole fetch has a time budget.
        """
        if not self.nlp.ready:
            return {}  # news that cannot be scored is not worth a network round trip
        places = sorted(set(places))
        pool = ThreadPoolExecutor(max_workers=max(1, len(places)))
        futures = {pool.submit(self.news_ingestor.fetch_headlines, p): p for p in places}
        done, _ = wait(futures, timeout=LIVE_INTEL_TIMEOUT_S)
        pool.shutdown(wait=False, cancel_futures=True)
        intel = {}
        for future in sorted(done, key=lambda f: futures[f]):
            headlines = future.result() if future.exception() is None else None
            if not headlines:
                continue
            place = futures[future]
            assessment = self.nlp.assess(headlines)
            score = assessment["score"]
            threat_type = assessment["type"] if score > 0 else "none"
            condition = "clear"
            if threat_type == "weather" and score >= RAIN_SCORE:
                condition = "stormy" if score >= STORM_SCORE else "rainy"
            report = {"place": place, "hubs": sorted(self._city_hubs.get(place, ())), "headlines": headlines,
                      "headline": assessment["headline"], "score": round(score, 3),
                      "threat_type": threat_type, "condition": condition}
            for hub_id in report["hubs"]:
                intel[hub_id] = report
        return intel

    def _intel_delays(self, intel):
        """Re-predicted delay quantiles for every lane touching a hub with live intel."""
        G = self.unified_graph
        edges = {}
        for h in intel:
            for n in self._hub_nodes.get(h, []):
                for u, v, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)):
                    if d["type"] == "transit":
                        edges[(u, v)] = d
        return self._predict_delays([(u, v, d) for (u, v), d in edges.items()], intel)

    # ---- Routing ------------------------------------------------------------

    def recommend(self, source: str, destination: str, transport_preference: str = "any",
                  routing_policy: str = "STRICT", cargo_type: str = "general",
                  priority: str = "normal", scenario: str = None,
                  overrides: dict = None, live_intel: bool = False) -> dict:

        overrides = overrides or {}
        avoid_hubs = set(overrides.get("avoid_chokepoints", []))
        cost_ceiling = overrides.get("cost_ceiling", 999999)
        max_delay = overrides.get("max_delay", 9999)

        # 1. Resolve Entry/Exit (Virtual Nodes)
        res_s = self.resolver.resolve_node_to_entry_point(source)
        res_d = self.resolver.resolve_node_to_entry_point(destination)

        if "error" in res_s: return {"error": res_s["error"]}
        if "error" in res_d: return {"error": res_d["error"]}

        G = self.unified_graph
        s_vnode, d_vnode = res_s["id"], res_d["id"]
        origin_id, dest_id = G.nodes[s_vnode]["physical_id"], G.nodes[d_vnode]["physical_id"]

        # 2. Scenario lookup (per request; never shared state)
        active_scenario = self.scenario_mgr.get_scenario(scenario)
        disruptions = self.scenario_mgr.get_disruptions(scenario)

        # 3. Live news at the origin and destination (optional; needs network)
        intel = {}
        if live_intel:
            intel = self._live_intel([G.nodes[n].get("parent_city") or G.nodes[n]["display_name"]
                                      for n in (s_vnode, d_vnode)])
        intel_delays = self._intel_delays(intel) if intel else {}

        def hub(n):
            return G.nodes[n]["physical_id"]

        def delays(u, v, d):
            return intel_delays.get((u, v)) or d.get("delay_q", NO_DELAY)

        def leg_news(u, v, d):
            """(threat, hub) of the live report that bears on this leg most, after CARF.

            A transfer is exposed to news about either of the two modes it joins.
            """
            modes = (d["transport_mode"],) if d["type"] == "transit" else (G.nodes[u]["mode"], G.nodes[v]["mode"])
            best = (0.0, None)
            for h in (hub(v), hub(u)):
                if h in intel:
                    threat = max(self.carf.apply_filter(intel[h]["score"], intel[h]["headlines"], m) for m in modes)
                    if threat > best[0]:
                        best = (threat, h)
            return best

        # 4. Routing policy and cargo rules
        strict_modes = None
        if transport_preference != "any" and routing_policy == "STRICT":
            strict_modes = {transport_preference, "transfer", "road"}
        soft_preference = transport_preference != "any" and routing_policy == "PREFERRED"
        time_weight, cost_weight = BALANCED_WEIGHTS.get(priority, BALANCED_WEIGHTS["normal"])

        def excluded(u, v, d):
            if avoid_hubs and (hub(u) in avoid_hubs or hub(v) in avoid_hubs):
                return True
            if d["type"] == "transit" and cargo_type in MODE_PROFILES.get(d["transport_mode"], {}).get("cargo_restrictions", []):
                return True  # e.g. no hazardous waste by air, no urgent perishables by sea
            return strict_modes is not None and d["transport_mode"] not in strict_modes

        def preference_factor(u, v, d):
            # PREFERRED policy: a soft bias instead of STRICT's hard exclusion. Road
            # legs out of the origin or into the destination are first/last-mile
            # access and stay unpenalised, as road access is allowed under STRICT.
            if not soft_preference or d["type"] != "transit" or d["transport_mode"] == transport_preference:
                return 1.0
            if d["transport_mode"] == "road" and (hub(u) == origin_id or hub(v) == dest_id):
                return 1.0
            return PREFERENCE_PENALTY

        # 5. Persona Optimization
        candidates = []
        for persona in ["FASTEST", "SAFEST", "BALANCED"]:
            def weight_func(u, v, d, persona=persona, qi=PERSONA_QUANTILE[persona]):
                if excluded(u, v, d):
                    return None  # hides the edge from Dijkstra without copying the graph
                impact = self._leg_impact(d, hub(u), hub(v), origin_id, disruptions,
                                          extra_threat=leg_news(u, v, d)[0])
                time_h = d["baseline_time"] + delays(u, v, d)[qi] + impact["delay"]
                cost = d.get("cost", 0) + impact["premium"]
                threat = impact["threat"]
                if persona == "FASTEST":
                    weight = time_h
                elif persona == "SAFEST":
                    weight = time_h * (1.0 + threat * 12.0)
                else:  # BALANCED (economic leaning)
                    weight = time_h * time_weight + (cost / 150.0) * cost_weight + (threat * 40.0) * 0.2
                return weight * preference_factor(u, v, d)

            try:
                path = nx.dijkstra_path(G, s_vnode, d_vnode, weight=weight_func)
            except nx.NetworkXNoPath:
                continue
            route = self._compose_route(persona, path, origin_id, disruptions, delays, leg_news, intel)
            if route["total_cost"] > cost_ceiling or route["adjusted_eta"] > max_delay * 24:
                continue
            route["override_applied"] = bool(avoid_hubs or cost_ceiling < 999999)
            candidates.append(route)

        if not candidates:
            return {"error": "No valid multimodal route under the current constraints."}

        # Deduplicate: personas that chose the same path become one route that
        # lists every persona it is best for.
        final = []
        by_path = {}
        for c in sorted(candidates, key=lambda x: x["adjusted_eta"]):
            path_sig = tuple((l["to"], l["mode"], l["type"]) for l in c["legs"])
            if path_sig in by_path:
                by_path[path_sig]["personas"].append(c["persona"])
                continue
            c["personas"] = [c["persona"]]
            by_path[path_sig] = c
            final.append(c)

        preference = transport_preference if soft_preference else None
        for c in final:
            c["delay_drivers"] = self._delay_drivers(c.pop("_leg_features"))
            c["explanation"] = self._explain(c, final, preference)

        return {
            "origin": source, "destination": destination,
            "active_scenario": active_scenario["name"] if active_scenario else None,
            "live_intel": list({id(r): r for r in intel.values()}.values()),
            "delay_model": self.delay_model is not None,
            "recommendations": final[:3]
        }

    @staticmethod
    def _leg_impact(d, from_id, to_id, origin_id, disruptions, charged=None, extra_threat=0.0):
        """Scenario impact of one leg, shared by the optimiser and the reported trace.

        A leg touching a disrupted hub is exposed to its threat. The hub's delay, plus
        a 10% risk premium on the leg's cost, is charged on the transit leg arriving
        at the hub, or on the transit leg leaving a disrupted origin. Transfers inside
        a hub belong to the same visit and are never charged; `charged` stops a hub
        being charged twice on one route. `extra_threat` adds live-news threat.
        """
        exposed = [h for h in dict.fromkeys((from_id, to_id)) if h in disruptions]
        threat = max([d.get("base_threat", 0.05), extra_threat] + [disruptions[h]["threat"] for h in exposed])
        due = []
        if d["type"] == "transit":
            if to_id in disruptions and to_id != from_id:
                due.append(to_id)
            if from_id == origin_id and from_id in disruptions:
                due.append(from_id)
        due = [h for h in due if charged is None or h not in charged]
        return {
            "threat": threat,
            "exposed": exposed,
            "charged": due,
            "delay": sum(disruptions[h]["delay"] for h in due),
            "premium": d.get("cost", 0) * 0.1 * len(due),
        }

    def _compose_route(self, persona, path, origin_id, disruptions, delays, leg_news, intel):
        G = self.unified_graph
        legs, leg_features, leg_quantiles = [], [], []
        total_cost = max_threat = 0.0
        fixed_hours = 0.0  # nominal transit and transfer time plus scenario delay
        trace = {
            "eta": {"transit": 0.0, "transfer": 0.0, "delay": 0.0, "scenario": 0.0},
            "cost": {"transit": 0.0, "transfer": 0.0, "scenario": 0.0},
            "risk": {"baseline": 0.0, "scenario": 0.0, "live": 0.0}
        }
        charged = set()  # disrupted hubs whose delay is already on this route

        for u, v in zip(path, path[1:]):
            d = G[u][v]
            from_id, to_id = G.nodes[u]["physical_id"], G.nodes[v]["physical_id"]
            news_threat, news_hub = leg_news(u, v, d)
            impact = self._leg_impact(d, from_id, to_id, origin_id, disruptions, charged, extra_threat=news_threat)
            charged.update(impact["charged"])
            base_time, base_cost = d["baseline_time"], d.get("cost", 0)
            q = delays(u, v, d) if d["type"] == "transit" else NO_DELAY

            # The trace splits each total into parts that add up exactly: base
            # transit/transfer time and cost, the model's typical (p50) operational
            # delay, and scenario delay/premium.
            bucket = "transfer" if d["type"] == "transfer" else "transit"
            trace["eta"][bucket] += base_time
            trace["eta"]["delay"] += q[0]
            trace["eta"]["scenario"] += impact["delay"]
            trace["cost"][bucket] += base_cost
            trace["cost"]["scenario"] += impact["premium"]
            if bucket == "transit":
                trace["risk"]["baseline"] = max(trace["risk"]["baseline"], d.get("base_threat", 0.05))
                leg_quantiles.append(q)
                if self.delay_model is not None:
                    leg_features.append(self._leg_features(u, v, d, intel))
            if impact["exposed"]:
                trace["risk"]["scenario"] = max(trace["risk"]["scenario"], impact["threat"])
            trace["risk"]["live"] = max(trace["risk"]["live"], news_threat)

            fixed_hours += base_time + impact["delay"]
            l_cost = base_cost + impact["premium"]
            total_cost += l_cost
            max_threat = max(max_threat, impact["threat"])

            scenario_hub = impact["exposed"][-1] if impact["exposed"] else None
            if scenario_hub:
                reason, source = disruptions[scenario_hub]["reason"], "SCENARIO"
            elif news_hub:
                reason, source = intel[news_hub]["headline"], "LIVE"
            else:
                reason, source = d.get("base_news", "Standard conditions"), "FALLBACK"
            legs.append({
                "from": from_id,
                "to": to_id,
                "to_name": G.nodes[v].get("display_name", to_id),
                "mode": d["transport_mode"].upper(),
                "type": d["type"],
                "eta": round(base_time + q[0] + impact["delay"], 1),
                "delay": dict(zip(QUANTILE_NAMES, (round(x, 1) for x in q))),
                "cost": round(l_cost, 2),
                "threat": round(impact["threat"], 2),
                "reason": reason,
                "intel_source": source
            })

        for part in trace:
            trace[part] = {k: round(x, 2) for k, x in trace[part].items()}
        typical = fixed_hours + sum(q[0] for q in leg_quantiles)
        return {
            "persona": persona,
            "primary_mode": "MULTIMODAL",
            "legs": legs,
            # Typical ETA adds each leg's median delay; the band is the simulated
            # distribution of the whole route's door-to-door time.
            "adjusted_eta": round(typical, 1),
            "eta_band": self._eta_band(fixed_hours, leg_quantiles),
            "total_cost": round(total_cost, 2),
            "threat_level": round(max_threat, 2),
            "audit_trace": trace,
            "_leg_features": leg_features,
        }

    @staticmethod
    def _explain(route, routes, preference=None):
        """Plain-language trade-off of one route against the other options returned."""
        transfers = sum(1 for l in route["legs"] if l["type"] == "transfer")
        eta, cost, threat = route["adjusted_eta"], route["total_cost"], route["threat_level"]
        others = [r for r in routes if r is not route]
        # Under the PREFERRED policy a persona optimises with a bias, so its route is
        # the best *among options favouring that mode*, not necessarily overall.
        favouring = f" that favours {preference.upper()}" if preference else ""
        parts = []

        if "FASTEST" in route["personas"]:
            label = "Fastest option" + ("" if all(eta <= r["adjusted_eta"] for r in others) else favouring)
            slower = [r for r in others if r["adjusted_eta"] - eta >= SAME_TIME_H]
            if slower:
                ref = min(slower, key=lambda r: r["total_cost"])
                ratio = cost / ref["total_cost"]
                price = ("at about the same cost" if abs(ratio - 1) < SAME_COST_RATIO
                         else f"at {ratio:.1f}x its cost")
                parts.append(f"{label}: {eta:.0f}h door to door, {ref['adjusted_eta'] - eta:.0f}h sooner "
                             f"than the {ref['persona'].lower()} route {price}.")
            else:
                parts.append(f"{label}: {eta:.0f}h door to door.")
        if "SAFEST" in route["personas"]:
            label = "Lowest-risk option" + ("" if all(threat <= r["threat_level"] for r in others) else favouring)
            riskier = [r for r in others if r["threat_level"] > threat]
            if riskier:
                ref = max(riskier, key=lambda r: r["threat_level"])
                parts.append(f"{label}: peak threat {threat:.0%} vs {ref['threat_level']:.0%} "
                             f"on the {ref['persona'].lower()} route.")
            else:
                parts.append(f"{label}: peak threat {threat:.0%}.")
        if "BALANCED" in route["personas"]:
            pricier = [r for r in others if (r["total_cost"] - cost) / r["total_cost"] >= SAME_COST_RATIO]
            if pricier:
                ref = max(pricier, key=lambda r: r["total_cost"])
                saving = (ref["total_cost"] - cost) / ref["total_cost"]
                saving_text = "over 99%" if saving >= 0.995 else f"{saving:.0%}"
                gap = eta - ref["adjusted_eta"]
                timing = "same ETA" if abs(gap) < SAME_TIME_H else f"{gap:+.0f}h on its ETA"
                parts.append(f"Best cost-time-risk balance: {saving_text} cheaper than the {ref['persona'].lower()} "
                             f"route, {timing}.")
            else:
                parts.append(f"Best cost-time-risk balance at ${cost:,.0f} landed.")

        parts.append(f"{transfers} mode transfer{'s' if transfers != 1 else ''}.")
        band = route["eta_band"]
        if band["p95"] > band["p50"]:
            parts.append(f"Plan for {band['p85']:.0f}h (85% confidence), up to {band['p95']:.0f}h in a bad case.")
        drivers = (route.get("delay_drivers") or {}).get("drivers") or []
        top = next((d for d in drivers if d["hours"] >= 1), None)
        if top:
            parts.append(f"Largest delay driver: {top['factor']} (+{top['hours']:.0f}h at p85).")
        if route["audit_trace"]["eta"]["scenario"] > 0:
            parts.append(f"Includes {route['audit_trace']['eta']['scenario']:.0f}h of scenario delay.")
        return " ".join(parts)
