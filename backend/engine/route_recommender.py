import networkx as nx
from .multimodal_network import create_multimodal_network
from .threat_intelligence import ContrastiveNLPEngine, CARFFilter
from .news_ingestion import DynamicNewsIngestor
from .node_resolver import NodeResolver

PREFERENCE_PENALTY = 1.5  # weight multiplier on non-preferred modes under PREFERRED policy

class RouteRecommender:
    """
    Supplychainer Unified Multimodal Optimization Engine.
    V8: Virtual-Node Forensic Edition.
    """

    def __init__(self, network, predictor, simulator, scenario_mgr, demo_mode=False):
        self.network = network # Legacy
        self.predictor = predictor
        self.simulator = simulator
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
        
        if self.demo_mode:
            self.is_warmed_up = True
            
        print(f"[STARTUP] Unified Engine Ready.")

    def run_background_warmup(self):
        if self.is_warmed_up: return
        print("[WARMUP] Calibrating global threat floor...")
        try:
            self.predictor.warmup()
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

    def recommend(self, source: str, destination: str, transport_preference: str = "any",
                  routing_policy: str = "STRICT", cargo_type: str = "general",
                  priority: str = "normal", scenario: str = None,
                  overrides: dict = None) -> dict:

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

        # 3. Routing policy
        strict_modes = None
        if transport_preference != "any" and routing_policy == "STRICT":
            strict_modes = {transport_preference, "transfer", "road"}
        soft_preference = transport_preference != "any" and routing_policy == "PREFERRED"

        def hub(n):
            return G.nodes[n]["physical_id"]

        def excluded(u, v, d):
            if avoid_hubs and (hub(u) in avoid_hubs or hub(v) in avoid_hubs):
                return True
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

        # 4. Persona Optimization
        candidates = []
        for persona in ["FASTEST", "SAFEST", "BALANCED"]:
            def weight_func(u, v, d, persona=persona):
                if excluded(u, v, d):
                    return None  # hides the edge from Dijkstra without copying the graph
                impact = self._leg_impact(d, hub(u), hub(v), origin_id, disruptions)
                time_h = d["baseline_time"] + impact["delay"]
                cost = d.get("cost", 0) + impact["premium"]
                threat = impact["threat"]
                if persona == "FASTEST":
                    weight = time_h
                elif persona == "SAFEST":
                    weight = time_h * (1.0 + threat * 12.0)
                else:  # BALANCED (economic leaning)
                    weight = time_h * 0.3 + (cost / 150.0) * 0.5 + (threat * 40.0) * 0.2
                return weight * preference_factor(u, v, d)

            try:
                path = nx.dijkstra_path(G, s_vnode, d_vnode, weight=weight_func)
            except nx.NetworkXNoPath:
                continue
            route = self._compose_route(persona, path, origin_id, disruptions)
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
            c["explanation"] = self._explain(c, final, preference)

        return {
            "origin": source, "destination": destination,
            "active_scenario": active_scenario["name"] if active_scenario else None,
            "recommendations": final[:3]
        }

    @staticmethod
    def _leg_impact(d, from_id, to_id, origin_id, disruptions, charged=None):
        """Scenario impact of one leg, shared by the optimiser and the reported trace.

        A leg touching a disrupted hub is exposed to its threat. The hub's delay, plus
        a 10% risk premium on the leg's cost, is charged on the transit leg arriving
        at the hub, or on the transit leg leaving a disrupted origin. Transfers inside
        a hub belong to the same visit and are never charged; `charged` stops a hub
        being charged twice on one route.
        """
        exposed = [h for h in dict.fromkeys((from_id, to_id)) if h in disruptions]
        threat = max([d.get("base_threat", 0.05)] + [disruptions[h]["threat"] for h in exposed])
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

    def _compose_route(self, persona, path, origin_id, disruptions):
        G = self.unified_graph
        legs = []
        total_time = total_cost = max_threat = 0.0
        trace = {
            "eta": {"transit": 0.0, "transfer": 0.0, "scenario": 0.0},
            "cost": {"transit": 0.0, "transfer": 0.0, "scenario": 0.0},
            "risk": {"baseline": 0.0, "scenario": 0.0}
        }
        charged = set()  # disrupted hubs whose delay is already on this route

        for u, v in zip(path, path[1:]):
            d = G[u][v]
            from_id, to_id = G.nodes[u]["physical_id"], G.nodes[v]["physical_id"]
            impact = self._leg_impact(d, from_id, to_id, origin_id, disruptions, charged)
            charged.update(impact["charged"])
            base_time, base_cost = d["baseline_time"], d.get("cost", 0)

            # The trace splits each total into parts that add up exactly:
            # base transit/transfer time and cost, plus scenario delay and premium.
            bucket = "transfer" if d["type"] == "transfer" else "transit"
            trace["eta"][bucket] += base_time
            trace["cost"][bucket] += base_cost
            trace["eta"]["scenario"] += impact["delay"]
            trace["cost"]["scenario"] += impact["premium"]
            if bucket == "transit":
                trace["risk"]["baseline"] = max(trace["risk"]["baseline"], d.get("base_threat", 0.05))
            if impact["exposed"]:
                trace["risk"]["scenario"] = max(trace["risk"]["scenario"], impact["threat"])

            l_time = base_time + impact["delay"]
            l_cost = base_cost + impact["premium"]
            total_time += l_time
            total_cost += l_cost
            max_threat = max(max_threat, impact["threat"])

            scenario_hub = impact["exposed"][-1] if impact["exposed"] else None
            legs.append({
                "from": from_id,
                "to": to_id,
                "to_name": G.nodes[v].get("display_name", to_id),
                "mode": d["transport_mode"].upper(),
                "type": d["type"],
                "eta": round(l_time, 1),
                "cost": round(l_cost, 2),
                "threat": round(impact["threat"], 2),
                "reason": disruptions[scenario_hub]["reason"] if scenario_hub else d.get("base_news", "Standard conditions"),
                "intel_source": "SCENARIO" if scenario_hub else "FALLBACK"
            })

        for part in trace:
            trace[part] = {k: round(x, 2) for k, x in trace[part].items()}
        return {
            "persona": persona,
            "primary_mode": "MULTIMODAL",
            "legs": legs,
            "adjusted_eta": round(total_time, 1),
            "total_cost": round(total_cost, 2),
            "threat_level": round(max_threat, 2),
            "audit_trace": trace
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
            slower = [r for r in others if r["adjusted_eta"] > eta]
            if slower:
                ref = min(slower, key=lambda r: r["total_cost"])
                parts.append(f"{label}: {eta:.0f}h door to door, {ref['adjusted_eta'] - eta:.0f}h sooner "
                             f"than the {ref['persona'].lower()} route at {cost / ref['total_cost']:.1f}x its cost.")
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
            pricier = [r for r in others if r["total_cost"] > cost]
            if pricier:
                ref = max(pricier, key=lambda r: r["total_cost"])
                saving = (ref["total_cost"] - cost) / ref["total_cost"]
                saving_text = "over 99%" if saving >= 0.995 else f"{saving:.0%}"
                parts.append(f"Best cost-time-risk balance: {saving_text} cheaper than the {ref['persona'].lower()} "
                             f"route, {eta - ref['adjusted_eta']:+.0f}h on its ETA.")
            else:
                parts.append(f"Best cost-time-risk balance at ${cost:,.0f} landed.")

        parts.append(f"{transfers} mode transfer{'s' if transfers != 1 else ''}.")
        if route["audit_trace"]["eta"]["scenario"] > 0:
            parts.append(f"Includes {route['audit_trace']['eta']['scenario']:.0f}h of scenario delay.")
        return " ".join(parts)
