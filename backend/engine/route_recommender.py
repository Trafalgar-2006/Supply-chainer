import networkx as nx
import math
import time
from typing import List, Dict, Any, Optional
from .multimodal_network import MODE_PROFILES, create_multimodal_network
from .threat_intelligence import ThreatIntelligencePredictor, ContrastiveNLPEngine, CARFFilter
from .news_ingestion import DynamicNewsIngestor
from .node_resolver import NodeResolver

PREFERENCE_PENALTY = 1.5  # weight multiplier on non-preferred modes under PREFERRED policy
LOCAL_ROAD_KM = 300       # road hops up to this length count as first/last-mile access

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
        
        t0 = time.perf_counter()
        overrides = overrides or {}
        avoid_hubs = overrides.get("avoid_chokepoints", [])
        cost_ceiling = overrides.get("cost_ceiling", 999999)
        max_delay = overrides.get("max_delay", 9999)
        
        # 1. Resolve Entry/Exit (Virtual Nodes)
        res_s = self.resolver.resolve_node_to_entry_point(source)
        res_d = self.resolver.resolve_node_to_entry_point(destination)
        
        if "error" in res_s: return {"error": res_s["error"]}
        if "error" in res_d: return {"error": res_d["error"]}
        
        s_vnode, d_vnode = res_s["id"], res_d["id"]
        
        # 2. Scenario lookup (per request; never shared state)
        active_scenario = self.scenario_mgr.get_scenario(scenario)
        disruptions = self.scenario_mgr.get_disruptions(scenario)
        
        # 3. Persona Optimization
        soft_preference = transport_preference != "any" and routing_policy == "PREFERRED"
        candidates = []
        for persona in ["FASTEST", "SAFEST", "BALANCED"]:
            try:
                # Build Persona Graph (Applying STRICT constraints)
                G_p = self.unified_graph.copy()
                
                # Apply Hub Avoidance (Prune all virtual nodes for the hub)
                for hub_id in avoid_hubs:
                    nodes_to_remove = [n for n, d in G_p.nodes(data=True) if d.get("physical_id") == hub_id]
                    G_p.remove_nodes_from(nodes_to_remove)
                
                # Apply Transport Preference
                if transport_preference != "any" and routing_policy == "STRICT":
                    allowed_modes = [transport_preference, "transfer", "road"]
                    edges_to_remove = []
                    for u, v, d in G_p.edges(data=True):
                        if d["transport_mode"] not in allowed_modes:
                            edges_to_remove.append((u, v))
                    G_p.remove_edges_from(edges_to_remove)

                def weight_func(u, v, d):
                    return persona_weight(u, v, d) * preference_factor(d)

                def preference_factor(d):
                    # PREFERRED policy: a soft bias instead of STRICT's hard exclusion.
                    # Local road hops (first/last mile) are exempt, as in STRICT.
                    if not soft_preference or d["type"] != "transit" or d["transport_mode"] == transport_preference:
                        return 1.0
                    if d["transport_mode"] == "road" and d.get("distance", 0) <= LOCAL_ROAD_KM:
                        return 1.0
                    return PREFERENCE_PENALTY

                def persona_weight(u, v, d):
                    mode = d["transport_mode"]
                    base_t = d["baseline_time"]
                    base_c = d.get("cost", 0)
                    
                    # Intelligence Factor (Mapped to physical node). A disruption is
                    # charged when a transit leg arrives at the hub; transfers inside
                    # the hub are part of the same visit and are not charged again.
                    p_id = G_p.nodes[v].get("physical_id")

                    threat = d.get("base_threat", 0.05)
                    delay = 0

                    if p_id in disruptions and d["type"] == "transit" and G_p.nodes[u].get("physical_id") != p_id:
                        threat = max(threat, disruptions[p_id]["threat"])
                        delay += disruptions[p_id]["delay"]
                    
                    if persona == "FASTEST":
                        return base_t + delay
                    elif persona == "SAFEST":
                        risk_penalty = 1.0 + (threat * 12.0)
                        return (base_t + delay) * risk_penalty
                    else: # BALANCED (ECONOMIC leaning)
                        # High cost penalty for transfers and expensive modes
                        time_weight = 0.3
                        cost_weight = 0.5
                        risk_weight = 0.2
                        return (base_t + delay)*time_weight + (base_c / 150.0)*cost_weight + (threat * 40.0)*risk_weight

                path = nx.dijkstra_path(G_p, s_vnode, d_vnode, weight=weight_func)
                
                # Compose Multimodal Path Details
                legs = []
                total_time, total_cost, max_threat = 0, 0, 0
                trace = {
                    "eta": {"transit": 0, "transfer": 0, "scenario": 0},
                    "cost": {"transit": 0, "transfer": 0, "scenario": 0},
                    "risk": {"baseline": 0, "scenario": 0}
                }

                charged = set()  # disrupted hubs whose delay is already on this route
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    d = G_p[u][v]
                    mode = d["transport_mode"]
                    v_data = G_p.nodes[v]
                    p_id = v_data.get("physical_id")
                    from_id = G_p.nodes[u].get("physical_id", u)

                    base_time = d["baseline_time"]
                    base_cost = d.get("cost", 0)
                    base_threat = d.get("base_threat", 0.05)
                    l_threat = base_threat
                    l_news = d.get("base_news", "Standard conditions")
                    l_source = "FALLBACK"
                    scenario_delay = scenario_premium = 0.0

                    # Every leg that touches a disrupted hub is exposed to it; its delay
                    # is charged once per route, on the first transit leg that departs
                    # from or arrives at the hub (this covers a disrupted origin too).
                    for hub in dict.fromkeys((from_id, p_id)):
                        if hub not in disruptions:
                            continue
                        l_threat = max(l_threat, disruptions[hub]["threat"])
                        l_news = disruptions[hub]["reason"]
                        l_source = "SCENARIO"
                        trace["risk"]["scenario"] = max(trace["risk"]["scenario"], l_threat)
                        if d["type"] == "transit" and hub not in charged:
                            charged.add(hub)
                            scenario_delay += disruptions[hub]["delay"]
                            scenario_premium += base_cost * 0.1

                    # The trace splits each total into parts that add up exactly:
                    # base transit/transfer time and cost, plus scenario delay/premium.
                    bucket = "transfer" if d["type"] == "transfer" else "transit"
                    trace["eta"][bucket] += base_time
                    trace["cost"][bucket] += base_cost
                    trace["eta"]["scenario"] += scenario_delay
                    trace["cost"]["scenario"] += scenario_premium
                    if bucket == "transit":
                        trace["risk"]["baseline"] = max(trace["risk"]["baseline"], base_threat)

                    l_time = base_time + scenario_delay
                    l_cost = base_cost + scenario_premium
                    total_time += l_time
                    total_cost += l_cost
                    max_threat = max(max_threat, l_threat)

                    legs.append({
                        "from": from_id,
                        "to": p_id,
                        "to_name": v_data.get("display_name", p_id),
                        "mode": mode.upper(),
                        "type": d["type"],
                        "eta": round(l_time, 1),
                        "cost": round(l_cost, 2),
                        "threat": round(l_threat, 2),
                        "reason": l_news,
                        "intel_source": l_source
                    })

                if total_cost > cost_ceiling or total_time > (max_delay * 24): continue

                for part in ("eta", "cost", "risk"):
                    trace[part] = {k: round(v, 2) for k, v in trace[part].items()}
                candidates.append({
                    "persona": persona,
                    "primary_mode": "MULTIMODAL",
                    "legs": legs,
                    "adjusted_eta": round(total_time, 1),
                    "total_cost": round(total_cost, 2),
                    "threat_level": round(max_threat, 2),
                    "audit_trace": trace,
                    "override_applied": bool(avoid_hubs or cost_ceiling < 999999)
                })

            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                print(f"[ROUTING ERROR] {persona}: {e}")

        if not candidates:
            return {"error": "No valid multimodal route under the current constraints."}

        # Deduplicate: personas that chose the same path become one route that
        # lists every persona it is best for.
        final = []
        by_path = {}
        for c in sorted(candidates, key=lambda x: x["adjusted_eta"]):
            path_sig = tuple(l["to"] for l in c["legs"])
            if path_sig in by_path:
                by_path[path_sig]["personas"].append(c["persona"])
                continue
            c["personas"] = [c["persona"]]
            by_path[path_sig] = c
            final.append(c)

        for c in final:
            c["explanation"] = self._explain(c, final)

        return {
            "origin": source, "destination": destination,
            "active_scenario": active_scenario["name"] if active_scenario else None,
            "recommendations": final[:3]
        }

    @staticmethod
    def _explain(route, routes):
        """Plain-language trade-off of one route against the other options returned."""
        transfers = sum(1 for l in route["legs"] if l["type"] == "transfer")
        eta, cost, threat = route["adjusted_eta"], route["total_cost"], route["threat_level"]
        others = [r for r in routes if r is not route]
        parts = []

        if "FASTEST" in route["personas"]:
            slower = [r for r in others if r["adjusted_eta"] > eta]
            if slower:
                ref = min(slower, key=lambda r: r["total_cost"])
                parts.append(f"Fastest option: {eta:.0f}h door to door, {ref['adjusted_eta'] - eta:.0f}h sooner "
                             f"than the {ref['persona'].lower()} route at {cost / ref['total_cost']:.1f}x its cost.")
            else:
                parts.append(f"Fastest option: {eta:.0f}h door to door.")
        if "SAFEST" in route["personas"]:
            riskier = [r for r in others if r["threat_level"] > threat]
            if riskier:
                ref = max(riskier, key=lambda r: r["threat_level"])
                parts.append(f"Lowest-risk option: peak threat {threat:.0%} vs {ref['threat_level']:.0%} "
                             f"on the {ref['persona'].lower()} route.")
            else:
                parts.append(f"Lowest-risk option: peak threat {threat:.0%}.")
        if "BALANCED" in route["personas"]:
            pricier = [r for r in others if r["total_cost"] > cost]
            if pricier:
                ref = max(pricier, key=lambda r: r["total_cost"])
                saving = (ref["total_cost"] - cost) / ref["total_cost"]
                parts.append(f"Best cost-time-risk balance: {saving:.0%} cheaper than the {ref['persona'].lower()} "
                             f"route, {eta - ref['adjusted_eta']:+.0f}h on its ETA.")
            else:
                parts.append(f"Best cost-time-risk balance at ${cost:,.0f} landed.")

        parts.append(f"{transfers} mode transfer{'s' if transfers != 1 else ''}.")
        if route["audit_trace"]["eta"]["scenario"] > 0:
            parts.append(f"Includes {route['audit_trace']['eta']['scenario']:.0f}h of scenario delay.")
        return " ".join(parts)
