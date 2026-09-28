import networkx as nx
import json
import os
import math
from typing import List, Dict, Any

MODE_PROFILES = {
    "road": {"speed": 80, "cost_per_km": 1.2, "cargo_restrictions": ["oversize_heavy"]},
    "rail": {"speed": 60, "cost_per_km": 0.5, "cargo_restrictions": []},
    "air": {"speed": 800, "cost_per_km": 15.0, "cargo_restrictions": ["hazardous_waste"]},
    "sea": {"speed": 35, "cost_per_km": 0.15, "cargo_restrictions": ["perishable_urgent"]}
}

PRIORITY_MULTIPLIERS = {
    "low": 1.2,
    "normal": 1.0,
    "urgent": 0.7
}

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _travel_time(dist, mode):
    # Speeds in km/h based on freight benchmarks
    speed = MODE_PROFILES.get(mode, {}).get("speed", 50)
    
    # Road friction factor: Real trucking time includes standard stops/traffic
    if mode == "road":
        # Effective speed is lower than max highway speed due to freight regulations
        effective_speed = speed * 0.85 # 68 km/h effective
        return dist / effective_speed
    
    return dist / speed

def load_canonical_hubs():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'canonical_hubs.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

TRANSFER_PROFILES = {
    "port_to_rail": {"delay": 8.0, "cost": 180, "risk": 0.05},
    "road_to_air": {"delay": 6.0, "cost": 150, "risk": 0.02},
    "road_to_sea": {"delay": 14.0, "cost": 250, "risk": 0.08},
    "road_to_rail": {"delay": 4.0, "cost": 100, "risk": 0.03},
    "rail_to_road": {"delay": 4.0, "cost": 80, "risk": 0.02},
    "default": {"delay": 4.0, "cost": 100, "risk": 0.03}
}

# Enclosed seas, and the Atlantic / Indo-Pacific split, can only be crossed through
# real straits. The hub data lists direct lanes such as Jebel Ali -> Haifa that
# would sail across the Arabian Peninsula; every cross-basin lane is routed through
# its chokepoints instead, so a Hormuz, Bab el-Mandeb, Suez or Gibraltar disruption
# reaches every ship that actually passes it.
SEA_GATES = {
    "CHOKE-HORMUZ": ("PERSIAN_GULF", "INDO_PACIFIC"),
    "CHOKE-BABEL": ("RED_SEA", "INDO_PACIFIC"),
    "CHOKE-SUEZ": ("RED_SEA", "MEDITERRANEAN"),
    "CHOKE-GIBRAL": ("MEDITERRANEAN", "ATLANTIC"),
    "CHOKE-DARDANELLES": ("MEDITERRANEAN", "MARMARA"),
    "CHOKE-BOSPHO": ("MARMARA", "BLACK_SEA"),
    "CHOKE-CAPEGOOD": ("INDO_PACIFIC", "ATLANTIC"),
    "CHOKE-PANAMA": ("INDO_PACIFIC", "ATLANTIC"),
}

def sea_basin(lat, lon):
    """Coarse ocean basin of a coastal point; the most enclosed seas are checked first."""
    if 38.0 <= lat <= 47.5 and 46.5 <= lon <= 55.0: return "CASPIAN"  # landlocked
    if 23.5 <= lat <= 30.5 and 47.5 <= lon < 56.3: return "PERSIAN_GULF"
    if 12.7 <= lat < 30.0 and 32.2 <= lon <= 43.6: return "RED_SEA"
    if 40.3 <= lat <= 41.08 and 26.5 <= lon <= 30.0: return "MARMARA"
    if 40.9 < lat <= 47.5 and 27.4 <= lon <= 42.0: return "BLACK_SEA"
    if 30.0 <= lat <= 46.0 and -5.7 <= lon <= 36.5 and not (lon < 3.0 and lat > 42.5): return "MEDITERRANEAN"
    if lon < -30:  # the Americas: Pacific coast vs Atlantic / Caribbean / Gulf of Mexico
        pacific = lon < -100 or (lat < 9.2 and lon < -77) or (lat < 7 and lon < -70)
        return "INDO_PACIFIC" if pacific else "ATLANTIC"
    if lat > 50 and lon < 60: return "ATLANTIC"  # North Sea, Baltic, Barents
    return "ATLANTIC" if lon < 20 else "INDO_PACIFIC"

# Road and rail cannot cross open sea. Islands are reachable over land only through
# a fixed link, and the Channel Tunnel is the only one between an island in the
# registry and a continent.
ISLAND_LANDMASSES = {"UK", "Ireland", "Japan", "Taiwan", "Sri Lanka", "Indonesia", "Philippines",
                     "Australia", "New Zealand", "Madagascar", "Iceland"}
CHANNEL_TUNNEL_PORTALS = ((51.096, 1.137), (50.925, 1.813))  # Folkestone (UK), Coquelles (FR)
TUNNEL_REACH_KM = 150

def landmass(hub):
    if hub["country"] in ISLAND_LANDMASSES:
        return hub["country"]
    return "AMERICAS" if hub["lon"] < -30 else "AFRO_EURASIA"

def land_link_possible(h1, h2):
    a, b = landmass(h1), landmass(h2)
    if a == b:
        return True
    if {a, b} == {"UK", "AFRO_EURASIA"}:
        uk, mainland = (h1, h2) if a == "UK" else (h2, h1)
        return (_haversine(uk["lat"], uk["lon"], *CHANNEL_TUNNEL_PORTALS[0]) <= TUNNEL_REACH_KM
                and _haversine(mainland["lat"], mainland["lon"], *CHANNEL_TUNNEL_PORTALS[1]) <= TUNNEL_REACH_KM)
    return False

def _hub_basins(hub):
    return set(SEA_GATES[hub["id"]]) if hub["id"] in SEA_GATES else {sea_basin(hub["lat"], hub["lon"])}

def _strait_chain(h1, h2, hub_lookup):
    """Shortest sequence of straits a ship must pass between two sea hubs.

    Returns [] for a lane inside one basin, the gate ids for a cross-basin lane,
    or None when no sea passage exists (e.g. the landlocked Caspian).
    """
    goal = _hub_basins(h2)
    best = None

    def walk(basin, pos, chain, dist, used):
        nonlocal best
        if basin in goal:
            total = dist + _haversine(*pos, h2["lat"], h2["lon"])
            if best is None or total < best[0]:
                best = (total, chain)
        for gate, sides in SEA_GATES.items():
            if gate in used or basin not in sides:
                continue
            g = hub_lookup[gate]
            nxt = sides[1] if basin == sides[0] else sides[0]
            walk(nxt, (g["lat"], g["lon"]), chain + [gate],
                 dist + _haversine(*pos, g["lat"], g["lon"]), used | {gate})

    for basin in _hub_basins(h1):
        walk(basin, (h1["lat"], h1["lon"]), [], 0.0, frozenset({h1["id"], h2["id"]}))
    return None if best is None else best[1]

def create_multimodal_network():
    """
    Supplychainer Unified Multimodal Optimization Graph.
    V3: Node Splitting Edition (The Forensic Fix).
    """
    G = nx.DiGraph()
    hubs = load_canonical_hubs()
    hub_lookup = {h["id"]: h for h in hubs}

    # 1. Add Mode-Specific Virtual Nodes
    # Each hub H with modes M gets nodes H:m1, H:m2...
    for hub in hubs:
        hub_id = hub["id"]
        for mode in hub["modes"]:
            v_node = f"{hub_id}:{mode}"
            G.add_node(v_node,
                physical_id=hub_id,
                display_name=hub["display_name"], 
                type=hub["type"],
                country=hub["country"], 
                lat=hub["lat"], 
                lon=hub["lon"],
                importance=hub.get("importance", 5),
                mode=mode,
                parent_city=hub.get("parent_city"),
                basins=sorted(_hub_basins(hub)) if mode == "sea" else None)

    # 2. Add Intra-Hub Transfer Edges (The Friction Layer)
    for hub in hubs:
        hub_id = hub["id"]
        modes = hub["modes"]
        for i, m1 in enumerate(modes):
            for m2 in modes[i+1:]:
                u, v = f"{hub_id}:{m1}", f"{hub_id}:{m2}"
                
                # Determine transfer profile
                t_type = hub["type"]
                profile_key = "default"
                if t_type == "port" and (m1 == "rail" or m2 == "rail"): profile_key = "port_to_rail"
                elif t_type == "airport" and (m1 == "road" or m2 == "road"): profile_key = "road_to_air"
                elif t_type == "port" and (m1 == "road" or m2 == "road"): profile_key = "road_to_sea"
                
                profile = TRANSFER_PROFILES.get(profile_key, TRANSFER_PROFILES["default"])
                
                G.add_edge(u, v, baseline_time=profile["delay"], distance=0.1, 
                           transport_mode="transfer", type="transfer", 
                           cost=profile["cost"], risk=profile["risk"])
                G.add_edge(v, u, baseline_time=profile["delay"], distance=0.1, 
                           transport_mode="transfer", type="transfer", 
                           cost=profile["cost"], risk=profile["risk"])

    # 3. Add Strategic Intra-Mode Transit Edges
    # `connections` is an undirected adjacency list: most lanes are listed on only
    # one of their two hubs, so every lane is added in both directions.
    for hub in hubs:
        u_base = hub["id"]
        for conn in hub.get("connections", []):
            v_base = conn["to"]
            mode = conn["mode"]

            u_vnode = f"{u_base}:{mode}"
            v_vnode = f"{v_base}:{mode}"

            if G.has_node(u_vnode) and G.has_node(v_vnode):
                waypoints = [hub, hub_lookup[v_base]]
                if mode in ("road", "rail") and not land_link_possible(hub, hub_lookup[v_base]):
                    continue
                if mode == "sea":
                    chain = _strait_chain(hub, hub_lookup[v_base], hub_lookup)
                    if chain is None:
                        continue  # no sea passage between these basins
                    waypoints[1:1] = [hub_lookup[g] for g in chain]

                for h1, h2 in zip(waypoints, waypoints[1:]):
                    dist = _haversine(h1["lat"], h1["lon"], h2["lat"], h2["lon"])
                    t = _travel_time(dist, mode)
                    cost = dist * MODE_PROFILES[mode]["cost_per_km"]
                    a, b = f"{h1['id']}:{mode}", f"{h2['id']}:{mode}"
                    for x, y in ((a, b), (b, a)):
                        G.add_edge(x, y, baseline_time=t, distance=round(dist, 1),
                                   transport_mode=mode, type="transit", cost=cost)

    # 4. Local Road Auto-wire (<200km)
    for i, h1 in enumerate(hubs):
        if "road" not in h1["modes"]: continue
        for h2 in hubs[i+1:]:
            if "road" not in h2["modes"] or not land_link_possible(h1, h2): continue
            d = _haversine(h1["lat"], h1["lon"], h2["lat"], h2["lon"])
            if d < 200:
                u, v = f"{h1['id']}:road", f"{h2['id']}:road"
                if G.has_node(u) and G.has_node(v) and not G.has_edge(u, v):
                    t = _travel_time(d, "road")
                    cost = d * MODE_PROFILES["road"]["cost_per_km"]
                    G.add_edge(u, v, baseline_time=t, distance=round(d, 1), 
                               transport_mode="road", type="transit", cost=cost)
                    G.add_edge(v, u, baseline_time=t, distance=round(d, 1), 
                               transport_mode="road", type="transit", cost=cost)

    print(f"Split-Node Multimodal Network: {G.number_of_nodes()} virtual nodes, {G.number_of_edges()} edges")
    return G

def get_city_capabilities(G):
    city_data = {}
    for node, data in G.nodes(data=True):
        city = data.get("parent_city", data.get("display_name"))
        if city not in city_data:
            city_data[city] = {"id": data.get("physical_id"), "display_name": city, "country": data.get("country"), 
                               "has_port": False, "has_airport": False, "has_rail": False, "nodes": []}
        
        city_data[city]["nodes"].append(node)
        mode = data.get("mode")
        if mode == "sea": city_data[city]["has_port"] = True
        if mode == "air": city_data[city]["has_airport"] = True
        if mode == "rail": city_data[city]["has_rail"] = True

    return sorted(list(city_data.values()), key=lambda c: c["display_name"])
