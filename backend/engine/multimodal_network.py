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

# Enclosed seas and the ocean basins can only be crossed through real straits and
# capes. The hub data lists direct lanes such as Jebel Ali -> Haifa, which would sail
# across the Arabian Peninsula, or Shanghai -> Jebel Ali, which would cross China and
# India. Every cross-basin lane is routed through its chokepoints instead, so a
# Hormuz, Bab el-Mandeb, Suez, Gibraltar or Malacca disruption reaches every ship
# that actually passes it.
SEA_GATES = {
    "CHOKE-HORMUZ": ("PERSIAN_GULF", "INDIAN_OCEAN"),
    "CHOKE-BABEL": ("RED_SEA", "INDIAN_OCEAN"),
    "CHOKE-SUEZ": ("RED_SEA", "MEDITERRANEAN"),
    "CHOKE-GIBRAL": ("MEDITERRANEAN", "ATLANTIC"),
    "CHOKE-DARDANELLES": ("MEDITERRANEAN", "MARMARA"),
    "CHOKE-BOSPHO": ("MARMARA", "BLACK_SEA"),
    "CHOKE-CAPEGOOD": ("INDIAN_OCEAN", "ATLANTIC"),
    "CHOKE-PANAMA": ("PACIFIC", "ATLANTIC"),
    "CHOKE-MALACCA": ("INDIAN_OCEAN", "PACIFIC"),
    "CHOKE-LOMBOK": ("INDIAN_OCEAN", "PACIFIC"),
    "CHOKE-CAPELEEUWIN": ("INDIAN_OCEAN", "PACIFIC"),  # south of Australia
}

def sea_basin(lat, lon):
    """Coarse ocean basin of a coastal point; the most enclosed seas are checked first."""
    if 38.0 <= lat <= 47.5 and 46.5 <= lon <= 55.0: return "CASPIAN"  # landlocked
    if 23.5 <= lat <= 30.5 and 47.5 <= lon < 56.3: return "PERSIAN_GULF"
    if 12.7 <= lat < 30.0 and 32.2 <= lon <= 43.6: return "RED_SEA"
    if 40.3 <= lat <= 41.08 and 26.5 <= lon <= 30.0: return "MARMARA"
    if 40.9 < lat <= 47.5 and 27.4 <= lon <= 42.0: return "BLACK_SEA"
    if 30.0 <= lat <= 46.0 and -5.7 <= lon <= 36.5 and not (lon < 3.0 and lat > 42.5): return "MEDITERRANEAN"
    if lon < -30:
        return "PACIFIC" if _pacific_americas(lat, lon) else "ATLANTIC"
    # North Sea, Baltic and Barents. The Arctic Northern Sea Route (CHOKE-NSR) is a
    # seasonal, ice-class passage, kept as a Pacific-side spur rather than a
    # year-round Asia-Europe gate.
    if lat > 50 and lon < 60: return "ATLANTIC"
    if lon < 20: return "ATLANTIC"
    return "PACIFIC" if _pacific_asia(lat, lon) else "INDIAN_OCEAN"

def _pacific_asia(lat, lon):
    """East of the Malacca Strait / Indonesian archipelago, and eastern Australia."""
    if lon < 97.5:
        return False                      # Arabian Sea, Bay of Bengal, Andaman Sea
    if lat < -10:
        return lon >= 118.0               # Australia: west coast faces the Indian Ocean
    if lon < 104.0:
        # Malay Peninsula: the Gulf of Thailand and the coast east of the Malacca
        # chokepoint face the South China Sea; the Strait of Malacca coast does not.
        return lat > 10.5 or lon > 102.25
    return True

# Continental divide of Central America as (lon, lat) points, from the Isthmus of
# Tehuantepec to Panama: a coast south-west of this line faces the Pacific.
_CENTRAL_AMERICA_DIVIDE = [(-100.0, 18.5), (-95.0, 17.2), (-91.0, 15.5), (-87.5, 14.0),
                           (-85.0, 11.0), (-82.5, 8.8), (-79.7, 9.15), (-77.0, 8.0)]

def _pacific_americas(lat, lon):
    if lon <= -100:
        return True
    if lon >= -77:  # South America east of the Darien: only the Pacific coast of Colombia to Chile
        return lat < 7 and lon < -70
    for (x1, y1), (x2, y2) in zip(_CENTRAL_AMERICA_DIVIDE, _CENTRAL_AMERICA_DIVIDE[1:]):
        if x1 <= lon <= x2:
            return lat < y1 + (lon - x1) / (x2 - x1) * (y2 - y1)
    return False

# Road and rail cannot cross open sea. Islands are reachable over land only through
# a fixed link, and the Channel Tunnel is the only one between an island in the
# registry and a continent: land lanes between Britain and the mainland run
# through its portals.
ISLAND_LANDMASSES = {"UK", "Ireland", "Japan", "Taiwan", "Sri Lanka", "Indonesia", "Philippines",
                     "Australia", "New Zealand", "Madagascar", "Iceland"}
CHANNEL_TUNNEL_PORTALS = ((51.096, 1.137), (50.925, 1.813))  # Folkestone (UK), Coquelles (FR)

def landmass(hub):
    if hub["country"] in ISLAND_LANDMASSES:
        return hub["country"]
    return "AMERICAS" if hub["lon"] < -30 else "AFRO_EURASIA"

def land_route_km(h1, h2):
    """Length of a road/rail lane between two hubs, or None if no land link exists."""
    a, b = landmass(h1), landmass(h2)
    if a == b:
        return _haversine(h1["lat"], h1["lon"], h2["lat"], h2["lon"])
    if {a, b} == {"UK", "AFRO_EURASIA"}:
        uk_portal, fr_portal = CHANNEL_TUNNEL_PORTALS
        start, end = (uk_portal, fr_portal) if a == "UK" else (fr_portal, uk_portal)
        return (_haversine(h1["lat"], h1["lon"], *start) + _haversine(*start, *end)
                + _haversine(*end, h2["lat"], h2["lon"]))
    return None

def land_link_possible(h1, h2):
    return land_route_km(h1, h2) is not None

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

def _add_lane(G, a, b, mode, dist):
    """Add a transit lane of `dist` km in both directions."""
    t = _travel_time(dist, mode)
    cost = dist * MODE_PROFILES[mode]["cost_per_km"]
    for x, y in ((a, b), (b, a)):
        G.add_edge(x, y, baseline_time=t, distance=round(dist, 1),
                   transport_mode=mode, type="transit", cost=cost)

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
                target = hub_lookup[v_base]
                if mode == "sea":
                    chain = _strait_chain(hub, target, hub_lookup)
                    if chain is None:
                        continue  # no sea passage between these basins
                    waypoints = [hub] + [hub_lookup[g] for g in chain] + [target]
                    segments = [(a, b, _haversine(a["lat"], a["lon"], b["lat"], b["lon"]))
                                for a, b in zip(waypoints, waypoints[1:])]
                elif mode in ("road", "rail"):
                    dist = land_route_km(hub, target)
                    if dist is None:
                        continue  # no land link between these landmasses
                    segments = [(hub, target, dist)]
                else:
                    segments = [(hub, target, _haversine(hub["lat"], hub["lon"], target["lat"], target["lon"]))]

                for h1, h2, dist in segments:
                    _add_lane(G, f"{h1['id']}:{mode}", f"{h2['id']}:{mode}", mode, dist)

    # 4. Local Road Auto-wire (hubs < 200 km apart on the same landmass)
    for i, h1 in enumerate(hubs):
        if "road" not in h1["modes"]: continue
        for h2 in hubs[i+1:]:
            if "road" not in h2["modes"]: continue
            if _haversine(h1["lat"], h1["lon"], h2["lat"], h2["lon"]) >= 200: continue
            dist = land_route_km(h1, h2)
            u, v = f"{h1['id']}:road", f"{h2['id']}:road"
            if dist is not None and G.has_node(u) and G.has_node(v) and not G.has_edge(u, v):
                _add_lane(G, u, v, "road", dist)

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
