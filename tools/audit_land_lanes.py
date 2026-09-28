"""Audit road/rail lanes against a 1 km global land mask.

Prints every land lane whose straight path crosses more than N km of open water
(default 20). Coastal lanes that cut across a bay show up too; the ones that
matter are crossings between landmasses with no bridge or tunnel.

Usage: python tools/audit_land_lanes.py [min_water_km]
"""
import math
import sys
from pathlib import Path

import numpy as np
from global_land_mask import globe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.engine.multimodal_network import create_multimodal_network


def gc_points(lat1, lon1, lat2, lon2, step_km=2.0):
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d = 2 * math.asin(math.sqrt(math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2))
    n = max(2, int(d * 6371 / step_km))
    f = np.linspace(0, 1, n)
    if d == 0:
        return np.full(n, lat1), np.full(n, lon1), 0.0
    A = np.sin((1 - f) * d) / math.sin(d)
    B = np.sin(f * d) / math.sin(d)
    x = A * math.cos(p1) * math.cos(l1) + B * math.cos(p2) * math.cos(l2)
    y = A * math.cos(p1) * math.sin(l1) + B * math.cos(p2) * math.sin(l2)
    z = A * math.sin(p1) + B * math.sin(p2)
    lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon, d * 6371


def longest_water_km(lat1, lon1, lat2, lon2, trim_km=10.0):
    lat, lon, dist = gc_points(lat1, lon1, lat2, lon2)
    if dist == 0:
        return 0.0, 0.0
    land = globe.is_land(lat, lon)
    step = dist / (len(lat) - 1)
    trim = int(trim_km / step)
    core = land[trim: len(land) - trim] if len(land) > 2 * trim else land[:0]
    best = run = 0
    for is_land in core:
        run = 0 if is_land else run + 1
        best = max(best, run)
    return best * step, dist


if __name__ == "__main__":
    G = create_multimodal_network()
    seen = set()
    rows = []
    for u, v, d in G.edges(data=True):
        if d["transport_mode"] not in ("road", "rail"):
            continue
        a, b = G.nodes[u], G.nodes[v]
        key = tuple(sorted((a["physical_id"], b["physical_id"]))) + (d["transport_mode"],)
        if key in seen:
            continue
        seen.add(key)
        water, dist = longest_water_km(a["lat"], a["lon"], b["lat"], b["lon"])
        if water > float(sys.argv[1] if len(sys.argv) > 1 else 20):
            rows.append((round(water), round(dist), d["transport_mode"], a["physical_id"], a["country"], b["physical_id"], b["country"]))
    rows.sort(reverse=True)
    print(len(seen), "land lanes checked;", len(rows), "cross open water")
    for r in rows:
        print(r)
