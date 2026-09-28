import json
from collections import Counter

from backend.engine.scenario_manager import ScenarioManager

from .conftest import ROOT

DATA = ROOT / "backend" / "data"


def test_hub_ids_are_unique(hubs):
    dupes = [hub_id for hub_id, n in Counter(h["id"] for h in hubs).items() if n > 1]
    assert dupes == []


def test_every_connection_targets_an_existing_hub(hubs, hub_index):
    dangling = sorted({c["to"] for h in hubs for c in h.get("connections", []) if c["to"] not in hub_index})
    assert dangling == []


def test_no_self_loop_connections(hubs):
    loops = [h["id"] for h in hubs for c in h.get("connections", []) if c["to"] == h["id"]]
    assert loops == []


def test_connection_modes_exist_on_both_ends(hubs, hub_index):
    bad = [
        (h["id"], c["to"], c["mode"])
        for h in hubs
        for c in h.get("connections", [])
        if c["to"] in hub_index and (c["mode"] not in h["modes"] or c["mode"] not in hub_index[c["to"]]["modes"])
    ]
    assert bad == []


def test_location_map_points_to_existing_hubs(hub_index):
    locations = json.loads((DATA / "canonical_locations.json").read_text())
    missing = sorted({hub_id for modes in locations.values() for hub_id in modes.values() if hub_id not in hub_index})
    assert missing == []


def test_scenario_nodes_exist(hub_index):
    missing = [
        (sid, node)
        for sid, s in ScenarioManager.SCENARIOS.items()
        for node in s["affected_nodes"]
        if node not in hub_index
    ]
    assert missing == []


def test_supplier_hubs_exist(hub_index):
    suppliers = json.loads((DATA / "suppliers.json").read_text())
    missing = [
        (s["id"], node)
        for s in suppliers
        for node in [s["location_hub"], *s.get("transit_choke_points", [])]
        if node not in hub_index
    ]
    assert missing == []
