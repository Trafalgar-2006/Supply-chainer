import networkx as nx
import pytest

from backend.engine.multimodal_network import (
    _haversine, create_multimodal_network, land_link_possible, land_route_km, sea_basin,
)


@pytest.fixture(scope="module")
def G():
    return create_multimodal_network()


@pytest.fixture(scope="module")
def sea(G):
    return G.subgraph(n for n, d in G.nodes(data=True) if d["mode"] == "sea")


def test_every_land_and_air_lane_is_navigable_both_ways(G, hubs):
    missing = [
        (h["id"], c["to"], c["mode"])
        for h in hubs
        for c in h.get("connections", [])
        if c["mode"] != "sea"
        and not (G.has_edge(f'{h["id"]}:{c["mode"]}', f'{c["to"]}:{c["mode"]}')
                 and G.has_edge(f'{c["to"]}:{c["mode"]}', f'{h["id"]}:{c["mode"]}'))
    ]
    assert missing == []


def test_every_possible_sea_lane_is_navigable_both_ways(sea, hubs, hub_index):
    missing = []
    for h in hubs:
        for c in h.get("connections", []):
            if c["mode"] != "sea":
                continue
            u, v = f'{h["id"]}:sea', f'{c["to"]}:sea'
            if "CASPIAN" in (sea.nodes[u]["basins"] + sea.nodes[v]["basins"]):
                continue
            if not (nx.has_path(sea, u, v) and nx.has_path(sea, v, u)):
                missing.append((u, v))
    assert missing == []


def test_sea_lanes_never_cross_land_between_basins(sea):
    crossing = [(a, b) for a, b in sea.edges if not set(sea.nodes[a]["basins"]) & set(sea.nodes[b]["basins"])]
    assert crossing == []


@pytest.mark.parametrize("inside, outside, straits", [
    ("PORT-JEBEL", "PORT-SINGAPORE", ["CHOKE-HORMUZ"]),
    ("PORT-JEDDAH", "PORT-COLOMBO", ["CHOKE-BABEL", "CHOKE-SUEZ"]),
    ("PORT-PIRAEUS", "PORT-ROTTERDAM", ["CHOKE-GIBRAL", "CHOKE-SUEZ"]),
    ("PORT-CONSTANTA", "PORT-PIRAEUS", ["CHOKE-BOSPHO"]),
])
def test_enclosed_seas_are_only_reachable_through_their_straits(sea, inside, outside, straits):
    assert nx.has_path(sea, f"{inside}:sea", f"{outside}:sea")
    sealed = sea.subgraph(n for n in sea if n not in {f"{s}:sea" for s in straits})
    assert not nx.has_path(sealed, f"{inside}:sea", f"{outside}:sea")


@pytest.mark.parametrize("a, b, strait", [
    ("PORT-SHANGHAI", "PORT-COLOMBO", "CHOKE-MALACCA"),
    ("PORT-SHANGHAI", "PORT-JEBEL", "CHOKE-HORMUZ"),
    ("PORT-FREMANTLE", "PORT-MELBOURNE", "CHOKE-CAPELEEUWIN"),
])
def test_shortest_sea_route_passes_the_real_strait(sea, a, b, strait):
    path = nx.shortest_path(sea, f"{a}:sea", f"{b}:sea", weight="distance")
    assert f"{strait}:sea" in path


def test_landlocked_caspian_has_no_ocean_lanes(sea):
    for port in ("PORT-BAKU:sea", "RAIL-AKTAU:sea"):
        neighbours = set(sea.successors(port)) | set(sea.predecessors(port))
        assert all("CASPIAN" in sea.nodes[n]["basins"] for n in neighbours)


@pytest.mark.parametrize("lat, lon, basin", [
    (25.01, 55.06, "PERSIAN_GULF"),   # Jebel Ali
    (25.12, 56.34, "INDIAN_OCEAN"),   # Fujairah, outside Hormuz
    (21.49, 39.18, "RED_SEA"),        # Jeddah
    (11.59, 43.15, "INDIAN_OCEAN"),   # Djibouti, outside Bab el-Mandeb
    (31.26, 32.30, "MEDITERRANEAN"),  # Port Said
    (36.13, -5.45, "MEDITERRANEAN"),  # Algeciras
    (43.26, -2.93, "ATLANTIC"),       # Bilbao
    (59.93, 30.30, "ATLANTIC"),       # St Petersburg (Baltic)
    (9.36, -79.90, "ATLANTIC"),       # Colon
    (8.95, -79.57, "PACIFIC"),   # Balboa
    (40.37, 49.85, "CASPIAN"),        # Baku
    (16.17, -95.20, "PACIFIC"),  # Salina Cruz
    (13.92, -90.79, "PACIFIC"),  # Puerto Quetzal
    (12.48, -87.17, "PACIFIC"),  # Corinto
    (9.98, -84.83, "PACIFIC"),   # Puntarenas
    (3.88, -77.07, "PACIFIC"),   # Buenaventura
    (19.20, -96.13, "ATLANTIC"),      # Veracruz
    (18.14, -94.41, "ATLANTIC"),      # Coatzacoalcos
    (10.00, -83.03, "ATLANTIC"),      # Puerto Limon
    (23.14, -82.36, "ATLANTIC"),      # Havana
    (10.40, -75.50, "ATLANTIC"),      # Cartagena
    (3.00, 101.39, "INDIAN_OCEAN"),   # Port Klang, Strait of Malacca
    (1.26, 103.82, "PACIFIC"),        # Singapore, east of the Malacca chokepoint
    (13.08, 100.88, "PACIFIC"),       # Laem Chabang, Gulf of Thailand
    (6.94, 79.84, "INDIAN_OCEAN"),    # Colombo
    (-32.06, 115.74, "INDIAN_OCEAN"), # Fremantle
    (-37.82, 144.91, "PACIFIC"),      # Melbourne
    (-6.10, 106.89, "PACIFIC"),       # Jakarta, Java Sea
])
def test_sea_basin_classification(lat, lon, basin):
    assert sea_basin(lat, lon) == basin


def test_land_lanes_stay_on_one_landmass(G, hub_index):
    crossing = [
        (a, b) for a, b, d in G.edges(data=True)
        if d["transport_mode"] in ("road", "rail")
        and not land_link_possible(hub_index[G.nodes[a]["physical_id"]], hub_index[G.nodes[b]["physical_id"]])
    ]
    assert crossing == []


@pytest.mark.parametrize("a, b, mode", [
    ("ICD-RIYADH", "PORT-PORTSUDAN", "rail"),       # Red Sea
    ("PORT-PIRAEUS", "PORT-ALEXANDR", "rail"),      # Mediterranean
    ("PORT-BANDARABBAS", "HUB-DAMMAM", "rail"),     # Persian Gulf
    ("PORT-SHENZHEN", "PORT-KAOHSIUNG", "rail"),    # Taiwan Strait
    ("PORT-TANGIER", "PORT-VALENCIA", "rail"),      # Strait of Gibraltar
    ("PORT-BUSAN", "PORT-DALIAN", "rail"),          # North Korea / Yellow Sea
    ("PORT-TPELEPAS", "PORT-TANJUNGSAUH", "road"),  # Singapore Strait
])
def test_impossible_land_lanes_are_absent(G, a, b, mode):
    assert not G.has_edge(f"{a}:{mode}", f"{b}:{mode}")
    assert not G.has_edge(f"{b}:{mode}", f"{a}:{mode}")


def test_channel_tunnel_links_britain_to_the_continent(G):
    assert G.has_edge("BORDER-DOVER:road", "HUB-CALAIS:road")
    assert G.has_edge("HUB-CALAIS:road", "BORDER-DOVER:road")


def test_britain_to_mainland_land_lanes_run_through_the_tunnel(G, hub_index):
    felixstowe, rotterdam = hub_index["PORT-FELIXSTOWE"], hub_index["HUB-ROTTERDAM"]
    straight = _haversine(felixstowe["lat"], felixstowe["lon"], rotterdam["lat"], rotterdam["lon"])
    edge = G["PORT-FELIXSTOWE:road"]["HUB-ROTTERDAM:road"]
    assert edge["distance"] == pytest.approx(land_route_km(felixstowe, rotterdam), abs=0.1)
    assert edge["distance"] > straight * 1.5  # the North Sea is not crossed by truck


def test_far_britain_mainland_links_are_kept_via_the_tunnel():
    london = {"country": "UK", "lat": 51.5, "lon": -0.12}
    paris = {"country": "France", "lat": 48.86, "lon": 2.35}
    assert land_route_km(london, paris) == pytest.approx(land_route_km(paris, london))
    assert land_route_km(london, paris) > _haversine(51.5, -0.12, 48.86, 2.35)


def test_every_chokepoint_can_be_entered_and_left(G, hubs):
    stuck = [
        h["id"] for h in hubs if h["type"] == "choke_point"
        if G.in_degree(f'{h["id"]}:sea') == 0 or G.out_degree(f'{h["id"]}:sea') == 0
    ]
    assert stuck == []


def test_every_seaport_outside_the_caspian_is_on_one_navigable_ocean(sea):
    largest = max(nx.strongly_connected_components(sea), key=len)
    stranded = sorted(n for n in sea if n not in largest and "CASPIAN" not in sea.nodes[n]["basins"])
    assert stranded == []


def test_every_hub_is_reachable_from_the_main_network(G):
    largest = max(nx.strongly_connected_components(G), key=len)
    physical = {d["physical_id"] for n, d in G.nodes(data=True)}
    reachable = {G.nodes[n]["physical_id"] for n in largest}
    assert len(reachable) / len(physical) > 0.9
