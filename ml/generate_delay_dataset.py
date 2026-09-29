"""Generate the synthetic training set for the leg-delay quantile models.

There is no public dataset of per-leg freight delays, so the target is simulated
with the physics-informed priors of the original generator
(Code/dataset_generator_geo.py in the starter commit 8f15416): a dwell floor per mode, 10% transit-time
variability, weather multipliers and fat-tailed log-normal noise. Two changes
make the model usable on the live network:

1. Legs are sampled from the real routing graph (mode, distance and what they
   arrive at), so training and serving see the same distribution, where the
   original model was trained on 16 hub names that do not exist in the graph.
2. Disruption incidents drive both the delay and a noisy news signal
   (nlp_score), including false alarms and missed incidents, so the model
   learns how far the NLP stage can be trusted.

Usage: python ml/generate_delay_dataset.py [out.csv] [n_samples]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.engine.multimodal_network import create_multimodal_network  # noqa: E402
from backend.engine.delay_features import CONDITIONS, arrival_kind  # noqa: E402

SEED = 42

# Dwell at an arrival terminal (h), cruising speed (km/h) and log-normal sigma.
MODE_PARAMS = {
    "sea": {"dwell": 48.0, "speed": 35.0, "sigma": 0.9},   # berthing, customs, yard
    "air": {"dwell": 6.0, "speed": 800.0, "sigma": 0.9},   # ground handling
    "rail": {"dwell": 12.0, "speed": 60.0, "sigma": 0.6},  # scheduling, marshalling
    "road": {"dwell": 2.0, "speed": 68.0, "sigma": 0.6},   # gate queues, rest stops
}
# Share of the terminal dwell when a ship arrives at a canal or strait instead:
# a canal convoy queue is about 12 h, a strait passage about an hour.
ARRIVAL_DWELL_SHARE = {"terminal": 1.0, "canal": 0.25, "strait": 0.02}
CONDITION_PROB = [0.6, 0.3, 0.1]                           # clear, rainy, stormy
RAIN_MULT = 1.3
STORM_MULT = {"sea": 4.0, "air": 4.0, "rail": 2.0, "road": 2.0}
TRANSIT_VARIABILITY = 0.10
INCIDENT_PROB = 0.08          # strikes, closures, groundings, outages
INCIDENT_IMPACT = 4.0         # delay multiplier is 1 + 4 x severity (a full closure ~5x)
FALSE_ALARM_PROB = 0.03       # news flags a threat that does not materialise
MISSED_INCIDENT_PROB = 0.15   # an incident the news feed does not pick up


def network_legs():
    """Unique transit legs of the routing graph as (mode, distance_km, arrival)."""
    G = create_multimodal_network()
    legs = {}
    for u, v, d in G.edges(data=True):
        if d["type"] != "transit":
            continue
        a, b = G.nodes[u]["physical_id"], G.nodes[v]["physical_id"]
        legs[(a, b, d["transport_mode"])] = (d["transport_mode"], d["distance"], arrival_kind(b))
    return pd.DataFrame(list(legs.values()), columns=["mode", "distance_km", "arrival"])


def generate(n_samples: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    legs = network_legs()
    df = legs.iloc[rng.integers(0, len(legs), n_samples)].reset_index(drop=True)

    params = df["mode"].map(MODE_PARAMS)
    dwell = params.map(lambda p: p["dwell"]) * df["arrival"].map(ARRIVAL_DWELL_SHARE)
    transit_h = df["distance_km"] / params.map(lambda p: p["speed"])
    base = dwell + TRANSIT_VARIABILITY * transit_h

    condition = rng.choice(len(CONDITIONS), size=n_samples, p=CONDITION_PROB)
    weather = np.where(condition == 1, RAIN_MULT, 1.0)
    weather = np.where(condition == 2, df["mode"].map(STORM_MULT), weather)

    incident = rng.random(n_samples) < INCIDENT_PROB
    severity = np.where(incident, rng.uniform(0.3, 1.0, n_samples), 0.0)
    impact = 1.0 + INCIDENT_IMPACT * severity

    # The news signal tracks incidents imperfectly.
    seen = incident & (rng.random(n_samples) >= MISSED_INCIDENT_PROB)
    false_alarm = ~incident & (rng.random(n_samples) < FALSE_ALARM_PROB)
    nlp = np.abs(rng.normal(0.0, 0.05, n_samples))
    nlp = np.where(seen, severity + rng.normal(0.0, 0.1, n_samples), nlp)
    nlp = np.where(false_alarm, rng.uniform(0.2, 0.6, n_samples), nlp)

    sigma = params.map(lambda p: p["sigma"]).to_numpy()
    delay = rng.lognormal(np.log(base * weather * impact), sigma)

    return pd.DataFrame({
        "mode": df["mode"],
        "distance_km": df["distance_km"].round(1),
        "arrival": df["arrival"],
        "condition": [CONDITIONS[c] for c in condition],
        "nlp_score": np.clip(nlp, 0.0, 1.0).round(4),
        "delay_hours": delay.round(2),
    })


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "ml" / "delay_dataset.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
    data = generate(n)
    data.to_csv(out, index=False)
    print(f"{len(data)} legs -> {out}")
    print(data.groupby("mode")["delay_hours"].describe(percentiles=[0.5, 0.85, 0.95]).round(1))
