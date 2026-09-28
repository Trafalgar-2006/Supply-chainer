"""Feature schema shared by the delay model (API) and its training scripts in ml/.

A leg is described by five features that exist for every edge of the live
network, so the model generalises to all hubs instead of a fixed list of names:

- mode         transport mode of the leg (categorical)
- distance_km  length of the leg
- arrival      what the leg arrives at: a terminal, a canal or a strait (categorical)
- condition    weather: 0 clear, 1 rainy, 2 stormy
- nlp_score    news threat score from the contrastive NLP stage, 0..1
"""
import numpy as np

MODES = ["air", "rail", "road", "sea"]
ARRIVALS = ["terminal", "canal", "strait"]
CONDITIONS = ["clear", "rainy", "stormy"]
FEATURES = ["mode", "distance_km", "arrival", "condition", "nlp_score"]
CATEGORICAL = [True, False, True, False, False]
# More distance, worse weather or a worse news signal can never shorten a delay.
MONOTONIC = [0, 1, 0, 1, 1]
QUANTILES = (0.50, 0.85, 0.95)
CANALS = {"CHOKE-SUEZ", "CHOKE-PANAMA"}


def arrival_kind(hub_id: str) -> str:
    if hub_id in CANALS:
        return "canal"
    return "strait" if hub_id.startswith("CHOKE-") else "terminal"


def encode(mode, distance_km, arrival, condition, nlp_score) -> np.ndarray:
    """Encode array-likes of raw feature values into the model's numeric matrix."""
    return np.column_stack([
        np.asarray([MODES.index(m) for m in np.atleast_1d(mode)], dtype=float),
        np.asarray(distance_km, dtype=float).reshape(-1),
        np.asarray([ARRIVALS.index(a) for a in np.atleast_1d(arrival)], dtype=float),
        np.asarray([CONDITIONS.index(c) for c in np.atleast_1d(condition)], dtype=float),
        np.asarray(nlp_score, dtype=float).reshape(-1),
    ])
