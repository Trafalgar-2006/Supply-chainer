"""Leg-delay quantile model (p50 / p85 / p95) used by the router.

Trained by ml/train_delay_model.py; see docs/MODEL_CARD.md. Predictions are the
extra hours a leg takes beyond its nominal transit time (dwell, handling,
variability, weather and disruption), at three quantiles.
"""
import hashlib
import io
import json
import os
from itertools import combinations
from math import factorial

import joblib
import numpy as np

from .delay_features import ARRIVALS, CONDITIONS, FEATURES, MODES, QUANTILES, encode

EXECUTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Execution")
MODEL_PATH = os.path.join(EXECUTION_DIR, "delay_quantile_model.joblib")
REPORT_PATH = os.path.join(EXECUTION_DIR, "delay_quantile_model.json")
# Pinned in code, so whoever can write the artifact cannot also update its hash.
# Retraining prints the new digest; updating it here is a reviewed code change.
EXPECTED_SHA256 = "85248c56c50f386f0de13916ef935d2360e5a3e94877c23659cbf2454f80cf35"

# Shapley reference: a short road hop into a terminal in clear weather with no news.
# Each feature's contribution is measured against this leg.
REFERENCE = {"mode": "road", "distance_km": 100.0, "arrival": "terminal", "condition": "clear", "nlp_score": 0.0}


class ModelIntegrityError(RuntimeError):
    pass


class DelayQuantileModel:
    def __init__(self, models, report):
        self._models = [models[str(q)] for q in QUANTILES]
        self.report = report
        n = len(FEATURES)
        # Exact Shapley weights |S|!(n-|S|-1)!/n! for every coalition S without feature i.
        self._coalitions = [
            (i, mask, factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n))
            for i in range(n)
            for size in range(n)
            for s in combinations([j for j in range(n) if j != i], size)
            for mask in [sum(1 << j for j in s)]
        ]

    @classmethod
    def load(cls, model_path=MODEL_PATH, report_path=REPORT_PATH, expected_sha256=EXPECTED_SHA256):
        """Load the model after checking its SHA-256 against the pinned digest.

        Unpickling runs code, so a tampered or swapped artifact is refused before
        it is ever deserialised. The bytes that are hashed are the bytes that are
        loaded, so the file cannot be swapped between the check and the load.
        """
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        with open(model_path, "rb") as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != expected_sha256:
            raise ModelIntegrityError(f"{os.path.basename(model_path)} does not match the pinned SHA-256")
        bundle = joblib.load(io.BytesIO(blob))
        return cls(bundle["models"], report)

    def _predict_matrix(self, X):
        raw = np.column_stack([m.predict(X) for m in self._models])
        # Enforce p50 <= p85 <= p95 and non-negative delays.
        return np.maximum.accumulate(np.clip(raw, 0.0, None), axis=1)

    def predict(self, mode, distance_km, arrival, condition, nlp_score):
        """Quantile delays (n x 3 hours, columns p50/p85/p95) for arrays of legs."""
        return self._predict_matrix(encode(mode, distance_km, arrival, condition, nlp_score))

    def explain(self, legs, quantile_index=1):
        """Exact Shapley contributions (hours) of each feature, per leg.

        `legs` is a list of dicts with the five raw features. With five features all
        32 coalitions are evaluated, so the values are exact (no sampling) and each
        leg's contributions sum to prediction(leg) - prediction(REFERENCE).
        Returns (reference_hours, array n_legs x n_features).
        """
        n = len(FEATURES)
        x = encode(*[[leg[f] for leg in legs] for f in FEATURES])
        ref = encode(*[[REFERENCE[f]] for f in FEATURES])[0]
        masks = np.arange(1 << n)
        present = (masks[:, None] >> np.arange(n)) & 1  # 32 x n
        rows = np.where(present[None, :, :], x[:, None, :], ref[None, None, :]).reshape(-1, n)
        model = self._models[quantile_index]
        values = np.clip(model.predict(rows), 0.0, None).reshape(len(legs), 1 << n)
        phi = np.zeros((len(legs), n))
        for i, mask, weight in self._coalitions:
            phi[:, i] += weight * (values[:, mask | (1 << i)] - values[:, mask])
        return float(values[0, 0]) if legs else 0.0, phi


def label(feature, value):
    if feature == "mode":
        return f"{value.upper()} leg"
    if feature == "distance_km":
        return "Distance"
    if feature == "arrival":
        return "Arrives at a terminal" if value == "terminal" else f"{value.title()} passage"
    if feature == "condition":
        return f"Weather: {value}"
    return "News threat signal"


__all__ = ["DelayQuantileModel", "ModelIntegrityError", "REFERENCE", "label", "ARRIVALS", "CONDITIONS", "MODES"]
