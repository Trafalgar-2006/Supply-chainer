"""Train and evaluate the p50/p85/p95 leg-delay quantile models.

One gradient-boosted quantile regressor per quantile, with native categorical
handling and monotonic constraints (distance, weather and news severity can only
push a delay up). Evaluated on a held-out 20% split against a naive baseline that
predicts the empirical quantile of each (mode, arrival) group:

- pinball loss per quantile (lower is better)
- coverage: share of held-out delays at or below the predicted quantile, which
  should be close to the quantile itself
- quantile crossing rate before the p50 <= p85 <= p95 ordering is enforced
- permutation importance of each feature for the p85 model

Writes Execution/delay_quantile_model.joblib and a JSON model report that
records the artifact's SHA-256, which the API verifies before unpickling.

Usage: python ml/train_delay_model.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.engine.delay_features import CATEGORICAL, FEATURES, MODES, MONOTONIC, QUANTILES, encode  # noqa: E402
from ml.generate_delay_dataset import SEED, generate  # noqa: E402

ARTIFACT = ROOT / "Execution" / "delay_quantile_model.joblib"
REPORT = ROOT / "Execution" / "delay_quantile_model.json"
N_SAMPLES = 60_000


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    return encode(df["mode"], df["distance_km"], df["arrival"], df["condition"], df["nlp_score"])


def fit_quantile(X, y, q):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=CATEGORICAL,
        monotonic_cst=MONOTONIC, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=SEED,
    ).fit(X, y)


def main():
    data = generate(N_SAMPLES)
    train, test = train_test_split(data, test_size=0.2, random_state=SEED)
    X_train, X_test = to_matrix(train), to_matrix(test)
    y_train, y_test = train["delay_hours"].to_numpy(), test["delay_hours"].to_numpy()

    models = {q: fit_quantile(X_train, y_train, q) for q in QUANTILES}
    raw = np.column_stack([models[q].predict(X_test) for q in QUANTILES])
    crossing_rate = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
    pred = np.maximum.accumulate(raw, axis=1)

    groups = ["mode", "arrival"]
    report = {"n_train": len(train), "n_test": len(test), "features": FEATURES,
              "quantile_crossing_rate": round(crossing_rate, 4), "quantiles": {}}
    for i, q in enumerate(QUANTILES):
        naive = train.groupby(groups)["delay_hours"].quantile(q)
        naive_pred = test.set_index(groups).index.map(naive).to_numpy(dtype=float)
        model_loss = mean_pinball_loss(y_test, pred[:, i], alpha=q)
        naive_loss = mean_pinball_loss(y_test, naive_pred, alpha=q)
        by_mode = {m: round(float(np.mean(y_test[test["mode"].to_numpy() == m]
                                          <= pred[test["mode"].to_numpy() == m, i])), 3) for m in MODES}
        report["quantiles"][f"p{int(q * 100)}"] = {
            "coverage": round(float(np.mean(y_test <= pred[:, i])), 3),
            "coverage_by_mode": by_mode,
            "pinball_loss": round(model_loss, 3),
            "naive_pinball_loss": round(naive_loss, 3),
            "improvement_vs_naive": round(1 - model_loss / naive_loss, 3),
            "iterations": int(models[q].n_iter_),
        }

    p85 = models[0.85]
    imp = permutation_importance(p85, X_test, y_test, n_repeats=5, random_state=SEED,
                                 scoring=lambda est, X, y: -mean_pinball_loss(y, est.predict(X), alpha=0.85))
    report["p85_permutation_importance"] = {
        f: round(float(v), 3) for f, v in sorted(zip(FEATURES, imp.importances_mean), key=lambda t: -t[1])
    }

    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump({"models": {str(q): models[q] for q in QUANTILES}, "features": FEATURES,
                 "quantiles": list(QUANTILES)}, ARTIFACT, compress=3)
    report["artifact"] = ARTIFACT.name
    report["sha256"] = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
