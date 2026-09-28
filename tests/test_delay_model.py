import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from backend.engine.delay_features import FEATURES, QUANTILES
from backend.engine.delay_model import (
    MODEL_PATH, REFERENCE, REPORT_PATH, DelayQuantileModel, ModelIntegrityError,
)


@pytest.fixture(scope="module")
def model():
    return DelayQuantileModel.load()


def leg(**overrides):
    base = {"mode": "sea", "distance_km": 5000.0, "arrival": "terminal", "condition": "clear", "nlp_score": 0.0}
    return {**base, **overrides}


def predict(model, *legs):
    return model.predict(*[[l[f] for l in legs] for f in FEATURES])


def test_held_out_coverage_matches_each_quantile():
    report = json.loads(Path(REPORT_PATH).read_text())
    for q in QUANTILES:
        stats = report["quantiles"][f"p{int(q * 100)}"]
        assert stats["coverage"] == pytest.approx(q, abs=0.03)
        assert stats["pinball_loss"] < stats["naive_pinball_loss"]


def test_quantiles_are_ordered_and_non_negative(model):
    legs = [leg(mode=m, distance_km=d, condition=c, nlp_score=n)
            for m in ("air", "rail", "road", "sea") for d in (50, 2000, 12000)
            for c in ("clear", "stormy") for n in (0.0, 0.9)]
    q = predict(model, *legs)
    assert (q >= 0).all()
    assert (np.diff(q, axis=1) >= 0).all()


@pytest.mark.parametrize("feature, low, high", [
    ("distance_km", 500.0, 12000.0),
    ("condition", "clear", "stormy"),
    ("nlp_score", 0.0, 0.9),
])
def test_monotonic_in_distance_weather_and_news(model, feature, low, high):
    lo, hi = predict(model, leg(**{feature: low}), leg(**{feature: high}))
    assert (hi >= lo).all()


def test_canal_and_strait_passages_dwell_less_than_a_port_call(model):
    terminal, canal, strait = predict(model, leg(), leg(arrival="canal"), leg(arrival="strait"))
    assert terminal[1] > canal[1] > strait[1]


def test_shapley_values_are_exact(model):
    legs = [leg(condition="stormy", nlp_score=0.8), leg(mode="air", distance_km=800.0), dict(REFERENCE)]
    reference, phi = model.explain(legs, quantile_index=1)
    raw = np.clip(model._models[1].predict(model_rows(legs)), 0, None)
    # Efficiency: contributions add up to prediction minus the reference prediction.
    assert phi.sum(axis=1) == pytest.approx(raw - reference, abs=1e-6)
    # Dummy: a feature at its reference value contributes nothing.
    assert phi[1][FEATURES.index("arrival")] == pytest.approx(0.0, abs=1e-9)
    assert phi[2] == pytest.approx(np.zeros(len(FEATURES)), abs=1e-9)


def model_rows(legs):
    from backend.engine.delay_features import encode
    return encode(*[[l[f] for l in legs] for f in FEATURES])


def test_the_artifact_matches_the_digest_pinned_in_code():
    import hashlib
    from backend.engine.delay_model import EXPECTED_SHA256
    assert hashlib.sha256(Path(MODEL_PATH).read_bytes()).hexdigest() == EXPECTED_SHA256
    assert json.loads(Path(REPORT_PATH).read_text())["sha256"] == EXPECTED_SHA256


def test_a_rewritten_report_hash_does_not_bypass_the_pin(tmp_path):
    import hashlib
    tampered = tmp_path / "model.joblib"
    tampered.write_bytes(Path(MODEL_PATH).read_bytes() + b"payload")
    report = json.loads(Path(REPORT_PATH).read_text())
    report["sha256"] = hashlib.sha256(tampered.read_bytes()).hexdigest()
    forged = tmp_path / "report.json"
    forged.write_text(json.dumps(report))
    with pytest.raises(ModelIntegrityError):
        DelayQuantileModel.load(model_path=str(tampered), report_path=str(forged))


def test_tampered_artifact_is_refused_before_unpickling(tmp_path):
    tampered = tmp_path / "model.joblib"
    shutil.copy(MODEL_PATH, tampered)
    data = bytearray(tampered.read_bytes())
    data[-1] ^= 0xFF
    tampered.write_bytes(bytes(data))
    with pytest.raises(ModelIntegrityError):
        DelayQuantileModel.load(model_path=str(tampered), report_path=REPORT_PATH)
