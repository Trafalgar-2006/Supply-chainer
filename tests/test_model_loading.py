from backend.engine.threat_intelligence import ThreatIntelligencePredictor


def test_model_loads_regardless_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    predictor = ThreatIntelligencePredictor()
    assert predictor.is_trained
    assert predictor.profiles, "calibration profiles should load too"
