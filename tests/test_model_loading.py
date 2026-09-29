from backend.engine.delay_model import DelayQuantileModel
from backend.engine.threat_intelligence import LABELLED_HEADLINES


def test_model_and_data_load_regardless_of_working_directory(tmp_path, monkeypatch):
    # Paths are resolved from the package, so the server can start from anywhere.
    monkeypatch.chdir(tmp_path)
    assert DelayQuantileModel.load().report["n_test"] > 0
    with open(LABELLED_HEADLINES, encoding="utf-8") as f:
        assert f.readline().startswith("split,label,threat_type")
