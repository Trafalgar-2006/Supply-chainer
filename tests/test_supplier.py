import pytest

from backend.engine.scenario_manager import ScenarioManager
from backend.engine.supplier_scorer import SupplierScorer

from .conftest import ROOT

scorer = SupplierScorer(str(ROOT / "backend" / "data" / "suppliers.json"))


def by_id(ranked, supplier_id):
    return next(s for s in ranked if s["id"] == supplier_id)


def test_suez_block_adds_documented_lead_time_penalty():
    # Spec (supplier_scorer.py): penalty is 10% of the disruption delay, in days.
    disruptions = ScenarioManager().get_disruptions("SUEZ_BLOCK")
    normal = by_id(scorer.get_ranked_suppliers("Electronics"), "SUP-GLOBAL-01")
    hit = by_id(scorer.get_ranked_suppliers("Electronics", disruptions), "SUP-GLOBAL-01")
    assert hit["effective_lead_time"] - normal["effective_lead_time"] == pytest.approx(240 / 24 * 0.1)


def test_unaffected_supplier_is_not_penalised():
    disruptions = ScenarioManager().get_disruptions("SUEZ_BLOCK")
    normal = by_id(scorer.get_ranked_suppliers("Electronics"), "SUP-RESIL-02")
    hit = by_id(scorer.get_ranked_suppliers("Electronics", disruptions), "SUP-RESIL-02")
    assert hit["decision_score"] == normal["decision_score"]


@pytest.mark.parametrize("inventory, safety, forecast, status", [
    (1000, 1500, 800, "SAFETY_STOCK_VIOLATION"),
    (500, 1500, 800, "CRITICAL_SHORTAGE"),
    (5000, 1500, 800, "HEALTHY"),
])
def test_procurement_advice_status(inventory, safety, forecast, status):
    assert scorer.get_procurement_advice(inventory, safety, forecast)["status"] == status


def test_supplier_endpoint_does_not_change_global_scenario(client):
    from backend.main import scenario_mgr
    before = scenario_mgr.active_scenario_id
    r = client.post("/api/suppliers", json={"category": "Electronics", "scenario": "SUEZ_BLOCK"})
    assert r.status_code == 200
    assert r.json()["active_disruptions"]
    assert scenario_mgr.active_scenario_id == before
