import React, { useState, useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';

// Inputs keep whatever the user typed (possibly empty mid-edit); requests use a
// clean count, never NaN or a negative number.
const toCount = value => Math.max(0, Number.parseInt(value, 10) || 0);
const REQUEST_DELAY_MS = 300;
const sentence = s => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase().replace(/_/g, ' ');

export default function SupplierIntelligence({ onNavigate }) {
  const [suppliers, setSuppliers] = useState([]);
  const [advice, setAdvice] = useState(null);
  const [disruptions, setDisruptions] = useState({});
  const [inventory, setInventory] = useState('1000');
  const [safetyStock, setSafetyStock] = useState('1500');
  const [forecast, setForecast] = useState('800');
  const [category, setCategory] = useState('Electronics');
  const [scenario, setScenario] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [hubNames, setHubNames] = useState({});
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/scenarios')
      .then(r => r.json())
      .then(setScenarios)
      .catch(e => console.error('Failed to load scenarios', e));
    fetch('/api/hubs')
      .then(r => r.json())
      .then(hubs => setHubNames(Object.fromEntries(hubs.map(h => [h.id, h.display_name]))))
      .catch(e => console.error('Failed to load hubs', e));
  }, []);
  const hubName = id => hubNames[id] || id;

  useEffect(() => {
    // Wait for typing to pause before asking the API, and drop superseded requests.
    const controller = new AbortController();
    const timer = setTimeout(() => fetch('/api/suppliers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        category,
        current_inventory: toCount(inventory),
        safety_stock: toCount(safetyStock),
        demand_forecast: toCount(forecast),
        scenario
      }),
      signal: controller.signal,
    })
      .then(async res => {
        const data = await res.json();
        if (!res.ok) throw new Error(Array.isArray(data.detail) ? data.detail.map(d => d.msg).join('; ') : data.detail || `Request failed (${res.status})`);
        setSuppliers(data.suppliers || []);
        setAdvice(data.advice);
        setDisruptions(data.active_disruptions || {});
        setError(null);
      })
      .catch(e => { if (e.name !== 'AbortError') setError(e.message); }), REQUEST_DELAY_MS);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [category, scenario, inventory, safetyStock, forecast]);

  const critical = advice && advice.urgency_level === 'CRITICAL';

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <h1>Supplier intelligence</h1>
          <p>Suppliers ranked by cost, lead time and reliability under disruption</p>
        </div>
        <nav>
          <button type="button" className="nav-button" onClick={() => onNavigate('recommend')}>
            <ArrowLeft size={15} aria-hidden="true" /> Route planner
          </button>
        </nav>
      </header>

      <main className="page-body">
        <section className="supplier-controls" aria-label="Sourcing inputs">
          <div className="field">
            <label htmlFor="category">Product category</label>
            <select id="category" value={category} onChange={e => setCategory(e.target.value)} className="control">
              <option value="Electronics">Electronics</option>
              <option value="Raw Materials">Raw materials</option>
              <option value="Chemicals">Chemicals</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="inventory">Current inventory (units)</label>
            <input id="inventory" type="number" min="0" value={inventory} onChange={e => setInventory(e.target.value)} className="control" />
          </div>
          <div className="field">
            <label htmlFor="safety">Safety stock target (units)</label>
            <input id="safety" type="number" min="0" value={safetyStock} onChange={e => setSafetyStock(e.target.value)} className="control" />
          </div>
          <div className="field">
            <label htmlFor="forecast">Demand forecast (units)</label>
            <input id="forecast" type="number" min="0" value={forecast} onChange={e => setForecast(e.target.value)} className="control" />
          </div>
          <div className="field">
            <label htmlFor="supplier-scenario">Disruption scenario</label>
            <select id="supplier-scenario" value={scenario || ''} onChange={e => setScenario(e.target.value || null)}
                    className={`control ${scenario ? 'alert' : ''}`}>
              <option value="">Normal operations</option>
              {scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </div>
        </section>

        {error && <div className="error" role="alert">{error}</div>}

        <div className="supplier-results">
          {advice && (
            <section className={`panel advice ${critical ? 'critical' : ''}`}>
              <span className="urgency">{sentence(advice.urgency_level)} urgency</span>
              <h2>{advice.recommendation}</h2>
              <p className="note">
                Status: {sentence(advice.status)}. Projected inventory after demand: {advice.projected_inventory.toLocaleString()} units.
                {advice.shortage_quantity > 0 && ` Short of the safety stock by ${advice.shortage_quantity.toLocaleString()} units.`}
              </p>
              {Object.keys(disruptions).length > 0 && (
                <p className="disrupted">Disrupted hubs: {Object.keys(disruptions).map(hubName).join(', ')}</p>
              )}
            </section>
          )}

          <section className="panel">
            <h2>Qualified suppliers, best first</h2>
            <div className="table-wrap">
              <table className="supplier-table">
                <thead>
                  <tr>
                    <th scope="col">Rank</th>
                    <th scope="col">Supplier</th>
                    <th scope="col">Unit cost</th>
                    <th scope="col">Lead time</th>
                    <th scope="col">Reliability</th>
                    <th scope="col">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {suppliers.length === 0 && (
                    <tr><td colSpan={6} className="muted">No suppliers in this category.</td></tr>
                  )}
                  {suppliers.map(s => {
                    const reliability = s.audit_trace.effective_metrics.stability_index;
                    const penalty = s.audit_trace.penalties.lead_time_impact;
                    return (
                      <tr key={s.id}>
                        <td className="num muted">{s.rank}</td>
                        <td>
                          {s.name}
                          <small>{hubName(s.location_hub)}</small>
                        </td>
                        <td className="num">${s.unit_cost.toLocaleString()}</td>
                        <td className="num">
                          {s.effective_lead_time} days
                          {penalty > 0 && <span className="added"> (+{penalty} from disruption)</span>}
                        </td>
                        <td className={`num ${reliability < 80 ? 'low' : ''}`}>{reliability}%</td>
                        <td>
                          <span className="score">
                            <span className="score-track"><span style={{ width: `${Math.round(s.decision_score * 100)}%` }} /></span>
                            <span className="num">{s.decision_score.toFixed(2)}</span>
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
