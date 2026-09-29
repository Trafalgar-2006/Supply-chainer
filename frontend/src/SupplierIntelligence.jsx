import React, { useState, useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';
import { readJson, requestError } from './api.js';

// Inputs keep whatever the user typed. A count must be a whole number from 0 to
// the API's limit; anything else is flagged on its field, and no advice is
// asked for or shown until it is fixed, so it never rests on a guessed number.
const MAX_UNITS = 1e9;
const countError = value => {
  if (value.trim() === '') return 'Enter a number of units.';
  const n = Number(value);
  if (!Number.isInteger(n)) return 'Use a whole number of units.';
  if (n < 0) return 'Units cannot be negative.';
  if (n > MAX_UNITS) return 'At most 1,000,000,000 units.';
  return null;
};
const units = n => n.toLocaleString('en-US');
const REQUEST_DELAY_MS = 300;
const sentence = s => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase().replace(/_/g, ' ');

export default function SupplierIntelligence({ onNavigate }) {
  const [suppliers, setSuppliers] = useState(null); // null until the first reply, and after a failed one
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

  const invalid = [inventory, safetyStock, forecast].some(countError);

  useEffect(() => {
    // Wait for typing to pause before asking the API, and drop superseded requests.
    // While a count is invalid only the ranking is asked for; it doesn't use the counts.
    // A failed reply clears the table, so it never shows another category's suppliers.
    const fail = message => {
      setError(message);
      setSuppliers(null);
      setAdvice(null);
    };
    const controller = new AbortController();
    const timer = setTimeout(() => fetch('/api/suppliers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        category,
        ...(invalid ? {} : {
          current_inventory: Number(inventory),
          safety_stock: Number(safetyStock),
          demand_forecast: Number(forecast),
        }),
        scenario
      }),
      signal: controller.signal,
    })
      .then(async res => {
        const data = await readJson(res);
        if (!res.ok || !data || !Array.isArray(data.suppliers)) {
          fail(res.ok ? 'The server sent a reply this page could not read.' : requestError(res, data));
          return;
        }
        setSuppliers(data.suppliers);
        setAdvice(invalid ? null : data.advice);
        setDisruptions(data.active_disruptions || {});
        setError(null);
      })
      .catch(e => { if (e.name !== 'AbortError') fail('The server cannot be reached. Check that the backend is running.'); }),
    REQUEST_DELAY_MS);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [category, scenario, inventory, safetyStock, forecast, invalid]);

  const critical = advice && advice.urgency_level === 'CRITICAL';

  const renderCount = (id, label, value, setValue) => {
    const problem = countError(value);
    return (
      <div className="field">
        <label htmlFor={id}>{label}</label>
        <input id={id} type="number" min="0" max={MAX_UNITS} step="1" inputMode="numeric" value={value}
               onChange={e => setValue(e.target.value)} className="control"
               aria-invalid={Boolean(problem)} aria-describedby={problem ? `${id}-error` : undefined} />
        {problem && <p id={`${id}-error`} className="field-error">{problem}</p>}
      </div>
    );
  };

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
          {renderCount('inventory', 'Current inventory (units)', inventory, setInventory)}
          {renderCount('safety', 'Safety stock target (units)', safetyStock, setSafetyStock)}
          {renderCount('forecast', 'Demand forecast (units)', forecast, setForecast)}
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
          {invalid && (
            <section className="panel advice">
              <h2>Fix the highlighted inputs to get procurement advice.</h2>
            </section>
          )}
          {advice && !invalid && (
            <section className={`panel advice ${critical ? 'critical' : ''}`}>
              <span className="urgency">{sentence(advice.urgency_level)} urgency</span>
              <h2>{advice.recommendation}</h2>
              <p className="note">
                Status: {sentence(advice.status)}.{' '}
                {advice.projected_inventory < 0
                  ? `Demand exceeds inventory by ${units(-advice.projected_inventory)} units: a stock-out.`
                  : `Projected inventory after demand: ${units(advice.projected_inventory)} units.`}
                {advice.shortage_quantity > 0 && ` Short of the safety stock by ${units(advice.shortage_quantity)} units.`}
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
                  {!suppliers?.length && (
                    <tr>
                      <td colSpan={6} className="muted">
                        {suppliers ? 'No suppliers in this category.' : error ? 'Could not load the suppliers.' : 'Loading suppliers…'}
                      </td>
                    </tr>
                  )}
                  {(suppliers || []).map(s => {
                    const reliability = s.audit_trace.effective_metrics.stability_index;
                    const penalty = s.audit_trace.penalties.lead_time_impact;
                    return (
                      <tr key={s.id}>
                        <td className="num muted">{s.rank}</td>
                        <td>
                          {s.name}
                          <small>{hubName(s.location_hub)}</small>
                        </td>
                        <td className="num">${s.unit_cost.toLocaleString('en-US')}</td>
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
            <p className="note">
              A disruption at a supplier's hub or on its shipping route adds 10% of the disruption's delay to the lead time
              (a 10-day Suez closure adds a day) and takes up to 30 points off its reliability.
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
