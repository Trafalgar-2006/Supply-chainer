import React, { useState, useEffect } from 'react';
import { Activity, ArrowLeft, Database, ShieldAlert } from 'lucide-react';

// Inputs keep whatever the user typed (possibly empty mid-edit); requests use a
// clean count, never NaN or a negative number.
const toCount = value => Math.max(0, Number.parseInt(value, 10) || 0);
const REQUEST_DELAY_MS = 300;

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
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/scenarios')
      .then(r => r.json())
      .then(setScenarios)
      .catch(e => console.error(e));
  }, []);

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
        if (!res.ok) throw new Error(Array.isArray(data.detail) ? data.detail.map(d => d.msg).join('; ') : `Request failed (${res.status})`);
        setSuppliers(data.suppliers || []);
        setAdvice(data.advice);
        setDisruptions(data.active_disruptions || {});
        setError(null);
      })
      .catch(e => { if (e.name !== 'AbortError') setError(e.message); }), REQUEST_DELAY_MS);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [category, scenario, inventory, safetyStock, forecast]);

  const critical = advice && advice.urgency_level === 'CRITICAL';
  const accent = critical ? '#ef4444' : '#8b5cf6';

  return (
    <div className="supplier-layout">
      <header className="dashboard-header" style={{ gridColumn: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <Database size={28} color="#8b5cf6" />
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 800 }}>Supplier Intelligence</h1>
            <p style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 700 }}>DISRUPTION-AWARE SOURCING DECISION MATRIX</p>
          </div>
        </div>
        <button className="sc-badge-active" style={{ cursor: 'pointer', borderColor: '#8b5cf6', color: '#8b5cf6' }} onClick={() => onNavigate('recommend')}>
          <ArrowLeft size={14} /> ROUTE RECOMMENDER
        </button>
      </header>

      <main className="supplier-body">
        <section className="supplier-controls">
          <div className="sc-input-group">
            <label className="sc-label">Product Category</label>
            <select value={category} onChange={e => setCategory(e.target.value)} className="sc-select">
              <option value="Electronics">Electronics</option>
              <option value="Raw Materials">Raw Materials</option>
              <option value="Chemicals">Chemicals</option>
            </select>
          </div>
          <div className="sc-input-group">
            <label className="sc-label">Current Inventory (units)</label>
            <input type="number" min="0" value={inventory} onChange={e => setInventory(e.target.value)} className="sc-input" />
          </div>
          <div className="sc-input-group">
            <label className="sc-label">Safety Stock Target (units)</label>
            <input type="number" min="0" value={safetyStock} onChange={e => setSafetyStock(e.target.value)} className="sc-input" />
          </div>
          <div className="sc-input-group">
            <label className="sc-label">Demand Forecast (units)</label>
            <input type="number" min="0" value={forecast} onChange={e => setForecast(e.target.value)} className="sc-input" />
          </div>
          <div className="sc-input-group">
            <label className="sc-label">Disruption Scenario</label>
            <select value={scenario || ''} onChange={e => setScenario(e.target.value || null)} className="sc-select">
              <option value="">Operational Normal</option>
              {scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </div>
        </section>

        {error && <div className="error-box">{error}</div>}

        <section className="supplier-results">
          {advice && (
            <div className="path-card" style={{ borderColor: accent, cursor: 'default' }}>
              <div className="card-header" style={{ background: critical ? 'rgba(239, 68, 68, 0.1)' : 'rgba(139, 92, 246, 0.1)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: accent }}>
                  <ShieldAlert size={18} />
                  <span style={{ fontWeight: 800, fontSize: '0.75rem' }}>{advice.urgency_level} URGENCY</span>
                </div>
              </div>
              <div style={{ padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: '0.75rem' }}>{advice.recommendation}</h3>
                <p style={{ fontSize: '0.85rem', color: '#94a3b8', lineHeight: 1.6 }}>
                  Status: {advice.status.replace(/_/g, ' ')} · Projected inventory after demand: {advice.projected_inventory} units
                  {advice.shortage_quantity > 0 && ` · Shortfall vs. safety stock: ${advice.shortage_quantity} units`}
                </p>
                {Object.keys(disruptions).length > 0 && (
                  <p style={{ fontSize: '0.8rem', color: '#ef4444', marginTop: '0.75rem' }}>
                    Disrupted: {Object.keys(disruptions).join(', ')}
                  </p>
                )}
              </div>
            </div>
          )}

          <div className="path-card" style={{ cursor: 'default' }}>
            <div className="card-header">
              <h3 style={{ fontSize: '0.9rem', fontWeight: 700 }}>Qualified Suppliers, ranked</h3>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="supplier-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Supplier</th>
                    <th>Unit cost</th>
                    <th>Effective lead time</th>
                    <th>Reliability (disruption-adjusted)</th>
                    <th>Decision score</th>
                  </tr>
                </thead>
                <tbody>
                  {suppliers.length === 0 && (
                    <tr><td colSpan={6} style={{ color: '#64748b' }}>No suppliers in this category.</td></tr>
                  )}
                  {suppliers.map(s => {
                    const reliability = s.audit_trace.effective_metrics.stability_index;
                    const penalty = s.audit_trace.penalties.lead_time_impact;
                    return (
                      <tr key={s.id}>
                        <td style={{ color: '#64748b' }}>{s.rank}</td>
                        <td>
                          <div style={{ fontWeight: 700 }}>{s.name}</div>
                          <div style={{ fontSize: '0.7rem', color: '#64748b' }}>{s.location_hub}</div>
                        </td>
                        <td className="mono">${s.unit_cost.toLocaleString()}</td>
                        <td className="mono">
                          {s.effective_lead_time} days
                          {penalty > 0 && <span style={{ color: '#ef4444' }}> (+{penalty})</span>}
                        </td>
                        <td>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: reliability < 80 ? '#ef4444' : '#10b981' }}>
                            <Activity size={14} /> {reliability}%
                          </span>
                        </td>
                        <td>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className="score-track"><span style={{ width: `${Math.round(s.decision_score * 100)}%` }} /></span>
                            <span className="mono">{s.decision_score.toFixed(2)}</span>
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
