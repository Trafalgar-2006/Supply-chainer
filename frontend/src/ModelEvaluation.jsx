import React, { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Activity, ArrowLeft, Brain } from 'lucide-react';

const AXIS = { stroke: '#64748b', fontSize: 11 };
const TOOLTIP = { contentStyle: { background: '#0f172a', border: '1px solid #1e293b', fontSize: 12 } };

export default function ModelEvaluation({ onNavigate }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/model')
      .then(r => r.json())
      .then(data => (data.available ? setReport(data) : setError(data.error || 'Delay model unavailable')))
      .catch(() => setError('Engine connection failed. Verify backend status.'));
  }, []);

  const quantiles = report ? Object.entries(report.quantiles) : [];
  const coverage = quantiles.map(([name, q]) => ({
    name, target: Number(name.slice(1)), measured: +(q.coverage * 100).toFixed(1),
  }));
  const byMode = report ? Object.keys(quantiles[0][1].coverage_by_mode).map(mode => ({
    mode: mode.toUpperCase(),
    ...Object.fromEntries(quantiles.map(([name, q]) => [name, +(q.coverage_by_mode[mode] * 100).toFixed(1)])),
  })) : [];
  const loss = quantiles.map(([name, q]) => ({ name, model: q.pinball_loss, naive: q.naive_pinball_loss }));
  const importance = report ? Object.entries(report.p85_permutation_importance).map(([feature, value]) => ({ feature, value })) : [];

  return (
    <div className="model-layout">
      <header className="dashboard-header" style={{ gridColumn: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <Brain size={28} color="#8b5cf6" />
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 800 }}>Delay Model Evaluation</h1>
            <p style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 700 }}>
              P50 / P85 / P95 QUANTILE MODELS · HELD-OUT TEST SET{report ? ` · ${report.n_test.toLocaleString()} LEGS` : ''}
            </p>
          </div>
        </div>
        <button className="sc-badge-active" onClick={() => onNavigate('recommend')} style={{ cursor: 'pointer' }}>
          <ArrowLeft size={14} /> ROUTE RECOMMENDER
        </button>
      </header>

      <main className="model-grid">
        {error && <div className="error-box">{error}</div>}
        {!report && !error && <p style={{ color: '#64748b' }}>Loading evaluation…</p>}
        {report && (
          <>
            <section className="model-card">
              <h2 className="panel-title"><Activity size={14} /> Calibration: coverage vs target</h2>
              <p className="model-note">Share of held-out delays at or below each predicted quantile. Well calibrated means measured ≈ target.</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={coverage}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="name" tick={AXIS} />
                  <YAxis domain={[0, 100]} unit="%" tick={AXIS} />
                  <Tooltip {...TOOLTIP} />
                  <Legend />
                  <Bar dataKey="target" fill="#334155" name="Target" />
                  <Bar dataKey="measured" fill="#8b5cf6" name="Measured" />
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="model-card">
              <h2 className="panel-title">Coverage by transport mode</h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={byMode}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="mode" tick={AXIS} />
                  <YAxis domain={[0, 100]} unit="%" tick={AXIS} />
                  <Tooltip {...TOOLTIP} />
                  <Legend />
                  <Bar dataKey="p50" fill="#3b82f6" />
                  <Bar dataKey="p85" fill="#8b5cf6" />
                  <Bar dataKey="p95" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="model-card">
              <h2 className="panel-title">Pinball loss vs naive baseline</h2>
              <p className="model-note">Naive baseline: the empirical quantile of each (mode, arrival) group. Lower is better.</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={loss}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="name" tick={AXIS} />
                  <YAxis tick={AXIS} />
                  <Tooltip {...TOOLTIP} />
                  <Legend />
                  <Bar dataKey="naive" fill="#334155" name="Naive baseline" />
                  <Bar dataKey="model" fill="#10b981" name="Quantile model" />
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="model-card">
              <h2 className="panel-title">What drives the p85 delay</h2>
              <p className="model-note">Permutation importance: rise in p85 pinball loss when a feature is shuffled.</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={importance} layout="vertical" margin={{ left: 30 }}>
                  <CartesianGrid stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" tick={AXIS} />
                  <YAxis type="category" dataKey="feature" tick={AXIS} width={90} />
                  <Tooltip {...TOOLTIP} />
                  <Bar dataKey="value" fill="#f59e0b" name="Importance" />
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="model-card model-facts">
              <h2 className="panel-title">Model facts</h2>
              <ul>
                <li>One gradient-boosted quantile regressor per quantile, trained on {report.n_train.toLocaleString()} legs sampled from the live routing graph.</li>
                <li>Monotonic in distance, weather and news severity; p50 ≤ p85 ≤ p95 enforced (raw crossing rate {(report.quantile_crossing_rate * 100).toFixed(2)}%).</li>
                <li>Pinball loss vs the naive baseline: {quantiles.map(([name, q]) => `${name} ${Math.round(q.improvement_vs_naive * 100)}% better`).join(', ')}.</li>
                <li>Artifact SHA-256 pinned in code and verified before loading: <code>{report.sha256.slice(0, 16)}…</code></li>
                <li>Training data is synthetic and physics-informed; see docs/MODEL_CARD.md for assumptions and limitations.</li>
              </ul>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
