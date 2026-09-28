import React, { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ArrowLeft } from 'lucide-react';

// Chart colours from the palette in index.css.
const MUTED = '#8FA3B5';
const GRID = '#22384D';
const AXIS = { fill: MUTED, fontSize: 12 };
const TOOLTIP = {
  contentStyle: { background: '#0B1622', border: `1px solid ${GRID}`, fontSize: 13 },
  cursor: { fill: 'rgba(24, 48, 71, 0.6)' },
};
const QUANTILE_COLOURS = { p50: '#DCE6EE', p85: '#6CC3D5', p95: '#E3A83B' };

export default function ModelEvaluation({ onNavigate }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/model')
      .then(r => r.json())
      .then(data => (data.available ? setReport(data) : setError(data.error || 'The delay model is not loaded.')))
      .catch(() => setError('Could not reach the routing engine. Check that the backend is running.'));
  }, []);

  const quantiles = report ? Object.entries(report.quantiles) : [];
  const coverage = quantiles.map(([name, q]) => ({
    name, target: Number(name.slice(1)), measured: +(q.coverage * 100).toFixed(1),
  }));
  const byMode = report ? Object.keys(quantiles[0][1].coverage_by_mode).map(mode => ({
    mode: mode.charAt(0).toUpperCase() + mode.slice(1),
    ...Object.fromEntries(quantiles.map(([name, q]) => [name, +(q.coverage_by_mode[mode] * 100).toFixed(1)])),
  })) : [];
  const loss = quantiles.map(([name, q]) => ({ name, model: q.pinball_loss, naive: q.naive_pinball_loss }));
  const importance = report ? Object.entries(report.p85_permutation_importance).map(([feature, value]) => ({ feature, value })) : [];

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <h1>Delay model evaluation</h1>
          <p>
            p50, p85 and p95 quantile models, scored on a held-out test set
            {report ? ` of ${report.n_test.toLocaleString()} legs` : ''}
          </p>
        </div>
        <nav>
          <button type="button" className="nav-button" onClick={() => onNavigate('recommend')}>
            <ArrowLeft size={15} aria-hidden="true" /> Route planner
          </button>
        </nav>
      </header>

      <main className="page-body">
        {error && <div className="error" role="alert">{error}</div>}
        {!report && !error && <p className="muted">Loading the evaluation…</p>}
        {report && (
          <div className="model-grid">
            <section className="panel">
              <h2>Calibration: measured coverage against target</h2>
              <p className="note">Share of held-out delays at or below each predicted quantile. A calibrated model measures close to its target.</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={coverage}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="name" tick={AXIS} stroke={GRID} />
                  <YAxis domain={[0, 100]} unit="%" tick={AXIS} stroke={GRID} />
                  <Tooltip {...TOOLTIP} />
                  <Legend />
                  <Bar dataKey="target" fill={MUTED} name="Target" />
                  <Bar dataKey="measured" fill="#6CC3D5" name="Measured" />
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="panel">
              <h2>Coverage by transport mode</h2>
              <p className="note">The same check within each mode, so no mode hides behind the average.</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={byMode}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="mode" tick={AXIS} stroke={GRID} />
                  <YAxis domain={[0, 100]} unit="%" tick={AXIS} stroke={GRID} />
                  <Tooltip {...TOOLTIP} />
                  <Legend />
                  {quantiles.map(([name]) => <Bar key={name} dataKey={name} fill={QUANTILE_COLOURS[name] || MUTED} />)}
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="panel">
              <h2>Pinball loss against a naive baseline</h2>
              <p className="note">The baseline predicts the empirical quantile of each mode and arrival group. Lower is better.</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={loss}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="name" tick={AXIS} stroke={GRID} />
                  <YAxis tick={AXIS} stroke={GRID} />
                  <Tooltip {...TOOLTIP} />
                  <Legend />
                  <Bar dataKey="naive" fill={MUTED} name="Naive baseline" />
                  <Bar dataKey="model" fill="#49B083" name="Quantile model" />
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="panel">
              <h2>What drives the p85 delay</h2>
              <p className="note">Permutation importance: how much the p85 pinball loss rises when a feature is shuffled.</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={importance} layout="vertical" margin={{ left: 30 }}>
                  <CartesianGrid stroke={GRID} horizontal={false} />
                  <XAxis type="number" tick={AXIS} stroke={GRID} />
                  <YAxis type="category" dataKey="feature" tick={AXIS} stroke={GRID} width={90} />
                  <Tooltip {...TOOLTIP} />
                  <Bar dataKey="value" fill="#E3A83B" name="Importance" />
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="panel">
              <h2>About the model</h2>
              <ul className="facts">
                <li>One gradient-boosted quantile regressor per quantile, trained on {report.n_train.toLocaleString()} legs sampled from the live routing graph.</li>
                <li>Monotonic in distance, weather and news severity; p50 ≤ p85 ≤ p95 is enforced (raw crossing rate {(report.quantile_crossing_rate * 100).toFixed(2)}%).</li>
                <li>Pinball loss against the naive baseline: {quantiles.map(([name, q]) => `${name} ${Math.round(q.improvement_vs_naive * 100)}% lower`).join(', ')}.</li>
                <li>The artifact's SHA-256 is pinned in code and checked before loading: <code>{report.sha256.slice(0, 16)}…</code></li>
                <li>Training data is synthetic and physics-informed. docs/MODEL_CARD.md lists the assumptions and limits.</li>
              </ul>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
