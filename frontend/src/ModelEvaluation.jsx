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
const pct = x => `${Math.round(x * 100)}%`;
// Threat-intelligence scores on the held-out headlines: [label, key, format, higher is better].
const NLP_METRICS = [
  ['Disruptions detected', 'recall', pct, true],
  ['False alarms on routine news', 'false_alarm_rate', pct, false],
  ['Precision', 'precision', pct, true],
  ['Detection AUC', 'auc', x => x.toFixed(2), true],
  ['Threat type correct', 'type_accuracy', pct, true],
  ['CARF: right call per mode', 'carf_accuracy', pct, true],
];

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
  // Naive baseline, this model and the best possible model, all on the same fresh legs.
  const ceiling = report?.delay_ceiling
    ? Object.entries(report.delay_ceiling.quantiles).map(([name, q]) => ({ name, ...q }))
    : [];
  const nlp = report?.nlp;
  const holdout = nlp?.holdout;
  const nlpNow = holdout && { ...holdout.detection, type_accuracy: holdout.type_accuracy, carf_accuracy: holdout.carf_accuracy };
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

            {ceiling.length > 0 && (
              <section className="panel">
                <h2>Pinball loss: naive baseline, this model, best possible</h2>
                <p className="note">
                  Lower is better. The best possible model is computed exactly from the known data generator; this model
                  gets {ceiling.map(q => `${pct(q.share_of_achievable_gain)} (${q.name})`).join(', ')} of the way
                  there, on {report.delay_ceiling.legs.toLocaleString()} fresh legs.
                </p>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={ceiling}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="name" tick={AXIS} stroke={GRID} />
                    <YAxis tick={AXIS} stroke={GRID} />
                    <Tooltip {...TOOLTIP} />
                    <Legend />
                    <Bar dataKey="naive" fill={MUTED} name="Naive baseline" />
                    <Bar dataKey="model" fill="#49B083" name="This model" />
                    <Bar dataKey="optimal" fill="#DCE6EE" name="Best possible" />
                  </BarChart>
                </ResponsiveContainer>
              </section>
            )}

            {holdout && (
              <section className="panel">
                <h2>Threat intelligence on held-out headlines</h2>
                <p className="note">
                  {holdout.headlines} labelled headlines written after all tuning and scored once: {holdout.disrupted} disruptions
                  and {holdout.safe} pieces of routine news, including disruptions that have ended.
                </p>
                <table className="scores">
                  <thead>
                    <tr><th scope="col">Measure</th><th scope="col">Before</th><th scope="col">Now</th></tr>
                  </thead>
                  <tbody>
                    {NLP_METRICS.map(([label, key, format, higherIsBetter]) => {
                      const before = nlp.baseline_holdout?.[key];
                      const now = nlpNow[key];
                      const better = before !== undefined && (higherIsBetter ? now > before : now < before);
                      return (
                        <tr key={key}>
                          <td>{label}</td>
                          <td className="num muted">{before === undefined ? '' : format(before)}</td>
                          <td className={`num ${better ? 'better' : ''}`}>{format(now)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="note">
                  Still missed: {holdout.errors.missed.map(h => `"${h}"`).join('; ') || 'none'}.
                  {' '}False alarms: {holdout.errors.false_alarms.map(h => `"${h}"`).join('; ') || 'none'}.
                </p>
              </section>
            )}

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
                <li>Pinball loss against the naive baseline on the held-out split: {quantiles.map(([name, q]) => `${name} ${Math.round(q.improvement_vs_naive * 100)}% lower`).join(', ')}.</li>
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
