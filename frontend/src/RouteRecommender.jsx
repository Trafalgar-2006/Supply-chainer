import React, { useState, useEffect, useRef, useMemo } from 'react';
import { animate, stagger } from 'animejs';
import { Truck, Ship, Plane, Train, ArrowRightLeft, ArrowUpDown, Navigation, Download } from 'lucide-react';
import RouteMap, { PERSONA_COLOURS, personaLabel, prefersReducedMotion } from './RouteMap.jsx';

const CARGO_TYPES = [
  { value: 'general', label: 'General cargo' },
  { value: 'perishable_urgent', label: 'Perishable, urgent (no sea)' },
  { value: 'hazardous_waste', label: 'Hazardous waste (no air)' },
  { value: 'oversize_heavy', label: 'Oversize or heavy (no road)' },
];
const THREAT_TYPE_LABELS = {
  weather: 'Weather', labour: 'Labour', geopolitical: 'Geopolitical', infrastructure: 'Infrastructure',
  cyber: 'Cyber', congestion: 'Congestion', general: 'General', none: 'No threat',
};
const MODE_ICONS = { AIR: Plane, SEA: Ship, RAIL: Train, ROAD: Truck, TRANSFER: ArrowRightLeft };
// The engine's learned severity, 1 (minor) to 3 (severe).
const severityWord = s => (s >= 2.5 ? 'severe' : s >= 1.5 ? 'significant' : 'minor');
// Engine status pushed over the WebSocket: [state for the indicator, what to show].
const ENGINE_STATES = {
  'FULLY OPERATIONAL': ['ready', 'Engine ready'],
  'WARMING RISK ENGINE': ['warming', 'Warming up the news model'],
  'WARM-UP FAILED': ['offline', 'News model failed to load'],
  'ENGINE OFFLINE': ['offline', 'Engine offline'],
};
const LEGEND = [
  ['Fastest', PERSONA_COLOURS.FASTEST], ['Best balance', PERSONA_COLOURS.BALANCED],
  ['Lowest risk', PERSONA_COLOURS.SAFEST], ['Scenario disruption', 'var(--signal)'], ['Live news', 'var(--orchid)'],
];
// One-click plans for a first visit, each showing a different part of the engine.
const EXAMPLES = [
  { label: 'Shanghai to Rotterdam', source: 'PORT-SHANGHAI', destination: 'PORT-ROTTERDAM', scenario: null, mode: 'any' },
  { label: 'Shanghai to Los Angeles by sea during the port strike', source: 'PORT-SHANGHAI', destination: 'HUB-LOSANGELES',
    scenario: 'LA_PORT_STRIKE', mode: 'sea' },
  { label: 'Chennai to Singapore during the monsoon floods', source: 'PORT-CHENNAI', destination: 'PORT-SINGAPORE',
    scenario: 'CHENNAI_FLOOD', mode: 'any' },
];
const RECENT_KEY = 'supplychainer.recent-plans';
const RECENT_LIMIT = 8;

// Hours, shown as days once a trip passes two days (a 640 h voyage reads 26.7
// days). Figures read together pass `days` so they share one unit.
const fmtH = (h, days = h >= 48) => (days ? `${(h / 24).toFixed(1)} days` : `${Math.round(h * 10) / 10} h`);
const fmtMoney = v => `$${Math.round(v).toLocaleString()}`;
const pct = x => `${Math.round(x * 100)}%`;
const capitalise = s => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();

function modeSummary(route) {
  const modes = route.legs.filter(l => l.type === 'transit').map(l => l.mode.toLowerCase())
    .filter((m, i, all) => m !== all[i - 1]);
  const transfers = route.legs.length - route.legs.filter(l => l.type === 'transit').length;
  return `${capitalise(modes.join(', '))}. ${transfers} transfer${transfers === 1 ? '' : 's'}.`;
}

// Recent plans are kept in this browser only; storage can be full or disabled,
// in which case they are simply not remembered.
const planKey = p => JSON.stringify([p.source, p.destination, p.transportMode, p.routingPolicy,
  p.cargoType, p.priority, p.liveIntel, [...p.avoid].sort()]);

function loadRecent() {
  try {
    const saved = JSON.parse(localStorage.getItem(RECENT_KEY));
    return Array.isArray(saved) ? saved.filter(p => p && typeof p.source === 'string' && typeof p.destination === 'string'
      && typeof p.sourceName === 'string' && typeof p.destName === 'string' && Array.isArray(p.avoid) && Array.isArray(p.hubs)) : [];
  } catch {
    return [];
  }
}

function saveRecent(plans) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(plans));
  } catch {
    // Not remembered; the plans still work for this session.
  }
}

// Every cell quoted. Headlines come from outside, so text that a spreadsheet
// would run as a formula (= + - @) gets a leading apostrophe.
const csvCell = v => {
  const text = typeof v === 'string' && /^[=+\-@\t\r]/.test(v) ? `'${v}` : String(v);
  return `"${text.replace(/"/g, '""')}"`;
};
const slug = s => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function downloadCsv(route, hubName) {
  const origin = hubName(route.legs[0].from);
  const destination = route.legs[route.legs.length - 1].to_name;
  const rows = [
    ['leg', 'type', 'mode', 'from', 'to', 'hours', 'delay_p50_h', 'delay_p85_h', 'delay_p95_h', 'cost_usd', 'threat', 'intel_source', 'reason'],
    ...route.legs.map((l, i) => [i + 1, l.type, l.mode, hubName(l.from), l.to_name, l.eta, l.delay.p50, l.delay.p85, l.delay.p95,
      l.cost, l.threat, l.intel_source, l.reason]),
    ['total', '', '', origin, destination, route.adjusted_eta, '', '', '', route.total_cost, route.threat_level, '',
      `Typical ${route.adjusted_eta} h; plan for ${route.eta_band.p85} h (p85); up to ${route.eta_band.p95} h (p95)`],
  ];
  // The byte-order mark lets Excel read accented hub names as UTF-8.
  const csv = '﻿' + rows.map(r => r.map(csvCell).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `route-${slug(origin)}-to-${slug(destination)}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const RouteRecommender = ({ onNavigate, engineStatus }) => {
  const [source, setSource] = useState('');
  const [destination, setDestination] = useState('');
  const [transportMode, setTransportMode] = useState('any');
  const [routingPolicy, setRoutingPolicy] = useState('STRICT');
  const [operationalConfig, setOperationalConfig] = useState('NORMAL');
  const [cargoType, setCargoType] = useState('general');
  const [priority, setPriority] = useState('normal');
  const [liveIntel, setLiveIntel] = useState(true);
  const [avoid, setAvoid] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [intelReports, setIntelReports] = useState([]);
  // The plan the displayed routes were computed for; the controls may have changed since.
  const [resultContext, setResultContext] = useState(null);
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState({ source: '', dest: '' });
  const [searchResults, setSearchResults] = useState({ source: [], dest: [] });
  const [scenarios, setScenarios] = useState([]);
  const [hubs, setHubs] = useState([]);
  const [recent, setRecent] = useState(loadRecent);
  const [focusLeg, setFocusLeg] = useState(null); // the voyage-plan leg highlighted on the map
  const latestQuery = useRef({ source: '', dest: '' });
  const optionList = useRef(null);

  useEffect(() => {
    // The scenario list comes from the backend so it can never drift from ScenarioManager.
    fetch('/api/scenarios')
      .then(r => r.json())
      .then(setScenarios)
      .catch(e => console.error('Failed to load scenarios', e));
    fetch('/api/hubs')
      .then(r => r.json())
      .then(setHubs)
      .catch(e => console.error('Failed to load hubs', e));
  }, []);

  // New results: the option rows come in one after another, alongside the
  // route being drawn on the chart.
  useEffect(() => {
    if (!optionList.current || !recommendations.length || prefersReducedMotion()) return undefined;
    const entrance = animate(optionList.current.children, {
      opacity: [0, 1], translateX: [-12, 0], duration: 450, delay: stagger(80), ease: 'outQuad',
    });
    return () => entrance.revert();
  }, [recommendations]);

  const hubNames = useMemo(() => Object.fromEntries(hubs.map(h => [h.id, h.display_name])), [hubs]);
  const hubName = id => hubNames[id] || id;
  const chokepoints = useMemo(
    () => hubs.filter(h => h.type === 'choke_point').sort((a, b) => a.display_name.localeCompare(b.display_name)),
    [hubs]
  );
  const selectedScenario = operationalConfig !== 'NORMAL' ? operationalConfig : null;
  const scenarioInfo = scenarios.find(s => s.id === selectedScenario);
  const resultScenario = scenarios.find(s => s.id === resultContext?.scenario);
  const scenarioChanged = resultContext && resultContext.scenario !== selectedScenario;
  const disruptedHubs = useMemo(() => (resultScenario ? resultScenario.affected_nodes : []), [resultScenario]);
  const liveHubs = useMemo(() => intelReports.filter(r => r.score > 0).flatMap(r => r.hubs), [intelReports]);
  const route = recommendations[selected];
  const scaleMax = Math.max(1, ...recommendations.map(r => r.eta_band.p95));
  const ledgerDays = route ? route.eta_band.p95 >= 48 : false; // one unit for the selected route's time ledger

  // Recent plans whose routes pass through a hub the chosen scenario disrupts,
  // and that have not been planned under it yet.
  const affectedPlans = useMemo(() => {
    if (!scenarioInfo) return [];
    const hit = new Set(scenarioInfo.affected_nodes);
    const done = new Set(recent.filter(p => p.scenario === scenarioInfo.id).map(planKey));
    return recent.filter(p => {
      const key = planKey(p);
      if (done.has(key) || !p.hubs.some(h => hit.has(h))) return false;
      done.add(key);
      return true;
    });
  }, [recent, scenarioInfo]);

  const applyPlan = plan => {
    setSource(plan.source);
    setDestination(plan.destination);
    setSearchQuery({ source: plan.sourceName, dest: plan.destName });
    latestQuery.current = { source: plan.sourceName, dest: plan.destName };
    setSearchResults({ source: [], dest: [] });
    setTransportMode(plan.transportMode);
    setRoutingPolicy(plan.routingPolicy);
    setCargoType(plan.cargoType);
    setPriority(plan.priority);
    setOperationalConfig(plan.scenario || 'NORMAL');
    setLiveIntel(plan.liveIntel);
    setAvoid(plan.avoid);
  };

  const remember = (plan, routes) => {
    const entry = { ...plan, hubs: [...new Set(routes.flatMap(r => r.legs.flatMap(l => [l.from, l.to])))] };
    const next = [entry, ...recent.filter(p => planKey(p) !== planKey(entry) || p.scenario !== entry.scenario)]
      .slice(0, RECENT_LIMIT);
    setRecent(next);
    saveRecent(next);
  };

  const runPlan = async plan => {
    applyPlan(plan);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: plan.source,
          destination: plan.destination,
          transport_preference: plan.transportMode,
          routing_policy: plan.routingPolicy,
          cargo_type: plan.cargoType,
          priority: plan.priority,
          scenario: plan.scenario,
          live_intel: plan.liveIntel,
          overrides: plan.avoid.length ? { avoid_chokepoints: plan.avoid } : null
        })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        const detail = Array.isArray(data.detail) ? data.detail.map(d => d.msg).join('; ') : data.detail;
        setError(data.error || detail || `Request failed (${res.status})`);
        setRecommendations([]);
        setIntelReports([]);
        setResultContext(null);
      } else {
        setRecommendations(data.recommendations);
        setIntelReports(data.live_intel || []);
        setResultContext(plan);
        setSelected(0);
        remember(plan, data.recommendations);
      }
    } catch {
      setError('Could not reach the routing engine. Check that the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const planFromControls = () => ({
    source, sourceName: searchQuery.source, destination, destName: searchQuery.dest,
    transportMode, routingPolicy, cargoType, priority, scenario: selectedScenario, liveIntel, avoid,
  });

  const handleSearch = async (type, query) => {
    setSearchQuery(prev => ({ ...prev, [type]: query }));
    // Editing the text un-selects the hub, so a stale choice is never submitted.
    if (type === 'source') setSource(''); else setDestination('');
    latestQuery.current[type] = query;
    if (query.length < 2) {
      setSearchResults(prev => ({ ...prev, [type]: [] }));
      return;
    }
    try {
      const res = await fetch(`/api/hubs/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      // Responses can arrive out of order while typing; keep only the latest.
      if (latestQuery.current[type] === query) {
        setSearchResults(prev => ({ ...prev, [type]: data }));
      }
    } catch {
      console.error('Hub search failed');
    }
  };

  const selectHub = (type, hub) => {
    if (type === 'source') setSource(hub.id); else setDestination(hub.id);
    setSearchQuery(prev => ({ ...prev, [type]: hub.display_name }));
    latestQuery.current[type] = hub.display_name;
    setSearchResults(prev => ({ ...prev, [type]: [] }));
  };

  const toggleAvoid = id => setAvoid(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]));

  const swapEnds = () => {
    setSource(destination);
    setDestination(source);
    setSearchQuery({ source: searchQuery.dest, dest: searchQuery.source });
    latestQuery.current = { source: searchQuery.dest, dest: searchQuery.source };
    setSearchResults({ source: [], dest: [] });
  };

  const runExample = example => runPlan({
    source: example.source, sourceName: hubName(example.source),
    destination: example.destination, destName: hubName(example.destination),
    transportMode: example.mode, routingPolicy: 'STRICT', cargoType: 'general', priority: 'normal',
    scenario: example.scenario, liveIntel, avoid: [],
  });

  // Pointing at a leg (mouse or keyboard) highlights its stretch on the map.
  const pointAt = leg => ({
    tabIndex: 0,
    onMouseEnter: () => setFocusLeg(leg), onMouseLeave: () => setFocusLeg(null),
    onFocus: () => setFocusLeg(leg), onBlur: () => setFocusLeg(null),
  });

  const [engineState, engineText] = ENGINE_STATES[engineStatus] || ['warming', 'Connecting to the engine'];

  const renderHubSearch = (type, label, placeholder) => (
    <div className="field">
      <label htmlFor={`hub-${type}`}>{label}</label>
      <input
        id={`hub-${type}`} type="text" value={searchQuery[type]} autoComplete="off"
        onChange={e => handleSearch(type, e.target.value)}
        className="control" placeholder={placeholder}
      />
      {searchResults[type].length > 0 && (
        <div className="suggestions">
          {searchResults[type].map((h, idx) => (
            <button key={`${h.id}-${idx}`} type="button" onClick={() => selectHub(type, h)}>{h.display_name}</button>
          ))}
        </div>
      )}
    </div>
  );

  const renderLeg = (leg, idx) => {
    const exposure = leg.intel_source === 'SCENARIO' ? 'scenario' : leg.intel_source === 'LIVE' ? 'live' : '';
    const Icon = MODE_ICONS[leg.mode] || Navigation;
    const note = exposure && (
      <div className="exposure">
        {exposure === 'live' ? 'Live news' : 'Scenario'}, {pct(leg.threat)} threat: {leg.reason}
      </div>
    );
    if (leg.type === 'transfer') {
      return (
        <li key={idx} className={`handoff ${exposure}`} {...pointAt(leg)}>
          Transfer at {leg.to_name}, {fmtH(leg.eta)}
          {note}
        </li>
      );
    }
    return (
      <li key={idx} className={exposure} {...pointAt(leg)}>
        {leg.to_name}
        <div className="how"><Icon size={13} aria-hidden="true" /> {capitalise(leg.mode)}, {fmtH(leg.eta)}, {fmtMoney(leg.cost)}</div>
        {note}
      </li>
    );
  };

  const renderDrivers = drivers => {
    const rows = drivers ? drivers.drivers.filter(d => Math.abs(d.hours) >= 0.5).slice(0, 5) : [];
    if (!rows.length) return null;
    const peak = Math.max(...rows.map(d => Math.abs(d.hours)), 1);
    return (
      <section>
        <h2>Why the delay</h2>
        <p className="note">What each factor adds to the 85% planning delay, compared with a short road hop (Shapley values).</p>
        <div className="bars">
          {rows.map(d => (
            <div key={d.factor} className="bar" title={`${d.factor}: ${d.hours > 0 ? '+' : ''}${d.hours} h`}>
              <span>{d.factor}</span>
              <span className="track">
                <span className={`fill ${d.hours < 0 ? 'saves' : ''}`}
                      style={{ width: `${Math.max(4, (Math.abs(d.hours) / peak) * 100)}%` }} />
              </span>
              <span className="value num">{d.hours > 0 ? '+' : ''}{Math.round(d.hours)} h</span>
            </div>
          ))}
        </div>
      </section>
    );
  };

  return (
    <div className="planner">
      <header className="topbar">
        <div className="brand">
          <h1>Supplychainer</h1>
          <p>Multimodal freight routes that account for disruption</p>
        </div>
        <nav>
          <span className="engine" data-state={engineState} role="status">{engineText}</span>
          <button type="button" className="nav-button" onClick={() => onNavigate('model')}>Model evaluation</button>
          <button type="button" className="nav-button" onClick={() => onNavigate('suppliers')}>Supplier intelligence</button>
        </nav>
      </header>

      <aside className="plan" aria-label="Plan a shipment">
        {renderHubSearch('source', 'Origin', 'City, port or airport')}
        <button type="button" className="link swap" onClick={swapEnds}
                disabled={!searchQuery.source && !searchQuery.dest}>
          <ArrowUpDown size={14} aria-hidden="true" /> Swap origin and destination
        </button>
        {renderHubSearch('dest', 'Destination', 'City, port or airport')}

        <div className="field">
          <label htmlFor="mode">Transport mode</label>
          <select id="mode" value={transportMode} onChange={e => setTransportMode(e.target.value)} className="control">
            <option value="any">Any mode</option>
            <option value="sea">Sea</option>
            <option value="air">Air</option>
            <option value="rail">Rail</option>
            <option value="road">Road</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="policy">Mode policy</label>
          <select id="policy" value={routingPolicy} onChange={e => setRoutingPolicy(e.target.value)} className="control"
                  disabled={transportMode === 'any'}>
            <option value="STRICT" title="Road is still allowed for the first and last mile">Only this mode</option>
            <option value="PREFERRED">Prefer this mode</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="cargo">Cargo</label>
          <select id="cargo" value={cargoType} onChange={e => setCargoType(e.target.value)} className="control">
            {CARGO_TYPES.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
          </select>
        </div>

        <div className="field">
          <label htmlFor="priority">Priority</label>
          <select id="priority" value={priority} onChange={e => setPriority(e.target.value)} className="control">
            <option value="low">Low: cost first</option>
            <option value="normal">Normal</option>
            <option value="urgent">Urgent: time first</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="scenario">Disruption scenario</label>
          <select id="scenario" value={operationalConfig} onChange={e => setOperationalConfig(e.target.value)}
                  className={`control ${selectedScenario ? 'alert' : ''}`}>
            <option value="NORMAL">Normal operations</option>
            {scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </div>

        <div className="field">
          <span className="label" id="avoid-label">Avoid chokepoints</span>
          <div className="chips" role="group" aria-labelledby="avoid-label">
            {chokepoints.map(c => (
              <button key={c.id} type="button" onClick={() => toggleAvoid(c.id)}
                      className="chip" aria-pressed={avoid.includes(c.id)}>
                {c.display_name.replace(/ (Strait|Canal)$/, '')}
              </button>
            ))}
          </div>
        </div>

        <label className="check">
          <input type="checkbox" checked={liveIntel} onChange={e => setLiveIntel(e.target.checked)} />
          Check live news and weather
        </label>

        <button type="button" className="primary" onClick={() => runPlan(planFromControls())}
                disabled={loading || !source || !destination}>
          {loading ? 'Planning routes…' : 'Plan routes'}
        </button>

        {recent.length > 0 && (
          <>
            <h2>Recent plans</h2>
            <ul className="recent">
              {recent.map(p => (
                <li key={planKey(p) + p.scenario}>
                  <button type="button" disabled={loading} onClick={() => runPlan(p)}>
                    {p.sourceName} to {p.destName}
                    <small>
                      {scenarios.find(s => s.id === p.scenario)?.name || (p.scenario ? p.scenario : 'Normal operations')}
                      {p.transportMode !== 'any' ? `, ${p.transportMode}` : ''}
                    </small>
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </aside>

      <main className="board">
        {resultScenario && (
          <div className="notice">
            <strong>Planned under {resultScenario.name}.</strong> {resultScenario.reason}
          </div>
        )}
        {scenarioChanged && !affectedPlans.some(p => planKey(p) === planKey(resultContext)) && (
          <div className="notice warn">
            <strong>Scenario changed.</strong> Plan again to apply it to the routes shown.
          </div>
        )}
        {affectedPlans.length > 0 && (
          <div className="notice warn">
            <strong>{scenarioInfo.name} disrupts {affectedPlans.length} recent plan{affectedPlans.length === 1 ? '' : 's'}.</strong> Plan again under it:
            <ul>
              {affectedPlans.map(p => (
                <li key={planKey(p)}>
                  <button type="button" className="link" disabled={loading}
                          onClick={() => runPlan({ ...p, scenario: scenarioInfo.id })}>
                    {p.sourceName} to {p.destName}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        {error && <div className="error" role="alert">{error}</div>}

        <RouteMap hubs={hubs} routes={recommendations} selected={selected} onSelect={setSelected}
                  disrupted={disruptedHubs} liveHubs={liveHubs} focusLeg={focusLeg} />

        {recommendations.length > 0 ? (
          <>
            <div className="legend" aria-label="Map legend">
              {LEGEND.map(([label, colour]) => <span key={label}><i style={{ background: colour }} />{label}</span>)}
            </div>
            <section>
              <h2>Route options</h2>
              <p className="note">
                Bars run from the typical door-to-door time to the worst case (p95), on one scale for all options.
                The coloured mark is the time to plan for: 85% of simulated trips arrive by then.
              </p>
            </section>
            <div className="options-head options-grid" aria-hidden="true">
              <span /><span>Route</span><span>Door-to-door time</span><span>Landed cost</span><span>Peak risk</span>
            </div>
            <ul className={`options ${loading ? 'busy' : ''}`} ref={optionList} aria-busy={loading}>
              {recommendations.map((rec, idx) => {
                const colour = PERSONA_COLOURS[rec.persona];
                const band = rec.eta_band;
                const at = h => `${(h / scaleMax) * 100}%`;
                const low = Math.min(rec.adjusted_eta, band.p50);
                const days = band.p95 >= 48;
                return (
                  <li key={idx}>
                    <button type="button" className="option options-grid" aria-pressed={idx === selected}
                            onClick={() => setSelected(idx)}>
                      <span className="swatch" style={{ background: colour }} />
                      <span className="who">
                        <span className="name">{personaLabel(rec)}</span>
                        <span className="modes">{modeSummary(rec)}</span>
                      </span>
                      <span className="gauge"
                            title={`Typical ${Math.round(rec.adjusted_eta)} h. Simulated p50 ${Math.round(band.p50)} h, p85 ${Math.round(band.p85)} h, p95 ${Math.round(band.p95)} h.`}>
                        <span className="reading">
                          <span><strong>{fmtH(rec.adjusted_eta, days)}</strong> typical</span>
                          <span>up to {fmtH(band.p95, days)}</span>
                        </span>
                        <span className="scale">
                          <span className="spread" style={{ left: at(low), width: `calc(${at(band.p95)} - ${at(low)})`, background: colour }} />
                          <span className="tick typical" style={{ left: at(rec.adjusted_eta) }} />
                          <span className="tick" style={{ left: at(band.p85), background: colour }} />
                        </span>
                      </span>
                      <span className="cost num">{fmtMoney(rec.total_cost)}</span>
                      <span className={`risk num ${rec.threat_level >= 0.5 ? 'high' : ''}`}>{pct(rec.threat_level)}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </>
        ) : (
          !loading && !error && (
            <div className="empty">
              <p>
                Choose an origin and a destination, then plan routes. You get up to three options: the fastest,
                the lowest-risk, and the best balance of cost, time and risk.
              </p>
              {hubs.length > 0 && (
                <>
                  <p>Or try one:</p>
                  <ul className="examples">
                    {EXAMPLES.map(example => (
                      <li key={example.label}>
                        <button type="button" className="link" onClick={() => runExample(example)}>{example.label}</button>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )
        )}
      </main>

      <aside className="voyage" aria-label="Selected route">
        {route ? (
          <>
            <section>
              <div className="heading">
                <h2>{personaLabel(route)}</h2>
                <button type="button" className="secondary" onClick={() => downloadCsv(route, hubName)}>
                  <Download size={14} aria-hidden="true" /> Export CSV
                </button>
              </div>
              <p className="lead">{route.explanation}</p>
            </section>

            <section>
              <h2>Legs</h2>
              <ol className="legs">
                <li className="handoff">Depart {hubName(route.legs[0].from)}</li>
                {route.legs.map(renderLeg)}
              </ol>
            </section>

            {renderDrivers(route.delay_drivers)}

            <section>
              <h2>Time</h2>
              <table className="ledger">
                <tbody>
                  <tr><td>Transit</td><td>{fmtH(route.audit_trace.eta.transit, ledgerDays)}</td></tr>
                  <tr><td>Transfers</td><td>{fmtH(route.audit_trace.eta.transfer, ledgerDays)}</td></tr>
                  <tr><td>Typical delay (model p50)</td><td>{fmtH(route.audit_trace.eta.delay, ledgerDays)}</td></tr>
                  <tr><td>Scenario delay</td><td>{route.audit_trace.eta.scenario > 0 ? fmtH(route.audit_trace.eta.scenario, ledgerDays) : 'None'}</td></tr>
                  <tr className="total"><td>Typical door to door</td><td>{fmtH(route.adjusted_eta, ledgerDays)}</td></tr>
                  <tr className="gap"><td>Plan for (p85)</td><td>{fmtH(route.eta_band.p85, ledgerDays)}</td></tr>
                  <tr><td>Worst case (p95)</td><td>{fmtH(route.eta_band.p95, ledgerDays)}</td></tr>
                </tbody>
              </table>
            </section>

            <section>
              <h2>Cost</h2>
              <table className="ledger">
                <tbody>
                  <tr><td>Transit</td><td>{fmtMoney(route.audit_trace.cost.transit)}</td></tr>
                  <tr><td>Transfer fees</td><td>{fmtMoney(route.audit_trace.cost.transfer)}</td></tr>
                  <tr><td>Scenario risk premium</td><td>{fmtMoney(route.audit_trace.cost.scenario)}</td></tr>
                  <tr className="total"><td>Landed cost</td><td>{fmtMoney(route.total_cost)}</td></tr>
                </tbody>
              </table>
            </section>

            <section>
              <h2>Risk</h2>
              <table className="ledger">
                <tbody>
                  <tr><td>Standing conditions</td><td>{pct(route.audit_trace.risk.baseline)}</td></tr>
                  <tr><td>Scenario</td><td>{pct(route.audit_trace.risk.scenario)}</td></tr>
                  <tr><td>Live news</td><td>{pct(route.audit_trace.risk.live)}</td></tr>
                  <tr className="total"><td>Peak on the route</td><td>{pct(route.threat_level)}</td></tr>
                </tbody>
              </table>
            </section>
          </>
        ) : (
          <p className="empty">The selected route's legs, delay drivers and cost breakdown appear here.</p>
        )}

        <section>
          <h2>Live news</h2>
          {!resultContext && (
            <p className="note">Reports for the origin, the destination and the chokepoints on your routes appear here after you plan.</p>
          )}
          {resultContext && !resultContext.liveIntel && <p className="note">These routes were planned with live news off.</p>}
          {resultContext?.liveIntel && intelReports.length === 0 && (
            <p className="note">No live reports: the feeds are quiet or offline, or the news model is still warming up.</p>
          )}
          {intelReports.some(r => r.weather) && <p className="note">Weather data by Open-Meteo.com (CC BY 4.0).</p>}
          {intelReports.map(r => (
            <div key={r.place} className={`report ${r.score > 0 ? 'hot' : ''}`}>
              <h3>
                {r.place}
                <span>
                  {THREAT_TYPE_LABELS[r.threat_type] || r.threat_type}
                  {r.score > 0 ? `, ${severityWord(r.severity)} (${pct(r.score)})` : ''}
                </span>
              </h3>
              <p>{r.headline || r.headlines || 'No disruption news right now.'}</p>
              {r.weather && (
                <p className="muted">Weather now: {r.weather.description}, wind {r.weather.wind_kmh} km/h.</p>
              )}
              {r.condition !== 'clear' && <p className="muted">The delay model treats the weather here as {r.condition}.</p>}
            </div>
          ))}
        </section>
      </aside>
    </div>
  );
};

export default RouteRecommender;
