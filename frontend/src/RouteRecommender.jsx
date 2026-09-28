import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Truck, Ship, Plane, Train,
  AlertTriangle, ShieldCheck, Clock, Navigation, Zap, Globe,
  ArrowRightLeft, BarChart3, Activity, Layers, Terminal, Radio, Brain
} from 'lucide-react';
import RouteMap from './RouteMap.jsx';

const CARGO_TYPES = [
  { value: 'general', label: 'General cargo' },
  { value: 'perishable_urgent', label: 'Perishable, urgent (no sea)' },
  { value: 'hazardous_waste', label: 'Hazardous waste (no air)' },
  { value: 'oversize_heavy', label: 'Oversize / heavy (no road)' },
];
const THREAT_TYPE_LABELS = {
  weather: 'Weather', labour: 'Labour', geopolitical: 'Geopolitical', infrastructure: 'Infrastructure',
  cyber: 'Cyber', congestion: 'Congestion', general: 'General', none: 'No threat',
};

// Hours, with days added for long legs so a 640 h sea voyage reads as ~27 days.
const fmtH = h => (h >= 48 ? `${Math.round(h)}h (${(h / 24).toFixed(1)}d)` : `${Math.round(h * 10) / 10}h`);
const fmtMoney = v => `$${Math.round(v).toLocaleString()}`;

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
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState({ source: '', dest: '' });
  const [searchResults, setSearchResults] = useState({ source: [], dest: [] });
  const [scenarios, setScenarios] = useState([]);
  const [hubs, setHubs] = useState([]);
  const latestQuery = useRef({ source: '', dest: '' });

  useEffect(() => {
    // Pull the live scenario list from the backend instead of hardcoding IDs here,
    // so this dropdown can never drift out of sync with ScenarioManager.SCENARIOS again.
    fetch('/api/scenarios')
      .then(r => r.json())
      .then(data => setScenarios(data))
      .catch(e => console.error("Failed to load scenarios", e));
    fetch('/api/hubs')
      .then(r => r.json())
      .then(data => setHubs(data))
      .catch(e => console.error("Failed to load hubs", e));
  }, []);

  const chokepoints = useMemo(
    () => hubs.filter(h => h.type === 'choke_point').sort((a, b) => a.display_name.localeCompare(b.display_name)),
    [hubs]
  );
  const activeScenario = scenarios.find(s => s.id === operationalConfig);
  const disruptedHubs = activeScenario ? activeScenario.affected_nodes : [];
  const liveHubs = intelReports.filter(r => r.score > 0).flatMap(r => r.hubs);
  const route = recommendations[selected];

  const getRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          destination,
          transport_preference: transportMode,
          routing_policy: routingPolicy,
          cargo_type: cargoType,
          priority: priority,
          scenario: operationalConfig !== 'NORMAL' ? operationalConfig : null,
          live_intel: liveIntel,
          overrides: avoid.length ? { avoid_chokepoints: avoid } : null
        })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        const detail = Array.isArray(data.detail) ? data.detail.map(d => d.msg).join('; ') : data.detail;
        setError(data.error || detail || `Request failed (${res.status})`);
        setRecommendations([]);
        setIntelReports([]);
      } else {
        setRecommendations(data.recommendations);
        setIntelReports(data.live_intel || []);
        setSelected(0);
      }
    } catch (err) {
      setError("Engine connection failed. Verify backend status.");
    } finally {
      setLoading(false);
    }
  };

  const getModeIcon = (mode) => {
    switch (mode.toLowerCase()) {
      case 'air': return <Plane size={12} />;
      case 'sea': return <Ship size={12} />;
      case 'rail': return <Train size={12} />;
      case 'road': return <Truck size={12} />;
      case 'transfer': return <ArrowRightLeft size={12} />;
      default: return <Navigation size={12} />;
    }
  };

  const handleSearch = async (type, query) => {
    setSearchQuery(prev => ({ ...prev, [type]: query }));
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
    } catch (err) { console.error("Search failed"); }
  };

  const selectHub = (type, hub) => {
    if (type === 'source') {
      setSource(hub.id);
      setSearchQuery(prev => ({ ...prev, source: hub.display_name }));
    } else {
      setDestination(hub.id);
      setSearchQuery(prev => ({ ...prev, dest: hub.display_name }));
    }
    setSearchResults(prev => ({ ...prev, [type]: [] }));
  };

  const toggleAvoid = id => setAvoid(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]));

  const personaClass = persona => (
    persona === 'FASTEST' ? 'tag-fastest' : persona === 'SAFEST' ? 'tag-safest' : 'tag-balanced'
  );

  const renderHubSearch = (type, label, placeholder) => (
    <div className="sc-input-group">
      <label className="sc-label">{label}</label>
      <input
        type="text" value={searchQuery[type]}
        onChange={(e) => handleSearch(type, e.target.value)}
        className="sc-input" placeholder={placeholder}
      />
      {searchResults[type].length > 0 && (
        <div className="search-results">
          {searchResults[type].map((h, idx) => (
            <button key={`${h.id}-${idx}`} onClick={() => selectHub(type, h)} className="search-result">
              {h.display_name}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  const renderDrivers = drivers => {
    if (!drivers) return null;
    const peak = Math.max(...drivers.drivers.map(d => Math.abs(d.hours)), 1);
    return (
      <div className="drivers">
        <div className="drivers-title">Why the p85 delay (Shapley, vs a short road hop)</div>
        {drivers.drivers.filter(d => Math.abs(d.hours) >= 0.5).slice(0, 4).map(d => (
          <div key={d.factor} className="driver-row" title={`${d.factor}: ${d.hours > 0 ? '+' : ''}${d.hours}h`}>
            <span className="driver-label">{d.factor}</span>
            <span className="driver-track">
              <span className={`driver-bar ${d.hours >= 0 ? 'up' : 'down'}`}
                    style={{ width: `${Math.max(4, (Math.abs(d.hours) / peak) * 100)}%` }} />
            </span>
            <span className="driver-value">{d.hours > 0 ? '+' : ''}{Math.round(d.hours)}h</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="dashboard-layout">
      {/* Header */}
      <header className="dashboard-header">
        <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
          <Globe size={28} color="#3b82f6" />
          <div>
            <h1 style={{fontSize: '1.25rem', fontWeight: 800}}>Supplychainer Command Console</h1>
            <p style={{fontSize: '0.7rem', color: '#64748b', fontWeight: 700}}>MULTIMODAL ROUTING · QUANTILE DELAY MODEL · LIVE THREAT INTELLIGENCE</p>
          </div>
        </div>
        <div style={{display: 'flex', gap: '1rem', alignItems: 'center'}}>
          {engineStatus && (
            <span className={`engine-status ${engineStatus === 'FULLY OPERATIONAL' ? 'ok' : 'warn'}`}>
              <Activity size={12} /> {engineStatus}
            </span>
          )}
          <button className="sc-badge-active" onClick={() => onNavigate('model')} style={{cursor: 'pointer', borderColor: '#8b5cf6', color: '#8b5cf6'}}>
            <Brain size={14} /> MODEL EVALUATION
          </button>
          <button className="sc-badge-active" onClick={() => onNavigate('suppliers')} style={{cursor: 'pointer'}}>
            <ShieldCheck size={14} /> SUPPLIER INTELLIGENCE
          </button>
        </div>
      </header>

      {/* Left Panel - Command Panel */}
      <aside className="sidebar-left">
        <h2 className="panel-title"><Terminal size={14} /> Strategic Input Panel</h2>

        {renderHubSearch('source', 'Origin Hub', 'Search origin...')}
        {renderHubSearch('dest', 'Destination Hub', 'Search destination...')}

        <div className="sc-input-group">
          <label className="sc-label">Transport Mode</label>
          <select value={transportMode} onChange={e => setTransportMode(e.target.value)} className="sc-select">
            <option value="any">Unconstrained</option>
            <option value="sea">SEA (Maritime Corridors)</option>
            <option value="air">AIR (Express Cargo)</option>
            <option value="rail">RAIL (Inland Freight)</option>
            <option value="road">ROAD (Local Distribution)</option>
          </select>
        </div>

        <div className="sc-input-group">
          <label className="sc-label">Routing Policy</label>
          <select value={routingPolicy} onChange={e => setRoutingPolicy(e.target.value)} className="sc-select">
            <option value="STRICT">STRICT (Hard Exclusion)</option>
            <option value="PREFERRED">PREFERRED (Soft Bias)</option>
          </select>
        </div>

        <div className="sc-input-group">
          <label className="sc-label">Cargo Type</label>
          <select value={cargoType} onChange={e => setCargoType(e.target.value)} className="sc-select">
            {CARGO_TYPES.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
          </select>
        </div>

        <div className="sc-input-group">
          <label className="sc-label">Shipment Priority</label>
          <select value={priority} onChange={e => setPriority(e.target.value)} className="sc-select">
            <option value="low">Low (cost first)</option>
            <option value="normal">Normal</option>
            <option value="urgent">Urgent (time first)</option>
          </select>
        </div>

        <div className="sc-input-group">
          <label className="sc-label">Disruption Scenario</label>
          <select value={operationalConfig} onChange={e => setOperationalConfig(e.target.value)} className="sc-select" style={{borderColor: operationalConfig !== 'NORMAL' ? '#ef4444' : '#1e293b'}}>
            <option value="NORMAL">Operational Normal</option>
            {scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </div>

        <div className="sc-input-group">
          <label className="sc-label">Avoid Chokepoints</label>
          <div className="chip-grid">
            {chokepoints.map(c => (
              <button key={c.id} type="button" onClick={() => toggleAvoid(c.id)}
                      className={`chip ${avoid.includes(c.id) ? 'chip-on' : ''}`} aria-pressed={avoid.includes(c.id)}>
                {c.display_name.replace(/ (Strait|Canal)$/, '')}
              </button>
            ))}
          </div>
        </div>

        <label className="toggle-row">
          <input type="checkbox" checked={liveIntel} onChange={e => setLiveIntel(e.target.checked)} />
          <span><Radio size={12} /> Live news intelligence (origin &amp; destination)</span>
        </label>

        <button className="sc-btn-execute" onClick={getRecommendations} disabled={loading || !source || !destination}>
          {loading ? <Zap className="animate-pulse" size={16} /> : "GENERATE STRATEGIC ROUTE OPTIONS"}
        </button>
      </aside>

      {/* Main Area */}
      <main className="main-content">
        {activeScenario && (
          <div className="scenario-banner animate-slide-in">
            <AlertTriangle size={20} />
            <div>
              <span style={{fontWeight: 800, fontSize: '0.75rem', display: 'block'}}>ACTIVE DISRUPTION SCENARIO</span>
              <span style={{fontSize: '0.875rem'}}>{activeScenario.name}: {activeScenario.reason}</span>
            </div>
          </div>
        )}

        {error && <div className="error-box">{error}</div>}

        <RouteMap hubs={hubs} routes={recommendations} selected={selected} onSelect={setSelected}
                  disrupted={disruptedHubs} liveHubs={liveHubs} />

        <div className="path-grid">
          {recommendations.map((rec, idx) => (
            <div key={idx} className={`path-card ${idx === selected ? 'path-card-selected' : ''}`}
                 onClick={() => setSelected(idx)} role="button" tabIndex={0}
                 onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && setSelected(idx)}>
              <div className="card-header">
                <span className={`persona-badge ${personaClass(rec.persona)}`}>{(rec.personas || [rec.persona]).join(' · ')}</span>
                <div style={{display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem', fontFamily: 'JetBrains Mono'}}>
                   <Clock size={12} /> {fmtH(rec.adjusted_eta)}
                </div>
              </div>
              <div style={{padding: '1.25rem'}}>
                <div className="band-row">
                  <span>Simulated ETA</span>
                  <span>p50 {Math.round(rec.eta_band.p50)}h</span>
                  <span>p85 {Math.round(rec.eta_band.p85)}h</span>
                  <span>p95 {Math.round(rec.eta_band.p95)}h</span>
                </div>
                <h3 style={{fontSize: '0.85rem', fontWeight: 600, margin: '1rem 0 1.25rem', color: '#cbd5e1', lineHeight: 1.45}}>{rec.explanation}</h3>

                <div style={{display: 'flex', flexDirection: 'column', gap: '0.75rem', borderLeft: '2px solid #1e293b', paddingLeft: '1rem', marginLeft: '0.5rem'}}>
                  {rec.legs.map((leg, lIdx) => {
                    const isTransfer = leg.type === 'transfer';
                    return (
                      <div key={lIdx} style={{display: 'flex', flexDirection: 'column', opacity: isTransfer ? 0.7 : 1}}>
                        <span style={{
                          fontSize: '0.65rem',
                          fontWeight: 800,
                          color: isTransfer ? '#94a3b8' : '#3b82f6',
                          letterSpacing: '0.05em',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}>
                          {getModeIcon(leg.mode)}
                          {isTransfer ? 'STRATEGIC HANDOFF' : `${leg.mode} TRANSIT`}
                          {leg.intel_source !== 'FALLBACK' && (
                            <span className={`intel-badge intel-${leg.intel_source.toLowerCase()}`} title={leg.reason}>
                              {leg.intel_source} {Math.round(leg.threat * 100)}%
                            </span>
                          )}
                        </span>
                        <span style={{fontSize: '0.8rem', fontWeight: 600}}>
                          {isTransfer ? `Processing at ${leg.to_name}` : `to ${leg.to_name}`}
                          <span className="leg-eta"> · {fmtH(leg.eta)}</span>
                        </span>
                      </div>
                    );
                  })}
                </div>
                {renderDrivers(rec.delay_drivers)}
              </div>
              <div style={{padding: '1.25rem', borderTop: '1px solid #1e293b', background: 'rgba(15, 23, 42, 0.3)'}}>
                 <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700}}>
                   <span style={{color: '#64748b'}}>TOTAL COST</span>
                   <span style={{color: '#10b981'}}>{fmtMoney(rec.total_cost)}</span>
                 </div>
                 <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700, marginTop: '0.4rem'}}>
                   <span style={{color: '#64748b'}}>PEAK THREAT</span>
                   <span style={{color: rec.threat_level >= 0.5 ? '#ef4444' : '#94a3b8'}}>{Math.round(rec.threat_level * 100)}%</span>
                 </div>
              </div>
            </div>
          ))}
        </div>
      </main>

      {/* Right Sidebar - Audit/Intel */}
      <aside className="sidebar-right">
        <h2 className="panel-title"><Layers size={14} /> Decision Integrity Audit</h2>

        {route ? (
          <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
            <div className="audit-trace-box" style={{borderLeft: '4px solid #3b82f6'}}>
               <div className="audit-title">ETA Audit: {(route.personas || [route.persona]).join(' · ')}</div>
               <div>Transit: {route.audit_trace.eta.transit}h</div>
               <div>Transfers: +{route.audit_trace.eta.transfer}h</div>
               <div>Typical delay (model p50): +{route.audit_trace.eta.delay}h</div>
               <div>Scenario impact: {route.audit_trace.eta.scenario > 0 ? `+${route.audit_trace.eta.scenario}h` : 'None'}</div>
               <div className="audit-total">= Typical ETA {route.adjusted_eta}h</div>
               <div style={{marginTop: '0.5rem'}}>Simulated range: p50 {route.eta_band.p50}h · p85 {route.eta_band.p85}h · p95 {route.eta_band.p95}h</div>
            </div>

            <div className="audit-trace-box" style={{borderLeft: '4px solid #10b981'}}>
               <div className="audit-title">Cost Composition</div>
               <div>Transit: {fmtMoney(route.audit_trace.cost.transit)}</div>
               <div>Transfer fees: {fmtMoney(route.audit_trace.cost.transfer)}</div>
               <div>Risk premium: {fmtMoney(route.audit_trace.cost.scenario)}</div>
               <div className="audit-total">= {fmtMoney(route.total_cost)}</div>
            </div>

            <div className="audit-trace-box" style={{borderLeft: '4px solid #f59e0b'}}>
               <div className="audit-title">Risk Exposure</div>
               <div>Baseline (fallback reports): {Math.round(route.audit_trace.risk.baseline * 100)}%</div>
               <div>Scenario: {Math.round(route.audit_trace.risk.scenario * 100)}%</div>
               <div>Live news: {Math.round(route.audit_trace.risk.live * 100)}%</div>
            </div>
          </div>
        ) : (
          <div style={{textAlign: 'center', color: '#64748b', marginTop: '2rem'}}>
            <Activity size={48} style={{opacity: 0.1, marginBottom: '1rem'}} />
            <p style={{fontSize: '0.8rem'}}>Pick an origin and destination to generate routes.</p>
          </div>
        )}

        <div>
          <h2 className="panel-title"><Radio size={14} /> Live Intelligence</h2>
          {!liveIntel && <p className="muted-note">Live news is off.</p>}
          {liveIntel && recommendations.length > 0 && intelReports.length === 0 && (
            <p className="muted-note">No live reports: the feed is quiet, offline, or the NLP engine is still warming up.</p>
          )}
          {intelReports.map(r => (
            <div key={r.place} className="intel-card">
              <div className="intel-head">
                <span>{r.place}</span>
                <span className={`intel-type ${r.score > 0 ? 'hot' : ''}`}>{THREAT_TYPE_LABELS[r.threat_type] || r.threat_type}</span>
              </div>
              <div className="threat-meter"><span style={{width: `${Math.round(r.score * 100)}%`}} /></div>
              <div className="intel-headline">{r.headline || r.headlines}</div>
              {r.condition !== 'clear' && <div className="intel-condition">Weather feature set to {r.condition}</div>}
            </div>
          ))}
        </div>
      </aside>

      {/* Bottom Tradeoff Strip */}
      <footer className="tradeoff-strip">
        <div style={{display: 'flex', alignItems: 'center', gap: '0.75rem'}}>
          <BarChart3 size={20} color="#64748b" />
          <span style={{fontSize: '0.75rem', fontWeight: 800, color: '#64748b'}}>TRADEOFF ANALYSIS</span>
        </div>
        <div style={{display: 'flex', gap: '3rem', flex: 1, justifyContent: 'center'}}>
          <div className="tradeoff-item">
            <span>FASTEST (TYPICAL)</span>
            <span style={{color: '#f59e0b'}}>{recommendations.length > 0 ? fmtH(Math.min(...recommendations.map(r => r.adjusted_eta))) : '--'}</span>
          </div>
          <div className="tradeoff-item">
            <span>LOWEST COST</span>
            <span style={{color: '#10b981'}}>{recommendations.length > 0 ? fmtMoney(Math.min(...recommendations.map(r => r.total_cost))) : '--'}</span>
          </div>
          <div className="tradeoff-item">
            <span>RISK FLOOR</span>
            <span style={{color: '#3b82f6'}}>{recommendations.length > 0 ? `${Math.round(Math.min(...recommendations.map(r => r.threat_level)) * 100)}%` : '--'}</span>
          </div>
          <div className="tradeoff-item">
            <span>SELECTED P85 BUFFER</span>
            <span style={{color: '#a855f7'}}>{route ? `+${Math.round(route.eta_band.p85 - route.adjusted_eta)}h` : '--'}</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default RouteRecommender;
