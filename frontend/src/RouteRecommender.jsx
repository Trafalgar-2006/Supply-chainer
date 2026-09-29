import React, { useState, useEffect, useRef, useMemo } from 'react';
import { animate, stagger } from 'animejs';
import { Truck, Ship, Plane, Train, ArrowRightLeft, ArrowUpDown, Navigation, Download, Printer } from 'lucide-react';
import RouteMap, { PERSONA_COLOURS, personaLabel, prefersReducedMotion } from './RouteMap.jsx';
import { readJson, requestError } from './api.js';

// `forbids`: the mode the engine never uses for this cargo (MODE_PROFILES' cargo_restrictions).
const CARGO_TYPES = [
  { value: 'general', label: 'General cargo' },
  { value: 'perishable_urgent', label: 'Perishable, urgent (no sea)', forbids: 'sea' },
  { value: 'hazardous_waste', label: 'Hazardous waste (no air)', forbids: 'air' },
  { value: 'oversize_heavy', label: 'Oversize or heavy (no road)', forbids: 'road' },
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
export const RECENT_KEY = 'supplychainer.recent-plans';
const RECENT_LIMIT = 8;
const SEARCH_MAX = 100; // the API's limit on a hub search

// Hours, shown as days once a trip passes two days (a 640 h voyage reads 26.7
// days). Figures read together pass `days` so they share one unit.
const fmtH = (h, days = h >= 48) => (days ? `${(h / 24).toFixed(1)} days` : `${Math.round(h * 10) / 10} h`);
// Costs come in US dollars; another currency is shown at the day's ECB reference rate.
// Always en-US grouping, so the same figure never reads $1,38,753 in one place and $138,753 in another.
const moneyFormat = (currency, rate) => {
  const f = new Intl.NumberFormat('en-US', { style: 'currency', currency, minimumFractionDigits: 0, maximumFractionDigits: 0 });
  return v => f.format(v * rate);
};
const CURRENCY_NAMES = {
  USD: 'US dollars', EUR: 'Euros', GBP: 'Pounds sterling', INR: 'Indian rupees', CNY: 'Chinese yuan',
  JPY: 'Japanese yen', SGD: 'Singapore dollars',
};
const fmtTime = iso => new Date(iso).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
const pct = x => `${Math.round(x * 100)}%`;
const capitalise = s => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();

function modeSummary(route) {
  const modes = route.legs.filter(l => l.type === 'transit').map(l => l.mode.toLowerCase())
    .filter((m, i, all) => m !== all[i - 1]);
  const transfers = route.legs.length - route.legs.filter(l => l.type === 'transit').length;
  return `${capitalise(modes.join(', '))}. ${transfers} transfer${transfers === 1 ? '' : 's'}.`;
}

// Recent plans are kept in this browser only, each with the result it got, so
// reopening one shows exactly what was recommended then. Storage can be full or
// disabled, in which case they are kept without results, or not at all.
const planKey = p => JSON.stringify([p.source, p.destination, p.transportMode, p.routingPolicy,
  p.cargoType, p.priority, p.liveIntel, [...p.avoid].sort()]);

// Saved plans come back from storage that an older version or a person may
// have changed, so everything the page reads is checked: a plan with an unknown
// value is dropped, and a result that doesn't look like a planner response is
// dropped while its plan is kept.
const isNum = x => typeof x === 'number' && Number.isFinite(x);
const isStr = x => typeof x === 'string';
const numbers = (o, keys) => Boolean(o) && typeof o === 'object' && keys.every(k => isNum(o[k]));
const QUANTILES = ['p50', 'p85', 'p95'];
const validLeg = l => Boolean(l) && isStr(l.from) && isStr(l.to) && isStr(l.to_name) && isStr(l.mode)
  && (l.type === 'transit' || l.type === 'transfer') && isNum(l.eta) && isNum(l.cost) && isNum(l.threat)
  && isStr(l.reason) && isStr(l.intel_source) && numbers(l.delay, QUANTILES);
const validRoute = r => Boolean(r) && isStr(r.persona) && Array.isArray(r.personas) && r.personas.every(isStr)
  && isStr(r.explanation) && Array.isArray(r.legs) && r.legs.length > 0 && r.legs.every(validLeg)
  && numbers(r.eta_band, QUANTILES) && isNum(r.adjusted_eta) && isNum(r.total_cost) && isNum(r.threat_level)
  && Boolean(r.audit_trace) && numbers(r.audit_trace.eta, ['transit', 'transfer', 'delay', 'scenario'])
  && numbers(r.audit_trace.cost, ['transit', 'transfer', 'scenario']) && numbers(r.audit_trace.risk, ['baseline', 'scenario', 'live'])
  && (r.delay_drivers == null || (Array.isArray(r.delay_drivers.drivers)
    && r.delay_drivers.drivers.every(d => d && isStr(d.factor) && isNum(d.hours))));
const validReport = r => Boolean(r) && isStr(r.place) && Array.isArray(r.hubs) && isNum(r.score) && isNum(r.severity)
  && isStr(r.threat_type) && isStr(r.condition) && (r.headline == null || isStr(r.headline))
  && (r.headlines == null || isStr(r.headlines))
  && (r.weather == null || (isStr(r.weather.description) && isNum(r.weather.wind_kmh)));
const validResult = r => Boolean(r) && Array.isArray(r.recommendations) && r.recommendations.length > 0
  && r.recommendations.every(validRoute) && Array.isArray(r.live_intel) && r.live_intel.every(validReport)
  && (r.pending == null || (Array.isArray(r.pending) && r.pending.every(isStr)));
const MODES = ['any', 'sea', 'air', 'rail', 'road'];
const validPlan = p => Boolean(p) && isStr(p.source) && isStr(p.destination) && isStr(p.sourceName) && isStr(p.destName)
  && MODES.includes(p.transportMode) && ['STRICT', 'PREFERRED'].includes(p.routingPolicy)
  && CARGO_TYPES.some(c => c.value === p.cargoType) && ['low', 'normal', 'urgent'].includes(p.priority)
  && typeof p.liveIntel === 'boolean' && (p.scenario === null || isStr(p.scenario))
  && Array.isArray(p.avoid) && p.avoid.every(isStr) && Array.isArray(p.hubs) && p.hubs.every(isStr);

function loadRecent() {
  try {
    const saved = JSON.parse(localStorage.getItem(RECENT_KEY));
    return Array.isArray(saved) ? saved.filter(validPlan)
      .map(({ result, plannedAt, ...p }) => (validResult(result) && isStr(plannedAt) && !Number.isNaN(Date.parse(plannedAt))
        ? { ...p, result, plannedAt } : p)) : [];
  } catch {
    return [];
  }
}

// A plan's inputs alone, without its saved result.
const planOnly = ({ result, plannedAt, saved, example, ...plan }) => plan;

function saveRecent(plans) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(plans));
  } catch {
    try {
      localStorage.setItem(RECENT_KEY, JSON.stringify(plans.map(planOnly)));
    } catch {
      // Not remembered; the plans still work for this session.
    }
  }
}

// What moved since the last time this plan was made, option by option.
function changesSince(before, after, fmtMoney) {
  const path = r => r.legs.map(l => l.to).join();
  return after.flatMap(r => {
    const old = before.find(b => b.persona === r.persona);
    if (!old) return [];
    const parts = [];
    if (path(old) !== path(r)) parts.push('a different route');
    const hours = r.eta_band.p85 - old.eta_band.p85;
    if (Math.abs(hours) >= 1) parts.push(`plan-for time ${hours > 0 ? 'up' : 'down'} ${fmtH(Math.abs(hours))}`);
    if (Math.abs(r.total_cost - old.total_cost) >= 0.02 * old.total_cost) {
      parts.push(`cost ${fmtMoney(old.total_cost)} to ${fmtMoney(r.total_cost)}`);
    }
    const risk = Math.round((r.threat_level - old.threat_level) * 100);
    if (Math.abs(risk) >= 5) parts.push(`peak risk ${pct(old.threat_level)} to ${pct(r.threat_level)}`);
    return parts.length ? [`${personaLabel(r)}: ${parts.join(', ')}.`] : [];
  });
}

// Every cell quoted. Headlines come from outside, so text that a spreadsheet
// would run as a formula (= + - @) gets a leading apostrophe.
const csvCell = v => {
  const text = typeof v === 'string' && /^[=+\-@\t\r]/.test(v) ? `'${v}` : String(v);
  return `"${text.replace(/"/g, '""')}"`;
};
const slug = s => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadCsv(route, hubName) {
  const origin = hubName(route.legs[0].from);
  const destination = route.legs[route.legs.length - 1].to_name;
  const rows = [
    // A leg's hours already include its typical (p50) delay; the delay columns are
    // for reference and must not be added to them.
    ['leg', 'type', 'mode', 'from', 'to', 'hours_incl_p50_delay', 'delay_p50_h', 'delay_p85_h', 'delay_p95_h',
      'cost_usd_estimate', 'threat', 'intel_source', 'reason'],
    ...route.legs.map((l, i) => [i + 1, l.type, l.mode, hubName(l.from), l.to_name, l.eta, l.delay.p50, l.delay.p85, l.delay.p95,
      l.cost, l.threat, l.intel_source, l.reason]),
    ['total', '', '', origin, destination, route.adjusted_eta, '', '', '', route.total_cost, route.threat_level, '',
      `Typical ${route.adjusted_eta} h (the legs' hours added up); simulated whole trip: median ${route.eta_band.p50} h, `
        + `plan for ${route.eta_band.p85} h (p85), up to ${route.eta_band.p95} h (p95)`],
  ];
  // The byte-order mark lets Excel read accented hub names as UTF-8.
  download(`route-${slug(origin)}-to-${slug(destination)}.csv`,
    '﻿' + rows.map(r => r.map(csvCell).join(',')).join('\r\n'), 'text/csv;charset=utf-8');
}

// Every option with its full audit trail, in the API's own shape, plus the
// request that produced it: what a TMS or ERP would import.
function downloadJson(context, recommendations, intelReports) {
  download(`routes-${slug(context.sourceName)}-to-${slug(context.destName)}.json`,
    JSON.stringify({ planned_at: context.plannedAt, currency: 'USD', request: apiRequest(context), recommendations,
      live_intel: intelReports }, null, 2),
    'application/json');
}

// The body /api/recommend takes, from a plan.
const apiRequest = plan => ({
  source: plan.source, destination: plan.destination, transport_preference: plan.transportMode,
  routing_policy: plan.routingPolicy, cargo_type: plan.cargoType, priority: plan.priority,
  scenario: plan.scenario, live_intel: plan.liveIntel,
  overrides: plan.avoid.length ? { avoid_chokepoints: plan.avoid } : null,
});

const RouteRecommender = ({ onNavigate, engineStatus, visible = true }) => {
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
  const [activeOption, setActiveOption] = useState({ source: -1, dest: -1 }); // highlighted suggestion
  const [searchNote, setSearchNote] = useState({ source: '', dest: '' }); // no matches, or search unavailable
  const [pendingIntel, setPendingIntel] = useState([]); // places whose news was still loading
  const [scenarios, setScenarios] = useState([]);
  const [hubs, setHubs] = useState([]);
  const [recent, setRecent] = useState(loadRecent);
  const [focusLeg, setFocusLeg] = useState(null); // the voyage-plan leg highlighted on the map
  const [changes, setChanges] = useState(null); // what moved since this plan was last made
  const [currency, setCurrency] = useState('USD');
  const [fx, setFx] = useState({ date: null, rates: {} });
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
    // Without rates (offline source), costs stay in US dollars.
    fetch('/api/currencies')
      .then(r => r.json())
      .then(data => data && data.rates && setFx(data))
      .catch(() => {});
  }, []);

  const rate = currency === 'USD' ? 1 : fx.rates[currency];
  const fmtMoney = useMemo(() => (rate ? moneyFormat(currency, rate) : moneyFormat('USD', 1)), [currency, rate]);

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
    setActiveOption({ source: -1, dest: -1 });
    setSearchNote({ source: '', dest: '' });
    setTransportMode(plan.transportMode);
    setRoutingPolicy(plan.routingPolicy);
    setCargoType(plan.cargoType);
    setPriority(plan.priority);
    setOperationalConfig(plan.scenario || 'NORMAL');
    setLiveIntel(plan.liveIntel);
    setAvoid(plan.avoid);
  };

  const sameAs = plan => p => planKey(p) === planKey(plan) && p.scenario === plan.scenario;

  const remember = (plan, result) => {
    const entry = { ...plan, result, hubs: [...new Set(result.recommendations.flatMap(r => r.legs.flatMap(l => [l.from, l.to])))] };
    const next = [entry, ...recent.filter(p => !sameAs(entry)(p))].slice(0, RECENT_LIMIT);
    setRecent(next);
    saveRecent(next);
  };

  // A saved plan opens as it was; planning again fetches current conditions.
  const openSaved = ({ result, ...plan }) => {
    applyPlan(plan);
    setError(null);
    setChanges(null);
    setRecommendations(result.recommendations);
    setIntelReports(result.live_intel);
    setPendingIntel(result.pending || []);
    setResultContext({ ...plan, saved: true });
    setSelected(0);
  };

  const runPlan = async (request, { example = false } = {}) => {
    const plan = planOnly(request);
    applyPlan(plan);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiRequest(plan)),
      });
      const data = await readJson(res);
      const result = data && { recommendations: data.recommendations, live_intel: data.live_intel || [],
        pending: data.live_intel_pending || [] };
      if (!res.ok || !validResult(result)) {
        setError(res.ok ? 'The routing engine sent a reply this page could not read. Plan again.' : requestError(res, data));
        setRecommendations([]);
        setIntelReports([]);
        setPendingIntel([]);
        setResultContext(null);
        setChanges(null);
      } else {
        const planned = { ...plan, plannedAt: new Date().toISOString() };
        const before = recent.find(p => sameAs(plan)(p) && p.result);
        setChanges(before ? { since: before.plannedAt, before: before.result.recommendations } : null);
        setRecommendations(result.recommendations);
        setIntelReports(result.live_intel);
        setPendingIntel(result.pending);
        setResultContext({ ...planned, example });
        setSelected(0);
        remember(planned, result);
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
  // Plans the engine would refuse, explained before they are sent, and a note
  // when the preferred mode is one this cargo can't use.
  const cargoRule = CARGO_TYPES.find(c => c.value === cargoType) || CARGO_TYPES[0];
  const cargoName = cargoRule.label.split(' (')[0];
  const conflict = source && source === destination ? 'Choose a destination different from the origin.'
    : cargoRule.forbids === transportMode && routingPolicy === 'STRICT'
      ? `${cargoName} cargo can't go by ${transportMode}. Choose another mode, or set the mode policy to prefer it.`
      : null;
  const planNote = !conflict && cargoRule.forbids === transportMode
    ? `${cargoName} cargo can't go by ${transportMode}, so the routes use other modes.` : null;
  const unpicked = ['source', 'dest'].filter(t => searchQuery[t].trim() && !(t === 'source' ? source : destination));
  const policyHint = transportMode === 'any' ? 'Choose a transport mode to set a policy.'
    : routingPolicy === 'STRICT'
      ? transportMode === 'road' ? 'Road only.' : `At least one ${transportMode} leg; road only to reach it and to leave it.`
      : `Favours ${transportMode}; other modes only when they are clearly better.`;
  // The routes shown no longer match the controls: they are dimmed and can't be exported.
  const inputsChanged = resultContext && planKey(planFromControls()) !== planKey(resultContext);
  const stale = !loading && (inputsChanged || scenarioChanged);
  const changeLines = changes ? changesSince(changes.before, recommendations, fmtMoney) : [];

  const setFor = (setter, type, value) => setter(prev => ({ ...prev, [type]: value }));
  const closeSuggestions = type => {
    setFor(setSearchResults, type, []);
    setFor(setActiveOption, type, -1);
  };

  const handleSearch = async (type, query) => {
    setFor(setSearchQuery, type, query);
    // Editing the text un-selects the hub, so a stale choice is never submitted.
    if (type === 'source') setSource(''); else setDestination('');
    latestQuery.current[type] = query;
    const q = query.trim();
    if (q.length < 2) {
      closeSuggestions(type);
      setFor(setSearchNote, type, '');
      return;
    }
    let results = [];
    let note = '';
    try {
      const res = await fetch(`/api/hubs/search?q=${encodeURIComponent(q)}`);
      const data = await readJson(res);
      if (!res.ok || !Array.isArray(data)) note = `Place search failed. ${requestError(res, data)}`;
      else if (!data.length) note = `No place matches "${q}".`;
      else results = data;
    } catch {
      note = 'Place search is unavailable: the routing engine cannot be reached.';
    }
    // Responses can arrive out of order while typing; keep only the latest.
    if (latestQuery.current[type] === query) {
      setFor(setSearchResults, type, results);
      setFor(setActiveOption, type, -1);
      setFor(setSearchNote, type, note);
    }
  };

  const selectHub = (type, hub) => {
    if (type === 'source') setSource(hub.id); else setDestination(hub.id);
    setFor(setSearchQuery, type, hub.display_name);
    latestQuery.current[type] = hub.display_name;
    closeSuggestions(type);
    setFor(setSearchNote, type, '');
  };

  // Up and down move through the suggestions, Enter picks one, Escape closes them.
  const onSearchKey = (type, e) => {
    const options = searchResults[type];
    if (e.key === 'Escape' && options.length) {
      e.preventDefault();
      closeSuggestions(type);
    } else if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && options.length) {
      e.preventDefault();
      const step = e.key === 'ArrowDown' ? 1 : -1;
      setFor(setActiveOption, type, (activeOption[type] + step + options.length + (activeOption[type] < 0 && step < 0 ? 1 : 0))
        % options.length);
    } else if (e.key === 'Enter' && options.length) {
      e.preventDefault();
      selectHub(type, options[Math.max(activeOption[type], 0)]);
    }
  };

  const toggleAvoid = id => setAvoid(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]));

  const swapEnds = () => {
    setSource(destination);
    setDestination(source);
    setSearchQuery({ source: searchQuery.dest, dest: searchQuery.source });
    latestQuery.current = { source: searchQuery.dest, dest: searchQuery.source };
    setSearchResults({ source: [], dest: [] });
    setActiveOption({ source: -1, dest: -1 });
    setSearchNote({ source: '', dest: '' });
  };

  // An example fills in the whole form, so the notice says so (see `example`).
  const runExample = example => runPlan({
    source: example.source, sourceName: hubName(example.source),
    destination: example.destination, destName: hubName(example.destination),
    transportMode: example.mode, routingPolicy: 'STRICT', cargoType: 'general', priority: 'normal',
    scenario: example.scenario, liveIntel, avoid: [],
  }, { example: true });

  // Pointing at a leg (mouse or keyboard) highlights its stretch on the map.
  const pointAt = leg => ({
    tabIndex: 0,
    onMouseEnter: () => setFocusLeg(leg), onMouseLeave: () => setFocusLeg(null),
    onFocus: () => setFocusLeg(leg), onBlur: () => setFocusLeg(null),
  });

  const [engineState, engineText] = ENGINE_STATES[engineStatus] || ['warming', 'Connecting to the engine'];

  const renderHubSearch = (type, label, placeholder) => {
    const options = searchResults[type];
    const id = `hub-${type}`;
    const note = searchNote[type] || (unpicked.includes(type) && !options.length ? 'Pick a place from the list.' : '');
    return (
      <div className="field">
        <label htmlFor={id}>{label}</label>
        <input
          id={id} type="text" value={searchQuery[type]} autoComplete="off" maxLength={SEARCH_MAX}
          role="combobox" aria-autocomplete="list" aria-expanded={options.length > 0} aria-controls={`${id}-list`}
          aria-activedescendant={activeOption[type] >= 0 ? `${id}-option-${activeOption[type]}` : undefined}
          aria-describedby={note ? `${id}-note` : undefined}
          onChange={e => handleSearch(type, e.target.value)} onKeyDown={e => onSearchKey(type, e)}
          onBlur={() => closeSuggestions(type)}
          className="control" placeholder={placeholder}
        />
        <ul id={`${id}-list`} role="listbox" aria-label={`${label} suggestions`} className="suggestions" hidden={!options.length}>
          {options.map((h, idx) => (
            <li key={h.id} id={`${id}-option-${idx}`} role="option" aria-selected={idx === activeOption[type]}
                onMouseDown={e => e.preventDefault()} onClick={() => selectHub(type, h)}>
              {h.display_name}
            </li>
          ))}
        </ul>
        {note && <p id={`${id}-note`} className="field-hint" role="status">{note}</p>}
      </div>
    );
  };

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
                  disabled={transportMode === 'any'} aria-describedby="policy-hint">
            <option value="STRICT">Only this mode</option>
            <option value="PREFERRED">Prefer this mode</option>
          </select>
          <p id="policy-hint" className="field-hint">{policyHint}</p>
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

        <div className="field">
          <label htmlFor="currency">Show costs in</label>
          <select id="currency" value={currency} onChange={e => setCurrency(e.target.value)} className="control"
                  disabled={!Object.keys(fx.rates).length}>
            {['USD', ...Object.keys(fx.rates)].map(c => <option key={c} value={c}>{CURRENCY_NAMES[c] || c}</option>)}
          </select>
        </div>

        <button type="button" className="primary" onClick={() => runPlan(planFromControls())}
                disabled={loading || !source || !destination || Boolean(conflict)}
                aria-describedby={conflict ? 'plan-conflict' : undefined}>
          {loading ? 'Planning routes…' : 'Plan routes'}
        </button>
        {conflict && <p id="plan-conflict" className="field-error" role="status">{conflict}</p>}
        {planNote && <p className="field-hint" role="status">{planNote}</p>}

        {recent.length > 0 && (
          <>
            <h2>Recent plans</h2>
            <ul className="recent">
              {recent.map(p => (
                <li key={planKey(p) + p.scenario}>
                  <button type="button" disabled={loading} onClick={() => (p.result ? openSaved(p) : runPlan(p))}>
                    {p.sourceName} to {p.destName}
                    <small>
                      {scenarios.find(s => s.id === p.scenario)?.name || (p.scenario ? p.scenario : 'Normal operations')}
                      {p.transportMode !== 'any' ? `, ${p.transportMode}` : ''}
                      {p.result ? `. Saved result from ${fmtTime(p.plannedAt)}` : ''}
                    </small>
                  </button>
                  {p.result && (
                    <button type="button" className="link again" disabled={loading} onClick={() => runPlan(p)}
                            aria-label={`Plan ${p.sourceName} to ${p.destName} again with current conditions`}>
                      Plan again with current conditions
                    </button>
                  )}
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
        {resultContext && recommendations.length > 0 && (
          <p className="print-only">
            {resultContext.sourceName} to {resultContext.destName}, planned {fmtTime(resultContext.plannedAt)}.
            Cargo: {CARGO_TYPES.find(c => c.value === resultContext.cargoType)?.label}. Priority: {resultContext.priority}.
            Mode: {resultContext.transportMode}. Live news and weather: {resultContext.liveIntel ? 'on' : 'off'}.
          </p>
        )}
        {resultContext?.example && !stale && (
          <div className="notice">
            <strong>Example plan.</strong> The form on the left now holds its settings
            ({resultContext.transportMode === 'any' ? 'any mode' : `${resultContext.transportMode} only`},
            {' '}{resultScenario ? resultScenario.name : 'normal operations'}, general cargo). Change them to plan your own shipment.
          </div>
        )}
        {resultContext?.saved && !stale && (
          <div className="notice warn">
            <strong>Saved result from {fmtTime(resultContext.plannedAt)}.</strong> Conditions may have changed since; plan
            again for current news, weather and delays.
          </div>
        )}
        {changes && !stale && (
          <div className="notice">
            <strong>Since your plan of {fmtTime(changes.since)}:</strong> {changeLines.length ? '' : 'no change.'}
            {changeLines.length > 0 && <ul className="changes">{changeLines.map(line => <li key={line}>{line}</li>)}</ul>}
          </div>
        )}
        {stale && (inputsChanged || !affectedPlans.some(p => planKey(p) === planKey(resultContext))) && (
          <div className="notice warn" role="status">
            <strong>{inputsChanged ? 'Inputs changed.' : 'Scenario changed.'}</strong> The routes shown are for the
            previous inputs; plan again to update them.
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

        <RouteMap hubs={hubs} routes={recommendations} selected={selected} onSelect={setSelected} visible={visible}
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
                The coloured mark is the time to plan for: 85% of simulated trips arrive by then. Typical adds up
                each leg's median delay; the simulated median of the whole trip is a little higher, because delays are skewed.
              </p>
            </section>
            <div className="options-head options-grid" aria-hidden="true">
              <span /><span>Route</span><span>Door-to-door time</span><span>Estimated cost</span><span>Peak risk</span>
            </div>
            <ul className={`options ${loading ? 'busy' : ''} ${stale ? 'stale' : ''}`} ref={optionList} aria-busy={loading}>
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
                            title={`Typical ${Math.round(rec.adjusted_eta)} h (each leg's median delay added up). Simulated whole trip: median ${Math.round(band.p50)} h, p85 ${Math.round(band.p85)} h, p95 ${Math.round(band.p95)} h.`}>
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

      <aside className={`voyage ${stale ? 'stale' : ''}`} aria-label="Selected route">
        {route ? (
          <>
            <section>
              <h2>{personaLabel(route)}</h2>
              <p className="lead">{route.explanation}</p>
              <div className="exports">
                <button type="button" className="secondary" disabled={stale} onClick={() => downloadCsv(route, hubName)}
                        title="This route's legs, times, costs and threats">
                  <Download size={14} aria-hidden="true" /> Export CSV
                </button>
                <button type="button" className="secondary" disabled={stale}
                        onClick={() => downloadJson(resultContext, recommendations, intelReports)}
                        title="Every option with its audit trail, as the API returns it">
                  <Download size={14} aria-hidden="true" /> Export JSON
                </button>
                <button type="button" className="secondary" disabled={stale} onClick={() => window.print()}
                        title="A report of all options and this route, to print or save as PDF">
                  <Printer size={14} aria-hidden="true" /> Print report
                </button>
              </div>
              {stale && <p className="note">Plan again to export the routes for the current inputs.</p>}
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
                  <tr>
                    <td>Scenario delay</td>
                    <td>
                      {route.audit_trace.eta.scenario > 0 ? fmtH(route.audit_trace.eta.scenario, ledgerDays)
                        : resultScenario ? 'None, route avoids it' : 'None'}
                    </td>
                  </tr>
                  <tr className="total"><td>Typical door to door</td><td>{fmtH(route.adjusted_eta, ledgerDays)}</td></tr>
                  <tr className="gap"><td>Plan for (p85)</td><td>{fmtH(route.eta_band.p85, ledgerDays)}</td></tr>
                  <tr><td>Worst case (p95)</td><td>{fmtH(route.eta_band.p95, ledgerDays)}</td></tr>
                </tbody>
              </table>
            </section>

            <section>
              <h2>Cost</h2>
              <p className="note">
                Estimates for comparing the options, not freight quotes: a flat rate per kilometre for each mode, fixed
                transfer fees, and a risk premium on legs through a disruption.
                {rate && currency !== 'USD' ? ` Converted from US dollars at the ECB reference rate of ${fx.date}.` : ''}
              </p>
              <table className="ledger">
                <tbody>
                  <tr><td>Transit</td><td>{fmtMoney(route.audit_trace.cost.transit)}</td></tr>
                  <tr><td>Transfer fees</td><td>{fmtMoney(route.audit_trace.cost.transfer)}</td></tr>
                  <tr><td>Scenario risk premium</td><td>{fmtMoney(route.audit_trace.cost.scenario)}</td></tr>
                  <tr className="total"><td>Estimated total</td><td>{fmtMoney(route.total_cost)}</td></tr>
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
          {resultContext?.liveIntel && pendingIntel.length > 0 && (
            <p className="note" role="status">
              News for {pendingIntel.join(' and ')} was still loading when these routes were planned. Plan again in a few
              seconds to include it.
            </p>
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
