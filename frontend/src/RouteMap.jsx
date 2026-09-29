import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { animate, svg } from 'animejs';
import 'leaflet/dist/leaflet.css';

// Mirrors the palette in index.css; one colour per route option and per kind of exposure.
export const PERSONA_COLOURS = { FASTEST: '#E3A83B', BALANCED: '#6CC3D5', SAFEST: '#49B083' };
const EXPOSURE_COLOURS = { SCENARIO: '#E0564A', LIVE: '#C58BE0' };
const PERSONA_NAMES = { FASTEST: 'Fastest', BALANCED: 'Best balance', SAFEST: 'Lowest risk' };

// "Fastest and best balance": every option a route turned out best for.
export function personaLabel(route) {
  const names = (route.personas || [route.persona]).map(p => PERSONA_NAMES[p] || p);
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1].toLowerCase()}`;
}

export const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Continue a longitude from the previous one: shift by +-360 so consecutive
// points are never more than 180 degrees apart, otherwise a transpacific leg
// would be drawn the long way round the world.
function continueLon(lon, prev) {
  let l = lon;
  while (l - prev > 180) l -= 360;
  while (l - prev < -180) l += 360;
  return l;
}

// Hub ids a route visits in order, and each one's position along the route with
// longitudes continued across the antimeridian.
function routePositions(route, byId) {
  const ids = [route.legs[0]?.from, ...route.legs.map(l => l.to)].filter((id, i, a) => id && id !== a[i - 1] && byId[id]);
  const positions = {};
  const points = [];
  ids.forEach(id => {
    const hub = byId[id];
    const lon = points.length ? continueLon(hub.lon, points[points.length - 1][1]) : hub.lon;
    positions[id] = positions[id] || [hub.lat, lon];
    points.push([hub.lat, lon]);
  });
  return { points, positions };
}

function pulse(className, label, latlng) {
  return L.marker(latlng, {
    icon: L.divIcon({ className: `pulse-marker ${className}`, iconSize: [14, 14] }),
    keyboard: false,
  }).bindTooltip(label);
}

export default function RouteMap({ hubs, routes, selected, onSelect, disrupted, liveHubs, focusLeg }) {
  const container = useRef(null);
  const map = useRef(null);
  const layers = useRef({});
  const fitted = useRef(null);

  useEffect(() => {
    map.current = L.map(container.current, { worldCopyJump: true, minZoom: 2 }).setView([22, 45], 2);
    // OpenStreetMap's standard tiles need no API key; CSS darkens the tile pane
    // (not the route overlays) to match the dashboard.
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 10,
    }).addTo(map.current);
    layers.current = {
      hubs: L.layerGroup().addTo(map.current),
      routes: L.layerGroup().addTo(map.current),
      focus: L.layerGroup().addTo(map.current),
    };
    const resize = setTimeout(() => map.current && map.current.invalidateSize(), 0);
    return () => { clearTimeout(resize); map.current.remove(); map.current = null; };
  }, []);

  useEffect(() => {
    const group = layers.current.hubs;
    group.clearLayers();
    hubs.forEach(h => {
      const choke = h.type === 'choke_point';
      L.circleMarker([h.lat, h.lon], {
        radius: choke ? 4 : 2, color: choke ? '#DCE6EE' : '#8FA3B5', weight: choke ? 1.5 : 1,
        fillColor: choke ? '#0B1622' : '#8FA3B5', fillOpacity: choke ? 1 : 0.5,
      }).bindTooltip(h.display_name).addTo(group);
    });
  }, [hubs]);

  useEffect(() => {
    const group = layers.current.routes;
    group.clearLayers();
    const byId = Object.fromEntries(hubs.map(h => [h.id, h]));
    let focus = null;
    let selectedLine = null;
    const exposures = [];
    let anchor = {}; // hub id -> position on the selected route's world copy

    routes.forEach((route, idx) => {
      const { points, positions } = routePositions(route, byId);
      if (points.length < 2) return;
      const isSelected = idx === selected;
      const line = L.polyline(points, {
        color: PERSONA_COLOURS[route.persona] || '#8FA3B5',
        weight: isSelected ? 4 : 2,
        opacity: isSelected ? 0.95 : 0.4,
        dashArray: isSelected ? null : '4 6',
      }).on('click', () => onSelect(idx))
        .bindTooltip(`${personaLabel(route)}: ${Math.round(route.adjusted_eta)} h typical`)
        .addTo(group);
      if (!isSelected) return;
      focus = line.getBounds();
      selectedLine = line;
      anchor = positions;

      // Legs exposed to a scenario or live news, drawn over the selected route
      // using the route's own (antimeridian-continued) positions.
      route.legs.forEach(leg => {
        const colour = EXPOSURE_COLOURS[leg.intel_source];
        if (!colour || leg.from === leg.to || !positions[leg.from] || !positions[leg.to]) return;
        const label = leg.intel_source === 'LIVE' ? 'Live news' : 'Scenario';
        exposures.push(L.polyline([positions[leg.from], positions[leg.to]], { color: colour, weight: 7, opacity: 0.55 })
          .bindTooltip(`${label}: ${leg.reason}`)
          .addTo(group));
      });
    });
    // Later options would otherwise be drawn over the selected one.
    selectedLine?.bringToFront();
    exposures.forEach(e => e.bringToFront());

    // Alert pulses sit on the selected route's world copy when the hub is on it.
    const addPulses = (ids, className, prefix) => ids.forEach(id => {
      const h = byId[id];
      if (h) pulse(className, `${prefix}: ${h.display_name}`, anchor[id] || [h.lat, h.lon]).addTo(group);
    });
    addPulses(disrupted, 'pulse-signal', 'Scenario disruption');
    addPulses(liveHubs, 'pulse-orchid', 'Live news');

    // Origin and destination of the selected route, labelled.
    const shown = routes[selected];
    if (shown && shown.legs.length) {
      [shown.legs[0].from, shown.legs[shown.legs.length - 1].to].forEach((id, i) => {
        const h = byId[id];
        if (!h) return;
        L.circleMarker(anchor[id] || [h.lat, h.lon], { radius: 6, color: '#DCE6EE', weight: 2, fillColor: '#0B1622', fillOpacity: 1 })
          .bindTooltip(h.display_name, { permanent: true, direction: i ? 'right' : 'left', className: 'endpoint' })
          .addTo(group);
      });
    }

    // Refit only when a different route is shown (new results or a new
    // selection), not when only the alerts change. That is also the one moment
    // the chart animates: the route is drawn from origin to destination, then
    // its exposed legs fade in.
    const running = [];
    if (focus && fitted.current !== routes[selected]) {
      fitted.current = routes[selected];
      map.current.fitBounds(focus, { padding: [30, 30], maxZoom: 5 });
      if (!prefersReducedMotion()) {
        running.push(animate(svg.createDrawable(selectedLine.getElement()), { draw: ['0 0', '0 1'], duration: 1400, ease: 'inOutQuad' }));
        const overlays = exposures.map(e => e.getElement());
        if (overlays.length) running.push(animate(overlays, { opacity: [0, 1], delay: 1100, duration: 500, ease: 'outQuad' }));
      }
    }
    // A redraw replaces these elements; stop animating the old ones.
    return () => running.forEach(a => a.revert());
  }, [routes, selected, hubs, onSelect, disrupted, liveHubs]);

  // The leg the user is pointing at in the voyage plan: its stretch of the
  // route, or the hub itself for a transfer.
  useEffect(() => {
    const group = layers.current.focus;
    group.clearLayers();
    const route = routes[selected];
    if (!focusLeg || !route) return;
    const { positions } = routePositions(route, Object.fromEntries(hubs.map(h => [h.id, h])));
    const [a, b] = [positions[focusLeg.from], positions[focusLeg.to]];
    if (!a || !b) return;
    (focusLeg.from === focusLeg.to
      ? L.circleMarker(a, { radius: 11, color: '#DCE6EE', weight: 2, fill: false, className: 'leg-focus' })
      : L.polyline([a, b], { color: '#DCE6EE', weight: 9, opacity: 0.5, className: 'leg-focus' })
    ).addTo(group);
  }, [focusLeg, routes, selected, hubs]);

  return <div ref={container} className="route-map" role="region" aria-label="Route map" />;
}
