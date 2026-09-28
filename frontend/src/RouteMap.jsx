import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

export const PERSONA_COLOURS = { FASTEST: '#f59e0b', SAFEST: '#10b981', BALANCED: '#3b82f6' };
const EXPOSURE_COLOURS = { SCENARIO: '#ef4444', LIVE: '#a855f7' };

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

export default function RouteMap({ hubs, routes, selected, onSelect, disrupted, liveHubs }) {
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
        radius: choke ? 4 : 2, color: choke ? '#f59e0b' : '#475569', weight: 1,
        fillColor: choke ? '#f59e0b' : '#64748b', fillOpacity: choke ? 0.9 : 0.6,
      }).bindTooltip(h.display_name).addTo(group);
    });
  }, [hubs]);

  useEffect(() => {
    const group = layers.current.routes;
    group.clearLayers();
    const byId = Object.fromEntries(hubs.map(h => [h.id, h]));
    let focus = null;
    let anchor = {}; // hub id -> position on the selected route's world copy

    routes.forEach((route, idx) => {
      const { points, positions } = routePositions(route, byId);
      if (points.length < 2) return;
      const isSelected = idx === selected;
      const line = L.polyline(points, {
        color: PERSONA_COLOURS[route.persona] || '#94a3b8',
        weight: isSelected ? 4 : 2,
        opacity: isSelected ? 0.95 : 0.35,
        dashArray: isSelected ? null : '4 6',
      }).on('click', () => onSelect(idx))
        .bindTooltip(`${(route.personas || [route.persona]).join(' · ')} · ${Math.round(route.adjusted_eta)}h`)
        .addTo(group);
      if (!isSelected) return;
      focus = line.getBounds();
      anchor = positions;

      // Legs exposed to a scenario or live news, drawn over the selected route
      // using the route's own (antimeridian-continued) positions.
      route.legs.forEach(leg => {
        const colour = EXPOSURE_COLOURS[leg.intel_source];
        if (!colour || leg.from === leg.to || !positions[leg.from] || !positions[leg.to]) return;
        L.polyline([positions[leg.from], positions[leg.to]], { color: colour, weight: 7, opacity: 0.55 })
          .bindTooltip(`${leg.intel_source}: ${leg.reason}`)
          .addTo(group);
      });
    });

    // Alert pulses sit on the selected route's world copy when the hub is on it.
    const addPulses = (ids, className, prefix) => ids.forEach(id => {
      const h = byId[id];
      if (h) pulse(className, `${prefix}: ${h.display_name}`, anchor[id] || [h.lat, h.lon]).addTo(group);
    });
    addPulses(disrupted, 'pulse-red', 'Scenario disruption');
    addPulses(liveHubs, 'pulse-violet', 'Live news');

    // Refit only when a different route is shown (new results or a new
    // selection), not when only the alerts change.
    if (focus && fitted.current !== routes[selected]) {
      fitted.current = routes[selected];
      map.current.fitBounds(focus, { padding: [30, 30], maxZoom: 5 });
    }
  }, [routes, selected, hubs, onSelect, disrupted, liveHubs]);

  return <div ref={container} className="route-map" role="region" aria-label="Route map" />;
}
