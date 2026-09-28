import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

export const PERSONA_COLOURS = { FASTEST: '#f59e0b', SAFEST: '#10b981', BALANCED: '#3b82f6' };
const EXPOSURE_COLOURS = { SCENARIO: '#ef4444', LIVE: '#a855f7' };

// Keep a line continuous across the antimeridian: shift each longitude by
// +-360 so consecutive points are never more than 180 degrees apart, otherwise a
// transpacific leg would be drawn the long way round the world.
function unwrap(points) {
  const out = [];
  for (const [lat, lon] of points) {
    let l = lon;
    if (out.length) {
      const prev = out[out.length - 1][1];
      while (l - prev > 180) l -= 360;
      while (l - prev < -180) l += 360;
    }
    out.push([lat, l]);
  }
  return out;
}

function pulse(className, label) {
  return L.marker([0, 0], {
    icon: L.divIcon({ className: `pulse-marker ${className}`, iconSize: [14, 14] }),
    keyboard: false,
  }).bindTooltip(label);
}

export default function RouteMap({ hubs, routes, selected, onSelect, disrupted, liveHubs }) {
  const container = useRef(null);
  const map = useRef(null);
  const layers = useRef({});

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
      alerts: L.layerGroup().addTo(map.current),
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

    routes.forEach((route, idx) => {
      const ids = [route.legs[0]?.from, ...route.legs.map(l => l.to)].filter((id, i, a) => id && id !== a[i - 1]);
      const points = unwrap(ids.map(id => byId[id]).filter(Boolean).map(h => [h.lat, h.lon]));
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

      // Legs exposed to a scenario or live news, drawn over the selected route.
      route.legs.forEach(leg => {
        const colour = EXPOSURE_COLOURS[leg.intel_source];
        if (!colour || leg.from === leg.to || !byId[leg.from] || !byId[leg.to]) return;
        const segment = unwrap([[byId[leg.from].lat, byId[leg.from].lon], [byId[leg.to].lat, byId[leg.to].lon]]);
        L.polyline(segment, { color: colour, weight: 7, opacity: 0.55 })
          .bindTooltip(`${leg.intel_source}: ${leg.reason}`)
          .addTo(group);
      });
    });
    if (focus) map.current.fitBounds(focus, { padding: [30, 30], maxZoom: 5 });
  }, [routes, selected, hubs, onSelect]);

  useEffect(() => {
    const group = layers.current.alerts;
    group.clearLayers();
    const byId = Object.fromEntries(hubs.map(h => [h.id, h]));
    const add = (ids, className, prefix) => ids.forEach(id => {
      const h = byId[id];
      if (h) pulse(className, `${prefix}: ${h.display_name}`).setLatLng([h.lat, h.lon]).addTo(group);
    });
    add(disrupted, 'pulse-red', 'Scenario disruption');
    add(liveHubs, 'pulse-violet', 'Live news');
  }, [hubs, disrupted, liveHubs]);

  return <div ref={container} className="route-map" role="region" aria-label="Route map" />;
}
