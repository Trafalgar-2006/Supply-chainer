import React, { useState, useEffect } from 'react';
import RouteRecommender from './RouteRecommender.jsx';
import SupplierIntelligence from './SupplierIntelligence.jsx';
import ModelEvaluation from './ModelEvaluation.jsx';

export default function App() {
  const [currentView, setCurrentView] = useState('recommend');
  const [engineStatus, setEngineStatus] = useState(null);

  useEffect(() => {
    // The backend pushes the engine's warm-up status every two seconds; reconnect
    // if the server restarts.
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let ws;
    let retry;
    let closed = false;
    const connect = () => {
      ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
      ws.onmessage = event => setEngineStatus(JSON.parse(event.data).engine_status);
      ws.onclose = () => {
        setEngineStatus('ENGINE OFFLINE');
        if (!closed) retry = setTimeout(connect, 3000);
      };
    };
    connect();
    return () => { closed = true; clearTimeout(retry); ws.close(); };
  }, []);

  if (currentView === 'suppliers') {
    return <SupplierIntelligence onNavigate={setCurrentView} />;
  }
  if (currentView === 'model') {
    return <ModelEvaluation onNavigate={setCurrentView} />;
  }
  return <RouteRecommender onNavigate={setCurrentView} engineStatus={engineStatus} />;
}
