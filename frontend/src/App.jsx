import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import RouteRecommender, { RECENT_KEY } from './RouteRecommender.jsx';
import SupplierIntelligence from './SupplierIntelligence.jsx';

// The charts library is only needed on this page, so it loads on first visit.
const ModelEvaluation = lazy(() => import('./ModelEvaluation.jsx'));

// Each page has an address (#model, #suppliers), so Back, Forward and reload
// keep you where you were.
const VIEWS = { '': 'recommend', '#model': 'model', '#suppliers': 'suppliers' };
const TITLES = { recommend: 'Supplychainer', model: 'Model evaluation: Supplychainer', suppliers: 'Supplier intelligence: Supplychainer' };
const viewFromHash = () => VIEWS[window.location.hash] || 'recommend';

// A rendering error would otherwise leave a blank page. Saved plans are the
// likeliest cause, so the reset clears them.
class ErrorBoundary extends React.Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  reset = () => {
    try {
      localStorage.removeItem(RECENT_KEY);
    } catch {
      // Storage unavailable: nothing saved to clear.
    }
    window.location.hash = '';
    window.location.reload();
  };

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="crash" role="alert">
        <h1>Something went wrong</h1>
        <p>The dashboard hit an error it could not recover from. Resetting clears the recent plans saved in this browser and reloads the page.</p>
        <button type="button" className="primary" onClick={this.reset}>Reset and reload</button>
      </main>
    );
  }
}

export default function App() {
  const [view, setView] = useState(viewFromHash);
  const [visited, setVisited] = useState(() => new Set([viewFromHash()]));
  const [engineStatus, setEngineStatus] = useState(null);
  const pages = useRef({});

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  // Pages stay mounted once opened, so planned routes and form inputs survive a
  // visit to another page. Focus moves to the page that is now shown.
  const firstView = useRef(true);
  useEffect(() => {
    setVisited(prev => (prev.has(view) ? prev : new Set(prev).add(view)));
    document.title = TITLES[view];
    if (firstView.current) {
      firstView.current = false;
      return;
    }
    window.scrollTo(0, 0);
    pages.current[view]?.focus();
  }, [view]);

  const navigate = next => {
    window.location.hash = next === 'recommend' ? '' : next;
  };

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
    return () => {
      closed = true;
      clearTimeout(retry);
      ws.onmessage = null;
      ws.onclose = null;
      // Closing a socket that is still connecting makes the browser log an error
      // (React's development mode mounts twice, which does this); close it once open.
      if (ws.readyState === WebSocket.CONNECTING) ws.onopen = () => ws.close();
      else ws.close();
    };
  }, []);

  const page = (name, content) => visited.has(name) && (
    <div hidden={view !== name} tabIndex={-1} ref={el => { pages.current[name] = el; }} className="view">
      {content}
    </div>
  );

  return (
    <ErrorBoundary>
      {page('recommend', <RouteRecommender onNavigate={navigate} engineStatus={engineStatus} visible={view === 'recommend'} />)}
      {page('model', (
        <Suspense fallback={<p className="muted" style={{ padding: '1.5rem' }}>Loading the evaluation…</p>}>
          <ModelEvaluation onNavigate={navigate} />
        </Suspense>
      ))}
      {page('suppliers', <SupplierIntelligence onNavigate={navigate} />)}
    </ErrorBoundary>
  );
}
