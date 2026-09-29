// Turning a failed API response into a sentence for the user, shared by the
// route planner and the supplier page.

// Request fields, as the user knows them from the form.
const FIELD_NAMES = {
  source: 'origin', destination: 'destination', transport_preference: 'transport mode', routing_policy: 'mode policy',
  cargo_type: 'cargo type', priority: 'priority', scenario: 'disruption scenario', avoid_chokepoints: 'avoided chokepoints',
  live_intel: 'live news setting', category: 'product category', current_inventory: 'current inventory',
  safety_stock: 'safety stock target', demand_forecast: 'demand forecast',
};

// The body of a response, or null when it isn't JSON (a proxy error page, say).
export const readJson = res => res.json().catch(() => null);

export function requestError(res, data) {
  if (res.status === 429) {
    const wait = Number(res.headers.get('Retry-After'));
    return `Too many requests. Try again in ${wait > 0 ? `${wait} second${wait === 1 ? '' : 's'}` : 'a minute'}.`;
  }
  if (res.status === 401) {
    return 'The server requires an API key that this page does not send. Start the dashboard as the README describes.';
  }
  if (data && typeof data.error === 'string') return data.error; // the engine's own reason, already plain
  if (data && Array.isArray(data.detail)) {
    // Validation errors name fields by their API names; say which inputs they are.
    const fields = [...new Set(data.detail.map(d => FIELD_NAMES[(d.loc || []).find(k => FIELD_NAMES[k])] || 'input'))];
    return `The server did not accept the ${fields.join(' and ')}. Check ${fields.length > 1 ? 'them' : 'it'} and try again.`;
  }
  return `The request failed (HTTP ${res.status}). Try again in a moment.`;
}
