// Forecast contract — must match scripts/make_forecast.py and README "Forecast contract".
export interface Forecast {
  schema_version: number
  region: string
  generated_at: string
  horizon_days: number
  probability: number
  label: 'likely' | 'unlikely'
  confidence: 'low' | 'medium' | 'high'
  is_placeholder: boolean
  model_version: string
  stations: string[]
  disclaimer: string
}

// Relative URL (no leading slash) so it resolves under base './' on web and in the
// Capacitor WebView. Cache-busted so a refreshed scheduled inference is picked up.
export async function loadForecast(): Promise<Forecast> {
  const res = await fetch(`forecast.json?t=${Date.now()}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`Failed to load forecast (HTTP ${res.status})`)
  return (await res.json()) as Forecast
}
