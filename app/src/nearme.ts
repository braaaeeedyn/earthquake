// Near-me subscribe API + largest-quake archive — talks to scripts/server.py via the Vite /api proxy.

export interface UsgsEvent {
  id: string
  mag: number
  lat: number
  lon: number
  place: string
  url: string
  time?: number
}

// One day's top-5 largest quakes, worldwide and in the California "area".
export interface DayTop {
  date: string
  world: UsgsEvent[]
  area: UsgsEvent[]
}

export async function subscribe(sub: { name: string; email: string; lat: number; lon: number }) {
  const res = await fetch('/api/subscribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sub),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
  return data as { ok: boolean; count: number; note?: string }
}

// Today's live top-5 (also folds the current USGS feed into the server-side daily archive).
export async function todayTop(): Promise<DayTop> {
  const res = await fetch('/api/events')
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as DayTop
}

// The list of dates the archive holds (newest first).
export async function archiveDates(): Promise<string[]> {
  const res = await fetch('/api/archive')
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()).dates as string[]
}

// A specific past day's top-5.
export async function archiveDay(date: string): Promise<DayTop> {
  const res = await fetch(`/api/archive?date=${encodeURIComponent(date)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as DayTop
}

// Top-5 largest California quakes for a named time window (day/week/month/year/all).
export interface CaWindow {
  window: string
  label: string
  events: UsgsEvent[]
}

export async function caTop(window: string): Promise<CaWindow> {
  const res = await fetch(`/api/ca?window=${encodeURIComponent(window)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as CaWindow
}

// Turn a place name into coordinates (server-side geocode).
export async function geocode(q: string): Promise<{ lat: number; lon: number; name: string }> {
  const res = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
  return data as { lat: number; lon: number; name: string }
}
