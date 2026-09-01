// Near-me subscribe API + largest-quake archive — talks to scripts/server.py.
// On web the Vite dev server proxies /api -> :8000. In the packaged Android app there is no
// proxy, so calls must hit an absolute base: the emulator reaches the host at 10.0.2.2, and a
// real device / deployment sets VITE_API_BASE at build time.

import { Capacitor } from '@capacitor/core'

const API_BASE = Capacitor.isNativePlatform()
  ? ((import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://10.0.2.2:8000')
  : ''

const api = (path: string) => API_BASE + path

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

// Register this device's FCM token + location for push alerts (mobile app only).
export async function registerPushToken(p: { token: string; lat: number; lon: number; name: string }) {
  const res = await fetch(api('/api/register-push'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(p),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
  return data as { ok: boolean; count: number }
}

// Unregister this device's FCM token — removes it from the alert list (unsubscribe).
export async function unregisterPushToken(p: { token: string }) {
  const res = await fetch(api('/api/unregister-push'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(p),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
  return data as { ok: boolean; count: number }
}

// Relay a support message to the (hidden) support inbox. The user's email is used only as
// Reply-To on the server and is never stored; the destination address is never sent to the client.
export async function sendContact(p: { email: string; message: string }) {
  const res = await fetch(api('/api/contact'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(p),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
  return data as { ok: boolean }
}

// Today's live top-5 (also folds the current USGS feed into the server-side daily archive).
export async function todayTop(): Promise<DayTop> {
  const res = await fetch(api('/api/events'))
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as DayTop
}

// The list of dates the archive holds (newest first).
export async function archiveDates(): Promise<string[]> {
  const res = await fetch(api('/api/archive'))
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()).dates as string[]
}

// A specific past day's top-5.
export async function archiveDay(date: string): Promise<DayTop> {
  const res = await fetch(api(`/api/archive?date=${encodeURIComponent(date)}`))
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
  const res = await fetch(api(`/api/ca?window=${encodeURIComponent(window)}`))
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as CaWindow
}

// Whether the live SeedLink watcher is connected (drives the green "live" dot in the nav).
export async function liveStatus(): Promise<boolean> {
  try {
    const res = await fetch(api('/api/status'))
    if (!res.ok) return false
    return Boolean((await res.json()).live)
  } catch {
    return false
  }
}

// Turn a place name into coordinates (server-side geocode).
export async function geocode(q: string): Promise<{ lat: number; lon: number; name: string }> {
  const res = await fetch(api(`/api/geocode?q=${encodeURIComponent(q)}`))
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
  return data as { lat: number; lon: number; name: string }
}
