// Estimated shaking (Modified Mercalli intensity, MMI) at the user's home, computed ON THE DEVICE with the same
// equations as the server (src/eq/shaking.py; coefficients from /api/shaking-model):
//   log10(PGV m/s) = a + b*M + c*log10(R) + d*R + e*log10(Vs30/Vs30_ref) + eventTerm  ->  MMI (Worden 2012)
// Vs30 (ground stiffness) at the home comes from vs30_socal.json (USGS map). The home location never leaves
// the device: it is kept in localStorage and Capacitor Preferences (so the native notification code can read it).
import { Preferences } from '@capacitor/preferences'
import { api } from './nearme'

export interface ShakingModel {
  gmpe: { a: number; b: number; c: number; d: number }
  site: { vs30_ref: number; e: number }
  worden: { lo: [number, number]; hi: [number, number]; brk: number }
  mmi_offset?: number
  levels: { mmi: number; label: string; desc: string }[]
  residual_sd_log10?: number | null
  validation?: { mae: number; within1: number; n_cells: number; n_quakes: number } | null
}
interface Vs30Grid { lat0: number; lon0: number; step: number; nlat: number; nlon: number; vs30: number[] }
export interface Home { lat: number; lon: number; label: string; vs30: number | null }

const HOME_KEY = 'seismic.home'
let modelP: Promise<ShakingModel> | null = null
let gridP: Promise<Vs30Grid> | null = null

export function loadShakingModel(): Promise<ShakingModel> {
  modelP ??= fetch(api('/api/shaking-model')).then(async (r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const m = (await r.json()) as ShakingModel
    // the native notification code (app/native/android) reads the same coefficients from here
    try { await Preferences.set({ key: 'seismic.shaking_model', value: JSON.stringify(m) }) } catch { /* web */ }
    return m
  })
  return modelP
}

function loadGrid(): Promise<Vs30Grid> {
  gridP ??= fetch('vs30_socal.json').then((r) => r.json() as Promise<Vs30Grid>)
  return gridP
}

export async function vs30At(lat: number, lon: number): Promise<number | null> {
  const g = await loadGrid()
  const i = Math.round((lat - g.lat0) / g.step), j = Math.round((lon - g.lon0) / g.step)
  if (i < 0 || j < 0 || i >= g.nlat || j >= g.nlon) return null           // outside Southern California
  return g.vs30[i * g.nlon + j] || null
}

export function kmBetween(aLat: number, aLon: number, bLat: number, bLon: number) {
  const R = 6371, rad = (d: number) => (d * Math.PI) / 180
  const s = Math.sin(rad(bLat - aLat) / 2) ** 2 + Math.cos(rad(aLat)) * Math.cos(rad(bLat)) * Math.sin(rad(bLon - aLon) / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(s))
}

export function estimateMMI(m: ShakingModel, mag: number, distKm: number, vs30: number | null, eventTerm = 0) {
  const r = Math.max(distKm, 1)
  let logPgv = m.gmpe.a + m.gmpe.b * mag + m.gmpe.c * Math.log10(r) + m.gmpe.d * r + eventTerm
  if (vs30) logPgv += m.site.e * Math.log10(vs30 / m.site.vs30_ref)
  const y = logPgv + 2                                                      // log10(PGV in cm/s)
  const [a, b] = y > m.worden.brk ? m.worden.hi : m.worden.lo
  return Math.max(1, Math.min(10, a + b * y + (m.mmi_offset ?? 0)))   // + DYFI-calibrated offset (validate_mmi.py)
}

const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
export function describeMMI(m: ShakingModel, mmi: number) {
  const lvl = Math.max(1, Math.min(10, Math.round(mmi)))
  const L = m.levels[lvl - 1]
  return { roman: ROMAN[lvl - 1], label: L.label, desc: L.desc }
}

export function getHome(): Home | null {
  try {
    const raw = localStorage.getItem(HOME_KEY)
    return raw ? (JSON.parse(raw) as Home) : null
  } catch {
    return null
  }
}

export async function setHome(lat: number, lon: number, label: string) {
  const home: Home = { lat: +lat.toFixed(3), lon: +lon.toFixed(3), label, vs30: await vs30At(lat, lon) }
  try { localStorage.setItem(HOME_KEY, JSON.stringify(home)) } catch { /* storage unavailable */ }
  try { await Preferences.set({ key: HOME_KEY, value: JSON.stringify(home) }) } catch { /* web or no plugin */ }
  window.dispatchEvent(new CustomEvent('home-changed'))
  return home
}

export async function clearHome() {
  try { localStorage.removeItem(HOME_KEY) } catch { /* ignore */ }
  try { await Preferences.remove({ key: HOME_KEY }) } catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent('home-changed'))
}

// One line for a quake: "Light (IV) at your home, 34 km away", or null when no home is set.
export async function homeShaking(lat: number, lon: number, mag: number, eventTerm = 0) {
  const home = getHome()
  if (!home) return null
  const m = await loadShakingModel()
  const km = kmBetween(home.lat, home.lon, lat, lon)
  const mmi = estimateMMI(m, mag, km, home.vs30, eventTerm)
  return { ...describeMMI(m, mmi), mmi, km, home }
}
