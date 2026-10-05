import { useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import { getStations, type Station } from './nearme'
import { SOCAL_LAND } from './socalOutline'

// Map frame (deg) and an equirectangular projection scaled by cos(mid-latitude) so shapes aren't stretched.
const LAT0 = 32.3, LAT1 = 36.9, LON0 = -121.3, LON1 = -114.0
const KX = Math.cos((((LAT0 + LAT1) / 2) * Math.PI) / 180)
const W = 600
const S = W / ((LON1 - LON0) * KX)
const H = Math.round((LAT1 - LAT0) * S)
const REACH_KM = 100 // a station "covers" a point within this distance (the confirmation rule's coverage)
const CELL = 0.1 // shading grid (deg)
const ZMAX = 8 // max zoom-in; zoom 1 = the whole frame (max zoom-out)
const TIER_ZOOM = { 1: 1, 2: 1.8, 3: 3.2 } as const // cities of a tier appear from this zoom
const OUTLINE_ZOOM = 2 // dotted city boundaries appear from this zoom

const px = (lon: number, lat: number): [number, number] => [(lon - LON0) * KX * S, (LAT1 - lat) * S]

function km(aLat: number, aLon: number, bLat: number, bLon: number) {
  const r = Math.PI / 180
  const s =
    Math.sin(((bLat - aLat) * r) / 2) ** 2 +
    Math.cos(aLat * r) * Math.cos(bLat * r) * Math.sin(((bLon - aLon) * r) / 2) ** 2
  return 2 * 6371 * Math.asin(Math.sqrt(s))
}

// City outlines + label points (U.S. Census 2023 place boundaries, simplified), loaded on first render.
interface City { name: string; tier: 1 | 2 | 3; lon: number; lat: number; rings: [number, number][][] }

const ringPath = (rings: [number, number][][]) =>
  rings.map((r) => 'M' + r.map(([lo, la]) => px(lo, la).map((v) => v.toFixed(1)).join(',')).join('L') + 'Z').join('')

interface View { x: number; y: number; z: number } // top-left of the visible box + zoom
const clampView = (v: View): View => {
  const z = Math.min(ZMAX, Math.max(1, v.z))
  const w = W / z, h = H / z
  return { z, x: Math.min(W - w, Math.max(0, v.x)), y: Math.min(H - h, Math.max(0, v.y)) }
}

// Where the network can see: darker = 3+ stations within 100 km (events there are located, sized and can
// alert); lighter = 2 (detected and logged, never alerted). Zoom in for more cities and their boundaries.
export default function Coverage() {
  const [stations, setStations] = useState<Station[] | null>(null)
  const [cities, setCities] = useState<City[]>([])
  const [failed, setFailed] = useState(false)
  const [view, setView] = useState<View>({ x: 0, y: 0, z: 1 })
  const svgRef = useRef<SVGSVGElement>(null)
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const gesture = useRef<{ view: View; dist?: number; cx?: number; cy?: number } | null>(null)

  useEffect(() => {
    getStations().then(setStations).catch(() => setFailed(true))
    fetch('socal_cities.json').then((r) => r.json()).then(setCities).catch(() => setCities([]))
  }, [])

  const land = useMemo(() => ringPath(SOCAL_LAND), [])
  const cityPaths = useMemo(() => cities.map((c) => ({ ...c, d: ringPath(c.rings), at: px(c.lon, c.lat) })), [cities])

  const cells = useMemo(() => {
    if (!stations) return { two: '', three: '' }
    let two = '', three = ''
    // cells are opaque and overlap by a hair so no sub-pixel seams show between them at high zoom
    const w = CELL * KX * S + 0.4, h = CELL * S + 0.4
    for (let la = LAT0; la < LAT1; la += CELL) {
      for (let lo = LON0; lo < LON1; lo += CELL) {
        const n = stations.filter((s) => km(la + CELL / 2, lo + CELL / 2, s.lat, s.lon) <= REACH_KM).length
        if (n < 2) continue
        const [x, y] = px(lo, la + CELL)
        const d = `M${x.toFixed(1)},${y.toFixed(1)}h${w.toFixed(1)}v${h.toFixed(1)}h${(-w).toFixed(1)}z`
        if (n >= 3) three += d
        else two += d
      }
    }
    return { two, three }
  }, [stations])

  // screen point -> SVG (viewBox) point
  const toSvg = (clientX: number, clientY: number, v: View) => {
    const r = svgRef.current!.getBoundingClientRect()
    return { x: v.x + ((clientX - r.left) / r.width) * (W / v.z), y: v.y + ((clientY - r.top) / r.height) * (H / v.z) }
  }
  // zoom by `f` keeping the SVG point (px, py) fixed under the cursor
  const zoomAt = (v: View, f: number, p: { x: number; y: number }) => {
    const z = Math.min(ZMAX, Math.max(1, v.z * f))
    const k = v.z / z
    return clampView({ z, x: p.x - (p.x - v.x) * k, y: p.y - (p.y - v.y) * k })
  }

  // wheel zoom needs a non-passive listener to stop the page from scrolling while zooming the map
  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      setView((v) => zoomAt(v, Math.exp(-e.deltaY * 0.0015), toSvg(e.clientX, e.clientY, v)))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  const onDown = (e: RPointerEvent<SVGSVGElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    gesture.current = { view }
  }
  const onMove = (e: RPointerEvent<SVGSVGElement>) => {
    if (!pointers.current.has(e.pointerId) || !gesture.current) return
    const prev = pointers.current.get(e.pointerId)!
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    const pts = [...pointers.current.values()]
    if (pts.length === 2) {
      // pinch: scale by the change in finger distance around their midpoint
      const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y)
      const g = gesture.current
      if (!g.dist) {
        g.dist = d
        g.cx = (pts[0].x + pts[1].x) / 2
        g.cy = (pts[0].y + pts[1].y) / 2
        g.view = view
        return
      }
      setView(zoomAt(g.view, d / g.dist, toSvg(g.cx!, g.cy!, g.view)))
      return
    }
    if (view.z === 1) return // nothing to pan at full extent
    const r = svgRef.current!.getBoundingClientRect()
    const dx = ((e.clientX - prev.x) / r.width) * (W / view.z)
    const dy = ((e.clientY - prev.y) / r.height) * (H / view.z)
    setView((v) => clampView({ ...v, x: v.x - dx, y: v.y - dy }))
  }
  const onUp = (e: RPointerEvent<SVGSVGElement>) => {
    pointers.current.delete(e.pointerId)
    if (pointers.current.size < 2 && gesture.current) gesture.current.dist = undefined
    if (pointers.current.size === 0) gesture.current = null
  }
  const zoomButton = (f: number) => setView((v) => zoomAt(v, f, { x: v.x + W / v.z / 2, y: v.y + H / v.z / 2 }))

  const z = view.z
  const u = 1 / z // keeps dots/labels/lines a constant on-screen size
  const shown = cityPaths.filter((c) => z >= TIER_ZOOM[c.tier])

  return (
    <section className="coverage">
      <h2>Where it can see</h2>
      <p className="nearme-lede">
        The 19 live sensor stations and the area they cover. A quake is only confirmed, located and sized
        where at least three stations are within about 100 km. Lighter areas reach two stations, so quakes
        there are logged but never alerted. Scroll or pinch to zoom in for more cities and their boundaries.
      </p>
      <div className="card cov-card">
        <div className="cov-frame">
          <svg ref={svgRef} className={`cov-map${z > 1 ? ' zoomed' : ''}`} viewBox={`${view.x} ${view.y} ${W / z} ${H / z}`}
            role="img" aria-label="Interactive map of Southern California showing the 19 sensor stations and their coverage"
            onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
            <defs>
              <clipPath id="cov-frame"><rect x="0" y="0" width={W} height={H} /></clipPath>
            </defs>
            <g clipPath="url(#cov-frame)">
              <path className="cov-land" d={land} />
              <path className="cov-2" d={cells.two} />
              <path className="cov-3" d={cells.three} />
              <path className="cov-coast" d={land} strokeWidth={0.8 * u} />
              {z >= OUTLINE_ZOOM && shown.map((c) => (
                <path key={`o-${c.name}`} className="cov-city-outline" d={c.d} strokeWidth={0.9 * u}
                  strokeDasharray={`${2 * u} ${2 * u}`} />
              ))}
              {shown.map((c) => (
                <g key={c.name} className={`cov-city tier-${c.tier}`}>
                  <circle cx={c.at[0]} cy={c.at[1]} r={2.5 * u} strokeWidth={u} />
                  <text x={c.at[0] + 5 * u} y={c.at[1] + 12 * u} fontSize={11 * u} strokeWidth={3 * u}>{c.name}</text>
                </g>
              ))}
              {stations?.map((s) => {
                const [x, y] = px(s.lon, s.lat)
                return (
                  <g key={s.code} className="cov-sta">
                    <circle cx={x} cy={y} r={4.5 * u} strokeWidth={1.5 * u} />
                    <text x={x + 7 * u} y={y - 5 * u} fontSize={10 * u} strokeWidth={3 * u}>{s.code}</text>
                  </g>
                )
              })}
            </g>
          </svg>
          <div className="cov-zoom" role="group" aria-label="Map zoom">
            <button type="button" onClick={() => zoomButton(1.6)} disabled={z >= ZMAX} aria-label="Zoom in">+</button>
            <button type="button" onClick={() => zoomButton(1 / 1.6)} disabled={z <= 1} aria-label="Zoom out">−</button>
            <button type="button" onClick={() => setView({ x: 0, y: 0, z: 1 })} disabled={z <= 1} aria-label="Reset map"
              className="cov-reset">Reset</button>
          </div>
        </div>
        <div className="cov-legend">
          <span><i className="sw sw-3" />3+ stations: located, sized, can alert</span>
          <span><i className="sw sw-2" />2 stations: logged only</span>
          <span><i className="sw sw-sta" />live station</span>
          {z >= OUTLINE_ZOOM && <span><i className="sw sw-city" />city boundary</span>}
        </div>
        {failed && <p className="muted cov-note">Couldn’t load the station list — the map shows the coastline only.</p>}
      </div>
    </section>
  )
}
