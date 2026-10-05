import { useEffect, useMemo, useState } from 'react'
import { getStations, type Station } from './nearme'
import { SOCAL_LAND } from './socalOutline'

// Map frame (deg) and an equirectangular projection scaled by cos(mid-latitude) so shapes aren't stretched.
const LAT0 = 32.3, LAT1 = 36.9, LON0 = -121.3, LON1 = -114.0
const KX = Math.cos((((LAT0 + LAT1) / 2) * Math.PI) / 180)
const W = 600
const S = W / ((LON1 - LON0) * KX)
const H = Math.round((LAT1 - LAT0) * S)
const REACH_KM = 100 // a station "covers" a point within this distance (matches the live pipeline's design)
const CELL = 0.1 // shading grid (deg)

const px = (lon: number, lat: number): [number, number] => [(lon - LON0) * KX * S, (LAT1 - lat) * S]

function km(aLat: number, aLon: number, bLat: number, bLon: number) {
  const r = Math.PI / 180
  const s =
    Math.sin(((bLat - aLat) * r) / 2) ** 2 +
    Math.cos(aLat * r) * Math.cos(bLat * r) * Math.sin(((bLon - aLon) * r) / 2) ** 2
  return 2 * 6371 * Math.asin(Math.sqrt(s))
}

// [name, lat, lon, label side] -- 'L' puts the label left of the dot where a station label sits to the right
const CITIES: [string, number, number, 'L' | 'R'][] = [
  ['Los Angeles', 34.05, -118.24, 'R'], ['San Diego', 32.72, -117.16, 'L'], ['Bakersfield', 35.37, -119.02, 'L'],
  ['Palm Springs', 33.83, -116.55, 'R'], ['Santa Barbara', 34.42, -119.7, 'R'], ['Ridgecrest', 35.62, -117.67, 'L'],
  ['El Centro', 32.79, -115.56, 'R'],
]

// Where the network can see: darker = 3+ stations within 100 km (events there are located, sized and can
// alert); lighter = 2 (detected and logged, never alerted). Stations come from /api/stations, the same
// single list the live daemon uses.
export default function Coverage() {
  const [stations, setStations] = useState<Station[] | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    getStations().then(setStations).catch(() => setFailed(true))
  }, [])

  const land = useMemo(
    () => SOCAL_LAND.map((ring) => 'M' + ring.map(([lo, la]) => px(lo, la).map((v) => v.toFixed(1)).join(',')).join('L') + 'Z').join(''),
    [],
  )

  const cells = useMemo(() => {
    if (!stations) return { two: '', three: '' }
    let two = '', three = ''
    const w = CELL * KX * S, h = CELL * S
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

  return (
    <section className="coverage">
      <h2>Where it can see</h2>
      <p className="nearme-lede">
        The 19 live sensor stations and the area they cover. A quake is only confirmed, located and sized
        where at least three stations are within about 100 km. Lighter areas reach two stations, so quakes
        there are logged but never alerted.
      </p>
      <div className="card cov-card">
        <svg className="cov-map" viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label="Map of Southern California showing the 19 sensor stations and where three or more stations cover">
          <defs>
            <clipPath id="cov-frame"><rect x="0" y="0" width={W} height={H} /></clipPath>
          </defs>
          <g clipPath="url(#cov-frame)">
            <path className="cov-land" d={land} />
            <path className="cov-2" d={cells.two} />
            <path className="cov-3" d={cells.three} />
            <path className="cov-coast" d={land} />
            {CITIES.map(([name, la, lo, side]) => {
              const [x, y] = px(lo, la)
              return (
                <g key={name} className="cov-city">
                  <circle cx={x} cy={y} r={2.5} />
                  <text x={side === 'L' ? x - 6 : x + 5} y={y + 12} textAnchor={side === 'L' ? 'end' : 'start'}>{name}</text>
                </g>
              )
            })}
            {stations?.map((s) => {
              const [x, y] = px(s.lon, s.lat)
              return (
                <g key={s.code} className="cov-sta">
                  <circle cx={x} cy={y} r={4.5} />
                  <text x={x + 7} y={y - 5}>{s.code}</text>
                </g>
              )
            })}
          </g>
        </svg>
        <div className="cov-legend">
          <span><i className="sw sw-3" />3+ stations: located, sized, can alert</span>
          <span><i className="sw sw-2" />2 stations: logged only</span>
          <span><i className="sw sw-sta" />live station</span>
        </div>
        {failed && <p className="muted cov-note">Couldn’t load the station list — the map shows the coastline only.</p>}
      </div>
    </section>
  )
}
