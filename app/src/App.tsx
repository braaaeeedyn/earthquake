import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import { loadSeismic, type Seismic, type Task } from './seismic'
import { subscribe, caTop, geocode, liveStatus, type UsgsEvent, type CaWindow } from './nearme'
import { enablePush, initPush, isNativeApp } from './push'

type State =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: Seismic }

// Detect -> Size -> Warn is a real pipeline, so the numbering carries meaning.
const ROWS = [
  {
    n: '01', kicker: 'Detect', key: 'detection', q: 'Is it an earthquake?', figure: 'detection_demo.png',
    desc: 'Give it 30 seconds of shaking recorded by a sensor and it decides whether a real earthquake is happening or whether it is just ordinary background noise like traffic or wind. It has learned what genuine quakes look like, so it spots ones that the older, simpler alarm would miss. On earthquakes it had never seen before, it makes the right call about 98% of the time.',
    tech: '',
  },
  {
    n: '02', kicker: 'Size', key: 'magnitude', q: 'How big is it?', figure: 'magnitude_demo.png',
    desc: 'It estimates the size, or magnitude, of the earthquake by looking at the readings from many sensor stations at the same time. Combining the whole network of stations is what makes the estimate accurate. Relying on only the one station nearest the quake barely works.',
    tech: 'Technical: network-magnitude regression. Per-station 3-component waveforms feed a CNN feature extractor, then a graph convolution across the 10-station network, then a transformer, then a magnitude head. The 5-seed ensemble scores R² = 0.846 (MAE 0.15) versus an amplitude + distance linear baseline at R² = 0.690. A nearest-single-station ablation collapses to R² −0.17, isolating the multi-station graph fusion as the source of the skill.',
  },
  {
    n: '03', kicker: 'Warn', key: 'eew_alert', q: 'How hard will it shake?', figure: 'eew_demo.png',
    desc: 'From just the first 8 seconds of an earthquake, it predicts how hard the ground will shake a few moments later, before the strong shaking reaches you, and decides whether to sound an alarm. This early warning can give people seconds to take cover. It correctly warns about 76% of the truly dangerous, strong-shaking earthquakes, and to catch that many the older method sets off twice as many false alarms.',
    tech: 'Technical: early-warning PGV regression from the first 8 s (≈5 s pre-P + 3 s early P-wave). A per-station CNN → graph convolution → transformer predicts future log₁₀(peak ground velocity), reported as a 5-model seed ensemble. The alert trigger is tuned on a chronological validation split with a recall-weighted F2 objective (the "strong shaking" definition is fixed at the train 70th-percentile PGV). Held-out: recall 0.76, precision 0.82, MCC +0.760 versus a GMPE-style baseline MCC +0.655; continuous R² 0.728 ≈ baseline 0.720.',
  },
]

export default function App() {
  const [state, setState] = useState<State>({ status: 'loading' })
  const [live, setLive] = useState(false)

  useEffect(() => {
    let active = true
    loadSeismic()
      .then((data) => active && setState({ status: 'ready', data }))
      .catch((e) => active && setState({ status: 'error', message: String(e?.message ?? e) }))
    initPush()          // register foreground push handlers (no-op on web)
    const checkLive = () => liveStatus().then((v) => active && setLive(v))
    checkLive()
    const id = setInterval(checkLive, 15000)   // poll the SeedLink watcher status
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  return (
    <div className="app">
      <nav className="nav">
        <span className="brand">SeismicSoCal</span>
        <span className="live">
          <span className={`dot ${live ? 'on' : ''}`} aria-hidden /> {live ? 'live · SeedLink' : 'offline · SeedLink'}
        </span>
      </nav>

      <main className="content">
        {state.status === 'loading' && <p className="state">Loading…</p>}
        {state.status === 'error' && (
          <div className="state">
            <h2>Couldn’t load results</h2>
            <p className="muted">{state.message}</p>
            <p className="muted">Ensure <code>seismic.json</code> is in <code>public/</code>.</p>
          </div>
        )}
        {state.status === 'ready' && <Console data={state.data} />}
      </main>

      <footer className="footer">
        Research &amp; education · real Southern California network data · not an official warning system
      </footer>
    </div>
  )
}

function Console({ data }: { data: Seismic }) {
  const d = data.dataset
  return (
    <>
      <section className="hero">
        <p className="eyebrow">Earthquake ML · Southern California</p>
        <h1>SeismicSoCal</h1>
        <p className="hero-sub">
          Three deep models on real held-out waveforms, each tested against the classic
          seismology baseline.
        </p>
        <Trace />
        <div className="meta">
          <span>{d.events.toLocaleString()} events</span>
          <span>{d.stations} stations</span>
          <span>M {d.mag_min}–{d.mag_max}</span>
          <span>out-of-sample</span>
        </div>
      </section>

      <Carousel data={data} />

      <NearMe />

      <CaLargest />
    </>
  )
}

// The Detect -> Size -> Warn sequence as an auto-cycling carousel.
// Tracks the previous index so the outgoing card slides left and the incoming enters from the right.
function Carousel({ data }: { data: Seismic }) {
  const byKey = Object.fromEntries(data.tasks.map((t) => [t.key, t]))
  const rows = ROWS.filter((r) => byKey[r.key])
  const n = rows.length
  const CYCLE = 7
  const [idx, setIdx] = useState<{ active: number; prev: number; dir: 'next' | 'prev' }>({ active: 0, prev: 0, dir: 'next' })
  const [paused, setPaused] = useState(false)
  const [remaining, setRemaining] = useState(CYCLE)
  const [evOpen, setEvOpen] = useState(false) // Evidence is shared: one toggle opens all three

  // One 1-second ticker drives both the countdown display and the 7s auto-advance.
  useEffect(() => {
    if (paused || n < 2) return
    const id = setInterval(() => {
      setRemaining((r) => {
        if (r <= 1) {
          setIdx((s) => ({ active: (s.active + 1) % n, prev: s.active, dir: 'next' }))
          return CYCLE
        }
        return r - 1
      })
    }, 1000)
    return () => clearInterval(id)
  }, [paused, n])

  const go = (i: number, dir: 'next' | 'prev' = 'next') => {
    setIdx((s) => ({ active: ((i % n) + n) % n, prev: s.active, dir }))
    setRemaining(CYCLE)
  }
  const cls = (i: number) =>
    i === idx.active ? 'is-active' : i === idx.prev ? 'is-leaving' : 'is-rest'

  return (
    <section className="carousel" aria-roledescription="carousel" aria-label="Detect, Size, Warn">
      <div className="carousel-tabs" role="tablist">
        {rows.map((r, i) => (
          <button
            key={r.key}
            role="tab"
            aria-selected={idx.active === i}
            className={`carousel-tab ${idx.active === i ? 'on' : ''}`}
            onClick={() => go(i, i < idx.active ? 'prev' : 'next')}
          >
            <span className="carousel-tab-n">{r.n}</span>
            {r.kicker}
          </button>
        ))}
      </div>

      <div className={`carousel-stage dir-${idx.dir}`}>
        {rows.map((r, i) => (
          <div key={r.key} className={`carousel-card ${cls(i)}`} aria-hidden={idx.active !== i}>
            <Result row={r} t={byKey[r.key]} evOpen={evOpen} onToggleEv={() => setEvOpen((o) => !o)} />
          </div>
        ))}
      </div>

      <div className="carousel-controls">
        <button className="carousel-ctrl" onClick={() => go(idx.active - 1, 'prev')}>
          ← Previous
        </button>
        <button
          className="carousel-ctrl carousel-ctrl-icon"
          onClick={() => setPaused((p) => !p)}
          aria-pressed={paused}
          aria-label={paused ? 'Play' : 'Pause'}
        >
          {paused ? '▶' : '❚❚'}
        </button>
        <div className="carousel-dots" aria-hidden>
          {rows.map((r, i) => (
            <span key={r.key} className={`dot-i ${idx.active === i ? 'on' : ''}`} />
          ))}
        </div>
        <span className="carousel-timer">{`${remaining}s`}</span>
        <button className="carousel-ctrl" onClick={() => go(idx.active + 1, 'next')}>
          Next →
        </button>
      </div>
    </section>
  )
}

function Result({
  row,
  t,
  evOpen,
  onToggleEv,
}: {
  row: (typeof ROWS)[number]
  t: Task
  evOpen: boolean
  onToggleEv: () => void
}) {
  const fmt = (v: number) => (t.metric === 'recall' ? v.toFixed(2) : v.toFixed(3))
  const verdict = t.winner === 'deep' ? 'deep wins' : t.winner === 'tie' ? 'tie' : 'baseline wins'
  return (
    <article className="result">
      <p className="eyebrow">
        <b>{row.n}</b>
        {row.kicker}
      </p>
      <h2 className="result-q">{row.q}</h2>
      <div className="stat">
        <span className="stat-num">{fmt(t.deep)}</span>
        <span className="stat-metric">{t.metric}</span>
        <span className="stat-vs">
          vs {t.baseline_name} {fmt(t.baseline)} · <span className={`verdict ${t.winner}`}>{verdict}</span>
        </span>
      </div>
      <p className="result-desc">{row.desc}</p>
      <Evidence figure={row.figure} kicker={row.kicker} tech={row.tech} open={evOpen} onToggle={onToggleEv} />
    </article>
  )
}

// Disclosure whose panel grows smoothly (grid-rows 0fr -> 1fr) instead of snapping open.
// Open state is shared across all three cards (controlled by the parent).
function Evidence({
  figure,
  kicker,
  tech,
  open,
  onToggle,
}: {
  figure: string
  kicker: string
  tech?: string
  open: boolean
  onToggle: () => void
}) {
  return (
    <div className={`evidence ${open ? 'open' : ''}`}>
      <button className="evidence-summary" onClick={onToggle} aria-expanded={open}>
        Evidence
      </button>
      <div className="evidence-wrap">
        <div className="evidence-inner">
          <img src={figure} alt={`${kicker} — deep model vs baseline on held-out data`} />
          {tech && <p className="evidence-tech">{tech}</p>}
        </div>
      </div>
    </div>
  )
}

// A synthetic seismograph that shakes in place — stationary, but the closer the mouse, the
// larger and more erratic the amplitude; calm and near-flat when the mouse is far away.
function Trace() {
  const svgRef = useRef<SVGSVGElement>(null)
  const pathRef = useRef<SVGPathElement>(null)
  const target = useRef(0) // proximity target from the mouse, 0 (far) .. 1 (over the trace)
  const level = useRef(0) // eased current intensity, so it ramps smoothly

  useEffect(() => {
    const N = 480
    const W = 1000
    const MID = 32
    // smooth value noise (continuous) — animating its time argument makes the line wobble
    const hash = (n: number) => {
      const s = Math.sin(n * 127.1) * 43758.5453
      return s - Math.floor(s)
    }
    const vnoise = (x: number) => {
      const i = Math.floor(x)
      const f = x - i
      const a = hash(i) * 2 - 1
      const b = hash(i + 1) * 2 - 1
      const u = f * f * (3 - 2 * f)
      return a * (1 - u) + b * u
    }
    // x stays fixed per point (no scrolling); only y moves. phase animates, k = intensity.
    // Amplitude is bell-shaped across x: the centre swells much more than the edges as k rises.
    const frame = (phase: number, k: number) => {
      const edgeAmp = 1.3 + k * 7 // edges: calm ~1.3px, medium ~8px near the mouse
      const centreExtra = k * 42 // extra amplitude concentrated in the middle — big swell when close
      let d = ''
      for (let i = 0; i < N; i++) {
        const nx = i / (N - 1)
        const x = nx * W
        const bell = Math.exp(-Math.pow((nx - 0.5) / 0.28, 2)) // 1 at centre, ~0 at edges
        const amp = edgeAmp + centreExtra * bell
        const s1 = vnoise(i * 0.4 + phase)
        const s2 = vnoise(i * 1.4 + phase * 1.9 + 40)
        const s3 = vnoise(i * 3.0 + phase * 3.4 + 120) // sharp, high-freq — only shows up near the mouse
        const shape = s1 * 0.55 + s2 * 0.3 + s3 * 0.15 * k
        d += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${(MID + shape * amp).toFixed(2)} `
      }
      pathRef.current?.setAttribute('d', d)
    }

    // Desktop (fine pointer) reacts to the cursor's proximity. Touch devices have no hover, so on
    // a coarse pointer the trace self-animates instead: a slow, continuous low -> high -> low pulse.
    const coarse = window.matchMedia('(pointer: coarse)').matches

    const setTargetFrom = (cx: number, cy: number) => {
      const el = svgRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const dx = Math.max(r.left - cx, 0, cx - r.right)
      const dy = Math.max(r.top - cy, 0, cy - r.bottom)
      target.current = Math.max(0, 1 - Math.hypot(dx, dy) / 420) // within ~420px it starts waking up
    }
    const onMove = (e: MouseEvent) => setTargetFrom(e.clientX, e.clientY)
    if (!coarse) window.addEventListener('mousemove', onMove)
    const unbind = () => window.removeEventListener('mousemove', onMove)

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      frame(0, 0)
      return unbind
    }

    let raf = 0
    let phase = 0
    let elapsed = 0
    let last = performance.now()
    const PULSE = 3.2 // seconds for one low -> high -> low cycle on touch devices
    const loop = (t: number) => {
      const dt = Math.min((t - last) / 1000, 0.05)
      last = t
      if (coarse) {
        // autonomous breathing: calm/low (~0.12) up to a big swell (~0.9) and back, forever
        elapsed += dt
        target.current = 0.12 + 0.78 * (0.5 - 0.5 * Math.cos((elapsed / PULSE) * Math.PI * 2))
      }
      level.current += (target.current - level.current) * Math.min(1, dt * 5) // ease toward target
      phase += (0.6 + level.current * 5) * dt // faster wobble when agitated (accumulated -> no jumps)
      frame(phase, level.current)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf)
      unbind()
    }
  }, [])

  return (
    <svg ref={svgRef} className="trace" viewBox="0 0 1000 64" preserveAspectRatio="none" aria-hidden>
      <path ref={pathRef} pathLength={1} />
    </svg>
  )
}

// sms-tsunami-warning.com resolves by USGS event id; the place-slug and date segments are
// cosmetic (any value returns the right page), so the id is what makes the link correct.
function eventLink(e: UsgsEvent): string {
  const slug =
    (e.place || 'earthquake')
      .replace(/^\d+\s*km\s+[NSEW]+\s+of\s+/i, '') // strip "36 km SSE of "
      .replace(/,/g, '')
      .trim()
      .replace(/\s+/g, '-')
      .replace(/[^A-Za-z0-9-]/g, '') || 'earthquake'
  const d = e.time ? new Date(e.time) : new Date()
  const date = `${String(d.getDate()).padStart(2, '0')}-${String(d.getMonth() + 1).padStart(2, '0')}-${d.getFullYear()}`
  return `https://www.sms-tsunami-warning.com/earthquakes-today/${e.id}/${slug}/${date}`
}

const fmtDate = (ms?: number) =>
  ms ? new Date(ms).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : ''

// A top-5 list (World or California) — each row links to the event's sms-tsunami-warning page.
function QuakeList({ title, events }: { title: string; events?: UsgsEvent[] }) {
  return (
    <div className="qlist">
      <p className="eyebrow">{title}</p>
      {events && events.length === 0 && <p className="muted">None recorded.</p>}
      <ul className="recent-list">
        {events?.map((e) => (
          <li key={e.id}>
            <a className="recent-link" href={eventLink(e)} target="_blank" rel="noopener noreferrer">
              <span className={`m ${e.mag >= 5 ? 'hi' : ''}`}>M{e.mag.toFixed(1)}</span>
              <span className="place">{e.place}</span>
              <span className="r">{fmtDate(e.time)}</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}

const CA_WINDOWS = [
  { key: 'day', label: 'Day' },
  { key: 'week', label: 'Week' },
  { key: 'month', label: 'Month' },
  { key: 'year', label: 'Year' },
  { key: 'all', label: 'All time' },
]

// The biggest California quakes across widening time windows, cycled like the model cards.
function CaLargest() {
  const n = CA_WINDOWS.length
  const CYCLE = 7
  const [idx, setIdx] = useState<{ active: number; prev: number; dir: 'next' | 'prev' }>({ active: 0, prev: 0, dir: 'next' })
  const [paused, setPaused] = useState(false)
  const [remaining, setRemaining] = useState(CYCLE)
  const [data, setData] = useState<Record<string, CaWindow | 'err'>>({})

  useEffect(() => {
    CA_WINDOWS.forEach((w) =>
      caTop(w.key)
        .then((r) => setData((d) => ({ ...d, [w.key]: r })))
        .catch(() => setData((d) => ({ ...d, [w.key]: 'err' })))
    )
  }, [])

  useEffect(() => {
    if (paused || n < 2) return
    const id = setInterval(() => {
      setRemaining((r) => {
        if (r <= 1) {
          setIdx((s) => ({ active: (s.active + 1) % n, prev: s.active, dir: 'next' }))
          return CYCLE
        }
        return r - 1
      })
    }, 1000)
    return () => clearInterval(id)
  }, [paused, n])

  const go = (i: number, dir: 'next' | 'prev' = 'next') => {
    setIdx((s) => ({ active: ((i % n) + n) % n, prev: s.active, dir }))
    setRemaining(CYCLE)
  }
  const cls = (i: number) => (i === idx.active ? 'is-active' : i === idx.prev ? 'is-leaving' : 'is-rest')

  return (
    <section className="carousel" aria-roledescription="carousel" aria-label="Largest California earthquakes">
      <div className="largest-head">
        <div>
          <p className="eyebrow">Largest earthquakes</p>
          <h2>Biggest Southern California quakes</h2>
          <p className="source-note">Source: USGS Earthquake Catalog</p>
        </div>
      </div>
      <div className="carousel-tabs" role="tablist">
        {CA_WINDOWS.map((w, i) => (
          <button
            key={w.key}
            role="tab"
            aria-selected={idx.active === i}
            className={`carousel-tab ${idx.active === i ? 'on' : ''}`}
            onClick={() => go(i, i < idx.active ? 'prev' : 'next')}
          >
            {w.label}
          </button>
        ))}
      </div>
      <div className={`carousel-stage dir-${idx.dir}`}>
        {CA_WINDOWS.map((w, i) => {
          const d = data[w.key]
          return (
            <div key={w.key} className={`carousel-card ${cls(i)}`} aria-hidden={idx.active !== i}>
              <div className="ca-panel">
                {d === undefined && <p className="muted">Loading…</p>}
                {d === 'err' && <p className="muted">Feed unavailable — is the backend running?</p>}
                {d && d !== 'err' && <QuakeList title={d.label} events={d.events} />}
              </div>
            </div>
          )
        })}
      </div>
      <div className="carousel-controls">
        <button className="carousel-ctrl" onClick={() => go(idx.active - 1, 'prev')}>
          ← Previous
        </button>
        <button
          className="carousel-ctrl carousel-ctrl-icon"
          onClick={() => setPaused((p) => !p)}
          aria-pressed={paused}
          aria-label={paused ? 'Play' : 'Pause'}
        >
          {paused ? '▶' : '❚❚'}
        </button>
        <div className="carousel-dots" aria-hidden>
          {CA_WINDOWS.map((w, i) => (
            <span key={w.key} className={`dot-i ${idx.active === i ? 'on' : ''}`} />
          ))}
        </div>
        <span className="carousel-timer">{`${remaining}s`}</span>
        <button className="carousel-ctrl" onClick={() => go(idx.active + 1, 'next')}>
          Next →
        </button>
      </div>
    </section>
  )
}

function NearMe() {
  const [form, setForm] = useState({ name: '', email: '', lat: '', lon: '' })
  const [msg, setMsg] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [city, setCity] = useState('')
  const [stateName, setStateName] = useState('CA')
  const [searching, setSearching] = useState(false)

  const set = (k: keyof typeof form) => (e: ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value })

  const searchLocation = async () => {
    if (!city.trim()) return
    setSearching(true)
    setMsg(null)
    try {
      const r = await geocode(`${city.trim()}, ${stateName.trim() || 'CA'}`)
      setForm((f) => ({ ...f, lat: r.lat.toFixed(4), lon: r.lon.toFixed(4) }))
      setMsg({ kind: 'ok', text: `Found: ${r.name}` })
    } catch (err) {
      setMsg({ kind: 'err', text: `Couldn’t find that place: ${(err as Error).message}` })
    } finally {
      setSearching(false)
    }
  }

  const useMyLocation = () => {
    if (!navigator.geolocation) return setMsg({ kind: 'err', text: 'Geolocation unavailable — enter it manually.' })
    navigator.geolocation.getCurrentPosition(
      (pos) => setForm((f) => ({ ...f, lat: pos.coords.latitude.toFixed(4), lon: pos.coords.longitude.toFixed(4) })),
      () => setMsg({ kind: 'err', text: 'Couldn’t read your location — enter it manually.' })
    )
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setMsg(null)
    try {
      const r = await subscribe({ name: form.name, email: form.email, lat: Number(form.lat), lon: Number(form.lon) })
      let text = r.note ?? `Subscribed — you’re #${r.count} on the watch list.`
      // On the mobile app, also register this device for push alerts.
      try {
        const pushed = await enablePush({ name: form.name, lat: Number(form.lat), lon: Number(form.lon) })
        if (pushed) text += ' Push alerts enabled on this device.'
      } catch (pErr) {
        text += ` (Email set; push couldn’t be enabled: ${(pErr as Error).message}.)`
      }
      setMsg({ kind: 'ok', text })
    } catch (err) {
      setMsg({ kind: 'err', text: `Couldn’t subscribe: ${(err as Error).message}. Is the backend running?` })
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="nearme">
      <p className="eyebrow">Alerts</p>
      <h2>Alert me near me</h2>
      <p className="nearme-lede">
        A model watches the live Southern California seismic stream and, when it detects a quake
        near you, emails you the detection, its size, and how hard it is likely to shake. This is
        rapid detection, not an official warning.
        {isNativeApp() && ' On this app, alerts also arrive as push notifications.'}
      </p>

      <div className="card">
        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="nm-name">Name</label>
            <input id="nm-name" value={form.name} onChange={set('name')} required placeholder="Your name" />
          </div>
          <div className="field">
            <label htmlFor="nm-email">Email</label>
            <input id="nm-email" type="email" value={form.email} onChange={set('email')} required placeholder="you@example.com" />
          </div>
          <div className="field">
            <label htmlFor="nm-city">Search location</label>
            <div className="search-row">
              <input
                id="nm-city"
                value={city}
                onChange={(e) => setCity(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    searchLocation()
                  }
                }}
                placeholder="City"
                aria-label="City"
              />
              <input
                id="nm-state"
                className="state-input"
                value={stateName}
                onChange={(e) => setStateName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    searchLocation()
                  }
                }}
                placeholder="State"
                aria-label="State"
              />
              <button type="button" className="btn-outline" onClick={searchLocation} disabled={searching}>
                {searching ? 'Searching…' : 'Search'}
              </button>
            </div>
          </div>
          <div className="field-pair">
            <div className="field">
              <label htmlFor="nm-lat">Latitude</label>
              <input id="nm-lat" value={form.lat} onChange={set('lat')} required placeholder="34.05" />
            </div>
            <div className="field">
              <label htmlFor="nm-lon">Longitude</label>
              <input id="nm-lon" value={form.lon} onChange={set('lon')} required placeholder="-118.24" />
            </div>
          </div>
          <div className="actions">
            <button type="submit" className="btn" disabled={busy}>
              {busy ? 'Subscribing…' : 'Subscribe'}
            </button>
            <button type="button" className="btn-outline" onClick={useMyLocation}>
              Use my location
            </button>
          </div>
          {msg && <p className={`form-msg ${msg.kind}`}>{msg.text}</p>}
        </form>
      </div>
    </section>
  )
}
