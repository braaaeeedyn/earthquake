import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import { loadSeismic, type Seismic, type Task } from './seismic'
import { caTop, geocode, liveStatus, sendContact, type UsgsEvent, type CaWindow } from './nearme'
import { enablePush, disablePush, initPush, isNativeApp, isSubscribed } from './push'
import { getMyLocation } from './geo'

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

// Tiny pathname router: push a new path and re-render (no router dependency for a handful of pages).
function navigate(path: string) {
  window.history.pushState({}, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

type Route = 'home' | 'app' | 'privacy' | 'notfound'
const ROUTE_OF = (p: string): Route =>
  p === '/' ? 'home' : p === '/app' ? 'app' : p === '/privacy' ? 'privacy' : 'notfound'
const PAGE_TITLES: Record<Route, string> = {
  home: 'SeismicSoCal · Southern California earthquake ML',
  app: 'Get the app · SeismicSoCal',
  privacy: 'Privacy policy · SeismicSoCal',
  notfound: 'Page not found · SeismicSoCal',
}

// Opens the in-app contact form. The support address lives only on the server, so it's never
// shown here; any "Contact support" control just fires this event and App renders the modal.
function contactSupport() {
  window.dispatchEvent(new CustomEvent('open-contact'))
}

export default function App() {
  const [state, setState] = useState<State>({ status: 'loading' })
  const [live, setLive] = useState(false)
  const [path, setPath] = useState(() => window.location.pathname)
  const [contactOpen, setContactOpen] = useState(false)

  useEffect(() => {
    let active = true
    loadSeismic()
      .then((data) => active && setState({ status: 'ready', data }))
      .catch((e) => active && setState({ status: 'error', message: String(e?.message ?? e) }))
    initPush()          // register foreground push handlers (no-op on web)
    const checkLive = () => liveStatus().then((v) => active && setLive(v))
    checkLive()
    const id = setInterval(checkLive, 15000)   // poll the SeedLink watcher status
    const onPop = () => setPath(window.location.pathname)
    const onContact = () => setContactOpen(true)
    window.addEventListener('popstate', onPop)
    window.addEventListener('open-contact', onContact)
    return () => {
      active = false
      clearInterval(id)
      window.removeEventListener('popstate', onPop)
      window.removeEventListener('open-contact', onContact)
    }
  }, [])

  const route = ROUTE_OF(path)
  useEffect(() => { document.title = PAGE_TITLES[route] }, [route])

  return (
    <div className="app">
      <nav className="nav">
        <a className="brand" href="/" onClick={(e) => { e.preventDefault(); navigate('/') }}>SeismicSoCal</a>
        <span className="live">
          <span className={`dot ${live ? 'on' : ''}`} aria-hidden /> {live ? 'live · SeedLink' : 'offline · SeedLink'}
        </span>
      </nav>

      <main className="content">
        {route === 'app' && <AppDownload />}
        {route === 'privacy' && <Privacy />}
        {route === 'notfound' && <NotFound />}
        {route === 'home' && (
          <>
            {state.status === 'loading' && <p className="state">Loading…</p>}
            {state.status === 'error' && (
              <div className="state">
                <h2>Couldn’t load results</h2>
                <p className="muted">{state.message}</p>
                <p className="muted">Ensure <code>seismic.json</code> is in <code>public/</code>.</p>
              </div>
            )}
            {state.status === 'ready' && <Console data={state.data} />}
          </>
        )}
      </main>

      <footer className="footer">
        <p>Research &amp; education · real Southern California network data · not an official warning system</p>
        <p className="footer-links">
          <a href="/privacy" onClick={(e) => { e.preventDefault(); navigate('/privacy') }}>Privacy</a>
          <span aria-hidden> · </span>
          <button type="button" className="linklike" onClick={contactSupport}>Contact support</button>
        </p>
      </footer>

      {contactOpen && <ContactModal onClose={() => setContactOpen(false)} />}
    </div>
  )
}

// In-app support form: the user enters their own email + a message; the server relays it to the
// hidden support inbox with their email as Reply-To. Nothing is stored, and the address is never
// exposed to the client.
function ContactModal({ onClose }: { onClose: () => void }) {
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setMsg(null)
    try {
      await sendContact({ email, message })
      setMsg({ kind: 'ok', text: 'Message sent. We’ll reply to the email you entered.' })
      setEmail('')
      setMessage('')
    } catch (err) {
      setMsg({ kind: 'err', text: `Couldn’t send: ${(err as Error).message}` })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="contact-title" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2 id="contact-title">Contact support</h2>
          <button type="button" className="modal-x" onClick={onClose} aria-label="Close">×</button>
        </div>
        <p className="modal-lede">
          Send us a message. Enter your email so we can reply - it’s used only for the reply and is
          not stored.
        </p>
        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="ct-email">Your email</label>
            <input id="ct-email" type="email" required value={email}
              onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
          </div>
          <div className="field">
            <label htmlFor="ct-msg">Message</label>
            <textarea id="ct-msg" required rows={5} maxLength={5000} value={message}
              onChange={(e) => setMessage(e.target.value)} placeholder="How can we help?" />
          </div>
          <div className="actions">
            <button type="submit" className="btn" disabled={busy}>{busy ? 'Sending…' : 'Send message'}</button>
            <button type="button" className="btn-outline" onClick={onClose}>Cancel</button>
          </div>
          {msg && <p className={`form-msg ${msg.kind}`}>{msg.text}</p>}
        </form>
      </div>
    </div>
  )
}

// Inline arrow - an SVG chevron that inherits the text color and centres cleanly (the unicode
// ← / → glyphs sat off the text baseline). Flip horizontally for the left-pointing variant.
function Arrow({ dir = 'right' }: { dir?: 'left' | 'right' }) {
  return (
    <svg className="ar" width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true"
      style={dir === 'left' ? { transform: 'scaleX(-1)' } : undefined}>
      <path d="M4 12h15M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2.2"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

// Custom 404 in the site's own style.
function NotFound() {
  return (
    <section className="notfound">
      <p className="eyebrow">404</p>
      <h1>Page not found</h1>
      <p className="app-lede">That page doesn’t exist. It may have moved, or the link was mistyped.</p>
      <p className="app-back">
        <a className="back-link" href="/" onClick={(e) => { e.preventDefault(); navigate('/') }}><Arrow dir="left" />Back to the console</a>
      </p>
    </section>
  )
}

// Privacy policy. Honest about what the app collects, that ML is used, and the third parties involved.
function Privacy() {
  return (
    <section className="legal">
      <p className="eyebrow">Legal</p>
      <h1>Privacy policy</h1>
      <p className="legal-updated">SeismicSoCal - research prototype. Last updated August 2026.</p>

      <h2>What we collect</h2>
      <p>
        SeismicSoCal collects data <strong>only if you subscribe to alerts inside the app</strong>.
        When you subscribe we store: the name you enter, the approximate location (latitude and
        longitude) you choose, and your device’s push-notification token. The public website
        collects no personal data and has no subscribe function - alerts exist only in the app.
      </p>

      <h2>How we use it</h2>
      <p>
        Your location is used solely to decide whether a detected earthquake is close enough to
        notify you, and your device token is used solely to deliver that push notification. We do
        not sell your data, use it for advertising, or share it except with the notification
        provider described below.
      </p>

      <h2>Use of AI / machine learning</h2>
      <p>
        SeismicSoCal uses machine-learning models to detect earthquakes on a live seismic stream,
        estimate their magnitude, and estimate expected shaking. These models generate the content
        of the alerts you receive. They run on our server against public seismic-network data - your
        personal data is never used to train them, and the alert decision is an automated estimate,
        not an official warning.
      </p>

      <h2>Third parties</h2>
      <p>
        Push notifications are delivered through <strong>Google Firebase Cloud Messaging (FCM)</strong>.
        To route a notification to your device, your push token is shared with Google as the message
        recipient; Google’s handling is governed by its own privacy policy. We also query public data
        services that receive no personal information beyond a normal web request - the USGS /
        EarthScope earthquake catalog, and OpenStreetMap’s Nominatim for the place name you type when
        searching for your location.
      </p>

      <h2>Unsubscribing and deletion</h2>
      <p>
        You can unsubscribe at any time with the <strong>Unsubscribe</strong> button in the app, which
        removes your device token and location from our records. If you <strong>uninstall the app</strong>,
        your device is unsubscribed automatically: the push token is invalidated and we purge it the
        next time an alert would have been sent.
      </p>

      <h2>Contact</h2>
      <p>
        Questions about your data? <button type="button" className="linklike" onClick={contactSupport}>Contact support</button>.
        The contact form uses the email you enter only to reply to you - it is not stored.
      </p>

      <p className="app-back">
        <a className="back-link" href="/" onClick={(e) => { e.preventDefault(); navigate('/') }}><Arrow dir="left" />Back to the console</a>
      </p>
    </section>
  )
}

// The /app subpage: download the Android app (APK) that unlocks subscription + push alerts.
function AppDownload() {
  // Read the real APK size from the file itself so the page can't advertise a stale number.
  const [sizeMB, setSizeMB] = useState<string | null>(null)
  useEffect(() => {
    let active = true
    fetch('/seismicsocal.apk', { method: 'HEAD' })
      .then((r) => {
        const len = r.headers.get('content-length')
        if (active && len) setSizeMB((Number(len) / 1048576).toFixed(1))
      })
      .catch(() => {})
    return () => { active = false }
  }, [])

  return (
    <section className="app-download">
      <p className="eyebrow">Get the app</p>
      <h1>SeismicSoCal for Android</h1>
      <p className="app-lede">
        Alerts run only in the app: install it to subscribe your location and receive push
        notifications when the live models detect a nearby earthquake. This is a research
        prototype, not an official warning system.
      </p>

      <div className="card app-card">
        <div className="app-card-row">
          <div>
            <p className="app-file">seismicsocal.apk</p>
            <p className="muted app-file-sub">Android{sizeMB ? ` · ${sizeMB} MB` : ''} · debug build</p>
          </div>
          <a className="btn" href="/seismicsocal.apk" download="seismicsocal.apk">Download APK</a>
        </div>
        <ol className="app-steps">
          <li>Download the APK on your Android device.</li>
          <li>Open it - Android will ask to allow installs from this source. Enable it for your browser.</li>
          <li>Install, open SeismicSoCal, set your location, and tap Subscribe to turn on alerts.</li>
        </ol>
        <p className="muted app-note">
          Android only. iOS is not available. The APK is an unsigned debug build for demonstration;
          your device may warn about installing outside the Play Store.
        </p>
      </div>

      <p className="app-back">
        <a className="back-link" href="/" onClick={(e) => { e.preventDefault(); navigate('/') }}><Arrow dir="left" />Back to the console</a>
      </p>
    </section>
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
          <Arrow dir="left" />Previous
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
          Next<Arrow dir="right" />
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
          <img src={figure} alt={`${kicker} - deep model vs baseline on held-out data`} />
          {tech && <p className="evidence-tech">{tech}</p>}
        </div>
      </div>
    </div>
  )
}

// A synthetic seismograph that shakes in place - stationary, but the closer the mouse, the
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
    // smooth value noise (continuous) - animating its time argument makes the line wobble
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
      const centreExtra = k * 42 // extra amplitude concentrated in the middle - big swell when close
      let d = ''
      for (let i = 0; i < N; i++) {
        const nx = i / (N - 1)
        const x = nx * W
        const bell = Math.exp(-Math.pow((nx - 0.5) / 0.28, 2)) // 1 at centre, ~0 at edges
        const amp = edgeAmp + centreExtra * bell
        const s1 = vnoise(i * 0.4 + phase)
        const s2 = vnoise(i * 1.4 + phase * 1.9 + 40)
        const s3 = vnoise(i * 3.0 + phase * 3.4 + 120) // sharp, high-freq - only shows up near the mouse
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

const fmtDate = (ms?: number) =>
  ms ? new Date(ms).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : ''

// A top-5 list (World or California) - each row is display-only (not a link).
function QuakeList({ title, events }: { title: string; events?: UsgsEvent[] }) {
  return (
    <div className="qlist">
      <p className="eyebrow">{title}</p>
      {events && events.length === 0 && <p className="muted">None recorded.</p>}
      <ul className="recent-list">
        {events?.map((e) => (
          <li key={e.id}>
            <div className="recent-link">
              <span className={`m ${e.mag >= 5 ? 'hi' : ''}`}>M{e.mag.toFixed(1)}</span>
              <span className="place">{e.place}</span>
              <span className="r">{fmtDate(e.time)}</span>
            </div>
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
                {d === 'err' && <p className="muted">Feed unavailable - is the backend running?</p>}
                {d && d !== 'err' && <QuakeList title={d.label} events={d.events} />}
              </div>
            </div>
          )
        })}
      </div>
      <div className="carousel-controls">
        <button className="carousel-ctrl" onClick={() => go(idx.active - 1, 'prev')}>
          <Arrow dir="left" />Previous
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
          Next<Arrow dir="right" />
        </button>
      </div>
    </section>
  )
}

function NearMe() {
  const [form, setForm] = useState({ name: '', lat: '', lon: '' })
  const [msg, setMsg] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [subscribed, setSubscribed] = useState(isSubscribed())
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

  const useMyLocation = async () => {
    try {
      const { lat, lon } = await getMyLocation()
      setForm((f) => ({ ...f, lat: lat.toFixed(4), lon: lon.toFixed(4) }))
    } catch {
      setMsg({ kind: 'err', text: 'Couldn’t read your location - allow the location permission, or enter it manually.' })
    }
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setMsg(null)
    try {
      const pushed = await enablePush({ name: form.name, lat: Number(form.lat), lon: Number(form.lon) })
      if (pushed === null) {
        // web build: no native push, so there's nothing to subscribe to here
        setMsg({ kind: 'err', text: 'Alerts arrive as push notifications - install the SeismicSoCal app to subscribe.' })
        return
      }
      setSubscribed(true)
      setMsg({ kind: 'ok', text: 'Subscribed - push alerts are enabled on this device.' })
    } catch (err) {
      setMsg({ kind: 'err', text: `Couldn’t subscribe: ${(err as Error).message}.` })
    } finally {
      setBusy(false)
    }
  }

  const unsubscribe = async () => {
    setBusy(true)
    setMsg(null)
    try {
      await disablePush()
      setSubscribed(false)
      setMsg({ kind: 'ok', text: 'Unsubscribed - this device will no longer receive alerts.' })
    } catch (err) {
      setMsg({ kind: 'err', text: `Couldn’t unsubscribe: ${(err as Error).message}.` })
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
        near you, sends a push notification with the detection, its size, and how hard it is likely
        to shake. This is rapid detection, not an official warning.
        {!isNativeApp() && ' Alerts are available in the SeismicSoCal app.'}
      </p>

      <div className="card">
        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="nm-name">Name</label>
            <input id="nm-name" value={form.name} onChange={set('name')} required placeholder="Your name" />
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
            {isNativeApp() ? (
              subscribed ? (
                <button type="button" className="btn" onClick={unsubscribe} disabled={busy}>
                  {busy ? 'Unsubscribing…' : 'Unsubscribe'}
                </button>
              ) : (
                <button type="submit" className="btn" disabled={busy}>
                  {busy ? 'Subscribing…' : 'Subscribe'}
                </button>
              )
            ) : (
              // Web has no push channel - subscription lives in the app only.
              <button type="button" className="btn btn-struck" disabled aria-disabled="true">
                Subscribe
              </button>
            )}
            <button type="button" className="btn-outline" onClick={useMyLocation}>
              Use my location
            </button>
          </div>
          {!isNativeApp() && (
            <p className="download-note">
              Download the app for subscription and notifications -{' '}
              <a href="/app" onClick={(e) => { e.preventDefault(); navigate('/app') }}>get the app</a>.
            </p>
          )}
          {msg && <p className={`form-msg ${msg.kind}`}>{msg.text}</p>}
        </form>
      </div>
    </section>
  )
}
