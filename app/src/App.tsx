import { useEffect, useRef, useState, type FormEvent } from 'react'
import { loadSeismic, type Seismic, type Task } from './seismic'
import { caTop, geocode, getAppVersion, getHealth, getStations, liveStatus, sendContact, type Health, type ModelInfo, type Station, type UsgsEvent, type CaWindow } from './nearme'
import { enablePush, disablePush, initPush, isNativeApp, subscribedStations, subscribedName, subscribedMode, type AlertMode } from './push'
import { getMyLocation } from './geo'
import { APP_VERSION, mustUpdate, updateAvailable } from './version'
import Coverage from './Coverage'
import { clearHome, getHome, homeShaking, loadShakingModel, setHome, type Home } from './shaking'

// The public site, for sending the app to the download/update page in an external browser.
const APP_SITE = 'https://seismicsocal.duckdns.org'

type State =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: Seismic }

// Detect -> Size is the live pipeline's order, so the numbering carries meaning.
const ROWS: { n: string; kicker: string; key: string; q: string; figure: string; desc: string; tech: string[] }[] = [
  {
    n: '01', kicker: 'Detect', key: 'detection', q: 'Is it an earthquake?', figure: 'detect_evidence.png',
    desc: 'Every 2 seconds, each of the 19 live sensors hands the model its last 30 seconds of ground motion, and the model decides whether an earthquake is in it or just traffic, wind or sensor noise. A quake only counts when at least three sensors see it and their timings point to one place. On held-out data it separates quakes from noise almost perfectly, and replayed on 80 held-out days it caught about 7 in 10 quakes of M3 and up (about 8 in 10 outside busy aftershock sequences). Every alert it sent was for a real quake, though about 1 in 12 was placed more than 60 km off.',
    tech: [
      'Input: the vertical channel of each station, 30 s at 100 Hz, causally band-limited (1 Hz high-pass, 18 Hz low-pass) and scaled to unit variance — the same preparation in training and live.',
      'Model: 4 strided 1-D convolutions (16→64 channels) turn the trace into a feature sequence; a 2-layer Transformer encoder reads it and a linear head gives P(earthquake). Selected from 5 seeds on validation AUC.',
      'Training data: 34,377 event windows (P-wave placed anywhere 1–25 s into the window), 14,304 noise windows from all hours with no catalogued M1+ quake nearby, and 2,062 hard negatives — the previous live system’s own false alarms. Chronological split, 2000–2026.',
      'Held-out test (7,303 windows, 2022–2026): ROC-AUC 0.9998, MCC 0.886; the classic STA/LTA trigger on the same input reaches AUC 0.816.',
      'Live: a window above 0.6 triggers a P-wave pick (STA/LTA onset refined by an Akaike picker). Picks from ≥3 stations are located by grid search; the event is confirmed only if one source fits them (RMS ≤ 1.5 s) and no working station closer to it stayed silent.',
      'Replay of the exact live code on 80 held-out days (2022–2026): 68 % of M3+ quakes caught (78 % outside aftershock sequences with ≥3 stations online), 52 % of M2+; 145 push alerts, every one a real quake, 92 % placed within 60 km; median location error 3–4 km.',
    ],
  },
  {
    n: '02', kicker: 'Size', key: 'magnitude', q: 'How big is it?', figure: 'size_evidence.png',
    desc: 'Once a quake is located, the size model reads 30 seconds from every nearby sensor — lined up on the moment the P-wave arrived at each one — and combines them into one magnitude. On held-out quakes it is typically within 0.1 of the official magnitude, clearly better than the classic amplitude-and-distance formula. A faster 4-second check sends the first alert; this full estimate confirms it.',
    tech: [
      'Input: for every working station within 200 km of the located epicentre, a 3-component velocity window from 5 s before its P arrival to 25 s after (instrument response removed, 18 Hz low-pass).',
      'Model: each window, scaled to unit peak, goes through a 1-D CNN (wave shape); its log peak velocity and log distance from the epicentre join it as graph-node features; two graph-convolution layers over the 19-station network (Gaussian distance weights, 150 km cut-off) and a Transformer mix the stations; the pooled vector plus 4 network amplitude/distance statistics give the magnitude. Average of 5 seeds.',
      'Training data: 6,243 catalogued SoCal quakes (M2.0–7.1, 2000–2026), with augmentation for live conditions — ±8 km epicentre jitter, ±0.5 s pick jitter, and only the nearest 3–n stations or random station drop-out.',
      'Held-out test (937 quakes, Aug 2021–2026): R² 0.951, MAE 0.10; amplitude + distance baseline R² 0.886, MAE 0.16; nearest single station only MAE 0.20. Live-like (10 km location error, 3–6 stations): MAE 0.11.',
      'Quick check (first alert): median over the picked stations of a·log10(peak velocity in the first 4 s of P) + b·log10(distance) + c, fitted on the training quakes. Test MAE 0.24; 93 % of M3+ quakes pass its validated threshold, 1 % of quakes under M2.5 do.',
      'Push rule: provisional alert if the quick check ≥ 3.04; the full estimate then confirms (M ≥ 3.0) or retracts it, replacing the first notification. Replay: provisional alerts ~33 s after origin on the Standard setting (~26 s on Fast: 2 s of P, sensitivity-scaled), confirmations ~55 s.',
    ],
  },
]

// Tiny pathname router: push a new path and re-render (no router dependency for a handful of pages).
function navigate(path: string) {
  window.history.pushState({}, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

type Route = 'home' | 'app' | 'privacy' | 'health' | 'quake' | 'notfound'
const ROUTE_OF = (p: string): Route =>
  p === '/' ? 'home' : p === '/app' ? 'app' : p === '/privacy' ? 'privacy' : p === '/health' ? 'health' : p === '/quake' ? 'quake' : 'notfound'
const PAGE_TITLES: Record<Route, string> = {
  home: 'SeismicSoCal · Southern California earthquake ML',
  app: 'Get the app · SeismicSoCal',
  privacy: 'Privacy policy · SeismicSoCal',
  health: 'Model health · SeismicSoCal',
  quake: 'Earthquake · SeismicSoCal',
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
  const [gate, setGate] = useState<{ blocked: boolean; latest: string } | null>(null)

  useEffect(() => {
    let active = true
    loadSeismic()
      .then((data) => active && setState({ status: 'ready', data }))
      .catch((e) => active && setState({ status: 'error', message: String(e?.message ?? e) }))
    initPush()          // register foreground push handlers (no-op on web)
    loadShakingModel().catch(() => {})   // cache the shaking equations (also for the native notification code)
    if (isNativeApp()) {
      // Version gate (installed app only): behind the server's MIN on major/minor -> block. Fail OPEN
      // on a network error so an offline user is never locked out by a failed check.
      getAppVersion()
        .then((v) => active && setGate({ blocked: mustUpdate(APP_VERSION, v.min), latest: v.latest }))
        .catch(() => active && setGate({ blocked: false, latest: APP_VERSION }))
    }
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

  // A required update (major/minor behind) blocks the whole app until re-downloaded.
  if (gate?.blocked) return <UpdateRequired latest={gate.latest} />

  const softUpdate = gate && !gate.blocked && updateAvailable(APP_VERSION, gate.latest)

  return (
    <div className="app">
      <nav className="nav">
        <a className="brand" href="/" onClick={(e) => { e.preventDefault(); navigate('/') }}>SeismicSoCal</a>
        <span className="live">
          <span className={`dot ${live ? 'on' : ''}`} aria-hidden /> {live ? 'live · SeedLink' : 'offline · SeedLink'}
        </span>
      </nav>

      {softUpdate && (
        <p className="update-banner">
          A newer version (v{gate!.latest}) is available.{' '}
          <a href={`${APP_SITE}/app`} target="_blank" rel="noopener">Update</a>
        </p>
      )}

      <main className="content">
        {route === 'app' && <AppDownload />}
        {route === 'privacy' && <Privacy />}
        {route === 'health' && <ModelHealth />}
        {route === 'quake' && <QuakeDetail />}
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
          <a href="/health" onClick={(e) => { e.preventDefault(); navigate('/health') }}>Model health</a>
          <span aria-hidden> · </span>
          <button type="button" className="linklike" onClick={contactSupport}>Contact support</button>
        </p>
        <p className="footer-ver">v{APP_VERSION}</p>
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

// QuakeOps model health: which model version is live, its held-out metrics with 95% CIs against the
// classic baseline, per-station drift of the live stream vs the training data, and promotion history.
const HEALTH_CARDS: { key: 'detector' | 'magnitude'; name: string; rows: [string, string, number][] }[] = [
  { key: 'detector', name: 'Detect', rows: [['test_auc', 'ROC-AUC', 4], ['sta_lta_auc', 'STA/LTA AUC', 3], ['test_mcc', 'MCC', 3], ['fpr_test_noise_at_trigger', 'False triggers / window (live trigger)', 4]] },
  { key: 'magnitude', name: 'Size', rows: [['ens_r2', 'R²', 3], ['baseline_r2', 'Amp + distance R²', 3], ['ens_mae', 'MAE (magnitude)', 3], ['live_like_r2', 'Live-like R²', 3]] },
]

function ModelHealth() {
  const [h, setH] = useState<Health | 'error' | null>(null)
  useEffect(() => { getHealth().then(setH).catch(() => setH('error')) }, [])
  const day = (s?: string) => (s ? s.slice(0, 10) : '—')
  const metric = (m: ModelInfo, key: string, dp: number) => {
    const v = m.metrics[key]
    if (typeof v !== 'number') return '—'
    const ci = m.metrics[`${key}_ci`]
    return Array.isArray(ci) ? `${v.toFixed(dp)} (${ci[0].toFixed(dp)}–${ci[1].toFixed(dp)})` : v.toFixed(dp)
  }
  return (
    <section className="legal health">
      <p className="eyebrow">QuakeOps</p>
      <h1>Model health</h1>
      <p className="legal-updated">
        Which model is live, how it scores on held-out data (95% confidence intervals), and whether the live
        stream still looks like the data it was trained on. A new model goes live only after it passes the
        promotion gate, including a replay of held-out days.
      </p>
      {h === null && <p className="state">Loading…</p>}
      {h === 'error' && <p className="state muted">Health data is unavailable right now.</p>}
      {h && h !== 'error' && (
        <>
          <h2>Live models</h2>
          {!h.models && <p className="muted">The model registry hasn’t reported yet.</p>}
          {h.models && (
            <div className="health-grid">
              {HEALTH_CARDS.map((c) => {
                const m = h.models?.[c.key]
                return (
                  <article className="card health-card" key={c.key}>
                    <p className="eyebrow">{c.name}</p>
                    {m ? (
                      <>
                        <p className="health-ver">v{m.version}{m.deployed != null && m.deployed !== m.version ? ` (v${m.deployed} running)` : ''}</p>
                        <dl className="health-dl">
                          {c.rows.map(([k, label, dp]) => (
                            <div key={k}><dt>{label}</dt><dd>{metric(m, k, dp)}</dd></div>
                          ))}
                          <div><dt>Trained</dt><dd>{day(m.trained)}</dd></div>
                          <div><dt>Data · code</dt><dd>{m.dataset_version ?? '—'} · {m.git_commit ?? '—'}</dd></div>
                        </dl>
                      </>
                    ) : <p className="muted">Not registered.</p>}
                  </article>
                )
              })}
            </div>
          )}

          <h2>Live stream vs training data</h2>
          {!h.drift && <p className="muted">The daily drift check hasn’t run yet.</p>}
          {h.drift && (
            <>
              <p>
                {day(h.drift.date)}: the detector’s score, spikiness and frequency content of each station’s
                live noise, compared with its training noise.
              </p>
              <ul className="drift-pills">
                {Object.entries(h.drift.stations).map(([code, s]) => (
                  <li key={code} className={`drift-pill ${s.status}`} title={s.drifted.length ? `changed: ${s.drifted.join(', ')}` : undefined}>
                    <span className="drift-code">{code}</span> {s.status === 'insufficient' ? 'no data' : s.status}
                  </li>
                ))}
              </ul>
            </>
          )}

          <h2>Promotion history</h2>
          {!h.models?.history?.length && <p className="muted">No promotions recorded yet.</p>}
          {!!h.models?.history?.length && (
            <table className="health-table">
              <thead><tr><th>Model</th><th>Version</th><th>Date</th><th>Why</th></tr></thead>
              <tbody>
                {h.models.history.map((r) => (
                  <tr key={`${r.model}-${r.version}-${r.promoted_at}`}>
                    <td>{r.model === 'detector' ? 'Detect' : 'Size'}</td><td>v{r.version}</td><td>{day(r.promoted_at)}</td><td>{r.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
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
      <p className="legal-updated">SeismicSoCal - research prototype. Last updated October 2026.</p>

      <h2>What we collect</h2>
      <p>
        SeismicSoCal collects data <strong>only if you subscribe to alerts inside the app</strong>.
        When you subscribe we store: the name you enter, the sensor stations you choose to follow, your alert-speed setting,
        and your device’s push-notification token. We do <strong>not</strong> store your location —
        it is used only on your device: to rank which stations are nearest you, and, saved on the device as your
        “home”, to estimate how strongly each quake shook there. It is never sent to us. The public website collects no personal data and has no subscribe function -
        alerts exist only in the app.
      </p>

      <h2>How we use it</h2>
      <p>
        The sensor stations you follow are used solely to decide which detections to notify you
        about, and your device token is used solely to deliver that push notification. We do
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
        removes your device token, name and followed sensors from our records. If you <strong>uninstall the app</strong>,
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
// Full-screen block shown when the installed app is behind the server's minimum version (major/minor).
// The app can't self-update (sideloaded APK), so it sends the user to the download page to reinstall.
function UpdateRequired({ latest }: { latest: string }) {
  const url = `${APP_SITE}/app`
  return (
    <div className="app update-gate">
      <section className="update-card card">
        <p className="eyebrow">Update required</p>
        <h1>Time to update</h1>
        <p className="update-lede">
          This version (v{APP_VERSION}) is out of date and can no longer run. Version {latest} is
          available — download and install it over this app, then reopen.
        </p>
        <a className="btn" href={url} target="_blank" rel="noopener">Download the update</a>
        <p className="muted update-url">Or open <strong>{APP_SITE.replace('https://', '')}/app</strong> in your browser.</p>
      </section>
    </div>
  )
}

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
        Alerts run only in the app: install it to follow the sensors near you and receive push
        notifications when the live models detect a nearby earthquake. This is a research
        prototype, not an official warning system.
      </p>

      <div className="card app-card">
        <div className="app-card-row">
          <div>
            <p className="app-file">seismicsocal.apk</p>
            <p className="muted app-file-sub">Android · v{APP_VERSION}{sizeMB ? ` · ${sizeMB} MB` : ''}</p>
          </div>
          <a className="btn" href="/seismicsocal.apk" download="seismicsocal.apk">Download APK</a>
        </div>
        <ol className="app-steps">
          <li>Download the APK on your Android device.</li>
          <li>Open it - Android will ask to allow installs from this source. Enable it for your browser.</li>
          <li>Install, open SeismicSoCal, find the sensors near you (city search or your location), follow a region, and tap Subscribe.</li>
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
          Two deep models running on a live 19-station stream, each tested on held-out
          waveforms against the classic seismology baseline.
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

      <Coverage />

      <NearMe />

      <CaLargest />
    </>
  )
}

// The Detect -> Size sequence as an auto-cycling carousel.
// Tracks the previous index so the outgoing card slides left and the incoming enters from the right.
function Carousel({ data }: { data: Seismic }) {
  const byKey = Object.fromEntries(data.tasks.map((t) => [t.key, t]))
  const rows = ROWS.filter((r) => byKey[r.key])
  const n = rows.length
  const CYCLE = 7
  const [idx, setIdx] = useState<{ active: number; prev: number; dir: 'next' | 'prev' }>({ active: 0, prev: 0, dir: 'next' })
  const [paused, setPaused] = useState(false)
  const [remaining, setRemaining] = useState(CYCLE)
  const [evOpen, setEvOpen] = useState(false) // Evidence is shared: one toggle opens both cards

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
    <section className="carousel" aria-roledescription="carousel" aria-label="Detect, Size">
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
  // 4 decimals near 1 so e.g. AUC 0.9998 never rounds up to a perfect-looking 1.000
  const fmt = (v: number) => (t.metric === 'recall' ? v.toFixed(2) : v >= 0.995 ? v.toFixed(4) : v.toFixed(3))
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
      {t.deep_ci && t.baseline_ci && (
        <p className="stat-ci">
          95% CI {fmt(t.deep_ci[0])}–{fmt(t.deep_ci[1])} · {t.baseline_name} {fmt(t.baseline_ci[0])}–{fmt(t.baseline_ci[1])}
          {t.n ? ` · n = ${t.n.toLocaleString()}` : ''}
        </p>
      )}
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
  tech?: string[]
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
          <img src={figure} alt={`${kicker} - evidence on held-out data and replayed live days`} />
          {tech && tech.length > 0 && (
            <div className="evidence-tech">
              <p className="evidence-tech-h">How it gets the result</p>
              <ol>{tech.map((t, k) => <li key={k}>{t}</li>)}</ol>
            </div>
          )}
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
const quakeHref = (e: UsgsEvent) =>
  `/quake?${new URLSearchParams({ lat: String(e.lat), lon: String(e.lon), mag: String(e.mag), t: String(e.time ?? ''), place: e.place })}`

// "Light (IV) at your home · 34 km away" under a quake, when the user has saved a home on this device.
function HomeShakingLine({ lat, lon, mag, term = 0 }: { lat: number; lon: number; mag: number; term?: number }) {
  const [line, setLine] = useState<string | null>(null)
  useEffect(() => {
    let on = true
    const run = () => homeShaking(lat, lon, mag, term)
      .then((h) => on && setLine(h ? `Est. at your home: ${h.label} (${h.roman}) · ${Math.round(h.km)} km away` : null))
      .catch(() => on && setLine(null))
    run()
    window.addEventListener('home-changed', run)
    return () => { on = false; window.removeEventListener('home-changed', run) }
  }, [lat, lon, mag, term])
  return line ? <p className="home-shaking">{line}</p> : null
}

// One quake (from a list or a tapped alert): size, place, time and the estimated shaking at the user's home.
function QuakeDetail() {
  const q = new URLSearchParams(window.location.search)
  const lat = Number(q.get('lat')), lon = Number(q.get('lon')), mag = Number(q.get('mag')), term = Number(q.get('term') || 0)
  const t = Number(q.get('t')), place = q.get('place') || 'Southern California', stage = q.get('stage')
  const [h, setH] = useState<Awaited<ReturnType<typeof homeShaking>> | 'none' | null>(null)
  useEffect(() => { homeShaking(lat, lon, mag, term).then((r) => setH(r ?? 'none')).catch(() => setH('none')) }, [lat, lon, mag, term])
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(mag)) return <NotFound />
  return (
    <section className="legal quake-detail">
      <p className="eyebrow">Earthquake{stage === 'early' ? ' · size being confirmed' : ''}</p>
      <h1>M{mag.toFixed(1)} · {place}</h1>
      <p className="legal-updated">{t ? new Date(t).toLocaleString() : ''} · {lat.toFixed(2)}, {lon.toFixed(2)}</p>
      <h2>How strong was it at your home?</h2>
      {h === null && <p className="muted">Estimating…</p>}
      {h === 'none' && <p>Set your home in <a href="/" onClick={(e) => { e.preventDefault(); navigate('/') }}>Alert me near me</a> (use your
        location or search a city) to see the estimated shaking there. Your home is stored only on this device.</p>}
      {h && h !== 'none' && (
        <div className="card mmi-card">
          <p className="mmi-big">{h.label} <span>MMI {h.roman}</span></p>
          <p>{h.desc[0].toUpperCase() + h.desc.slice(1)}. About {Math.round(h.km)} km from {h.home.label}.</p>
          <p className="muted">An estimate from the magnitude, the distance and the ground type at your home{term ? ', adjusted by how strongly this quake shook our sensors' : ''}.
            Checked against USGS “Did You Feel It?” reports: within one level in about 9 of 10 places. Not an official measurement.</p>
        </div>
      )}
      <p className="app-back"><a className="back-link" href="/" onClick={(e) => { e.preventDefault(); navigate('/') }}><Arrow dir="left" />Back to the console</a></p>
    </section>
  )
}

function QuakeList({ title, events }: { title: string; events?: UsgsEvent[] }) {
  return (
    <div className="qlist">
      <p className="eyebrow">{title}</p>
      {events && events.length === 0 && <p className="muted">None recorded.</p>}
      <ul className="recent-list">
        {events?.map((e) => (
          <li key={e.id}>
            <a className="recent-link" href={quakeHref(e)} onClick={(ev) => { ev.preventDefault(); navigate(quakeHref(e)) }}>
              <span className={`m ${e.mag >= 5 ? 'hi' : ''}`}>M{e.mag.toFixed(1)}</span>
              <span className="place">{e.place}</span>
              <span className="r">{fmtDate(e.time)}</span>
            </a>
            {e.caught && <CaughtBadge c={e.caught} />}
            <HomeShakingLine lat={e.lat} lon={e.lon} mag={e.mag} />
          </li>
        ))}
      </ul>
    </div>
  )
}

// How the live pipeline did on this USGS quake (matched within 30 s and 60 km of its origin).
function CaughtBadge({ c }: { c: NonNullable<UsgsEvent['caught']> }) {
  if (c.status === 'caught')
    return (
      <p className="caught caught-yes">
        Caught by our model{c.mag != null ? ` · estimated M${c.mag.toFixed(1)}` : ''}
        {c.n_stations ? ` · ${c.n_stations} sensors` : ''}
      </p>
    )
  if (c.status === 'seen') return <p className="caught caught-seen">Seen by our sensors · not confirmed</p>
  return <p className="caught caught-no">Not caught by our model</p>
}

// Under the quake list: why a quake can be seen but not confirmed, and how often quakes are caught
// (80 held-out replay days + the 2026-10-07 swarm tuning; HOW_IT_WORKS.md sections 6.3-6.4).
function CatchNotes() {
  const [open, setOpen] = useState(false)
  return (
    <div className={`evidence ${open ? 'open' : ''}`}>
      <button className="evidence-summary" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        What the marks mean
      </button>
      <div className="evidence-wrap">
        <div className="evidence-inner">
          <div className="evidence-tech">
            <p className="evidence-tech-h">Why a quake can be seen but not confirmed</p>
            <p>
              “Seen” means at least one sensor picked up the quake, but it didn’t pass the checks that make a detection
              trustworthy enough to alert on. Usually the quake was small, but not always. A quake is confirmed only when:
            </p>
            <ol>
              <li><b>At least 3 sensors pick it up.</b> The most common miss. A small quake stands out from background
                noise only at the nearest one or two sensors, which are tens of kilometres apart.</li>
              <li><b>Their timings point to one place</b> (within 1.5 s). Two quakes overlapping, or one mistimed pick,
                breaks this.</li>
              <li><b>No more than one working sensor closer to it stayed quiet.</b> A real quake reaches nearer sensors
                first; this rule throws out noise that hits three sensors by coincidence, and occasionally a real small quake.</li>
              <li><b>The nearest sensor that picked it is within 120 km.</b> Quakes outside the network (north of
                Ridgecrest, south of the border) can fail this.</li>
              <li><b>It isn’t within 30 seconds and 100 km of a quake already declared</b>, which is treated as that
                quake’s echo. In busy aftershock sequences this can hide a second quake.</li>
            </ol>
            <p className="evidence-tech-h">How often it catches quakes</p>
            <p>The exact live code, replayed on 80 days of archived data from 2022–2026 that it never trained on:</p>
            <ol>
              <li>About <b>7 in 10 quakes of M3 and up</b> were caught (68%); about 8 in 10 outside busy aftershock
                sequences with enough sensors online (78%). About half of M2+ quakes (52%), and few below M2.</li>
              <li><b>Every one of its 145 alerts was a real quake.</b> 92% were placed within 60 km; the other 12 were
                outside the network and placed 63–145 km off.</li>
              <li>Sizes were within 0.12 magnitude units on average; median location error 3–4 km.</li>
              <li>About 7 detections a week on ordinary days don’t match a catalogued quake. Those are logged, never sent.</li>
            </ol>
            <p className="evidence-tech-h">The aftershock fix (October 2026)</p>
            <p>
              Most misses were in aftershock swarms, where a second quake arrived before the system was ready for it. We tried
              32 combinations of the echo window, the rest time each sensor takes after a pick, and splitting overlapping
              detections, on separate tuning days only. The best that added no false detections and no double counts was a
              <b> 30-second echo window (was 120 s) and a 45-second sensor rest (was 60 s), without splitting</b>. Checked once
              on the 80 days, it raised M3+ catches from 71% to 78% (counting quakes with 3+ sensors online), with no quake
              counted twice and real-detection rate 85% → 84%.
            </p>
          </div>
        </div>
      </div>
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
          <p className="source-note">
            Only quakes our live pipeline can catch: M2+ with 3+ sensors within 100 km · Source: USGS ·
            each is checked against what our models caught in real time
          </p>
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
                {d === 'err' && <p className="muted">The USGS feed is unavailable right now. Try again in a minute.</p>}
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
      <CatchNotes />
    </section>
  )
}

// Alert-speed choices (first message only). Numbers: median after the quake starts on 20 replayed days
// (HOW_IT_WORKS.md section 5.3); live adds a few seconds of stream delay.
const SPEEDS: { key: AlertMode; name: string; when: string; desc: string }[] = [
  { key: 'standard', name: 'Standard', when: 'first notice ~30–35 s', desc: 'Most safeguards. Fewer notices that later get retracted, and a closer first size.' },
  { key: 'fast', name: 'Fast', when: 'first notice ~25 s', desc: 'Earlier, from a rougher 2-second size. Expect more retractions, occasionally for a quake that turns out too small to feel.' },
]

// "Use my location" selects the nearest REGION (all of its sensors), but only if one of its sensors is
// within this cap (roughly an M3.5's felt distance) — farther than that, the user picks manually.
const NEAR_CAP_KM = 150

// Great-circle distance in km (haversine), for ranking stations by how close they are to the user.
function kmBetween(aLat: number, aLon: number, bLat: number, bLon: number) {
  const R = 6371
  const rad = (d: number) => (d * Math.PI) / 180
  const dLat = rad(bLat - aLat)
  const dLon = rad(bLon - aLon)
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(rad(aLat)) * Math.cos(rad(bLat)) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(s))
}

// Stations grouped by region, in the network's own order (src/eq/network.py via /api/stations).
function regionsOf(stations: Station[]): Map<string, Station[]> {
  const m = new Map<string, Station[]>()
  for (const st of stations) m.set(st.region, [...(m.get(st.region) ?? []), st])
  return m
}

// Are two sets of station codes identical? Used to tell a live subscription from a pending edit.
function sameSet(a: Set<string>, b: Set<string>) {
  return a.size === b.size && [...a].every((x) => b.has(x))
}

function NearMe() {
  // Restore the persisted subscription so the choice survives closing the app: the stations the
  // device is subscribed to are shown pre-selected (and locked-in), not blank.
  const savedSubs = subscribedStations()
  const [name, setName] = useState(subscribedName())
  const [stations, setStations] = useState<Station[]>([])
  const [dist, setDist] = useState<Record<string, number>>({})   // code -> km, empty until located
  const [selected, setSelected] = useState<Set<string>>(new Set(savedSubs))
  const [subscribedSet, setSubscribedSet] = useState<Set<string>>(new Set(savedSubs)) // live server subscription
  const [located, setLocated] = useState(false)
  const [msg, setMsg] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [city, setCity] = useState('')
  const [stateName, setStateName] = useState('CA')
  const [searching, setSearching] = useState(false)
  const [opened, setOpened] = useState<Set<string>>(new Set())   // regions the user expanded by hand
  const [mode, setMode] = useState<AlertMode>(subscribedMode())   // alert speed for the first message
  const [home, setHomeState] = useState<Home | null>(getHome())          // for shaking estimates (device only)
  const [liveMode, setLiveMode] = useState<AlertMode>(subscribedMode())

  const subscribed = subscribedSet.size > 0                    // subscribed iff we track live stations
  const dirty = !sameSet(selected, subscribedSet) || mode !== liveMode   // selection or speed differs from what's live

  useEffect(() => {
    getStations()
      .then((list) => {
        setStations(list)
        // The network was re-chosen (stations that never streamed live were replaced). Drop saved codes
        // the network no longer has; the server already moved those subscriptions to the nearest new sensor.
        const known = new Set(list.map((s) => s.code))
        const stale = savedSubs.filter((c) => !known.has(c))
        if (stale.length) {
          setSelected((prev) => new Set([...prev].filter((c) => known.has(c))))
          setSubscribedSet((prev) => new Set([...prev].filter((c) => known.has(c))))
          setMsg({ kind: 'err', text: `The sensor network was upgraded and ${stale.join(', ')} ${stale.length > 1 ? 'were' : 'was'} retired. Your alerts were moved to the nearest new sensor — review your sensors below and tap Subscribe/Update to confirm.` })
        }
      })
      .catch(() => setMsg({ kind: 'err', text: 'Couldn’t load the sensor list — is the backend running?' }))
  }, [])

  // Given the user's coordinates, compute distance to every station and auto-select the nearest
  // few within the cap. The coordinates are used only here to rank stations — they are never stored;
  // only the chosen station codes are sent to the server.
  const applyLocation = (lat: number, lon: number, label = 'your location') => {
    setHome(lat, lon, label).then(setHomeState).catch(() => {})   // saved on this device only (shaking estimates)
    const d: Record<string, number> = {}
    for (const st of stations) d[st.code] = kmBetween(lat, lon, st.lat, st.lon)
    setDist(d)
    setLocated(true)
    const best = [...regionsOf(stations)].map(([r, sts]) => [r, Math.min(...sts.map((x) => d[x.code])), sts] as const)
      .sort((a, b) => a[1] - b[1])[0]
    if (best && best[1] <= NEAR_CAP_KM) {
      const [r, km, sts] = best
      setSelected(new Set(sts.map((x) => x.code)))
      setOpened(new Set([r]))
      setMsg({ kind: 'ok', text: `Selected ${r} — its ${sts.length} sensors (nearest ${Math.round(km)} km from you). Turn individual sensors off below, or add other regions.` })
    } else {
      setMsg({ kind: 'err', text: `No sensor within ${NEAR_CAP_KM} km — you may be outside the covered region. You can still pick a region below.` })
    }
  }

  const searchLocation = async () => {
    if (!city.trim()) return
    setSearching(true)
    setMsg(null)
    try {
      const r = await geocode(`${city.trim()}, ${stateName.trim() || 'CA'}`)
      applyLocation(r.lat, r.lon, city.trim())
    } catch (err) {
      setMsg({ kind: 'err', text: `Couldn’t find that place: ${(err as Error).message}` })
    } finally {
      setSearching(false)
    }
  }

  const useMyLocation = async () => {
    setMsg(null)
    try {
      const { lat, lon } = await getMyLocation()
      applyLocation(lat, lon)
    } catch {
      setMsg({ kind: 'err', text: 'Couldn’t read your location - allow the location permission, or search by city.' })
    }
  }

  const toggle = (code: string) => setSelected((prev) => {
    const next = new Set(prev)
    if (next.has(code)) next.delete(code)
    else next.add(code)
    return next
  })

  // Tapping a region follows ALL of its sensors (and opens it so single sensors can be turned off);
  // tapping a region that has any sensor selected clears the whole region.
  const toggleRegion = (region: string, codes: string[]) => {
    const any = codes.some((c) => selected.has(c))
    setSelected((prev) => {
      const next = new Set(prev)
      codes.forEach((c) => (any ? next.delete(c) : next.add(c)))
      return next
    })
    setOpened((prev) => {
      const next = new Set(prev)
      if (any) next.delete(region)
      else next.add(region)
      return next
    })
  }
  const toggleOpen = (region: string) => setOpened((prev) => {
    const next = new Set(prev)
    if (next.has(region)) next.delete(region)
    else next.add(region)
    return next
  })

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!selected.size) {
      setMsg({ kind: 'err', text: 'Pick at least one sensor to be alerted for.' })
      return
    }
    const wasSubscribed = subscribed
    setBusy(true)
    setMsg(null)
    try {
      // register-push upserts by device token, so the same call both subscribes and swaps the set.
      const pushed = await enablePush({ name, stations: [...selected], mode })
      if (pushed === null) {
        // web build: no native push, so there's nothing to subscribe to here
        setMsg({ kind: 'err', text: 'Alerts arrive as push notifications - install the SeismicSoCal app to subscribe.' })
        return
      }
      setSubscribedSet(new Set(selected))
      setLiveMode(mode)
      const n = selected.size
      setMsg({
        kind: 'ok',
        text: wasSubscribed
          ? `Updated - now alerting on ${n} sensor${n > 1 ? 's' : ''}.`
          : `Subscribed to ${n} sensor${n > 1 ? 's' : ''} - push alerts are enabled on this device.`,
      })
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
      setSubscribedSet(new Set())      // no live subscription; selection is kept for easy re-subscribe
      setMsg({ kind: 'ok', text: 'Unsubscribed - this device will no longer receive alerts.' })
    } catch (err) {
      setMsg({ kind: 'err', text: `Couldn’t unsubscribe: ${(err as Error).message}.` })
    } finally {
      setBusy(false)
    }
  }

  // Regions (from the network's station list), nearest-first once located.
  const regions = [...regionsOf(stations)]
  if (located) {
    const near = (sts: Station[]) => Math.min(...sts.map((x) => dist[x.code] ?? Infinity))
    regions.sort((a, b) => near(a[1]) - near(b[1]))
  }

  return (
    <section className="nearme">
      <p className="eyebrow">Alerts</p>
      <h2>Alert me near me</h2>
      <p className="nearme-lede">
        The models watch a live 19-sensor Southern California stream. When a quake is located near a
        sensor you follow, you get a first notice about half a minute after it begins (or sooner on the Fast
        setting) and a confirmed magnitude by about a minute (or a retraction if it turns out too small to feel). Follow a region, then fine-tune its sensors.
        This is rapid detection, not an official warning.
        {!isNativeApp() && ' Alerts are available in the SeismicSoCal app.'}
      </p>

      <div className="card">
        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="nm-name">Name</label>
            <input id="nm-name" value={name} onChange={(e) => setName(e.target.value)} required placeholder="Your name" />
          </div>
          <div className="field">
            <label htmlFor="nm-city">Find sensors near you</label>
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

          <button type="button" className="btn-outline locate-btn" onClick={useMyLocation}>
            Use my location
          </button>

          {home && (
            <p className="home-note">
              Home for shaking estimates: <strong>{home.label}</strong> (saved on this device only) ·{' '}
              <button type="button" className="linklike" onClick={() => { clearHome(); setHomeState(null) }}>Clear</button>
            </p>
          )}

          <fieldset className="station-picker">
            <legend>
              Regions{located ? ' — distance from you' : ''}
              <span className="picker-hint"> · tap a region to follow all its sensors, then turn single sensors off</span>
            </legend>
            <ul className="region-list">
              {regions.map(([region, sts]) => {
                const codes = sts.map((x) => x.code)
                const nSel = codes.filter((c) => selected.has(c)).length
                const nLive = codes.filter((c) => selected.has(c) && subscribedSet.has(c)).length
                const expanded = nSel > 0 || opened.has(region)
                const near = located ? Math.min(...codes.map((c) => dist[c] ?? Infinity)) : undefined
                const cls = nSel && nLive === nSel ? ' subscribed' : nSel ? ' on' : ''
                const label = !nSel ? 'Off' : nLive === nSel ? `Subscribed ${nSel}/${codes.length}` : `Selected ${nSel}/${codes.length}`
                return (
                  <li key={region} className="region-item">
                    <div className="region-head">
                      <button type="button" className={`station-row region-row${cls}`} aria-pressed={nSel > 0}
                        onClick={() => toggleRegion(region, codes)}>
                        <span className="region-name">{region}</span>
                        <span className="station-dist">
                          {codes.length} sensors{near !== undefined && isFinite(near) ? ` · ${Math.round(near)} km` : ''}
                        </span>
                        <span className="station-toggle">{label}</span>
                      </button>
                      <button type="button" className="region-caret" aria-expanded={expanded}
                        aria-label={`${expanded ? 'Hide' : 'Show'} ${region} sensors`} onClick={() => toggleOpen(region)}
                        disabled={nSel > 0}>
                        {expanded ? '−' : '+'}
                      </button>
                    </div>
                    {expanded && (
                      <ul className="station-list region-sensors">
                        {sts.map((st) => {
                          const on = selected.has(st.code)
                          const live = subscribedSet.has(st.code)
                          const km = dist[st.code]
                          const scls = live && on ? ' subscribed' : on ? ' on' : ''
                          const slabel = live && on ? 'Subscribed' : on ? 'Selected' : 'Off'
                          return (
                            <li key={st.code}>
                              <button type="button" className={`station-row${scls}`} aria-pressed={on} onClick={() => toggle(st.code)}>
                                <span className="station-code">{st.code}</span>
                                <span className="station-dist">{located && km !== undefined ? `${Math.round(km)} km` : '—'}</span>
                                <span className="station-toggle">{slabel}</span>
                              </button>
                            </li>
                          )
                        })}
                      </ul>
                    )}
                  </li>
                )
              })}
            </ul>
          </fieldset>

          <fieldset className="speed-picker">
            <legend>Alert speed <span className="picker-hint"> · for the first notice; the confirmed size is the same</span></legend>
            <div className="speed-options" role="radiogroup" aria-label="Alert speed">
              {SPEEDS.map((s) => (
                <button key={s.key} type="button" role="radio" aria-checked={mode === s.key}
                  className={`speed-option${mode === s.key ? ' on' : ''}`} onClick={() => setMode(s.key)}>
                  <span className="speed-name">{s.name}</span>
                  <span className="speed-when">{s.when}</span>
                  <span className="speed-desc">{s.desc}</span>
                </button>
              ))}
            </div>
          </fieldset>

          <div className="actions">
            {!isNativeApp() ? (
              // Web has no push channel - subscription lives in the app only.
              <button type="button" className="btn btn-struck" disabled aria-disabled="true">
                Subscribe
              </button>
            ) : !subscribed ? (
              // Not subscribed: pick stations, then Subscribe.
              <button type="submit" className="btn" disabled={busy || !selected.size}>
                {busy ? 'Subscribing…' : `Subscribe${selected.size ? ` (${selected.size})` : ''}`}
              </button>
            ) : (
              // Subscribed: Unsubscribe all, plus Update to commit a changed selection (swap sensors).
              <>
                {dirty && selected.size > 0 && (
                  <button type="submit" className="btn" disabled={busy}>
                    {busy ? 'Updating…' : `Update (${selected.size})`}
                  </button>
                )}
                <button
                  type="button"
                  className={dirty && selected.size > 0 ? 'btn-outline' : 'btn'}
                  onClick={unsubscribe}
                  disabled={busy}
                >
                  {busy ? 'Unsubscribing…' : 'Unsubscribe all'}
                </button>
              </>
            )}
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
