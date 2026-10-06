// Ten homepage design drafts (/home1../home10) for side-by-side comparison. Each is a full page built from
// the SAME real components and numbers as the live site (seismic.json, the coverage map, the alert picker,
// the USGS list, the evidence figures); only layout, typography, rhythm and color differ.
// Numbers not in seismic.json come from FACTS below, each with its source in the repo.
import { useEffect, useState, type ReactNode } from 'react'
import type { Seismic, Task } from '../seismic'
import Coverage from '../Coverage'
import { Arrow, CaLargest, Carousel, Evidence, NearMe, ROWS, Trace, navigate } from '../App'
import { liveDetail } from '../nearme'
import './drafts.css'

// Measured numbers (HOW_IT_WORKS.md section 4 + 6, data/processed/v2/replay/score_test_2stage.json).
const FACTS = {
  replayDays: 10,                 // held-out test days replayed through the exact live code
  confirmedReal: 93,              // % of confirmed events matching a catalogued quake
  chance: 0,                      // % for the same declarations shifted +1 h
  pushes: 5, falsePushes: 0,
  locErrKm: 2.5,                  // median location error
  provisionalS: 33, confirmedS: 55,   // median seconds after origin (replay)
  falseConfirmedPerWeek: 3.5,     // confirmed but unmatched (all sized <= M2.9, never pushed)
  catchRate: [['M1–1.5', 9], ['M1.5–2', 48], ['M2–2.5', 81], ['M2.5–3', 75], ['M3+', 86]] as [string, number][],
  quickMae: 0.24, quickM3Pass: 93, quickSmallPass: 1.1,
  detWindows: 50743, years: '2000–2026',
  fprLive: 1.41,                  // % of 30 s test-noise windows over the live 0.6 trigger
}

const STEPS = [
  { k: 'Wave', t: 'A quake starts', b: 'P-waves travel outward at about 6 km/s. The nearest of 19 live sensors feels them first, seconds later.', f: '19 sensors · 100 samples/s' },
  { k: 'Detect', t: 'Is it a quake?', b: 'Every 2 seconds each sensor hands its last 30 seconds to the detector, which tells earthquakes from traffic, wind and glitches.', f: 'AUC 0.9998 vs STA/LTA 0.816' },
  { k: 'Locate', t: 'Where is it?', b: 'Three sensors must agree. Their P-wave arrival times are fitted to one source, and a closer sensor staying silent vetoes it.', f: `median error ${FACTS.locErrKm} km` },
  { k: 'Size', t: 'How big?', b: 'A 4-second quick check gives a first size; the graph network then reads 30 seconds from every nearby sensor for the confirmed magnitude.', f: 'R² 0.951 vs 0.886 baseline' },
  { k: 'Alert', t: 'Who should know?', b: 'Phones following a nearby sensor get a provisional notice, then the confirmed size, or a retraction if it was too small to feel.', f: `~${FACTS.provisionalS} s, then ~${FACTS.confirmedS} s` },
]

const fmt = (v: number) => (v >= 0.995 ? v.toFixed(4) : v.toFixed(3))
const task = (d: Seismic, k: string) => d.tasks.find((t) => t.key === k) as Task

function useFont(href: string | null) {
  useEffect(() => {
    if (!href || document.querySelector(`link[data-draft-font="${href}"]`)) return
    const l = document.createElement('link')
    l.rel = 'stylesheet'
    l.href = href
    l.dataset.draftFont = href
    document.head.appendChild(l)
  }, [href])
}
const gf = (q: string) => `https://fonts.googleapis.com/css2?${q}&display=swap`

// Deep model vs baseline, each as a point with its 95% CI, on a shared axis. Honest comparison as a picture.
function Range({ t, lo, hi }: { t: Task; lo: number; hi: number }) {
  const W = 320, x = (v: number) => ((v - lo) / (hi - lo)) * W
  const row = (y: number, v: number, ci: [number, number] | undefined, name: string, cls: string) => (
    <g className={cls}>
      <text x={0} y={y - 8} className="rg-name">{name} {fmt(v)}</text>
      {ci && <line x1={x(ci[0])} x2={x(Math.min(ci[1], hi))} y1={y} y2={y} className="rg-ci" />}
      <circle cx={x(v)} cy={y} r={5} className="rg-dot" />
    </g>
  )
  return (
    <svg className="rg" viewBox={`-4 0 ${W + 8} 92`} role="img"
      aria-label={`${t.name}: deep ${fmt(t.deep)}, ${t.baseline_name} ${fmt(t.baseline)}`}>
      {row(28, t.deep, t.deep_ci, 'Deep model', 'rg-deep')}
      {row(66, t.baseline, t.baseline_ci, t.baseline_name, 'rg-base')}
      <line x1={0} x2={W} y1={84} y2={84} className="rg-axis" />
      <text x={0} y={92} className="rg-tick">{lo}</text>
      <text x={W} y={92} className="rg-tick" textAnchor="end">{hi} {t.metric}</text>
    </svg>
  )
}

function CatchBars() {
  return (
    <div className="cb" role="img" aria-label="Share of in-coverage quakes caught by size, replayed days">
      {FACTS.catchRate.map(([m, p]) => (
        <div className="cb-row" key={m}>
          <span className="cb-m">{m}</span>
          <span className="cb-track"><span className="cb-fill" style={{ width: `${p}%` }} /></span>
          <span className="cb-p">{p}%</span>
        </div>
      ))}
    </div>
  )
}

function Fig({ k, cap }: { k: 'detection' | 'magnitude'; cap: ReactNode }) {
  const row = ROWS.find((r) => r.key === k)!
  return (
    <figure className="dfig">
      <img src={row.figure} alt={`${row.kicker}: evidence on held-out data and replayed live days`} loading="lazy" />
      <figcaption>{cap}</figcaption>
    </figure>
  )
}

function Link({ to, children }: { to: string; children: ReactNode }) {
  return <a href={to} onClick={(e) => { e.preventDefault(); navigate(to) }}>{children}</a>
}

function Foot() {
  return (
    <footer className="dfoot">
      <p>Research &amp; education · real Southern California network data · not an official warning system</p>
      <p><Link to="/privacy">Privacy</Link> · <Link to="/health">Model health</Link> · <Link to="/app">Get the app</Link></p>
    </footer>
  )
}

const NAMES = ['The sequence', 'Seismogram', 'Against the baseline', 'Map first', 'Front page', 'The paper',
  'Instrument', 'Questions', 'Drench', '55 seconds']

function Switcher({ n }: { n: number }) {
  const go = (i: number) => navigate(`/home${((i - 1 + 10) % 10) + 1}`)
  useEffect(() => { window.scrollTo(0, 0) }, [n])
  return (
    <nav className="dsw" aria-label="Design drafts">
      <button onClick={() => go(n - 1)} aria-label="Previous draft"><Arrow dir="left" /></button>
      <span><b>{n}</b>/10 · {NAMES[n - 1]}</span>
      <button onClick={() => go(n + 1)} aria-label="Next draft"><Arrow /></button>
      <a href="/" onClick={(e) => { e.preventDefault(); navigate('/') }}>Current site</a>
    </nav>
  )
}

// ---------------------------------------------------------------- 1. The sequence (story scroll, sticky rail)
function D1({ data }: { data: Seismic }) {
  useFont(gf('family=Literata:opsz,wght@7..72,400;7..72,600'))
  return (
    <div className="d1">
      <header className="d1-hero">
        <p className="d1-brand">SeismicSoCal</p>
        <h1>What happens in the minute after Southern California shakes</h1>
        <p className="d1-lede">A live network of 19 seismometers, two neural networks, and a strict rule about evidence.
          Follow one earthquake from the first wave to the alert on a phone.</p>
        <Trace />
      </header>
      <ol className="d1-steps">
        {STEPS.map((s, i) => (
          <li key={s.k} className="d1-step">
            <div className="d1-rail"><span>{i + 1}</span>{s.k}</div>
            <div className="d1-body">
              <h2>{s.t}</h2>
              <p>{s.b}</p>
              <p className="d1-fact">{s.f}</p>
              {i === 1 && <Fig k="detection" cap="Held-out windows: the detector against the classic STA/LTA trigger, and how often each size of quake was caught on replayed days." />}
              {i === 3 && <Fig k="magnitude" cap="Estimated against catalogue magnitude on 937 held-out quakes; the quick check against the full estimate." />}
            </div>
          </li>
        ))}
      </ol>
      <section className="d1-proof">
        <h2>Does it hold up on real days?</h2>
        <p>We replayed {FACTS.replayDays} days it had never seen through the exact code that runs live. {FACTS.confirmedReal}% of the
          quakes it confirmed were real (a random-time baseline scores {FACTS.chance}%). It sent {FACTS.pushes} alerts and none were false.</p>
        <CatchBars />
      </section>
      <Coverage />
      <NearMe />
      <CaLargest />
      <p className="d1-meta">{data.dataset.events.toLocaleString()} catalogued quakes · {data.dataset.stations} stations · M{data.dataset.mag_min}–{data.dataset.mag_max} · {FACTS.years}</p>
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 2. Seismogram (full-bleed annotated trace)
function D2({ data }: { data: Seismic }) {
  useFont(gf('family=Archivo:wdth,wght@62..125,400;62..125,700;62..125,800'))
  const det = task(data, 'detection'), mag = task(data, 'magnitude')
  return (
    <div className="d2">
      <header className="d2-hero">
        <div className="d2-trace">
          <Trace />
          <span className="d2-mark d2-p" style={{ left: '38%' }}>P · detected</span>
          <span className="d2-mark d2-s" style={{ left: '52%' }}>3 sensors agree · located</span>
          <span className="d2-mark d2-a" style={{ left: '71%' }}>sized · alert</span>
        </div>
        <h1>SEISMIC<br />SOCAL</h1>
        <p className="d2-lede">Earthquakes read straight off the wire. Two neural networks listen to 19 live seismometers
          and turn the first seconds of ground motion into a location, a magnitude and an alert.</p>
      </header>
      <section className="d2-band">
        <p>Detection AUC <b>{fmt(det.deep)}</b>, against {fmt(det.baseline)} for the classic STA/LTA trigger.</p>
        <p>Magnitude R² <b>{fmt(mag.deep)}</b>, against {fmt(mag.baseline)} for amplitude and distance.</p>
        <p><b>{FACTS.falsePushes}</b> false alerts out of {FACTS.pushes}, over {FACTS.replayDays} replayed real days.</p>
      </section>
      <div className="d2-wrap">
        <Carousel data={data} />
        <Coverage />
        <NearMe />
        <CaLargest />
      </div>
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 3. Against the baseline (comparison-first)
function D3({ data }: { data: Seismic }) {
  useFont(gf('family=Manrope:wght@400;600;800'))
  const det = task(data, 'detection'), mag = task(data, 'magnitude')
  const [ev, setEv] = useState(false)
  return (
    <div className="d3">
      <header className="d3-hero">
        <h1>Deep learning, held to the classic standard.</h1>
        <p>Every model here is measured against the method seismologists already trust, on data it never saw,
          with 95% confidence intervals. If the deep model weren’t better, this page would say so.</p>
      </header>
      <section className="d3-grid">
        <article>
          <h2>Is it an earthquake?</h2>
          <Range t={det} lo={0.78} hi={1} />
          <p>On {det.n?.toLocaleString()} held-out 30-second windows the detector separates quakes from noise almost perfectly; the
            STA/LTA trigger, the classic method on the same input, does not.</p>
          <Evidence figure="detect_evidence.png" kicker="Detect" tech={ROWS[0].tech} open={ev} onToggle={() => setEv((o) => !o)} />
        </article>
        <article>
          <h2>How big is it?</h2>
          <Range t={mag} lo={0.84} hi={0.98} />
          <p>On {mag.n} held-out quakes (M2–5.2) the graph network’s magnitude explains more of the variance than the
            amplitude-and-distance formula, and the intervals don’t overlap.</p>
          <Fig k="magnitude" cap="Left: deep estimate vs catalogue. The baseline’s scatter is visibly wider." />
        </article>
      </section>
      <section className="d3-live">
        <h2>And on live-like days</h2>
        <div className="d3-pair">
          <div><p className="d3-k">Deep pipeline</p><p className="d3-v">{FACTS.pushes} alerts, {FACTS.falsePushes} false</p></div>
          <div><p className="d3-k">The system it replaced, same days</p><p className="d3-v">6 alerts, 6 false</p></div>
        </div>
      </section>
      <div className="d3-wrap"><Coverage /><NearMe /><CaLargest /></div>
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 4. Map first (residents; locked design system)
function D4({ data, live }: { data: Seismic; live: boolean }) {
  return (
    <div className="d4">
      <header className="d4-hero">
        <p className="d4-live"><span className={`dot ${live ? 'on' : ''}`} /> {live ? 'Listening live to 19 sensors' : 'Live stream offline'}</p>
        <h1>Does it cover where you live?</h1>
        <p>Where at least three sensors are within 100 km, a quake is located, sized and can alert your phone,
          usually within a minute of it starting.</p>
      </header>
      <div className="d4-map"><Coverage /></div>
      <section className="d4-how">
        <div><h3>Detect</h3><p>Two models, tested on held-out data. Detection AUC {fmt(task(data, 'detection').deep)}.</p></div>
        <div><h3>Size</h3><p>Typically within 0.1 of the official magnitude (MAE 0.10 on 937 held-out quakes).</p></div>
        <div><h3>Alert</h3><p>A first notice ~{FACTS.provisionalS} s after it begins, confirmed ~{FACTS.confirmedS} s.</p></div>
      </section>
      <NearMe />
      <Carousel data={data} />
      <CaLargest />
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 5. Front page (broadsheet)
function D5({ data }: { data: Seismic }) {
  useFont(gf('family=Spectral:ital,wght@0,400;0,600;0,800;1,400'))
  const det = task(data, 'detection'), mag = task(data, 'magnitude')
  return (
    <div className="d5">
      <header className="d5-mast">
        <p className="d5-date">{new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}</p>
        <h1 className="d5-title">The SeismicSoCal Record</h1>
        <p className="d5-sub">Research prototype · Southern California · not an official warning system</p>
      </header>
      <div className="d5-cols">
        <article className="d5-lead">
          <h2>Neural networks now locate and size local quakes within a minute</h2>
          <p className="d5-deck">A 19-station live stream, two deep models and a replay of real days put the classic methods on notice.</p>
          <Trace />
          <p>SeismicSoCal listens to 19 seismometers that stream in real time across Southern California. Every two seconds
            a detector reads each station’s last half-minute; when three agree on a single source, the quake is located,
            and a graph network reads every nearby station to estimate its magnitude.</p>
          <p>Measured on data it had never seen, the detector scores an AUC of {fmt(det.deep)} where the textbook STA/LTA
            trigger scores {fmt(det.baseline)}. The magnitude model reaches R² {fmt(mag.deep)} against {fmt(mag.baseline)} for the
            amplitude-and-distance formula.</p>
          <p>Replayed through the live code on {FACTS.replayDays} held-out days, {FACTS.confirmedReal}% of confirmed quakes were real and none of
            its {FACTS.pushes} alerts were false.</p>
        </article>
        <aside className="d5-side">
          <h3>By the numbers</h3>
          <dl>
            <dt>Catalogued quakes trained on</dt><dd>{data.dataset.events.toLocaleString()}</dd>
            <dt>Median location error</dt><dd>{FACTS.locErrKm} km</dd>
            <dt>First notice / confirmed</dt><dd>~{FACTS.provisionalS} s / ~{FACTS.confirmedS} s</dd>
            <dt>M3+ quakes caught in coverage</dt><dd>86%</dd>
          </dl>
          <h3>The fine print</h3>
          <p>Detection after a quake begins, not prediction. Coverage is thin offshore and south of the border.</p>
        </aside>
      </div>
      <div className="d5-below">
        <Fig k="detection" cap="Fig. The detector against STA/LTA, on held-out windows and replayed days." />
        <Fig k="magnitude" cap="Fig. Estimated against official magnitude on 937 held-out quakes." />
      </div>
      <div className="d5-wrap"><Coverage /><NearMe /><CaLargest /></div>
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 6. The paper (research-paper layout)
function D6({ data }: { data: Seismic }) {
  useFont(gf('family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400'))
  const det = task(data, 'detection'), mag = task(data, 'magnitude')
  return (
    <div className="d6">
      <header className="d6-head">
        <h1>Rapid earthquake detection, location and sizing on a live 19-station Southern California stream</h1>
        <p className="d6-auth">SeismicSoCal · research prototype · {new Date().getFullYear()}</p>
      </header>
      <section className="d6-abs">
        <h2>Abstract</h2>
        <p>We present a deep-learning pipeline that detects (CNN → Transformer), locates (grid search over picked P arrivals)
          and sizes (CNN → GNN → Transformer) earthquakes on the real-time SeedLink stream of 19 SCSN stations. On a chronological
          held-out test set the detector reaches ROC-AUC {fmt(det.deep)} (95% CI {det.deep_ci?.map(fmt).join('–')}) against {fmt(det.baseline)} for STA/LTA;
          the magnitude ensemble reaches R² {fmt(mag.deep)} (CI {mag.deep_ci?.map((v) => v.toFixed(3)).join('–')}) against {fmt(mag.baseline)} for an
          amplitude-distance regression. Replaying {FACTS.replayDays} held-out days through the live code, {FACTS.confirmedReal}% of confirmed events match
          catalogued quakes (time-shifted chance {FACTS.chance}%), with {FACTS.falsePushes} false alerts among {FACTS.pushes} and a median location error of {FACTS.locErrKm} km.</p>
      </section>
      <div className="d6-figs">
        <Fig k="detection" cap={<><b>Figure 1.</b> Detection on held-out data and replayed live days, against STA/LTA.</>} />
        <Fig k="magnitude" cap={<><b>Figure 2.</b> Magnitude against catalogue (n = 937); quick check vs full estimate.</>} />
      </div>
      <section className="d6-sec">
        <h2>1 · Data</h2>
        <p>{data.dataset.events.toLocaleString()} catalogued events (M{data.dataset.mag_min}–{data.dataset.mag_max}) and {FACTS.detWindows.toLocaleString()} detection windows
          from {FACTS.years}, response-removed and low-passed identically in training and live. Splits are chronological 70/15/15.</p>
        <h2>2 · Limits</h2>
        <p>Locations outside the network are one-sided and can be tens of kilometres off; alerts arrive 30–60 s after origin,
          so this is rapid detection, not early warning. {FACTS.fprLive}% of quiet 30-second windows cross the live trigger;
          the three-station location rule is what keeps them from becoming alerts.</p>
        <h2>3 · Try it</h2>
      </section>
      <div className="d6-wrap"><Coverage /><NearMe /><CaLargest /></div>
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 7. Instrument (light console, live station grid)
function D7({ data, live }: { data: Seismic; live: boolean }) {
  useFont(gf('family=JetBrains+Mono:wght@400;600'))
  const [st, setSt] = useState<Record<string, { up: boolean; latency_s: number | null }> | null>(null)
  useEffect(() => {
    const load = () => liveDetail().then((s) => setSt(s.stations ?? null)).catch(() => setSt(null))
    load()
    const id = setInterval(load, 15000)
    return () => clearInterval(id)
  }, [])
  const det = task(data, 'detection'), mag = task(data, 'magnitude')
  return (
    <div className="d7">
      <header className="d7-top">
        <span className="d7-id">SEISMICSOCAL / LIVE</span>
        <span className={`d7-led ${live ? 'on' : ''}`}>{live ? 'STREAM OK' : 'STREAM OFFLINE'}</span>
      </header>
      <section className="d7-panel d7-trace"><p className="d7-l">ground motion · synthetic preview</p><Trace /></section>
      <section className="d7-row">
        <div className="d7-panel"><p className="d7-l">detect · auc</p><p className="d7-n">{fmt(det.deep)}</p><p className="d7-s">sta/lta {fmt(det.baseline)} · ci {det.deep_ci?.map(fmt).join('–')}</p></div>
        <div className="d7-panel"><p className="d7-l">size · r²</p><p className="d7-n">{fmt(mag.deep)}</p><p className="d7-s">amp+dist {fmt(mag.baseline)} · ci {mag.deep_ci?.map((v) => v.toFixed(3)).join('–')}</p></div>
        <div className="d7-panel"><p className="d7-l">replay · false alerts</p><p className="d7-n">{FACTS.falsePushes}/{FACTS.pushes}</p><p className="d7-s">{FACTS.confirmedReal}% confirmed real · chance {FACTS.chance}%</p></div>
      </section>
      <section className="d7-panel">
        <p className="d7-l">stations · {st ? `${Object.values(st).filter((s) => s.up).length}/${Object.keys(st).length} up` : 'status unavailable'}</p>
        <ul className="d7-grid">
          {data.stations.map((c) => {
            const s = st?.[c]
            return (
              <li key={c} className={s ? (s.up ? 'up' : 'down') : 'na'}>
                <b>{c}</b><span>{s ? (s.up ? `${s.latency_s ?? '–'} s` : 'down') : '—'}</span>
              </li>
            )
          })}
        </ul>
      </section>
      <div className="d7-wrap"><Coverage /><CaLargest /><NearMe /></div>
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 8. Questions (FAQ-led explainer)
function D8({ data }: { data: Seismic }) {
  useFont(gf('family=Young+Serif'))
  const det = task(data, 'detection'), mag = task(data, 'magnitude')
  const QA: { q: string; a: ReactNode; x?: ReactNode }[] = [
    { q: 'Can it predict earthquakes?', a: <>No. Nobody can, reliably. It detects a quake after it begins and tells you where and how big, usually within a minute.</> },
    { q: 'How does it know it’s an earthquake?', a: <>A neural network reads 30 seconds from each of 19 live sensors every 2 seconds. On data it never saw it scores AUC {fmt(det.deep)}; the classic STA/LTA trigger scores {fmt(det.baseline)}. Three sensors must agree on one location before anything counts.</>, x: <Fig k="detection" cap="The detector against STA/LTA, and catch rate by magnitude on replayed days." /> },
    { q: 'How close is its magnitude?', a: <>Typically within 0.1 of the official value (R² {fmt(mag.deep)}, against {fmt(mag.baseline)} for the textbook formula). A 4-second quick check comes first; the full estimate confirms or retracts it.</>, x: <Fig k="magnitude" cap="Estimated vs official magnitude, 937 held-out quakes." /> },
    { q: 'Does it ever cry wolf?', a: <>Replayed on {FACTS.replayDays} held-out days it sent {FACTS.pushes} alerts and none were false; {FACTS.confirmedReal}% of the quakes it confirmed were real. Small false detections do happen (about {FACTS.falseConfirmedPerWeek} a week) but they size below the alert floor and are only logged.</> },
    { q: 'Does it cover where I live?', a: <>Where three sensors sit within about 100 km: the LA basin, Inland Empire, Mojave, Ridgecrest and Kern are strongest.</>, x: <Coverage /> },
    { q: 'How do I get alerts?', a: <>Follow the sensors near you in the Android app.</>, x: <NearMe /> },
  ]
  return (
    <div className="d8">
      <header className="d8-hero">
        <h1>Six questions about the earthquake detector that listens to Southern California</h1>
        <Trace />
      </header>
      {QA.map((x) => (
        <section className="d8-qa" key={x.q}>
          <h2>{x.q}</h2>
          <p>{x.a}</p>
          {x.x}
        </section>
      ))}
      <CaLargest />
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 9. Drench (committed color, poster type)
function D9({ data }: { data: Seismic }) {
  useFont(gf('family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,700;12..96,800'))
  const det = task(data, 'detection'), mag = task(data, 'magnitude')
  return (
    <div className="d9">
      <header className="d9-hero">
        <p className="d9-brand">SeismicSoCal</p>
        <h1>Southern California, listened to every two seconds.</h1>
        <Trace />
        <p className="d9-lede">Nineteen live seismometers. Two neural networks. When the ground moves, they find it, place it
          and size it, then tell the phones that follow nearby sensors.</p>
      </header>
      <section className="d9-facts">
        <p>The detector beats the classic trigger <b>{fmt(det.deep)}</b> to <b>{fmt(det.baseline)}</b> on data it never saw.</p>
        <p>Its magnitude lands within about <b>0.1</b> of the official one (R² {fmt(mag.deep)}).</p>
        <p>On {FACTS.replayDays} replayed real days: <b>{FACTS.pushes}</b> alerts, <b>none</b> false.</p>
      </section>
      <div className="d9-wrap">
        <Carousel data={data} />
        <Coverage />
        <NearMe />
        <CaLargest />
      </div>
      <Foot />
    </div>
  )
}

// ---------------------------------------------------------------- 10. 55 seconds (timeline of one quake)
function D10({ data }: { data: Seismic }) {
  useFont(gf('family=Schibsted+Grotesk:wght@400;600;800'))
  const T = [
    { t: '0 s', h: 'Origin', b: 'Rock slips. P-waves leave at ~6 km/s, S-waves behind them.' },
    { t: '+ seconds', h: 'First sensor', b: 'The nearest station feels the P-wave. Its next 2-second scan scores the window: quake.' },
    { t: '3 picks', h: 'Located', b: 'A third station picks the P-wave; one source fits all three, and no closer sensor stayed silent.' },
    { t: `~${FACTS.provisionalS} s`, h: 'First notice', b: '4 seconds of P-wave give a quick size. If it clears the bar, followers get a provisional alert.' },
    { t: `~${FACTS.confirmedS} s`, h: 'Confirmed', b: '30 seconds from every nearby sensor give the full magnitude. The alert updates, or is retracted.' },
  ]
  return (
    <div className="d10">
      <header className="d10-hero">
        <h1>55 seconds</h1>
        <p>From the moment a fault slips to a confirmed magnitude on your phone: the median, measured on {FACTS.replayDays} replayed real days.</p>
      </header>
      <ol className="d10-line">
        {T.map((x) => (
          <li key={x.h}><span className="d10-t">{x.t}</span><h2>{x.h}</h2><p>{x.b}</p></li>
        ))}
      </ol>
      <section className="d10-why">
        <h2>Why trust the timeline</h2>
        <p>Every step is the same code that runs live, replayed over archived data. On held-out tests the detector scores
          AUC {fmt(task(data, 'detection').deep)} (STA/LTA {fmt(task(data, 'detection').baseline)}) and the magnitude R² {fmt(task(data, 'magnitude').deep)}.</p>
        <CatchBars />
      </section>
      <div className="d10-wrap"><Carousel data={data} /><Coverage /><NearMe /><CaLargest /></div>
      <Foot />
    </div>
  )
}

const DRAFTS = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10]

export default function Drafts({ n, data, live }: { n: number; data: Seismic; live: boolean }) {
  const D = DRAFTS[n - 1]
  return (
    <div className={`draft draft-${n}`}>
      <D data={data} live={live} />
      <Switcher n={n} />
    </div>
  )
}
