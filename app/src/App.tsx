import { useEffect, useState } from 'react'
import { loadSeismic, type Seismic, type Task } from './seismic'

type State =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: Seismic }

export default function App() {
  const [state, setState] = useState<State>({ status: 'loading' })

  useEffect(() => {
    let active = true
    loadSeismic()
      .then((data) => active && setState({ status: 'ready', data }))
      .catch((e) => active && setState({ status: 'error', message: String(e?.message ?? e) }))
    return () => {
      active = false
    }
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <span className="logo">
          <span className="pulse" aria-hidden /> SEISMIC&nbsp;ML
        </span>
        <span className="topbar-sub">CNN · GNN · Transformer</span>
      </header>

      <main className="content">
        {state.status === 'loading' && <p className="muted center">Loading results…</p>}
        {state.status === 'error' && (
          <div className="card error">
            <h2>Couldn’t load results</h2>
            <p className="muted">{state.message}</p>
            <p className="muted">
              Ensure <code>seismic.json</code> is present in <code>public/</code>.
            </p>
          </div>
        )}
        {state.status === 'ready' && <Results data={state.data} />}
      </main>

      <footer className="footer muted">
        Research &amp; education only · real California network data · not an official warning system
      </footer>
    </div>
  )
}

function Results({ data }: { data: Seismic }) {
  const d = data.dataset
  return (
    <>
      <section className="hero">
        <p className="eyebrow">Earthquake ML · {d.region}</p>
        <h1>
          Seismic deep learning that <em>earns its keep</em>
        </h1>
        <p className="lede">
          One shared <strong>{data.model}</strong> backbone for earthquake detection, magnitude, and
          early warning — evaluated honestly against the classic seismology baselines, on real
          waveform data.
        </p>
        <div className="chips">
          <span className="chip">{d.events.toLocaleString()} events</span>
          <span className="chip">{d.stations} stations</span>
          <span className="chip">M {d.mag_min}–{d.mag_max}</span>
          <span className="chip">out-of-sample · chronological split</span>
        </div>
      </section>

      <section>
        <h2 className="section-title">Results vs. classic baselines</h2>
        <div className="grid">
          {data.tasks.map((t) => (
            <TaskCard key={t.key} t={t} />
          ))}
        </div>
      </section>

      <section>
        <h2 className="section-title">At a glance</h2>
        <img className="figure" src="seismic_results.png" alt="Deep model vs classic baselines across four tasks" />
      </section>

      <section className="note">
        <h2 className="section-title">Honest framing</h2>
        <p>
          The deep model <strong>doesn’t</strong> beat the physics where the task <em>is</em> physics
          (raw shaking level ≈ amplitude + distance) — there it ties. It <strong>does</strong> win where
          the task needs learned pattern recognition (detection) or a better operating-point trade-off
          (alerting), and the hybrid magnitude head beats the baseline by adding waveform information on
          top of the amplitude–distance physics. Every number is out-of-sample with the baseline shown
          alongside; the early-warning figures are 5-seed ensembles.
        </p>
        <p className="disclaimer">{data.disclaimer}</p>
        <div className="stations">
          {data.stations.map((s) => (
            <span key={s} className="sta">{s}</span>
          ))}
        </div>
      </section>
    </>
  )
}

function TaskCard({ t }: { t: Task }) {
  const max = Math.max(t.deep, t.baseline, 1)
  const fmt = (v: number) => (t.metric === 'recall' ? v.toFixed(2) : v.toFixed(3))
  const label =
    t.winner === 'deep' ? 'Deep wins' : t.winner === 'tie' ? 'Tie' : 'Baseline wins'
  return (
    <article className={`card task ${t.winner}`}>
      <div className="task-head">
        <h3>{t.name}</h3>
        <span className={`pill pill-${t.winner}`}>{label}</span>
      </div>
      <p className="metric-label">{t.metric}</p>

      <Bar name="Deep model" value={t.deep} max={max} kind="deep" text={fmt(t.deep)} />
      <Bar name={t.baseline_name} value={t.baseline} max={max} kind="base" text={fmt(t.baseline)} />

      <p className="task-desc">{t.desc}</p>
    </article>
  )
}

function Bar({
  name,
  value,
  max,
  kind,
  text,
}: {
  name: string
  value: number
  max: number
  kind: 'deep' | 'base'
  text: string
}) {
  return (
    <div className="bar-row">
      <span className="bar-name">{name}</span>
      <div className="bar-track">
        <div className={`bar-fill ${kind}`} style={{ width: `${(value / max) * 100}%` }} />
      </div>
      <span className="bar-val">{text}</span>
    </div>
  )
}
