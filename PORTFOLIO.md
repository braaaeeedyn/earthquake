# SeismicSoCal: live deep-learning earthquake detection for Southern California

**Live:** https://seismicsocal.duckdns.org · **Code:** github.com/braaaeeedyn/earthquake · **Role:** sole developer
(data, models, live system, MLOps, web + Android app, deployment)

> A live system that listens to 19 seismometers across Southern California. Two neural networks turn the
> first seconds of ground motion into a detection, a location and a magnitude, and phones that follow nearby
> sensors get a two-stage alert about 25–55 seconds after a quake begins. Every claim on the site sits next to
> its classic-seismology baseline, a 95% confidence interval and a replay of real days.

Research demonstration, not an official warning system. It detects earthquakes after they begin; it does not predict them.

---

## 1. The story in one paragraph

The project started as an attempt to **forecast** earthquakes a week ahead from geomagnetic data. On real
data that idea came back null: a superposed-epoch test gave p = 0.83 and the ROC was at chance, which is what
the literature predicts. I kept the null result on record, removed the forecasting pipeline, and pivoted
the same CNN / GNN / Transformer toolkit to a problem where deep learning genuinely works: reading seismic
waveforms. The result is a production system that runs 24/7 on a free cloud VM.

## 2. What a visitor sees (the site, top to bottom)

1. **Hero and live status.** The project name, a one-line summary, an animated seismograph trace (it reacts to
   the cursor on desktop and "breathes" on phones), the dataset size, and a green dot when the live SeedLink
   stream is connected.
2. **Detect → Size cards.** An auto-cycling, two-step carousel that follows the pipeline's real order. Each
   card answers one question ("Is it an earthquake?", "How big is it?"), shows the deep model's held-out score
   against the classic baseline with a **95% confidence interval**, and opens an *Evidence* panel: a figure built
   from the current models plus a step-by-step technical rundown.
3. **Where it can see.** An interactive SVG map of Southern California, drawn from Census outlines, showing
   the 19 stations and their coverage. Dark areas have 3+ stations within 100 km, so a quake there is located,
   sized and can alert. Light areas reach 2 stations and are logged only. Wheel or pinch to zoom; more cities
   and dotted city boundaries appear as you zoom in.
4. **Alert me near me.** Follow a region (all its sensors), then switch single sensors off. "Use my location"
   or a city search picks the nearest region. Coordinates never leave the device; only station codes are sent.
   Subscribing happens in the Android app, which receives push notifications.
5. **Biggest Southern California quakes.** Day / week / month / year / all-time windows from the USGS
   catalogue, filtered to quakes the pipeline *could* catch. Each one is marked **caught** (with our magnitude),
   **seen** (logged, not confirmed) or **not caught**, by matching against the live event log.
6. **Model health** (`/health`). Which model version is live, its metrics with CIs, per-station drift
   status, and the promotion history (QuakeOps, section 7).

## 3. How it works

```
SeedLink (19 stations, real time) ─► per-station buffers ─► data-quality gate
   ─► DETECT  30 s window every 2 s per station (CNN → Transformer)  ─► P-wave pick (STA/LTA + AIC)
   ─► LOCATE  ≥ 3 picks that fit one source (grid search), no silent nearer station   = CONFIRMED
   ─► QUICK CHECK  per subscriber: Standard 4 s of P / Fast 2 s  ─► provisional push   (~33 s / ~26 s)
   ─► SIZE    30 s from every station ≤ 200 km (CNN → GNN → Transformer, 5-seed ensemble)
   ─► DECIDE  M ≥ 3.0 → confirmed push, else retraction (replaces the first notification)   (~55 s)
```

- **One engine, two uses.** The same Python module (`pipeline.py`) runs live and in an offline replay harness
  over archived data, so what is measured offline is exactly what runs in production. Everything runs on
  waveform (data) time, never wall-clock time.
- **Training input == live input.** The training windows and the live windows are cut by the same picker,
  filtered by the same causal filters and normalized by the same function. One subtle bug this caught: the
  detector was partly learning the difference between response-corrected training positives and raw-count
  negatives. A shared 1 Hz high-pass removed that shortcut.
- **Negative evidence.** With exactly 3 picks a location fit is mathematically exact, so a low misfit proves
  nothing. A detection is confirmed only if no healthy station *closer* to the epicentre stayed silent; a real
  quake reaches nearer stations first.
- **Two-stage alerts with a user-chosen speed.** Five variants of the first message were replayed over 20 days,
  and the trade-off is exposed as a setting:
  - **Standard** (4 s of P, full instrument correction) keeps the most safeguards;
  - **Fast** (2 s, sensitivity-scaled) arrives about 7 s sooner, with a rougher size and more retractions;
  - pushing on location alone was rejected: about 7 alerts a day, nearly all retracted. The full model then
  confirms or retracts it, and both carry the same notification tag, so the second message replaces the first.

## 4. Data and models

| | Detect | Size |
|---|---|---|
| Question | Is there a quake in this 30 s window? | Given a located quake, what is its magnitude? |
| Architecture | 4 strided 1-D conv layers → 2-layer Transformer | per-station CNN → 2 graph-conv layers over the station network → Transformer |
| Training data | 34,377 event windows, 14,304 noise windows, 2,062 *hard negatives* (the previous live system's own false alarms) | 6,243 catalogued quakes (M2.0–7.1), 37,474 station records |
| Held-out test | **ROC-AUC 0.9998** (95% CI 0.9997–0.9999), MCC 0.886 | **R² 0.951** (0.943–0.959), MAE 0.10 magnitude units |
| Classic baseline | STA/LTA trigger: AUC 0.816 (0.805–0.828) | amplitude + distance: R² 0.886 (0.871–0.898) |

- **Data:** the USGS catalogue (2000 → 2026), plus one SCEDC waveform request per event with instrument
  response removed. Pre-2010 data is 40 Hz, so a common 18 Hz low-pass makes 40 Hz and 100 Hz data look alike.
- **Splits:** strictly chronological 70/15/15, never random. Hard negatives are split by date.
- **Statistics:** bootstrap CIs, clustered by event so that correlated station windows aren't treated as
  independent. Paired comparisons for deep vs baseline. A 10-seed study separates seed variance (single model
  R² 0.949, t-CI 0.948–0.951) from sampling variance (ensemble 0.952, bootstrap 0.944–0.959).
- **Augmentation that mimics live conditions:** the P-wave can fall anywhere in the window; ±8 km epicentre
  jitter; ±0.5 s pick jitter; random station drop-out.

## 5. Proving it works on real days (the acceptance test)

Test AUC alone doesn't say whether a live system can be trusted, so the acceptance test is a **replay**:
the exact live engine runs over **80 archived days** (2022–2026) it never trained or calibrated on: 60 days with an
M3+ quake in coverage and 20 random days.

| Replay result (80 held-out days) | |
|---|---|
| Push alerts | **145**, every one a real quake; 92% placed within 60 km (the other 12 were out-of-network quakes located 63–145 km off). Time-shifted chance baseline: 0% |
| Pushed magnitudes vs catalogue | bias +0.05, MAE 0.12 |
| Confirmed events that were real quakes | 86% busy days / 79% random days (chance 4% / 0%) |
| Median location error | **3–4 km** |
| First message (Standard / Fast) / confirmation | ~31 s / ~25 s / ~50 s after origin |
| Catch rate | **68%** of M3+ (95% CI 60–74%); 78% outside aftershock swarms with ≥3 stations online; 52% of M2+ |

The misses are concentrated in aftershock swarms (30% caught vs 73% for isolated M2+ quakes), where a
second quake within two minutes was absorbed into the first. Tuning that window and the station rest
period on validation swarms, then scoring once on the 80 test days, raised M3+ catch from 71% to 78%
(detection-only scoring) with no duplicates and precision 85% → 84%. A smaller 10-day check had suggested 86% at M3+
with zero false pushes; the larger test is the honest number. The system it replaced sent 6 pushes on 4 of those
days, all false.

The thresholds (trigger level, pick SNR, misfit, station count) were calibrated on **separate validation days**.
Every live and replay precision is reported next to a chance baseline. After deployment the system ran in
**shadow mode** (detecting and logging, sending no pushes), with a nightly job that scores the live log
against USGS before alerts are switched on.

## 6. Engineering highlights

- **Shaking at your home, without sending your home anywhere.** For every quake the app estimates the Modified
  Mercalli intensity at the user's saved home: a ground-motion equation fitted on our own 25k station records,
  a USGS Vs30 ground-type term, a per-quake correction from how hard it actually shook our sensors, and an
  offset calibrated on USGS "Did You Feel It?" reports. Held out on 28 later quakes (1,892 report cells): MAE
  0.42 levels, 94% within one level. The same equations run on the server, in the web app and in native
  Android code that writes the notification, and the location never leaves the phone.

- **One station list** (`network.py`) is imported by the dataset builder, the daemon, the API, the scorer
  and the replay harness. An earlier version had drifted apart and silently ran on 5 stations instead of 10.
  Every station must stream on the public SeedLink relay, and the builder checks this.
- **Checkpoints carry their own normalizers and station list.** The daemon refuses a model trained on a
  different network.
- **Self-healing daemon.** The API server supervises the SeedLink daemon and respawns it if the stream drops.
- **Privacy by design.** Subscriptions store station codes and a push token, never a location.
- **$0 infrastructure.** Oracle Always-Free ARM VM, Caddy (HTTPS), systemd, DuckDNS, Firebase Cloud Messaging.

## 7. QuakeOps: making it maintain itself (MLOps)

| Capability | What it does here |
|---|---|
| Experiment tracking + registry (**MLflow**) | Every training run logs params, lineage (commit, dataset version, seeds), metrics with CIs and the checkpoint. Models are versioned with `champion` / `challenger` aliases. |
| Statistically gated promotion | A challenger is promoted only if it beats the classic baseline (paired CI > 0), is non-inferior to the champion on a test split neither has seen, and **passes the 10-day replay**. Six rules (G1–G6), every one logged. |
| Orchestration (**Dagster**) | A monthly job grows the dataset, retrains, evaluates, replays, gates and promotes. Every stage can be resumed. |
| Growing data | Dataset builds are append-stable: earlier selections are kept and only the new month downloads. |
| Drift monitoring (**Evidently**) | The daemon logs scale-free features of the live stream. A daily job compares each station with its training noise (ok / watch / drifting) and emails on a 2-day streak. |
| CI/CD (**GitHub Actions**) | Lint, tests, daemon selftest and site build on every change; tar-stream deploy to the VM on `main`. |
| Reproducibility | Every model traces back to its code commit, dataset hash and seeds. |

## 8. Design

- **The live site:** a deliberately minimal, monochrome system (black ink, gray body text, filled-black pills,
  12 px cards, light only) with serif (Literata) headings and a layout and type scale that grow with the
  screen. The chrome stays quiet so the evidence stands out.
- **Design process:** ten alternative homepage directions were prototyped on the real components and data,
  from a story scroll to a research-paper layout. The original layout won; it adopted the explorations'
  serif headings, fluid type and higher-contrast text.
- **Accessibility:** WCAG AA contrast, status never shown by color alone, reduced-motion support for every
  animation, and no sideways scroll at 320 px.

## 9. Honest limits

- **Coverage:** strongest where 3+ stations sit within ~100 km (LA basin, Inland Empire, Mojave, Ridgecrest,
  Kern). Quakes outside the network are located from a one-sided set of stations and can be tens of km off.
- **Latency:** alerts arrive ~30–60 s after origin. This is rapid detection, not pre-arrival warning.
- **Small quakes:** quakes below M2 are outside the magnitude model's training range and read slightly high.
- **Station availability:** stations can drop off the public relay; the system then uses the ones that remain.

## 10. Stack

Python · PyTorch (CUDA) · ObsPy · NumPy / SciPy / pandas / scikit-learn · MLflow · Evidently · Dagster ·
React + TypeScript + Vite · Capacitor (Android) · Firebase Cloud Messaging · Caddy · systemd · Oracle Cloud ·
GitHub Actions · pytest · ruff

## 11. What I'd point a reviewer to

- `src/eq/pipeline.py`: the detect → locate → size → decide engine shared by live and replay.
- `scripts/replay_archive.py`: the acceptance test, with chance baselines.
- `scripts/retrain.py`: the statistical + replay promotion gate.
- `HOW_IT_WORKS.md`: the full method, with every number.
