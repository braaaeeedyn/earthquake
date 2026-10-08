# SeismicSoCal ML

Deep-learning seismology for **Southern California**: detect an earthquake, size it, and warn —
each shown working on real held-out SCEDC waveforms and against the classic seismology baseline —
plus a **live detection daemon** that runs the models on a real-time station stream and **push-alerts**
nearby subscribers. **Deployed live at https://seismicsocal.duckdns.org.**

> **Honest disclaimer.** This is a research prototype, not an official warning system. Short-term
> earthquake *prediction* (whether one will occur) remains unsolved; this does detection,
> characterization, and rapid shaking estimation. See "Honesty notes" below.

---

## Background: the pivot

The project began as near-term earthquake *forecasting* from geomagnetic (INTERMAGNET) data. On
real data that thesis came up **null** (superposed-epoch p=0.83, ROC ≈ chance) — exactly as the
literature predicts. That track is retired as a rigorous replication/null result, and the geomagnetic
pipeline was later removed — `src/eq/` is the live library: station network, catalogue, waveform access, picker/locator, models, pipeline and stats. Work
pivoted to **seismic-waveform deep learning**, where the same CNN/GNN/Transformer architecture
genuinely works.

## The two models (Detect → Size)

Real numbers, out-of-sample on a **chronological** held-out split. Detect and Size are trained on the
**19 stations that actually stream live** (`src/eq/network.py`), **2000 → Aug 2026**: 6,243 magnitude
events (M2.0–7.1) and 50,743 detection windows (34,377 event / 14,304 noise / 2,062 hard negatives).

| Task | Model | Baseline | Verdict |
|------|-------|----------|---------|
| **Detect** — is it a quake? | CNN+Transformer, AUC **0.9998** (MCC 0.886) | STA/LTA 0.816 | deep wins decisively |
| **Size** — how big? | multi-station GNN, R² **0.951** (MAE 0.10) | amp+dist 0.886 (MAE 0.16) | deep wins (nearest-1-station ablation → R² 0.808) |

A third, much simpler estimator — the **quick check** — sizes a located quake from the first 4 s of its
P-wave (classic amplitude scaling, test MAE 0.24) so a provisional alert can go out ~20 s before the full
magnitude. Full method: `HOW_IT_WORKS.md`.

```bash
.venv/Scripts/python scripts/build_dataset.py                       # check -> select -> fetch -> assemble
.venv/Scripts/python scripts/demo_detect.py --retrain --seeds 5     # detection vs STA/LTA
.venv/Scripts/python scripts/demo_magnitude.py --retrain --seeds 5  # 5-seed magnitude ensemble
.venv/Scripts/python scripts/fit_early_magnitude.py                 # quick-check fit (first 4 s of P)
.venv/Scripts/python scripts/make_figures.py                        # site evidence figures
```

## Live alert daemon — `scripts/live_watch.py` (engine: `src/eq/pipeline.py`)

The models run **continuously on a live SeedLink stream** (USGS bypassed), on **data time**, and every
model input is cut exactly as in training:

```
SeedLink (19 CI stations, pinned location codes)  ->  per-station buffers
  ->  data-quality gate  ->  DETECT: 30 s windows (1 Hz high-pass, unit std) every 2 s
  ->  PICK the P onset (STA/LTA + AIC, the same picker that aligned the training data)
  ->  LOCATE: >= 3 picks must fit ONE source (grid search) and no healthy nearer station may be silent
        -> CONFIRMED;  1-2 stations -> TENTATIVE (logged, never pushed)
  ->  QUICK CHECK ~10 s after P (first 4 s of P amplitude + distance): provisional push if >= 3.04
  ->  SIZE once P+25 s has arrived: windows [P-5 s, P+25 s], distances from the LOCATED epicentre
  ->  CONFIRM (M >= 3.0) or RETRACT the provisional push, replacing it on the device (same notification
      tag). All pushes need PUSH_ENABLED=1 and go to devices following a station within 150 km.
```

**Why 3 stations:** with 3 picks the location (lat, lon, origin time) is exactly determined, which is the
minimum that tests whether the picks come from one place; the "no silent nearer station" rule then rejects
coincident noise (a real quake reaches nearer stations first). Two stations always "fit" and prove nothing.
On the full validation set 4 stations confirmed 31 events vs 86 and missed a real out-of-network M3.6 that
3 stations caught; unmatched 3-station confirmations are logged only and sit at the network edges. The **M3.0 floor** keeps pushes to quakes people can feel.

### Replay harness — the acceptance test (`scripts/replay_archive.py`)

Runs the exact live engine over **archived continuous data** (SCEDC), so the live behaviour is measured
offline. Thresholds were chosen on 10 validation days (Sep 29 – Oct 2 2026 + Sep 7–14 2020) and scored
once on held-out days.

**80 held-out days** (Apr 2022 – Aug 2026: 60 with an M3+ quake in coverage + 20 random; `replay/bigtest/`):

| | result |
|---|---|
| pushes | **145**, all real quakes; 133 (92 %) matched a catalogued M2.5+ within 60 km, 12 were out-of-network quakes located 63–145 km off (chance baseline 0 %) |
| pushed size vs catalog | bias +0.05, MAE 0.12, 89 % within 0.3 |
| confirmed events that are real | 86 % busy days / 79 % random days (chance 4 % / 0 %); 7.4 false confirmations/week, logged only |
| location error (median) | 3.0–3.6 km |
| catch rate | M3+ **68 %** (CI 60–74 %); with ≥3 stations up: isolated M3+ **83 %** (86 % good coverage / 80 % thin) vs **60 %** inside sequences; M2+ 52 % |
| first message / confirmation | Standard ~31 s, Fast ~25 s / ~50 s after origin |

The first 10-day check (Oct 2–5 + Aug 18–25 2026: 93 % real, 5 pushes / 0 false, 4 of 5 M3+ caught) was too
small to quote. **Swarm tuning** (2026-10-07; echo window 120 → 30 s, station rest 60 → 45 s, chosen on
validation swarms) then raised 80-day M3+ catch 71 → 78 % (detection-only scoring), precision 85 → 84 %,
no duplicates; HOW_IT_WORKS §6.4. On Oct 2–5 the **old** daemon pushed 6 alerts, **all false**, with a 37 km station-proxy
location error. Event-centric test on 833 held-out events (live geometry): magnitude bias **+0.08**,
MAE 0.135, median location error 3.8 km.

```bash
.venv/Scripts/python scripts/replay_archive.py scan --start 2026-10-02 --end 2026-10-05   # score windows (cached)
.venv/Scripts/python scripts/replay_archive.py run  --start 2026-10-02 --end 2026-10-05   # engine + scorecard
.venv/Scripts/python scripts/live_watch.py --selftest                                    # logic checks
```

## Web console — `app/`

React + Vite. `npm run dev`, opens on `localhost:5173`.

- **Hero** with a mouse-reactive synthetic seismograph.
- **Detect / Size** as an auto-cycling card carousel (7 s, pause, prev/next, shared "Evidence"
  disclosures with the evidence figures and a step-by-step technical rundown).
- **Where it can see** — interactive coverage map (zoom/pan; more cities and dotted city boundaries as you
  zoom in) of the 19 stations and where 3+ / 2 stations cover.
- **Biggest Southern California quakes** — a second carousel cycling Day / Week / Month / Year /
  All time, live from the USGS FDSN catalog, each with its caught mark; each row opens a quake page with the
  **estimated shaking at your home** (MMI, computed on the device; see below).
- **Alert me near me** — follow a **region** (all its sensors), then turn single sensors off (city/state
  or "use my location" selects the nearest region). No coordinates
  are stored — only the chosen station codes + your device's push token. Push alerts are **mobile-app
  only**; the daemon does the alerting when a station you follow triggers.

**Shaking at your home (MMI).** Setting your location in Alert me near me also saves it, on the device only,
as your home. For every quake (past, or the one just pushed) the app estimates the Modified Mercalli intensity
there from magnitude, distance and ground type (USGS Vs30), plus — for live quakes — how hard the quake shook our
sensors. Checked on held-out USGS "Did You Feel It?" reports: MAE 0.42 levels, 94 % within one level (28 quakes,
1,892 cells). APK 2.01.00 adds native code that puts it in the notification itself. Details: HOW_IT_WORKS §13.

Run the full stack:

```bash
.venv/Scripts/python scripts/server.py    # backend on :8000; auto-spawns the live daemon
cd app && npm run dev                      # frontend on :5173, proxies /api -> :8000
```

Push alerts need `fcm-service-account.json` (Firebase) on the host; the contact form needs
`SMTP_USER` / `SMTP_PASS` in a gitignored `.env`. Without them those paths print instead of send.
See `DEPLOY.md` for the Oracle A1 deployment (Caddy + systemd) and update procedure.

---

## How a model gets to production (QuakeOps)

```
monthly (Dagster, PC)            gate (retrain.py)                          VM (daily timer)
build_dataset --append  ──►  G1 beats baseline (paired CI)      ──►  @champion  ──►  tracking.py pull: sha256-
train challenger (MLflow)    G2 non-inferior to champion             in MLflow        verified install (opt-in)
 --compare champion          G3 replay of 10 held-out days                            → daemon restarts on it
                             G4 tests  G5 lineage  G6 reason                          → drift_check.py daily
```

Every training run is tracked in MLflow: code commit, dataset version, seeds, metrics with 95% CIs, and
the checkpoint. A challenger replaces the champion only if it beats the classic baseline, is
statistically non-inferior to the champion on a test split neither model has seen, and passes the same
replay acceptance test the live system was accepted on (precision against a chance baseline, no false
pushes, magnitudes matching the catalogue). The live stream is checked every day for drift from the
training data, and the result is shown on the site's `/health` page. Details: `HOW_IT_WORKS.md` §12.

### Magnitude: seed-averaged result

The magnitude model was trained with 10 seeds on the same chronological split (937 held-out quakes,
M2–5.2). The two sources of uncertainty are reported separately:

| | R² | 95% CI |
|---|---|---|
| Single model, mean over 10 seeds (seed variance) | 0.949 | 0.948–0.951 (t-interval) |
| 10-seed ensemble (sampling variance) | 0.952 | 0.944–0.959 (event bootstrap) |
| Live 5-seed ensemble | 0.951 | 0.943–0.959 |
| Amplitude + distance baseline | 0.886 | 0.871–0.898 |
| Ensemble − baseline (paired) | +0.067 | +0.056…+0.079 |

The deep model's advantage over the baseline is about 6× the width of either uncertainty. Ensembling
adds +0.003 over a single model, well inside the sampling CI, so the live system keeps 5 seeds.

## Honesty notes

- **Coverage** is strongest where >= 3 stations sit within ~100 km (LA basin, Inland Empire, Mojave,
  Ridgecrest, Kern); San Diego / Imperial and offshore are thinner. Only stations on IRIS's public
  SeedLink relay can be used (five archived SCSN stations the old network relied on never stream there).
- **Latency:** SeedLink is seconds to tens of seconds and sizing waits for P+25 s, so pushes go out
  ~30–60 s after the origin: *rapid detection*, not pre-arrival warning (ShakeAlert's job).
- **Location** is a fixed-depth grid search with a 1-D travel-time model calibrated on 31k picks — good
  to a few km inside the network, coarser outside it.
- **Small quakes** (M1–2) are detected and logged; sizes below ~M2 read slightly high (the magnitude set
  starts at M2), which is harmless for the M3 push floor.
- The "Biggest SoCal quakes" browser lists **USGS catalog** quakes the pipeline could catch (M2+, 3+
  stations within 100 km) and marks each one caught / seen / not caught against the live event log.

## Repository layout (current)

```
src/eq/
  network.py       # THE live station list (19 stations) — every consumer imports it
  models.py        # DetectorNet (detect), MultiStationModel + station graph (size), STA/LTA baseline
  pipeline.py      # live engine: DQ gate, detect, pick, locate/associate, size, decide (live + replay)
  locate.py        # P picker, travel times (+ fitted correction), grid-search locator
  seismic.py       # SCEDC waveform access (response removal, 18 Hz common low-pass, compact cache)
  catalog.py       # USGS catalog (monthly chunks, auto-split, date-ranged cache)
  stats.py         # bootstrap / cluster / paired bootstrap CIs, seed t-CI
  shaking.py       # estimated shaking (MMI) at a place: equation, site + event terms, DYFI offset
scripts/
  build_dataset.py      # v2 datasets on the live network (check / select / fetch / assemble)
  demo_detect.py / demo_magnitude.py   # train + evaluate (95% CIs, --compare a champion)
  tracking.py / retrain.py / quakeops_dagster.py / drift_check.py   # QuakeOps: MLflow, gate, Dagster, drift
  fit_early_magnitude.py / make_figures.py   # quick-check fit / site evidence figures
  replay_archive.py     # replay harness: scan / calibrate / run / events / compare-live
  live_watch.py         # LIVE SeedLink daemon (pushes only with PUSH_ENABLED=1)
  select_network.py     # reproduce the station selection (streamable, quiet, spaced, coverage)
  crosscheck_events.py  # score the live log vs USGS, with a time-shifted chance baseline
  server.py             # API + supervisor of live_watch.py
  push_fcm.py / mailer.py   # FCM pushes (normal or data-only) / .env + operator email
  calibrate_shaking.py  # fits shaking_calibration.json (ground-motion equation + Vs30 site term)
  validate_mmi.py       # scores the MMI estimate vs USGS Did You Feel It?, fits the DYFI offset
  build_apk.sh          # Android APK from the web build
app/                    # React + Vite + Capacitor console (web + Android)
tests/                  # pytest: network, picker/locator, pipeline rules, overfit-one-batch models
```

Datasets (`data/processed/v2/*.npz`), checkpoints (`*.pt`) and the waveform cache (`data/raw/`) are
gitignored. `data/processed/v2/pipeline_config.json` + `tt_correction.json` ship with the models.
