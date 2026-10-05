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
pipeline was later removed — `src/eq/` now holds only the seismic catalog/waveform helpers. Work
pivoted to **seismic-waveform deep learning**, where the same CNN/GNN/Transformer architecture
genuinely works.

## The three models (Detect → Size → Warn)

Real numbers, out-of-sample on a **chronological** held-out split. Detect and Size are trained on the
**19 stations that actually stream live** (`src/eq/network.py`), **2000 → Aug 2026**: 6,243 magnitude
events (M2.0–7.1) and 50,743 detection windows (34,377 event / 14,304 noise / 2,062 hard negatives).

| Task | Model | Baseline | Verdict |
|------|-------|----------|---------|
| **Detect** — is it a quake? | CNN+Transformer, AUC **0.9998** (MCC 0.886) | STA/LTA 0.816 | deep wins decisively |
| **Size** — how big? | multi-station GNN, R² **0.951** (MAE 0.10) | amp+dist 0.886 (MAE 0.16) | deep wins (nearest-1-station ablation → R² 0.808) |
| **Warn** — how hard will it shake? | EEW ensemble, alert **MCC 0.760** | GMPE-style 0.655 | deep wins (recall 0.76 @ precision 0.82) |

_EEW numbers are from the earlier 10-station run and EEW is not in the live loop (offline evidence)._

```bash
.venv/Scripts/python scripts/build_dataset.py                       # check -> select -> fetch -> assemble
.venv/Scripts/python scripts/demo_detect.py --retrain --seeds 5     # detection vs STA/LTA
.venv/Scripts/python scripts/demo_magnitude.py --retrain --seeds 5  # 5-seed magnitude ensemble
.venv/Scripts/python scripts/demo_eew.py                            # early-warning ensemble (legacy data)
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
  ->  SIZE once P+25 s has arrived: windows [P-5 s, P+25 s], distances from the LOCATED epicentre
  ->  PUSH iff confirmed AND M >= 3.0 AND PUSH_ENABLED=1: devices subscribed to a station within 150 km
```

**Why 3 stations:** with 3 picks the location (lat, lon, origin time) is exactly determined, which is the
minimum that tests whether the picks come from one place; the "no silent nearer station" rule then rejects
coincident noise (a real quake reaches nearer stations first). Two stations always "fit" and prove nothing.
On the full validation set 4 stations confirmed 31 events vs 86 and missed a real out-of-network M3.6 that
3 stations caught; unmatched 3-station confirmations are logged only and sit at the network edges. The **M3.0 floor** keeps pushes to quakes people can feel.

### Replay harness — the acceptance test (`scripts/replay_archive.py`)

Runs the exact live engine over **archived continuous data** (SCEDC), so the live behaviour is measured
offline. Thresholds were chosen on 10 validation days (Sep 29 – Oct 2 2026 + Sep 7–14 2020) and scored
once on held-out days:

| 10 held-out days (Oct 2–5 + Aug 18–25 2026) | new pipeline |
|---|---|
| confirmed events that are real catalogued quakes | **93 %** (chance baseline 0 %) |
| false confirmations (logged, never pushed) | 3.5 / week |
| pushes / false pushes | 5 / **0** — sized 3.63, 3.10, 3.39, 4.09, 3.49 vs catalog 3.6, 3.0, 3.3, 4.0, 3.5 |
| location error (median) | **2.5 km** |
| in-coverage M3+ caught | 4 / 5 (the miss came 80 s after an M4.0 at the same spot — coda suppression) |

On the same Oct 2–5 days the **old** daemon pushed 6 alerts, **all false**, with a 37 km station-proxy
location error. Event-centric test on 616 held-out events (live geometry): magnitude bias **+0.06**,
MAE 0.12 (the old live path read **≈2.2 units low**), median location error 4.3 km.

```bash
.venv/Scripts/python scripts/replay_archive.py scan --start 2026-10-02 --end 2026-10-05   # score windows (cached)
.venv/Scripts/python scripts/replay_archive.py run  --start 2026-10-02 --end 2026-10-05   # engine + scorecard
.venv/Scripts/python scripts/live_watch.py --selftest                                    # logic checks
```

## Web console — `app/`

React + Vite. `npm run dev`, opens on `localhost:5173`.

- **Hero** with a mouse-reactive synthetic seismograph.
- **Detect / Size / Warn** as an auto-cycling card carousel (7 s, pause, prev/next, shared "Evidence"
  disclosures with the demo figures + a technical paragraph for Size and Warn).
- **Biggest Southern California quakes** — a second carousel cycling Day / Week / Month / Year /
  All time, live from the USGS FDSN catalog, each row linking to its `sms-tsunami-warning.com` page.
- **Alert me near me** — subscribe to the **sensor stations** nearest you (city/state or "use my
  location" ranks the 19 stations by distance and auto-picks the nearest 3; toggle any). No coordinates
  are stored — only the chosen station codes + your device's push token. Push alerts are **mobile-app
  only**; the daemon does the alerting when a station you follow triggers.

Run the full stack:

```bash
.venv/Scripts/python scripts/server.py    # backend on :8000; auto-spawns the live daemon
cd app && npm run dev                      # frontend on :5173, proxies /api -> :8000
```

Push alerts need `fcm-service-account.json` (Firebase) on the host; the contact form needs
`SMTP_USER` / `SMTP_PASS` in a gitignored `.env`. Without them those paths print instead of send.
See `DEPLOY.md` for the Oracle A1 deployment (Caddy + systemd) and update procedure.

---

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
- The "Biggest SoCal quakes" browser uses the **USGS catalog** (metadata, not model output).

## Repository layout (current)

```
src/eq/
  network.py       # THE live station list (19 stations) — every consumer imports it
  pipeline.py      # live engine: DQ gate, detect, pick, locate/associate, size, decide (live + replay)
  locate.py        # P picker, travel times (+ fitted correction), grid-search locator
  seismic.py       # SCEDC waveform access (response removal, 18 Hz common low-pass, compact cache)
  quakecast.py     # USGS catalog (monthly chunks, auto-split, date-ranged cache)
scripts/
  build_dataset.py      # v2 datasets on the live network (check / select / fetch / assemble)
  demo_detect.py / demo_magnitude.py / demo_eew.py   # train + evaluate (EEW = legacy data)
  replay_archive.py     # replay harness: scan / calibrate / run / events / compare-live
  live_watch.py         # LIVE SeedLink daemon (pushes only with PUSH_ENABLED=1)
  select_network.py     # reproduce the station selection (streamable, quiet, spaced, coverage)
  migrate_subscriptions.py  # map retired stations in push_tokens.json to the nearest new one
  crosscheck_events.py  # score the live log vs USGS, with a time-shifted chance baseline
  server.py / push_fcm.py / shaking_model.py / quake_archive.py / nearme_watch.py
  seismic_train*.py / seismic_eew*.py   # model definitions (+ legacy research mains)
app/                    # React + Vite + Capacitor console (web + Android)
tests/                  # pytest: network, picker/locator, pipeline rules, overfit-one-batch models
```

Datasets (`data/processed/v2/*.npz`), checkpoints (`*.pt`) and the waveform cache (`data/raw/`) are
gitignored. `data/processed/v2/pipeline_config.json` + `tt_correction.json` ship with the models.
