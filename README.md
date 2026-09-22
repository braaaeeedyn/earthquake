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

Real numbers, out-of-sample on a **chronological** held-out split (10-station SoCal network,
**2000–2025**, 1,126 magnitude events / 5,863 detection windows):

| Task | Model | Baseline | Verdict |
|------|-------|----------|---------|
| **Detect** — is it a quake? | CNN+Transformer, AUC **0.992** (MCC 0.930) | STA/LTA 0.550 | deep wins decisively |
| **Size** — how big? | multi-station GNN, R² **0.840** | amp+dist 0.749 | deep wins (nearest-1-station ablation → R² +0.42) |
| **Warn** — how hard will it shake? | EEW ensemble, alert **MCC 0.760** | GMPE-style 0.655 | deep wins (recall 0.76 @ precision 0.82) |

_Detect + Size retrained on 2000–2025 data; EEW numbers are from the earlier run (not retrained)._

Reproducible one-command demos (train-once-and-cache):

```bash
.venv/Scripts/python scripts/demo_detect.py       # detection vs STA/LTA
.venv/Scripts/python scripts/demo_magnitude.py    # 5-seed magnitude ensemble
.venv/Scripts/python scripts/demo_eew.py          # early-warning ensemble (val-tuned alert)
```

## Live alert daemon — `scripts/live_watch.py`

The models run **continuously on a live waveform stream** (USGS bypassed):

```
SeedLink stream (10 CI/SCEDC SoCal stations)  ->  rolling per-station buffers
  ->  DETECTION model on sliding 30 s windows, continuously
  ->  COINCIDENCE: event declared only when >= K stations agree within ~12 s  (kills false alarms)
  ->  location proxy + deep MAGNITUDE ensemble sizes it
  ->  devices subscribed to a triggering station get a PUSH alert: detection + size + estimated shaking
```

```bash
.venv/Scripts/python scripts/live_watch.py --selftest   # deterministic pipeline check (dry-run)
.venv/Scripts/python scripts/live_watch.py --replay     # verify detection + sizing on cached events
.venv/Scripts/python scripts/live_watch.py              # LIVE (needs an always-on host + network)
```

## Web console — `app/`

React + Vite. `npm run dev`, opens on `localhost:5173`.

- **Hero** with a mouse-reactive synthetic seismograph.
- **Detect / Size / Warn** as an auto-cycling card carousel (7 s, pause, prev/next, shared "Evidence"
  disclosures with the demo figures + a technical paragraph for Size and Warn).
- **Biggest Southern California quakes** — a second carousel cycling Day / Week / Month / Year /
  All time, live from the USGS FDSN catalog, each row linking to its `sms-tsunami-warning.com` page.
- **Alert me near me** — subscribe to the **sensor stations** nearest you (city/state or "use my
  location" ranks the 10 stations by distance and auto-picks the nearest 3; toggle any). No coordinates
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

- **Coverage is the 10 SoCal stations the magnitude/EEW models were trained on.** Statewide needs a
  dataset rebuild + retrain (deferred).
- **Latency:** SeedLink is seconds-to-tens-of-seconds, so this is *rapid detection*, not sub-second
  pre-arrival warning (that's what ShakeAlert does on a dedicated low-latency pipeline).
- **Location is a proxy** (the strongest-triggering station), not a true locator, so distance and
  shaking are estimates. A real associator/locator is the natural next step.
- The magnitude regressor **underpredicts the very largest events** (a known trait); mid-range is closer.
- The "Biggest SoCal quakes" browser and the largest-quake links use the **USGS catalog** (real, but
  it's catalog metadata — the models don't produce those).

## Repository layout (current)

```
scripts/
  demo_detect.py / demo_magnitude.py / demo_eew.py   # reproducible per-model demos
  seismic_train.py / seismic_train_multi.py / seismic_eew*.py  # model definitions + training
  live_watch.py        # LIVE SeedLink daemon: detect -> coincidence -> size -> push alert
  push_fcm.py          # FCM push sender + device-token store (push_tokens.json)
  shaking_model.py     # magnitude/distance -> estimated shaking (MMI) + alert decision
  nearme_watch.py      # legacy USGS near-me watcher (retired from alerting; helpers reused by server)
  quake_archive.py     # per-day top-5 archive (min-heap), used by the largest-quakes feature
  server.py            # stdlib backend: /api/register-push, /api/stations, /api/ca, /api/geocode, ...
src/eq/                # seismic catalog + waveform helpers (quakecast.py, seismic.py)
app/                   # React + Vite + Capacitor console (web + Android)
data/processed/        # datasets (.npz) + model checkpoints (.pt) + result JSONs (gitignored)
figures/               # demo figures
tests/                 # pytest (pipeline + overfit-one-batch per sub-model)
```

Datasets/checkpoints and `data/subscribers.json` are gitignored. Rebuild seismic data with
`scripts/seismic_build*.py` (first run is network-bound via ObsPy/SCEDC).
