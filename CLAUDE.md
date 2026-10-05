# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project: SeismicSoCal ML

### What this is
Deep-learning seismology for **Southern California**, **deployed live at https://seismicsocal.duckdns.org**
(Oracle Always-Free A1 VM). Three models — **Detect** (is it a quake?), **Size** (magnitude), **Warn**
(early-warning shaking) — each shown working on real held-out SCEDC waveforms vs the classic seismology
baseline, plus a **live SeedLink daemon** that detects, picks, locates and sizes quakes on a real-time
19-station stream and **push-alerts** subscribers. Framed as a *research demonstration*, not an operational warning system.

_History: the project began as near-term (7-day) earthquake **forecasting** from geomagnetic (INTERMAGNET)
data. On real data that thesis came up **null** — superposed-epoch p=0.83, ROC ≈ chance, as the literature
predicts — and the geomagnetic pipeline was later **removed** (`src/eq/` now holds only the seismic
catalog/waveform helpers). Work pivoted to seismic-waveform deep learning, where the same
CNN/GNN/Transformer architecture genuinely works._

### The three models
Current numbers — **v2 dataset on the live network** (19 stations, 2000 → Aug 2026), 5-seed models,
**chronological 70/15/15** split:

| Task | Architecture | Deep (held-out test) | Baseline |
|------|--------------|----------------------|----------|
| **Detect** | CNN → Transformer (single-station, 30 s, 1 Hz HP) | AUC **0.9998**, MCC 0.886 (n=7,303) | STA/LTA 0.816 |
| **Size** | CNN → **GNN** → Transformer (multi-station, unit-peak + log-amp node feature) | R² **0.951**, MAE 0.10 (n=937, M2–5.2) | amp+dist 0.886 |
| **Warn (EEW)** | CNN → GNN → Transformer, first ~8 s → future PGV | alert **MCC 0.760** | GMPE-style 0.655 |

- Data: 6,243 magnitude events (M2.0–7.1) / 50,743 detection windows (34,377 event, 14,304 noise,
  2,062 hard negatives = the old daemon's own Sep-22..Oct-5 false declarations). `build_dataset.py`.
- Size: nearest-1-station ablation R² 0.808; live-like (10 km loc error, 3–6 stations) R² 0.939.
- **EEW is legacy** (earlier 10-station data, not retrained, not live).
- **Replay harness (the acceptance test)** on 10 held-out days: 93 % of confirmed events real (chance 0 %),
  5 pushes / 0 false, magnitudes within ±0.13 of catalog, median location error 2.5 km. Event-centric
  (616 test events, live geometry): mag bias +0.06, MAE 0.12, loc err 4.3 km. Old daemon, same days:
  6 pushes, all false.

### Locked rules — do not change without asking
- **Chronological splits only**, never random (temporal leakage). 70/15/15 by time. Hard negatives split
  by date (train Sep 22–28, val Sep 29–Oct 1, test Oct 2–5 2026); replay calibration uses validation days
  only, test days are scored once.
- **Honest metrics:** ROC-AUC + MCC (detection), R²/MAE vs baseline (magnitude), alerts by MCC/precision
  not recall alone; every live/replay precision is reported next to a **time-shifted chance baseline**.
- **Conservative public claims** — detection / characterization / rapid shaking estimation, NOT prediction.
- **Design: LIGHT MODE ONLY** — Ollama-referenced, monochrome (black ink / gray body / one black accent,
  filled-black pills, 12px cards), tokens in plain `:root` in `app/src/index.css`, sourced from `DESIGN.md`.
  No dark theme or toggle. Off-palette color is flagged by the impeccable design hook.
- **One station list:** `src/eq/network.py`. Builder, daemon, API, scorer, replay all import it. A station
  must stream on the public SeedLink relay (`build_dataset.py --stage check` enforces it).
- **Training input == live input.** Shared code only: `locate.pick_p` aligns training windows AND live
  windows; `pipeline.det_prep` is the detector input everywhere; `seismic.lowpass` (18 Hz) on every trace.
  Checkpoints carry their normalizers + station list; the daemon refuses a checkpoint whose stations differ.
  (There is no hand-kept `SCALE` constant any more.)

### How it fits together (Frontend / Online / Offline)
- **OFFLINE (PC).** `build_dataset.py` (USGS catalog M1+ via `quakecast.py`; one SCEDC request per event,
  response-removed, cached in `data/raw/v2/`) → `data/processed/v2/{detection,magnitude}.npz`;
  `demo_detect.py` / `demo_magnitude.py` train on the GPU → `detector.pt`, `magnitude_ensemble.pt`;
  `replay_archive.py` scans archived continuous data, calibrates `data/processed/v2/pipeline_config.json`
  (validation days) and scores test days. `app/public/seismic.json` holds the published numbers.
- **ONLINE (VM).** `server.py` auto-spawns `live_watch.py`, a thin SeedLink shell around
  `src/eq/pipeline.py` (detect → pick → locate → size → decide, all on data time). `/api/status` includes
  per-station health from `data/processed/live_status.json`. Other routes: `/api/ca`, `/api/geocode`,
  `/api/stations`, `/api/register-push`, `/api/unregister-push`, `/api/version`, `/api/contact`.
  EEW is NOT live; the push wording uses `shaking_model.py` (mag + distance).
- **FRONTEND (`app/`).** React + Vite + Capacitor. Detect/Size/Warn carousel (reads `seismic.json`),
  biggest-quakes carousel, "Alert me near me" (station subscription, mobile app only).

### Alerts: station subscription + located, 3-station confirmation (2026-10-05)
- **Subscribe to STATIONS, not a coordinate.** `push_tokens.json` = `[{token, stations:[codes], name}]`,
  no lat/lon stored. A device is alerted when a pushed event is within 150 km (`ALERT_REACH_KM`) of a
  station it follows; one message, distance from the located epicentre to its nearest followed station.
- **CONFIRMED** = >= 3 P picks that one grid-search location fits (RMS <= 1.5 s), at most 1 healthy
  nearer station silent, nearest pick <= 120 km. 1–2 stations → TENTATIVE (logged, never pushed).
- **Push** iff CONFIRMED and M >= 3.0 and `PUSH_ENABLED=1` (env; default OFF = shadow mode).
  Rationale + validation numbers: `data/processed/v2/pipeline_config_reason.json`, README.
- Coda of a big quake can re-trigger: picks within 120 s / 100 km of a declared event are absorbed
  (costs: an aftershock inside that window is only logged as tentative).

### Run it
- **Full stack (local):** `python scripts/server.py` + `cd app && npm run dev` (Vite proxies `/api` → `:8000`).
- **Rebuild data:** `python scripts/build_dataset.py` (network-bound ~2 h; `--stage check|select|fetch|assemble`).
- **Train:** `python scripts/demo_detect.py --retrain --seeds 5`, `python scripts/demo_magnitude.py --retrain --seeds 5`
  (CUDA torch is in `.venv`; RTX 4060).
- **Replay / acceptance:** `python scripts/replay_archive.py scan|calibrate|run|events|compare-live` (see docstring).
- **Daemon checks:** `python scripts/live_watch.py --selftest`; `pytest` (16 tests).

### Env / secrets
- `.venv`: torch 2.12.1+cu126, scikit-learn, matplotlib, obspy, scipy, pandas, ruff. Node/Vite for `app/`.
- **Gitignored:** `.env` (SMTP, `PUSH_ENABLED`), `fcm-service-account.json`, `data/processed/*` (npz, pt,
  json), `data/raw/*`. Models + `data/processed/v2/{pipeline_config,tt_correction}.json` ship by scp —
  **verify sha256 on the VM** (DEPLOY.md).

### Gotchas
- Imbalanced data → report AUC/MCC vs base rate, never accuracy/F1 alone.
- `data/raw/` holds the legacy cache (~11 GB) + `data/raw/v2/` (~2 GB) — exclude from any transfer.
- Pre-2010 SCEDC continuous archive is 40 Hz **BH**, not HH: the builder falls back to BH and every trace
  gets the common 18 Hz low-pass.
- Stations can drop off the public SeedLink relay (SCZ2 did during selection) — re-run
  `build_dataset.py --stage check` / `select_network.py` before relying on a station.
- USGS FDSN answers HTTP 400 (not a truncated list) above 20k rows; `quakecast` splits the interval.
- Low RAM (16 GB, other apps): long background jobs can be reaped; everything is resumable/cached.
- `crosscheck_events.py` line ~118 has a pre-existing unused variable (`c`) flagged by ruff F841.

### Deployed
Live at **https://seismicsocal.duckdns.org** (Oracle A1, `ubuntu@167.234.214.169`, `/opt/seismicsocal`,
Caddy + systemd). Update procedure and shadow-mode policy in `DEPLOY.md`. The VM was found (2026-10-04)
running Aug-10 models on the old 10-station network — always verify checkpoint hashes after shipping.

### Open TODOs
- **Shadow mode → pushes:** after deploying v2, run >= 7 days with `PUSH_ENABLED=0`, score with
  `crosscheck_events.py`, then enable.
- **Rebuild the APK** (station-network message in `App.tsx`; needs the Android SDK).
- **Replay the Sep 2020 validation week** (scan was reaped at 15/168 h for low memory) to widen calibration.
- **Seed-averaged magnitude R² with a CI** write-up (QuakeOps Phase 2).
- **Retrain EEW on the live network** and decide whether it belongs in the live alert.
- QuakeOps (MLflow / gate / drift): see `QUAKEOPS_PLAN.md`, `QUAKEOPS_IMPLEMENTATION.md` (to be re-based on v2).
