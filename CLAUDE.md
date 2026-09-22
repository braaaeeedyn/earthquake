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
baseline, plus a **live SeedLink daemon** that runs detection + magnitude on a real-time station stream
and **push-alerts** subscribers. Framed as a *research demonstration*, not an operational warning system.

_History: the project began as near-term (7-day) earthquake **forecasting** from geomagnetic (INTERMAGNET)
data. On real data that thesis came up **null** — superposed-epoch p=0.83, ROC ≈ chance, as the literature
predicts — and the geomagnetic pipeline was later **removed** (`src/eq/` now holds only the seismic
catalog/waveform helpers). Work pivoted to seismic-waveform deep learning, where the same
CNN/GNN/Transformer architecture genuinely works._

### The three models
Current numbers — **2000–2025 dataset**, 5-seed ensembles on a **chronological 70/15/15** split:

| Task | Architecture | Deep (held-out test) | Baseline |
|------|--------------|----------------------|----------|
| **Detect** | CNN → Transformer (single-station) | AUC **0.992**, MCC 0.930 (n=880) | STA/LTA 0.550 |
| **Size** | CNN → **GNN** → Transformer (multi-station) | R² **0.840**, MAE 0.12 (n=169) | amp+dist 0.749 |
| **Warn (EEW)** | CNN → GNN → Transformer, first ~8 s → future PGV | alert **MCC 0.760**, recall 0.76 | GMPE-style 0.655 |

- Detect + Size were **retrained on the 2000–2025 data** (1,126 magnitude events / 5,863 detection
  windows). EEW numbers are from the earlier run — **EEW has not been retrained.**
- Size: the nearest-1-station ablation drops to R² **+0.42** (vs 0.840) — the multi-station **GNN fusion**
  is what earns the win. Detect uses no GNN (one window, one station).

### Locked rules — do not change without asking
- **Chronological splits only**, never random (temporal leakage). `split_chrono` = 70/15/15 by event time.
- **Honest metrics:** ROC-AUC + MCC (detection), R²/MAE vs baseline (magnitude), and for alerts report
  **MCC, not recall alone** (recall is gameable once you tune the threshold). Data is imbalanced.
- **Conservative public claims** — this does detection / characterization / rapid shaking estimation,
  NOT earthquake prediction (whether one will occur; unsolved).
- **Design: LIGHT MODE ONLY** — Ollama-referenced, monochrome (black ink / gray body / one black accent,
  filled-black pills, 12px cards), tokens in plain `:root` in `app/src/index.css`, sourced from `DESIGN.md`.
  No dark theme or toggle. Off-palette color is flagged by the impeccable design hook.
- **Live magnitude SCALE:** `live_watch.py` normalises waveforms by a hardcoded `SCALE` = the training
  set's `X[mask].std()`. If you rebuild the magnitude dataset, **update `SCALE`** to match or live
  magnitudes are biased (last set to `6.954687e-4` for the 2000–2025 data).

### How it fits together (Frontend / Online / Offline)
- **OFFLINE (training).** `seismic_build*.py` fetch SCEDC waveforms (labeled against the USGS catalog via
  `quakecast.py`) → `.npz` datasets; `seismic_train*.py` / `demo_*.py` train → checkpoints
  (`detector.pt`, `magnitude_ensemble.pt`, `eew_ensemble.pt`) + figures + `app/public/seismic.json`.
  Nothing user-facing computes live; the app reads the cached numbers.
- **ONLINE (live).** `server.py` (stdlib backend, no framework) **auto-spawns `live_watch.py`**.
  `live_watch` streams the 10 CI/SCEDC stations over **SeedLink**, runs **detection continuously**,
  declares on **graded coincidence** (below), sizes **confirmed** events with the magnitude ensemble, and
  **pushes (FCM)** devices subscribed to any triggering station. `server.py` also serves `/api/ca`
  (biggest SoCal quakes, USGS/FDSN), `/api/geocode`, `/api/stations`, `/api/register-push`,
  `/api/unregister-push`, `/api/status`, `/api/contact`. **EEW is NOT in the live loop** (offline evidence
  only); the live "warning" shaking is the `shaking_model.py` mag+distance formula, not the trained net.
- **FRONTEND (`app/`).** React + Vite + Capacitor. Detect/Size/Warn carousel (reads `seismic.json`),
  a biggest-quakes carousel, and "Alert me near me" (station-subscription push — **mobile app only**;
  web has no push). Android APK downloadable from `/app`.

### Alerts: station-subscription + graded coincidence (design, 2026-09-21)
- **Subscribe to STATIONS, not a coordinate.** `data/processed/push_tokens.json` = `[{token, stations:[codes], name}]`
  — **no lat/lon stored.** Signup ranks the 10 stations by distance, auto-selects the nearest 3 within a
  150 km cap (`NEAR_TOP_N`/`NEAR_CAP_KM` in `App.tsx`), and each is a tap-toggle (per-sensor unsubscribe).
- **Graded declaration (`declare_graded`):** ≥2 stations agree + move-out check → **CONFIRMED** (sized);
  a lone station at prob ≥ `LONE_THRESH=0.85` → **TENTATIVE** (labelled a possible false alarm); weak lone
  triggers suppressed. Cooldown is per-strongest-station. `events.jsonl` logs each declaration (with a
  `confirmed` flag) for `crosscheck_events.py`.
- **One combined push per device**, personalised by distance from the epicenter-proxy (strongest station)
  to the user's nearest subscribed station; `shaking_model` gives the intensity string. `shaking_model` is
  on the MESSAGE path (wording), NOT the alert DECISION (which is station membership + coincidence tier).

### Run it
- **Full stack (local):** `python scripts/server.py` + `cd app && npm run dev` (Vite proxies `/api` → `:8000`).
- **Demos:** `python scripts/demo_{detect,magnitude,eew}.py` (add `--retrain --seeds 5`, or `--k 5` for eew).
- **Daemon checks:** `python scripts/live_watch.py --selftest` (dry-run, both alert tiers) / `--replay` (cached).
- **Rebuild data:** `python scripts/seismic_build.py` / `seismic_build_multi.py` (accept `--start`/`--end`;
  first run is network-bound via ObsPy/SCEDC). See `DEPLOY.md` for the live-host update procedure.

### Env / secrets
- `.venv` has `torch`, `scikit-learn`, `matplotlib`, `obspy` (`requirements-ml.txt`). Node/Vite for `app/`.
- **Gitignored:** `.env` (SMTP), `fcm-service-account.json` (FCM push creds), `data/subscribers.json`,
  `data/processed/*.npz` + `*.pt`. Datasets/checkpoints don't travel with git — **scp them on deploy.**

### Gotchas
- Imbalanced data → report AUC/MCC vs base rate, never accuracy/F1 alone (a high F1 can be pure base rate).
- SCEDC waveforms are cached under `data/raw/` (~11 GB) — **exclude it from any transfer.**
- `load_catalog` caches to a **date-agnostic** CSV (`data/raw/usgs_california_seis_m2.5.csv`) — move it
  aside before re-fetching a different date range, or the old range is silently reused.
- The magnitude regressor **underpredicts the very largest events** (a known trait); mid-range is closer.
- In the last detection rebuild **PFO returned no data** → 5 of the 6 detection stations were used;
  investigate its channel availability to restore the 6th.

### Deployed (2026-09-22)
Live at **https://seismicsocal.duckdns.org** (Oracle A1, `ubuntu@167.234.214.169`, `/opt/seismicsocal`,
Caddy + systemd; the `seismicsocal` service runs the backend and auto-spawns the daemon). The
backend + daemon run the **2000–2025 retrained models** with the corrected `SCALE`; the site shows
0.992 / 0.840; the **Android APK** (built against the live backend) is served at `/app`. Subscribers live
in `data/processed/push_tokens.json` **on the VM**. Update procedure = ship code + `.pt` models, `npm run
build`, `sudo systemctl restart seismicsocal` (details in `DEPLOY.md`).

### Open TODOs
- **"Replay a real earthquake" mode** — pick a historical CA event → walk it Detect → Size → Warn, showing
  the alert fire + lead time. Self-contained, high demonstration payoff.
- **Seed-averaged magnitude R² with a CI** — a 5-variant fine-tune search (augment / cosine LR / Huber /
  dropout / physics-blend) found **no reliable gain** over 0.840 (the apparent cosine win evaporated at
  matched seeds), so the ceiling looks real; a mean±CI writeup is still owed.
- **Wire the EEW net into the live alert** (currently the mag/dist shaking formula) + surface the latency
  caveat in the alert text.
- **Statewide coverage** (dataset rebuild + magnitude/EEW retrain); restore PFO; add `@fontsource`
  nunito/inter/geist-mono for design fidelity on Windows.
