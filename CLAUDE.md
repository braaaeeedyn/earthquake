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
(Oracle Always-Free A1 VM). Two models — **Detect** (is it a quake?) and **Size** (magnitude) — plus a
4-second **quick check**, each shown on held-out SCEDC waveforms vs the classic baseline, run live by a
SeedLink daemon that detects, picks, locates and sizes quakes on a 19-station stream and push-alerts
subscribers in two stages (provisional, then confirmed or retracted). Research demonstration, not an
official warning system. Full method: `HOW_IT_WORKS.md`.

_History: the project began as near-term (7-day) earthquake **forecasting** from geomagnetic (INTERMAGNET)
data. On real data that thesis came up **null** — superposed-epoch p=0.83, ROC ≈ chance, as the literature
predicts — and the geomagnetic pipeline was later **removed** (`src/eq/` is the live library: station network, catalogue, waveform access, picker/locator, models, pipeline and stats). Work pivoted to seismic-waveform deep learning, where the same
CNN/GNN/Transformer architecture genuinely works._

### The models
Current numbers — **v2 dataset on the live network** (19 stations, 2000 → Aug 2026), 5-seed models,
**chronological 70/15/15** split:

| Task | Architecture | Deep (held-out test) | Baseline |
|------|--------------|----------------------|----------|
| **Detect** | CNN → Transformer (single-station, 30 s, 1 Hz HP) | AUC **0.9998** (CI .9997–.9999), MCC 0.886 (.868–.902) (n=7,303) | STA/LTA 0.816 (.805–.828) |
| **Size** | CNN → **GNN** → Transformer (multi-station, unit-peak + log-amp node feature) | R² **0.951** (.943–.959), MAE 0.10 (n=937, M2–5.2) | amp+dist 0.886 (.871–.898) |
| **Quick check** | median over stations of a·log10(peak vel, first 4 s of P) + b·log10(dist) + c | test MAE 0.24 | — |

- Data: 6,243 magnitude events (M2.0–7.1) / 50,743 detection windows (34,377 event, 14,304 noise,
  2,062 hard negatives = the old daemon's own Sep-22..Oct-5 false declarations). `build_dataset.py`.
- Detect false triggers per 30 s window at the LIVE trigger 0.6: 1.41 % test noise, 0.25 % held-out live noise
  (at the checkpoint's 0.9987: 0 % / 0.25 %). False EVENTS come from replay (3.5 confirmed/week, 0 pushes).
- Size: nearest-1-station ablation R² 0.808; live-like (10 km loc error, 3–6 stations) R² 0.939.
  Seed-averaged (10 seeds): single model R² 0.949 (t-CI 0.948–0.951), 10-seed ensemble 0.952 (bootstrap
  0.944–0.959); ΔR² vs baseline +0.067 (+0.056…+0.079). CIs = event-clustered bootstrap (`src/eq/stats.py`).
- **Replay harness (the acceptance test), 80 held-out days** (2022-04..2026-08, `data/processed/v2/replay/bigtest/`):
  145 pushes, all real quakes, 92 % within 60 km (12 out-of-network quakes mislocated 63–145 km, all 3-station);
  pushed mag bias +0.05 / MAE 0.12; confirmed precision 86 % busy / 79 % random days (chance 4 / 0 %), 7.4 false
  confirmed/week (logged only); median loc err 3–3.6 km; catch M3+ 68 % (CI 60–74); with ≥3 stations online: isolated M3+ 83 %
  (86 % good coverage / 80 % thin, 2nd-nearest station > 60 km) vs 60 % in sequences (`bigtest/misses.py`); M2+ 52 %; swarms 30 % vs isolated 73 %; 0 of 75 quakes inside the 120 s echo window caught.
  Timing: Standard ~31 s, Fast ~25 s, confirmation ~50 s. (Old 10-day check, 93 % / 0 false, was too small.)

### Locked rules — do not change without asking
- **Chronological splits only**, never random (temporal leakage). 70/15/15 by time. Hard negatives split
  by date (train Sep 22–28, val Sep 29–Oct 1, test Oct 2–5 2026); replay calibration uses validation days
  only, test days are scored once.
- **Honest metrics:** ROC-AUC + MCC (detection), R²/MAE vs baseline (magnitude), alerts by MCC/precision
  not recall alone; every live/replay precision is reported next to a **time-shifted chance baseline**.
- **Conservative public claims** — detection / characterization / rapid shaking estimation, NOT prediction.
- **Design: LIGHT MODE ONLY** — Ollama-referenced, monochrome (black ink / gray body / one black accent,
  filled-black pills, 12px cards), tokens in plain `:root` in `app/src/index.css`, sourced from `DESIGN.md`.
  Headings: self-hosted Literata serif; fluid column + clamp() type scale; prose #525252 (DESIGN.md "Project overrides").
  No dark theme or toggle. Off-palette color is flagged by the impeccable design hook.
- **One station list:** `src/eq/network.py`. Builder, daemon, API, scorer, replay all import it. A station
  must stream on the public SeedLink relay (`build_dataset.py --stage check` enforces it).
- **Training input == live input.** Shared code only: `locate.pick_p` aligns training windows AND live
  windows; `pipeline.det_prep` is the detector input everywhere; `seismic.lowpass` (18 Hz) on every trace.
  Checkpoints carry their normalizers + station list; the daemon refuses a checkpoint whose stations differ.
  (There is no hand-kept `SCALE` constant any more.)
- **Models reach production only through the QuakeOps gate** (`retrain.py`, rules G1–G6 in HOW_IT_WORKS §12)
  or an explicit `retrain.py rollback`. The gate compares champion vs challenger PAIRED on the challenger's
  chronological test split and replays the held-out days; it never edits `pipeline_config.json`.
  `scripts/tracking.py` is the only module that imports mlflow (no-op without `MLFLOW_TRACKING_URI`).

### How it fits together (Frontend / Online / Offline)
- **OFFLINE (PC).** `build_dataset.py` (USGS catalog M1+ via `src/eq/catalog.py`; one SCEDC request per event,
  response-removed, cached in `data/raw/v2/`) → `data/processed/v2/{detection,magnitude}.npz`;
  `demo_detect.py` / `demo_magnitude.py` train on the GPU (networks in `src/eq/models.py`) → `detector.pt`,
  `magnitude_ensemble.pt`;
  `replay_archive.py` scans archived continuous data, calibrates `data/processed/v2/pipeline_config.json`
  (validation days) and scores test days. `app/public/seismic.json` holds the published numbers.
- **ONLINE (VM).** `server.py` (env + operator email via `scripts/mailer.py`) auto-spawns `live_watch.py`, a thin SeedLink shell around
  `src/eq/pipeline.py` (detect → pick → locate → size → decide, all on data time). `/api/status` includes
  per-station health from `data/processed/live_status.json`. Other routes: `/api/ca`, `/api/geocode`,
  `/api/stations`, `/api/register-push`, `/api/unregister-push`, `/api/version`, `/api/contact`.
  `/api/shaking-model` (MMI coefficients for the app). Push wording uses `src/eq/shaking.py` (MMI at the
  subscriber's nearest followed station). `/api/ca` lists only catchable quakes and
  marks each caught / seen / missed against `events.jsonl`.
- **FRONTEND (`app/`).** React + Vite + Capacitor. Detect/Size carousel (reads `seismic.json`, evidence
  figures from `make_figures.py`), interactive coverage map (`Coverage.tsx`, `socal_cities.json`),
  biggest-quakes carousel with caught badges + "shaking at your home" line, `/quake` page (MMI at home),
  "Alert me near me" (region-first, mobile app only; also saves the home on the device),
  `/health` model-health page (from `/api/health`), 95% CIs under the card numbers. (Ten homepage drafts
  were prototyped 2026-10-05 and removed 2026-10-06; the original layout was kept.) `PRODUCT.md` = audience/voice/principles for the impeccable skill; `PORTFOLIO.md` = portfolio write-up.
- **QUAKEOPS (MLOps loop, HOW_IT_WORKS §12, design in `QUAKEOPS_IMPLEMENTATION.md`).** PC: MLflow-tracked
  training (`tracking.py`), monthly Dagster job (`quakeops_dagster.py` → `retrain.py`: data `--append` →
  train `--compare` champion → replay → gate → promote), CIs via `src/eq/stats.py`, `make_figures.py publish`.
  VM: `mlflow.service` (registry, Caddy basic auth), `seismicsocal-quakeops.timer` 09:30 UTC = `tracking.py pull`
  (models.json; installs only with `QUAKEOPS_AUTO_DEPLOY=1` / `--apply`) + `drift_check.py` (Evidently vs
  training noise, features logged by `live_watch.py`). CI: `.github/workflows/ci.yml` (ruff, pytest, selftest,
  build; tar deploy on main once secrets exist).

### Alerts: station subscription + located, 3-station confirmation (2026-10-05)
- **Subscribe to STATIONS, not a coordinate.** `push_tokens.json` = `[{token, stations:[codes], name, mode, caps}]`,
  no lat/lon stored (`caps: ["local_text"]` = app has native shaking text → gets data-only pushes). A device is alerted when a pushed event is within 150 km (`ALERT_REACH_KM`) of a
  station it follows; one message, distance from the located epicentre to its nearest followed station.
- **CONFIRMED** = >= 3 P picks that one grid-search location fits (RMS <= 1.5 s), at most 1 healthy
  nearer station silent, nearest pick <= 120 km. 1–2 stations → TENTATIVE (logged, never pushed).
- **Two-stage push** (needs `PUSH_ENABLED=1`; default OFF = shadow mode): (1) provisional push when CONFIRMED
  and the quick check of the subscriber's ALERT SPEED clears its threshold — per-device `mode` in push_tokens.json:
  `standard` (default; 4 s of P, response removed + 6 s taper margin, `early_mag.json` >= 3.04, ~33 s) or `fast`
  (2 s of P, sensitivity-scaled, no margin, `early_mag_T2.json` >= 3.05, ~26 s, rougher size, more retractions).
  Both run per event (`EARLY_PROFILES`); 20-day variant test in HOW_IT_WORKS §5.3; (2) after full sizing, a confirmation if
  M >= 3.0, else a retraction if (1) went out. Both carry the tag `quake-<event id>` so (2) replaces (1).
  Rationale + validation numbers: `data/processed/v2/pipeline_config_reason.json`, README.
- Coda of a big quake can re-trigger: picks within 30 s / 100 km of a declared event are absorbed; station rest 45 s
  (swarm tuning 2026-10-07, was 120 s / 60 s; HOW_IT_WORKS §6.4: 80-day M3+ catch 71.4 → 78.2 %, precision
  85.4 → 84.2 %, 0 duplicates). `split_picks` exists in the code but is off (adds false detections).

### Shaking at your location (MMI, 2026-10-07; HOW_IT_WORKS §13)
- `src/eq/shaking.py` = the one model: log10 PGV = a + bM + c log R + dR + e log(Vs30/631) + event_term
  (fit `calibrate_shaking.py` on v2 train events; test scatter 0.33 log10), Worden 2012 PGV→MMI, + DYFI
  `mmi_offset` 0.76. Same equations in `app/src/shaking.ts` and `app/native/android/QuakeMessagingService.java`,
  coefficients served by `/api/shaking-model` so all three stay identical.
- Vs30: USGS global slope-proxy grid cropped to SoCal → `app/public/vs30_socal.json` (0.02°, + station_vs30).
- Event term = clipped (±0.5) mean log residual of observed PGV at the sized stations (`Event.pgv_term`).
- Validation (`validate_mmi.py`, DYFI 10 km cells, offset fitted on older 29 quakes, scored on newer 28 / 1,892
  cells): MAE 0.42, bias +0.14, 93.8 % within one level; without offset 0.68; without site term 0.40.
- **Privacy:** the home is saved on the device only (localStorage + Preferences `seismic.home`); the server
  never sees it. Server push text uses the subscriber's nearest followed station instead.
- Current AND past quakes: pushes carry `{type, stage, mode, id, lat, lon, t0, mag, pgv_term, region}`; a tap opens
  `/quake`; the quake list shows a home-shaking line per quake.

### Run it
- **Full stack (local):** `python scripts/server.py` + `cd app && npm run dev` (Vite proxies `/api` → `:8000`).
- **Rebuild data:** `python scripts/build_dataset.py` (network-bound ~2 h; `--stage check|select|fetch|assemble`).
- **Train:** `python scripts/demo_detect.py --retrain --seeds 5`, `python scripts/demo_magnitude.py --retrain --seeds 5`
  (CUDA torch is in `.venv`; RTX 4060).
- **Replay / acceptance:** `python scripts/replay_archive.py scan|calibrate|run|events|compare-live` (see docstring).
- **Daemon checks:** `python scripts/live_watch.py --selftest`; `pytest` (32 tests); `ruff check scripts src tests`.
- **Shaking (MMI):** `python scripts/calibrate_shaking.py` (fit on v2 train events) → `python scripts/validate_mmi.py`
  (DYFI offset + held-out score; writes both into `data/processed/shaking_calibration.json`).
- **Long jobs on this PC:** launch detached (`Start-Process powershell -WindowStyle Hidden -File <job>.ps1`), log to a
  file, make them resumable — Claude Code's own background shells get reaped under low RAM. Examples:
  `data/processed/v2/replay/bigtest/run_bigtest.ps1`, `.../swarmtune/run_swarmtune.ps1`. Don't auto-restart a reaped job.
- **QuakeOps:** `python scripts/retrain.py --month YYYY-MM [--stage S] [--dry-run]`;
  `dagster dev -f scripts/quakeops_dagster.py`; `python scripts/tracking.py register-legacy|pull|status`;
  `python scripts/drift_check.py [--build-reference]`. Needs `MLFLOW_TRACKING_URI` (+ basic-auth user/pass) in `.env`.

### Env / secrets
- `.venv`: torch 2.12.1+cu126, scikit-learn, matplotlib, obspy, scipy, pandas, ruff, mlflow 3.16, evidently 0.7.23,
  dagster 1.13. Node/Vite for `app/`. `.env` QuakeOps keys: `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME/PASSWORD`
  (PC), `OPS_EMAIL_TO`, `QUAKEOPS_AUTO_DEPLOY` (VM).
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
- USGS FDSN answers HTTP 400 (not a truncated list) above 20k rows; `eq/catalog.py` splits the interval.
- Low RAM (16 GB, other apps): long background jobs can be reaped; everything is resumable/cached.
- `crosscheck_events.py` line ~118 has a pre-existing unused variable (`c`) flagged by ruff F841 (ignored per file in
  `ruff.toml`, with the other legacy-script F errors). Its `--time-tol` / `--dist-tol` help text says 180 s / 100 km;
  the real defaults are 30 s / 60 km.
- `shaking_calibration.json` is gitignored data that ships by scp. `shaking.load()` falls back to the built-in v2
  coefficients if it finds the pre-v2 file (no `site` key) — the VM had that file until 2026-10-07.
- Shaking levels round half UP everywhere (Python `math.floor(x+0.5)`, JS/Java `Math.round`); Python's `round`
  is banker's rounding and once disagreed with the app at MMI 2.5.
- The VM has a stray `x.ai/DESIGN.md` folder (Jul 7, a design reference); harmless, not part of the app.
- **Android builds happen on THIS PC now** (moved from the other device 2026-10-07): real project in `app/android/`
  (gitignored; `com.seismicsocal`), Firebase `google-services.json` inside it, signing key `~/.android/debug.keystore`
  (SHA-256 7dd3cbc8…; originals + the project zip in gitignored `other device/`). Never commit or lose the keystore.
  `build_apk.sh` handles Node 20 (Capacitor 8 needs 22 → npx node@22 for cap sync) and Anaconda's cygpath.
  The app module needs `firebase-messaging:25.0.1` declared itself (QuakeMessagingService).
- MLflow prints emoji; `tracking.run` makes stdout tolerant (a cp1252 console used to crash at run end).
- Low RAM: a local `mlflow server` + training at the same time can get reaped.
- Deploys (tar) never delete files: CI removes server files git no longer tracks under scripts/src/tests before
  restarting; do the same by hand (DEPLOY.md QuakeOps preamble). A stale `src/eq/models/` package once shadowed
  `src/eq/models.py` and crash-looped the daemon for ~2 min (2026-10-06).

### Deployed (status as of 2026-10-07)
Live at **https://seismicsocal.duckdns.org** (Oracle A1, `ubuntu@167.234.214.169`, `/opt/seismicsocal`,
Caddy + systemd; SSH key `~/.ssh/oracle_seismic` has a passphrase, so every SSH session needs the user).
- **Running:** `main` (deployed by CI on every push; v2 system, 19 stations — all up, two-stage alerts with the
  per-device alert speed, /api/health, serif headings + fluid layout + CIs). Detection live since 2026-10-05 09:13 UTC.
- **App 2.00.00 is a FORCED update** (server LATEST = MIN = 2.00.00; APK on /app built 2026-10-06). The APK is
  signed with the other device's debug key (different signature from 1.01.00), so updating = uninstall +
  reinstall; that wipes the app's stored subscription, so users must re-subscribe. Stale tokens are pruned
  automatically when FCM answers UNREGISTERED on the next push.
- **Both alert-speed profiles running** (`early_mag_T2.json` copied 2026-10-06; daemon log: `alert-speed profiles: ['fast', 'standard']`).
- **Shadow mode:** `PUSH_ENABLED=0` in the VM `.env` — detects, sizes and logs, sends NO pushes.
- **Swarm-tuned config live (2026-10-07 ~19:40 PDT):** `pipeline_config.json` event_sep_s 30, refractory 45
  (scp'd, sha256 verified). The shadow-mode precision check should cover days after this change.
- **Nightly crosscheck timer installed** (09:00 UTC) → `data/processed/crosscheck_report.json`.
- **MMI feature (2026-10-07):** server side deployed with `main` (shaking.py, /api/shaking-model, data fields on
  pushes, v2 `shaking_calibration.json` scp'd, sha256 verified); website shows home shaking. The native
  notification text needs **APK 2.01.00+** (2.01.01 on /app since 2026-10-07).
- **QuakeOps live on the VM (2026-10-06):** `mlflow.service` (registry, `https://mlflow.seismicsocal.duckdns.org`,
  basic auth user `quakeops`); detector + magnitude registered as v1 `@champion`; `seismicsocal-quakeops.timer`
  (09:30 UTC) pulls the champion (`QUAKEOPS_AUTO_DEPLOY=0`) and runs the drift check; `/health` shows both models.
  GitHub secrets `VM_HOST` / `VM_SSH_KEY` set (key `~/.ssh/seismic_ci`, no passphrase): pushes to `main` deploy.
- Deploy = one SSH session streaming a tar (git archive + app/dist + data/processed/v2/*.json); verify
  checkpoint sha256 on the VM (DEPLOY.md).

### Open TODOs
- **Shadow mode → pushes (~2026-10-12):** read the nightly report; if `confirmed` precision is well above
  its chance baseline and pushed magnitudes match USGS, set `PUSH_ENABLED=1` and restart. Replay predicts
  ~7 confirmed/day — if live stays far below that, replay the same hours and compare.
- **Out-of-network locations:** quakes north of MPM / south of the border are located with a one-sided
  station triple (e.g. a real M3.6 placed 66 km off); consider an azimuthal-gap flag in the push wording.
- **First retrain (not run yet):** `retrain.py --month 2026-09 --dry-run` (~hours: fetch + 2 trainings + re-scan of
  10 replay days), then schedule monthly (Dagster schedule or the Task Scheduler entry in DEPLOY.md step 9).
- **APK 2.01.01 RELEASED 2026-10-07 (soft update; LATEST 2.01.01, MIN 2.00.00), user testing it from the site.**
  2.01.00 = native home-shaking notifications; 2.01.01 = same + updated accuracy text (the APK bundles the site).
  Built on this PC, same debug key as 2.00.00 (installs over it). Not phone-tested before release: confirm
  push_tokens.json shows `caps: ["local_text"]` for the user's device and that a quake push shows the home-shaking
  line. Rollback: VM `app/dist/seismicsocal-2.01.00.apk.bak` (or `-2.00.00.apk.bak`) → `seismicsocal.apk`, and
  LATEST back. Release steps: bump `version.ts` + `app/android/app/build.gradle` + server LATEST, `build_apk.sh`,
  scp the APK to `app/dist/` (CI never ships .apk files).
- **Push wording is too long** (user, 2026-10-06): shorter alternatives were drafted, none chosen yet. Wording lives
  in `live_watch.push_message`; the native app prepends its own home-shaking line.
- **Vs30 site term** gave no measurable gain on DYFI (MAE 0.40 without vs 0.42 with). Kept for physics; re-check
  with more quakes or point (not 10 km cell) data before claiming it helps.
- Drift reference has only ~280–650 training noise windows per station; if `watch` flaps, grow `NOISE_TIMES`.
