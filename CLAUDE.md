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

## Project: Earthquake Forecasting App

### What this is
Near-term (7-day) earthquake forecasting from INTERMAGNET geomagnetic sensor data.
Three fused models (CNN spatial, GNN inter-station, Transformer temporal) feed a
binary classifier. Target: beat the ~60% accuracy benchmark from Liu et al. (2022).

### Locked rules — do not change without asking
- Data shape: minute readings -> hourly averages -> daily vectors of 24 ->
  stacked into 24x27 matrices (24 hours x 27 days). This pipeline is deterministic.
- Labels: each matrix is paired with whether an earthquake occurred in the
  following 7 days, per fixed magnitude/proximity/time rules.
- Train/val/test splits MUST be chronological, never random (avoid temporal leakage).
- Report precision, recall, and F1 — not accuracy alone. The data is heavily
  imbalanced (earthquakes are rare); recall on the positive class matters most.

### Build order
1. Acquire + define inputs (station data, earthquake catalog, labeling rules).
2. Build the deterministic data pipeline (raw -> labeled 24x27 matrices).
3. CNN, then 4. GNN, then 5. Transformer.
6. Fuse into final classifier, validate vs 60% baseline.
7. Build the app around the trained model.
8. Produce evidence figures.

### Working style
- Before writing pipeline or model code, confirm the relevant locked rules with me.
- Keep public-facing prediction claims conservative — geomagnetic earthquake
  prediction is scientifically contested.

---

## Progress & handoff (last updated 2026-07-07)

Read this before continuing. The project now has **three tracks**. The original thesis
(geomagnetic *forecasting*) came up **null on real data**, so work pivoted to **seismic
waveform deep learning**. Frame everything below by which track it belongs to — the
numbers only make sense that way. Prior handoffs describing a single "fused forecaster
beats 60%" are obsolete: that number was on the synthetic fixture and does not survive
real data.

### Track 1 — Geomagnetic earthquake *forecasting* (original thesis) → NULL, retired
The full pipeline + fused model flow was built and run against **real** INTERMAGNET +
USGS data. It does not work, as the literature predicts:
- Superposed epoch analysis (`scripts/superposed_epoch.py` → `data/processed/sea_results.json`):
  CA, 26 events, pre-quake mean z=0.15, **p=0.83**. No precursor signal.
- ROC AUC across `methodology`/`horizon`/`model_breakdown` experiments: **~0.47–0.60**
  (at/near chance), with wide CIs.
- The superficially "good" F1/accuracy (e.g. Japan mag5/500km vector exp F1=0.80) is
  **base-rate inflation** — base rate there is 0.685, and ROC≈0.5 confirms no real skill.
- **Do not present this as a working forecaster.** It IS legitimate as a rigorous *null /
  replication* result (headline: superposed-epoch p=0.83). Keep public claims conservative.

Assets: `src/eq/config.py`, `src/eq/pipeline.py` (minute→hourly→daily→24×27, labeling,
chronological split + 34-day embargo, train-only normalizer), `src/eq/graph.py`,
`src/eq/synthetic.py`, `src/eq/features.py`, and the many `scripts/*_experiment.py`.
Model flow in `src/eq/models/` (`cnn`/`gnn`/`transformer`/`fusion` + `data`/`metrics`/`train`);
every sub-model is a feature extractor with a uniform `forward(x)->(B,out_dim)` interface,
`Classifier` wraps one standalone, `FusedClassifier` late-fuses all three reusing the same
backbones — keep this contract if extending.

### Track 2 — Seismic *detection* from waveforms (the pivot) → STRONG, real
`src/eq/seismic.py` (SCEDC/ObsPy fetch, 30s vertical-component windows) + `scripts/seismic_*`.
```
DETECTION (test n=300, 80 event): CNN+Transformer AUC=0.988  MCC=+0.848
                                  STA/LTA baseline AUC=0.625
```
Genuinely works, decisively beats the classical baseline. This is the flagship deliverable.
Caveat: detection is a known-solved problem (PhaseNet/EQTransformer exist) — strong as a
**demonstration/portfolio** result, not a novel one.

### Track 3 — Seismic *magnitude* + *early-warning (EEW)* → deep WINS on multi-station data (with caveats)
IMPORTANT: two data regimes exist and give opposite verdicts. The **single-station, small-data**
runs (`seismic_train.log`/`seismic_eew.log`, n=60–80) show deep LOSING. The **multi-station,
794-event** runs (`seismic_xl_eval.log`, 10 stations) show deep WINNING. The multi-station
result is the real, current one — use it, and don't quote the old single-station numbers.
```
MAGNITUDE (multi, seismic_xl_eval.log): deep R2=0.771 MAE=0.153 | amp+dist R2=0.690 MAE=0.153
                                        deep wins on R² (MAE tied); 1-station ablation R2=-0.54
                                        → multi-station GNN fusion is doing the real work
EEW · PGV regression:  deep ENSEMBLE R2=0.728 | baseline R2=0.720        → ~tie
EEW · shaking alert:   deep recall 0.65 MCC 0.739 | base 0.55 MCC 0.686  → deep wins (recall)
```
Caveats to keep honest: (1) magnitude R² swung **0.570 (big run) → 0.771 (xl run)** — the
figure uses the better one; a seed-averaged number with CI is still owed. (2) EEW single
models range R² −0.06..0.79; only the **5-model ensemble** clears the baseline. Report the
ensemble, honestly. These caveats shape rigor, not the verdict: on the multi-station data
deep genuinely beats the physics baselines.

### Also present
- **Web app surface.** `app/` React+Vite+Capacitor, runs at `localhost:5173`
  (`cd app && npm run dev`). Already pivoted off geomagnetic: `app/src/App.tsx` loads
  `seismic.json` and renders the honest results page (`seismic_results.html` is the
  standalone version). `scripts/make_forecast.py` is a retired geomagnetic PLACEHOLDER.
- **Tests:** pipeline/graph + overfit-one-batch per sub-model & fusion (`pytest`).
- **Env:** `.venv` has `torch`, `scikit-learn`, `matplotlib`, `obspy` (`requirements-ml.txt`).
- **Figures:** `figures/seismic_results.png` (summary), plus the per-demo figures below.

### LOCKED DIRECTION (decided 2026-07-06) — the three-panel console
The project's deliverable is a **seismic early-warning demonstration console**: one app with
three panels, one per model, each shown working on real held-out California data. Framed
honestly as a *research demonstration*, NOT an operational system. The three answer the
monitoring pipeline's questions in order — **Detect → Size → Warn** — but are independent
parts (they don't need to feed each other).

**Reproducible per-model demos (build order 1 — DONE, all train-once-and-cache):**
- `scripts/demo_detect.py`   → detection. Deep AUC ~0.988 vs STA/LTA ~0.62; prints held-out
  windows where deep is right and STA/LTA wrong. Artifacts: `data/processed/detector.pt`,
  `detection_demo.json`, `figures/detection_demo.png`.
- `scripts/demo_magnitude.py` → magnitude. Seed-ENSEMBLE (single models swing 0.57..0.86, so
  ensembling is the honest fix). Reports ensemble R² vs amp+dist baseline 0.690 + nearest-1-
  station ablation. Artifacts: `magnitude_ensemble.pt`, `magnitude_demo.json`, `figures/magnitude_demo.png`.
- `scripts/demo_eew.py`      → early warning. Seed-ENSEMBLE; predicts future PGV from first 8 s;
  headline metric is strong-shaking **alert recall**. Artifacts: `eew_ensemble.pt`,
  `eew_demo.json`, `figures/eew_demo.png`.
  Full/publishable runs: add `--retrain --seeds 5` (detect/mag) or `--k 5` (eew).

**Console UI (build order 2 — DONE, with a two-theme redesign; replay mode still TODO):**
`app/src/App.tsx` renders three result ROWS (01 Detect / 02 Size / 03 Warn — a real pipeline,
so numbered), each = big mono stat + baseline + verdict + one sentence + an "Evidence" `<details>`
disclosure holding its demo figure (`app/public/{detection,magnitude,eew}_demo.png`). Reads
`app/public/seismic.json` (real reproducible ensemble numbers: detection AUC 0.977, magnitude
R² 0.846, EEW R² 0.728 / alert recall 0.65). Hero has a synthetic seismogram-trace signature.
**LIGHT MODE ONLY (decided 2026-07-06).** The dark/xAI theme + the theme toggle were REMOVED —
the user chose the light look and wants it for the rest of the project. Do NOT reintroduce a
dark theme or toggle. The single theme is **Ollama-referenced light** (paper-white, center
README hero, monochrome — black ink / gray body / one black "accent", filled-black pills, 12px
cards), sourced from `DESIGN.md`. Tokens live in plain `:root` in `app/src/index.css` (no
`data-theme` scoping) with a 4pt spacing scale. Kept monochrome: error text and M5+ magnitude
use font-weight for emphasis, no color (impeccable design hook flags any off-palette color).
Verified via headless screenshot; TS build passes. (`x.ai/DESIGN.md` is now unused reference.)
Near-me subscribe section (`NearMe` + `app/src/nearme.ts`) + backend `scripts/server.py`
(`POST /api/subscribe` → `data/subscribers.json`, `GET /api/events` live USGS); Vite proxies
`/api` → `:8000`. Run the full stack: `python scripts/server.py` + `cd app && npm run dev`.
FONT CAVEAT: exact Ollama "rounded" display needs the Nunito webfont; on Windows it falls back
to system-ui (looks like a normal sans). Add `@fontsource/nunito` + `@fontsource/inter` +
`@fontsource/geist-mono` for pixel-fidelity to both brands.
STILL TODO: the **"replay a real earthquake" mode** (pick a historical CA event → walk it
through detect → magnitude → alert + lead time).

**"Near me" alert feature — DECIDED + FIRST VERSION BUILT (2026-07-06).**
Key honesty point that shaped the design: the project's ML models do NOT detect quakes in real
time (they run offline on cached waveforms — no live seismic stream). So the alert product rides
on the **public USGS realtime feed** (real events, ~1-min latency), which already detects/locates
quakes; the ML models are a separate research-demo/enrichment layer. Decisions:
- **Event source:** real USGS realtime GeoJSON feed (`.../summary/{feed}.geojson`).
- **Notifier:** EMAIL first (SMTP), as a testable opt-in; SMS/native-push later behind the same
  `notify()` seam; a mobile app is the eventual target.
- **"Near" zone:** MAGNITUDE-SCALED radius `felt_radius_km(mag)=10**(0.30+0.333*mag)` clamped
  [10,400] km (M3≈20, M4.5≈70, M6≈200). NOT a fixed 1-mile radius (seismologically wrong).
- **Server-side:** all work on the server; a subscriber only supplies {location, contact}.

BUILT: `scripts/nearme_watch.py` — polls USGS, dedups seen ids (`data/processed/nearme_seen.json`),
matches subscribers (`data/subscribers.json` = [{name,lat,lon,email}]) in the felt radius, emails
them. Modes: `--selftest` (deterministic: fabricates a quake on subscriber[0], verified working),
`--feed 2.5_day --once --dry-run` (test vs real past-day events, verified: 24 events, radii + match
correct), `--interval 60` (live loop). Email creds via env `SMTP_USER`/`SMTP_PASS` (Gmail app
password); `--dry-run` or missing creds prints instead of sends. NOT yet running as a hosted
service (needs an always-on host + a subscribe UI/endpoint — next step for this feature).
The three-panel console's EEW model can later enrich alerts with expected shaking AT the user.

### 2026-07-07 update — SeismicSoCal ML: LIVE detection + product hardening
The project is now branded **SeismicSoCal ML** (renamed from "Seismic ML") and scoped to
**Southern California** (the 10 CI/SCEDC stations the models were trained on). Statewide coverage
is deferred (needs a dataset rebuild + retrain of the magnitude/EEW models).

- **LIVE detection daemon — `scripts/live_watch.py` (NEW, the headline).** Bypasses USGS: streams
  the 10 SoCal stations over **SeedLink** (ObsPy `EasySeedLinkClient`, default IRIS rtserve), runs
  the pre-trained **detection** model continuously on sliding 30 s windows, declares an event only
  on **multi-station COINCIDENCE** (≥K stations agree within ~12 s — the false-alarm killer), sizes
  it with the in-distribution **deep magnitude ensemble**, and emails nearby subscribers detection +
  size + estimated shaking. Verified: `--replay` (detection 0.99/0.01, deep magnitude runs),
  `--selftest` (coincidence→alert dry-run), and a live SeedLink connect + IRIS response fetch.
  Honest caveats baked into the emails: SeedLink latency is seconds-to-tens (RAPID detection, not
  sub-second pre-arrival warning), and event **location is a proxy** (strongest-triggering station,
  not a real locator) → distance/shaking are estimates. Needs an always-on host to actually run.
  **(2026-07-07) `server.py` now AUTO-SPAWNS `live_watch.py` as a child process — it is THE alert
  daemon for "Alert me near me". USGS is no longer used for alerts, only for the `/api/ca`
  largest-quakes display. `nearme_watch.py` is retired from alerting (its USGS poll loop is unused);
  `server.py` still imports its helpers (`fetch_usgs`/`load_json`/`send_email`/`SUBS`).**
- **`scripts/shaking_model.py` (NEW).** magnitude+distance → estimated MMI + description + alert
  decision (`estimate_mmi`, `describe`, `should_alert`). Used by both `live_watch.py` and the
  (legacy) `nearme_watch.py`, which was upgraded from a felt-radius decision to a shaking-based one.
- **EEW alert honesty fix.** `demo_eew.py` now tunes the alert trigger on the VALIDATION split with
  a recall-weighted F2 objective (fixed "strong shaking" = train 70th-pct PGV), applied fairly to
  both models. Recall 0.65→**0.76** @ precision 0.82; the app's Warn headline switched to **MCC
  0.760 vs 0.655** (recall alone is gameable once you tune thresholds — MCC is the fair summary).
- **"Biggest SoCal quakes" browser.** `server.py` `/api/ca?window=day|week|month|year|all` queries
  the USGS FDSN catalog for the top-5 largest SoCal quakes (calendar month/year, place-filtered to
  drop NV/Baja border events). Frontend renders it as a **second carousel** (same interaction as the
  Detect/Size/Warn cards) cycling the five windows. `quake_archive.py` (per-day top-5 min-heap) also
  exists but the live FDSN queries superseded it for this feature.
- **Geocoded location search.** `server.py` `/api/geocode` (Nominatim, viewbox-bounded to SoCal).
  The subscribe form has **separate City + State inputs** joined before search (`City, CA`).
- **Console UI (heavily iterated).** Detect/Size/Warn is now an **auto-cycling carousel** (7 s,
  pause, prev/next, dots, shared "Evidence" disclosures that animate open via grid-rows; Size/Warn
  evidence includes a technical paragraph). Card copy rewritten in **plain language**. Hero has a
  **mouse-reactive synthetic seismograph** (amplitude bells toward the centre, grows as the cursor
  nears). A Plasma WebGL background was tried and **removed** (kept monochrome light per the locked
  design). Nav badge now reads **"live · SeedLink"** (was USGS). Names updated app-wide.
- **Secrets/PII:** `.env` (SMTP) and now **`data/subscribers.json`** are gitignored — subscriber
  emails must not be committed.

### 2026-09-21 update — alerts moved to STATION-SUBSCRIPTION + GRADED coincidence
Reverses the earlier "magnitude-scaled felt radius" + "store {location, contact}" alert decisions.
Rationale: with a proxy epicenter (strongest station), distance-based targeting was fuzzy, and the
5-of-10 coincidence gate missed small (M~3.5) quakes only 1–2 stations can feel. Verified via
`live_watch.py --selftest` (both tiers + targeting) and a clean `app` TS build.
- **Subscription unit is now a STATION, not a coordinate.** `push_tokens.json` stores
  `{token, stations:[codes], name}` — **no lat/lon**. `push_fcm.save_token(token, stations, name)`.
  A device's location is used only client-side at signup to rank stations, then discarded.
- **Signup UX (`App.tsx` `NearMe`).** "Use my location" / city search → computes the user's distance
  to each of the 10 stations, shows them nearest-first, **auto-selects the nearest 3 within a 150 km
  cap** (`NEAR_TOP_N`/`NEAR_CAP_KM`), and each station is a tap-toggle (that's the per-sensor
  unsubscribe). New `GET /api/stations` (server) serves codes+coords; `nearme.ts` `getStations()`.
  `/api/register-push` now takes `{token, stations}` and validates codes ⊂ `STATION_CODES`.
- **Graded declaration (`live_watch.py` `declare_graded`).** The hard 5-station gate is gone.
  `MIN_STATIONS=2` = CONFIRM tier (≥2 agree + move-out check → confirmed, sized by the magnitude
  ensemble). A lone station at prob ≥ `LONE_THRESH=0.85` → TENTATIVE alert, labelled a possible false
  alarm (false positives > false negatives, but flagged). Weak lone triggers are suppressed. Cooldown
  is now per-strongest-station, not one global timestamp. `events.jsonl` gains a `confirmed` flag.
- **Alerts are per-station, combined, and distance-personalised.** `alert_push_devices` pushes each
  device subscribed to ANY triggering station exactly once. The message states the quake is nearest
  to the strongest station and **how far that is from the user** — distance from the proxy epicentre
  to the user's NEAREST subscribed station (from station coords; still no stored user location).
  `shaking_model` then estimates intensity at that distance for the message. Confirmed → "N sensors
  agree — nearest to <strongest>, ~D km from you. Estimated M<mag>. <intensity> shaking expected.";
  tentative → "one sensor … <D km from you> — unconfirmed, may be a false alarm". `shaking_model` is
  on the MESSAGE path (intensity string), not the alert DECISION (which is station membership + tier).
- Privacy policy copy updated (we no longer store location). `@capacitor/geolocation` was declared
  but missing from `node_modules` on this checkout — `npm install` fixed it (pre-existing, unrelated).

### Next steps (prioritized, decided 2026-07-07)
Direction after wiring `live_watch.py` as the auto-spawned alert daemon. Roughly in order:
1. **Deploy the stack to an always-on host.** The alert product only watches while running;
   `live_watch.py` needs continuous SeedLink + an always-on process (`server.py` now auto-spawns it).
   Hosting is the only blocker between demo and working product. Needs: a small always-on VM, SMTP
   creds in the host env, and a decision on whether the watcher auto-restarts when the stream drops
   (today it exits and the API keeps serving).
2. **Build the "replay a real earthquake" mode** (the locked demo TODO). Pick a historical CA event →
   walk it Detect → Size → Warn, showing the alert fire + LEAD TIME. Exercises all three models on one
   real event; self-contained, no hosting. Highest demonstration payoff.
3. **Seed-averaged magnitude R² with a CI** (rigor debt). R² swung 0.57 → 0.77 → 0.85 across runs; the
   app shows the best single number. Report mean ± CI across seeds — small script change, matches the
   project's honest-demo framing.
4. Polish/scope (lower priority): add `@fontsource` nunito/inter/geist-mono for design fidelity on
   Windows; wire the EEW/PGV deep model into alert emails (shaking is currently the mag/dist formula,
   not the trained net); statewide coverage (needs a dataset rebuild + magnitude/EEW retrain).

### Gotchas
- Real data is heavily imbalanced (earthquakes rare) — always report precision/recall/F1
  AND ROC/PR-AUC vs base rate; a high F1 alone can be pure base-rate inflation (see Track 1).
- Seismic data is fetched from SCEDC via ObsPy and cached under `data/raw/seismic*/`; first
  runs are network-bound. Geomagnetic raw lives in `data/raw/intermagnet*`.
- Result JSONs + run logs are in `data/processed/`; the `.npz` datasets there are gitignored.
  Rebuild geomag with `scripts/build_dataset.py`, seismic with `scripts/seismic_build*.py`.
- Everything through the initial commit (`51d9af1`) is committed; the only uncommitted change
  is this handoff rewrite. Commit it when you pick the work back up.
