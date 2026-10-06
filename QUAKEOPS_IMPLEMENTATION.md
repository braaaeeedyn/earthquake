# QuakeOps: implementation strategy (re-based on v2, 2026-10-05)

> Companion to `QUAKEOPS_PLAN.md` (the *what* and *why*). This file is the *how*, rewritten against the
> **v2 system** as it runs today (19-station network, `src/eq/pipeline.py` engine, replay harness as the
> acceptance test, two-stage pushes). The pre-v2 version of this file assumed a 10-station network, an
> EEW model, a hand-kept `SCALE`, a frozen `test_v1` and monthly npz shards; none of that applies any more.
>
> **Guiding rule:** extend what exists. QuakeOps adds four technologies (**MLflow, Evidently, Dagster,
> GitHub Actions**) and otherwise reuses the current scripts: `demo_detect.py` / `demo_magnitude.py`
> (training + evaluation), `build_dataset.py` (data), `replay_archive.py` (acceptance), `server.py`
> (API), the `crosscheck` systemd-timer pattern, a shared mailer (`mailer.send_email`), and the
> `server.py` supervisor that already respawns the daemon.

---

## Summary: what was built, how it works, when you'll see it

**What it did.** It turned the two models into a system that keeps itself up to date and leaves a record
of every step:
- every training run is tracked (MLflow), with lineage and 95% confidence intervals;
- models are versioned in a registry (`champion` / `challenger`);
- a monthly job (Dagster, on the PC) adds the newest month of data, retrains, and promotes a new model
  **only** if it passes a statistical and replay gate;
- the live stream is checked every day for drift from the training data;
- a `/health` page shows all of this;
- CI tests and builds every change, and deploys `main`.

**How it works, in one pass:**
1. `retrain.py --month M` grows the dataset with `build_dataset.py --append`, which keeps every earlier
   selection.
2. It trains a challenger detector and magnitude ensemble with the existing demo scripts.
3. It scores champion and challenger **paired** on the challenger's chronological test split, which
   neither model has seen.
4. It replays the 10 held-out days with each challenger.
5. It applies rules G1–G6: beat the baseline, be no worse than the champion (including false triggers at
   the live 0.6 threshold), pass the replay (precision vs chance, no false pushes, magnitudes vs the
   catalogue), tests green, clean lineage, and a reason to switch.
6. Each model that passes becomes `@champion`.
7. On the VM, a daily timer runs `tracking.py pull`. It records the champion in `models.json` and
   installs it only if `QUAKEOPS_AUTO_DEPLOY=1`, or when you run `pull --apply`. The daemon then
   restarts on the new model.
8. The same timer runs `drift_check.py`. It compares the features `live_watch.py` logs (detector score,
   spikiness, high-frequency share; all scale-free) with each station's training noise.

**When you'll see it:**

| Where | When |
|---|---|
| 95% CIs under the Detect / Size numbers on the site, `/health` link in the footer | live (2026-10-06) |
| `/health` → "Live models" and "Promotion history" | live: detector v1 and magnitude v1 |
| `/health` → drift pills | from the 2026-10-07 daily run (it needs a full UTC day of features) |
| A new model on the site | after the first retrain passes the gate (`retrain.py --month 2026-09`) and you install it |
| Android app | 2.00.00 (forced update), with the alert-speed choice |
| MLflow UI | `https://mlflow.seismicsocal.duckdns.org` (user `quakeops`) |

**What needs retraining or testing:** nothing needs retraining now. The live checkpoints are unchanged and
are the v1 champions; this work only re-evaluated them, with CIs. The server side is set up and verified
(2026-10-06). Still to run: one real `retrain.py --month 2026-09` (hours: the month's download, two
trainings, and a re-scan of the 10 replay days with the challenger detector), starting with `--dry-run` so no
alias moves, then the monthly schedule.

**False positives during training: yes, at three levels:**
- *per window* (`demo_detect.py`, every training run): false-trigger rate on test noise and on held-out
  live noise, at the checkpoint threshold **and** at the live trigger 0.6. Today that is 1.41% and 0.25%
  of windows at 0.6. These numbers are logged to MLflow and gated in G2;
- *per event* (the replay harness, the gate's G3): false confirmed events per week (3.5 today) and false
  provisional/final pushes (0), each next to a time-shifted chance baseline;
- *live* (the nightly crosscheck): precision of confirmed, pushed and tentative declarations against
  USGS.

---

## 0. What changed since the old strategy (verified in the code, 2026-10-05)

| Old finding | Status in v2 | Effect |
|---|---|---|
| F1 no tests | **Fixed**: 24 pytest tests (18 before QuakeOps) + `live_watch.py --selftest` | The gate runs them as-is |
| F3 VM is not a git checkout | Still true; deploy = `git archive` tar stream | CI deploy streams the same tar (no VM conversion) |
| F4 CPU-only torch | **Fixed**: torch 2.12.1+cu126, scripts pick CUDA | Seeds give statistically equivalent, not bit-identical, models (documented) |
| F5 seed chosen on test | **Fixed**: `demo_detect.py` selects on validation AUC | v1 champion numbers can be used as recorded |
| F6 `seismic.json` hand-written | Still true | `make_figures.py` gains a `publish` step for the numeric fields + CIs |
| F7 EEW model | **Gone** (Warn card removed, not in the live loop) | Out of scope |
| F8 builder not append-only | Still true: the M2–3 sample and noise times are random over the whole range | `build_dataset.py --append` extends the selections instead of re-drawing them (§3) |
| F9 catalogue cache ignores dates | **Fixed** (cache name carries the range) | — |
| F10 / F11 station mismatch, `SCALE` | **Fixed**: one `network.py`; checkpoints carry stations + normalizers | Drift reference exists for all 19 stations |
| F12 ruff | 7 `F` errors, all in legacy research scripts | `ruff.toml` excludes those files; CI lints everything else |
| F14 crosscheck timer pattern | Installed on the VM | Copied for the QuakeOps timer |
| new | **The acceptance test is the replay harness**, not test AUC | The promotion gate replays held-out days (§5) |
| new | `tt_correction.json` and `early_mag.json` are fitted from the dataset | They are versioned and shipped **with** the magnitude model |
| new | Detection threshold for live is calibrated (`pipeline_config.json`, 0.6), separate from the checkpoint's MCC threshold | The gate keeps the calibrated config fixed; re-calibration stays a deliberate, manual step |

## 1. Decisions (signed off 2026-10-05)

- **D1 Split = rolling chronological 70/15/15 (the locked rule, unchanged).** Each retrain re-splits the
  grown dataset by time. Champion and challenger are compared, *paired*, on the **challenger's** test
  set. That set is clean for both: the champion's train+val end before the champion's own test start,
  which is earlier than the new test start, so neither model has trained or validated on it. The
  newest month is automatically in the test set, so "no regression on new data" needs no separate
  window. No frozen `test_v1`, no embargo, no shard manifests.
  The replay test days (Oct 2–5 and Aug 18–25 2026) must stay inside the challenger's test period;
  the gate checks this and fails loudly when it stops being true (years away at ~1 month/month).
- **D2 Orchestration = Dagster on the PC.** `scripts/retrain.py` holds the stage functions with a CLI;
  `scripts/quakeops_dagster.py` (~60 lines) wraps them as assets plus a monthly schedule. Each asset
  shells out to the stage CLI, so Dagster's dependencies never touch the training code. The daily VM
  job (pull champion + drift) is a systemd timer, like crosscheck.
- **D3 MLflow on the VM** behind Caddy basic auth (`mlflow.seismicsocal.duckdns.org`), SQLite + local
  artifacts. The PC logs runs to it remotely. The daemon never talks to MLflow: a daily
  `tracking.py pull` on the VM downloads `@champion` into `data/processed/`, verifies sha256 and writes
  `data/processed/models.json`.
- **D4 "Statistically better" = non-inferior AND a reason to switch.** With AUC at 0.9998 and n≈937 for
  size, strict superiority would never pass. A challenger must beat the classical baseline (CI > 0),
  be non-inferior to the champion (paired CI), pass replay acceptance, and either be significantly
  better somewhere or be trained on a newer dataset version.
- **D5 Auto-deploy is opt-in.** The replay rule in the gate is an offline shadow run. The VM applies a
  new champion only if `QUAKEOPS_AUTO_DEPLOY=1` in its `.env`; otherwise `pull` reports "new champion
  available" (email + `/health`) and you apply it with `tracking.py pull --apply`. This keeps the
  "shadow mode after a model change" policy in your hands until you trust the gate.
- **D6 Ops email** goes to `OPS_EMAIL_TO`, defaulting to `SMTP_USER`.

## 2. Architecture after QuakeOps

```
OFFLINE (PC, RTX 4060)
  build_dataset.py [ext: --append, dataset version in dataset_meta.json]
  demo_detect.py / demo_magnitude.py [ext: bootstrap CIs, --compare CKPT (paired vs champion), MLflow run]
  replay_archive.py [ext: --det/--mag checkpoints; scan cache keyed by detector sha]
  src/eq/stats.py   [new] bootstrap / cluster bootstrap / paired bootstrap / seed t-CI
  scripts/tracking.py [new] the only mlflow import: run logging, register, aliases, register-legacy, pull
  scripts/retrain.py  [new] stages: data → train → evaluate → replay → gate → promote (state.json, resumable)
  scripts/quakeops_dagster.py [new] Dagster assets + monthly schedule over retrain.py stages
  make_figures.py [ext: publish numeric fields + CIs into app/public/seismic.json]
        │  HTTPS + basic auth
        ▼
VM (Oracle A1)
  mlflow.service [new]  127.0.0.1:5000 (SQLite, local artifacts), Caddy site mlflow.<host> with basic_auth
  seismicsocal.service (unchanged unit)
    server.py [ext: GET /api/health = models.json + drift_status.json]
      └ live_watch.py [ext: sampled detector features → data/processed/features/<date>.csv;
                        exits when models.json names a new version → supervisor respawns it]
  seismicsocal-quakeops.{service,timer} [new, crosscheck copy, 09:30 UTC]
      1) tracking.py pull   (champion → data/processed, sha256, models.json; applies only if AUTO_DEPLOY)
      2) drift_check.py     (Evidently per station vs training noise → drift_status.json, email on 2-day streak)

FRONTEND  /health route [new]: live model versions, metrics ± CI, baselines, drift pills, promotion history
          Result cards [ext]: "95% CI lo–hi" under the headline number
CI/CD     .github/workflows/ci.yml [new]: ruff + pytest (CPU) + npm build → on main: tar-stream deploy
```

## 3. Gap #4: larger datasets (monthly growth)

`build_dataset.py --append --end <first of next month>` (stage `select`):
- **M3+ magnitude events:** declustering is per (0.2° cell, calendar month), so months already selected
  never change. New months are selected the same way and appended to `sel_mag.csv`.
- **M2–3 events:** the original 4,000 are a random sample of the 2000–2026-08 candidates. Append keeps
  them and samples the new interval's candidates at the **same rate** (`len(old) / old candidates`).
- **Noise times:** kept; new uniform times in the new interval at the same density per day.
- **Hard negatives:** unchanged (fixed Sep 22 – Oct 5 2026 set, split by date).
- `fetch` is already cached per event, so only the new month downloads. `assemble` rebuilds both npz
  files and refits `tt_correction.json`.
- `dataset_meta.json` gains `version`: the first 12 hex chars of sha256 over the selection CSVs + the end
  date. Checkpoints already embed `dataset_meta.json`, so every model names its data version.

## 4. Gap #7: formal statistics (`src/eq/stats.py`, ~60 lines)

- `bootstrap_ci(metric, y, p, clusters=None)`: percentile bootstrap, 2,000 resamples. Detection
  **clusters by event**: every station window of one quake shares its origin time (`pos_time`), and
  every noise window of one random time shares `noise_time`. Resampling single windows would make
  the CI too narrow.
- `paired_bootstrap(metric, y, p_a, p_b, clusters=None)`: resample once and score both → Δ with CI.
  Used for deep vs baseline and challenger vs champion.
- `seed_ci(values)`: t-interval across seeds.
- `demo_detect.py` / `demo_magnitude.py` add these to their summaries: AUC/MCC (detect) and R²/MAE (size)
  with 95% CIs, the baseline with its CI, and the paired Δ(deep − baseline). `--compare CKPT` adds
  paired Δ(this − CKPT) on the same test set. That is the gate's input.
- **Seed-averaged magnitude R² ± CI** (the open TODO): `demo_magnitude.py --retrain --seeds 10` reports
  two separate uncertainties: seed variance (single-model R² mean, t-CI over seeds) and sampling variance
  (ensemble R², event bootstrap). It is written up in README "Magnitude: seed-averaged result".
- `make_figures.py publish` writes `deep`, `baseline`, `deep_ci`, `baseline_ci`, `n`, `model_version`,
  `generated_at` into `app/public/seismic.json`. The `desc` prose is never rewritten; the command
  prints a reminder to check it by hand.

## 5. Gaps #12 / #13 / #18: tracking, registry, promotion gate

**Tracking (`scripts/tracking.py`).** `run(experiment, name, params)` is a context manager that is a
**no-op when `MLFLOW_TRACKING_URI` is unset**, so every script runs offline and in CI exactly as now.
Each training run logs: params (architecture, lr, epochs, seeds, batch), lineage (git commit, a
`git_dirty` flag, dataset version, torch/CUDA), metrics with CI bounds, and artifacts (checkpoint,
summary JSON, figure; for size also `early_mag.json` and `tt_correction.json`).

**Registry.** Registered models `detector` and `magnitude`, one version per training run
(`runs:/<id>/model`, a plain artifact directory holding the checkpoint plus its sidecars). The
checkpoints are custom dicts, so no pyfunc flavour is needed. **Aliases** `champion` / `challenger`, not
stages. `tracking.py register-legacy` registers today's `detector.pt` and `magnitude_ensemble.pt` as
v1 `@champion`, with the metrics from `detection_demo.json` / `magnitude_demo.json` (idempotent).

**Promotion gate (`retrain.py`).** Detect and size are decided independently. Every rule is logged with
its numbers to an MLflow `quakeops-gate` run and to `data/processed/promotions.jsonl`. A skipped rule
is logged `SKIPPED (reason)`, never as a silent pass.

| Rule | Detect | Size |
|---|---|---|
| G1 beats baseline (challenger's test) | paired Δ(AUC − STA/LTA) CI low > 0 | paired Δ(R² − amp+dist) CI low > 0 |
| G2 non-inferior to champion (same test, paired) | ΔAUC ≥ −0.001 and CI high ≥ 0; ΔMCC ≥ −0.02; per-window FPR **at the live trigger (0.6)** ≤ champion + 0.005 (test noise) and + 0.0025 (held-out live noise) | ΔR² ≥ −0.01 and CI high ≥ 0; ΔMAE ≤ +0.01 |
| G3 replay acceptance (10 held-out days, `pipeline_config.json` fixed) | re-scan with the challenger: confirmed precision ≥ 0.85 and ≥ chance + 0.5; 0 false pushes; M3 recall ≥ champion's | `run` with the challenger: 0 false pushes; pushed-quake \|mag − catalogue\| ≤ 0.3; event-centric MAE ≤ champion + 0.02 |
| G4 tests | `pytest` + `live_watch.py --selftest` green | same |
| G5 lineage | clean git tree; dataset version, seeds, commit recorded; replay test days inside the challenger's test period | same |
| G6 reason to switch | newer dataset version than the champion, or a superiority CI > 0 on any G2 metric | same |

On pass: alias `champion` moves, the version is tagged (`promoted_at`, `reason`, `gate_run_id`,
`previous_champion`), a record is appended to `promotions.jsonl`, `make_figures.py publish` refreshes
`seismic.json`, and an email goes out. On reject: `@challenger` stays, tagged `rejected_reasons`.
`retrain.py rollback --model detector --to N` moves `@champion` back and tags `rolled_back_from`.

**Stages** (`data/processed/retrain/<YYYY-MM>/state.json`, re-runs skip finished stages):
`data` (build_dataset `--append` select/fetch/assemble) → `train` (both demo scripts with
`--out` challenger paths, logged + registered `@challenger`) → `evaluate` (`--compare` champion) →
`replay` (scan the test days with the challenger detector into its own cache dir, `run` + `events`
for both) → `gate` → `promote`. `--dry-run` evaluates without moving aliases.

## 6. Gap #11: orchestration (Dagster, PC)

`scripts/quakeops_dagster.py`: one asset per stage (`dataset → challengers → evaluation → replay →
gate_decision → promotion`), each running `retrain.py --month <partition> --stage <name>` in a
subprocess, plus a monthly schedule (day 3, 03:00 local, after late catalogue revisions settle).
`dagster dev -f scripts/quakeops_dagster.py` gives the UI and runs the schedule while it is up.
For when it isn't, a Windows Task Scheduler entry runs
`dagster job execute -f scripts/quakeops_dagster.py -j retrain_monthly` (with "run as soon as possible
after a missed start").

## 7. Gap #5: drift monitoring

**Features (live).** `live_watch.py` wraps `Pipeline.on_window` and, for every 15th clean window per
station (one per 30 s), appends `ts, station, prob, crest, hf_ratio` to
`data/processed/features/<date>.csv` (~55k rows/day, ~2 MB). All three are computed on
`det_prep(window)`, which is identical in training and live, so they are **scale-free and directly
comparable to the training data**. Raw amplitude is not: training noise is sensitivity-scaled and
normalized, while live is raw counts.
- `prob`: detector P(quake);
- `crest`: log10(max |x| / rms), for spikiness and glitches;
- `hf_ratio`: the fraction of 5–18 Hz power in 1–18 Hz, for spectral content (cultural noise, a
  failing sensor).
Logging is wrapped in `try/except`: it can never break detection.

**Reference.** `drift_check.py --build-reference` (PC, and again on every promotion) computes the same
features on the **training-split noise windows** of `detection.npz` (time-random)
with the champion (~280–650 per station). Noise only, because the live stream is almost all noise. The reference is
`data/processed/drift_reference.csv`, logged with the champion's run and pulled to the VM with it.

**Daily check (VM).** For each station, yesterday's rows are compared with that station's reference
using Evidently's `DataDriftPreset` (pinned version; normed Wasserstein at large n, the default).
Status: `insufficient` (< 500 rows), `ok` (0 drifted columns), `watch` (1), `drifting` (≥ 2).
Outputs: `data/processed/drift_status.json`, `drift/<date>/<station>.html`, a line in
`drift_history.jsonl`. An email goes out when a station is `drifting` on 2 consecutive days, once per
streak. Features and HTML older than 90 days are deleted.

## 8. Gap #13: CI/CD + health page

- `.github/workflows/ci.yml`. On PR and push: `ruff check` (rules `F`, `E9`; legacy research scripts
  excluded in `ruff.toml`), `pytest` on CPU torch, and `npm ci && npm run build`. On push to `main`,
  if the `VM_SSH_KEY` secret exists: build the site, then stream `git archive HEAD` + `app/dist` to
  the VM over SSH (the same tar deploy as DEPLOY.md) and restart `seismicsocal`. Models are never in
  git; they arrive through the registry.
- `server.py` `GET /api/health` returns `models.json` (versions, metrics ± CI, lineage, promotion
  history, written by `pull`) and `drift_status.json`, each `null` when absent.
- `/health` page: Detect and Size cards (version, trained date, metric ± CI vs baseline ± CI, dataset
  version, commit), 19 station drift pills (monochrome: ok = outlined, watch = gray, drifting = filled
  black, insufficient = dashed, each with a text label), and the promotion history table.

## 9. Order of work

| Step | Ships | Check |
|---|---|---|
| 1 | `stats.py` + tests; CIs in the demo scripts; `--compare` | pytest green; CIs on the real test sets |
| 2 | `tracking.py`, run logging, `register-legacy`, `pull` | Local MLflow server: v1 registered, `pull` writes `models.json` |
| 3 | `build_dataset.py --append`; replay `--det/--mag` | Append on a one-month range only adds rows |
| 4 | `retrain.py` gate + Dagster defs | `--dry-run` gate of champion vs itself passes G1–G5 and fails G6 |
| 5 | feature logging, `drift_check.py` | Synthetic 10× spiky day → `drifting` |
| 6 | `/api/health`, `/health`, CIs on cards, CI workflow, deploy units | `npm run build`, pytest, ruff |
| VM (you) | MLflow service + Caddy block, `.env` keys, QuakeOps timer, GitHub secrets | DEPLOY.md "QuakeOps" section |

## 10. Out of scope

Statewide coverage; containers; re-calibrating `pipeline_config.json` automatically (a new detector
that needs a new trigger threshold fails G3 and is calibrated by hand); rewriting `desc` prose;
widening ruff rules over the legacy scripts.
