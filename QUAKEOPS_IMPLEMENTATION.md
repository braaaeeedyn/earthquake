# QuakeOps: implementation plan and manual

> **2026-10-05 note:** written before the v2 rebuild (see `URGENT_PLAN.md` status). Already resolved
> by v2: F1 (tests exist), F5 (seed selection on val), F8/F9 (catalog cache dated; v2 builder), F10
> (live network), F11 (normalizers in the checkpoint). Re-base the remaining phases on `data/processed/v2/`
> and the replay harness before implementing.
>
> Companion to `QUAKEOPS_PLAN.md` (the *what* and *why*). This file covers the *how*: where each
> item hooks into the existing SeismicSoCal code, what to build, in what order, and how to check it.
> Nothing here has been implemented yet.
>
> **Guiding rule:** extend what already exists before adding anything. QuakeOps adds exactly
> three new technologies (**MLflow, Evidently, GitHub Actions**). Everything else reuses current
> patterns: stdlib `server.py`, the watcher supervisor, systemd timers (like `crosscheck`),
> `nearme_watch.send_email`, the JSONL/JSON report pattern, `split_chrono`, the existing baselines,
> and the existing build/train scripts.

---

## 0. Current-state findings that change the plan

I read the code before writing this. Several assumptions in `QUAKEOPS_PLAN.md` don't match the repo
as it stands. Each finding below is handled in the phase noted.

| # | Finding | Evidence | Effect on plan |
|---|---|---|---|
| F1 | **There is no pytest suite.** `tests/` holds only stale `.pyc` files from the geomagnetic code, which commit `ae8fd27` deleted. `pytest.ini` points at an empty folder. | `ls tests/` → `__pycache__` only | The plan's "existing pytest suite (pipeline + overfit-one-batch tests)" has to be **re-created**. This moves to Phase 0, because the gate (R4) and every later refactor depend on it. |
| F2 | **TransitPulse / Dagster doesn't exist** on this machine. Nothing named transit is under `coding/`. | filesystem | "Reuse Dagster" isn't possible yet. See **D1**. |
| F3 | **The VM isn't a git checkout.** `DEPLOY.md` §4 rsyncs the code with `--exclude .git`. | `DEPLOY.md:61` | "SSH deploy: git pull" needs a one-time conversion first (Phase 5.2). |
| F4 | **Training is CPU-only.** The `.venv` has `torch 2.12.1+cpu`, and no script has device code. | `torch.cuda.is_available() == False` | "Train on the RTX 4060" needs a CUDA torch install plus a `device` argument. See **D5**. |
| F5 | **The live detector was picked by *test* AUC.** `train_and_cache` keeps the best of 5 seeds by test-set AUC. The site's 0.992 is the 5-seed *mean*, not the deployed model's score. | `scripts/demo_detect.py:76` | Selecting on test is leakage, and a gate can't be built on it. Challengers must select on **val**. The v1 champion numbers have to be **re-measured from the actual `.pt`**, not copied from `CLAUDE.md` (Phase 0.3). |
| F6 | **`seismic.json` is hand-written.** No script produces it. | `git grep seismic.json` | Phase 2 adds a small writer that updates only the numeric fields. |
| F7 | **The EEW checkpoint was trained on a dataset that no longer exists.** `eew_ensemble.pt` is dated Jul 7, and `seismic_phase2a_xl.npz` has since been rebuilt for 2000–2025. | file dates, `CLAUDE.md` | EEW's training set can't be identified, so its `test_v1` score may be contaminated. Register it as `legacy` and **leave EEW out of the gate** until it's retrained. It isn't in the live loop anyway. |
| F8 | **Builders aren't append-only.** `build_multistation` shuffles the catalog (`rng.permutation`) and caps `max_events`. `build` caps per station. Re-running over a longer date range gives a *different sample*, not "old data plus new". | `src/eq/seismic.py` | Monthly growth uses **immutable monthly shards** plus a manifest (Phase 3.1), not one npz rebuilt each month. |
| F9 | **The catalog cache ignores dates** (the known gotcha). | `src/eq/quakecast.py:23` | A monthly fetch would silently reuse the 2000–2025 CSV. Fix it at the source by putting the date range in the cache filename (Phase 3.1). |
| F10 | **Detection was trained on 5 stations, but the daemon runs on 10.** PFO was dropped. SVD, RIO, MWC, DGR and BAK have no detection training windows. | `STATIONS` vs `STATIONS2` | Five stations have no training reference for drift. See Phase 4.2. |
| F11 | **`SCALE` is hard-coded** in `live_watch.py:54` and has to be edited by hand after a magnitude rebuild. | `CLAUDE.md` locked rule | Store `scale` **inside the magnitude checkpoint** so it always travels with the model. The constant stays only as a fallback for old checkpoints. |
| F12 | **ruff would reject the current style.** Scripts use `a; b` one-liners (E702) throughout. | e.g. `seismic_train_multi.py` | Start ruff on a narrow rule set (`F`, `E9`) so CI checks for bugs without reformatting the codebase. |
| F13 | **`nearme_watch.py` already has `send_email()` and `load_env()`.** | `scripts/nearme_watch.py:38,100` | Reuse them for promotion and drift emails. No new mailer. |
| F14 | **The repo already has a nightly-job pattern:** `crosscheck_events.py` writes a JSON report and runs from `seismicsocal-crosscheck.{service,timer}`. The server's supervisor already respawns the daemon when it exits. | `deploy/`, `server.py:supervise_watcher` | The drift job copies the crosscheck pattern. Model reload reuses the supervisor (Phase 1.5). |
| F15 | **Test-set sizes:** detection test starts **2021-08-29** (n≈880). Magnitude test starts **2020-06-23** (n=169). The data ends **Dec 2025**. | `wtime` quantiles | The first retrain has a **9-month backlog** (2026-01 → 2026-09). The newest month alone will hold only about 5–15 M3.5+ events. See **D3**. |

---

## 1. Decisions that need your sign-off

Each decision has a recommendation, and the manual is written to that recommendation. Before
implementing, confirm it or pick the alternative.

**D1. Orchestrator: Dagster vs. what already exists.**
Training runs on your PC (GPU). Drift runs on the VM (where the live features are). A Dagster
instance on the VM can't start GPU training on your PC, and running Dagster on the PC means
keeping its daemon alive on a desktop.
- **Recommended:** no Dagster for now. `scripts/retrain.py` is written as **separate stage
  functions** (`fetch → rebuild → train → evaluate → gate → promote`) with a CLI. Your PC runs it
  monthly through **Windows Task Scheduler**. The VM runs the daily drift job on a **systemd timer**,
  the same way `crosscheck` runs. This adds zero new infrastructure.
- **Later, once TransitPulse's Dagster exists:** add one ~60-line `ops/dagster_defs.py` that wraps
  those same stage functions as assets. That's an adapter, not a rewrite, so the skill claim is
  still available.

**D2. "Statistically better" vs. the gate rules as written.** The one-liner says "only promotes a
model that is **statistically better**". The rules in Phase 3 describe **non-inferiority** ("no
worse than −0.002, CI not entirely below 0"). These are different tests. With n=169 magnitude
events the 95% CI on ΔR² is about ±0.03–0.05, and detection AUC is already at 0.99. A strict
superiority test (CI lower bound > 0) would almost never pass, so nothing would ever be promoted.
- **Recommended:** non-inferiority on `test_v1` **and** a reason to switch. The challenger must
  either be significantly better on some metric, or be non-inferior while trained on a **newer
  dataset version**. That's the honest version of "keeps up with new data without regressing".
  Change the one-liner wording to match.

**D3. "No regression on the newest month" with a tiny n.** One month has about 5–15 magnitude
events and about 10–40 detection event windows. R² on 8 points is noise.
- **Recommended:** evaluate on a **trailing window of the newest 3 months not yet in training**.
  Use MAE (magnitude) and MCC (detection) with wide tolerances. **Skip the rule, and log that it was
  skipped,** when n is below a floor (detection < 20 event windows, magnitude < 15 events).

**D4. Chronological rule vs. a frozen `test_v1` (locked rule, so I need your approval).** Freezing
`test_v1` (2021-08 → 2025-12 detection, 2020-06 → 2025-12 magnitude) and then adding 2026+ months to
*training* means training data comes chronologically *after* the test set. For per-event tasks
(detect, size) the real leakage risk is correlated events, such as aftershock sequences that cross
the boundary. The general march of time matters less here.
- **Recommended:** allow it, with a **30-day embargo**: drop training rows within 30 days after
  `test_v1`'s last timestamp. `test_v1` stays the fixed yardstick for comparing models, and the
  rolling newest-months window (D3) is the true forward-in-time check.
- **Alternative:** never train past the `test_v1` start. Training then can't grow, which defeats
  monthly retraining.

**D5. GPU.** Install the CUDA build of torch in `.venv` and add a `device` argument to the three
`train_one` functions. Note that CUDA kernels aren't bit-deterministic, so "seed → identical model"
becomes "seed → statistically equivalent model". The lineage docs should say so.
- **Recommended:** first time one CPU retrain (5 seeds × detect + size). If it's under about 2 h,
  stay on CPU, which means fewer moving parts and exact reproducibility. Only add GPU if it's
  actually slow.

**D6. Email recipient for ops alerts.** Reuse `SUPPORT_TO` from `server.py`, or set a new
`OPS_EMAIL_TO` in `.env`. **Recommended:** `OPS_EMAIL_TO`, defaulting to `SMTP_USER`.

---

## 2. Architecture after QuakeOps

Same OFFLINE / ONLINE / FRONTEND split as `CLAUDE.md`. New pieces are marked **[new]**. Changed
existing pieces are marked **[ext]**.

```
OFFLINE (your PC)
  seismic_build*.py [ext: --out shard, dated catalog cache]
     └─> data/processed/shards/{phase1,phase2a}/<base|YYYY-MM>.npz + manifest.json     [new layout]
  src/eq/dataset.py [new]  load shards · dataset hash · frozen split (test_v1 + embargo)
  demo_detect / demo_magnitude / demo_eew [ext: val-based seed pick, --out, mlflow logging,
                                           thr/scale saved in ckpt, train fn importable]
  scripts/evaluate.py [new]  freeze test_v1 · eval ckpt w/ bootstrap CIs · seed-CI write-up · publish seismic.json
  scripts/retrain.py  [new]  monthly: fetch → rebuild → train → evaluate → gate → promote/reject
  scripts/tracking.py [new]  thin MLflow wrapper (no-op when MLFLOW_TRACKING_URI unset)
  src/eq/stats.py     [new]  bootstrap / paired bootstrap / seed t-CI (numpy + scipy)
          │  logs runs, registers versions, sets aliases (HTTPS + basic auth)
          ▼
VM (Oracle A1)
  mlflow.service [new]  127.0.0.1:5000, SQLite + local artifacts, Caddy basic_auth at mlflow.<host>
  seismicsocal.service (unchanged unit)
     server.py [ext: /api/models, /api/drift]
       └─ live_watch.py [ext: champion from registry w/ local cache fallback, scale from ckpt,
                         sampled per-window features → Parquet, hourly champion check → exit →
                         existing supervisor respawns with the new model]
  seismicsocal-drift.{service,timer} [new, copy of crosscheck units] → scripts/drift_check.py [new]
       Evidently per station → drift_status.json + HTML · email on ≥2 days drifting
  seismicsocal-crosscheck (unchanged)

FRONTEND (app/)
  /health route [new, in App.tsx's tiny router] ← /api/models + /api/drift
  Result cards [ext] show 95% CI from seismic.json

CI/CD
  .github/workflows/ci.yml [new]  ruff + pytest (CPU, synthetic fixtures) + npm run build
                                  → on main: SSH deploy (git pull, npm build, restart)
```

### New and changed files

| New | Purpose |
|---|---|
| `src/eq/dataset.py` | Shard loading/concat, `dataset_version()`, `split_frozen()` |
| `src/eq/stats.py` | `bootstrap_ci`, `paired_bootstrap`, `cluster_bootstrap_ci`, `seed_ci` |
| `scripts/tracking.py` | MLflow helpers: `lineage`, `run`, `log_checkpoint`, `register`, `set_alias`, `resolve` |
| `scripts/evaluate.py` | `freeze`, `eval`, `seeds`, `publish` subcommands |
| `scripts/retrain.py` | Monthly pipeline and gate. `rollback` subcommand |
| `scripts/drift_check.py` | `--build-reference` (PC) and daily run (VM) |
| `data/splits/test_v1.json` | Frozen split manifest (**committed**, small) |
| `PROMOTION_RULES.md` | Gate rules, the single source of truth |
| `tests/` (5 files) | Re-created suite (F1) |
| `ruff.toml`, `requirements-vm.txt` | Lint config. Codifies the VM's pip line from `DEPLOY.md` |
| `.github/workflows/ci.yml` | CI and deploy |
| `deploy/mlflow.service`, `deploy/seismicsocal-drift.{service,timer}` | VM units |
| `app/src/health.ts` | Types and loaders for `/health` |

| Changed | Change |
|---|---|
| `src/eq/quakecast.py` | Cache filename includes `start`/`end` (F9) |
| `scripts/seismic_build.py`, `seismic_build_multi.py` | `--out` into the shard dir. Write build args into the manifest |
| `scripts/demo_detect.py`, `demo_magnitude.py`, `demo_eew.py` | See Phase 1.3 |
| `scripts/seismic_train.py`, `seismic_train_multi.py`, `seismic_eew*.py` | Metric/param logging only (they're research scripts with no checkpoints) |
| `scripts/live_watch.py` | Registry load, scale from ckpt, feature logging, champion check |
| `scripts/server.py` | `/api/models`, `/api/drift` (stdlib `urllib` to MLflow's REST API, no new dependency) |
| `app/src/App.tsx`, `app/src/seismic.ts` | `/health` route. Optional CI fields on `Task` |
| `deploy/Caddyfile`, `DEPLOY.md`, `README.md`, `CLAUDE.md`, `requirements-ml.txt` | Docs and config |

---

## 3. Phase-by-phase manual

Each step lists **What**, **Where**, **How**, and **Verify**. The phases run in order. One
exception: ship the feature logging from Phase 4.1 right after Phase 1, so that two or more weeks of
live data have built up by the time Phase 4 starts (see §5).

### Phase 0: Safety net and frozen baseline (week 0)

**0.1 Re-create the test suite (F1) and ruff**
- `ruff.toml`:
  ```toml
  line-length = 120
  [lint]
  select = ["F", "E9"]          # real bugs only; widen later, never reformat wholesale
  ```
- `tests/` uses **synthetic arrays only**. CI has no `data/processed`.
  - `test_models.py`: overfit one batch. `SeisModel` on 32 random windows reaches loss < 0.1 in
    200 steps. `MultiStationModel` on 8 random events reaches MSE close to 0. This is the logic of the
    existing `--overfit` flags, moved into tests.
  - `test_split.py`: `split_chrono` is strictly chronological with no overlap. `split_frozen` (0.2)
    puts every `test_v1` key in test, applies the embargo, and never leaks a test key into train/val.
  - `test_stats.py`: a bootstrap CI covers the true value on a known distribution. A paired
    bootstrap of identical predictions gives Δ=0 with a CI that contains 0.
  - `test_live.py`: `clean_window` rejects zero-fill, stuck runs, clipping and lone spikes, and
    accepts a synthetic P-onset. `declare_graded` gives confirmed, tentative and suppressed for
    each case. Use fixture coordinates so it doesn't need the npz the `--selftest` mode reads.
  - `test_gate.py` (added in Phase 3): crafted metric dicts pass or fail each rule.
- **Verify:** `pytest` is green locally, and `ruff check scripts src tests` is clean.

**0.2 Freeze `test_v1`**
- **Where:** `src/eq/dataset.py` (logic) and `scripts/evaluate.py freeze` (CLI).
- **How:** load the current `seismic_phase1.npz` and `seismic_phase2a_xl.npz`, run the
  existing `split_chrono`, and write the test and val keys:
  ```json
  { "name": "test_v1", "created": "2026-10-..", "embargo_days": 30,
    "detection": { "source_sha256": "...", "key": "station|wtime",
                   "test": ["CI.CCC|1630224000", ...], "val": [...],
                   "test_wtime_range": [t0, t1] },
    "magnitude": { "source_sha256": "...", "key": "wtime",
                   "test": [1592870400, ...], "val": [...], "test_wtime_range": [t0, t1] } }
  ```
  Detection key = `stations[staid] | int(wtime)`. Magnitude key = `int(wtime)` (origin epoch).
  `freeze` **asserts the keys are unique** and refuses to overwrite an existing manifest.
- `split_frozen(d, manifest, task, newest_months=3)` → `(train, val, test, recent)`:
  - `test` = rows whose key is in `test_v1.test`
  - rows before the `test_v1` range that aren't in test → chronological train/val. On the
    original dataset this must reproduce **exactly** the original split (assert this in a test).
  - rows after the range: drop the 30-day embargo (D4). Rows in the newest 3 months go to `recent`
    (D3). The rest go to train.
- Commit `data/splits/test_v1.json`. `.gitignore` only ignores `data/raw/*` and
  `data/processed/*`, so `data/splits/` is tracked.
- **Verify:** n_test = 880 (detection) / 169 (magnitude). Running `split_frozen` on the original
  npz gives index sets identical to `split_chrono`.

**0.3 Re-measure the v1 champions (F5)**
- `python scripts/evaluate.py eval --task detection --ckpt data/processed/detector.pt` (and
  `--task magnitude`). This runs the **deployed** checkpoint on `test_v1` and gives point
  estimates with 95% CIs (Phase 2 provides the statistics, so in practice 0.3 lands together with
  2.1).
- The detection threshold comes from **val** (`best_threshold` already exists in `demo_detect.py`).
  Don't tune it on test.
- Save the result as `data/splits/test_v1_v1_champions.json` (committed). These numbers, not the
  `CLAUDE.md` ones, become version 1's registry metrics.
- **Expect:** the deployed detector's AUC will differ a little from 0.992, because that figure is a
  5-seed mean. Record whatever comes out.

### Phase 1: MLflow (week 1) · gap #12

**1.1 MLflow server on the VM**
- Give it **its own venv** (`/opt/mlflow/.venv`, `pip install mlflow`). MLflow pins many
  dependencies, and a separate venv keeps them away from the live daemon's torch/obspy environment.
- `deploy/mlflow.service`:
  ```ini
  [Unit]
  Description=MLflow tracking server + model registry (QuakeOps)
  After=network-online.target
  [Service]
  User=ubuntu
  WorkingDirectory=/opt/mlflow
  ExecStart=/opt/mlflow/.venv/bin/mlflow server --host 127.0.0.1 --port 5000 \
      --backend-store-uri sqlite:////opt/mlflow/mlflow.db \
      --artifacts-destination /opt/mlflow/artifacts --serve-artifacts --workers 1
  Restart=always
  [Install]
  WantedBy=multi-user.target
  ```
  It binds to localhost only and uses one worker (RAM). Newer MLflow versions validate the Host
  header: check `mlflow server --help` for `--allowed-hosts` and add
  `mlflow.seismicsocal.duckdns.org` if it's there.
- `deploy/Caddyfile`, new site block (DuckDNS resolves sub-subdomains to the same IP; confirm
  with `nslookup mlflow.seismicsocal.duckdns.org` first):
  ```
  mlflow.seismicsocal.duckdns.org {
      basic_auth {
          quakeops <hash from `caddy hash-password`>
      }
      reverse_proxy 127.0.0.1:5000
  }
  ```
  (Caddy older than 2.8 spells this `basicauth`.) Port 443 is already open, so no firewall change.
- **Verify:** the UI loads at `https://mlflow.seismicsocal.duckdns.org` after the password
  prompt. `curl 127.0.0.1:5000/health` on the VM returns OK.

**1.2 `scripts/tracking.py`: the only file that imports mlflow**
- `load_env()` is reused from `nearme_watch` to pick up `MLFLOW_TRACKING_URI`,
  `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` from `.env`. MLflow reads the last
  two natively as HTTP basic auth.
- API:
  - `lineage(dataset_kind) -> dict`: `git rev-parse HEAD`, a `git_dirty` flag, `dataset_version`
    (from `dataset.py`), torch version, device, hostname.
  - `run(experiment, run_name, params)`: a context manager. **If `MLFLOW_TRACKING_URI` is unset,
    it yields a no-op logger**, so every script still runs offline and in CI exactly as it does now.
  - `log_checkpoint(name, ckpt_path, extra_files=())`: wraps the `.pt` as an MLflow pyfunc model
    with `artifacts={"ckpt": ckpt_path}`. The pyfunc `predict` raises `NotImplementedError` with
    a comment explaining that inference stays in `live_watch`'s loaders. The registry is there to
    version and locate the file, not to serve it. The checkpoints are custom dicts of state_dicts,
    so `mlflow.pytorch` doesn't fit.
  - `register(name, model_uri, tags) -> version` and `set_alias(name, alias, version)` via
    `MlflowClient` (**aliases, not stages**).
  - `resolve(name, alias="champion", fallback=Path) -> (path, meta)`:
    1. Look up the alias, then download into `data/processed/models/<name>/v<N>/` (skip if
       already cached) and write `data/processed/models/<name>/current.json`.
    2. If MLflow can't be reached, use `current.json` (the last champion that loaded).
    3. If neither works, use the legacy `fallback` path (`detector.pt` etc.).
    4. Return `meta.source ∈ {registry, cache, legacy}` so the daemon can log where the model came
       from.
- Registered model names: `detector`, `magnitude_ensemble`, `eew_ensemble`. Experiments:
  `detect`, `size`, `warn`, `quakeops-gate`.

**1.3 Instrument the training scripts**
- `demo_detect.py`:
  - **Choose the best seed by val AUC, not test AUC** (F5, `:76`). This is a real behaviour fix
    and should be called out in the commit message.
  - Save the val-tuned `thr`, the `split` name and `dataset_version` into the checkpoint dict.
  - Add `--out PATH` (default stays `detector.pt`) so `retrain.py` can write a challenger without
    overwriting the local champion.
  - Wrap training in `tracking.run(...)`. Log params (arch=`cnn-transformer`, lr, wd, epochs, batch,
    seeds, n_train/val/test), per-seed AUC as a stepped metric, test AUC/MCC, STA/LTA AUC, and the
    artifacts (ckpt, `detection_demo.json`, figure).
- `demo_magnitude.py`:
  - Move the inline training block (`main`, roughly lines 85–110) into
    `train_ensemble(...) -> (states, summary)` so `retrain.py` can import it. `main` calls it.
  - Save **`scale`** (F11) and `dataset_version` in the checkpoint.
  - Add `--out`. Log ensemble R²/MAE, single-seed R²s, the ablation and the baseline.
- `demo_eew.py`: logging and `--out` only. It isn't gated (F7).
- All three switch `np.load(NPZ)` to `dataset.load(kind)`, which falls back to the single npz when
  no shard manifest exists. Current behaviour stays identical until Phase 3 creates shards.
- `seismic_train.py`, `seismic_train_multi.py`, `seismic_eew*.py`: wrap `main` in
  `tracking.run` and log the printed metrics. Two to four lines each.
- **Verify:** with `MLFLOW_TRACKING_URI` unset, outputs are byte-identical apart from the
  val-based seed pick. With it set, a run shows up with params, metrics, artifacts and the git
  commit.

**1.4 Register the existing checkpoints as v1**
- `python scripts/tracking.py register-legacy`, a one-shot that is idempotent (it skips if v1
  already exists). For `detector` and `magnitude_ensemble` it logs a run tagged `legacy=true`
  with `git_commit=cceaaa4` (the retrain commit) and the metrics **from Phase 0.3**, registers the
  model, and sets `@champion`.
- For `eew_ensemble`: register v1 tagged `lineage=unreproducible` with the historical metrics
  copied from `eew_demo.json`, and set `@champion` for completeness. The gate never touches it.

**1.5 `live_watch.py` loads the champion**
- `load_detector()` and `load_magnitude()` get their path from
  `tracking.resolve("detector", fallback=DETECTOR)`. Print `model detector v3 (registry)`
  at startup.
- `estimate_magnitude` takes the scale from `ck.get("scale", SCALE)`. Keep the constant as the
  fallback for the v1 checkpoint, which has no `scale` key.
- **Reload without new machinery:** inside `scan()`, once an hour, compare the alias's current
  version to the loaded one. If it changed, print a line and `sys.exit(3)`. The **existing
  `supervise_watcher`** in `server.py` respawns the daemon within about 5–15 s, and the new
  process loads the new champion. The cost is about 30 s of buffer warm-up once per promotion.
  Make the check cheap and failure-tolerant: MLflow being down means "no change".
- VM daemon venv: add `mlflow-skinny` (client only, no server) to `requirements-vm.txt`. Add
  `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` to the VM's `.env`. The unit already loads it through
  `EnvironmentFile`.
- **Verify:**
  - `live_watch.py --replay` prints source=registry and the same detection and magnitude values as
    before.
  - Stop `mlflow.service` and restart `seismicsocal`: the daemon logs source=cache and keeps working.
  - Move an alias by hand: within an hour the daemon exits, respawns, and logs the new version.
- ✅ **Checkpoint:** every live model can be traced to a run, which records the commit, dataset hash
  and seeds.

### Phase 2: Statistics write-up (week 2) · gap #7

**2.1 `src/eq/stats.py`** (numpy, plus scipy for the t-quantile; scipy is already installed as an
obspy dependency)
- `bootstrap_ci(metric, y, p, n_boot=2000, alpha=0.05, seed=0) -> (point, lo, hi)`: percentile
  bootstrap. For AUC, skip resamples that contain only one class.
- `cluster_bootstrap_ci(metric, y, p, clusters, ...)`: **use this for detection.** One quake shows
  up as several correlated windows (one per station). Resampling individual windows makes the CI
  too narrow. Cluster by event (the catalog origin, i.e. the `wtime` minus travel time, or
  `evidx`). Each noise window is its own cluster.
- `paired_bootstrap(metric, y, p_a, p_b, clusters=None) -> (delta, lo, hi, p_worse)`: resample once
  and score both models on the same rows. Use it for challenger vs. champion and for model vs.
  baseline.
- `seed_ci(values) -> (mean, lo, hi)`: a t-interval across seeds.

**2.2 `evaluate.py eval`**: runs a checkpoint (or `models:/name@alias`) on `test_v1`. It reports
the metric with a CI, the baseline with a CI (STA/LTA via the existing `sta_lta_scores`; amp+dist
via the existing `baseline_features` + `LinearRegression` fit on the train rows), and the paired Δ
against the baseline. Output is JSON, plus a log to MLflow when it's configured.

**2.3 Seed-averaged magnitude R² ± CI** (the open TODO). `evaluate.py seeds --task magnitude
--seeds 10` reuses `demo_magnitude.train_one` on the frozen split and reports **two separate
uncertainties** (don't merge them):
- *Seed variance:* single-model R² mean with a 95% t-CI over 10 seeds.
- *Sampling variance:* the ensemble's R² with an event-bootstrap 95% CI on the 169 test events.
- Also: baseline R² with its CI, the paired Δ(ensemble − baseline) CI, and the nearest-1-station
  ablation mean ± CI.
- Write it up as a short README section, "Magnitude: seed-averaged result", and as an MLflow run.
  The 5-variant fine-tune search result ("no reliable gain") goes in the same section.

**2.4 Put CIs on the site**
- `evaluate.py publish` updates only the numeric fields of `app/public/seismic.json`
  (`deep`, `baseline`, `generated_at`) and adds optional `deep_ci: [lo, hi]`,
  `baseline_ci: [lo, hi]`, `n`, `model_version`. **The `desc` prose isn't rewritten.** It prints
  a reminder to check the numbers quoted in `desc` by hand.
- `seismic.ts`: add the optional fields to `Task`. In `App.tsx` `Result`, show
  `95% CI 0.985–0.997` in the existing muted style under `stat-num`, only when present. EEW rows
  have no CI and render as they do today.
- **Verify:** `npm run build` passes and the cards show CIs. The impeccable hook raises no
  off-palette flags.

### Phase 3: Retraining and the promotion gate (week 3) · gaps #11 and #13

**3.1 Dataset versioning with monthly shards (F8, F9)**
- **Fix the catalog cache at the source:** in `load_catalog`, change the cache filename to
  `usgs_{region}_m{min_mag}_{start}_{end}.csv`. **Rename the existing CSV** by hand to the new name
  for 2000-01-01 → 2025-12-31 so it isn't refetched. This removes the `CLAUDE.md` gotcha instead of
  working around it.
- Layout:
  ```
  data/processed/shards/phase1/{base.npz, 2026-01.npz, ...}
  data/processed/shards/phase2a/{base.npz, 2026-01.npz, ...}
  data/processed/shards/<kind>/manifest.json   # [{name, sha256, n, wtime_min, wtime_max, build_args}]
  ```
  `base.npz` is the current `seismic_phase1.npz` / `seismic_phase2a_xl.npz` (copied or hard-linked).
  Find its original `build_args` in `seismic_build.log` / `seismic_multi_build.log` and record them.
- `dataset.load(kind)` concatenates the shards and asserts that `stations` / `coords` match across
  them (the same `STATIONS`/`STATIONS2` lists). `dataset_version(kind)` is the first 12 hex
  characters of sha256 over the ordered shard hashes.
- Monthly builds reuse the existing scripts with `--start/--end` for the month and `--out` pointing
  at the shard. **Keep the class balance:** pass `--max-noise` of about 2.5 × that month's event
  windows per station. The default of 200 per call would swamp a month that has about 10 events.
- `prep()` in `seismic_train_multi.py` computes `scale` over the whole array. In `retrain.py`,
  compute it on **train rows only** and store it in the checkpoint (1.3). The old v1 behaviour isn't
  touched.

**3.2 `scripts/retrain.py`**
- Each stage is a plain function. State is recorded in
  `data/processed/retrain/<YYYY-MM>/state.json`, so a re-run resumes and skips finished stages.
  ```
  python scripts/retrain.py --month 2026-09            # all stages
  python scripts/retrain.py --month 2026-09 --stage gate
  python scripts/retrain.py --backfill 2026-01:2026-09 # first run (F15): fetch each, train/gate once
  python scripts/retrain.py rollback --model detector --to 2
  ```
  1. **fetch:** calls `seismic_build.main` / `seismic_build_multi.main` (via argv, or after a small
     `main(argv=None)` refactor) for the month into the shard paths. Idempotent.
  2. **rebuild:** updates the manifest and records `dataset_version`.
  3. **train:** `demo_detect.train_and_cache(..., out=challenger.pt)` and
     `demo_magnitude.train_ensemble(...)` on `split_frozen`. Five seeds, best by **val**
     (detect) or the full ensemble (size). Logs to MLflow, registers each new version, and sets
     `@challenger`.
  4. **evaluate:** champion and challenger, both on `test_v1` **and** `recent` (D3). Each model
     uses its own stored val threshold. Paired bootstrap for every comparison.
  5. **gate:** applies `PROMOTION_RULES.md` (below) and runs `pytest -q` as a subprocess. Logs a
     `quakeops-gate` run with each rule's pass/fail and numbers.
  6. **promote / reject:** detect and size are decided **independently**.
     - *Promote:* set `@champion`. Tag the version with `promoted_at`, `promotion_reason`,
       `gate_run_id` and `previous_champion`. Append to `data/processed/promotions.jsonl` and log
       it as a gate-run artifact. Rebuild the drift reference (4.2). Email through
       `nearme_watch.send_email`. The VM daemon picks up the new model within an hour (1.5).
     - *Reject:* keep `@challenger`, tag `rejected_reasons`, and email a one-line summary.
     - Then run `evaluate.py publish` so `seismic.json` reflects the champion, and commit it.
       Committing triggers the deploy.
- **Scheduling (D1):** Windows Task Scheduler, monthly on day 3 (this lets late catalog revisions
  settle):
  ```
  schtasks /Create /TN QuakeOpsRetrain /SC MONTHLY /D 3 /ST 03:00 ^
    /TR "C:\Users\brady\desktop\coding\earthquake\.venv\Scripts\python.exe C:\Users\brady\desktop\coding\earthquake\scripts\retrain.py --month last"
  ```
  In Task Scheduler → Settings, enable "Run task as soon as possible after a scheduled start is
  missed", because the PC may be off.

**3.3 `PROMOTION_RULES.md`** (proposed numbers, per D2 and D3)

| Rule | Detection (`detector`) | Magnitude (`magnitude_ensemble`) |
|---|---|---|
| R1 beats the classical baseline on `test_v1` | paired Δ(AUC − STA/LTA AUC), CI lower bound > 0 | paired Δ(R² − amp+dist R²), CI lower bound > 0 |
| R2 non-inferior to the champion on `test_v1` | ΔAUC ≥ −0.002 **and** ΔMCC ≥ −0.02 **and** the upper CI of ΔAUC ≥ 0 | ΔR² ≥ −0.01 **and** ΔMAE ≤ +0.01 **and** the upper CI of ΔR² ≥ 0 |
| R3 no regression on `recent` (3 months) | ΔMCC ≥ −0.05. Skipped if < 20 event windows | ΔMAE ≤ +0.05. Skipped if < 15 events |
| R4 tests | `pytest` green | same |
| R5 lineage | clean git tree, `dataset_version`, seeds and commit all recorded | same |
| R6 reason to switch | a superiority CI > 0 on any metric, **or** a newer `dataset_version` than the champion | same |

Every rule is logged with its numbers. A skipped rule is logged as `SKIPPED (n=…)`, never as a
silent pass. Detection uses **cluster** bootstrap (2.1). Rollback is `retrain.py rollback`: it moves
`@champion` and tags `rolled_back_from`.

- **Verify (`test_gate.py`):** crafted inputs check each rule's pass and fail. An end-to-end dry run
  (`--dry-run`) on the backfill months logs a gate run without moving aliases.

### Phase 4: Drift monitoring (week 4) · gap #5

**4.1 Live feature logging in `live_watch.py`** (ship this early, see §5)
- In `scan()`, compute features for each station window **before** the `clean_window` gate
  (`live_watch.py:318`):
  `{ts, station, dq_ok, log_maxamp=log10(max|w|+1), log_rms=log10(rms+1), prob}`.
  `prob` is NaN when the window is DQ-rejected. `log_maxamp` uses the same definition as the training
  `logamp`.
- **Sample** to keep the volume down: log every 15th scan (once every 30 s per station), which
  works out to about 29k rows/day. Logging every scan would be about 430k rows/day.
- Buffer rows in memory and flush hourly to
  `data/processed/features/date=YYYY-MM-DD/HH.parquet` (pandas plus pyarrow; Parquet files can't
  be appended to, so the job writes one part file per hour). Wrap the logging in
  `try/except`, because **it must never break detection**.

**4.2 Reference profile**
- `drift_check.py --build-reference` (runs on the PC, and again automatically on promotion). It
  takes the same three features from the **noise windows (`ydet==0`) in the champion's train
  split**, scores `prob` with the champion, and logs `reference_features.parquet` as an **artifact
  on the champion's run**. The reference is therefore versioned with the model, and the VM pulls it
  through `tracking`, with no scp.
- **Why noise only:** the live stream is almost entirely noise. A reference with 29% events would
  show permanent fake drift.
- **The 5 stations with no detection training data (F10):** SVD, RIO, MWC, DGR, BAK. Use a
  **self-reference**, the first 14 days of their own live features, and label them that way on the
  health page. This detects *change over time*, not train/serve skew, and the UI says so honestly.
  Don't pool amplitude across stations, because instrument gains differ.

**4.3 `scripts/drift_check.py`** (daily, on the VM)
- For each station, compare yesterday's Parquet to the reference with Evidently: a `Report` with
  `DataDriftPreset` on `log_maxamp`, `log_rms` and `prob`. **Pin the Evidently version** in
  `requirements-vm.txt` and check the import paths against that version's docs. The API changed
  substantially around 0.7 (`from evidently import Report`,
  `from evidently.presets import DataDriftPreset`, `report.run(...)` returns a snapshot with
  `save_html` / `dict`).
- With n in the thousands, Evidently's default numeric test switches to normed Wasserstein distance
  (threshold 0.1) instead of KS, which avoids "everything is significant at large n". Keep that
  default and record it.
- Status per station:
  - `insufficient` if there are < 500 rows (station offline)
  - `ok` if no column drifted
  - `watch` if 1 column drifted, or the DQ-reject rate is more than 2× its 30-day median
  - `drifting` if ≥ 2 columns drifted
- Outputs:
  - `data/processed/drift_status.json` (`{date, stations: {CCC: {status, drifted: [..], n, reference: "train"|"self"}}}`)
  - HTML in `data/processed/drift/<date>/<station>.html`
  - one appended line in `drift_history.jsonl`
- Email (`nearme_watch.send_email`) when any station is `drifting` for **2 or more consecutive
  days**, sent once per streak.
- **Retention:** delete feature dirs and drift HTML older than 90 days. Model artifacts are
  0.5–2 MB each, so registry cleanup can wait. If it's ever needed, keep the last 5 non-aliased
  versions.
- `deploy/seismicsocal-drift.{service,timer}`: copies of the crosscheck units, with `ExecStart`
  set to `drift_check.py` and the timer at `09:30 UTC` (after crosscheck).
- **Verify:**
  - Run it by hand on one day of logged features. HTML and JSON appear.
  - Feed it a synthetic day with the amplitudes scaled 10×: the station shows `drifting`.
  - Two such days produce exactly one email.

### Phase 5: CI/CD and the health page (week 5) · gap #13

**5.1 `.github/workflows/ci.yml`** (triggered on `pull_request` and on `push` to `main`)
- Job `python` (ubuntu-latest, **the same Python minor version as the VM**; check with
  `python3 --version` there):
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then
    `numpy pandas scipy scikit-learn obspy pytest ruff`
  - `ruff check scripts src tests`
  - `pytest`
- Job `app`: Node 20, `npm ci && npm run build` in `app/`.
- Job `deploy`: `needs: [python, app]`, and runs only on a push to `main` (below).

**5.2 SSH deploy**
- **One-time VM conversion (F3).** Take a backup first:
  ```bash
  tar czf ~/seismicsocal-pre-git.tgz --exclude=.venv --exclude=node_modules -C /opt seismicsocal
  cd /opt/seismicsocal && git init && git remote add origin https://github.com/braaaeeedyn/earthquake.git
  git fetch origin && git checkout -f -B main origin/main   # tracked files now match main; gitignored data/.env/.venv untouched
  git status                                                # should show nothing tracked as modified
  ```
  If the repo is private, add a read-only deploy key on the VM first.
- Secrets in GitHub: `VM_HOST`, `VM_USER`, and `VM_SSH_KEY` (a **dedicated** ed25519 key, added
  to `~ubuntu/.ssh/authorized_keys`).
- Deploy step (`appleboy/ssh-action`, pinned to a release tag):
  ```bash
  set -e
  cd /opt/seismicsocal
  git pull --ff-only                 # fails loudly on divergence, never discards VM changes
  .venv/bin/pip install -q -r requirements-vm.txt
  cd app && npm ci && npm run build && cd ..
  sudo systemctl restart seismicsocal
  sleep 5 && curl -fsS 127.0.0.1:8000/api/status
  ```
  Check `sudo -n true` on the VM. The Oracle Ubuntu image normally gives `ubuntu` passwordless
  sudo. Models aren't in git. The daemon pulls `@champion` from the registry (1.5). The APK in
  `app/public/` is gitignored, so `git pull` leaves it alone.

**5.3 `server.py`: `/api/models` and `/api/drift`** (stdlib only)
- `/api/models` uses `urllib` against MLflow's REST API at `MLFLOW_TRACKING_URI` (localhost, so no
  auth):
  - `GET /api/2.0/mlflow/registered-models/alias?name=detector&alias=champion` gives the version,
    `run_id` and tags.
  - `GET /api/2.0/mlflow/runs/get?run_id=…` gives the metrics and their CI bounds, the baseline, and
    the `dataset_version` / `git_commit` tags.
  - `GET /api/2.0/mlflow/model-versions/search?filter=name='detector'` gives the promotion history
    (versions tagged `promoted_at`).
  - The response is cached in memory for 60 s. If MLflow can't be reached it returns
    `{"available": false}`, and the page says "registry unavailable" instead of erroring.
- `/api/drift` returns `drift_status.json`, or `{"available": false}`.
- Add both to the route list in `CLAUDE.md`.

**5.4 The `/health` page**
- In `App.tsx`, extend `ROUTE_OF` and `PAGE_TITLES` with `'/health' → 'health'`, and add a footer
  link next to Privacy. Caddy's `try_files` already handles a hard refresh on client routes.
- `app/src/health.ts`: types plus `loadModels()` and `loadDrift()`, in the same shape as
  `seismic.ts`.
- `Health` component, three sections:
  1. **Live models:** one card each for Detect and Size, showing version, trained date, test metric
     ± CI, baseline ± CI, `dataset_version`, short commit, and source (registry/cache).
  2. **Drift:** 10 station pills. **Monochrome per `DESIGN.md`:** `ok` = outlined, `watch` = gray
     fill, `drifting` = filled black, `insufficient` = dashed. Every pill has a **text label**, so
     colour isn't the only signal. Self-reference stations are marked.
  3. **History:** a table of promotions (version, date, metrics, reason).
- Run the impeccable design check. Light mode only.

**5.5 Docs**
- `README.md`: add "How a model gets to production" (train → registry → gate → alias →
  daemon reload → drift watch). Also add the magnitude CI section (2.3).
- `CLAUDE.md`: update the run commands. Add locked rules for "`test_v1` is frozen" and "promotion
  only via the gate", and replace the `SCALE` rule with "scale lives in the checkpoint".
- `DEPLOY.md`: add the MLflow service, drift timer, git-based deploy and `requirements-vm.txt`.

---

## 4. Configuration and secrets

| Variable | Where | Used by |
|---|---|---|
| `MLFLOW_TRACKING_URI` | PC `.env` = `https://mlflow.seismicsocal.duckdns.org`. VM `.env` = `http://127.0.0.1:5000` | `tracking.py`, `server.py` |
| `MLFLOW_TRACKING_USERNAME` / `_PASSWORD` | PC `.env` only | MLflow client (Caddy basic auth) |
| `OPS_EMAIL_TO` (D6) | both `.env` files | promotion and drift emails |
| `SMTP_*` | existing | reused unchanged |
| `VM_HOST`, `VM_USER`, `VM_SSH_KEY` | GitHub secrets | deploy |

`.env` is already gitignored and already loaded by the systemd unit (`EnvironmentFile`) and by
`nearme_watch.load_env`.

## 5. Order of work and checkpoints

| Step | Ships | Checkpoint |
|---|---|---|
| 0 | tests, ruff, `test_v1.json`, v1 metrics re-measured | `pytest` green. Split reproduces the original exactly |
| 1 | MLflow server, `tracking.py`, instrumented scripts, v1 registered, daemon loads `@champion` | Live site models can be traced to runs. Fallback tested with MLflow stopped |
| **1b** | **feature logging from 4.1** (deploy right away) | Parquet parts appear hourly on the VM. Two or more weeks of data build up before Phase 4 |
| 2 | `stats.py`, `evaluate.py`, seed-CI write-up, CIs on the site | Cards show CIs. README section published |
| 3 | shards, catalog-cache fix, `retrain.py`, `PROMOTION_RULES.md`, scheduler | Backfill 2026-01 → 09 runs end to end. The gate decision is logged and emailed |
| 4 | reference profile, `drift_check.py`, drift timer | `drift_status.json` updates daily. Synthetic drift raises an alert |
| 5 | CI/CD, `/api/models`, `/api/drift`, `/health`, docs | A PR runs CI. A merge to main deploys. `/health` shows live versions, CIs, drift and history |

## 6. Explicitly out of scope

These are unchanged from the plan's "not used" list, plus a few items from this review:
- Wiring EEW into the live loop.
- Statewide coverage.
- Restoring PFO.
- Containerization.
- Dagster (until D1's "later" applies).
- Rewriting `desc` prose automatically.
- Reformatting existing code to satisfy a wider ruff rule set.

**Pre-existing issues noticed and left alone, except where QuakeOps requires the change:**
- **Fixed, because the gate depends on it:** the test-AUC seed selection (F5).
- **Left alone:**
  - `CLAUDE.md` describes Detect as a "5-seed ensemble", but the deployed detector is a single
    model.
  - `seismic_eew_ensemble.py` still defaults to the old `seismic_phase2a.npz`.
