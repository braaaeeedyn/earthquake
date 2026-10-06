# QuakeOps: project plan

> **Type:** an **upgrade to SeismicSoCal** (same repo: `github.com/braaaeeedyn/earthquake`, same live site
> `seismicsocal.duckdns.org`, same Oracle VM). Not a new project.
> **One line:** turn SeismicSoCal's models into a production ML system that maintains itself. Every training run
> is tracked, models are versioned in a registry, retraining is automatic and **only promotes a model that is
> statistically better**, and the live stream is watched for drift.
> **Effort:** ~4–5 weeks part-time (models, data and deployment already exist). **Cost target:** $0.

Originally planned for after TransitPulse, reusing its Dagster. TransitPulse doesn't exist yet, so QuakeOps sets up
Dagster itself, on the PC (decision 2026-10-05). **How it is built on the current v2 system: `QUAKEOPS_IMPLEMENTATION.md`.**
See [the overlap plan](#overlap-plan-with-transitpulse).

---

> **Status 2026-10-06:** everything checked below is built and live (MLflow on the VM, v1 champions registered,
> daily pull + drift job, `/health`, CI deploys on `main`). Not yet run: the first real monthly retrain. Summary and "when will I see it": top of `QUAKEOPS_IMPLEMENTATION.md`.

## 1. What it does

### What a visitor sees (new)
| Feature | Description |
|---|---|
| **Model health page** (`/health` in the React console) | Which model version is live for Detect and Size, when it was trained, its test metrics **with 95% CIs**, how it compares to the STA/LTA and amplitude+distance baselines, and a **drift status** badge (OK / watch / drifting) per station. |
| **Model history** | A table of every promoted model: version, date, metrics, and why it was promoted. |
| **Magnitude write-up with CIs** | The overdue "seed-averaged magnitude R² with a CI" result, published properly. |

Everything else (live detection, sizing, push alerts, biggest-quakes browser, Android app) keeps working as it does now.

### What runs behind the scenes (new)
1. **MLflow** records every training run: parameters, metrics, seeds, dataset version, model artifact.
2. Models live in the **MLflow Model Registry** with aliases **`champion`** (live) and **`challenger`** (candidate).
3. A monthly **Dagster** job (on the PC) fetches new SCEDC events, rebuilds the dataset, trains a challenger, evaluates it, **replays held-out days** with it and runs the **promotion gate**.
4. If the gate passes, the challenger becomes `champion`; the VM pulls it daily (applied automatically only when `QUAKEOPS_AUTO_DEPLOY=1`).
5. A daily **Evidently** job compares the live stream with the training data and updates the drift status.
6. **GitHub Actions** (reused) tests every change and deploys code to the VM.

---

## 2. Gaps this project closes

| # | Gap | How QuakeOps covers it | New or reused |
|---|---|---|---|
| 4 | Larger datasets | Existing ~11 GB SCEDC waveform cache + monthly growth | existing |
| 5 | Monitoring (**model drift part**) | Evidently drift reports on live features and detection scores | **new** |
| 7 | Formal statistics | Bootstrap CIs on AUC / R², a paired comparison inside the promotion gate, the seed-averaged CI write-up | reused |
| 11 | Orchestration | Dagster retraining (PC) + systemd timer for the daily drift/pull job | **new** (TransitPulse not built yet) |
| 12 | Experiment tracking + registry | MLflow tracking server + Model Registry with aliases | **new** |
| 13 | CI/CD | GitHub Actions tests, SSH deploy, plus **model-promotion gates** | reused + new concept |
| 18 | Reproducible | Every model traceable to code commit + data version + seed | existing, extended |

**Not covered, on purpose:**
- #1 cloud / Terraform, #2 SQL, #3 load testing, #6 A/B, #8 business metrics, #9 BI, #10 warehouse/dbt, #19 agents, #21 causal (all owned by TransitPulse)
- #16 fine-tuning and #17 Spark (owned by TransitPulse)
- #14 Kubernetes, #15 distributed training, #20 novel methods (too senior / not needed)

---

## 3. Stack: what is and isn't used

### ✅ Used

| Area | Tool | Notes |
|---|---|---|
| Existing | PyTorch, ObsPy, NumPy, scikit-learn, React + Vite, Capacitor, FCM, Caddy, systemd, Oracle A1 VM | unchanged |
| Experiment tracking | **MLflow OSS** tracking server | Runs on the **Oracle VM** (always free) with a **SQLite** backend store and **local-disk** artifact store, behind **Caddy** with HTTP basic auth at e.g. `mlflow.seismicsocal.duckdns.org`. **Not** Databricks / managed MLflow. |
| Model registry | **MLflow Model Registry** with **aliases** (`champion`, `challenger`) | Aliases are the current way to do this; avoid the old "stages" API. |
| Orchestration | **Dagster OSS** | Runs on the **PC** (where the GPU is); assets shell out to `retrain.py` stages. The daily VM job is a systemd timer. |
| Training compute | **Your local RTX 4060** | Free. Training runs locally and logs to the remote MLflow server. **No cloud GPUs.** |
| Drift monitoring | **Evidently** (open-source library) | Generates HTML/JSON reports; the JSON summary feeds the health page. **Not** Evidently Cloud. |
| Statistics | **numpy** bootstrap; `scikit-learn` metrics | Reused methods from TransitPulse. |
| CI/CD | **GitHub Actions** | **Reused.** New: SSH deploy to the VM using `appleboy/ssh-action`, secrets in GitHub. |
| Alerts | Existing **SMTP** email (already in `.env`) | Emails you when drift is "drifting" or a promotion happens. |
| Tests | Existing **pytest** suite (pipeline + overfit-one-batch tests) + **ruff** | |

### ❌ Not used, and why
| Not used | Why |
|---|---|
| Terraform / IaC | The VM is already hand-built and documented in `DEPLOY.md`. IaC is learned in TransitPulse. |
| GCP, BigQuery, dbt, Looker Studio | Belong to TransitPulse. |
| An LLM agent / Langfuse | Belongs to TransitPulse. *(Optional later: a tiny "explain this event" endpoint, but not part of this plan.)* |
| Kubernetes, Docker Swarm | The single VM with systemd is enough. Don't containerize the daemon just to learn it. |
| Cloud GPUs / SageMaker / Vertex AI | Your local GPU is free and fast enough for these models. |
| Weights & Biases | MLflow covers the same skill, self-hosted and free. |
| Feature store | Not needed at this scale. |
| Airflow / Prefect | Dagster is already learned; one orchestrator only. |

---

## 4. Build plan

### Phase 0: prep (week 0)
- [x] ~~Freeze a `test_v1`~~ → **rolling chronological split** (locked rule unchanged): champion and challenger are compared, paired, on the challenger's test set, which neither has seen.
- [x] Record the current v2 models' metrics (detection AUC 0.9998 / MCC 0.886, magnitude R² 0.951 / MAE 0.098), with bootstrap CIs, as the **v1 champions**.

### Phase 1: MLflow (week 1) · gap #12
- [x] Install the MLflow server on the VM:
  - systemd unit `mlflow.service`
  - `--backend-store-uri sqlite:////opt/mlflow/mlflow.db --artifacts-destination /opt/mlflow/artifacts`
  - Caddy reverse proxy with `basic_auth`
- [x] Add MLflow run logging (via `scripts/tracking.py`) to `demo_detect.py` and `demo_magnitude.py` (the only scripts that produce live checkpoints):
  - **params:** architecture, learning rate, epochs, seed, dataset hash, git commit
  - **metrics:** AUC, MCC, R², MAE, each against its baseline
  - **artifacts:** `.pt` files, figures, `seismic.json`
- [x] Register the existing checkpoints as version 1 of `detector` and `magnitude`; set alias `champion`. (EEW is gone; `SCALE` is gone: checkpoints carry their normalizers.)
- [x] A daily `tracking.py pull` on the VM downloads `@champion` (sha256-verified) and writes `models.json`; the daemon restarts itself when the version changes.
- ✅ **Checkpoint:** every model on the live site is traceable to its run.

### Phase 2: statistics write-up (week 2) · gap #7
- [x] Bootstrap 95% CIs (event-clustered for detection) for detection AUC/MCC and magnitude R²/MAE on the test split.
- [x] Finish the **seed-averaged magnitude R² ± CI** (open TODO in the earthquake repo's `CLAUDE.md`).
- [x] Add CIs to `app/public/seismic.json` so the site shows them.

### Phase 3: retraining + promotion gate (week 3) · gaps #11 #13
- [x] Dagster job `retrain_monthly`, running on your PC or triggered from the VM, with these steps:
  1. **data:** `build_dataset.py --append` (new month only; earlier selections kept) and record the dataset version
  2. **train** a 5-seed challenger and log it to MLflow
  3. **evaluate** paired against the champion on the challenger's test split (includes the newest data)
  4. **replay** the 10 held-out days with the challenger (the acceptance test)
  5. **gate**
  6. **promote or reject**
- [x] **Promotion gate rules** (G1–G6, in `HOW_IT_WORKS.md` §12 and `retrain.py`):
  - beats each classical baseline (STA/LTA, amplitude+distance), paired CI above 0
  - non-inferior to the champion on the same test set (paired bootstrap CI)
  - passes the replay acceptance test (precision vs chance, 0 false pushes, magnitudes vs catalogue)
  - passes pytest + selftest, clean lineage, and has a reason to switch (newer data or a significant gain)
- [x] On pass: set alias `champion` to the new version, write a `promotions.jsonl` record with metrics and the reason, and send an email.

### Phase 4: drift monitoring (week 4) · gap #5
- [x] Small change to `live_watch.py`: append scale-free per-window features (detector probability, crest factor, high-frequency power ratio, all on `det_prep` input) to a daily CSV. Raw amplitude isn't comparable to training (live = counts, training = normalized).
- [x] Save a **reference profile** of the same features from the training set.
- [x] Daily VM systemd timer: Evidently `DataDriftPreset` per station (current day vs. reference). Write an HTML report and a `drift_status.json` (OK / watch / drifting).
- [x] Email alert on "drifting" for ≥ 2 consecutive days.

### Phase 5: CI/CD + health page (week 5) · gap #13
- [x] **GitHub Actions**:
  - On PR: ruff, pytest (CPU, small fixtures), `npm run build` for the app
  - On merge to `main`: **SSH deploy** to the VM (stream `git archive` + `app/dist`, `systemctl restart seismicsocal`)
  - Models are **not** in git; the VM pulls the `champion` from MLflow
- [x] React **`/health` page**: champion versions, metrics with CIs, baseline comparison, drift badges, promotion history (reads one `/api/health` endpoint added to `server.py`).
- [x] README section: "How a model gets to production".

---

## 5. Skills you will be able to claim
**MLOps:** MLflow tracking + Model Registry (aliases), automated retraining, statistically gated model promotion, drift monitoring with Evidently, reproducible lineage (code + data + seed → model).
**Production ML:** a live inference daemon serving the registered champion, CI/CD to a VM, alerting.
**Statistics:** bootstrap CIs, paired model comparison.

Example résumé bullet (fill in your real numbers):
> Productionized SeismicSoCal's earthquake models with MLflow tracking and a model registry, monthly Dagster
> retraining with a statistically gated champion/challenger promotion (bootstrap CIs on a frozen test set), and
> Evidently drift monitoring over 19 live SeedLink stations; CI/CD via GitHub Actions.

---

## 6. Overlap plan with TransitPulse
| Skill / tool | Learned in TransitPulse | Used here |
|---|:-:|:-:|
| Dagster OSS | ✅ (planned) | ✅ set up here first, on the PC |
| GitHub Actions | ✅ | ✅ reused (+ SSH deploy, promotion gates) |
| Bootstrap CIs / paired tests | ✅ | ✅ reused |
| MLflow + Model Registry | ❌ | ✅ **new** |
| Evidently drift monitoring | ❌ | ✅ **new** |
| Champion/challenger gating | ❌ | ✅ **new** |
| GCP / Terraform / BigQuery / dbt / agent / Langfuse | ✅ | ❌ not used |

## 7. Free-resource notes
- Oracle Always-Free A1 allows up to 4 OCPU / 24 GB RAM / 200 GB block storage in total. The current VM uses 2 OCPU / 12 GB; MLflow + Dagster fit alongside the daemon. Watch disk usage (artifacts + Parquet). Add a cleanup job keeping the last N model versions and 90 days of features.
- No paid services are required.

## 8. Adding it to the portfolio
See `docs/PORTFOLIO_CONTEXT.md` → "Adding QuakeOps". It goes **into the existing SeismicSoCal page**, not a new page.
