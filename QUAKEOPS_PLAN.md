# QuakeOps: project plan

> **Type:** an **upgrade to SeismicSoCal** (same repo: `github.com/braaaeeedyn/earthquake`, same live site
> `seismicsocal.duckdns.org`, same Oracle VM). Not a new project.
> **One line:** turn SeismicSoCal's models into a production ML system that maintains itself. Every training run
> is tracked, models are versioned in a registry, retraining is automatic and **only promotes a model that is
> statistically better**, and the live stream is watched for drift.
> **Effort:** ~4–5 weeks part-time (models, data and deployment already exist). **Cost target:** $0.

Do this **after TransitPulse**. It **reuses** Dagster, GitHub Actions and the statistics methods from there and
introduces only three new things: **MLflow, Evidently and gated model promotion.**
See [the overlap plan](#overlap-plan-with-transitpulse).

---

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
3. A monthly **Dagster** job (reused from TransitPulse) fetches new SCEDC events, rebuilds the dataset, trains a challenger, evaluates it and runs the **promotion gate**.
4. If the gate passes, the challenger becomes `champion`, and the live daemon reloads it on its next restart.
5. A daily **Evidently** job compares the live stream with the training data and updates the drift status.
6. **GitHub Actions** (reused) tests every change and deploys code to the VM.

---

## 2. Gaps this project closes

| # | Gap | How QuakeOps covers it | New or reused |
|---|---|---|---|
| 4 | Larger datasets | Existing ~11 GB SCEDC waveform cache + monthly growth | existing |
| 5 | Monitoring (**model drift part**) | Evidently drift reports on live features and detection scores | **new** |
| 7 | Formal statistics | Bootstrap CIs on AUC / R², a paired comparison inside the promotion gate, the seed-averaged CI write-up | reused |
| 11 | Orchestration | Dagster retraining and drift jobs | reused |
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
| Orchestration | **Dagster OSS** | **Reused** from TransitPulse, same VM, separate code location. |
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
- [ ] Freeze the current test split as **`test_v1`**: save its event IDs to a versioned file. Every future model is compared on this same set, so the comparison is fair.
- [ ] Record the current models' metrics (detection AUC 0.992 / MCC 0.930, magnitude R² 0.840, EEW alert MCC 0.760) as the **v1 champions**.

### Phase 1: MLflow (week 1) · gap #12
- [ ] Install the MLflow server on the VM:
  - systemd unit `mlflow.service`
  - `--backend-store-uri sqlite:////opt/mlflow/mlflow.db --artifacts-destination /opt/mlflow/artifacts`
  - Caddy reverse proxy with `basic_auth`
- [ ] Add `mlflow.start_run()` logging to `seismic_train.py`, `seismic_train_multi.py`, `seismic_eew*.py` and the `demo_*.py` scripts:
  - **params:** architecture, learning rate, epochs, seed, dataset hash, git commit
  - **metrics:** AUC, MCC, R², MAE, each against its baseline
  - **artifacts:** `.pt` files, figures, `seismic.json`
- [ ] Register the existing checkpoints as version 1 of `detector`, `magnitude_ensemble` and `eew_ensemble`; set alias `champion`.
- [ ] Change `live_watch.py` to load the **`champion`** model from the registry, falling back to the local `.pt` if MLflow is down. Keep the `SCALE` constant tied to the dataset version.
- ✅ **Checkpoint:** every model on the live site is traceable to its run.

### Phase 2: statistics write-up (week 2) · gap #7
- [ ] Bootstrap 95% CIs for detection AUC/MCC and magnitude R²/MAE on `test_v1`.
- [ ] Finish the **seed-averaged magnitude R² ± CI** (open TODO in the earthquake repo's `CLAUDE.md`).
- [ ] Add CIs to `app/public/seismic.json` so the site shows them.

### Phase 3: retraining + promotion gate (week 3) · gaps #11 #13
- [ ] Dagster job `retrain_monthly`, running on your PC or triggered from the VM, with these steps:
  1. **fetch** new events/waveforms (existing `seismic_build*.py`, date range = last month)
  2. **rebuild** the dataset and record its hash
  3. **train** a 5-seed challenger and log it to MLflow
  4. **evaluate** on `test_v1` plus the newest held-out month
  5. **gate**
  6. **promote or reject**
- [ ] **Promotion gate rules** (write them in the repo as `PROMOTION_RULES.md`):
  - beats each classical baseline (STA/LTA, amplitude+distance) on `test_v1`
  - no worse than the champion beyond a small tolerance (e.g. AUC −0.002); the 95% bootstrap CI of (challenger − champion) must not be entirely below 0
  - no regression on the newest month
  - passes all pytest checks
- [ ] On pass: set alias `champion` to the new version, write a `promotions.jsonl` record with metrics and the reason, and send an email.

### Phase 4: drift monitoring (week 4) · gap #5
- [ ] Small change to `live_watch.py`: append per-window features (station, max amplitude, RMS, detector probability, timestamp) to a rolling **daily Parquet** file.
- [ ] Save a **reference profile** of the same features from the training set.
- [ ] Daily Dagster job: Evidently `DataDriftPreset` per station (current day vs. reference). Write an HTML report and a `drift_status.json` (OK / watch / drifting).
- [ ] Email alert on "drifting" for ≥ 2 consecutive days.

### Phase 5: CI/CD + health page (week 5) · gap #13
- [ ] **GitHub Actions**:
  - On PR: ruff, pytest (CPU, small fixtures), `npm run build` for the app
  - On merge to `main`: **SSH deploy** to the VM (pull, build the app, `systemctl restart seismicsocal`)
  - Models are **not** in git; the VM pulls the `champion` from MLflow
- [ ] React **`/health` page**: champion versions, metrics with CIs, baseline comparison, drift badges, promotion history (reads `drift_status.json` and a small `/api/models` endpoint added to `server.py`).
- [ ] README section: "How a model gets to production".

---

## 5. Skills you will be able to claim
**MLOps:** MLflow tracking + Model Registry (aliases), automated retraining, statistically gated model promotion, drift monitoring with Evidently, reproducible lineage (code + data + seed → model).
**Production ML:** a live inference daemon serving the registered champion, CI/CD to a VM, alerting.
**Statistics:** bootstrap CIs, paired model comparison.

Example résumé bullet (fill in your real numbers):
> Productionized SeismicSoCal's earthquake models with MLflow tracking and a model registry, monthly Dagster
> retraining with a statistically gated champion/challenger promotion (bootstrap CIs on a frozen test set), and
> Evidently drift monitoring over 10 live SeedLink stations; CI/CD via GitHub Actions.

---

## 6. Overlap plan with TransitPulse
| Skill / tool | Learned in TransitPulse | Used here |
|---|:-:|:-:|
| Dagster OSS (Oracle VM) | ✅ | ✅ reused |
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
