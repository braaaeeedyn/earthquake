# How SeismicSoCal works

The current system: data, training, the live pipeline, alerts, the website and app, and deployment.
Every number below comes from the current models, datasets and replay runs.

SeismicSoCal detects earthquakes in Southern California from a live seismic stream. It locates and sizes
each one, and pushes alerts to people who follow nearby sensors.

- It is **rapid detection**: alerts go out tens of seconds after a quake begins.
- It is **not** an official early-warning system.
- It does **not** predict earthquakes.

---

## 1. The system at a glance

```
OFFLINE (Windows PC, RTX 4060)                          ONLINE (Oracle A1 VM, Ubuntu, systemd + Caddy)
──────────────────────────────                          ─────────────────────────────────────────────
USGS catalogue (M1+)  ─┐                                IRIS SeedLink (19 stations, real time)
SCEDC waveform archive ─┤                                         │
                        ▼                                         ▼
 build_dataset.py  ──► detection.npz / magnitude.npz     live_watch.py ── pipeline.py (the engine)
                        │                                  detect → pick → locate → quick check
 demo_detect.py ──► detector.pt                            → full size → decide → push (FCM) + log
 demo_magnitude.py ──► magnitude_ensemble.pt                      │
 fit_early_magnitude.py ──► early_mag.json                        ▼
 replay_archive.py ──► pipeline_config.json             server.py (API)  ◄── website + Android app
   (+ scores, acceptance)                                 /api/stations, /api/ca, /api/status, …
 make_figures.py ──► evidence figures, seismic.json      Caddy: HTTPS, static site, /api proxy
```

One engine (`src/eq/pipeline.py`) runs both live and in the replay harness. What is measured offline is
the same code that runs on the server.

**QuakeOps** (§12) wraps this in an MLOps loop: training runs are tracked in MLflow, models are versioned
in its registry (`champion` / `challenger`), a monthly Dagster job on the PC grows the dataset, retrains,
and promotes a challenger only through a statistical + replay gate, and a daily job on the VM records the
champion (installing it only when allowed) and checks the live stream for drift. **Status (2026-10-05):
the code is in the repo and tested on the PC; none of it runs on the VM or the live site yet** (§12.0).

---

## 2. The station network — `src/eq/network.py`

This file defines the network, and it is the only place that does. The dataset builder, the live
daemon, the API, the scorer and the replay harness all import it.

| Region | Stations |
|---|---|
| LA basin | PASC (sensor location code `10`), BFS |
| Inland Empire | SVD, DGR |
| San Diego / Imperial | BAR, IKP, SWS, BEL |
| Mojave | GSC, GMR, EDW2 |
| Ridgecrest | LRL, MPM |
| Kern | ISA, ARV |
| Central coast / offshore | SMM, MPP, SNCC, CIA |

How the stations were chosen (`scripts/select_network.py` reproduces it):

- **Streams in real time:** the station sends HH channels on the public relay
  `rtserve.iris.washington.edu`. `build_dataset.py --stage check` refuses to build otherwise.
- **Enough history:** HH data back to 2010 or earlier, so there are years to train on.
- **Quiet:** median 2–8 Hz ground noise ≤ 1e-7 m/s, measured at four times of day.
- **Spread out:** at least about 44 km apart, so neighbours can't "confirm" each other on shared
  local noise.
- **Coverage:** chosen greedily to cover SoCal's M3+ seismic areas and major cities.

PASC is pinned to location code `10` because it streams two sensors.

---

## 3. Data

### 3.1 Catalogue — `src/eq/catalog.py`

- **What:** the USGS FDSN catalogue for 2000-01-01 → 2026-10-05, M ≥ 1.0, in a box around SoCal
  (31.5–37.5°N, 122–113.5°W). That's about 424,000 events with id, time, location, depth and
  magnitude.
- **How it's fetched:** in monthly chunks. If a chunk exceeds the service's 20,000-row cap (the
  service answers HTTP 400), the interval is split in half and retried.
- **Caching:** each chunk is cached. The final CSV's name includes the date range, so a different
  range can never reuse a stale cache.

### 3.2 What goes into the datasets — `scripts/build_dataset.py`

Data runs from 2000-01-01 to 2026-09-01. September 2026 is held back for the live-noise sets below.

- **Magnitude events:**
  - All catalogue M ≥ 3.0 quakes in 32.0–36.8°N, 121.5–114.0°W with at least 3 network stations within
    200 km.
  - Lightly declustered: at most 25 per 0.2° cell per month, keeping the largest, so one aftershock
    sequence can't dominate.
  - Plus the M2–3 events below.
  - Total: **6,243 events, M2.0–7.1**, with 37,474 station records.
- **M2–3 events:** a random sample of 4,000 catalogue M2.0–3.0 quakes with a station within 100 km.
  Only those stations are used. These events feed both detection and magnitude.
- **Noise:** 900 random times across all hours.
  - For each station, the 45 s window is kept only if no catalogued quake was nearby: no M ≥ 1 within
    ±150 s and 150 km, and no M ≥ 3 within 500 km.
- **Hard negatives:**
  - These are the previous live system's own false declarations (Sep 22 – Oct 5, 2026): 2,131
    declarations, of which 147 matched a real quake and were dropped.
  - Each is the declaring stations' vertical record from the 80 s before the declaration.

### 3.3 Fetching and preparing waveforms — `src/eq/seismic.py`

1. **One request per event:** a single SCEDC FDSN request covers every network station in range.
   It asks for `HH?` (100 Hz) and falls back to `BH?` (40 Hz), because before about 2010 the
   continuous archive is BH only.
2. **Instrument response removed** to ground velocity (m/s), with water level 60.
3. **Resampled** to 100 Hz.
4. **Low-passed:** every trace gets the same causal 18 Hz low-pass, so 40 Hz and 100 Hz sources look
   alike to the models. Live data gets the identical filter.
5. **Cut** to a long window per station: 32 s before to 36 s after the predicted P arrival.
6. **Rejected** if more than 2% zeros, non-finite, or dead.
7. **Stored compactly:** float16 plus a per-trace scale, under `data/raw/v2/` (about 2 GB).

Noise and hard-negative windows use the same steps but are only scaled by instrument sensitivity.
§4.1 explains why the detector's filter makes this difference irrelevant.

### 3.4 P-wave picking and travel times — `src/eq/locate.py`

Training data and live data are aligned by the **same picker**.

- **Picker:**
  1. A causal 2–10 Hz band-pass.
  2. An STA/LTA trigger (0.5 s / 5 s windows, on-threshold 3.5).
  3. An Akaike (AIC) refinement of the onset.
  4. SNR is the RMS of 2 s after the pick divided by the RMS of 5 s before.
- **Travel times:** first-arriving P, the faster of
  - the direct wave: √(d² + 8²) / 6.1 km/s;
  - the mantle-refracted wave: d / 7.9 + 6.0 s.
- **Correction:** an empirical correction per 10 km distance bin, fitted from 31,427 training picks,
  is added (`data/processed/v2/tt_correction.json`).

### 3.5 Assembled datasets — `data/processed/v2/`

| File | Contents |
|---|---|
| `magnitude.npz` | Per event: catalogue id, time, location, depth, magnitude. Per station record: 3-component velocity from 6 s before to 26 s after the **picked** P (predicted P if the pick SNR is < 2.5), distance, pick flag. |
| `detection.npz` | 34,377 event windows: the vertical channel from 26 s before to 30 s after P, kept when the pick SNR is ≥ 3. Plus 14,304 noise windows (45 s) and 2,062 hard-negative windows (80 s), normalized float16. |
| `tt_correction.json` | The travel-time correction table. |
| `dataset_meta.json` | Date range, counts, station list. |

### 3.6 Splits (no random splitting)

- **Events and noise:** chronological 70/15/15 by time.
  - Detection: validation starts 2019-07-07, test starts 2022-04-15.
  - Magnitude: validation starts 2019-07-10, test starts 2021-08-09.
  - As QuakeOps adds months (§12), the split is recomputed by the same rule, so the newest data is always
    in the test set.
- **Hard negatives:** split by date.
  - Train: Sep 22–28.
  - Validation: Sep 29 – Oct 1.
  - Test: Oct 2–5, 2026. This is the held-out "live noise" set.

---

## 4. The models

### 4.1 Detect — `scripts/demo_detect.py`, `DetectorNet` in `src/eq/models.py`

- **Question:** is there an earthquake in this 30 s vertical window?
- **Input preparation (`pipeline.det_prep`):**
  - Demean, then a causal 1 Hz high-pass, then unit standard deviation.
  - The identical function runs in training, replay and live.
  - The 1 Hz high-pass matters. Training positives are response-corrected velocity, while noise and
    live data are sensitivity-scaled counts. For broadband sensors the two differ only below about
    1 Hz, so removing that band stops the model learning the preprocessing instead of the earthquake.
- **Architecture:**
  - 4 strided 1-D convolutions (1 → 16 → 32 → 64 → 64 channels, kernel 7, stride 2, ReLU,
    batch-norm).
  - A 2-layer Transformer encoder (64-dim, 4 heads) over the resulting feature sequence.
  - Mean-pool, then a linear projection and a linear head giving the logit.
- **Training:**
  - Balanced batches of 256: half events; the other half noise, about a third of it hard negatives.
  - The P onset is placed at a random position 1–25 s into each crop, so detection doesn't depend on
    where the quake starts.
  - Random polarity flips.
  - Adam (learning rate 1e-3, weight decay 1e-4), cosine schedule, 12 epochs.
  - 5 seeds; the best is **chosen on validation AUC**.
- **Threshold:** the MCC-optimal value on validation (0.9987) is stored in the checkpoint. The live
  trigger threshold is set separately by calibration (§6.2).
- **Held-out test** (7,303 windows, 4,964 of them events):

  | | Result |
  |---|---|
  | ROC-AUC | **0.9998** (95% CI 0.9997–0.9999) |
  | MCC | **0.886** (0.868–0.902) |
  | STA/LTA on the same input | AUC 0.816 (0.805–0.828); paired difference +0.184 (+0.172…+0.195) |
  | AUC with P at 2 / 12 / 22 s into the window | 0.9997 / 0.9999 / 0.9997 |


  False positives per 30 s window. The live daemon triggers at **0.6** (`pipeline_config.json`), not at
  the checkpoint's 0.9987, so both are reported:

  | Threshold | Test noise (n=2,339) | Held-out live noise, Oct 2–5 (n=397) | Event windows caught |
  |---|---|---|---|
  | 0.6 (live trigger) | **1.41%** (33) | 0.25% (1) | 99.6% |
  | 0.9987 (checkpoint MCC) | 0.00% | 0.25% (1) | 91.9% |

  A per-window trigger is not an alert. It must also give a pick with SNR ≥ 3, and the picks of 3
  stations must locate one source with no silent nearer station. False *events* are therefore measured
  by the replay harness (§6.3: 3.5 false confirmed events/week, 0 false pushes) and, live, by the
  nightly crosscheck.

- **Memorization check:** train, validation and test AUC are 0.9998 / 0.9999 / 0.9998.
- **Confidence intervals** (`src/eq/stats.py`): percentile bootstrap, 2,000 resamples, **clustered by
  event**. Every station window of one quake (and every station's noise window at one random time)
  resamples together, because treating them as independent would make the interval too narrow.

### 4.2 Size — `scripts/demo_magnitude.py`, `MultiStationModel` in `src/eq/models.py`

- **Question:** given a located quake, what's its magnitude?
- **Input:** for each working station within 200 km, a 3-component window from 5 s before its P to
  25 s after.
- **Representation:**
  - Each window is normalized to unit peak, so the CNN reads **shape**.
  - Its standardized log₁₀ peak velocity and log₁₀ distance from the epicentre are **graph-node
    features**, so the graph reads **size**.
  - Four network statistics go to the head: mean and max log peak, mean and min log distance.
- **Architecture:**
  - A per-station 3-channel CNN.
  - Two graph-convolution layers over the 19-station graph. Edges are Gaussian distance weights
    (σ = 50 km, cut-off 150 km), symmetrically normalized.
  - A Transformer layer across stations.
  - Masked mean-pool, concatenated with the 4 statistics, then a 2-layer head.
- **Training:**
  - Mean squared error on catalogue magnitude; Adam (1e-3, weight decay 1e-4), cosine schedule,
    40 epochs, batch 32.
  - Augmentation mimics live conditions:
    - epicentre jittered by about 8 km, with distances recomputed;
    - ±0.5 s pick jitter;
    - only the nearest k ≥ 3 stations, or random 20% station drop-out.
- **Ensemble:** 5 seeds, averaged. The live output is their mean, with their spread as an
  uncertainty.
- **Checkpoint:** stores every normalizer (log-amplitude mean and std, statistics mean and std) plus
  the station list. The daemon refuses a checkpoint whose stations differ from `network.py`.
- **Held-out test** (937 quakes, M2.0–5.2):

  | | R² | MAE (magnitude units) |
  |---|---|---|
  | Deep ensemble | **0.951** (95% CI 0.943–0.959) | **0.098** (0.093–0.104) |
  | Live-like (10 km location error, nearest 3–6 stations) | 0.939 | 0.109 |
  | Nearest single station only | 0.808 | 0.203 |
  | Amplitude + distance linear baseline | 0.886 (0.871–0.898) | 0.158 |

- **Memorization check:** train, validation and test MAE are 0.089 / 0.099 / 0.098.
- **Seed-averaged result (10 seeds, `demo_magnitude.py --retrain --seeds 10`).** Two separate
  uncertainties:
  - *Seed variance:* a single model scores R² **0.949**, 95% t-CI 0.948–0.951 over 10 seeds (range
    0.943–0.952).
  - *Sampling variance:* the 10-seed ensemble scores R² **0.952** (event bootstrap 0.944–0.959), MAE 0.097.
  - The ensemble beats the baseline by ΔR² **+0.067** (paired CI +0.056…+0.079). The ensemble's gain over
    one model (+0.003) is smaller than the sampling CI, so 5 seeds are kept live.

### 4.3 Quick check — `scripts/fit_early_magnitude.py` → `data/processed/v2/early_mag.json`

A deliberately simple, fast preliminary size from the first **T = 4 s** after each picked P:

```
M_station = 0.709 · log10(peak 3-C velocity in [P, P+4 s]) + 1.477 · log10(distance km) + 4.026
M_quick   = median over the picked stations
```

- **Fit:** least squares on the training events. T (2, 3 or 4 s) was chosen on validation.
- **Threshold:** 3.04, the 5th percentile of the quick estimate for validation M3+ quakes.
- **Held-out test:**
  - MAE 0.24 (bias +0.08);
  - **93%** of M3+ quakes pass the threshold;
  - **1.1%** of quakes under M2.5 pass.

---

## 5. The live pipeline — `scripts/live_watch.py` + `src/eq/pipeline.py`

`server.py` starts `live_watch.py` as a child process and restarts it if it exits. The daemon
connects to SeedLink and subscribes to `HH?` for the 19 stations (with pinned location codes). Each
station keeps a 300 s rolling 3-component buffer.

Everything runs on **data time** (sample timestamps), never wall-clock time. Once a second, the engine
runs these steps:

### 5.1 Detect (every 2 s per station)

1. Take the station's newest 30 s vertical window, demeaned and 18 Hz low-passed exactly like
   training.
2. **Data-quality gate:** reject the window if any of these is true:
   - more than 5% exact zeros (gap fill);
   - a constant run longer than 0.5 s (stuck sensor);
   - more than 1% of samples at the clipping rail;
   - a lone glitch spike.
3. Score it with the detector.
4. **Trigger:** if P(quake) ≥ **0.6**, pick the P onset in the window. Keep the pick if SNR ≥ 3.
5. **Refractory:** after a pick, that station isn't picked again for 60 s. This stops S-waves and
   coda being picked as new P arrivals.

### 5.2 Locate and confirm (`associate`)

1. Gather the unused picks from the last 90 s, one (the earliest) per station.
2. With 3 or more, run the grid search:
   - coarse 0.05° grid over 31.8–37.0°N, 121.6–113.8°W, then a 0.01° refinement;
   - fixed 8 km depth; the origin time is the median implied by the picks;
   - with 4+ picks, the worst pick is dropped if that halves the misfit.
3. **CONFIRMED** only if all of these hold:
   - at least **3 picks** fit one source with RMS ≤ 1.5 s;
   - at most **1** working station *closer* to the epicentre than the farthest picking station stayed
     silent. A real quake reaches nearer stations first; with exactly 3 picks the fit is exactly
     determined, so this negative evidence is what rejects coincident noise;
   - the nearest picking station is within 120 km.
4. **Duplicates:** a new solution within 120 s and 100 km of an already-declared event is treated as
   that event's later phases or coda, and its picks are absorbed.
5. **TENTATIVE:** picks still unused after 40 s become a tentative declaration (1–2 stations, or no
   consistent fit). These are logged and **never** pushed.

### 5.3 Quick check (`early_ready`)

Each subscriber chooses an **alert speed** for the first message (`mode` in `push_tokens.json`; no mode means
standard). Both profiles run for every confirmed event (`EARLY_PROFILES` in `pipeline.py`):

| Profile | P-wave used | Conversion | Runs when | Fit |
|---|---|---|---|---|
| **Standard** (default, most safeguards) | 4 s | full response removal on [P − 30 s, P + 4 + 6 s] | the third pick + 4 s + 6 s margin | `early_mag.json`, threshold 3.04 |
| **Fast** | 2 s | sensitivity scaling only on [P − 30 s, P + 2 s]; broadband responses are flat in 1–18 Hz, so there's no end taper and no margin | the third pick + 2 s | `early_mag_T2.json`, threshold 3.05 |

Each profile computes M_quick = median over the picked stations of a·log10(peak 3-C velocity) + b·log10(distance) + c,
with its own a, b, c.

**What the choice costs** (`replay_archive.py early-variants`, 20 replayed days, `early_variants.json`):

| First-message variant | Held-out test days: first messages / real M2.5+ / retracted / no quake | Median after origin | Quick-size MAE (test / validation) |
|---|---|---|---|
| **Standard** | 6 / 6 / 1 / 0 | 33.4 s | 0.21 / 0.36 |
| 4 s, sensitivity-scaled, no margin (not offered) | 6 / 6 / 1 / 0 | 27.7 s | 0.20 / 0.35 |
| **Fast** (2 s, sensitivity-scaled, no margin) | 7 / 6 / 2 / 0 | 25.9 s | 0.26 / 0.56 |
| push at location, no size (not offered) | 72 / 8 / 67 / 5 | 25.6 s | — |

- **Fast** on validation days: 5 first messages, 3 for real M2.5+ quakes, 1 retracted, and 1 for no catalogued
  quake (Standard: 4 / 3 / 0 / 1).
- **The middle row** costs nothing in accuracy or false alerts. It's the candidate if Standard should ever
  become faster.
- **Pushing at location** would mean about 7 first messages a day, nearly all retracted.
- **The floor** is the third station's confirmation, about 25 s after origin.

### 5.4 Full sizing (`size_ready`)

Once the nearest 3 picking stations each have data to P + 25 s plus the 6 s margin (or up to 30 s more
if stations are late):

1. Cut every working station within 200 km at [P − 5 s, P + 25 s]. P is the station's own pick, or the
   travel time from the located origin if it didn't pick.
2. Response-correct it on the margin-padded segment. Without the margin, the correction's end taper
   would attenuate the newest seconds of the window.
3. Compute distances from the **located** epicentre.
4. Run the 5-model ensemble, giving magnitude ± spread.

### 5.5 Decide and push

| Stage | Condition | Push |
|---|---|---|
| Provisional (standard subscribers) | Confirmed and the standard quick check ≥ 3.04 | "Earthquake detected (*region*) — confirming size" with the preliminary size |
| Provisional (fast subscribers) | Confirmed and the fast quick check ≥ 3.05 | "Fast alert: earthquake detected (*region*)" with a rough size, marked "earlier, less certain" |
| Confirmation | Confirmed and full magnitude ≥ **3.0** | "M*x.x* earthquake confirmed (*region*)" with the distance from your nearest sensor and the expected shaking |
| Retraction | A provisional push went out but the full magnitude is < 3.0 (or couldn't be sized) | "Update: smaller quake (*region*) … you can disregard the earlier alert" |

- **Replacement:** every stage of an event carries the Android notification tag `quake-<event id>`, so
  a later stage **replaces** the earlier one in the notification tray.
- **Shaking wording:** "Weak shaking possible near you" and similar come from `shaking_model.py`, which
  maps magnitude and distance to an intensity label.
- **Who gets it:** each device following any station within **150 km** of the epicentre gets one
  message per stage.
- **Off switch:** pushes only go out when `PUSH_ENABLED=1` is in the server's `.env`. With it off
  (shadow mode), everything else still runs and is logged.

### 5.6 Logging and health

- **Event log:** every declaration is appended to `data/processed/events.jsonl`, including:
  - id, origin time and location;
  - stations, picks, location misfit, silent-station count;
  - quick-check magnitude and its timing;
  - full magnitude, its spread and timing;
  - push eligibility and the number of devices pushed.
- **Station health:** every 30 s the daemon writes `data/processed/live_status.json` (each station up
  or down, with latency). `/api/status` serves it.

### 5.7 Timing

On replayed days (no network delay), the median provisional push comes ≈ **33–35 s** after the origin on the
standard setting and ≈ **26 s** on fast. The median confirmation is ≈ **55 s** after for both. Live adds the
SeedLink delay (about 2–5 s). The confirmation goes to every subscriber in reach. A retraction goes only to
subscribers whose profile sent them a provisional push.

---

## 6. Evaluation and calibration — `scripts/replay_archive.py`

The **replay harness** is the acceptance test. It runs the exact live engine over archived continuous
data.

### 6.1 Commands

- **`scan`:** fetches each hour of archived vertical data for the 19 stations and prepares it like live.
  It scores every 30 s window on a 2 s grid and caches the score plus the picker result for windows at
  P ≥ 0.2 (`data/raw/v2/replay/`).
- **`run`:** feeds the cached windows through the same trigger and refractory logic
  (`Pipeline.triggerable` / `accept_pick`), then `Pipeline.advance` (locate → quick check → size).
  3-component windows are fetched on demand. The events are scored against the local catalogue.
- **`events`:** locates and sizes held-out test-period catalogue quakes from the cached event windows,
  in live geometry.
- **`calibrate`:** a grid search of the operating thresholds on validation days only.
- **`compare-live`:** scores a live daemon log with the same scorer.

### 6.2 Scoring and choices

- **Scoring:** a declared event is "real" if a catalogued quake (M ≥ 1) lies within ±20 s of its
  origin and 60 km of its location. Every precision is reported next to a **chance baseline**: the
  same declarations shifted +1 h.
- **Calibration** on 10 validation days (Sep 29 – Oct 2, 2026 and Sep 7–14, 2020) set the operating
  point:
  - trigger threshold 0.6;
  - pick SNR 3;
  - location RMS 1.5 s;
  - at most 1 silent nearer station;
  - **3 stations** to confirm;
  - a push floor of M3.0.
- **Why 3 stations and not 4:** 4 would have missed a real out-of-network M3.6 that 3 caught. The
  unmatched 3-station confirmations are logged only, sized ≤ M2.9, and cluster at the network edges.

### 6.3 Results on held-out days

On 10 test days (Oct 2–5 and Aug 18–25, 2026):

| | Result |
|---|---|
| Confirmed events that are real | **93%** (chance 0%) |
| Pushes | 5 confirmations, **0 false** |
| Provisional pushes | 6, all for real M2.5+ quakes, one later retracted |
| Location error | median **2.5 km** |
| Final magnitude vs catalogue on pushed quakes | within 0.13 |
| In-coverage M3+ caught | 4 of 5; the miss came 80 s after an M4.0 at the same spot, inside the coda-suppression window |

Event-centric test (833 test quakes, live geometry):
- located 89%;
- median location error 3.8 km;
- magnitude MAE 0.135 (bias +0.08).

Catch rate by size inside coverage, over all 20 replayed days:

| Magnitude | M1–1.5 | M1.5–2 | M2–2.5 | M2.5–3 | M3+ |
|---|---|---|---|---|---|
| Caught | 9% | 48% | 81% | 75% | 86% |

Quakes below M2 are outside the magnitude training range and size slightly high (around M2), which is
still below the push floor.

---

## 7. The backend API — `scripts/server.py` (stdlib HTTP, behind Caddy)

| Route | What it does |
|---|---|
| `GET /api/stations` | The 19 stations (code, lat, lon, region) from `network.py`. |
| `GET /api/status` | Whether the daemon is running, plus per-station up/latency and whether pushes are enabled. |
| `GET /api/ca?window=day\|week\|month\|year\|all` | The largest SoCal quakes in the window from USGS (details below). |
| `GET /api/geocode?q=` | City lookup (Nominatim, limited to California) for "find sensors near you". |
| `GET /api/version` | `{latest, min}` app versions for the in-app update gate. |
| `GET /api/health` | QuakeOps: `models` (registry champions, metrics ± CI, lineage, promotion history, from `models.json`), `drift` (`drift_status.json`), and `loaded` (the versions the daemon actually runs). Each is `null` until its job has run. |
| `POST /api/register-push` | Store `{token, stations, name, mode}`; it upserts by device token. Station codes are validated; `mode` is `standard` (default, also for older app versions) or `fast`. |
| `POST /api/unregister-push` | Remove a device. |
| `POST /api/contact` | Relay a support message by SMTP (nothing stored). |


**How `/api/ca` filters and marks quakes:**
- It lists only quakes the pipeline **could catch**:
  - **M ≥ 2.0**, with each window's own minimum applied on top;
  - a California place name;
  - **at least 3 stations within 100 km**.
- It marks each one against `events.jsonl` (matched within 30 s and 60 km):
  - `caught`: a confirmed event, with our magnitude;
  - `seen`: a tentative event only;
  - `missed`: no match;
  - none: the quake predates the live v2 pipeline (2026-10-05 09:13 UTC).

**Subscriptions** live in `data/processed/push_tokens.json` as `[{token, stations:[codes], name}]`.
No location is stored.

---

## 8. The website and app — `app/` (React + Vite; Android through Capacitor)

- **Detect / Size cards** (`seismic.json`):
  - the headline metric against the baseline;
  - a plain-language description;
  - an "Evidence" panel with a figure (`detect_evidence.png`, `size_evidence.png`, from
    `make_figures.py`) and a step-by-step technical rundown.
- **Where it can see** (`Coverage.tsx`):
  - an interactive map: the California outline (Census 1:20M), the 19 stations, and shading where 3+ or
    2 stations are within 100 km;
  - zoom with the wheel, a pinch or the buttons (1× = the full region, up to 8×); drag to pan;
  - more cities appear at 1.8× and 3.2×, and dotted city boundaries (Census 2023 places, simplified,
    `socal_cities.json`) from 2×.
- **Alert me near me** (mobile app only):
  - follow a **region**, which selects all its sensors and opens them, then turn single sensors off;
  - choose an **alert speed** for the first notice: Standard (most safeguards) or Fast (earlier, rougher size,
    more retractions), §5.3;
  - "Use my location" or a city search selects the nearest region if a sensor is within 150 km;
  - your coordinates are only used on the device to rank regions;
  - registering sends the chosen station codes and the push token.
- **Biggest Southern California quakes:** Day / Week / Month / Year / All time from `/api/ca`, each
  with its caught / seen / not-caught mark.
- **Model health** (`/health`, footer link; **in the code, not on the live site until the next deploy**):
  the live Detect and Size versions with held-out metrics ± 95% CI against their baselines, a drift pill
  per station (outlined = ok, gray = watch, black = drifting, dashed = no data; each labelled in text),
  and the promotion history. Each part reads "hasn't reported yet" until its VM job has run (§12.0). The
  result cards also show their 95% CI under the headline number (from `seismic.json`; same deploy).
- **Design context:** `PRODUCT.md` (audience, voice, principles) and `DESIGN.md` (visual system: monochrome,
  light only, Literata serif headings, fluid column and type scale).
- **App version gate:** the app compares its version with `/api/version`. Behind `min` on major or
  minor means blocked, with a link to the `/app` download page. A patch gap is only a notice.

---

## 9. Deployment — `DEPLOY.md`, `deploy/`

- **Host:** an Oracle Cloud Always-Free A1 VM, `/opt/seismicsocal`, with its own Python venv.
- **`seismicsocal.service`** (systemd) runs `server.py`, which starts and supervises `live_watch.py`.
  It loads `.env` (SMTP credentials and `PUSH_ENABLED`).
- **`seismicsocal-crosscheck.timer`** (installed) runs `crosscheck_events.py` every night at 09:00 UTC
  (02:00 Pacific), after the previous day's USGS catalogue has settled. It scores the whole live log
  since go-live and writes `data/processed/crosscheck_report.json`, split into **confirmed** (can
  alert), **pushed** and **tentative** (logged only), each with a +1 h chance baseline. Each USGS quake in
  the "Biggest quakes" list is also checked live, at request time (§7).
- **QuakeOps units (in `deploy/`, NOT installed yet; DEPLOY.md "QuakeOps"):**
- **`seismicsocal-quakeops.timer`** (09:30 UTC) runs `tracking.py pull` (resolves the registry
  champion, writes `models.json`, and installs a new champion only if `QUAKEOPS_AUTO_DEPLOY=1`), then
  `drift_check.py` (§12).
- **`mlflow.service`** runs the MLflow tracking server and registry on `127.0.0.1:5000`. It uses its own
  venv, SQLite and local artifacts. Caddy publishes it at `mlflow.seismicsocal.duckdns.org` behind
  basic auth, for the PC's training runs.
- **Caddy** provides HTTPS (Let's Encrypt), serves `app/dist`, and proxies `/api/*` to `127.0.0.1:8000`.
- **What ships:**
  - the code (from git);
  - `data/processed/detector.pt`, `magnitude_ensemble.pt` and `shaking_calibration.json`;
  - `data/processed/v2/{pipeline_config,tt_correction,early_mag}.json`;
  - the built `app/dist` with the APK;
  - `.env` and `fcm-service-account.json`.
- **After shipping:** checkpoint sha256 hashes are compared between the PC and the VM, then the service
  is restarted.
- **Policy:** after any pipeline change, run in **shadow mode** (`PUSH_ENABLED=0`). Score it with
  `crosscheck_events.py`, then enable pushes. A model change that passed the QuakeOps gate has already
  passed an offline shadow run, the replay rule G3. It still installs only when you allow it, either with
  `QUAKEOPS_AUTO_DEPLOY=1` or by running `tracking.py pull --apply` by hand.
- **CI/CD** (`.github/workflows/ci.yml`): every PR and push runs ruff, pytest, the daemon selftest and the
  site build. A push to `main` then streams `git archive` and `app/dist` to the VM, the same tar deploy as
  above, and restarts the service. This step is skipped until the `VM_HOST` / `VM_SSH_KEY` secrets exist.
  Models never go through git.
- **Android:** `scripts/build_apk.sh` with `VITE_API_BASE=https://seismicsocal.duckdns.org` builds the
  APK from the web build. That needs the Android project with `google-services.json`. The APK is
  served at `/app`.

---

## 10. Rebuild everything from scratch

```bash
python scripts/build_dataset.py                       # check → select → fetch (~2 h) → assemble
python scripts/demo_detect.py --retrain --seeds 5     # detector (GPU)
python scripts/demo_magnitude.py --retrain --seeds 5  # magnitude ensemble (GPU)
python scripts/fit_early_magnitude.py                 # quick check
python scripts/replay_archive.py scan --start 2026-09-29,2020-09-07,2026-08-18 --end 2026-10-05,2020-09-14,2026-08-25
python scripts/replay_archive.py calibrate --start 2026-09-29,2020-09-07 --end 2026-10-02,2020-09-14
python scripts/replay_archive.py run --start 2026-10-02,2026-08-18 --end 2026-10-05,2026-08-25 --tag test
python scripts/replay_archive.py events
python scripts/make_figures.py                        # site figures + publish metrics/CIs to seismic.json
python scripts/live_watch.py --selftest && pytest     # logic checks (24 tests)
python scripts/tracking.py register-legacy            # QuakeOps: current models -> registry v1 @champion
cd app && npm run build                               # site
```

---

## 11. Known limits

- **Coverage:** strongest where 3+ stations sit within about 100 km (LA basin, Inland Empire, Mojave,
  Ridgecrest, Kern). Outside the network (north of MPM, south of the border), locations come from a
  one-sided station triple and can be tens of kilometres off.
- **Latency:** pushes go out 30–60 s after the origin, so this is rapid detection, not pre-arrival
  warning.
- **Locator:** fixed 8 km depth, 1-D travel-time model.
- **Small quakes:** below M2, magnitudes read slightly high.
- **Station availability:** a station can drop off the public SeedLink relay. The pipeline then works
  with the stations that remain.

---

## 12. QuakeOps: tracking, retraining, promotion, drift

The design and its decisions are in `QUAKEOPS_IMPLEMENTATION.md`. This section describes how it runs.

### 12.0 Status (2026-10-05)

| Part | State |
|---|---|
| CIs, paired comparisons, seed-averaged write-up, `seismic.json` CIs | done on the PC (numbers in §4) |
| `retrain.py` gate, Dagster job, `build_dataset.py --append`, replay `--det/--mag` | in the repo. Gate checked on a dry run (champion vs itself); the append selection checked to reproduce today's lists. **No end-to-end retrain run yet** |
| MLflow tracking + `register-legacy` | checked against a throwaway local server. **No MLflow server on the VM yet** |
| `tracking.py pull`, the daemon's restart on a new version | in the repo, **untested against a live server** |
| Drift features (`live_watch.py`) + `drift_check.py` | checked on synthetic days. **Features start logging once the new `live_watch.py` is deployed** |
| `/health` page, `/api/health`, CIs on the cards | built (`npm run build`). **Not on the site until the next deploy** |
| CI (`.github/workflows/ci.yml`) | runs on the first push to GitHub. The deploy step waits for the `VM_HOST` / `VM_SSH_KEY` secrets |

### 12.1 Tracking and registry — `scripts/tracking.py`

- `tracking.py` is the **only** module that imports mlflow. Without `MLFLOW_TRACKING_URI` every call is a
  no-op, so all scripts run offline and in CI unchanged.
- **What a training run logs:** `demo_detect.py` and `demo_magnitude.py` log their params, lineage (git
  commit, a dirty-tree flag, dataset version, torch/CUDA), every metric with its CI bounds, and the
  checkpoint plus summary JSON under the run's `model/` directory.
- **Registered models:** `detector` (with sidecar `drift_reference.csv`) and `magnitude` (with
  `early_mag.json` and `tt_correction.json`, which are fitted from the same dataset). Each version is
  tagged with its checkpoint sha256. Aliases are `champion` and `challenger`.
- `register-legacy` registers the models that went live on 2026-10-05 as v1 `@champion`. It has been
  checked against a local test server, including that a second run doesn't duplicate anything. It has
  not yet been run against the VM registry.
- **`pull`** (VM, daily) writes `data/processed/models.json` (versions, metrics, lineage, promotion
  history). With `--apply`, or `QUAKEOPS_AUTO_DEPLOY=1`, it also downloads a new champion, verifies its
  sha256, and swaps it in atomically at the live paths. `live_watch.py` sees the new `deployed` version
  in `models.json` and exits, and `server.py`'s supervisor restarts it on the new model.

### 12.2 Monthly retrain — `scripts/retrain.py` + `scripts/quakeops_dagster.py`

Run by Dagster on the PC, on day 3 of each month for the previous month. The schedule is off until you
switch it on in the Dagster UI, and no retrain has run yet. Each stage can be resumed
(`data/processed/retrain/<month>/state.json`).

1. **data:** `build_dataset.py --append --end <month end>`.
   - M3+ events are declustered per cell and month, so months already in the dataset never change.
   - The M2–3 sample and the noise times are drawn for the new month only, at the original rate.
   - The fetch cache means only the new month downloads.
   - `dataset_meta.json` gets a new `version`, a hash of the selection lists. The 2026-09-01 dataset is
     `2105a42878a2`.
2. **train:** both demo scripts with `--retrain --out <month dir> --compare <champion>`. The champion is
   re-scored on the **challenger's** test split, which neither model has trained or validated on (the
   split rule is unchanged, §3.6). Then the quick-check fit, the travel-time table and the drift
   reference are built, and the run is registered `@challenger`.
3. **replay:** the 10 held-out replay days, run three ways: (champion, champion), (challenger detector,
   champion size) and (champion detector, challenger size). The challenger detector re-scans into its own
   cache `replay/<sha12>/`. Event-centric sizing runs for both magnitude models.
4. **gate:** the rules below. Results go to `gate.json` and an MLflow `quakeops-gate` run.
5. **promote:** for each model that passes, `@champion` moves. The version is tagged with the reason, gate
   run and previous champion, a record is appended to `promotions.jsonl`, the files are installed at the
   PC's live paths, `seismic.json` is republished, and an email is sent. A model that fails stays
   `@challenger`, tagged with the rules it failed. `retrain.py rollback --model M --to N` reverts.

### 12.3 Promotion gate

Detect and size are decided independently. A rule that can't be evaluated is logged as `SKIPPED`, never
as a silent pass.

| Rule | Detect | Size |
|---|---|---|
| G1 beats the classic baseline | paired ΔAUC vs STA/LTA, CI low > 0 | paired ΔR² vs amp+dist, CI low > 0 |
| G2 non-inferior to the champion (same test set, paired) | ΔAUC ≥ −0.001 and CI high ≥ 0; ΔMCC ≥ −0.02; per-window false-trigger rate **at the live trigger (0.6)** ≤ champion + 0.5 pp on test noise and + 0.25 pp on held-out live noise | ΔR² ≥ −0.01 and CI high ≥ 0; ΔMAE ≤ +0.01 |
| G3 replay acceptance (calibrated config fixed) | confirmed precision ≥ 0.85 and ≥ chance + 0.5; no false provisional or final pushes; in-coverage M3 recall ≥ champion; pushed magnitudes within 0.3 of the catalogue | the same push and magnitude checks; event-centric MAE ≤ champion + 0.02 |
| G4 tests | pytest + `live_watch.py --selftest` | same |
| G5 lineage | clean git tree; commit, dataset version and seeds recorded; replay test days inside the challenger's test period | same |
| G6 reason to switch | newer data than the champion, or a superiority CI > 0 | same |

The gate never changes `pipeline_config.json`. A detector that needs a new trigger threshold fails G3,
and calibrating it is a deliberate, manual step.

### 12.4 Drift — `scripts/drift_check.py`

- **Live features:** `live_watch.py` logs one window per station every 30 s to
  `data/processed/features/<date>.csv`:
  - the detector's score;
  - `crest` = log10(max/rms);
  - `hf_ratio` = the share of 1–18 Hz power above 5 Hz.
  All three are computed on the `det_prep` input. That makes them scale-free and comparable with
  training data. Raw amplitude isn't comparable: training noise is normalized, live data is raw counts.
- **Reference:** the same features on each station's **training-split noise windows**, about 280–650
  per station, scored by the champion.
- **Daily check:** Evidently `DataDriftPreset` (normed Wasserstein > 0.1) per station, yesterday against
  the reference. Status is `ok` (no feature drifted), `watch` (1), `drifting` (2 or more) or
  `insufficient` (fewer than 500 rows).
- **Outputs:** HTML reports in `data/processed/drift/<date>/`, `drift_status.json` and
  `drift_history.jsonl`. An email is sent when a station has been drifting for 2 days in a row. Data
  older than 90 days is deleted.
