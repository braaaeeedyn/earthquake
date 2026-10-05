# URGENT fix plan (v2): redesign the live pipeline and the station network

> Responds to `URGENT.md` (2026-10-04). Supersedes v1 of this file. **Implemented 2026-10-05 (status below); the plan text is kept as written, and
> the VM is not to be touched** until the whole plan has been built and passed the offline replay
> harness (Phase 2). The diagnostics behind every claim here were run on 2026-10-04: a simulation
> on held-out events, the VM's `events.jsonl` relabelled against USGS, a read-only inspection of
> the VM, the IRIS SeedLink inventory, SCEDC metadata and archive, and noise measurements. The
> scripts are in the session scratchpad and get promoted to `scripts/` in Phase 0.

## Status (2026-10-05): implemented on the PC

Phases 0–5 are done and validated offline; Phase 6 (deploy) follows with shadow mode. Deviations from the
plan, each forced by evidence found while building:

- **Network:** SCZ2 dropped off the SeedLink relay during selection → replaced by **SNCC**; IKP added
  (19 stations). PASC pinned to location `10` (the STS-2 with continuous history).
- **Archive:** pre-2010 SCEDC continuous data is 40 Hz BH → BH fallback + a common 18 Hz low-pass on
  every trace (training and live).
- **Detector shortcut found and removed:** positives were response-removed velocity, noise was
  sensitivity-scaled counts; the model partly learned that difference. Fix: shared `det_prep`
  (1 Hz high-pass) everywhere; verified that remaining high scores on pre-P windows are real quakes.
- **Magnitude representation:** a single global amplitude scale trained unstably on the larger data
  (R² 0.64 < baseline 0.79) → unit-peak waveforms + log-peak node feature (R² 0.95 vs baseline 0.89).
- **Magnitude range:** M≥3.0-only training sized live M1.5–2 detections at ≈M3 (replay found it) →
  magnitude set now M2.0–7.1 (adds the cached M2–3 events). The plan's "M3.0" item is superseded.
- **3 picks are exactly determined** (lat, lon, t0), so RMS can't validate them → added the
  negative-evidence rule (no silent healthy nearer station).
- **Hard negatives** date range is Sep 22 – Oct 5 (the old log's real span), split train/val/test by date.
- Replay of the Sep 2020 validation week was reaped for low memory at 15/168 h; calibration used the
  Sep 29 – Oct 2 validation days.

Held-out results and the full methodology: README ("The three models", "Replay harness").

---

## 1. What is actually wrong (root causes)

### 1.1 The core design flaw: training is event-aligned, live is clock-driven

Every training sample was cut **knowing the answer**. Each station's window starts exactly 5 s
before the catalog-predicted P arrival (`fetch_window_3c`: `origin + d/VP − LEAD_S`), and its
distances come from the **catalog epicentre**.

Live inference has neither. It slides a 30 s window on a 2 s wall-clock scan and measures distance
from the strongest station. Every live stage inherits that mismatch:

| Stage | Training saw | Live gives it | Measured effect |
|---|---|---|---|
| Detect | P always about 5 s into the window | the onset at *any* position, first appearing at the window's end | trigger timing is scan-clock noise. The detector never saw a partial onset |
| Coincidence | — | trigger time = **wall-clock scan time**, not data time. Move-out bound is 2 km/s plus 4 s slack | for PASC–MWC (13 km apart) any Δt ≤ 10.5 s passes. Shared city noise "confirms" itself |
| Size, window | P at 5 s, then 25 s of S and coda | sized at the trigger moment, so only the first seconds of P are in the window | **bias −1.5 to −1.8 mag units** |
| Size, distance | catalog epicentre | distance from the strongest *station* | **bias −0.7** |
| Size, combined | — | — | **bias −2.1 to −2.4**, estimates M1.0–2.8. This is exactly the "everything reads M2.5" collapse |
| Size, mask | stations ≤ 200 km with data | every station with any buffer | out-of-distribution station sets |

With windows aligned as in training, the same model gets R² 0.83. It still gets 0.78 with a 25 km
location error. **The models are fine. The pipeline feeds them inputs they were never trained on.**
`SCALE` matters little: R² is 0.84 vs 0.83 between the two candidate values.

### 1.2 What happened to the network

- **IRIS's public SeedLink server (`rtserve`) carries only a subset of CI stations**, about 54
  backbone broadband sites. **CCC, TOW2 and RIO are still recording.** SCEDC has their data from
  2026-10-03. They're just **not relayed in real time**. CLC and WBM had no archive data either at
  the checked minute.
- The training network was chosen from the **SCEDC archive**, which has every station. Nobody
  checked it against the **live** source. Five of ten trained stations could never stream.
- The live network is therefore effectively **PASC, SVD, MWC, DGR and BAK**:
  - The detector was trained on only one of them (PASC).
  - **BAK** is one of the noisiest sites (≈7e-7 m/s in daytime, about 100× MPM), which accounts for
    1,046 lone false triggers.
  - **PASC and MWC** are 13 km apart in the LA basin, so correlated cultural noise between them
    passes the coincidence check (84/116 false confirmations happen between 06:00 and 18:00 PDT).
  - Only **9%** of SoCal seismic areas have ≥3 live stations within 100 km.
- The station list is **hard-coded in four places**, so nothing kept them in sync:
  - `src/eq/seismic.py` `STATIONS`/`STATIONS2`
  - `server.py` `STATIONS`
  - `live_watch.py` (read from the npz)
  - `crosscheck_events.py` (read from the npz)
- **PASC** streams two sensors (`00`, `10`). The daemon subscribes without a location code and
  merges them into one buffer.

### 1.3 Deployment drift

The VM runs **Aug-10 models trained on 2010–2023 data** (794 magnitude events). The site and
`CLAUDE.md` claim the 2000–2025 models. `SCALE` was edited back and forth between the two (7.77e-4
fits the VM's model, 6.95e-4 fits the PC's). Nothing checks that a checkpoint, its scale, its
station list and the configured network belong together.

### 1.4 The scorecard overstates success

`crosscheck`'s "generous" match (±180 s, 100 km, **any magnitude**) gives 22/118, but a +1 h
time-shifted control still gives 9/118. Strict matching (±60 s, ≤50 km, M≥2) gives **2/118**. Small
real quakes (about 1,800 M0+ events in the region in 32 days) make chance matches common.

### 1.5 Label definitions

- Detection "noise" windows only exclude **M≥2.5** catalog events. Thousands of real M0.5–2.5
  quakes can sit inside windows labelled noise.
- Positives are M≥3 only.
- So the detector's notion of "earthquake" isn't defined consistently with what live sees or with
  how it's scored.

---

## 2. The redesign: methodology

Two principles:

1. **Live inference reconstructs the training alignment.** Pick, then associate, then locate, then
   cut aligned windows, then size. No stage uses wall-clock time or a station-as-epicentre proxy.
2. **Training covers the live conditions that can't be reconstructed:** onset at any position in
   the window, real live noise including daytime urban noise and the system's own past false
   triggers, the actual live stations, and station dropouts.

### 2.1 New live pipeline (`live_watch.py`, rewritten stage by stage, same daemon and supervisor)

```
SeedLink (explicit NET.STA.LOC.HH? per station)  →  per-station ring buffers in DATA time
  │  station health: latency, gap %, last packet → stale stations excluded, exposed in /api/status
  ▼
DETECT   per station, sliding 30 s window every 1–2 s (DQ gate kept) → trigger(station, data-time)
  ▼
PICK     AIC/STA-LTA onset picker on the Z trace around the trigger → P arrival time (≈0.1–0.5 s)
  ▼
ASSOCIATE + LOCATE   grid search (lat, lon, t0; fixed depth ~8 km; Vp≈6.2 km/s) over picks
         CONFIRMED  = ≥3 stations, RMS residual ≤ R (tuned), epicentre inside covered region
         negative evidence: a healthy station near the epicentre that did NOT trigger lowers confidence
         TENTATIVE  = 2 stations or a poor fit → logged, never pushed
  ▼
SIZE     wait until P + 25 s at the farthest station used (≈30–45 s after origin), then
         cut [P_i − 5 s, P_i + 25 s] per station (picked or predicted from the location),
         mask = healthy stations ≤ 200 km, dist = from the LOCATED epicentre → magnitude ensemble
         (+ ensemble spread as an uncertainty)
  ▼
DECIDE   push iff CONFIRMED ∧ M̂ ≥ ALERT_MIN_MAG (calibrated in the harness) ∧ PUSH_ENABLED
  ▼
PUSH + LOG   message says "near <place of located epicentre>", distance from the epicentre to the user's
         nearest subscribed station; events.jsonl gets picks, residual, location, mag, spread, health
```

- Pushes come about 30–45 s after origin. That's still "rapid detection", which is what the app
  claims. It isn't EEW, and the alert text already says so.
- Requiring **≥3 stations** is now feasible because the new network is denser where it matters
  (§3). With stations at least 49 km apart, correlated local noise can't satisfy a 3-station
  physical fit.
- The picker and locator are about 150 lines of numpy. **No new dependencies.**

### 2.2 Training methodology (rebuild plus retrain; this is where retraining is needed)

| Change | Why |
|---|---|
| **Build on the new live network only.** The builder checks every station against the SeedLink INFO list and fails if one isn't streamable | prevents §1.2 from happening again |
| **Longer raw windows** (P − 40 s → P + 50 s) cached once. Training crops are taken from them | enables onset-position augmentation (below) and leaves room for EEW later |
| **Detection: onset-position augmentation.** Positive crops place P anywhere in [1 s, 25 s] | the detector becomes time-invariant, as live requires |
| **Detection labels:** positive = any catalog event M≥2 within 100 km or M≥3 within 200 km. Noise = no catalog event of **any** magnitude within ±120 s / 300 km, sampled **across all hours** | removes the mislabelled small quakes. Daytime noise is in training |
| **Hard negatives:** the 2,127 logged live declarations (station + time are known) are fetched from the SCEDC archive. The ones with no catalog event are added as negatives. Sep 1–24 is used for training; **Sep 25–Oct 4 is held out** as a live-noise test | the system learns from its own false alarms. This sits outside the chronological 70/15/15 split, which stays as locked |
| **Magnitude: `--min-mag 3.0`** (was 3.5) | the push floor of about M3 then falls inside the trained range |
| **Magnitude robustness:** during training, jitter the epicentre by an empirical locator-error distribution and randomly drop stations | live locations are estimates and stations drop out |
| **Selection on val, not test** (fixes `demo_detect.py:76`) | test leakage |
| **The checkpoint carries `scale`, the threshold, the station list with location codes, `dataset_version` and the git commit.** The daemon **refuses to start** if the checkpoint's station list ≠ the configured network | ends the `SCALE` and VM-drift class of bug (§1.3) |
| **Operational metrics** reported alongside AUC/MCC: per-window false-positive rate on continuous live noise at the operating threshold; per-station breakdown | AUC 0.99 can still mean hundreds of false triggers a day at 43k windows per station per day |

Locked rules are unchanged: chronological 70/15/15, MCC/AUC/R² vs baselines, conservative claims.

### 2.3 Evaluation methodology

- **The replay harness is the acceptance test, not test-set AUC.** It runs the *exact* live code
  over **archived continuous data** from SCEDC FDSN for any stations and dates, offline,
  deterministic and repeatable. It also delivers the "replay a real earthquake" TODO.
- **Honest scorecard** (`crosscheck_events.py`):
  - strict matching by default
  - a **time-shifted chance baseline** printed next to every precision
  - separate numbers for *detection* (vs any-magnitude catalog), *push* (vs M ≥ floor), recall of
    in-coverage M≥3, **location error** and **magnitude bias/MAE** on matched events
- **Shadow mode before pushes:** once deployed (later), the new pipeline runs with
  `PUSH_ENABLED=0` for at least 7 days, and the live scorecard must match what the harness
  predicted.

### 2.4 One source of truth for the network

- A single `LIVE_NETWORK` table (net, sta, loc, lat, lon) in `src/eq/seismic.py`.
- Builders, `live_watch`, `server.py` `/api/stations` and `crosscheck` all import it.
- The app already fetches `/api/stations` dynamically, so **no APK update is needed** for the
  station list.
- A one-time **migration of `push_tokens.json`** maps each removed station to its nearest new
  station(s). The app also drops locally saved station codes that `/api/stations` no longer lists.

---

## 3. Proposed network: 18 quiet, spread-out, streamable stations

Selection rules:
- On `rtserve` with HH channels.
- Recording since 2006 or earlier, so there are 20+ years of training data.
- Median 2–8 Hz noise ≤ 1e-7 m/s across 4 times of day.
- **Minimum spacing ≈ 50 km**, so neighbours can't confirm each other on shared local noise.

Chosen greedily for coverage of SoCal's seismic areas (0.2° cells with any M3+ quake 2000–2025)
plus the major cities, then hand-adjusted to cover San Diego and Imperial.

| Region | Stations |
|---|---|
| LA basin / San Gabriels | **PASC** (loc 10 pinned), **BFS** |
| Inland Empire / Elsinore–San Jacinto | **SVD**, **DGR** |
| San Diego / border | **BAR** |
| Imperial / Salton | **SWS**, **BEL** |
| Mojave / Eastern CA shear zone | **GSC**, **GMR**, **EDW2** |
| Ridgecrest / Coso | **LRL**, **MPM** |
| Southern Sierra / Kern | **ISA**, **ARV** |
| Central coast / Santa Barbara | **SMM**, **MPP**, **SCZ2** |
| Offshore / Catalina | **CIA** |

| | Current live (5) | Proposed (18) |
|---|---|---|
| Seismic areas with ≥3 stations ≤100 km | 9% | **50%** (≥2: 70%) |
| M3+ events 2000–2025 with ≥3 stations ≤100 km | — | **61%** |
| Closest station pair | 13 km (PASC–MWC) | **49 km** (BFS–PASC) |
| LA / Irvine / Riverside / Ridgecrest / Lancaster (stations ≤100 km) | 2/4/4/0/2 | **4/5/4/5/5** |
| San Diego / El Centro | 0/0 | 1/1 → add **IKP** (2010+) to reach 2/2 |

- **Dropped:** BAK (very noisy), MWC (13 km from PASC), plus the five unstreamable stations.
- The uncovered half of the seismic areas is mostly offshore, the Baja border and remote edges.
  The UI should say coverage is strongest in the LA / Inland Empire / Mojave / Ridgecrest areas.
- Final selection becomes a committed, reproducible `scripts/select_network.py`.
- **Caveats:**
  - Noise was measured on one day. Phase 0 re-measures over 7 days.
  - Island and coastal stations (CIA, SCZ2) carry ocean microseism, which is below 1 Hz and largely
    outside the detection band. Their effect is checked in the harness.

---

## 4. Phases (all offline. The VM is untouched until Phase 6)

**Phase 0: Tooling and network (1–2 days)**
- Promote the diagnostics to `scripts/`:
  - `select_network.py` (candidate query, 7-day noise, coverage, spacing)
  - `mag_diag.py` becomes a regression test of live-path vs training-path sizing
- Add the `LIVE_NETWORK` table and wire every consumer to it (§2.4).
- Make `crosscheck` honest (§2.3).
- Re-create a minimal `tests/` folder (also QuakeOps F1): picker, locator, `clean_window`,
  association, push decision, scale/station-list guard.

**Phase 1: Replay harness (2–3 days)**
- `scripts/replay_archive.py` pulls archived data and runs the live code against it.
- **First, reproduce today's failure:** the old 5-station live set and the VM's models over Sep 1 –
  Oct 4 should give about 116 false confirmations and mags of about 2.5. That validates the
  harness before anything new is measured.

**Phase 2: Pipeline fix with the current models (3–4 days)**
- Implement pick, associate/locate, deferred aligned sizing, and the decision stage (§2.1), with
  station health.
- Measure in the harness on the *current* models and the *proposed* network:
  - location error
  - magnitude bias/MAE on real M2.5+ events
  - false confirmations per week

  This shows how much the pipeline fix alone delivers.
- Note: the current detector has never seen 17 of the 18 proposed stations, so its false-trigger
  rate there is unknown. That's why Phase 3 exists.

**Phase 3: Rebuild and retrain (about 1 week, mostly fetch time)**
- Rebuild both datasets on the 18 stations, 2000–2025, with long windows (§2.2).
- Mine hard negatives.
- Retrain the detector (augmented) and the magnitude ensemble (M≥3.0, locator jitter, station
  dropout), 5 seeds, selected on val.
- Store all metadata in the checkpoint.
- EEW is untouched (not live).

**Phase 4: Calibrate and accept (2 days)**
- In the harness, on held-out periods: tune `DET_THRESH`, the locator residual R and
  `ALERT_MIN_MAG` on **val months**, then report once on **test months** plus the Sep 25–Oct 4
  live-noise set.
- **Acceptance** (proposed, adjust as you see fit):
  - confirmed precision ≥ 0.5 vs a chance baseline < 0.05
  - ≤ 1 false confirmation per week
  - recall ≥ 0.8 for in-coverage M≥3
  - median location error ≤ 15 km
  - magnitude MAE ≤ 0.4 and |bias| ≤ 0.2 on matched events

**Phase 5: Docs**
- Update `seismic.json` with the new honest numbers.
- README: method, coverage map, limits.
- `CLAUDE.md`: network, rules, and the "Deployed" section corrected.
- `DEPLOY.md`: hash-verified model shipping.
- Retire `URGENT.md`.

**Phase 6: Deploy (only after your go-ahead)**
- Ship the code and models, and verify the sha256 hashes on the VM.
- Migrate `push_tokens.json`.
- Run **shadow mode for 7 days** with `PUSH_ENABLED=0`, compare against the harness, then
  enable pushes.

QuakeOps starts after Phase 4, since its frozen `test_v1` must be cut from the *new* datasets.

---

## 5. Decisions needed

1. **Network:** the 18 stations above, plus **IKP**? Or do you have preferences, for example
   keeping MWC instead of PASC (PASC's two sensors need a pinned location code)?
2. **Confirmation rule:** ≥3 stations required for a push (recommended), with 2-station events
   logged only?
3. **Push latency of about 30–45 s** after origin in exchange for correct sizing and location. The
   alternative is pushing an unsized "detected" notice first and a sized update after.
4. **Magnitude training floor of M3.0** (or 2.5, which gives more data but noisier labels).
5. **The acceptance thresholds** in Phase 4.
6. **The live VM keeps running unchanged until Phase 6**, including its current false pushes of
   about 2–3 a day, as you asked. Do you want **only** the one-line `PUSH_ENABLED=0` pause shipped
   now, with no model or pipeline change? I recommend it: the alerts are wrong far more often than
   right. Your call.
