# URGENT — live alert quality + magnitude fix (handoff 2026-10-04)

Cross-checked the live daemon's declared events against the USGS catalog and found the live
alerting is noisy and the live magnitude is effectively broken. Code fixes are **committed + pushed**
(`3c92912`, `46867f5`) but **NOT yet deployed to the VM** — the live site still runs the old behavior
until you ship + restart (steps at the bottom).

## What the data showed (VM `data/processed/events.jsonl`, 32.4-day window)

- 2,127 declarations = **118 CONFIRMED** + 2,009 TENTATIVE (lone-station, never pushed) + **43 PUSHED**.
- Of the 118 confirmed events, only **10–22 matched a real USGS quake**
  (precision **0.08** strict / **0.19** generous match window).
- Of the 43 pushes that reached phones, **7–16 were real, 27–36 were false** → ~**2–3 false push alerts/day**.
- **Every live magnitude collapsed to ~M2.5**, even for real M3.6 (Hermosa Beach) and M3.8 (Piru) quakes.
  The live mag does not discriminate size at all right now.
- Recall of M≥3 in-region: 3/4 at the generous window (missed one M3.2), 1/4 at the strict window.

Reproduce anytime:
```
scp -i ~/.ssh/oracle_seismic ubuntu@167.234.214.169:/opt/seismicsocal/data/processed/events.jsonl data/processed/events_vm.jsonl
python scripts/crosscheck_events.py --log data/processed/events_vm.jsonl           # all tiers
# (filter confirmed-only first if you want to limit USGS calls — see data/processed/events_vm_confirmed.jsonl)
```

## What was fixed (committed + pushed, main)

1. **Magnitude SCALE bug** (`46867f5`, `scripts/live_watch.py`)
   - `SCALE` was `6.954687e-4`; the deployed `magnitude_ensemble.pt` trains with
     `X[mask].std() = 7.773395e-4` over `seismic_phase2a_xl.npz`.
   - **Verified**: the checkpoint's stored `am`/`asd` match that npz *exactly*, so the checkpoint
     trained on this npz and `7.773395e-4` is unambiguously correct. Live inference had been
     under-normalizing waveforms ~12% vs training.

2. **Felt-shaking push floor** (`46867f5`, `scripts/live_watch.py`)
   - New `ALERT_MIN_MAG` (default **M3.0**, CLI `--min-mag`): a CONFIRMED event is PUSHED only if sized
     ≥ the floor. Smaller or unsized events are still logged (`confirmed=True`) but **not pushed** —
     below perception and outside the net's M≥3.5 trained range.
   - Logic is one `push_eligible()` helper, asserted in `--selftest` (passes).

3. **Cross-check tooling** (`3c92912`, `scripts/crosscheck_events.py`)
   - `--log <path>` to score any pulled log; prints CONFIRMED / TENTATIVE / PUSHED precision breakdown.

## ⚠️ Known limitation — do NOT trust the M3 floor yet

The SCALE fix (a ~12% correction) does **not** explain the ~M2.5 collapse. The real causes:
- the magnitude net is trained only on **M≥3.5**, so the mostly tiny/noise confirmed events are
  out-of-distribution;
- the live distance proxy is the strongest **station**, not the true epicentre, so the per-station
  `dist`/`logdist` aux features differ from training.

So **until the under-reading is diagnosed, an M3.0 floor may suppress real quakes too.** Validate and
recalibrate before relying on it (next section).

Local `.venv` can't run the model (Application Control policy blocks a scipy DLL), so `--replay` /
live validation must run on the VM or another working Python env.

## Deploy + validate (run on your PC / the VM)

```bash
# 1. ship the code (models unchanged — SCALE is a constant, no retrain needed)
cd /c/Users/braaa/coding/earthquake
git pull                                  # get 46867f5

rsync -av scripts/live_watch.py scripts/crosscheck_events.py \
    ubuntu@167.234.214.169:/opt/seismicsocal/scripts/     # or scp, with -i ~/.ssh/oracle_seismic

# 2. on the VM: confirm the SCALE fix restores in-range magnitudes BEFORE restarting the daemon
ssh -i ~/.ssh/oracle_seismic ubuntu@167.234.214.169
cd /opt/seismicsocal
python scripts/live_watch.py --replay        # expect estimate ≈ true for cached M3.5+ events
python scripts/live_watch.py --selftest      # push-floor + tier asserts

# 3. recalibrate ALERT_MIN_MAG from the post-fix mags of confirmed events that matched real USGS
#    quakes vs. the false ones, then edit ALERT_MIN_MAG (or pass --min-mag). Only then restart:
sudo systemctl restart seismicsocal

# 4. after a day, re-score to confirm false pushes dropped:
python scripts/crosscheck_events.py --log data/processed/events.jsonl
```

## Bigger follow-up (not done)
The honest fix for the collapse is a **statewide dataset rebuild + magnitude retrain that includes
M<3.5**, so the net is in-distribution for the small events the detector actually confirms, and a real
epicentre estimate instead of the station proxy. Tracked in `CLAUDE.md` Open TODOs.
