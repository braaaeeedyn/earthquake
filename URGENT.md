# URGENT — resolved (2026-10-05)

The 2026-10-04 handoff (noisy live alerts, every live magnitude ≈ M2.5) is resolved on `main`; its
suggested steps (edit `SCALE`, recalibrate `ALERT_MIN_MAG` on the old pipeline) are **obsolete** — do not
follow them.

Root causes found (details + evidence: `URGENT_PLAN.md`):
- **Magnitude collapse** was a pipeline bug, not the model: live sized at trigger time (only the first
  seconds of P in the window) with distance from the strongest *station* (bias −2.1 to −2.4 units).
- **False alarms:** five of the ten trained stations never streamed on the public SeedLink relay; the live
  network was five mostly-urban stations, two of them 13 km apart, with a detector that had seen one of them.
- The VM ran Aug-10 models on the old dataset while the site advertised newer ones (`SCALE` flip-flopped).

Fix (all on the PC, validated offline with the replay harness): new 19-station live network
(`src/eq/network.py`), v2 datasets, retrained detector + magnitude ensemble, and a new live engine
(`src/eq/pipeline.py`: pick → locate → aligned sizing → 3-station confirmation). See README "Replay harness"
for the held-out scorecard and `DEPLOY.md` for the hash-verified, shadow-mode deployment.
