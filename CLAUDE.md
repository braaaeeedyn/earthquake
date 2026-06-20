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

## Progress & handoff (last updated 2026-06-18)

Read this before continuing. Build order ≠ chronological order: the web app surface
(step 7) and the model flow (steps 3–6) were built on the **synthetic fixture** first,
deliberately, so real-data acquisition (a gated dependency) doesn't block anything.

### Done ✅
- **Pipeline (build order 1–2).** `src/eq/config.py` (single source of truth for all
  locked rules), `src/eq/pipeline.py` (minute→hourly→daily→24×27, labeling, chronological
  split with 34-day embargo, train-only normalizer), `src/eq/graph.py` (station adjacency),
  `src/eq/synthetic.py` (fake INTERMAGNET data with a learnable precursor signal).
- **Model flow (build order 3–6).** `src/eq/models/`: `cnn.py`, `gnn.py` (hand-rolled GCN),
  `transformer.py`, `fusion.py`, plus `data.py` (npz→tensors, adjacency, `pos_weight`),
  `metrics.py` (precision/recall/F1 + val-only threshold), `train.py` (generic loop).
  **Key design:** every sub-model is a feature extractor with a uniform
  `forward(x) -> (B, out_dim)` interface; `Classifier` wraps one for standalone
  train/test, `FusedClassifier` late-fuses all three reusing the same backbones — no
  rewrite to go standalone → fused. Keep this contract when extending.
- **Web app surface (build order 7, partial).** `app/` React+Vite+Capacitor reads
  `app/public/forecast.json`. Runs at `localhost:5173` (`cd app && npm run dev`).
- **Tests:** 18 green (`pytest`) — pipeline/graph + overfit-one-batch per sub-model & fusion.
- **Env:** `.venv` has `torch`, `scikit-learn`, `matplotlib` installed (`requirements-ml.txt`).

### Current results (SYNTHETIC data — not a scientific result)
`python scripts/train_models.py` → fused model ~0.79 acc / F1 0.73 / **recall 0.57**,
beats the 60% accuracy baseline. Recall is the weak spot and the metric that matters most.

### Next steps (do these next session, roughly in order)
1. **Improve recall** on the positive class (currently 0.57). Options: tune `pos_weight`,
   threshold strategy already selects on val, class-balanced sampling, more epochs/capacity.
   This is still on synthetic data, so treat it as wiring/validation, not science.
2. **Phase 3 — real data (GATED, needs the user).** Real INTERMAGNET access requires the
   user's credentials/registration; USGS catalog is public. Build a USGS fetcher + an
   INTERMAGNET loader behind the SAME `build_dataset` interface so models are unchanged.
   **Ask the user for INTERMAGNET access before starting.**
3. **Phase 5 — honest evaluation** on real held-out test: precision/recall/F1 vs the 60%
   baseline. Re-confirm chronological splits hold on real data.
4. **Phase 6 — wire real inference into the app.** `scripts/make_forecast.py` is still a
   PLACEHOLDER (hardcoded probability, `is_placeholder: true`). Replace `compute_forecast`
   with real model inference over the latest window; keep the `forecast.json` contract
   (see README "Forecast contract"). Then produce evidence figures for Dr. Sanders/WDSI.

### Gotchas
- Synthetic train pos_rate (~0.77) differs from val/test (~0.45) — an artifact of clustered
  positive anchors, harmless for a smoke test. Real data will be heavily imbalanced instead.
- Run `python scripts/build_dataset.py` to (re)materialize `data/processed/dataset.npz`
  before `scripts/train_models.py`; the npz is gitignored.
- Nothing has been committed yet this work — changes are in the working tree only.
