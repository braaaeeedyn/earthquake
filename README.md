# Earthquake Forecasting App

Near-term (7-day) earthquake forecasting from geomagnetic (INTERMAGNET) sensor data,
fused through CNN + GNN + Transformer sub-models, surfaced to the public through a
simple cross-platform app (web first, Android later via Capacitor).

> **Honest disclaimer.** Near-term earthquake prediction from geomagnetic precursors is
> scientifically contested. This is a research prototype. Public-facing claims are
> deliberately hedged and validation is conservative.

---

## Repository layout

```
Earthquake/
├── src/eq/              # Python package
│   ├── config.py        # LOCKED labeling / preprocessing / graph / split rules (single source of truth)
│   ├── pipeline.py      # deterministic raw -> labeled 24x27 matrices -> chronological splits
│   ├── graph.py         # station adjacency (haversine + Gaussian kernel)
│   ├── synthetic.py     # synthetic INTERMAGNET + quake catalog (dev/test fixture with learnable signal)
│   ├── data/            # REAL data acquisition (Phase 3)
│   │   ├── usgs.py        # USGS FDSN earthquake catalog
│   │   ├── intermagnet.py # INTERMAGNET GIN web service + IAGA-2002 parser
│   │   └── real.py        # western-US cluster -> readings/catalog (relaxed proximity/graph rules)
│   └── models/          # CNN, GNN, Transformer, fusion + data/metrics/train/evaluate (Phase 4-5)
├── tests/               # pytest suite (run: pytest)
├── scripts/             # build_dataset.py, train_models.py (synthetic), eval_real.py (real), make_forecast.py
├── app/                 # React + Vite web app (Capacitor-ready for Android)
├── progress.html        # standalone build-progress dashboard (light/dark)
├── data/raw/            # raw INTERMAGNET / USGS, cached (gitignored)
└── data/processed/      # materialized datasets (gitignored)
```

## Build phases (each is independently testable)

| Phase | Deliverable | Status | How to test |
|------|-------------|--------|-------------|
| 1 | Data pipeline + synthetic fixture | ✅ | `pytest` |
| 2 | Web app surface (reads `forecast.json`) | ✅ | `cd app && npm run dev` |
| 4 | CNN, GNN, Transformer, fusion | ✅ | overfit-one-batch tests; `python scripts/train_models.py` |
| 3 | Real INTERMAGNET + USGS behind same interface | ✅ | `python scripts/eval_real.py` (fetches + caches real data) |
| 5 | Evaluation vs 60% baseline (recall-focused, CIs) | 🔄 | repeated-seed mean ± 95% CI in `eval_real.py` |
| 6 | Wire real inference into app + figures for Dr. Sanders | ⬜ | end-to-end run |

> Real observatories are sparse, so the real cluster (Fresno, Tucson, Boulder, Newport)
> **relaxes** the locked SoCal-cluster rules: `proximity_km` is swept over {500, 700, 1000} km
> and graph distances are widened so the far-apart stations stay connected. See `src/eq/data/real.py`.

---

## Quick start

### Python pipeline (Phase 1)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q                 # run the test suite
.venv/Scripts/python scripts/build_dataset.py     # materialize a (synthetic) dataset
.venv/Scripts/python scripts/make_forecast.py     # emit app/public/forecast.json (placeholder until models exist)
```

### Models & real data (Phase 3-5)

```bash
.venv/Scripts/python -m pip install -r requirements-ml.txt  # torch, scikit-learn, matplotlib
.venv/Scripts/python scripts/train_models.py   # train CNN/GNN/Transformer/fusion on synthetic (with CIs)
.venv/Scripts/python scripts/eval_real.py      # fetch real INTERMAGNET+USGS, sweep proximity, report CIs
```

The first `eval_real.py` run downloads ~3 years of minute data for 4 observatories (cached
under `data/raw/`, resumable). No credentials needed — INTERMAGNET adj-or-rep data and the
USGS catalog are public.

### Web app (Phase 2)

```bash
cd app
npm install
npm run dev        # open the printed http://localhost:5173
```

The web app fetches `forecast.json` and renders the 7-day status. The same build later
becomes an Android app via Capacitor (`npx cap add android`) with no code changes.

---

## Locked rules (MVP §3.1) — see `src/eq/config.py`

- **Magnitude threshold:** M ≥ 5.0 counts as a target event.
- **Proximity:** within 300 km of *any* cluster station.
- **Horizon:** earthquake in the next **7 days**.
- **Matrix:** trailing **27 days × 24 hours** of hourly-averaged field, per station.
- **Splits:** chronological 70/15/15 with a **34-day embargo** (= 27 + 7) between splits
  so no input/label window straddles a split boundary (leakage-safe).
- **Graph:** stations within 500 km connected, edge weight `exp(-d²/2σ²)`, σ = 300 km.

## Forecast contract (`app/public/forecast.json`)

```json
{
  "schema_version": 1,
  "region": "string",
  "generated_at": "ISO-8601 UTC",
  "horizon_days": 7,
  "probability": 0.0,
  "label": "likely | unlikely",
  "confidence": "low | medium | high",
  "is_placeholder": true,
  "model_version": "string",
  "stations": ["CODE", "..."],
  "disclaimer": "string"
}
```
