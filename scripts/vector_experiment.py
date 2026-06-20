"""Lever 1: vector / Z-H polarization features on top of common-mode subtraction.

Per trustworthy cell (large positive count), build common-mode-residual H/Z/F components,
compute [ULF(H), ULF(Z), log Z/H, |dF/dt|], and evaluate each backbone + fusion. Reports the
HONEST metric set — ROC-AUC, precision, recall, F1, accuracy, balanced accuracy — next to the
majority-class baseline, so no accuracy gain can hide behind class imbalance.

  python scripts/vector_experiment.py --seeds 3 --epochs 20
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eq.config import Config  # noqa: E402
from eq.data.real import FULL_START, build_region_components, region_config  # noqa: E402
from eq.features import (  # noqa: E402
    BAND_MINUTE, POLAR_FEATURE_NAMES, assemble_feature_samples, vector_feature_grid,
)
from eq.models import (  # noqa: E402
    Classifier, FusedClassifier, SpatialCNN, StationGNN, TemporalTransformer,
    adjacency_tensor, pos_weight, predict_proba, train_model,
)
from eq.models.metrics import auc_metrics, best_threshold, binary_metrics  # noqa: E402
import grid_experiment as ge  # noqa: E402

CELLS = [("japan", 500.0, 5.0), ("japan", 300.0, 5.0), ("california", 500.0, 4.5)]


def common_mode_residual_components(components: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Remove the cross-station common-mode per component (H, Z, F)."""
    out: dict[str, dict] = {c: {} for c in components}
    for col in ("H", "Z", "F"):
        df = pd.DataFrame({c: components[c][col] for c in components}).astype("float32")
        dev = df.sub(df.median())
        resid = dev.sub(dev.mean(axis=1, skipna=True), axis=0)
        for c in components:
            out[c][col] = resid[c]
    return {c: pd.DataFrame(out[c]).dropna(how="all") for c in components}


def _ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    return float(vals.mean()), 1.96 * std / math.sqrt(n) if n > 1 else 0.0


def evaluate_full(make, loaders, seeds, pos_weight_t, epochs):
    """Train across seeds; return mean±CI for roc/precision/recall/f1/accuracy/balanced_acc, plus
    the Brier skill score vs. a climatology (base-rate) forecast — the probabilistic-forecast metric."""
    acc = {k: [] for k in ("roc", "precision", "recall", "f1", "accuracy", "bal_acc", "brier_ss")}
    base_rate = None
    for s in seeds:
        torch.manual_seed(s)
        model = make()
        train_model(model, loaders["train"], epochs=epochs, lr=1e-3, pos_weight=pos_weight_t,
                    weight_decay=1e-4, val_loader=loaders["val"], select_beta=2.0)
        yv, pv = predict_proba(model, loaders["val"])
        t = best_threshold(yv, pv, beta=2.0)
        yt, pt = predict_proba(model, loaders["test"])
        bm, am = binary_metrics(yt, pt, t), auc_metrics(yt, pt)
        pred = (np.asarray(pt) >= t).astype(int)
        y = np.asarray(yt).astype(int); p = np.asarray(pt, dtype=float)
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        base_rate = am["base_rate"]
        brier = np.mean((p - y) ** 2)
        brier_clim = base_rate * (1 - base_rate)        # always-forecast-base-rate
        bss = 1 - brier / brier_clim if brier_clim > 0 else float("nan")
        acc["roc"].append(am["roc_auc"]); acc["precision"].append(bm["precision"])
        acc["recall"].append(bm["recall"]); acc["f1"].append(bm["f1"])
        acc["accuracy"].append(bm["accuracy"]); acc["bal_acc"].append((bm["recall"] + spec) / 2)
        acc["brier_ss"].append(bss)
    out = {k: _ci(v) for k, v in acc.items()}
    out["base_rate"] = base_rate
    out["majority_acc"] = max(base_rate, 1 - base_rate)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "processed" / "vector_results.json"))
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    base = Config()

    results, cache = [], {}
    for region, prox, mag in CELLS:
        if region not in cache:
            comps, catalog, n_days, stations = build_region_components(region, min_magnitude=4.5)
            res = common_mode_residual_components(comps)
            cfg0 = region_config(base, stations, 300.0, 5.0, FULL_START, n_days)
            grids = vector_feature_grid(res, cfg0, BAND_MINUTE, samples_per_hour=60)
            cache[region] = (grids, catalog, n_days, stations, adjacency_tensor(cfg0))
        grids, catalog, n_days, stations, adj = cache[region]
        n_st, n_f = len(stations), len(POLAR_FEATURE_NAMES)

        cfg = region_config(base, stations, prox, mag, FULL_START, n_days)
        X, y, anchors = assemble_feature_samples(grids, catalog, cfg)
        loaders, y_tr = ge._loaders_for_cell(X, y, anchors, cfg)
        pw = pos_weight(torch.tensor(y_tr, dtype=torch.float32))

        makers = {
            "CNN": lambda: Classifier(SpatialCNN(n_st, n_f)),
            "GNN": lambda: Classifier(StationGNN(n_st, adj, n_f)),
            "Transformer": lambda: Classifier(TemporalTransformer(n_st, n_f)),
            "Fused": lambda: FusedClassifier([
                SpatialCNN(n_st, n_f), StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)]),
        }
        print(f"\n=== {region} M{mag} {prox:.0f}km — vector features (n+={int(y.sum())} total) ===",
              flush=True)
        for name, make in makers.items():
            m = evaluate_full(make, loaders, seeds, pw, args.epochs)
            row = {"region": region, "magnitude": mag, "proximity_km": prox, "model": name,
                   "base_rate": m["base_rate"], "majority_acc": m["majority_acc"],
                   **{k: {"mean": m[k][0], "ci": m[k][1]} for k in
                      ("roc", "precision", "recall", "f1", "accuracy", "bal_acc")}}
            results.append(row)
            print(f"  {name:<12} ROC={m['roc'][0]:.3f}±{m['roc'][1]:.3f}  "
                  f"bal_acc={m['bal_acc'][0]:.3f}  acc={m['accuracy'][0]:.3f} "
                  f"(maj {m['majority_acc']:.3f})  R={m['recall'][0]:.3f} P={m['precision'][0]:.3f} "
                  f"F1={m['f1'][0]:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"seeds": args.seeds, "epochs": args.epochs,
                                          "results": results}, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
