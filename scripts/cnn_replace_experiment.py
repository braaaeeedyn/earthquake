"""Lever 2: the CNN is the weak link — replace it with a different backbone.

In the best config (scalar common-mode features), compare late-fusion with the original CNN,
with a statistics-pooling MLP in its place, and with the CNN dropped entirely. A different
inductive bias (global distribution + recency trend, not local hour x day convolutions) is
the point. Honest metric set, trustworthy cells only.

  python scripts/cnn_replace_experiment.py --seeds 3 --epochs 20
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eq.config import Config  # noqa: E402
from eq.data.real import FULL_START, build_region_inputs, region_config  # noqa: E402
from eq.features import BAND_MINUTE, FEATURE_NAMES, assemble_feature_samples, feature_grid  # noqa: E402
from eq.models import (  # noqa: E402
    Classifier, FusedClassifier, SpatialCNN, StationGNN, TemporalTransformer,
    adjacency_tensor, pos_weight,
)
import grid_experiment as ge  # noqa: E402
from methodology_experiments import common_mode_residual  # noqa: E402
from vector_experiment import evaluate_full  # noqa: E402

CELLS = [("japan", 500.0, 5.0), ("japan", 300.0, 5.0), ("california", 500.0, 4.5)]


class StatPoolMLP(nn.Module):
    """Replacement for the 2D CNN: per (station, feature) summary stats over the 24x27 field
    (mean, std, max, recency trend) -> MLP. A global/trend inductive bias, not local convs."""

    def __init__(self, n_stations: int, n_features: int = 1, out_dim: int = 32,
                 dropout: float = 0.2, recent_days: int = 5):
        super().__init__()
        self.out_dim = out_dim
        self.recent_days = recent_days
        self.net = nn.Sequential(
            nn.Linear(n_stations * n_features * 4, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:                       # (B,S,24,27) -> (B,S,1,24,27)
            x = x.unsqueeze(2)
        b, s, f, h, d = x.shape
        flat = x.reshape(b, s, f, h * d)
        mean, std, mx = flat.mean(-1), flat.std(-1), flat.amax(-1)
        k = min(self.recent_days, d - 1) if d > 1 else d
        trend = x[..., -k:].mean(dim=(-2, -1)) - x[..., :-k].mean(dim=(-2, -1))
        feats = torch.stack([mean, std, mx, trend], dim=-1)   # (B,S,F,4)
        return self.net(feats.reshape(b, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "processed" / "cnn_replace_results.json"))
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    base = Config()

    results, cache = [], {}
    for region, prox, mag in CELLS:
        if region not in cache:
            readings, catalog, n_days, stations = build_region_inputs(region, min_magnitude=4.5)
            res = common_mode_residual(readings)
            cfg0 = region_config(base, stations, 300.0, 5.0, FULL_START, n_days)
            grids = feature_grid(res, cfg0, BAND_MINUTE, samples_per_hour=60)
            cache[region] = (grids, catalog, n_days, stations, adjacency_tensor(cfg0))
        grids, catalog, n_days, stations, adj = cache[region]
        n_st, n_f = len(stations), len(FEATURE_NAMES)

        cfg = region_config(base, stations, prox, mag, FULL_START, n_days)
        X, y, anchors = assemble_feature_samples(grids, catalog, cfg)
        loaders, y_tr = ge._loaders_for_cell(X, y, anchors, cfg)
        pw = pos_weight(torch.tensor(y_tr, dtype=torch.float32))

        makers = {
            "Fused_CNN": lambda: FusedClassifier([
                SpatialCNN(n_st, n_f), StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)]),
            "Fused_MLP": lambda: FusedClassifier([
                StatPoolMLP(n_st, n_f), StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)]),
            "Fused_noCNN": lambda: FusedClassifier([
                StationGNN(n_st, adj, n_f), TemporalTransformer(n_st, n_f)]),
            "StatMLP_solo": lambda: Classifier(StatPoolMLP(n_st, n_f)),
        }
        print(f"\n=== {region} M{mag} {prox:.0f}km — scalar common-mode (n+={int(y.sum())}) ===",
              flush=True)
        for name, make in makers.items():
            m = evaluate_full(make, loaders, seeds, pw, args.epochs)
            results.append({"region": region, "magnitude": mag, "proximity_km": prox,
                            "model": name, "base_rate": m["base_rate"],
                            "majority_acc": m["majority_acc"],
                            **{k: {"mean": m[k][0], "ci": m[k][1]} for k in
                               ("roc", "precision", "recall", "f1", "accuracy", "bal_acc")}})
            print(f"  {name:<13} ROC={m['roc'][0]:.3f}±{m['roc'][1]:.3f}  "
                  f"bal_acc={m['bal_acc'][0]:.3f}  R={m['recall'][0]:.3f} "
                  f"P={m['precision'][0]:.3f} F1={m['f1'][0]:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"seeds": args.seeds, "epochs": args.epochs,
                                          "results": results}, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
