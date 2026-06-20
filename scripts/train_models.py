"""End-to-end Phase 4-5 run: train each sub-model + the fusion, evaluate vs baseline.

Trains CNN, GNN, Transformer standalone and then the fused classifier on the (currently
synthetic) chronological dataset, selects each decision threshold on validation, and
reports precision/recall/F1 on the held-out test split against the ~60% accuracy baseline
from Liu et al. (2022). Run `python scripts/build_dataset.py` first.

------------------------------------------------------------------------------------------
GATE (read before running): deep-model work is PAUSED pending a feasibility check.
Run `python feasibility.py` first. It tests — with simple features under an honest
time-ordered split and a storm (Kp) confounder control — whether any out-of-sample signal
exists at all. The harness writes ./out/feasibility_report.txt with a GO / NO-GO verdict.
Only resume training here if that report says GO, and if so switch the headline metrics to
out-of-sample AUC / recall / FAR / precision / MCC on the SAME time-ordered split (never a
shuffle split), with the harness's best OOS AUC/MCC as the baseline this model must beat.
------------------------------------------------------------------------------------------
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eq.models import (  # noqa: E402
    Classifier,
    FusedClassifier,
    SpatialCNN,
    StationGNN,
    TemporalTransformer,
    adjacency_tensor,
    best_threshold,
    binary_metrics,
    load_dataset,
    make_loaders,
    pos_weight,
    predict_proba,
    train_model,
)
from eq.models.evaluate import format_row, repeated_evaluate  # noqa: E402

BASELINE_ACC = 0.60
EPOCHS = 80
SEEDS = (0, 1, 2, 3, 4)
SELECT_BETA = 2.0  # recall-weighted: threshold + epoch selection favor recall on the positive class


def _build_backbones(n_stations: int, adj):
    # Fresh instances per use so standalone training never leaks weights into fusion.
    return {
        "CNN": lambda: SpatialCNN(n_stations),
        "GNN": lambda: StationGNN(n_stations, adj),
        "Transformer": lambda: TemporalTransformer(n_stations),
    }


def _evaluate(model, loaders, name: str, pw):
    train_model(model, loaders["train"], epochs=EPOCHS, lr=1e-3, pos_weight=pw,
                weight_decay=1e-4, val_loader=loaders["val"], select_beta=SELECT_BETA)
    yv, pv = predict_proba(model, loaders["val"])
    t = best_threshold(yv, pv, beta=SELECT_BETA)   # operating point chosen on val only
    yt, pt = predict_proba(model, loaders["test"])
    m = binary_metrics(yt, pt, t)
    print(f"  {name:<12} thr={t:0.2f}  P={m['precision']:0.3f}  "
          f"R={m['recall']:0.3f}  F1={m['f1']:0.3f}  F2={m['f2']:0.3f}  "
          f"acc={m['accuracy']:0.3f}  (n={m['n']}, pos={m['positives']})")
    return m


def main() -> None:
    torch.manual_seed(0)
    data = load_dataset()
    loaders = make_loaders(data, batch_size=32)
    n_stations = data["train"][0].shape[1]
    adj = adjacency_tensor()
    pw = pos_weight(data["train"][1])

    print(f"Train n={len(data['train'][1])}  Val n={len(data['val'][1])}  "
          f"Test n={len(data['test'][1])}  pos_weight={pw.item():0.2f}")
    print(f"Baseline to beat: {BASELINE_ACC:.0%} accuracy (Liu et al. 2022)\n")
    print("Sub-models (standalone):")

    backbones = _build_backbones(n_stations, adj)
    for name, make in backbones.items():
        _evaluate(Classifier(make()), loaders, name, pw)

    print(f"\nFusion (CNN + GNN + Transformer) — {len(SEEDS)} seeds, mean ± 95% CI:")
    make_fused = lambda: FusedClassifier([
        SpatialCNN(n_stations), StationGNN(n_stations, adj), TemporalTransformer(n_stations)
    ])
    agg = repeated_evaluate(make_fused, loaders, seeds=SEEDS, pos_weight=pw, epochs=EPOCHS)
    print(format_row("Fused", agg))

    verdict = "BEATS" if agg["accuracy"]["mean"] > BASELINE_ACC else "does NOT beat"
    print(f"\nFused {verdict} the {BASELINE_ACC:.0%} accuracy baseline "
          f"(synthetic data — not a scientific result).")


if __name__ == "__main__":
    main()
