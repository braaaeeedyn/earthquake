"""Phase 4 model tests: each sub-model (and the fusion) can overfit one batch.

"Overfit one batch" is the standard capacity/gradient-flow smoke test: a model with
working forward + backward passes should memorize a tiny random batch. It validates the
architecture wiring, not generalization. torch is an optional (Phase 4+) dependency, so
the whole module is skipped if it is not installed.
"""
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from eq.models import (  # noqa: E402
    Classifier,
    FusedClassifier,
    SpatialCNN,
    StationGNN,
    TemporalTransformer,
    adjacency_tensor,
    binary_metrics,
    best_threshold,
    train_model,
)

S = 5  # matches DEFAULT_STATIONS / the default adjacency


def _batch(n: int = 16):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, S, 24, 27, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float()
    return x, y


def _overfit_accuracy(model) -> float:
    torch.manual_seed(0)
    x, y = _batch()
    clf = Classifier(model)
    train_model(clf, [(x, y)], epochs=300, lr=1e-2)
    with torch.no_grad():
        pred = (torch.sigmoid(clf(x)) >= 0.5).float()
    return (pred == y).float().mean().item()


# dropout is disabled here: overfit-one-batch tests capacity/gradient flow, and a
# regularizer legitimately prevents memorizing a single batch.
def test_cnn_overfits_one_batch():
    assert _overfit_accuracy(SpatialCNN(S, dropout=0.0)) >= 0.95


def test_gnn_overfits_one_batch():
    assert _overfit_accuracy(StationGNN(S, adjacency_tensor(), dropout=0.0)) >= 0.95


def test_transformer_overfits_one_batch():
    assert _overfit_accuracy(TemporalTransformer(S, dropout=0.0)) >= 0.95


def test_fusion_overfits_one_batch():
    torch.manual_seed(0)
    x, y = _batch()
    model = FusedClassifier([
        SpatialCNN(S, dropout=0.0),
        StationGNN(S, adjacency_tensor(), dropout=0.0),
        TemporalTransformer(S, dropout=0.0),
    ])
    train_model(model, [(x, y)], epochs=300, lr=1e-2)
    with torch.no_grad():
        pred = (torch.sigmoid(model(x)) >= 0.5).float()
    assert (pred == y).float().mean().item() >= 0.95


def test_backbones_share_uniform_interface():
    """Every sub-model returns (B, out_dim) so fusion can concatenate them."""
    x, _ = _batch(4)
    for m in (SpatialCNN(S), StationGNN(S, adjacency_tensor()), TemporalTransformer(S)):
        out = m(x)
        assert out.shape == (4, m.out_dim)


def test_metrics_and_threshold():
    y = [0, 0, 1, 1]
    prob = [0.1, 0.4, 0.6, 0.9]
    m = binary_metrics(y, prob, threshold=0.5)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    # A threshold exists that perfectly separates these probabilities.
    t = best_threshold(y, prob)
    assert binary_metrics(y, prob, t)["f1"] == 1.0
