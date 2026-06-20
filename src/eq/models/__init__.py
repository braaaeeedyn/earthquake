"""Model flow (MVP §4): CNN + GNN + Transformer sub-models fused into one classifier.

Design: every sub-model is a *feature extractor* exposing a uniform interface
``forward(x) -> (B, out_dim)`` and an ``out_dim`` attribute. ``Classifier`` wraps any
one backbone with a linear head for standalone training / overfit tests;
``FusedClassifier`` reuses the same backbones via late fusion. This keeps each model
independently testable AND fusible with no rewrite.
"""
from .cnn import SpatialCNN
from .gnn import StationGNN
from .transformer import TemporalTransformer
from .fusion import Classifier, FusedClassifier
from .data import load_dataset, make_loaders, adjacency_tensor, pos_weight
from .metrics import binary_metrics, best_threshold
from .train import train_model, predict_proba

__all__ = [
    "SpatialCNN",
    "StationGNN",
    "TemporalTransformer",
    "Classifier",
    "FusedClassifier",
    "load_dataset",
    "make_loaders",
    "adjacency_tensor",
    "pos_weight",
    "binary_metrics",
    "best_threshold",
    "train_model",
    "predict_proba",
]
