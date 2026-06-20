"""Generic training / inference loop shared by every model (sub-models and fusion).

Any module with signature ``forward(x) -> logit (B,)`` trains here, which is why the
sub-models expose a uniform interface. Loss is BCEWithLogits with optional ``pos_weight``
for the rare-event imbalance.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .metrics import best_threshold, binary_metrics


def train_model(model, train_loader, *, epochs: int = 30, lr: float = 1e-3,
                pos_weight: torch.Tensor | None = None, device: str = "cpu",
                weight_decay: float = 0.0, val_loader=None, select_beta: float = 2.0):
    """Train with BCEWithLogits (optional ``pos_weight`` for imbalance).

    If ``val_loader`` is given, keep the weights from the epoch with the best validation
    F-beta (``select_beta`` favors recall) instead of the last epoch. Returns the model in
    eval mode so dropout is off for inference.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )
    best_score, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if val_loader is not None:
            yv, pv = predict_proba(model, val_loader, device)
            m = binary_metrics(yv, pv, best_threshold(yv, pv, beta=select_beta))
            score = m["f2"] if select_beta >= 2 else m["f1"]
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, loader, device: str = "cpu"):
    """Return ``(y_true, prob)`` as numpy arrays over the whole loader."""
    model.to(device)
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        p = torch.sigmoid(model(xb.to(device)))
        ps.append(p.cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)
