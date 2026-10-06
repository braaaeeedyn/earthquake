
import numpy as np
import torch
import torch.nn as nn

from eq import network
from eq.models import DetectorNet, MultiStationModel, adjacency


def test_detector_overfits_tiny_batch():
    torch.manual_seed(0)
    x = torch.randn(16, 3000)
    x[:8, 1500:1800] += 6 * torch.sin(torch.arange(300) * 0.3)
    y = torch.tensor([1.0] * 8 + [0.0] * 8)
    m = DetectorNet()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(60):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.2


def test_magnitude_model_overfits_tiny_batch():
    torch.manual_seed(0)
    S = len(network.CODES)
    m = MultiStationModel(adjacency(network.COORDS.astype(np.float32)), hybrid=True, amp_feature=True)
    X = torch.randn(6, S, 3, 3000)
    M = torch.zeros(6, S, dtype=torch.bool)
    M[:, :4] = True
    L = torch.rand(6, S)
    A = torch.randn(6, 4)
    y = torch.tensor([3.0, 3.5, 4.0, 4.5, 5.0, 3.2])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad()
        loss = nn.functional.mse_loss(m(X, M, L, A, torch.randn(6, S)), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
