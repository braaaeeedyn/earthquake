"""GNN sub-model (MVP §4): inter-station relationships over the geographic graph.

Hand-rolled GCN (plain PyTorch, not torch_geometric — avoids Windows install pain for
a 3-5 node graph). Each station's 24x27 matrix is flattened and encoded into a node
feature, then two GCN layers propagate information across the normalized adjacency, and
a mean over stations yields the embedding. The adjacency is a registered buffer so the
forward signature stays uniform with the other sub-models.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """H' = A_hat @ (H W). A_hat is the precomputed symmetric-normalized adjacency."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)  adj: (S, S)
        return torch.einsum("ij,bjf->bif", adj, self.lin(x))


class StationGNN(nn.Module):
    def __init__(self, n_stations: int, adj: torch.Tensor, n_features: int = 1, hours: int = 24,
                 days: int = 27, hidden: int = 32, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer("adj", adj)
        self.encode = nn.Linear(n_features * hours * days, hidden)
        self.gcn1 = GCNLayer(hidden, hidden)
        self.gcn2 = GCNLayer(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:  # (B, S, 24, 27) -> add feature axis
            x = x.unsqueeze(2)
        b, s = x.shape[0], x.shape[1]
        z = x.reshape(b, s, -1)  # per-station node feature: F*24*27
        z = self.drop(torch.relu(self.encode(z)))
        z = self.drop(torch.relu(self.gcn1(z, self.adj)))
        z = torch.relu(self.gcn2(z, self.adj))
        return z.mean(dim=1)  # pool over stations -> (B, out_dim)
