from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    def __init__(self, dim: int, rank: int = 16, alpha: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)

        self.dropout = nn.Dropout(dropout)
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.B(self.A(self.dropout(x))) * self.scaling
        return x + update
