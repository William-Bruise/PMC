from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn


class ParameterEncoder(nn.Module):
    """MLP \Phi_\theta(p) mapping scalar parameter p -> latent coefficients."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        layers: List[nn.Module] = [nn.Linear(1, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        if p.ndim == 1:
            p = p.unsqueeze(-1)
        return self.net(p)


class TensorHead(nn.Module):
    """Compute G(p) = C x_3 Phi(p) where C in R^{n1 x n2 x d}."""

    def __init__(self, out_shape: Tuple[int, int], latent_dim: int):
        super().__init__()
        self.out_shape = out_shape
        self.latent = nn.Parameter(torch.randn(*out_shape, latent_dim) * 0.02)

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        # coeff: [B, d] -> output: [B, n1, n2]
        return torch.einsum("ijd,bd->bij", self.latent, coeff)


class NeuMatC(nn.Module):
    """General NeuMatC implementation with m matrix-valued output components."""

    def __init__(
        self,
        output_shapes: Sequence[Tuple[int, int]],
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.encoder = ParameterEncoder(latent_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.heads = nn.ModuleList(TensorHead(shape, latent_dim) for shape in output_shapes)

    def forward(self, p: torch.Tensor) -> List[torch.Tensor]:
        coeff = self.encoder(p)
        return [head(coeff) for head in self.heads]


@dataclass
class TrainBatch:
    p: torch.Tensor
    targets: List[torch.Tensor]
