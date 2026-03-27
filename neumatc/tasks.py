from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch



@dataclass
class ParametricTask:
    name: str
    matrix_fn: Callable[[torch.Tensor], torch.Tensor]
    target_fn: Callable[[torch.Tensor], List[torch.Tensor]]
    residual_fn: Callable[[torch.Tensor, List[torch.Tensor]], torch.Tensor]
    output_shapes: Sequence[Tuple[int, int]]


def make_spd_matrix(p: torch.Tensor, n: int = 16, device: str = "cpu") -> torch.Tensor:
    p = p.reshape(-1, 1, 1)
    idx = torch.arange(n, device=device, dtype=torch.float32)
    grid = idx[:, None] - idx[None, :]
    base = torch.exp(-0.15 * grid.abs())[None, :, :]
    eye = torch.eye(n, device=device).reshape(1, n, n)
    # smooth continuous variation with parameter p
    return base + (0.1 + p) * eye + 0.03 * torch.sin((idx[None, :, None] + 1) * p)


def inversion_task(n: int = 16, device: str = "cpu") -> ParametricTask:
    def matrix_fn(p: torch.Tensor) -> torch.Tensor:
        return make_spd_matrix(p, n=n, device=device)

    def target_fn(p: torch.Tensor) -> List[torch.Tensor]:
        A = matrix_fn(p)
        return [torch.linalg.inv(A)]

    def residual_fn(p: torch.Tensor, outputs: List[torch.Tensor]) -> torch.Tensor:
        A = matrix_fn(p)
        G = outputs[0]
        I = torch.eye(n, device=A.device).unsqueeze(0).expand_as(A)
        return A @ G - I

    return ParametricTask(
        name="inversion",
        matrix_fn=matrix_fn,
        target_fn=target_fn,
        residual_fn=residual_fn,
        output_shapes=[(n, n)],
    )


def svd_task(n: int = 16, device: str = "cpu") -> ParametricTask:
    def matrix_fn(p: torch.Tensor) -> torch.Tensor:
        A = make_spd_matrix(p, n=n, device=device)
        # Make the task non-trivial (not always symmetric diagonalizable with same vectors).
        rot = torch.eye(n, device=device)
        rot[0, 1] = 0.15
        rot[1, 0] = -0.15
        return A @ rot

    def target_fn(p: torch.Tensor) -> List[torch.Tensor]:
        A = matrix_fn(p)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        Sigma = torch.diag_embed(S)
        return [U, Sigma, Vh.transpose(-1, -2)]

    def residual_fn(p: torch.Tensor, outputs: List[torch.Tensor]) -> torch.Tensor:
        A = matrix_fn(p)
        U, Sigma, V = outputs
        recon = U @ Sigma @ V.transpose(-1, -2)
        I = torch.eye(A.shape[-1], device=A.device).unsqueeze(0)
        ortho_u = U.transpose(-1, -2) @ U - I
        ortho_v = V.transpose(-1, -2) @ V - I
        return torch.cat([
            (recon - A).reshape(A.shape[0], -1),
            ortho_u.reshape(A.shape[0], -1),
            ortho_v.reshape(A.shape[0], -1),
        ], dim=1)

    return ParametricTask(
        name="svd",
        matrix_fn=matrix_fn,
        target_fn=target_fn,
        residual_fn=residual_fn,
        output_shapes=[(n, n), (n, n), (n, n)],
    )


def relative_error(pred: List[torch.Tensor], target: List[torch.Tensor]) -> float:
    num = 0.0
    den = 0.0
    for p, t in zip(pred, target):
        num += torch.norm(p - t).item()
        den += torch.norm(t).item() + 1e-12
    return num / den
