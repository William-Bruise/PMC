from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from neumatc.model import NeuMatC
from neumatc.tasks import ParametricTask, relative_error


@dataclass
class TrainConfig:
    latent_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 3
    lr: float = 1e-3
    steps: int = 800
    lambda_residual: float = 1.0
    residual_threshold: float = 1e-3
    failure_tolerance: float = 0.03
    adaptive_every: int = 100
    n_add: int = 8


def train_neumatc(
    task: ParametricTask,
    p_train: torch.Tensor,
    p_collocation: torch.Tensor,
    cfg: TrainConfig,
    device: str = "cpu",
    targets_train: List[torch.Tensor] | None = None,
) -> NeuMatC:
    model = NeuMatC(
        output_shapes=task.output_shapes,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    p_train = p_train.to(device)
    p_collocation = p_collocation.to(device)
    if targets_train is None:
        targets_train = [t.to(device) for t in task.target_fn(p_train)]
    else:
        targets_train = [t.to(device) for t in targets_train]

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for step in range(cfg.steps):
        model.train()
        optimizer.zero_grad()

        pred_train = model(p_train)
        sup_loss = sum(nn.functional.mse_loss(p, t) for p, t in zip(pred_train, targets_train))

        pred_col = model(p_collocation)
        residual = task.residual_fn(p_collocation, pred_col)
        res_loss = (residual**2).mean()

        loss = sup_loss + cfg.lambda_residual * res_loss
        loss.backward()
        optimizer.step()

        if step > 0 and step % cfg.adaptive_every == 0:
            with torch.no_grad():
                candidates = torch.linspace(0, 1, 400, device=device)
                r = task.residual_fn(candidates, model(candidates))
                r_score = (r**2).mean(dim=tuple(range(1, r.ndim))) if r.ndim > 2 else (r**2).mean(dim=1)
                failures = r_score > cfg.residual_threshold
                failure_prob = failures.float().mean().item()
                if failure_prob < cfg.failure_tolerance:
                    continue
                idx = torch.topk(r_score, k=min(cfg.n_add, candidates.numel())).indices
                p_collocation = torch.unique(torch.cat([p_collocation, candidates[idx]]))

    return model


def evaluate(
    task: ParametricTask,
    model: NeuMatC,
    p_test: torch.Tensor,
    device: str = "cpu",
    targets_test: List[torch.Tensor] | None = None,
) -> Dict[str, float]:
    with torch.no_grad():
        p_test = p_test.to(device)
        pred = model(p_test)
        if targets_test is None:
            target = [t.to(device) for t in task.target_fn(p_test)]
        else:
            target = [t.to(device) for t in targets_test]
        relerr = relative_error(pred, target)
        residual = task.residual_fn(p_test, pred)
        res = (residual**2).mean().sqrt().item()
    return {"relative_error": relerr, "residual_rmse": res}
