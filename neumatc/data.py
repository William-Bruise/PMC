from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from neumatc.tasks import ParametricTask


@dataclass
class TaskDataset:
    p_train: torch.Tensor
    p_collocation: torch.Tensor
    p_test: torch.Tensor
    targets_train: List[torch.Tensor]
    targets_test: List[torch.Tensor]


def generate_dataset(
    task: ParametricTask,
    train_samples: int,
    test_samples: int,
    collocation_samples: int,
    seed: int,
    device: str = "cpu",
) -> TaskDataset:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    p_train = torch.rand(train_samples, generator=g, device=device)
    p_collocation = torch.rand(collocation_samples, generator=g, device=device)
    p_test = torch.linspace(0, 1, test_samples, device=device)

    with torch.no_grad():
        targets_train = [t.detach().cpu() for t in task.target_fn(p_train)]
        targets_test = [t.detach().cpu() for t in task.target_fn(p_test)]

    return TaskDataset(
        p_train=p_train.detach().cpu(),
        p_collocation=p_collocation.detach().cpu(),
        p_test=p_test.detach().cpu(),
        targets_train=targets_train,
        targets_test=targets_test,
    )


def save_dataset(path: str | Path, dataset: TaskDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "p_train": dataset.p_train,
        "p_collocation": dataset.p_collocation,
        "p_test": dataset.p_test,
        "targets_train": dataset.targets_train,
        "targets_test": dataset.targets_test,
    }
    torch.save(payload, path)


def load_dataset(path: str | Path) -> TaskDataset:
    payload = torch.load(Path(path), map_location="cpu")
    return TaskDataset(
        p_train=payload["p_train"],
        p_collocation=payload["p_collocation"],
        p_test=payload["p_test"],
        targets_train=payload["targets_train"],
        targets_test=payload["targets_test"],
    )


def ensure_dataset(
    path: str | Path,
    task: ParametricTask,
    train_samples: int,
    test_samples: int,
    collocation_samples: int,
    seed: int,
    force_regenerate: bool = False,
    device: str = "cpu",
) -> TaskDataset:
    path = Path(path)
    if path.exists() and not force_regenerate:
        return load_dataset(path)

    dataset = generate_dataset(
        task=task,
        train_samples=train_samples,
        test_samples=test_samples,
        collocation_samples=collocation_samples,
        seed=seed,
        device=device,
    )
    save_dataset(path, dataset)
    return dataset
