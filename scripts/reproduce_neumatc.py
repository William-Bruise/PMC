#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from neumatc.data import ensure_dataset
from neumatc.tasks import inversion_task, svd_task
from neumatc.train import TrainConfig, evaluate, train_neumatc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce NeuMatC on synthetic parametric matrix tasks.")
    p.add_argument("--task", choices=["inversion", "svd"], default="inversion")
    p.add_argument("--n", type=int, default=16, help="Matrix size")
    p.add_argument("--train-samples", type=int, default=60)
    p.add_argument("--test-samples", type=int, default=100)
    p.add_argument("--collocation", type=int, default=24)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--force-regenerate-data", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    task = inversion_task(n=args.n, device=device) if args.task == "inversion" else svd_task(n=args.n, device=device)

    dataset_path = (
        Path(args.data_dir)
        / f"{args.task}_n{args.n}_tr{args.train_samples}_te{args.test_samples}_col{args.collocation}_seed{args.seed}.pt"
    )

    dataset = ensure_dataset(
        path=dataset_path,
        task=task,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        collocation_samples=args.collocation,
        seed=args.seed,
        force_regenerate=args.force_regenerate_data,
        device=device,
    )

    cfg = TrainConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
    )

    t0 = time.perf_counter()
    model = train_neumatc(
        task,
        dataset.p_train,
        dataset.p_collocation,
        cfg,
        device=device,
        targets_train=dataset.targets_train,
    )
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    metrics = evaluate(
        task,
        model,
        dataset.p_test,
        device=device,
        targets_test=dataset.targets_test,
    )
    infer_ms = 1000.0 * (time.perf_counter() - t1)

    out = {
        "task": args.task,
        "dataset_path": str(dataset_path),
        "matrix_size": args.n,
        "train_seconds": train_sec,
        "test_eval_milliseconds": infer_ms,
        **metrics,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
