#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import torch

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
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    task = inversion_task(n=args.n, device=device) if args.task == "inversion" else svd_task(n=args.n, device=device)

    p_train = torch.rand(args.train_samples)
    p_colloc = torch.rand(args.collocation)
    p_test = torch.linspace(0, 1, args.test_samples)

    cfg = TrainConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
    )

    t0 = time.perf_counter()
    model = train_neumatc(task, p_train, p_colloc, cfg, device=device)
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    metrics = evaluate(task, model, p_test, device=device)
    infer_ms = 1000.0 * (time.perf_counter() - t1)

    out = {
        "task": args.task,
        "matrix_size": args.n,
        "train_seconds": train_sec,
        "test_eval_milliseconds": infer_ms,
        **metrics,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
