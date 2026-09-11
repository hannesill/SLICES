#!/usr/bin/env python3
"""Print a credentialed core or TS2Vec extension recipe; never launches training."""

import argparse
import shlex


def commands(dataset, objective, task, seed, protocol):
    out = f"outputs/reproduction/{dataset}/{objective}/seed{seed}"
    common = [
        f"dataset={dataset}",
        f"seed={seed}",
        "logging.use_wandb=false",
        "data.num_workers=0",
        "training.accelerator=cpu",
        "training.devices=1",
        "experiment_class="
        + ("ts2vec_extension" if objective == "ts2vec" else "core_ssl_benchmark"),
        "revision=" + ("extension-v1" if objective == "ts2vec" else "reproduction-v1"),
    ]
    pre = [
        "uv",
        "run",
        "python",
        "scripts/training/pretrain.py",
        f"ssl={objective}",
        *common,
        f"hydra.run.dir={out}/pretrain",
    ]
    downstream = [
        "uv",
        "run",
        "python",
        "scripts/training/finetune.py",
        *common,
        f"paradigm={objective}",
        f"tasks={task}",
        f"protocol={protocol}",
        f"checkpoint={out}/pretrain/encoder.pt",
        f"hydra.run.dir={out}/{task}/{protocol}",
    ]
    return [pre, downstream]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("miiv", "eicu", "combined"), default="miiv")
    p.add_argument("--objective", choices=("mae", "jepa", "contrastive", "ts2vec"), default="mae")
    p.add_argument("--tree", choices=("historical", "framework"), default="historical")
    p.add_argument(
        "--task",
        choices=("mortality_24h", "mortality_hospital", "aki_kdigo", "los_remaining"),
        default="mortality_hospital",
    )
    p.add_argument("--seed", type=int, choices=(42, 123, 456, 789, 1011), default=42)
    p.add_argument("--protocol", choices=("linear_probe", "full_finetune"), default="linear_probe")
    a = p.parse_args()
    print(f"# Run from {a.tree}/ after credentialed preprocessing. This only prints commands.")
    for command in commands(a.dataset, a.objective, a.task, a.seed, a.protocol):
        print(shlex.join(command))


if __name__ == "__main__":
    main()
