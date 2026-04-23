#!/usr/bin/env python3
"""Class-based experiment runner for the final SLICES thesis rerun corpus.

Generates the thesis experiment matrix by scientific experiment class, resolves
pretrain dependencies, and executes runs with crash recovery and resumable state.

Usage:
    uv run python scripts/internal/run_experiments.py warmup \
        --experiment-class core_ssl_benchmark label_efficiency
    uv run python scripts/internal/run_experiments.py run \
        --experiment-class core_ssl_benchmark label_efficiency \
        --project slices-thesis --revision thesis-v1 --entity <entity>
    uv run python scripts/internal/run_experiments.py run \
        --experiment-class core_ssl_benchmark --dry-run \
        --project slices-thesis --revision thesis-v1 --entity <entity>
    uv run python scripts/internal/run_experiments.py status \
        --experiment-class core_ssl_benchmark --revision thesis-v1
    uv run python scripts/internal/run_experiments.py retry --failed --skipped \
        --experiment-class core_ssl_benchmark --revision thesis-v1 \
        --project slices-thesis --entity <entity> --launch-commit <commit>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from slices.constants import THESIS_TASKS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_CLASSES = [
    "core_ssl_benchmark",
    "label_efficiency",
    "cross_dataset_transfer",
    "hp_robustness",
    "capacity_study",
    "classical_baselines",
    "ts2vec_extension",
    "smart_external_reference",
]

DEFAULT_EXPERIMENT_CLASSES = list(EXPERIMENT_CLASSES)

DOWNSTREAM_EXPERIMENT_CLASSES = [
    "core_ssl_benchmark",
    "label_efficiency",
    "cross_dataset_transfer",
    "hp_robustness",
    "capacity_study",
    "classical_baselines",
    "ts2vec_extension",
    "smart_external_reference",
]

EXPECTED_CLASS_COUNTS = {
    "core_ssl_benchmark": 465,
    "label_efficiency": 1155,
    "cross_dataset_transfer": 120,
    "hp_robustness": 150,
    "capacity_study": 100,
    "classical_baselines": 330,
    "ts2vec_extension": 135,
    "smart_external_reference": 135,
}

SSL_PARADIGMS = ["mae", "jepa", "contrastive"]
DATASETS = ["miiv", "eicu", "combined"]
TASKS = list(THESIS_TASKS)
SEEDS_EXTENDED = [42, 123, 456, 789, 1011]
LABEL_FRACTIONS_FULL = [0.01, 0.05, 0.1, 0.25, 0.5]
LABEL_FRACTIONS_MORTALITY_24H = [0.05, 0.1, 0.25, 0.5]
LABEL_FRACTIONS_TREND = [0.1]
LABEL_FRACTIONS_CAPACITY = [0.05, 0.1, 0.5]

MODEL_SIZES = {
    "medium": {
        "model": "transformer_medium",
        "ssl_scale": {
            "mae": {
                "ssl.decoder_d_model": 128,
                "ssl.decoder_n_layers": 2,
                "ssl.decoder_n_heads": 8,
                "ssl.decoder_d_ff": 512,
            },
        },
    },
    "large": {
        "model": "transformer_large",
        "ssl_scale": {
            "mae": {
                "ssl.decoder_d_model": 256,
                "ssl.decoder_n_layers": 2,
                "ssl.decoder_n_heads": 8,
                "ssl.decoder_d_ff": 1024,
            },
        },
    },
}

LR_ROBUSTNESS = [2e-4, 5e-4, 2e-3]
MASK_RATIO_ROBUSTNESS = [0.3, 0.75]
VIEW_MASK_EXPERIMENT_SUBTYPE = "view_mask_sensitivity"
TRANSFER_PAIRS = [("miiv", "eicu"), ("eicu", "miiv")]

STATE_FILE = Path("outputs/experiment_state.json")
LOG_DIR = Path("logs/runner")
LAUNCH_IDENTITY_FILE = ".runner_launch_identity.json"
LAUNCH_IDENTITY_KEYS = ("revision", "wandb_project", "wandb_entity", "launch_commit")

PROTO_A = {"freeze_encoder": True, "max_epochs": 50, "patience": 10, "lr": 1e-4}
PROTO_B = {"freeze_encoder": False, "max_epochs": 100, "patience": 10, "lr": 3e-4}


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


@dataclass
class Run:
    id: str
    experiment_class: str
    run_type: str  # "pretrain" | "finetune" | "supervised" | "gru_d" | "xgboost"
    paradigm: str
    dataset: str
    seed: int
    output_dir: str
    run_key: str | None = None
    depends_on: list[str] = field(default_factory=list)
    task: str | None = None
    label_fraction: float = 1.0
    freeze_encoder: bool | None = None
    extra_overrides: dict = field(default_factory=dict)
    source_dataset: str | None = None
    upstream_pretrain_lr: float | None = None
    upstream_pretrain_mask_ratio: float | None = None
    experiment_subtype: str | None = None
    model_size: str | None = None

    def __post_init__(self) -> None:
        if self.run_key is None:
            self.run_key = self.id

    @property
    def phase(self) -> str:
        if self.run_type == "pretrain":
            return "pretrain"
        if self.run_type == "supervised":
            return "supervised"
        if self.run_type in {"gru_d", "xgboost"}:
            return "baseline"
        return "finetune"

    @property
    def protocol(self) -> str | None:
        if self.freeze_encoder is True:
            return "A"
        if self.freeze_encoder is False or self.run_type in {"supervised", "gru_d", "xgboost"}:
            return "B"
        return None

    @property
    def protocol_selector(self) -> str | None:
        protocol = self.protocol
        return protocol.lower() if protocol is not None else None

    def build_command(self, runs_by_id: dict[str, "Run"]) -> list[str]:
        """Build the subprocess command for this run."""
        if self.run_type == "pretrain":
            return self._pretrain_cmd()
        if self.run_type == "finetune":
            return self._finetune_cmd(runs_by_id)
        if self.run_type == "supervised":
            return self._supervised_cmd()
        if self.run_type == "gru_d":
            return self._gru_d_cmd()
        if self.run_type == "xgboost":
            return self._xgboost_cmd()
        raise ValueError(f"Unknown run_type: {self.run_type}")

    def _metadata_overrides(self) -> list[str]:
        overrides = [
            f"experiment_class={self.experiment_class}",
            f"+phase={self.phase}",
        ]
        if self.experiment_subtype is not None:
            overrides.append(f"experiment_subtype={self.experiment_subtype}")
        if self.protocol_selector is not None and self.run_type == "finetune":
            overrides.append(f"protocol={self.protocol_selector}")
        if self.model_size is not None:
            overrides.append(f"+model_size={self.model_size}")
        for k, v in self.extra_overrides.items():
            overrides.append(f"{k}={v}")
        return overrides

    def _append_resume(self, cmd: list[str]) -> None:
        last_ckpt = Path(self.output_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists() and _can_resume_from_last_checkpoint(self):
            cmd.append(f"ckpt_path={last_ckpt}")

    def _pretrain_cmd(self) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/pretrain.py",
            f"dataset={self.dataset}",
            f"ssl={self.paradigm}",
            f"seed={self.seed}",
            f"hydra.run.dir={self.output_dir}",
        ]
        self._append_resume(cmd)
        cmd.extend(self._metadata_overrides())
        return cmd

    def _finetune_cmd(self, runs_by_id: dict[str, "Run"]) -> list[str]:
        pretrain_ids = [dep_id for dep_id in self.depends_on if dep_id in runs_by_id]
        if len(pretrain_ids) != 1:
            raise ValueError(f"Finetune run {self.id} must have exactly one pretrain dependency")
        pretrain_dir = runs_by_id[pretrain_ids[0]].output_dir

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/finetune.py",
            f"dataset={self.dataset}",
            f"checkpoint={pretrain_dir}/encoder.pt",
            f"tasks={self.task}",
            f"seed={self.seed}",
            f"hydra.run.dir={self.output_dir}",
        ]
        self._append_resume(cmd)
        if self.freeze_encoder is True:
            cmd.extend(
                [
                    "training.freeze_encoder=true",
                    f"training.max_epochs={PROTO_A['max_epochs']}",
                    f"training.early_stopping_patience={PROTO_A['patience']}",
                    f"optimizer.lr={PROTO_A['lr']}",
                    "task.head_type=linear",
                    "task.hidden_dims=[]",
                    "task.dropout=0.0",
                ]
            )
        elif self.freeze_encoder is False:
            cmd.extend(
                [
                    "training.freeze_encoder=false",
                    f"training.max_epochs={PROTO_B['max_epochs']}",
                    f"training.early_stopping_patience={PROTO_B['patience']}",
                    f"optimizer.lr={PROTO_B['lr']}",
                ]
            )
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        if self.source_dataset is not None:
            cmd.append(f"+source_dataset={self.source_dataset}")
        if self.upstream_pretrain_lr is not None:
            cmd.append(f"+upstream_pretrain_lr={self.upstream_pretrain_lr}")
        if self.upstream_pretrain_mask_ratio is not None:
            cmd.append(f"+upstream_pretrain_mask_ratio={self.upstream_pretrain_mask_ratio}")
        cmd.extend(self._metadata_overrides())
        return cmd

    def _supervised_cmd(self) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/supervised.py",
            f"dataset={self.dataset}",
            f"tasks={self.task}",
            f"seed={self.seed}",
            f"hydra.run.dir={self.output_dir}",
        ]
        self._append_resume(cmd)
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        cmd.extend(self._metadata_overrides())
        return cmd

    def _gru_d_cmd(self) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/supervised.py",
            "--config-name",
            "gru_d",
            f"dataset={self.dataset}",
            f"tasks={self.task}",
            f"seed={self.seed}",
            f"hydra.run.dir={self.output_dir}",
        ]
        self._append_resume(cmd)
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        cmd.extend(self._metadata_overrides())
        return cmd

    def _xgboost_cmd(self) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/xgboost_baseline.py",
            f"dataset={self.dataset}",
            f"tasks={self.task}",
            f"seed={self.seed}",
            f"hydra.run.dir={self.output_dir}",
        ]
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        cmd.extend(self._metadata_overrides())
        return cmd


# ---------------------------------------------------------------------------
# Matrix Generation
# ---------------------------------------------------------------------------


def _short_value(value) -> str:
    return str(value).replace(".", "")


def _name_suffix(extra: dict | None) -> str:
    if not extra:
        return ""
    parts = []
    for key, value in sorted(extra.items()):
        short_key = key.split(".")[-1].replace("_", "")
        parts.append(f"{short_key}{_short_value(value)}")
    return "_" + "_".join(parts)


def _pretrain_key(paradigm: str, dataset: str, seed: int, extra: dict | None = None) -> str:
    return f"pretrain_{paradigm}_{dataset}_seed{seed}{_name_suffix(extra)}"


def _output_dir(experiment_class: str, run_key: str) -> str:
    return f"outputs/{experiment_class}/{run_key}"


class MatrixBuilder:
    """Generate the final thesis matrix by experiment class."""

    def __init__(self) -> None:
        self.runs: list[Run] = []
        self.pretrain_index: dict[tuple[str, str, int, tuple[tuple[str, str], ...]], Run] = {}

    def _pretrain_index_key(
        self,
        paradigm: str,
        dataset: str,
        seed: int,
        extra: dict | None = None,
    ) -> tuple[str, str, int, tuple[tuple[str, str], ...]]:
        extra_items = tuple(sorted((str(k), str(v)) for k, v in (extra or {}).items()))
        return paradigm, dataset, seed, extra_items

    def _add_run(self, run: Run) -> Run:
        self.runs.append(run)
        return run

    def _add_pretrain(
        self,
        experiment_class: str,
        paradigm: str,
        dataset: str,
        seed: int,
        extra: dict | None = None,
        experiment_subtype: str | None = None,
        model_size: str | None = None,
    ) -> Run:
        extra = extra or {}
        index_key = self._pretrain_index_key(paradigm, dataset, seed, extra)
        if index_key in self.pretrain_index:
            return self.pretrain_index[index_key]

        run_key = _pretrain_key(paradigm, dataset, seed, extra)
        run = Run(
            id=f"{experiment_class}_{run_key}",
            run_key=run_key,
            experiment_class=experiment_class,
            run_type="pretrain",
            paradigm=paradigm,
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(experiment_class, run_key),
            extra_overrides=extra,
            experiment_subtype=experiment_subtype,
            model_size=model_size,
        )
        self._add_run(run)
        self.pretrain_index[index_key] = run
        return run

    def _get_pretrain(
        self,
        paradigm: str,
        dataset: str,
        seed: int,
        extra: dict | None = None,
    ) -> Run:
        index_key = self._pretrain_index_key(paradigm, dataset, seed, extra)
        if index_key not in self.pretrain_index:
            raise KeyError(f"Pretrain not found: {_pretrain_key(paradigm, dataset, seed, extra)}")
        return self.pretrain_index[index_key]

    def _add_finetune(
        self,
        experiment_class: str,
        paradigm: str,
        dataset: str,
        seed: int,
        task: str,
        freeze: bool,
        pretrain_run: Run,
        label_fraction: float = 1.0,
        extra: dict | None = None,
        source_dataset: str | None = None,
        name_extra: dict | None = None,
        upstream_pretrain_lr: float | None = None,
        upstream_pretrain_mask_ratio: float | None = None,
        experiment_subtype: str | None = None,
        model_size: str | None = None,
    ) -> Run:
        prefix = "probe" if freeze else "finetune"
        run_key = f"{prefix}_{paradigm}_{task}_{dataset}_seed{seed}"
        if source_dataset:
            run_key += f"_from_{source_dataset}"
        if label_fraction < 1.0:
            run_key += f"_frac{_short_value(label_fraction)}"
        run_key += _name_suffix(name_extra)
        run_key += _name_suffix(extra)

        return self._add_run(
            Run(
                id=f"{experiment_class}_{run_key}",
                run_key=run_key,
                experiment_class=experiment_class,
                run_type="finetune",
                paradigm=paradigm,
                dataset=dataset,
                seed=seed,
                output_dir=_output_dir(experiment_class, run_key),
                depends_on=[pretrain_run.id],
                task=task,
                label_fraction=label_fraction,
                freeze_encoder=freeze,
                extra_overrides=extra or {},
                source_dataset=source_dataset,
                upstream_pretrain_lr=upstream_pretrain_lr,
                upstream_pretrain_mask_ratio=upstream_pretrain_mask_ratio,
                experiment_subtype=experiment_subtype,
                model_size=model_size,
            )
        )

    def _add_supervised(
        self,
        experiment_class: str,
        dataset: str,
        seed: int,
        task: str,
        label_fraction: float = 1.0,
        extra: dict | None = None,
        model_size: str | None = None,
    ) -> Run:
        run_key = f"supervised_{task}_{dataset}_seed{seed}"
        if model_size:
            run_key += f"_{model_size}"
        if label_fraction < 1.0:
            run_key += f"_frac{_short_value(label_fraction)}"
        return self._add_run(
            Run(
                id=f"{experiment_class}_{run_key}",
                run_key=run_key,
                experiment_class=experiment_class,
                run_type="supervised",
                paradigm="supervised",
                dataset=dataset,
                seed=seed,
                output_dir=_output_dir(experiment_class, run_key),
                task=task,
                label_fraction=label_fraction,
                extra_overrides=extra or {},
                model_size=model_size,
            )
        )

    def _add_gru_d(
        self,
        experiment_class: str,
        dataset: str,
        seed: int,
        task: str,
        label_fraction: float = 1.0,
    ) -> Run:
        run_key = f"gru_d_{task}_{dataset}_seed{seed}"
        if label_fraction < 1.0:
            run_key += f"_frac{_short_value(label_fraction)}"
        return self._add_run(
            Run(
                id=f"{experiment_class}_{run_key}",
                run_key=run_key,
                experiment_class=experiment_class,
                run_type="gru_d",
                paradigm="gru_d",
                dataset=dataset,
                seed=seed,
                output_dir=_output_dir(experiment_class, run_key),
                task=task,
                label_fraction=label_fraction,
            )
        )

    def _add_xgboost(
        self,
        experiment_class: str,
        dataset: str,
        seed: int,
        task: str,
        label_fraction: float = 1.0,
    ) -> Run:
        run_key = f"xgboost_{task}_{dataset}_seed{seed}"
        if label_fraction < 1.0:
            run_key += f"_frac{_short_value(label_fraction)}"
        return self._add_run(
            Run(
                id=f"{experiment_class}_{run_key}",
                run_key=run_key,
                experiment_class=experiment_class,
                run_type="xgboost",
                paradigm="xgboost",
                dataset=dataset,
                seed=seed,
                output_dir=_output_dir(experiment_class, run_key),
                task=task,
                label_fraction=label_fraction,
            )
        )

    def build_core_ssl_benchmark(self) -> None:
        experiment_class = "core_ssl_benchmark"
        for seed in SEEDS_EXTENDED:
            for dataset in DATASETS:
                for paradigm in SSL_PARADIGMS:
                    pretrain = self._add_pretrain(experiment_class, paradigm, dataset, seed)
                    for task in TASKS:
                        self._add_finetune(
                            experiment_class, paradigm, dataset, seed, task, False, pretrain
                        )
                        self._add_finetune(
                            experiment_class, paradigm, dataset, seed, task, True, pretrain
                        )
                for task in TASKS:
                    self._add_supervised(experiment_class, dataset, seed, task)

    def build_label_efficiency(self) -> None:
        experiment_class = "label_efficiency"
        for seed in SEEDS_EXTENDED:
            for dataset in DATASETS:
                for paradigm in SSL_PARADIGMS:
                    pretrain = self._get_pretrain(paradigm, dataset, seed)
                    for frac in LABEL_FRACTIONS_MORTALITY_24H:
                        self._add_finetune(
                            experiment_class,
                            paradigm,
                            dataset,
                            seed,
                            "mortality_24h",
                            False,
                            pretrain,
                            frac,
                        )
                        self._add_finetune(
                            experiment_class,
                            paradigm,
                            dataset,
                            seed,
                            "mortality_24h",
                            True,
                            pretrain,
                            frac,
                        )
                    for task in TASKS[1:]:
                        fractions = (
                            LABEL_FRACTIONS_FULL
                            if task == "mortality_hospital"
                            else LABEL_FRACTIONS_TREND
                        )
                        for frac in fractions:
                            self._add_finetune(
                                experiment_class,
                                paradigm,
                                dataset,
                                seed,
                                task,
                                False,
                                pretrain,
                                frac,
                            )
                            self._add_finetune(
                                experiment_class,
                                paradigm,
                                dataset,
                                seed,
                                task,
                                True,
                                pretrain,
                                frac,
                            )
                for frac in LABEL_FRACTIONS_MORTALITY_24H:
                    self._add_supervised(experiment_class, dataset, seed, "mortality_24h", frac)
                for task in TASKS[1:]:
                    fractions = (
                        LABEL_FRACTIONS_FULL
                        if task == "mortality_hospital"
                        else LABEL_FRACTIONS_TREND
                    )
                    for frac in fractions:
                        self._add_supervised(experiment_class, dataset, seed, task, frac)

    def build_cross_dataset_transfer(self) -> None:
        experiment_class = "cross_dataset_transfer"
        for seed in SEEDS_EXTENDED:
            for source_dataset, target_dataset in TRANSFER_PAIRS:
                for paradigm in SSL_PARADIGMS:
                    pretrain = self._get_pretrain(paradigm, source_dataset, seed)
                    for task in TASKS:
                        self._add_finetune(
                            experiment_class,
                            paradigm,
                            target_dataset,
                            seed,
                            task,
                            False,
                            pretrain,
                            source_dataset=source_dataset,
                        )

    def build_hp_robustness(self) -> None:
        experiment_class = "hp_robustness"
        task = "mortality_24h"
        dataset = "miiv"
        for seed in SEEDS_EXTENDED:
            for paradigm in SSL_PARADIGMS:
                for lr in LR_ROBUSTNESS:
                    extra = {"optimizer.lr": lr}
                    subtype = "lr_sensitivity"
                    pretrain = self._add_pretrain(
                        experiment_class,
                        paradigm,
                        dataset,
                        seed,
                        extra,
                        experiment_subtype=subtype,
                    )
                    self._add_finetune(
                        experiment_class,
                        paradigm,
                        dataset,
                        seed,
                        task,
                        False,
                        pretrain,
                        name_extra=extra,
                        upstream_pretrain_lr=lr,
                        experiment_subtype=subtype,
                    )
                for mask_ratio in MASK_RATIO_ROBUSTNESS:
                    extra = {"ssl.mask_ratio": mask_ratio}
                    subtype = "mask_ratio_sensitivity"
                    if paradigm == "contrastive":
                        # Contrastive complementary views make the per-view mask
                        # budgets asymmetric away from 0.5. The robustness slice
                        # therefore uses independent views and is labeled as a
                        # view/mask sensitivity test rather than a pure
                        # mask-ratio sweep.
                        extra["ssl.complementary_masks"] = False
                        subtype = VIEW_MASK_EXPERIMENT_SUBTYPE
                    pretrain = self._add_pretrain(
                        experiment_class,
                        paradigm,
                        dataset,
                        seed,
                        extra,
                        experiment_subtype=subtype,
                    )
                    self._add_finetune(
                        experiment_class,
                        paradigm,
                        dataset,
                        seed,
                        task,
                        False,
                        pretrain,
                        name_extra=extra,
                        upstream_pretrain_mask_ratio=mask_ratio,
                        experiment_subtype=subtype,
                    )

    def build_capacity_study(self) -> None:
        experiment_class = "capacity_study"
        dataset = "miiv"
        task = "mortality_24h"
        for seed in SEEDS_EXTENDED:
            for model_size, size_cfg in MODEL_SIZES.items():
                model_extra = {"model": size_cfg["model"]}
                pretrain_extra = {**model_extra, **size_cfg["ssl_scale"]["mae"]}
                pretrain = self._add_pretrain(
                    experiment_class,
                    "mae",
                    dataset,
                    seed,
                    pretrain_extra,
                    model_size=model_size,
                )
                for frac in LABEL_FRACTIONS_CAPACITY:
                    self._add_finetune(
                        experiment_class,
                        "mae",
                        dataset,
                        seed,
                        task,
                        False,
                        pretrain,
                        frac,
                        extra=model_extra,
                        name_extra={"size": model_size},
                        model_size=model_size,
                    )
                    self._add_finetune(
                        experiment_class,
                        "mae",
                        dataset,
                        seed,
                        task,
                        True,
                        pretrain,
                        frac,
                        extra=model_extra,
                        name_extra={"size": model_size},
                        model_size=model_size,
                    )
                    self._add_supervised(
                        experiment_class,
                        dataset,
                        seed,
                        task,
                        frac,
                        extra=model_extra,
                        model_size=model_size,
                    )

    def build_classical_baselines(self) -> None:
        experiment_class = "classical_baselines"
        for seed in SEEDS_EXTENDED:
            for dataset in DATASETS:
                for task in TASKS:
                    self._add_xgboost(experiment_class, dataset, seed, task)
                    self._add_gru_d(experiment_class, dataset, seed, task)
                for frac in LABEL_FRACTIONS_MORTALITY_24H:
                    self._add_xgboost(experiment_class, dataset, seed, "mortality_24h", frac)
                    self._add_gru_d(experiment_class, dataset, seed, "mortality_24h", frac)
                for task in TASKS[1:]:
                    for frac in LABEL_FRACTIONS_TREND:
                        self._add_xgboost(experiment_class, dataset, seed, task, frac)
                        self._add_gru_d(experiment_class, dataset, seed, task, frac)

    def build_ts2vec_extension(self) -> None:
        experiment_class = "ts2vec_extension"
        for seed in SEEDS_EXTENDED:
            for dataset in DATASETS:
                pretrain = self._add_pretrain(experiment_class, "ts2vec", dataset, seed)
                for task in TASKS:
                    self._add_finetune(
                        experiment_class, "ts2vec", dataset, seed, task, False, pretrain
                    )
                    self._add_finetune(
                        experiment_class, "ts2vec", dataset, seed, task, True, pretrain
                    )

    def build_smart_external_reference(self) -> None:
        experiment_class = "smart_external_reference"
        pretrain_extra = {"model": "smart"}
        finetune_extra = {"model": "smart"}
        model_size = "default"
        for seed in SEEDS_EXTENDED:
            for dataset in DATASETS:
                pretrain = self._add_pretrain(
                    experiment_class,
                    "smart",
                    dataset,
                    seed,
                    pretrain_extra,
                    model_size=model_size,
                )
                for task in TASKS:
                    self._add_finetune(
                        experiment_class,
                        "smart",
                        dataset,
                        seed,
                        task,
                        False,
                        pretrain,
                        extra=finetune_extra,
                        model_size=model_size,
                    )
                    self._add_finetune(
                        experiment_class,
                        "smart",
                        dataset,
                        seed,
                        task,
                        True,
                        pretrain,
                        extra=finetune_extra,
                        model_size=model_size,
                    )

    def build_all(self) -> list[Run]:
        self.build_core_ssl_benchmark()
        self.build_label_efficiency()
        self.build_cross_dataset_transfer()
        self.build_hp_robustness()
        self.build_capacity_study()
        self.build_classical_baselines()
        self.build_ts2vec_extension()
        self.build_smart_external_reference()
        return self.runs


def generate_all_runs() -> list[Run]:
    return MatrixBuilder().build_all()


def scientific_fingerprint(run: Run) -> tuple:
    """Scientific identity used by matrix snapshot tests."""
    pretrain_overrides = tuple(sorted((str(k), str(v)) for k, v in run.extra_overrides.items()))
    return (
        run.run_type,
        run.experiment_class,
        run.experiment_subtype,
        run.paradigm,
        run.dataset,
        run.task,
        run.seed,
        run.protocol,
        run.label_fraction,
        run.source_dataset,
        run.model_size,
        run.upstream_pretrain_lr,
        run.upstream_pretrain_mask_ratio,
        pretrain_overrides if run.run_type == "pretrain" else (),
    )


def apply_revision(runs: list[Run], revision: str, reason: str | None = None) -> list[Run]:
    """Inject revision metadata into IDs, output dirs, dependencies, and overrides."""
    old_to_new: dict[str, str] = {}

    for run in runs:
        old_id = run.id
        run.id = f"{run.experiment_class}_rev-{revision}_{run.run_key}"
        old_to_new[old_id] = run.id
        run.output_dir = _output_dir(
            f"{run.experiment_class}_rev-{revision}", run.run_key or old_id
        )
        run.extra_overrides["revision"] = revision
        if reason:
            run.extra_overrides["rerun_reason"] = reason

    for run in runs:
        run.depends_on = [old_to_new.get(dep_id, dep_id) for dep_id in run.depends_on]

    return runs


def apply_wandb_target(
    runs: list[Run],
    project: str | None = None,
    entity: str | None = None,
) -> list[Run]:
    """Inject W&B project/entity overrides into generated runs."""
    if project is None and entity is None:
        return runs
    for run in runs:
        if project is not None:
            run.extra_overrides["project_name"] = project
            run.extra_overrides["logging.wandb_project"] = project
        if entity is not None:
            run.extra_overrides["logging.wandb_entity"] = entity
    return runs


def apply_launch_commit(runs: list[Run], launch_commit: str | None = None) -> list[Run]:
    """Inject the exact git commit used for launch into each Hydra config."""
    if not launch_commit:
        return runs
    for run in runs:
        run.extra_overrides["+launch_commit"] = launch_commit
    return runs


def _git_stdout(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or shlex.join(["git", *args])
        raise RuntimeError(message)
    return result.stdout.strip()


def _git_quiet(args: list[str]) -> bool:
    result = subprocess.run(
        ["git", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode in {0, 1}:
        return result.returncode == 0
    message = result.stderr.strip() or shlex.join(["git", *args])
    raise RuntimeError(message)


def _validate_clean_final_launch_state(launch_commit: str) -> str | None:
    """Return an error string if a final launch would have ambiguous provenance."""
    try:
        resolved_launch_commit = _git_stdout(
            ["rev-parse", "--verify", f"{launch_commit}^{{commit}}"]
        )
        current_commit = _git_stdout(["rev-parse", "--verify", "HEAD"])
        if current_commit != resolved_launch_commit:
            return (
                "final launch commit does not match the current checkout "
                f"(current={current_commit}, launch_commit={resolved_launch_commit})"
            )
        if not _git_quiet(["diff", "--quiet"]) or not _git_quiet(["diff", "--cached", "--quiet"]):
            return (
                "final run/retry requires a clean tracked worktree. "
                "Commit or stash tracked changes first; dry runs may be used for local audits."
            )
    except RuntimeError as exc:
        return f"could not validate final launch git provenance: {exc}"
    return None


def validate_direct_final_launch_policy(args, parser: argparse.ArgumentParser) -> None:
    """Require auditable provenance for direct final run/retry invocations."""
    if args.command not in {"run", "retry"}:
        return

    launch_commit = getattr(args, "launch_commit", None)
    is_run_dry_run = args.command == "run" and getattr(args, "dry_run", False)
    if is_run_dry_run:
        if not launch_commit:
            print(
                "WARNING: dry run has no launch provenance. Final run/retry requires "
                "--launch-commit or SLICES_LAUNCH_COMMIT.",
                file=sys.stderr,
            )
        return

    if not launch_commit:
        parser.error(
            "final run/retry requires --launch-commit or SLICES_LAUNCH_COMMIT " "for W&B provenance"
        )

    error = _validate_clean_final_launch_state(str(launch_commit))
    if error:
        parser.error(error)


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"version": 1, "runs": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _normalize_identity_value(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_launch_identity(identity: dict | None) -> dict[str, str | None]:
    if not isinstance(identity, dict):
        identity = {}
    return {key: _normalize_identity_value(identity.get(key)) for key in LAUNCH_IDENTITY_KEYS}


def _override_value(overrides: dict, *keys: str) -> str | None:
    for key in keys:
        value = overrides.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _run_launch_identity(run: Run) -> dict[str, str | None]:
    return _normalize_launch_identity(
        {
            "revision": _override_value(run.extra_overrides, "revision"),
            "wandb_project": _override_value(
                run.extra_overrides,
                "logging.wandb_project",
                "project_name",
            ),
            "wandb_entity": _override_value(run.extra_overrides, "logging.wandb_entity"),
            "launch_commit": _run_launch_commit(run),
        }
    )


def _command_override(command: str | None, *keys: str) -> str | None:
    if not command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = str(command).split()

    key_set = set(keys)
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.lstrip("+")
        if key in key_set and value:
            return value
    return None


def _state_launch_identity(info: dict | None) -> dict[str, str | None]:
    if not isinstance(info, dict):
        return _normalize_launch_identity(None)

    identity = {}
    if isinstance(info.get("launch_identity"), dict):
        identity.update(info["launch_identity"])

    for key in LAUNCH_IDENTITY_KEYS:
        if not identity.get(key) and info.get(key):
            identity[key] = info[key]

    command = str(info.get("command") or "")
    if not identity.get("revision"):
        identity["revision"] = _command_override(command, "revision")
    if not identity.get("wandb_project"):
        identity["wandb_project"] = _command_override(
            command,
            "logging.wandb_project",
            "project_name",
        )
    if not identity.get("wandb_entity"):
        identity["wandb_entity"] = _command_override(command, "logging.wandb_entity")
    if not identity.get("launch_commit"):
        identity["launch_commit"] = _state_launch_commit(info)

    return _normalize_launch_identity(identity)


def _has_scoped_launch_identity(identity: dict[str, str | None]) -> bool:
    return any(identity.get(key) is not None for key in LAUNCH_IDENTITY_KEYS)


def _launch_identity_matches(
    expected: dict[str, str | None],
    actual: dict[str, str | None],
) -> bool:
    expected = _normalize_launch_identity(expected)
    actual = _normalize_launch_identity(actual)
    return all(expected[key] == actual[key] for key in LAUNCH_IDENTITY_KEYS)


def _launch_identity_changed_keys(
    expected: dict[str, str | None],
    actual: dict[str, str | None],
) -> list[str]:
    expected = _normalize_launch_identity(expected)
    actual = _normalize_launch_identity(actual)
    return [key for key in LAUNCH_IDENTITY_KEYS if expected[key] != actual[key]]


def _format_launch_identity(identity: dict[str, str | None]) -> str:
    identity = _normalize_launch_identity(identity)
    return ", ".join(
        f"{key}={identity[key] if identity[key] is not None else 'none'}"
        for key in LAUNCH_IDENTITY_KEYS
    )


def _read_output_launch_identity(output_dir: Path) -> dict[str, str | None] | None:
    marker = output_dir / LAUNCH_IDENTITY_FILE
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return _normalize_launch_identity(None)
    if isinstance(payload, dict) and isinstance(payload.get("launch_identity"), dict):
        return _normalize_launch_identity(payload["launch_identity"])
    if isinstance(payload, dict):
        return _normalize_launch_identity(payload)
    return _normalize_launch_identity(None)


def _write_output_launch_identity(run: Run) -> None:
    output_dir = Path(run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run.id,
        "launch_identity": _run_launch_identity(run),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=f"{LAUNCH_IDENTITY_FILE}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, output_dir / LAUNCH_IDENTITY_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _can_resume_from_last_checkpoint(run: Run) -> bool:
    expected = _run_launch_identity(run)
    actual = _read_output_launch_identity(Path(run.output_dir))
    if actual is None:
        return not _has_scoped_launch_identity(expected)
    return _launch_identity_matches(expected, actual)


def get_run_status(state: dict, run_id: str) -> str:
    return state["runs"].get(run_id, {}).get("status", "pending")


def set_run_status(state: dict, run_id: str, status: str, **kwargs) -> None:
    state["runs"].setdefault(run_id, {})
    state["runs"][run_id]["status"] = status
    state["runs"][run_id].update(kwargs)


def _completed_pretrain_missing_encoder(run: Run) -> bool:
    """Return whether a supposedly completed pretrain lacks its encoder artifact."""
    return run.run_type == "pretrain" and not (Path(run.output_dir) / "encoder.pt").exists()


def _mark_completed_pretrain_missing_encoder(
    run: Run,
    runs: list[Run],
    state: dict,
    required_by: str | None = None,
) -> None:
    encoder_path = Path(run.output_dir) / "encoder.pt"
    reason = f"completed pretrain missing encoder.pt at {encoder_path}"
    now = datetime.now(timezone.utc).isoformat()
    suffix = f"; required by {required_by}" if required_by else ""
    print(f"  FAILED {run.id}: {reason}{suffix}")
    set_run_status(state, run.id, "failed", finished_at=now, exit_code=None, reason=reason)
    _propagate_failure(run.id, runs, state)


def _revalidate_completed_pretrain_artifacts(runs: list[Run], state: dict) -> None:
    """Fail stale completed pretrains whose encoder artifact is no longer present."""
    for run in runs:
        if get_run_status(state, run.id) == "completed" and _completed_pretrain_missing_encoder(
            run
        ):
            _mark_completed_pretrain_missing_encoder(run, runs, state)


def _dependencies_ready(
    run: Run,
    runs: list[Run],
    runs_by_id: dict[str, Run],
    state: dict,
) -> bool:
    """Check dependency state and revalidate completed pretrain artifacts."""
    for dep_id in run.depends_on:
        if get_run_status(state, dep_id) != "completed":
            return False
        dep = runs_by_id.get(dep_id)
        if dep is not None and _completed_pretrain_missing_encoder(dep):
            _mark_completed_pretrain_missing_encoder(dep, runs, state, required_by=run.id)
            return False
    return True


def _run_launch_commit(run: Run) -> str | None:
    commit = run.extra_overrides.get("+launch_commit")
    return str(commit) if commit else None


def _state_launch_commit(info: dict) -> str | None:
    commit = info.get("launch_commit")
    if commit:
        return str(commit)
    command = str(info.get("command") or "")
    match = re.search(r"(?:^|\s)\+?launch_commit=([^\s]+)", command)
    return match.group(1) if match else None


def _safe_path_component(value: str | None) -> str:
    text = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text)[:32] or "unknown"


def _quarantine_stale_output_dir(
    run: Run,
    actual_identity: str | None,
    expected_identity: str,
) -> str | None:
    output_dir = Path(run.output_dir)
    if not output_dir.exists():
        return None

    parent = output_dir.parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    actual = _safe_path_component(actual_identity)
    expected = _safe_path_component(expected_identity)
    stem = f"{output_dir.name}.stale-{actual}-to-{expected}-{timestamp}"
    candidate = parent / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = parent / f"{stem}-{suffix}"

    output_dir.rename(candidate)
    return str(candidate)


def reset_state_for_launch_identity_mismatch(runs: list[Run], state: dict) -> None:
    """Do not trust state or artifacts from a different launch target."""
    reset = 0
    quarantined = 0
    for run in runs:
        expected_identity = _run_launch_identity(run)
        info = state["runs"].get(run.id)
        status = info.get("status", "pending") if info else "pending"
        actual_identity = _state_launch_identity(info)
        reasons = []

        if info and not _launch_identity_matches(expected_identity, actual_identity):
            changed_keys = _launch_identity_changed_keys(expected_identity, actual_identity)
            if changed_keys == ["launch_commit"]:
                reasons.append(
                    "launch_commit changed from "
                    f"{actual_identity['launch_commit'] or 'unknown'} "
                    f"to {expected_identity['launch_commit'] or 'none'}"
                )
            else:
                reasons.append(
                    "launch identity changed from "
                    f"{_format_launch_identity(actual_identity)} to "
                    f"{_format_launch_identity(expected_identity)}"
                )

        output_dir = Path(run.output_dir)
        output_identity = _read_output_launch_identity(output_dir) if output_dir.exists() else None
        if output_dir.exists():
            if output_identity is None:
                if _has_scoped_launch_identity(expected_identity):
                    reasons.append("output directory has no launch identity marker")
            elif not _launch_identity_matches(expected_identity, output_identity):
                reasons.append(
                    "output directory launch identity is "
                    f"{_format_launch_identity(output_identity)}, expected "
                    f"{_format_launch_identity(expected_identity)}"
                )

        if not reasons:
            continue

        if status == "running":
            print(
                f"  WARNING {run.id}: state/output identity mismatch while run is marked "
                "running; leaving it untouched."
            )
            continue

        quarantined_output_dir = _quarantine_stale_output_dir(
            run,
            _format_launch_identity(output_identity or actual_identity),
            _format_launch_identity(expected_identity),
        )
        if quarantined_output_dir:
            quarantined += 1

        state["runs"][run.id] = {
            "status": "pending",
            "launch_identity": expected_identity,
            "reset_reason": "; ".join(dict.fromkeys(reasons)),
        }
        if quarantined_output_dir:
            state["runs"][run.id]["quarantined_output_dir"] = quarantined_output_dir
        reset += 1

    if reset:
        print(f"Reset {reset} stale run state entries for launch identity mismatch.")
    if quarantined:
        print(f"Quarantined {quarantined} stale output directories before relaunch.")


def reset_state_for_launch_commit_mismatch(runs: list[Run], state: dict) -> None:
    """Backward-compatible wrapper for launch identity reconciliation."""
    reset_state_for_launch_identity_mismatch(runs, state)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _pid_matches_command(pid: int, expected_command: str | None) -> bool:
    if not expected_command:
        return True
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return True
    if result.returncode != 0:
        return False
    actual_command = result.stdout.strip()
    return bool(actual_command) and (
        expected_command == actual_command or expected_command in actual_command
    )


def recover_stale_running(state: dict) -> None:
    """Reset running entries whose PIDs are dead or no longer match the run."""
    for run_id, info in state["runs"].items():
        if info.get("status") == "running":
            pid = info.get("pid")
            if (
                pid is None
                or not is_pid_alive(int(pid))
                or not _pid_matches_command(int(pid), info.get("command"))
            ):
                info["status"] = "pending"
                info.pop("pid", None)
                info.pop("command", None)
                print(f"  Recovered stale run: {run_id}")


RUN_SLOT_COSTS = {
    "pretrain": 4,
    "finetune": 1,
    "supervised": 1,
    "gru_d": 1,
    "xgboost": 1,
}


def _slot_cost(run: Run, slot_budget: int) -> int:
    return min(RUN_SLOT_COSTS.get(run.run_type, 1), slot_budget)


def _select_ready_runs(
    ready: list[Run],
    active_run_ids: set[str],
    runs_by_id: dict[str, Run],
    slot_budget: int,
) -> list[Run]:
    active_slots = sum(_slot_cost(runs_by_id[run_id], slot_budget) for run_id in active_run_ids)
    available_slots = slot_budget - active_slots
    selected: list[Run] = []

    indexed_ready = list(enumerate(ready))
    indexed_ready.sort(key=lambda item: (-_slot_cost(item[1], slot_budget), item[0]))
    for _, run in indexed_ready:
        cost = _slot_cost(run, slot_budget)
        if cost <= available_slots:
            selected.append(run)
            available_slots -= cost
    return selected


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def run_scheduler(runs: list[Run], state: dict, parallel: int, dry_run: bool) -> int:
    if parallel < 1:
        raise ValueError(f"--parallel must be >= 1, got {parallel}")

    runs_by_id = {run.id: run for run in runs}
    active: dict[str, subprocess.Popen] = {}
    shutting_down = False

    def handle_signal(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        for proc in active.values():
            try:
                proc.terminate()
            except OSError:
                pass
        deadline = time.time() + 30
        for proc in list(active.values()):
            try:
                proc.wait(timeout=max(0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()
        for run_id in active:
            set_run_status(state, run_id, "pending")
        save_state(state)
        print("State saved. Interrupted runs reset to pending.")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _print_dry_run(runs, runs_by_id)
        return 0

    reset_state_for_launch_identity_mismatch(runs, state)
    _revalidate_completed_pretrain_artifacts(runs, state)

    print(f"Scheduler started (slot_budget={parallel})")
    print(f"State file: {STATE_FILE}\n")

    while not shutting_down:
        finished = []
        for run_id, proc in active.items():
            exit_code = proc.poll()
            if exit_code is not None:
                finished.append((run_id, exit_code))

        for run_id, exit_code in finished:
            proc = active.pop(run_id)
            log_fh = getattr(proc, "_log_fh", None)
            if log_fh:
                log_fh.close()
            run = runs_by_id[run_id]
            now = datetime.now(timezone.utc).isoformat()
            elapsed = _format_elapsed(state, run_id, now)

            if exit_code == 0:
                if (
                    run.run_type == "pretrain"
                    and not (Path(run.output_dir) / "encoder.pt").exists()
                ):
                    print(f"  FAILED {run_id}: exit 0 but encoder.pt missing {elapsed}")
                    set_run_status(state, run_id, "failed", finished_at=now, exit_code=exit_code)
                    _propagate_failure(run_id, runs, state)
                    continue
                print(f"  DONE {run_id} {elapsed}")
                set_run_status(state, run_id, "completed", finished_at=now, exit_code=0)
            else:
                print(f"  FAILED {run_id} (exit {exit_code}) {elapsed}")
                set_run_status(state, run_id, "failed", finished_at=now, exit_code=exit_code)
                _propagate_failure(run_id, runs, state)

        ready = []
        for run in runs:
            if get_run_status(state, run.id) != "pending":
                continue
            if run.id in active:
                continue
            if _dependencies_ready(run, runs, runs_by_id, state):
                ready.append(run)

        launch_batch = _select_ready_runs(ready, set(active), runs_by_id, parallel)
        for run in launch_batch:
            cmd = run.build_command(runs_by_id)
            log_file = LOG_DIR / f"{run.id}.log"
            now = datetime.now(timezone.utc).isoformat()
            print(f"  START {run.id}")
            _write_output_launch_identity(run)
            log_fh = open(log_file, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            proc._log_fh = log_fh  # type: ignore[attr-defined]
            active[run.id] = proc
            status_updates = {
                "started_at": now,
                "pid": proc.pid,
                "log_file": str(log_file),
                "command": shlex.join(cmd),
                "launch_identity": _run_launch_identity(run),
            }
            for key, value in status_updates["launch_identity"].items():
                if value is not None:
                    status_updates[key] = value
            set_run_status(state, run.id, "running", **status_updates)

        save_state(state)

        if not active and not ready:
            break
        time.sleep(2)

    for proc in active.values():
        log_fh = getattr(proc, "_log_fh", None)
        if log_fh:
            log_fh.close()

    print()
    _print_summary(runs, state)
    return _scheduler_exit_code(runs, state)


def _format_elapsed(state: dict, run_id: str, now_iso: str) -> str:
    started = state["runs"].get(run_id, {}).get("started_at")
    if not started:
        return ""
    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(now_iso)
        secs = int((t1 - t0).total_seconds())
        if secs < 60:
            return f"({secs}s)"
        if secs < 3600:
            return f"({secs // 60}m{secs % 60:02d}s)"
        hours, rem = divmod(secs, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"({hours}h{minutes:02d}m{seconds:02d}s)"
    except (TypeError, ValueError):
        return ""


def _propagate_failure(failed_id: str, runs: list[Run], state: dict) -> None:
    queue = [failed_id]
    while queue:
        dep_id = queue.pop(0)
        for run in runs:
            if dep_id in run.depends_on and get_run_status(state, run.id) == "pending":
                set_run_status(state, run.id, "skipped", reason=f"dependency {dep_id} failed")
                print(f"  SKIP {run.id} (dep {dep_id} failed)")
                queue.append(run.id)


def _print_dry_run(runs: list[Run], runs_by_id: dict[str, Run]) -> None:
    print(f"DRY RUN: {len(runs)} runs\n")
    for run in runs:
        deps = ", ".join(run.depends_on) if run.depends_on else "(none)"
        cmd = shlex.join(run.build_command(runs_by_id))
        print(f"[{run.id}]")
        print(f"  experiment_class={run.experiment_class}  type={run.run_type}  deps={deps}")
        print(f"  dir={run.output_dir}")
        print(f"  cmd: {cmd}\n")


# ---------------------------------------------------------------------------
# Status / Retry
# ---------------------------------------------------------------------------


def _extract_revision_from_id(run_id: str) -> str | None:
    match = re.search(r"_rev-([^_]+)_", run_id)
    return match.group(1) if match else None


def _extract_experiment_class_from_id(run_id: str) -> str | None:
    for experiment_class in sorted(EXPERIMENT_CLASSES, key=len, reverse=True):
        if run_id.startswith(f"{experiment_class}_"):
            return experiment_class
    return None


def print_status(
    experiment_class_filter: list[str] | None = None,
    revision_filter: str | None = None,
) -> None:
    all_runs = generate_all_runs()
    if revision_filter:
        all_runs = apply_revision(all_runs, revision_filter)
    state = load_state()
    generated_ids = {run.id for run in all_runs}

    groups: dict[tuple[str, str | None], list[str]] = {}
    for run in all_runs:
        groups.setdefault((run.experiment_class, revision_filter), []).append(run.id)

    for run_id in state.get("runs", {}):
        if run_id not in generated_ids and "_rev-" in run_id:
            experiment_class = _extract_experiment_class_from_id(run_id)
            revision = _extract_revision_from_id(run_id)
            if revision_filter and revision != revision_filter:
                continue
            if experiment_class and revision:
                groups.setdefault((experiment_class, revision), []).append(run_id)

    group_keys = sorted(groups, key=lambda key: (key[0], key[1] is not None, key[1] or ""))
    if experiment_class_filter:
        allowed = set(experiment_class_filter)
        group_keys = [key for key in group_keys if key[0] in allowed]

    print(
        f"{'Experiment Class':>28} | {'Total':>5} | {'Done':>4} | {'Run':>3} | "
        f"{'Fail':>4} | {'Pend':>4} | {'Skip':>4}"
    )
    print("-" * 76)

    totals = {"total": 0, "completed": 0, "running": 0, "failed": 0, "pending": 0, "skipped": 0}
    for experiment_class, revision in group_keys:
        run_ids = groups[(experiment_class, revision)]
        label = f"{experiment_class}/{revision}" if revision else experiment_class
        counts = {
            "total": len(run_ids),
            "completed": 0,
            "running": 0,
            "failed": 0,
            "pending": 0,
            "skipped": 0,
        }
        for run_id in run_ids:
            status = get_run_status(state, run_id)
            counts[status if status in counts else "pending"] += 1
        print(
            f"{label:>28} | {counts['total']:>5} | {counts['completed']:>4} | "
            f"{counts['running']:>3} | {counts['failed']:>4} | "
            f"{counts['pending']:>4} | {counts['skipped']:>4}"
        )
        for key in totals:
            totals[key] += counts[key]

    print("-" * 76)
    print(
        f"{'TOTAL':>28} | {totals['total']:>5} | {totals['completed']:>4} | "
        f"{totals['running']:>3} | {totals['failed']:>4} | "
        f"{totals['pending']:>4} | {totals['skipped']:>4}"
    )


def _print_summary(runs: list[Run], state: dict) -> None:
    counts = {"completed": 0, "failed": 0, "skipped": 0, "pending": 0}
    for run in runs:
        status = get_run_status(state, run.id)
        counts[status] = counts.get(status, 0) + 1
    total = len(runs)
    print(
        f"Summary: {counts['completed']}/{total} completed, "
        f"{counts['failed']} failed, {counts['skipped']} skipped, {counts['pending']} pending"
    )


def _scheduler_exit_code(runs: list[Run], state: dict) -> int:
    failed = [run.id for run in runs if get_run_status(state, run.id) == "failed"]
    skipped = [run.id for run in runs if get_run_status(state, run.id) == "skipped"]
    running = [run.id for run in runs if get_run_status(state, run.id) == "running"]
    pending = [run.id for run in runs if get_run_status(state, run.id) == "pending"]
    if failed or skipped or running or pending:
        print(
            "Scheduler incomplete: "
            f"{len(failed)} failed, {len(skipped)} skipped due to dependency failure, "
            f"{len(running)} running, {len(pending)} pending"
        )
        return 1
    return 0


def _collect_dependency_closure(
    runs: list[Run],
    all_by_id: dict[str, Run],
) -> tuple[list[Run], dict[str, set[str]], set[str]]:
    deps: list[Run] = []
    required_by: dict[str, set[str]] = {}
    missing: set[str] = set()
    seen: set[str] = set()
    queue = [(dep_id, run.id) for run in runs for dep_id in run.depends_on]
    while queue:
        dep_id, parent_id = queue.pop(0)
        required_by.setdefault(dep_id, set()).add(parent_id)
        if dep_id in seen:
            continue
        seen.add(dep_id)
        dep = all_by_id.get(dep_id)
        if dep is None:
            missing.add(dep_id)
            continue
        deps.append(dep)
        queue.extend((child_dep_id, dep.id) for child_dep_id in dep.depends_on)
    return deps, required_by, missing


def _collect_skipped_dependent_closure(
    runs: list[Run],
    all_runs: list[Run],
    state: dict,
) -> list[Run]:
    """Find skipped downstream runs that will remain skipped after retry --failed."""
    downstream: dict[str, list[Run]] = {}
    for run in all_runs:
        for dep_id in run.depends_on:
            downstream.setdefault(dep_id, []).append(run)

    start_ids = {run.id for run in runs}
    queue = list(start_ids)
    seen: set[str] = set()
    skipped_dependents: list[Run] = []
    while queue:
        dep_id = queue.pop(0)
        for run in downstream.get(dep_id, []):
            if run.id in seen or run.id in start_ids:
                continue
            seen.add(run.id)
            if get_run_status(state, run.id) == "skipped":
                skipped_dependents.append(run)
                queue.append(run.id)
    return skipped_dependents


def _retry_command_suggestion(args, *, include_failed: bool, include_skipped: bool) -> str:
    cmd = ["uv", "run", "python", "scripts/internal/run_experiments.py", "retry"]
    if include_failed:
        cmd.append("--failed")
    if include_skipped:
        cmd.append("--skipped")
    if args.experiment_class:
        cmd.append("--experiment-class")
        cmd.extend(args.experiment_class)
    if args.revision:
        cmd.extend(["--revision", str(args.revision)])
    if args.reason:
        cmd.extend(["--reason", str(args.reason)])
    if args.project:
        cmd.extend(["--project", str(args.project)])
    if args.entity:
        cmd.extend(["--entity", str(args.entity)])
    launch_commit = getattr(args, "launch_commit", None)
    if launch_commit:
        cmd.extend(["--launch-commit", str(launch_commit)])
    cmd.extend(["--parallel", str(args.parallel)])
    return shlex.join(cmd)


def _fail_for_blocked_retry_dependencies(
    blocked: list[Run],
    missing: set[str],
    state: dict,
    required_by: dict[str, set[str]],
    args,
) -> None:
    print("Cannot retry the selected runs because required dependencies are blocked.")
    if blocked:
        print("\nBlocked dependencies:")
        for dep in sorted(blocked, key=lambda run: (get_run_status(state, run.id), run.id)):
            dependents = ", ".join(sorted(required_by.get(dep.id, set())))
            print(f"  {dep.id} [{get_run_status(state, dep.id)}] required by: {dependents}")
    if missing:
        print("\nMissing generated dependencies:")
        for dep_id in sorted(missing):
            dependents = ", ".join(sorted(required_by.get(dep_id, set())))
            print(f"  {dep_id} required by: {dependents}")
    suggested = _retry_command_suggestion(
        args,
        include_failed=args.failed
        or any(get_run_status(state, dep.id) == "failed" for dep in blocked),
        include_skipped=args.skipped
        or any(get_run_status(state, dep.id) == "skipped" for dep in blocked),
    )
    print("\nSuggested command:")
    print(f"  {suggested}")
    raise SystemExit(1)


def _warn_for_skipped_dependents_after_failed_retry(
    runs_to_retry: list[Run],
    all_runs: list[Run],
    state: dict,
    args,
) -> list[Run]:
    if not args.failed or args.skipped:
        return []

    skipped_dependents = _collect_skipped_dependent_closure(runs_to_retry, all_runs, state)
    if not skipped_dependents:
        return []

    print(
        "WARNING: retry --failed will leave skipped downstream runs selected for "
        "accounting only. Prefer retry --failed --skipped for final recovery."
    )
    print("\nSkipped dependents:")
    for run in skipped_dependents[:20]:
        print(f"  {run.id} [skipped]")
    if len(skipped_dependents) > 20:
        print(f"  ... {len(skipped_dependents) - 20} more")
    suggested = _retry_command_suggestion(args, include_failed=True, include_skipped=True)
    print("\nSuggested command:")
    print(f"  {suggested}")
    return skipped_dependents


def _filter_requested_runs(
    all_runs: list[Run],
    experiment_classes: list[str],
) -> list[Run]:
    requested = list(dict.fromkeys(experiment_classes))
    runs = [
        run
        for experiment_class in requested
        for run in all_runs
        if run.experiment_class == experiment_class
    ]
    run_ids = {run.id for run in runs}
    all_by_id = {run.id: run for run in all_runs}
    deps_needed = {
        dep_id
        for run in runs
        for dep_id in run.depends_on
        if dep_id not in run_ids and dep_id in all_by_id
    }
    deps = [run for run in all_runs if run.id in deps_needed]
    return deps + runs


def cmd_run(args):
    all_runs = generate_all_runs()
    state = load_state()
    recover_stale_running(state)

    runs = _filter_requested_runs(all_runs, args.experiment_class)
    if not runs:
        print(f"No runs found for experiment class(es): {', '.join(args.experiment_class)}")
        return 0

    if args.revision:
        runs = apply_revision(runs, args.revision, args.reason)
    runs = apply_wandb_target(runs, args.project, args.entity)
    launch_commit = getattr(args, "launch_commit", None)
    runs = apply_launch_commit(runs, launch_commit)

    print(f"Experiment class(es) {', '.join(args.experiment_class)}: {len(runs)} runs")
    if args.revision:
        print(f"Revision: {args.revision}" + (f" ({args.reason})" if args.reason else ""))
    if launch_commit:
        print(f"Launch commit: {launch_commit}")
    if args.project or args.entity:
        target = f"{args.entity}/{args.project}" if args.entity else args.project
        print(f"W&B target: {target}")
    return run_scheduler(runs, state, args.parallel, args.dry_run)


def cmd_status(args):
    print_status(args.experiment_class, args.revision)


def cmd_retry(args):
    all_runs = generate_all_runs()
    class_filter = set(args.experiment_class) if args.experiment_class else None

    if args.revision:
        revised = _filter_requested_runs(all_runs, args.experiment_class)
        revised_ids = {run.id for run in revised}
        rest = [run for run in all_runs if run.id not in revised_ids]
        revised = apply_revision(revised, args.revision, args.reason)
        all_runs = rest + revised

    all_runs = apply_wandb_target(all_runs, args.project, args.entity)
    launch_commit = getattr(args, "launch_commit", None)
    all_runs = apply_launch_commit(all_runs, launch_commit)
    state = load_state()
    recover_stale_running(state)

    candidate_runs = [
        run for run in all_runs if class_filter is None or run.experiment_class in class_filter
    ]
    runs_to_retry = []
    for run in candidate_runs:
        status = get_run_status(state, run.id)
        if args.failed and status == "failed":
            runs_to_retry.append(run)
        elif args.skipped and status == "skipped":
            runs_to_retry.append(run)

    if not runs_to_retry:
        print("No runs to retry.")
        return 0

    all_by_id = {run.id: run for run in all_runs}
    dependencies, required_by, missing = _collect_dependency_closure(runs_to_retry, all_by_id)
    blocked = [
        dep
        for dep in dependencies
        if (get_run_status(state, dep.id) == "failed" and not args.failed)
        or (get_run_status(state, dep.id) == "skipped" and not args.skipped)
    ]
    unresolved_missing = {
        dep_id for dep_id in missing if get_run_status(state, dep_id) != "completed"
    }
    if blocked or unresolved_missing:
        _fail_for_blocked_retry_dependencies(blocked, unresolved_missing, state, required_by, args)

    accounting_only = _warn_for_skipped_dependents_after_failed_retry(
        runs_to_retry,
        all_runs,
        state,
        args,
    )

    runs_by_id: dict[str, Run] = {}
    for run in dependencies + runs_to_retry + accounting_only:
        runs_by_id.setdefault(run.id, run)
    for run in dependencies + runs_to_retry:
        if get_run_status(state, run.id) in {"failed", "skipped"}:
            set_run_status(state, run.id, "pending")

    save_state(state)
    print(f"Retrying {len(runs_by_id)} runs")
    if args.project or args.entity:
        target = f"{args.entity}/{args.project}" if args.entity else args.project
        print(f"W&B target: {target}")
    if launch_commit:
        print(f"Launch commit: {launch_commit}")
    return run_scheduler(list(runs_by_id.values()), state, args.parallel, dry_run=False)


def cmd_warmup(args):
    if args.revision:
        print(f"Note: --revision={args.revision} ignored (warmup is revision-independent)")

    runs = _filter_requested_runs(generate_all_runs(), args.experiment_class)
    datasets = dict.fromkeys(run.dataset for run in runs)

    print(f"Warming up tensor caches for experiment class(es) {', '.join(args.experiment_class)}")
    print(f"  {len(datasets)} unique datasets to cache\n")

    failed = []
    for index, dataset in enumerate(datasets, 1):
        processed_dir = f"data/processed/{dataset}"
        print(f"[{index}/{len(datasets)}] {dataset}")
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                "from slices.data.dataset import ICUDataset; "
                f"ds = ICUDataset(data_dir={processed_dir!r}, task_name=None, normalize=False); "
                "print(len(ds))",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            n_samples = result.stdout.strip().split("\n")[-1]
            print(f"  -> Cached ({int(n_samples):,} samples)")
        else:
            stderr_tail = result.stderr.strip().split("\n")[-3:]
            print(f"  -> ERROR: {' '.join(stderr_tail)}")
            failed.append(dataset)

    if failed:
        print(f"\nWarmup failed for dataset(s): {', '.join(failed)}")
        return 1
    print("\nWarmup complete. Raw tensor caches saved to data/processed/<dataset>/.tensor_cache/")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="SLICES class-based experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run experiments for given experiment classes")
    p_run.add_argument("--experiment-class", nargs="+", required=True, choices=EXPERIMENT_CLASSES)
    p_run.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Max scheduler slots (default: 4). Pretrains consume 4 slots, other runs 1.",
    )
    p_run.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    p_run.add_argument("--revision", type=str, default=None, help="Revision name")
    p_run.add_argument("--reason", type=str, default=None, help="Reason for rerun")
    p_run.add_argument(
        "--launch-commit",
        type=str,
        default=os.environ.get("SLICES_LAUNCH_COMMIT"),
        help="Exact git commit hash for launch provenance (default: SLICES_LAUNCH_COMMIT)",
    )
    p_run.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT"),
        help="W&B project override for launched runs (default: WANDB_PROJECT env var)",
    )
    p_run.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity override for launched runs (default: WANDB_ENTITY env var)",
    )

    p_status = sub.add_parser("status", help="Show experiment status")
    p_status.add_argument("--experiment-class", nargs="*", choices=EXPERIMENT_CLASSES, default=None)
    p_status.add_argument("--revision", type=str, default=None, help="Show status for one revision")

    p_retry = sub.add_parser("retry", help="Retry failed/skipped runs")
    p_retry.add_argument("--failed", action="store_true", help="Retry failed runs")
    p_retry.add_argument(
        "--skipped",
        action="store_true",
        help="Retry dependency-skipped runs; prefer with --failed for final recovery",
    )
    p_retry.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Max scheduler slots (default: 4). Pretrains consume 4 slots, other runs 1.",
    )
    p_retry.add_argument("--experiment-class", nargs="+", choices=EXPERIMENT_CLASSES, default=None)
    p_retry.add_argument("--revision", type=str, default=None, help="Revision name to retry")
    p_retry.add_argument("--reason", type=str, default=None, help="Reason for rerun")
    p_retry.add_argument(
        "--launch-commit",
        type=str,
        default=os.environ.get("SLICES_LAUNCH_COMMIT"),
        help="Exact git commit hash for launch provenance (default: SLICES_LAUNCH_COMMIT)",
    )
    p_retry.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT"),
        help="W&B project override for relaunched runs (default: WANDB_PROJECT env var)",
    )
    p_retry.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity override for relaunched runs (default: WANDB_ENTITY env var)",
    )

    p_warmup = sub.add_parser(
        "warmup", help="Pre-build tensor caches to avoid OOM during parallel runs"
    )
    p_warmup.add_argument(
        "--experiment-class", nargs="+", required=True, choices=EXPERIMENT_CLASSES
    )
    p_warmup.add_argument(
        "--revision", type=str, default=None, help="Ignored (warmup is revision-independent)"
    )
    p_warmup.add_argument("--reason", type=str, default=None, help="Ignored for warmup")

    args = parser.parse_args()

    if getattr(args, "reason", None) and not getattr(args, "revision", None):
        parser.error("--reason requires --revision")
    if args.command == "run":
        if not args.revision:
            parser.error("run requires --revision to tag run IDs, output dirs, and W&B metadata")
        if not args.project:
            parser.error(
                "run requires --project or WANDB_PROJECT to avoid logging to config defaults"
            )
        if not args.entity:
            parser.error("run requires --entity or WANDB_ENTITY to make W&B ownership explicit")
    if args.command == "retry":
        if not args.revision:
            parser.error("retry requires --revision to select the revisioned state namespace")
        if not args.experiment_class:
            parser.error("retry requires --experiment-class to scope which classes to revise")
        if not args.project:
            parser.error(
                "retry requires --project or WANDB_PROJECT to avoid logging to config defaults"
            )
        if not args.entity:
            parser.error("retry requires --entity or WANDB_ENTITY to make W&B ownership explicit")
        if not args.failed and not args.skipped:
            parser.error("retry requires --failed and/or --skipped")

    validate_direct_final_launch_policy(args, parser)

    if args.command == "run":
        exit_code = cmd_run(args)
    elif args.command == "status":
        exit_code = cmd_status(args)
    elif args.command == "retry":
        exit_code = cmd_retry(args)
    elif args.command == "warmup":
        exit_code = cmd_warmup(args)
    else:
        exit_code = 0
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
