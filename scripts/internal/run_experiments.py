#!/usr/bin/env python3
"""
Parallel experiment runner for SLICES.

Generates all experiment configurations across sprints, resolves dependencies,
and executes them in parallel with crash recovery and state persistence.

Usage:
    uv run python scripts/internal/run_experiments.py warmup --sprint 1
    uv run python scripts/internal/run_experiments.py run --sprint 1 --parallel 4 \\
        --project slices-thesis --revision thesis-v1 --entity <entity>
    uv run python scripts/internal/run_experiments.py run --sprint 1 2 3 --parallel 6 \\
        --dry-run --project slices-thesis --revision thesis-v1 --entity <entity>
    uv run python scripts/internal/run_experiments.py run --sprint 1 \\
        --project slices-thesis --revision thesis-v1 --entity <entity> --reason "fix LR"
    uv run python scripts/internal/run_experiments.py run --sprint 1 2 \\
        --project slices-thesis --revision thesis-v1 --entity <entity>
    uv run python scripts/internal/run_experiments.py status
    uv run python scripts/internal/run_experiments.py status --sprint 1
    uv run python scripts/internal/run_experiments.py retry --failed --parallel 4 \\
        --project slices-thesis --revision thesis-v1 --entity <entity>
    uv run python scripts/internal/run_experiments.py retry --skipped --failed --parallel 4 \\
        --project slices-thesis --revision thesis-v1 --entity <entity>
    uv run python scripts/internal/run_experiments.py retry --failed \\
        --sprint 1 --revision thesis-v1 --parallel 4 \\
        --project slices-thesis --entity <entity>
    uv run python scripts/internal/run_experiments.py tag --sprint 2 --dry-run \\
        --project slices-thesis --entity <entity>
    uv run python scripts/internal/run_experiments.py tag --sprint 2 3 5 \\
        --project slices-thesis --entity <entity>
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SSL_PARADIGMS = ["mae", "jepa", "contrastive"]
DATASETS = ["miiv", "eicu", "combined"]
TASKS = ["mortality_24h", "mortality_hospital", "aki_kdigo", "los_remaining"]
SEEDS = [42, 123, 456]
SEEDS_EXTENDED = [42, 123, 456, 789, 1011]  # 5 seeds for cheap baselines
LABEL_FRACTIONS_FULL = [0.01, 0.05, 0.1, 0.25, 0.5]
LABEL_FRACTIONS_TREND = [0.1]
LABEL_FRACTIONS_PILOT = [0.01, 0.1, 0.5]

# Model capacity variants for Sprint 7 pilot (capacity ablation)
MODEL_SIZES = {
    "medium": {
        "model": "transformer_medium",
        "encoder": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff": 512},
        "ssl_scale": {
            "mae": {
                "ssl.decoder_d_model": 128,
                "ssl.decoder_n_layers": 2,
                "ssl.decoder_n_heads": 8,
                "ssl.decoder_d_ff": 512,
            },
            "jepa": {
                "ssl.predictor_d_model": 64,
                "ssl.predictor_n_layers": 2,
                "ssl.predictor_n_heads": 4,
                "ssl.predictor_d_ff": 256,
            },
            "contrastive": {
                "ssl.proj_hidden_dim": 512,
                "ssl.proj_output_dim": 128,
            },
        },
    },
    "large": {
        "model": "transformer_large",
        "encoder": {"d_model": 256, "n_layers": 4, "n_heads": 8, "d_ff": 1024},
        "ssl_scale": {
            "mae": {
                "ssl.decoder_d_model": 256,
                "ssl.decoder_n_layers": 2,
                "ssl.decoder_n_heads": 8,
                "ssl.decoder_d_ff": 1024,
            },
            "jepa": {
                "ssl.predictor_d_model": 128,
                "ssl.predictor_n_layers": 2,
                "ssl.predictor_n_heads": 8,
                "ssl.predictor_d_ff": 512,
            },
            "contrastive": {
                "ssl.proj_hidden_dim": 1024,
                "ssl.proj_output_dim": 256,
            },
        },
    },
}
LR_ABLATION = [2e-4, 5e-4, 2e-3]  # 1e-3 reused from Phase 1
MASK_RATIO_ABLATION = [0.3, 0.75]  # 0.5 reused from Phase 1
TRANSFER_PAIRS = [("miiv", "eicu"), ("eicu", "miiv")]

# Baseline inheritance: which earlier sprints' runs should be tagged as baselines
# for later sprints. See docs/internal/EXPERIMENT_PLAN.md § "Baseline Inheritance Across Sprints".
# Intentionally coarse-grained: ALL runs from source sprints are tagged, even when
# only a subset is relevant (e.g. 1b only needs mortality_24h from Sprint 1). Use
# secondary W&B tags (task:, protocol:, phase:, etc.) to filter down when querying.
BASELINE_SPRINTS: dict[str, list[str]] = {
    "1": [],
    "1b": ["1"],
    "1c": ["1"],
    "2": ["1"],
    "3": [],
    "4": [],
    "5": ["1", "2", "3", "4"],
    "6": ["1", "2", "3", "4", "5", "10"],
    "7": ["1", "3", "5", "10"],
    "7p": ["6", "10"],
    "8": ["1", "1b", "1c", "5", "10"],
}

STATE_FILE = Path("outputs/experiment_state.json")
LOG_DIR = Path("logs/runner")

# Protocol defaults
PROTO_A = {"freeze_encoder": True, "max_epochs": 50, "patience": 10, "lr": 1e-4}
PROTO_B = {"freeze_encoder": False, "max_epochs": 100, "patience": 10, "lr": 3e-4}


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------
@dataclass
class Run:
    id: str
    sprint: str
    run_type: str  # "pretrain" | "finetune" | "supervised" | "gru_d" | "xgboost"
    paradigm: str  # "mae" | "jepa" | "contrastive" | "supervised" | "gru_d" | "xgboost"
    dataset: str
    seed: int
    output_dir: str
    depends_on: list[str] = field(default_factory=list)
    task: str | None = None
    label_fraction: float = 1.0
    freeze_encoder: bool | None = None
    extra_overrides: dict = field(default_factory=dict)
    # For transfer learning: dataset the encoder was pretrained on
    source_dataset: str | None = None
    # Upstream pretrain metadata for provenance tracking in finetune runs
    upstream_pretrain_lr: float | None = None
    upstream_pretrain_mask_ratio: float | None = None
    experiment_subtype: str | None = None  # "lr_ablation" | "mask_ablation" | None

    def build_command(self, runs_by_id: dict[str, Run]) -> list[str]:
        """Build the subprocess command for this run."""
        if self.run_type == "pretrain":
            return self._pretrain_cmd()
        elif self.run_type == "finetune":
            return self._finetune_cmd(runs_by_id)
        elif self.run_type == "supervised":
            return self._supervised_cmd()
        elif self.run_type == "gru_d":
            return self._gru_d_cmd()
        elif self.run_type == "xgboost":
            return self._xgboost_cmd()
        else:
            raise ValueError(f"Unknown run_type: {self.run_type}")

    def _pretrain_cmd(self) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/pretrain.py",
            f"dataset={self.dataset}",
            f"ssl={self.paradigm}",
            f"seed={self.seed}",
            f"sprint={self.sprint}",
            f"hydra.run.dir={self.output_dir}",
        ]
        last_ckpt = Path(self.output_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            cmd.append(f"ckpt_path={last_ckpt}")
        for k, v in self.extra_overrides.items():
            cmd.append(f"{k}={v}")
        return cmd

    def _finetune_cmd(self, runs_by_id: dict[str, Run]) -> list[str]:
        # Find the pretrain dependency to get encoder path
        pretrain_id = [d for d in self.depends_on if d in runs_by_id]
        if not pretrain_id:
            raise ValueError(f"Finetune run {self.id} has no pretrain dependency")
        pretrain_dir = runs_by_id[pretrain_id[0]].output_dir

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/finetune.py",
            f"dataset={self.dataset}",
            f"checkpoint={pretrain_dir}/encoder.pt",
            f"tasks={self.task}",
            f"seed={self.seed}",
            f"sprint={self.sprint}",
            f"hydra.run.dir={self.output_dir}",
        ]
        last_ckpt = Path(self.output_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            cmd.append(f"ckpt_path={last_ckpt}")
        if self.freeze_encoder is True:
            cmd += [
                "training.freeze_encoder=true",
                f"training.max_epochs={PROTO_A['max_epochs']}",
                f"training.early_stopping_patience={PROTO_A['patience']}",
                f"optimizer.lr={PROTO_A['lr']}",
                "task.head_type=linear",
                "task.hidden_dims=[]",
                "task.dropout=0.0",
            ]
        elif self.freeze_encoder is False:
            cmd += [
                "training.freeze_encoder=false",
                f"training.max_epochs={PROTO_B['max_epochs']}",
                f"training.early_stopping_patience={PROTO_B['patience']}",
                f"optimizer.lr={PROTO_B['lr']}",
            ]
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        for k, v in self.extra_overrides.items():
            cmd.append(f"{k}={v}")
        if self.source_dataset is not None:
            cmd.append(f"+source_dataset={self.source_dataset}")
        # Propagate upstream pretrain metadata so it lands in W&B config
        if self.upstream_pretrain_lr is not None:
            cmd.append(f"+upstream_pretrain_lr={self.upstream_pretrain_lr}")
        if self.upstream_pretrain_mask_ratio is not None:
            cmd.append(f"+upstream_pretrain_mask_ratio={self.upstream_pretrain_mask_ratio}")
        if self.experiment_subtype is not None:
            cmd.append(f"+experiment_subtype={self.experiment_subtype}")
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
            f"sprint={self.sprint}",
            f"hydra.run.dir={self.output_dir}",
        ]
        last_ckpt = Path(self.output_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            cmd.append(f"ckpt_path={last_ckpt}")
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        for k, v in self.extra_overrides.items():
            cmd.append(f"{k}={v}")
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
            f"sprint={self.sprint}",
            f"hydra.run.dir={self.output_dir}",
        ]
        last_ckpt = Path(self.output_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            cmd.append(f"ckpt_path={last_ckpt}")
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        for k, v in self.extra_overrides.items():
            cmd.append(f"{k}={v}")
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
            f"sprint={self.sprint}",
            f"hydra.run.dir={self.output_dir}",
        ]
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        for k, v in self.extra_overrides.items():
            cmd.append(f"{k}={v}")
        return cmd


# ---------------------------------------------------------------------------
# Matrix Generation
# ---------------------------------------------------------------------------
def _pretrain_key(paradigm: str, dataset: str, seed: int, extra: dict | None = None) -> str:
    """Canonical key for deduplicating pretrain runs."""
    key = f"pretrain_{paradigm}_{dataset}_seed{seed}"
    if extra:
        for k, v in sorted(extra.items()):
            short_k = k.split(".")[-1]
            short_v = str(v).replace(".", "")
            key += f"_{short_k}{short_v}"
    return key


def _output_dir(sprint: str, name: str) -> str:
    return f"outputs/sprint{sprint}/{name}"


class MatrixBuilder:
    """Generates all Run objects across sprints with pretrain deduplication."""

    def __init__(self):
        self.runs: list[Run] = []
        self.pretrain_index: dict[str, Run] = {}  # canonical_key -> Run

    def _add_pretrain(
        self, sprint: str, paradigm: str, dataset: str, seed: int, extra: dict | None = None
    ) -> Run:
        """Add a pretrain run, deduplicating by canonical key."""
        extra = extra or {}
        key = _pretrain_key(paradigm, dataset, seed, extra)
        if key in self.pretrain_index:
            return self.pretrain_index[key]

        dir_name = _pretrain_key(paradigm, dataset, seed, extra)
        run = Run(
            id=f"s{sprint}_{dir_name}",
            sprint=sprint,
            run_type="pretrain",
            paradigm=paradigm,
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(sprint, dir_name),
            extra_overrides=extra,
        )
        self.runs.append(run)
        self.pretrain_index[key] = run
        return run

    def _get_pretrain(
        self, paradigm: str, dataset: str, seed: int, extra: dict | None = None
    ) -> Run:
        """Retrieve an existing pretrain run."""
        key = _pretrain_key(paradigm, dataset, seed, extra)
        if key not in self.pretrain_index:
            raise KeyError(f"Pretrain not found: {key}")
        return self.pretrain_index[key]

    def _add_finetune(
        self,
        sprint: str,
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
    ) -> Run:
        """Add a finetune run.

        Args:
            extra: Hydra overrides added to the finetune command AND name.
            name_extra: Dict used only for directory/ID disambiguation
                (e.g. pretrain ablation params). Not passed to command.
            upstream_pretrain_lr: LR used in the upstream pretrain run (for provenance).
            upstream_pretrain_mask_ratio: Mask ratio used in upstream pretrain run.
            experiment_subtype: "lr_ablation", "mask_ablation", or None.
        """
        prefix = "probe" if freeze else "finetune"
        name = f"{prefix}_{paradigm}_{task}_{dataset}_seed{seed}"
        if source_dataset:
            name += f"_from_{source_dataset}"
        if label_fraction < 1.0:
            frac_str = str(label_fraction).replace(".", "")
            name += f"_frac{frac_str}"
        # Append both extra and name_extra to the directory name
        for d in [name_extra, extra]:
            if d:
                for k, v in sorted(d.items()):
                    short_k = k.split(".")[-1]
                    short_v = str(v).replace(".", "")
                    name += f"_{short_k}{short_v}"

        run = Run(
            id=f"s{sprint}_{name}",
            sprint=sprint,
            run_type="finetune",
            paradigm=paradigm,
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(sprint, name),
            depends_on=[pretrain_run.id],
            task=task,
            label_fraction=label_fraction,
            freeze_encoder=freeze,
            extra_overrides=extra or {},
            source_dataset=source_dataset,
            upstream_pretrain_lr=upstream_pretrain_lr,
            upstream_pretrain_mask_ratio=upstream_pretrain_mask_ratio,
            experiment_subtype=experiment_subtype,
        )
        self.runs.append(run)
        return run

    def _add_supervised(
        self, sprint: str, dataset: str, seed: int, task: str, label_fraction: float = 1.0
    ) -> Run:
        name = f"supervised_{task}_{dataset}_seed{seed}"
        if label_fraction < 1.0:
            frac_str = str(label_fraction).replace(".", "")
            name += f"_frac{frac_str}"
        run = Run(
            id=f"s{sprint}_{name}",
            sprint=sprint,
            run_type="supervised",
            paradigm="supervised",
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(sprint, name),
            task=task,
            label_fraction=label_fraction,
        )
        self.runs.append(run)
        return run

    def _add_gru_d(
        self, sprint: str, dataset: str, seed: int, task: str, label_fraction: float = 1.0
    ) -> Run:
        name = f"gru_d_{task}_{dataset}_seed{seed}"
        if label_fraction < 1.0:
            frac_str = str(label_fraction).replace(".", "")
            name += f"_frac{frac_str}"
        run = Run(
            id=f"s{sprint}_{name}",
            sprint=sprint,
            run_type="gru_d",
            paradigm="gru_d",
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(sprint, name),
            task=task,
            label_fraction=label_fraction,
        )
        self.runs.append(run)
        return run

    def _add_xgboost(
        self, sprint: str, dataset: str, seed: int, task: str, label_fraction: float = 1.0
    ) -> Run:
        name = f"xgboost_{task}_{dataset}_seed{seed}"
        if label_fraction < 1.0:
            frac_str = str(label_fraction).replace(".", "")
            name += f"_frac{frac_str}"
        run = Run(
            id=f"s{sprint}_{name}",
            sprint=sprint,
            run_type="xgboost",
            paradigm="xgboost",
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(sprint, name),
            task=task,
            label_fraction=label_fraction,
        )
        self.runs.append(run)
        return run

    @staticmethod
    def _add_model_size_metadata(run: Run, size_name: str) -> Run:
        """Add Hydra metadata used by W&B naming/tagging without changing run IDs."""
        run.extra_overrides = {**run.extra_overrides, "+model_size": size_name}
        return run

    # --- Sprint builders ---

    def build_sprint1(self):
        """MIMIC, all tasks, Protocol B + supervised, seed=42.

        Classical baselines (GRU-D, XGBoost) are canonicalized in Sprint 11
        so the final thesis matrix reports them once rather than duplicating
        them across the early SSL/supervised sprints.
        """
        ds, seed, sprint = "miiv", 42, "1"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)

    def build_sprint1b(self):
        """LR sensitivity, MIMIC, mortality_24h, seed=42."""
        ds, seed, task, sprint = "miiv", 42, "mortality_24h", "1b"
        for p in SSL_PARADIGMS:
            for lr in LR_ABLATION:
                extra = {"optimizer.lr": lr}
                pt = self._add_pretrain(sprint, p, ds, seed, extra)
                self._add_finetune(
                    sprint,
                    p,
                    ds,
                    seed,
                    task,
                    False,
                    pt,
                    name_extra=extra,
                    upstream_pretrain_lr=lr,
                    experiment_subtype="lr_ablation",
                )

    def build_sprint1c(self):
        """Mask ratio sensitivity, MIMIC, mortality_24h, seed=42."""
        ds, seed, task, sprint = "miiv", 42, "mortality_24h", "1c"
        for p in SSL_PARADIGMS:
            for mr in MASK_RATIO_ABLATION:
                extra = {"ssl.mask_ratio": mr}
                pt = self._add_pretrain(sprint, p, ds, seed, extra)
                self._add_finetune(
                    sprint,
                    p,
                    ds,
                    seed,
                    task,
                    False,
                    pt,
                    name_extra=extra,
                    upstream_pretrain_mask_ratio=mr,
                    experiment_subtype="mask_ablation",
                )

    def build_sprint2(self):
        """MIMIC Protocol A, seed=42 — reuses Sprint 1 pretrains."""
        ds, seed, sprint = "miiv", 42, "2"
        for p in SSL_PARADIGMS:
            pt = self._get_pretrain(p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, True, pt)

    def build_sprint3(self):
        """eICU, both protocols + supervised, seed=42.

        Classical baselines are launched in Sprint 11 only.
        """
        ds, seed, sprint = "eicu", 42, "3"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
                self._add_finetune(sprint, p, ds, seed, task, True, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)

    def build_sprint4(self):
        """Combined, both protocols + supervised, seed=42.

        Classical baselines are launched in Sprint 11 only.
        """
        ds, seed, sprint = "combined", 42, "4"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
                self._add_finetune(sprint, p, ds, seed, task, True, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)

    def build_sprint5(self):
        """Seeds 123, 456 for datasets miiv, eicu, combined.

        Extends the SSL + supervised matrix to three seeds. Classical baselines
        are canonicalized in Sprint 11.
        """
        sprint = "5"
        for seed in [123, 456]:
            for ds in DATASETS:
                for p in SSL_PARADIGMS:
                    pt = self._add_pretrain(sprint, p, ds, seed)
                    for task in TASKS:
                        self._add_finetune(sprint, p, ds, seed, task, False, pt)
                        self._add_finetune(sprint, p, ds, seed, task, True, pt)
                for task in TASKS:
                    self._add_supervised(sprint, ds, seed, task)

    def build_sprint6(self):
        """Label efficiency ablation — reuses Phase 1 encoders.

        Includes SSL Protocol A/B plus the supervised Transformer baseline.
        Classical baselines are canonicalized separately in Sprint 11.
        """
        sprint = "6"
        for seed in SEEDS:
            for ds in DATASETS:
                for p in SSL_PARADIGMS:
                    pt = self._get_pretrain(p, ds, seed)
                    # mortality_24h gets full sweep
                    for frac in LABEL_FRACTIONS_FULL:
                        self._add_finetune(sprint, p, ds, seed, "mortality_24h", False, pt, frac)
                        self._add_finetune(sprint, p, ds, seed, "mortality_24h", True, pt, frac)
                    # Other tasks get trend check
                    for task in TASKS[1:]:
                        for frac in LABEL_FRACTIONS_TREND:
                            self._add_finetune(sprint, p, ds, seed, task, False, pt, frac)
                            self._add_finetune(sprint, p, ds, seed, task, True, pt, frac)
                # Supervised label efficiency
                for frac in LABEL_FRACTIONS_FULL:
                    self._add_supervised(sprint, ds, seed, "mortality_24h", frac)
                for task in TASKS[1:]:
                    for frac in LABEL_FRACTIONS_TREND:
                        self._add_supervised(sprint, ds, seed, task, frac)

    def build_sprint7p(self):
        """Focused model-capacity study for the thesis.

        MIIV only, mortality_24h only, MAE + supervised.
        Two larger model sizes (medium=128d/4L, large=256d/4L) at 3 label fractions
        across all five thesis seeds. Baseline comparisons come from Sprint 6
        (seeds 42/123/456) and Sprint 10 (seeds 789/1011), so the default-size
        baseline is inherited rather than rerun here.

        Total: 5 seeds × [2 pretrain + 2×3 finetune + 2×3 probe + 2×3 supervised] = 100 runs.
        """
        sprint = "7p"
        ds, task = "miiv", "mortality_24h"

        for seed in SEEDS_EXTENDED:
            for size_name, size_cfg in MODEL_SIZES.items():
                # Common model override for all runs at this size
                model_extra = {"model": size_cfg["model"]}

                # --- MAE pretrain (new encoder size, full data) ---
                pretrain_extra = {
                    **model_extra,
                    **size_cfg["ssl_scale"]["mae"],
                }
                pt = self._add_pretrain(sprint, "mae", ds, seed, extra=pretrain_extra)
                self._add_model_size_metadata(pt, size_name)

                # --- MAE finetune (Protocol B) + probe (Protocol A) ---
                finetune_extra = {**model_extra}
                for frac in LABEL_FRACTIONS_PILOT:
                    self._add_model_size_metadata(
                        self._add_finetune(
                            sprint,
                            "mae",
                            ds,
                            seed,
                            task,
                            False,
                            pt,
                            label_fraction=frac,
                            extra=finetune_extra,
                            name_extra={"size": size_name},
                        ),
                        size_name,
                    )
                    self._add_model_size_metadata(
                        self._add_finetune(
                            sprint,
                            "mae",
                            ds,
                            seed,
                            task,
                            True,
                            pt,
                            label_fraction=frac,
                            extra=finetune_extra,
                            name_extra={"size": size_name},
                        ),
                        size_name,
                    )

                # --- Supervised baseline (same larger encoder, from scratch) ---
                sup_extra = {**model_extra}
                for frac in LABEL_FRACTIONS_PILOT:
                    sup_name = f"supervised_{task}_{ds}_seed{seed}_{size_name}"
                    if frac < 1.0:
                        frac_str = str(frac).replace(".", "")
                        sup_name += f"_frac{frac_str}"
                    run = Run(
                        id=f"s{sprint}_{sup_name}",
                        sprint=sprint,
                        run_type="supervised",
                        paradigm="supervised",
                        dataset=ds,
                        seed=seed,
                        output_dir=_output_dir(sprint, sup_name),
                        task=task,
                        label_fraction=frac,
                        extra_overrides=sup_extra,
                    )
                    self._add_model_size_metadata(run, size_name)
                    self.runs.append(run)

    def build_sprint7(self):
        """Cross-dataset transfer — Protocol B only (full finetune)."""
        sprint = "7"
        for seed in SEEDS:
            for source_ds, target_ds in TRANSFER_PAIRS:
                for p in SSL_PARADIGMS:
                    pt = self._get_pretrain(p, source_ds, seed)
                    for task in TASKS:
                        self._add_finetune(
                            sprint, p, target_ds, seed, task, False, pt, source_dataset=source_ds
                        )

    def build_sprint8(self):
        """LR + mask ablations, seeds 123, 456."""
        sprint = "8"
        for seed in [123, 456]:
            for p in SSL_PARADIGMS:
                # LR ablation
                for lr in LR_ABLATION:
                    extra = {"optimizer.lr": lr}
                    pt = self._add_pretrain(sprint, p, "miiv", seed, extra)
                    self._add_finetune(
                        sprint,
                        p,
                        "miiv",
                        seed,
                        "mortality_24h",
                        False,
                        pt,
                        name_extra=extra,
                        upstream_pretrain_lr=lr,
                        experiment_subtype="lr_ablation",
                    )
                # Mask ratio ablation
                for mr in MASK_RATIO_ABLATION:
                    extra = {"ssl.mask_ratio": mr}
                    pt = self._add_pretrain(sprint, p, "miiv", seed, extra)
                    self._add_finetune(
                        sprint,
                        p,
                        "miiv",
                        seed,
                        "mortality_24h",
                        False,
                        pt,
                        name_extra=extra,
                        upstream_pretrain_mask_ratio=mr,
                        experiment_subtype="mask_ablation",
                    )

    def build_sprint10(self):
        """Seeds 789, 1011 for Sprints 1-8 scope (SSL + supervised)."""
        sprint = "10"
        new_seeds = [789, 1011]

        # --- Core experiments (Sprint 1-4 scope) ---
        for seed in new_seeds:
            for ds in DATASETS:
                for p in SSL_PARADIGMS:
                    pt = self._add_pretrain(sprint, p, ds, seed)
                    for task in TASKS:
                        self._add_finetune(sprint, p, ds, seed, task, False, pt)
                        self._add_finetune(sprint, p, ds, seed, task, True, pt)
                for task in TASKS:
                    self._add_supervised(sprint, ds, seed, task)

        # --- Label efficiency (Sprint 6 scope) ---
        for seed in new_seeds:
            for ds in DATASETS:
                for p in SSL_PARADIGMS:
                    pt = self._get_pretrain(p, ds, seed)
                    for frac in LABEL_FRACTIONS_FULL:
                        self._add_finetune(sprint, p, ds, seed, "mortality_24h", False, pt, frac)
                        self._add_finetune(sprint, p, ds, seed, "mortality_24h", True, pt, frac)
                    for task in TASKS[1:]:
                        for frac in LABEL_FRACTIONS_TREND:
                            self._add_finetune(sprint, p, ds, seed, task, False, pt, frac)
                            self._add_finetune(sprint, p, ds, seed, task, True, pt, frac)
                for frac in LABEL_FRACTIONS_FULL:
                    self._add_supervised(sprint, ds, seed, "mortality_24h", frac)
                for task in TASKS[1:]:
                    for frac in LABEL_FRACTIONS_TREND:
                        self._add_supervised(sprint, ds, seed, task, frac)

        # --- Cross-dataset transfer (Sprint 7 scope) ---
        for seed in new_seeds:
            for source_ds, target_ds in TRANSFER_PAIRS:
                for p in SSL_PARADIGMS:
                    pt = self._get_pretrain(p, source_ds, seed)
                    for task in TASKS:
                        self._add_finetune(
                            sprint,
                            p,
                            target_ds,
                            seed,
                            task,
                            False,
                            pt,
                            source_dataset=source_ds,
                        )

        # --- HP ablations (Sprint 8 scope) ---
        for seed in new_seeds:
            for p in SSL_PARADIGMS:
                for lr in LR_ABLATION:
                    extra = {"optimizer.lr": lr}
                    pt = self._add_pretrain(sprint, p, "miiv", seed, extra)
                    self._add_finetune(
                        sprint,
                        p,
                        "miiv",
                        seed,
                        "mortality_24h",
                        False,
                        pt,
                        name_extra=extra,
                        upstream_pretrain_lr=lr,
                        experiment_subtype="lr_ablation",
                    )
                for mr in MASK_RATIO_ABLATION:
                    extra = {"ssl.mask_ratio": mr}
                    pt = self._add_pretrain(sprint, p, "miiv", seed, extra)
                    self._add_finetune(
                        sprint,
                        p,
                        "miiv",
                        seed,
                        "mortality_24h",
                        False,
                        pt,
                        name_extra=extra,
                        upstream_pretrain_mask_ratio=mr,
                        experiment_subtype="mask_ablation",
                    )

    def build_sprint11(self):
        """Canonical classical baselines (XGBoost + GRU-D), 5 seeds.

        These baselines are intentionally centralized here rather than being
        duplicated across earlier sprints. That keeps the final thesis matrix
        aligned with the docs and export logic: one canonical experiment family
        for contextual non-SSL baselines.
        """
        sprint = "11"
        for seed in SEEDS_EXTENDED:
            for ds in DATASETS:
                for task in TASKS:
                    self._add_xgboost(sprint, ds, seed, task)
                    self._add_gru_d(sprint, ds, seed, task)
                # Baseline label-efficiency (mirrors Sprint 6 baseline scope)
                for frac in LABEL_FRACTIONS_FULL:
                    self._add_gru_d(sprint, ds, seed, "mortality_24h", frac)
                    self._add_xgboost(sprint, ds, seed, "mortality_24h", frac)
                for task_name in TASKS[1:]:
                    for frac in LABEL_FRACTIONS_TREND:
                        self._add_gru_d(sprint, ds, seed, task_name, frac)
                        self._add_xgboost(sprint, ds, seed, task_name, frac)

    def build_sprint12(self):
        """SMART (NeurIPS 2024) as external SSL reference, 5 seeds.

        SMART uses its own encoder (MART architecture, d_model=32) and element-wise
        masking — NOT part of the controlled comparison (different architecture).
        Included as an external SSL SOTA reference point in the appendix.
        """
        sprint = "12"
        # model=smart selects the MART encoder; ssl=smart is set via paradigm arg
        pretrain_extra = {"model": "smart"}
        finetune_extra = {"model": "smart"}

        for seed in SEEDS_EXTENDED:
            for ds in DATASETS:
                # Pretrain (ssl=smart is set by _pretrain_cmd via self.paradigm)
                pt = self._add_pretrain(sprint, "smart", ds, seed, pretrain_extra)
                # Both protocols + all tasks
                for task in TASKS:
                    self._add_finetune(
                        sprint,
                        "smart",
                        ds,
                        seed,
                        task,
                        False,
                        pt,
                        extra=finetune_extra,
                    )
                    self._add_finetune(
                        sprint,
                        "smart",
                        ds,
                        seed,
                        task,
                        True,
                        pt,
                        extra=finetune_extra,
                    )

    def build_sprint13(self):
        """TS2Vec temporal contrastive variant, 5 seeds, both protocols.

        Addresses the "contrastive was set up to fail" vulnerability by giving
        the contrastive paradigm its natural augmentations (noise + masking)
        and a temporal contrastive loss. Same encoder, same training budget.
        Evaluated under both Protocol A (probe) and Protocol B (full finetune)
        so it fits the main thesis A/B framing.
        """
        sprint = "13"
        for seed in SEEDS_EXTENDED:
            for ds in DATASETS:
                pt = self._add_pretrain(sprint, "ts2vec", ds, seed)
                for task in TASKS:
                    self._add_finetune(sprint, "ts2vec", ds, seed, task, False, pt)
                    self._add_finetune(sprint, "ts2vec", ds, seed, task, True, pt)

    def build_all(self) -> list[Run]:
        """Build full experiment matrix. Order matters for dedup."""
        self.build_sprint1()
        self.build_sprint1b()
        self.build_sprint1c()
        self.build_sprint2()
        self.build_sprint3()
        self.build_sprint4()
        self.build_sprint5()
        self.build_sprint6()
        self.build_sprint7p()
        self.build_sprint7()
        self.build_sprint8()
        self.build_sprint10()
        self.build_sprint11()
        self.build_sprint12()
        self.build_sprint13()
        return self.runs


def generate_all_runs() -> list[Run]:
    builder = MatrixBuilder()
    return builder.build_all()


def apply_revision(runs: list[Run], revision: str, reason: str | None = None) -> list[Run]:
    """Post-process runs to inject revision into IDs, output dirs, and overrides.

    Rewrites:
    - Run IDs: s1_pretrain_... → s1_rev-v2_pretrain_...
    - Output dirs: outputs/sprint1/... → outputs/sprint1_rev-v2/...
    - depends_on: remap references within the revised set
    - extra_overrides: inject revision= and rerun_reason=
    """
    old_to_new: dict[str, str] = {}

    for r in runs:
        old_id = r.id
        # Insert rev-<name> after the sprint prefix (e.g. s1_ → s1_rev-v2_)
        prefix = f"s{r.sprint}_"
        if old_id.startswith(prefix):
            new_id = f"{prefix}rev-{revision}_{old_id[len(prefix) :]}"
        else:
            new_id = f"rev-{revision}_{old_id}"
        old_to_new[old_id] = new_id

        r.id = new_id
        # Rewrite output dir: sprint1/ → sprint1_rev-v2/
        r.output_dir = r.output_dir.replace(
            f"outputs/sprint{r.sprint}/",
            f"outputs/sprint{r.sprint}_rev-{revision}/",
        )
        r.extra_overrides["revision"] = revision
        if reason:
            r.extra_overrides["rerun_reason"] = reason

    # Remap depends_on references within the revised set
    for r in runs:
        r.depends_on = [old_to_new.get(d, d) for d in r.depends_on]

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


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"version": 1, "runs": {}}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write
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


def get_run_status(state: dict, run_id: str) -> str:
    return state["runs"].get(run_id, {}).get("status", "pending")


def set_run_status(state: dict, run_id: str, status: str, **kwargs):
    if run_id not in state["runs"]:
        state["runs"][run_id] = {}
    entry = state["runs"][run_id]
    entry["status"] = status
    entry.update(kwargs)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _pid_matches_command(pid: int, expected_command: str | None) -> bool:
    """Best-effort guard against PID reuse before trusting a running state."""
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
    if not actual_command:
        return False
    return expected_command == actual_command or expected_command in actual_command


def recover_stale_running(state: dict):
    """Reset running entries whose PIDs are dead or no longer match the run."""
    for run_id, info in state["runs"].items():
        if info.get("status") == "running":
            pid = info.get("pid")
            if (
                pid is None
                or not is_pid_alive(pid)
                or not _pid_matches_command(
                    int(pid),
                    info.get("command"),
                )
            ):
                info["status"] = "pending"
                info.pop("pid", None)
                info.pop("command", None)
                print(f"  Recovered stale run: {run_id}")


# Scheduler slot weights. Pretraining occupies the full default budget of 4
# slots so `--parallel 4` does not mix heavyweight pretrains with downstream
# GPU jobs on a single device.
RUN_SLOT_COSTS = {
    "pretrain": 4,
    "finetune": 1,
    "supervised": 1,
    "gru_d": 1,
    "xgboost": 1,
}


def _slot_cost(run: Run, slot_budget: int) -> int:
    """Return the scheduler slot cost for a run under the active budget."""
    return min(RUN_SLOT_COSTS.get(run.run_type, 1), slot_budget)


def _select_ready_runs(
    ready: list[Run],
    active_run_ids: set[str],
    runs_by_id: dict[str, Run],
    slot_budget: int,
) -> list[Run]:
    """Choose a launch batch that fits the available scheduler slots.

    Heavier runs are prioritized first so pretrains claim the budget before
    lighter downstream jobs, avoiding accidental co-scheduling on one GPU.
    """
    active_slots = sum(_slot_cost(runs_by_id[rid], slot_budget) for rid in active_run_ids)
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

    runs_by_id = {r.id: r for r in runs}
    active: dict[str, subprocess.Popen] = {}  # run_id -> Popen
    shutting_down = False

    def handle_signal(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        for rid, proc in active.items():
            try:
                proc.terminate()
            except OSError:
                pass
        # Wait up to 30s for children
        deadline = time.time() + 30
        for rid, proc in list(active.items()):
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Reset interrupted runs to pending
        for rid in active:
            set_run_status(state, rid, "pending")
        save_state(state)
        print("State saved. Interrupted runs reset to pending.")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _print_dry_run(runs, runs_by_id)
        return 0

    print(f"Scheduler started (slot_budget={parallel})")
    print(f"State file: {STATE_FILE}")
    print()

    while not shutting_down:
        # 1. Poll active processes
        finished = []
        for rid, proc in active.items():
            ret = proc.poll()
            if ret is not None:
                finished.append((rid, ret))

        # 2. Handle completions
        for rid, exit_code in finished:
            proc = active.pop(rid)
            fh = getattr(proc, "_log_fh", None)
            if fh:
                fh.close()
            run = runs_by_id[rid]
            now = datetime.now(timezone.utc).isoformat()

            elapsed = _format_elapsed(state, rid, now)

            if exit_code == 0:
                # For pretrain, verify encoder.pt exists
                if run.run_type == "pretrain":
                    encoder_path = Path(run.output_dir) / "encoder.pt"
                    if not encoder_path.exists():
                        print(f"  FAILED {rid}: exit 0 but encoder.pt missing {elapsed}")
                        set_run_status(state, rid, "failed", finished_at=now, exit_code=exit_code)
                        _propagate_failure(rid, runs, state)
                        continue
                print(f"  DONE {rid} {elapsed}")
                set_run_status(state, rid, "completed", finished_at=now, exit_code=0)
            else:
                print(f"  FAILED {rid} (exit {exit_code}) {elapsed}")
                set_run_status(state, rid, "failed", finished_at=now, exit_code=exit_code)
                _propagate_failure(rid, runs, state)

        # 3. Find ready runs
        ready = []
        for r in runs:
            status = get_run_status(state, r.id)
            if status != "pending":
                continue
            if r.id in active:
                continue
            # Check all dependencies completed
            deps_ok = all(get_run_status(state, d) == "completed" for d in r.depends_on)
            if deps_ok:
                ready.append(r)

        # 4. Launch runs that fit the remaining scheduler slots
        launch_batch = _select_ready_runs(ready, set(active), runs_by_id, parallel)
        for r in launch_batch:
            cmd = r.build_command(runs_by_id)
            log_file = LOG_DIR / f"{r.id}.log"
            now = datetime.now(timezone.utc).isoformat()

            print(f"  START {r.id}")
            log_fh = open(log_file, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Store log handle for cleanup
            proc._log_fh = log_fh  # type: ignore[attr-defined]
            active[r.id] = proc
            set_run_status(
                state,
                r.id,
                "running",
                started_at=now,
                pid=proc.pid,
                log_file=str(log_file),
                command=shlex.join(cmd),
            )

        # 5. Save state
        save_state(state)

        # 6. Exit condition
        if not active and not ready:
            break

        time.sleep(2)

    # Close any remaining log handles
    for proc in active.values():
        fh = getattr(proc, "_log_fh", None)
        if fh:
            fh.close()

    # Print summary
    print()
    _print_summary(runs, state)
    return _scheduler_exit_code(runs, state)


def _format_elapsed(state: dict, run_id: str, now_iso: str) -> str:
    """Format elapsed time as (Xh Ym Zs)."""
    started = state["runs"].get(run_id, {}).get("started_at")
    if not started:
        return ""
    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(now_iso)
        secs = int((t1 - t0).total_seconds())
        if secs < 60:
            return f"({secs}s)"
        elif secs < 3600:
            return f"({secs // 60}m{secs % 60:02d}s)"
        else:
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            return f"({h}h{m:02d}m{s:02d}s)"
    except (ValueError, TypeError):
        return ""


def _propagate_failure(failed_id: str, runs: list[Run], state: dict):
    """Skip runs that depend on a failed run (transitive)."""
    queue = [failed_id]
    while queue:
        fid = queue.pop(0)
        for r in runs:
            if fid in r.depends_on and get_run_status(state, r.id) == "pending":
                set_run_status(state, r.id, "skipped", reason=f"dependency {fid} failed")
                print(f"  SKIP {r.id} (dep {fid} failed)")
                queue.append(r.id)


def _print_dry_run(runs: list[Run], runs_by_id: dict[str, Run]):
    """Print all runs and their commands without executing."""
    print(f"DRY RUN: {len(runs)} runs\n")
    for r in runs:
        deps = ", ".join(r.depends_on) if r.depends_on else "(none)"
        cmd = shlex.join(r.build_command(runs_by_id))
        print(f"[{r.id}]")
        print(f"  sprint={r.sprint}  type={r.run_type}  deps={deps}")
        print(f"  dir={r.output_dir}")
        print(f"  cmd: {cmd}")
        print()


# ---------------------------------------------------------------------------
# Status / Retry
# ---------------------------------------------------------------------------
def _extract_revision_from_id(run_id: str) -> str | None:
    """Extract revision name from a run ID (e.g. 's1_rev-v2_pretrain_...' → 'v2')."""
    m = re.search(r"_rev-([^_]+)_", run_id)
    return m.group(1) if m else None


def _sprint_display(sprint: str, revision: str | None) -> str:
    """Format sprint column, e.g. '1' or '1/v2'."""
    if revision:
        return f"{sprint}/{revision}"
    return sprint


def _extract_sprint_from_id(run_id: str) -> str | None:
    """Extract sprint from a run ID (e.g. 's1_rev-v2_pretrain_...' → '1')."""
    m = re.match(r"s([^_]+)_", run_id)
    return m.group(1) if m else None


def print_status(sprint_filter: list[str] | None = None):
    all_runs = generate_all_runs()
    state = load_state()
    generated_ids = {r.id for r in all_runs}

    # Group generated runs by (sprint, revision=None)
    groups: dict[tuple[str, str | None], list[str]] = {}
    for r in all_runs:
        key = (r.sprint, None)
        groups.setdefault(key, []).append(r.id)

    # Add revised runs from state (not in generated set) by parsing their IDs
    for run_id in state.get("runs", {}):
        if run_id not in generated_ids and "_rev-" in run_id:
            sprint = _extract_sprint_from_id(run_id)
            rev = _extract_revision_from_id(run_id)
            if sprint and rev:
                key = (sprint, rev)
                groups.setdefault(key, []).append(run_id)

    group_keys = sorted(groups.keys(), key=lambda k: (_sprint_sort_key(k[0]), k[1] or ""))
    if sprint_filter:
        group_keys = [k for k in group_keys if k[0] in sprint_filter]

    print(
        f"{'Sprint':>8} | {'Total':>5} | {'Done':>4} | {'Run':>3} | "
        f"{'Fail':>4} | {'Pend':>4} | {'Skip':>4}"
    )
    print("-" * 54)

    totals = {"total": 0, "completed": 0, "running": 0, "failed": 0, "pending": 0, "skipped": 0}

    for sprint, rev in group_keys:
        run_ids = groups[(sprint, rev)]
        label = _sprint_display(sprint, rev)
        counts = {
            "total": len(run_ids),
            "completed": 0,
            "running": 0,
            "failed": 0,
            "pending": 0,
            "skipped": 0,
        }
        for rid in run_ids:
            status = get_run_status(state, rid)
            if status in counts:
                counts[status] += 1
            else:
                counts["pending"] += 1  # unknown → pending
        print(
            f"{label:>8} | {counts['total']:>5} | {counts['completed']:>4} | "
            f"{counts['running']:>3} | {counts['failed']:>4} | "
            f"{counts['pending']:>4} | {counts['skipped']:>4}"
        )
        for k in totals:
            totals[k] += counts[k]

    print("-" * 54)
    print(
        f"{'TOTAL':>8} | {totals['total']:>5} | {totals['completed']:>4} | "
        f"{totals['running']:>3} | {totals['failed']:>4} | "
        f"{totals['pending']:>4} | {totals['skipped']:>4}"
    )


def _print_summary(runs: list[Run], state: dict):
    """Print completion summary for a scheduler run."""
    counts = {"completed": 0, "failed": 0, "skipped": 0, "pending": 0}
    for r in runs:
        s = get_run_status(state, r.id)
        counts[s] = counts.get(s, 0) + 1
    total = len(runs)
    print(
        f"Summary: {counts['completed']}/{total} completed, "
        f"{counts['failed']} failed, {counts['skipped']} skipped, "
        f"{counts['pending']} pending"
    )


def _scheduler_exit_code(runs: list[Run], state: dict) -> int:
    """Return nonzero when the requested run set did not complete cleanly."""
    failed = [r.id for r in runs if get_run_status(state, r.id) == "failed"]
    skipped = [r.id for r in runs if get_run_status(state, r.id) == "skipped"]
    running = [r.id for r in runs if get_run_status(state, r.id) == "running"]
    pending = [r.id for r in runs if get_run_status(state, r.id) == "pending"]

    if failed or skipped or running or pending:
        print(
            "Scheduler incomplete: "
            f"{len(failed)} failed, {len(skipped)} skipped due to dependency failure, "
            f"{len(running)} running, {len(pending)} pending"
        )
        return 1

    return 0


def _sprint_sort_key(s: str) -> tuple[int, str]:
    """Sort sprints: 1, 1b, 2, 3, ..."""
    num = ""
    suffix = ""
    for c in s:
        if c.isdigit():
            num += c
        else:
            suffix += c
    return (int(num) if num else 0, suffix)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_run(args):
    all_runs = generate_all_runs()
    state = load_state()
    recover_stale_running(state)

    # Filter to requested sprints
    sprints = [str(s) for s in args.sprint]
    runs = [r for r in all_runs if r.sprint in sprints]

    # Also include dependency runs from earlier sprints (pretrain reuse)
    run_ids = {r.id for r in runs}
    deps_needed = set()
    for r in runs:
        for d in r.depends_on:
            if d not in run_ids:
                deps_needed.add(d)
    if deps_needed:
        all_by_id = {r.id: r for r in all_runs}
        extra = [all_by_id[d] for d in deps_needed if d in all_by_id]
        runs = extra + runs
        if extra and not args.dry_run:
            print(f"Including {len(extra)} dependency run(s) from earlier sprints")

    if not runs:
        print(f"No runs found for sprint(s): {', '.join(sprints)}")
        return 0

    # Apply revision transform if requested
    if args.revision:
        runs = apply_revision(runs, args.revision, args.reason)
    runs = apply_wandb_target(runs, args.project, args.entity)
    runs = apply_launch_commit(runs, args.launch_commit)

    print(f"Sprint(s) {', '.join(sprints)}: {len(runs)} runs")
    if args.revision:
        print(f"Revision: {args.revision}" + (f" ({args.reason})" if args.reason else ""))
    if args.launch_commit:
        print(f"Launch commit: {args.launch_commit}")
    if args.project or args.entity:
        target = f"{args.entity}/{args.project}" if args.entity else args.project
        print(f"W&B target: {target}")
    return run_scheduler(runs, state, args.parallel, args.dry_run)


def cmd_status(args):
    sprint_filter = [str(s) for s in args.sprint] if args.sprint else None
    print_status(sprint_filter)


def _collect_dependency_closure(
    runs: list[Run],
    all_by_id: dict[str, Run],
) -> tuple[list[Run], dict[str, set[str]], set[str]]:
    """Return transitive dependencies for runs, with provenance for reporting."""
    deps: list[Run] = []
    required_by: dict[str, set[str]] = {}
    missing: set[str] = set()
    seen: set[str] = set()
    queue: list[tuple[str, str]] = [(dep_id, run.id) for run in runs for dep_id in run.depends_on]

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


def _retry_command_suggestion(args, *, include_failed: bool, include_skipped: bool) -> str:
    """Build a retry command that preserves the user's current retry scope."""
    cmd = ["uv", "run", "python", "scripts/internal/run_experiments.py", "retry"]
    if include_failed:
        cmd.append("--failed")
    if include_skipped:
        cmd.append("--skipped")
    if args.sprint:
        cmd.append("--sprint")
        cmd.extend(str(s) for s in args.sprint)
    if args.revision:
        cmd.extend(["--revision", str(args.revision)])
    if args.reason:
        cmd.extend(["--reason", str(args.reason)])
    if args.project:
        cmd.extend(["--project", str(args.project)])
    if args.entity:
        cmd.extend(["--entity", str(args.entity)])
    if args.launch_commit:
        cmd.extend(["--launch-commit", str(args.launch_commit)])
    cmd.extend(["--parallel", str(args.parallel)])
    return shlex.join(cmd)


def _fail_for_blocked_retry_dependencies(
    blocked: list[Run],
    missing: set[str],
    state: dict,
    required_by: dict[str, set[str]],
    args,
) -> None:
    """Print a dependency report for retry requests that cannot make progress."""
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


def cmd_retry(args):
    all_runs = generate_all_runs()
    sprint_filter = {str(s) for s in args.sprint} if args.sprint else None

    # Apply revision if specified (must happen before ID matching against state)
    if args.revision:
        if sprint_filter:
            revised = [r for r in all_runs if r.sprint in sprint_filter]
            revised_ids = {r.id for r in revised}
            all_by_id = {r.id: r for r in all_runs}
            deps_needed = {
                dep_id
                for r in revised
                for dep_id in r.depends_on
                if dep_id not in revised_ids and dep_id in all_by_id
            }
            revised = [all_by_id[dep_id] for dep_id in deps_needed] + revised
            revised_ids = {r.id for r in revised}
            rest = [r for r in all_runs if r.id not in revised_ids]
            revised = apply_revision(revised, args.revision, args.reason)
            all_runs = rest + revised
        else:
            all_runs = apply_revision(all_runs, args.revision, args.reason)
    all_runs = apply_wandb_target(all_runs, args.project, args.entity)
    all_runs = apply_launch_commit(all_runs, args.launch_commit)

    state = load_state()
    recover_stale_running(state)

    candidate_runs = [r for r in all_runs if sprint_filter is None or r.sprint in sprint_filter]
    runs_to_retry = []
    for r in candidate_runs:
        status = get_run_status(state, r.id)
        if args.failed and status == "failed":
            runs_to_retry.append(r)
        elif args.skipped and status == "skipped":
            runs_to_retry.append(r)

    if not runs_to_retry:
        print("No runs to retry.")
        return 0

    all_by_id = {r.id: r for r in all_runs}
    dependencies, required_by, missing = _collect_dependency_closure(runs_to_retry, all_by_id)

    # Failed/skipped dependencies cannot satisfy downstream runs. Treat --failed
    # and --skipped as the explicit request to reset those dependency states too.
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
        _fail_for_blocked_retry_dependencies(
            blocked,
            unresolved_missing,
            state,
            required_by,
            args,
        )

    runs_by_retry_id: dict[str, Run] = {}
    for run in dependencies + runs_to_retry:
        runs_by_retry_id.setdefault(run.id, run)

    for run in runs_by_retry_id.values():
        if get_run_status(state, run.id) in {"failed", "skipped"}:
            set_run_status(state, run.id, "pending")

    runs_to_retry = list(runs_by_retry_id.values())

    save_state(state)
    print(f"Retrying {len(runs_to_retry)} runs")
    if args.project or args.entity:
        target = f"{args.entity}/{args.project}" if args.entity else args.project
        print(f"W&B target: {target}")
    if args.launch_commit:
        print(f"Launch commit: {args.launch_commit}")
    return run_scheduler(runs_to_retry, state, args.parallel, dry_run=False)


def cmd_warmup(args):
    """Pre-build raw tensor caches for requested sprints.

    Creates one raw tensor cache per dataset (not per task/seed/fraction).
    The cache stores unnormalized tensors; normalization is applied at runtime.
    This means only ~0.8GB total cache instead of 140GB+.
    """
    if args.revision:
        print(f"Note: --revision={args.revision} ignored (warmup is revision-independent)")

    all_runs = generate_all_runs()

    # Filter to requested sprints
    sprints = [str(s) for s in args.sprint]
    runs = [r for r in all_runs if r.sprint in sprints]

    # Also include dependency runs
    run_ids = {r.id for r in runs}
    deps_needed = set()
    for r in runs:
        for d in r.depends_on:
            if d not in run_ids:
                deps_needed.add(d)
    if deps_needed:
        all_by_id = {r.id: r for r in all_runs}
        runs = [all_by_id[d] for d in deps_needed if d in all_by_id] + runs

    # Collect unique datasets — raw tensor cache is per-dataset only
    datasets: dict[str, None] = {}  # ordered set via dict
    for r in runs:
        datasets[r.dataset] = None

    print(f"Warming up tensor caches for sprint(s) {', '.join(sprints)}")
    print(f"  {len(datasets)} unique datasets to cache\n")

    failed = []
    for i, dataset in enumerate(datasets, 1):
        processed_dir = f"data/processed/{dataset}"
        print(f"[{i}/{len(datasets)}] {dataset}")

        # Run in a subprocess so that memory is fully returned to the OS.
        # Uses task_name=None (unsupervised) to load ALL stays without
        # any task-specific filtering — this populates the raw cache.
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                "from slices.data.dataset import ICUDataset; "
                f"ds = ICUDataset("
                f"  data_dir={processed_dir!r},"
                f"  task_name=None,"
                f"  normalize=False,"
                f"); "
                f"print(len(ds))",
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
    print("You can now run experiments in parallel without OOM.")
    return 0


def cmd_tag(args):
    """Add sprint tags to inherited baseline runs in W&B.

    For each requested sprint, finds runs from inherited baseline sprints
    (per BASELINE_SPRINTS mapping) and adds the target sprint tag via the
    W&B API. Idempotent — already-tagged runs are skipped.

    Usage:
        uv run python scripts/internal/run_experiments.py tag --sprint 2 \\
            --project slices-thesis --entity <entity>
        uv run python scripts/internal/run_experiments.py tag --sprint 2 3 --dry-run \\
            --project slices-thesis --entity <entity>
        uv run python scripts/internal/run_experiments.py tag --sprint 2 \\
            --project slices-thesis --entity <entity>
    """
    try:
        import wandb  # noqa: F811
    except ImportError:
        print("Error: wandb is required for tagging. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api()

    project = args.project
    entity = args.entity
    # Resolve entity: CLI flag > WANDB_ENTITY env var > wandb default entity
    if entity is None:
        entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    if entity is None:
        print("Error: Could not determine W&B entity. Set --entity or WANDB_ENTITY env var.")
        sys.exit(1)

    sprints = [str(s) for s in args.sprint]
    dry_run = args.dry_run
    revision_tag = f"revision:{args.revision}" if args.revision else None

    for target_sprint in sprints:
        source_sprints = BASELINE_SPRINTS.get(target_sprint, [])
        if not source_sprints:
            print(f"Sprint {target_sprint}: no baseline sprints to inherit from — skipping.")
            continue

        target_tag = f"sprint:{target_sprint}"
        sources = ", ".join(source_sprints)
        print(f"\nSprint {target_sprint}: inheriting baselines from sprint(s) {sources}")
        if revision_tag:
            print(f"  Filtering source runs to {revision_tag}")

        # Query runs from each source sprint
        tagged = 0
        skipped = 0
        skipped_revision = 0
        for source_sprint in source_sprints:
            source_tag = f"sprint:{source_sprint}"
            runs = api.runs(
                f"{entity}/{project}",
                filters={"tags": {"$in": [source_tag]}},
            )

            for run in runs:
                if revision_tag and revision_tag not in run.tags:
                    skipped_revision += 1
                    continue
                if target_tag in run.tags:
                    skipped += 1
                    continue

                if dry_run:
                    print(f"  [dry-run] Would tag: {run.name} ({run.id}) with {target_tag}")
                    tagged += 1
                else:
                    run.tags = run.tags + [target_tag]
                    run.update()
                    tagged += 1

        action = "would tag" if dry_run else "tagged"
        print(f"  {action} {tagged} runs, skipped {skipped} (already tagged)")
        if skipped_revision:
            print(f"  skipped {skipped_revision} source runs outside {revision_tag}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="SLICES parallel experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run experiments for given sprints")
    p_run.add_argument("--sprint", nargs="+", required=True, help="Sprint(s) to run (e.g. 1 1b 2)")
    p_run.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Max scheduler slots (default: 4). Pretrains consume 4 slots, other runs 1.",
    )
    p_run.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    p_run.add_argument("--revision", type=str, default=None, help="Revision name (e.g. v2)")
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

    # status
    p_status = sub.add_parser("status", help="Show experiment status")
    p_status.add_argument("--sprint", nargs="*", default=None, help="Filter by sprint(s)")

    # retry
    p_retry = sub.add_parser("retry", help="Retry failed/skipped runs")
    p_retry.add_argument("--failed", action="store_true", help="Retry failed runs")
    p_retry.add_argument("--skipped", action="store_true", help="Retry skipped runs")
    p_retry.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Max scheduler slots (default: 4). Pretrains consume 4 slots, other runs 1.",
    )
    p_retry.add_argument("--sprint", nargs="+", default=None, help="Scope retry to sprint(s)")
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

    # warmup
    p_warmup = sub.add_parser(
        "warmup", help="Pre-build tensor caches to avoid OOM during parallel runs"
    )
    p_warmup.add_argument(
        "--sprint", nargs="+", required=True, help="Sprint(s) to warmup (e.g. 1 1b 2)"
    )
    p_warmup.add_argument(
        "--revision", type=str, default=None, help="Ignored (warmup is revision-independent)"
    )
    p_warmup.add_argument("--reason", type=str, default=None, help="Ignored for warmup")

    # tag
    p_tag = sub.add_parser("tag", help="Tag inherited baseline runs in W&B with target sprint tags")
    p_tag.add_argument(
        "--sprint",
        nargs="+",
        required=True,
        help="Sprint(s) whose baselines to tag (e.g. 2 or 2 3 5)",
    )
    p_tag.add_argument("--dry-run", action="store_true", help="Show what would be tagged")
    p_tag.add_argument(
        "--revision",
        type=str,
        default=os.environ.get("REVISION") or os.environ.get("WANDB_REVISION"),
        help="Only inherit source runs with this revision tag (default: REVISION/WANDB_REVISION)",
    )
    p_tag.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "slices"),
        help="W&B project name (default: WANDB_PROJECT env var or 'slices')",
    )
    p_tag.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (default: WANDB_ENTITY env var or wandb default)",
    )

    args = parser.parse_args()

    # Validate --reason requires --revision
    if getattr(args, "reason", None) and not getattr(args, "revision", None):
        parser.error("--reason requires --revision")

    # Validate retry --revision requires --sprint
    if args.command == "retry" and args.revision and not args.sprint:
        parser.error("--revision requires --sprint to scope which sprints to revise")

    if args.command == "run":
        exit_code = cmd_run(args)
    elif args.command == "status":
        exit_code = cmd_status(args)
    elif args.command == "retry":
        exit_code = cmd_retry(args)
    elif args.command == "warmup":
        exit_code = cmd_warmup(args)
    elif args.command == "tag":
        exit_code = cmd_tag(args)
    else:
        exit_code = 0

    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
