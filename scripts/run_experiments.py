#!/usr/bin/env python3
"""
Parallel experiment runner for SLICES.

Generates all experiment configurations across sprints, resolves dependencies,
and executes them in parallel with crash recovery and state persistence.

Usage:
    uv run python scripts/run_experiments.py warmup --sprint 1
    uv run python scripts/run_experiments.py run --sprint 1 --parallel 4
    uv run python scripts/run_experiments.py run --sprint 1 2 3 --parallel 6 --dry-run
    uv run python scripts/run_experiments.py run --sprint 1 --revision v2 --reason "fix LR"
    uv run python scripts/run_experiments.py status
    uv run python scripts/run_experiments.py status --sprint 1
    uv run python scripts/run_experiments.py retry --failed --parallel 4
    uv run python scripts/run_experiments.py retry --failed --sprint 1 --revision v2 --parallel 4
    uv run python scripts/run_experiments.py tag --sprint 2 --dry-run
    uv run python scripts/run_experiments.py tag --sprint 2 3 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
# for later sprints. See docs/EXPERIMENT_PLAN.md § "Baseline Inheritance Across Sprints".
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
    "6": ["1", "2", "3", "4", "5"],
    "7": ["1", "3", "5"],
    "7p": ["6"],
    "8": ["1", "1b", "1c", "5"],
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
        if self.freeze_encoder is True:
            cmd += [
                "training.freeze_encoder=true",
                f"training.max_epochs={PROTO_A['max_epochs']}",
                f"training.early_stopping_patience={PROTO_A['patience']}",
                f"optimizer.lr={PROTO_A['lr']}",
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
    ) -> Run:
        """Add a finetune run.

        Args:
            extra: Hydra overrides added to the finetune command AND name.
            name_extra: Dict used only for directory/ID disambiguation
                (e.g. pretrain ablation params). Not passed to command.
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

    # --- Sprint builders ---

    def build_sprint1(self):
        """MIMIC, all tasks, Protocol B + supervised + baselines, seed=42."""
        ds, seed, sprint = "miiv", 42, "1"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)
            self._add_gru_d(sprint, ds, seed, task)
            self._add_xgboost(sprint, ds, seed, task)

    def build_sprint1b(self):
        """LR sensitivity, MIMIC, mortality_24h, seed=42."""
        ds, seed, task, sprint = "miiv", 42, "mortality_24h", "1b"
        for p in SSL_PARADIGMS:
            for lr in LR_ABLATION:
                extra = {"optimizer.lr": lr}
                pt = self._add_pretrain(sprint, p, ds, seed, extra)
                self._add_finetune(sprint, p, ds, seed, task, False, pt, name_extra=extra)

    def build_sprint1c(self):
        """Mask ratio sensitivity, MIMIC, mortality_24h, seed=42."""
        ds, seed, task, sprint = "miiv", 42, "mortality_24h", "1c"
        for p in SSL_PARADIGMS:
            for mr in MASK_RATIO_ABLATION:
                extra = {"ssl.mask_ratio": mr}
                pt = self._add_pretrain(sprint, p, ds, seed, extra)
                self._add_finetune(sprint, p, ds, seed, task, False, pt, name_extra=extra)

    def build_sprint2(self):
        """MIMIC Protocol A, seed=42 — reuses Sprint 1 pretrains."""
        ds, seed, sprint = "miiv", 42, "2"
        for p in SSL_PARADIGMS:
            pt = self._get_pretrain(p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, True, pt)

    def build_sprint3(self):
        """eICU, both protocols + supervised + baselines, seed=42."""
        ds, seed, sprint = "eicu", 42, "3"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
                self._add_finetune(sprint, p, ds, seed, task, True, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)
            self._add_gru_d(sprint, ds, seed, task)
            self._add_xgboost(sprint, ds, seed, task)

    def build_sprint4(self):
        """Combined, both protocols + supervised + baselines, seed=42."""
        ds, seed, sprint = "combined", 42, "4"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
                self._add_finetune(sprint, p, ds, seed, task, True, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)
            self._add_gru_d(sprint, ds, seed, task)
            self._add_xgboost(sprint, ds, seed, task)

    def build_sprint5(self):
        """Seeds 123, 456 for datasets miiv, eicu, combined."""
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
                    self._add_gru_d(sprint, ds, seed, task)
                    self._add_xgboost(sprint, ds, seed, task)

    def build_sprint6(self):
        """Label efficiency ablation — reuses Phase 1 encoders."""
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
                    self._add_gru_d(sprint, ds, seed, "mortality_24h", frac)
                    self._add_xgboost(sprint, ds, seed, "mortality_24h", frac)
                for task in TASKS[1:]:
                    for frac in LABEL_FRACTIONS_TREND:
                        self._add_supervised(sprint, ds, seed, task, frac)
                        self._add_gru_d(sprint, ds, seed, task, frac)
                        self._add_xgboost(sprint, ds, seed, task, frac)

    def build_sprint7p(self):
        """Model capacity pilot — test whether bigger models widen SSL-supervised gap.

        MIIV only, seed 42, mortality_24h, MAE + supervised.
        Two model sizes (medium=128d/4L, large=256d/4L) at 3 label fractions.
        Compares against Sprint 6 baseline (64d/2L) — no need to rerun baseline.

        Total: 2 pretrain + 2×3 finetune + 2×3 probe + 2×3 supervised = 20 runs.
        """
        sprint = "7p"
        ds, seed, task = "miiv", 42, "mortality_24h"

        for size_name, size_cfg in MODEL_SIZES.items():
            # Common model override for all runs at this size
            model_extra = {"model": size_cfg["model"]}

            # --- MAE pretrain (new encoder size, full data) ---
            pretrain_extra = {
                **model_extra,
                **size_cfg["ssl_scale"]["mae"],
            }
            pt = self._add_pretrain(sprint, "mae", ds, seed, extra=pretrain_extra)

            # --- MAE finetune (Protocol B) + probe (Protocol A) ---
            finetune_extra = {**model_extra}
            for frac in LABEL_FRACTIONS_PILOT:
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
                )
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
                        sprint, p, "miiv", seed, "mortality_24h", False, pt, name_extra=extra
                    )
                # Mask ratio ablation
                for mr in MASK_RATIO_ABLATION:
                    extra = {"ssl.mask_ratio": mr}
                    pt = self._add_pretrain(sprint, p, "miiv", seed, extra)
                    self._add_finetune(
                        sprint, p, "miiv", seed, "mortality_24h", False, pt, name_extra=extra
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
                    )

    def build_sprint11(self):
        """Classical baselines (XGBoost + GRU-D), all datasets/tasks, 5 seeds."""
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
        """TS2Vec temporal contrastive variant, 5 seeds.

        Addresses the "contrastive was set up to fail" vulnerability by giving
        the contrastive paradigm its natural augmentations (noise + masking)
        and a temporal contrastive loss. Same encoder, same training budget.
        Protocol B (full finetune) only — matches the primary evaluation protocol.
        """
        sprint = "13"
        for seed in SEEDS_EXTENDED:
            for ds in DATASETS:
                pt = self._add_pretrain(sprint, "ts2vec", ds, seed)
                for task in TASKS:
                    self._add_finetune(sprint, "ts2vec", ds, seed, task, False, pt)

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
            new_id = f"{prefix}rev-{revision}_{old_id[len(prefix):]}"
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


def recover_stale_running(state: dict):
    """Reset running entries whose PIDs are dead back to pending."""
    for run_id, info in state["runs"].items():
        if info.get("status") == "running":
            pid = info.get("pid")
            if pid is None or not is_pid_alive(pid):
                info["status"] = "pending"
                info.pop("pid", None)
                print(f"  Recovered stale run: {run_id}")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
def run_scheduler(runs: list[Run], state: dict, parallel: int, dry_run: bool):
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
        return

    print(f"Scheduler started (parallel={parallel})")
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
                        print(f"  FAILED {rid}: exit 0 but encoder.pt " f"missing {elapsed}")
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

        # 4. Launch runs up to parallel limit
        slots = parallel - len(active)
        for r in ready[:slots]:
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
                state, r.id, "running", started_at=now, pid=proc.pid, log_file=str(log_file)
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
        cmd = " ".join(r.build_command(runs_by_id))
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
            print(f"Including {len(extra)} dependency run(s) from earlier " f"sprints")

    if not runs:
        print(f"No runs found for sprint(s): {', '.join(sprints)}")
        return

    # Apply revision transform if requested
    if args.revision:
        runs = apply_revision(runs, args.revision, args.reason)

    print(f"Sprint(s) {', '.join(sprints)}: {len(runs)} runs")
    if args.revision:
        print(f"Revision: {args.revision}" + (f" ({args.reason})" if args.reason else ""))
    run_scheduler(runs, state, args.parallel, args.dry_run)


def cmd_status(args):
    sprint_filter = [str(s) for s in args.sprint] if args.sprint else None
    print_status(sprint_filter)


def cmd_retry(args):
    all_runs = generate_all_runs()

    # Apply revision if specified (must happen before ID matching against state)
    if args.revision:
        sprint_filter = [str(s) for s in args.sprint] if args.sprint else None
        if sprint_filter:
            revised = [r for r in all_runs if r.sprint in sprint_filter]
            rest = [r for r in all_runs if r.sprint not in sprint_filter]
            revised = apply_revision(revised, args.revision, args.reason)
            all_runs = rest + revised
        else:
            all_runs = apply_revision(all_runs, args.revision, args.reason)

    state = load_state()
    recover_stale_running(state)

    runs_to_retry = []
    for r in all_runs:
        status = get_run_status(state, r.id)
        if args.failed and status == "failed":
            set_run_status(state, r.id, "pending")
            runs_to_retry.append(r)
        elif args.skipped and status == "skipped":
            set_run_status(state, r.id, "pending")
            runs_to_retry.append(r)

    if not runs_to_retry:
        print("No runs to retry.")
        return

    # Include their dependencies that are completed (already fine) or pending
    all_by_id = {r.id: r for r in all_runs}
    retry_ids = {r.id for r in runs_to_retry}
    deps_needed = set()
    for r in runs_to_retry:
        for d in r.depends_on:
            if d not in retry_ids:
                deps_needed.add(d)
    extra = [all_by_id[d] for d in deps_needed if d in all_by_id]
    runs_to_retry = extra + runs_to_retry

    save_state(state)
    print(f"Retrying {len(runs_to_retry)} runs")
    run_scheduler(runs_to_retry, state, args.parallel, dry_run=False)


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

    import subprocess
    import sys

    for i, dataset in enumerate(datasets, 1):
        processed_dir = f"data/processed/{dataset}"
        print(f"[{i}/{len(datasets)}] {dataset}")

        # Run in a subprocess so that memory is fully returned to the OS.
        # Uses task_name=None (unsupervised) to load ALL stays without
        # any task-specific filtering — this populates the raw cache.
        result = subprocess.run(
            [
                sys.executable,
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

    print("\nWarmup complete. Raw tensor caches saved to data/processed/<dataset>/.tensor_cache/")
    print("You can now run experiments in parallel without OOM.")


def cmd_tag(args):
    """Add sprint tags to inherited baseline runs in W&B.

    For each requested sprint, finds runs from inherited baseline sprints
    (per BASELINE_SPRINTS mapping) and adds the target sprint tag via the
    W&B API. Idempotent — already-tagged runs are skipped.

    Usage:
        uv run python scripts/run_experiments.py tag --sprint 2
        uv run python scripts/run_experiments.py tag --sprint 2 3 --dry-run
        uv run python scripts/run_experiments.py tag --sprint 2 --project slices --entity myteam
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

    for target_sprint in sprints:
        source_sprints = BASELINE_SPRINTS.get(target_sprint, [])
        if not source_sprints:
            print(f"Sprint {target_sprint}: no baseline sprints to inherit from — skipping.")
            continue

        target_tag = f"sprint:{target_sprint}"
        sources = ", ".join(source_sprints)
        print(f"\nSprint {target_sprint}: inheriting baselines from sprint(s) {sources}")

        # Query runs from each source sprint
        tagged = 0
        skipped = 0
        for source_sprint in source_sprints:
            source_tag = f"sprint:{source_sprint}"
            runs = api.runs(
                f"{entity}/{project}",
                filters={"tags": {"$in": [source_tag]}},
            )

            for run in runs:
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

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="SLICES parallel experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run experiments for given sprints")
    p_run.add_argument("--sprint", nargs="+", required=True, help="Sprint(s) to run (e.g. 1 1b 2)")
    p_run.add_argument("--parallel", type=int, default=4, help="Max parallel jobs (default: 4)")
    p_run.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    p_run.add_argument("--revision", type=str, default=None, help="Revision name (e.g. v2)")
    p_run.add_argument("--reason", type=str, default=None, help="Reason for rerun")

    # status
    p_status = sub.add_parser("status", help="Show experiment status")
    p_status.add_argument("--sprint", nargs="*", default=None, help="Filter by sprint(s)")

    # retry
    p_retry = sub.add_parser("retry", help="Retry failed/skipped runs")
    p_retry.add_argument("--failed", action="store_true", help="Retry failed runs")
    p_retry.add_argument("--skipped", action="store_true", help="Retry skipped runs")
    p_retry.add_argument("--parallel", type=int, default=4, help="Max parallel jobs (default: 4)")
    p_retry.add_argument("--sprint", nargs="+", default=None, help="Scope retry to sprint(s)")
    p_retry.add_argument("--revision", type=str, default=None, help="Revision name to retry")
    p_retry.add_argument("--reason", type=str, default=None, help="Reason for rerun")

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
        "--project", type=str, default="slices", help="W&B project name (default: slices)"
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
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "retry":
        cmd_retry(args)
    elif args.command == "warmup":
        cmd_warmup(args)
    elif args.command == "tag":
        cmd_tag(args)


if __name__ == "__main__":
    main()
