#!/usr/bin/env python3
# ruff: noqa: E402 -- start the timeout before numerical imports
"""Run the TS2Vec-style extension through shared trainers using synthetic inputs only.

From an extracted supplement: uv run --project framework python extension.py --out extension-output
Two SSL epochs and two adaptation epochs per task/protocol, one CPU thread,
120-second process limit. Synthetic scores cannot enter the frozen evidence grid.
"""

import signal
import time

START = time.monotonic()
signal.alarm(120)

import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "framework" if (ROOT / "framework").exists() else ROOT.parents[1]
sys.path.insert(0, str(SOURCE / "src"))

import lightning.pytorch as pl
import torch
from audit import digest, read_csv, verify_manifest
from extension_report import extension_report
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from slices.benchmark.contract import compare_contracts, fingerprint
from slices.data.splits import _split_patient_sets
from slices.data.tensor_preprocessing import (
    apply_normalization_and_imputation,
    compute_normalization_stats,
)
from slices.models.pretraining.factory import CONFIG_REGISTRY, SSL_REGISTRY
from slices.models.pretraining.ts2vec import TS2VecConfig, TS2VecObjective
from slices.training.finetune_module import FineTuneModule
from slices.training.pretrain_module import SSLPretrainModule
from slices.training.utils import resolve_balanced_class_weights


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def config(name, overrides, short=True):
    with initialize_config_dir(config_dir=str(SOURCE / "configs"), version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.encoder.d_input = 84
    cfg.encoder.max_seq_length = 24
    cfg.output_dir = "."
    if not short:
        return cfg
    cfg.training.max_epochs = 2
    cfg.training.batch_size = 8
    cfg.training.accelerator = "cpu"
    cfg.training.devices = 1
    cfg.logging.use_wandb = False
    cfg.output_dir = "."
    if name == "pretrain":
        cfg.scheduler.warmup_epochs = 0
    return cfg


def check_scientific_configs(reference):
    """Bind the intended declaration to composed settings before shortening them."""
    pre = config("pretrain", ["ssl=ts2vec"], short=False)
    checks = []

    def same(field, actual, expected):
        if actual != expected:
            raise ValueError(f"Composed scientific config drift at {field}: {actual} != {expected}")
        checks.append(field)

    for key in ("obs_aware", "d_model", "n_layers", "n_heads", "d_ff", "dropout"):
        same(f"encoder.{key}", pre.encoder[key], reference["encoder"][key])
    same("encoder.pooling", pre.encoder.pooling, "none")
    same("pretraining.epochs", pre.training.max_epochs, reference["pretraining_epochs"])
    for field, value in {
        "batch_size": pre.training.batch_size,
        "lr": pre.optimizer.lr,
        "weight_decay": pre.optimizer.weight_decay,
        "warmup_epochs": pre.scheduler.warmup_epochs,
        "early_stopping": pre.training.early_stopping_patience,
        "gradient_clip": pre.training.gradient_clip_val,
    }.items():
        same(f"pretraining.{field}", value, reference["pretraining"][field])
    for protocol in ("linear_probe", "full_finetune"):
        cfg = config("finetune", [f"protocol={protocol}"], short=False)
        for field, value in {
            "epochs": cfg.training.max_epochs,
            "lr": cfg.optimizer.lr,
            "freeze": cfg.training.freeze_encoder,
            "hidden_dims": list(cfg.task.hidden_dims),
            "dropout": cfg.task.dropout,
        }.items():
            same(f"{protocol}.{field}", value, reference["adaptation"][protocol][field])
        for field, value in {
            "batch_size": cfg.training.batch_size,
            "weight_decay": cfg.optimizer.weight_decay,
            "patience": cfg.training.early_stopping_patience,
            "gradient_clip": cfg.training.gradient_clip_val,
        }.items():
            same(f"adaptation.{protocol}.{field}", value, reference["adaptation"][field])
        same(
            f"{protocol}.label_smoothing",
            cfg.training.label_smoothing,
            reference["losses"]["classification"]["label_smoothing"],
        )
    return checks


def trainer(folder, callbacks=(), checkpointing=False):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=2,
        logger=False,
        enable_checkpointing=checkpointing,
        callbacks=list(callbacks),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(folder),
        deterministic=True,
        gradient_clip_val=1.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=ROOT)
    args = parser.parse_args()
    # Audit before creating any outputs or running any model operations.
    verify_manifest(args.evidence_root)
    if args.out.exists():
        raise ValueError("Output exists; choose a fresh directory to preserve prior evidence")
    args.out.mkdir(parents=True)
    frozen = extension_report(
        read_csv(args.evidence_root / "results/per_seed.csv"),
        json.loads((args.evidence_root / "results/scientific_configs.json").read_text()),
        args.out,
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pl.seed_everything(42, workers=True)
    if (
        SSL_REGISTRY.get("ts2vec") is not TS2VecObjective
        or CONFIG_REGISTRY.get("ts2vec") is not TS2VecConfig
    ):
        raise ValueError("TS2Vec objective/config registration is incomplete")
    pre_cfg = config("pretrain", ["ssl=ts2vec"])
    reference = json.loads((args.evidence_root / "benchmark.json").read_text())
    composed_config_checks = check_scientific_configs(reference)
    candidate = copy.deepcopy(reference)
    candidate["objective"] = "ts2vec"
    candidate["objective_config"] = OmegaConf.to_container(pre_cfg.ssl, resolve=True)
    if compare_contracts(reference, candidate):
        raise ValueError("Extension changed a declared benchmark invariant")
    write_json(args.out / "scientific-declaration.json", candidate)
    actual = copy.deepcopy(candidate)
    actual.update(
        benchmark="synthetic-ts2vec-workflow-v1",
        datasets=["synthetic"],
        seeds=[42],
        pretraining_epochs=2,
    )
    actual["tasks"] = ["synthetic_binary", "synthetic_regression"]
    actual["task_definitions"] = {
        "synthetic_binary": "synthetic patient index modulo 2",
        "synthetic_regression": "1 + index modulo 7 / 2; arbitrary units",
    }
    actual["pretraining"].update(batch_size=8, warmup_epochs=0)
    actual["adaptation"]["batch_size"] = 8
    actual["adaptation"]["linear_probe"]["epochs"] = 2
    actual["adaptation"]["full_finetune"]["epochs"] = 2
    actual["adaptation"]["patience"] = None
    actual["evidence_policy"] = "single-seed synthetic execution; no scientific performance claim"
    actual["primary_metrics"]["regression"] = "mean_absolute_error_synthetic_units"
    actual["preprocessing"]["source_versions"] = {"synthetic_generator": "v1"}
    actual["preprocessing"]["ricu_version"] = None
    actual["cohort"]["minimum_stay_hours"] = None
    actual_differences = compare_contracts(candidate, actual)
    if not actual_differences:
        raise ValueError("Synthetic execution must fail scientific comparability")
    write_json(args.out / "execution-declaration.json", actual)
    negatives = {}
    for key, value in [
        ("feature_names", list(reversed(reference["feature_names"]))),
        ("checkpoint_policy", "best SSL validation"),
        ("pretraining_epochs", 99),
    ]:
        bad = copy.deepcopy(candidate)
        bad[key] = value
        negatives[key] = compare_contracts(candidate, bad)
        if not negatives[key]:
            raise ValueError(f"Comparator accepted changed {key}")

    # Same split algorithm and normalizer, with entirely fictitious patient IDs.
    splits = [sorted(s) for s in _split_patient_sets(set(range(80)), 42, 0.7, 0.15)]
    if any(set(a) & set(b) for i, a in enumerate(splits) for b in splits[i + 1 :]):
        raise ValueError("Synthetic partitions overlap")
    raw = torch.randn(80, 24, 84)
    mask = torch.rand_like(raw) < 0.12
    mask[:, 0] = False
    raw[~mask] = float("nan")
    mean, std = compute_normalization_stats(raw, mask, splits[0], 84, True)
    values = apply_normalization_and_imputation(raw, mean, std, True, 84)
    if not torch.isfinite(values).all() or not (values[~mask] == 0).all():
        raise ValueError("Synthetic normalization or zero-fill failed")

    def loaders(task):
        label = (
            torch.arange(80) % 2 if task == "mortality_hospital" else 1 + torch.arange(80) % 7 / 2
        )
        return [
            DataLoader(
                [{"timeseries": values[i], "mask": mask[i], "label": label[i]} for i in indices],
                batch_size=8,
                shuffle=False,
                num_workers=0,
            )
            for indices in splits
        ]

    train, val, _ = loaders("mortality_hospital")
    pre = SSLPretrainModule(pre_cfg)
    before = copy.deepcopy(pre.encoder.state_dict())
    pre_trainer = trainer(args.out)
    pre_trainer.fit(pre, train, val)
    if state_equal(before, pre.encoder.state_dict()):
        raise ValueError("Pretraining did not update the encoder")
    encoder_path = args.out / "encoder.pt"
    pre.save_encoder(str(encoder_path))
    saved = torch.load(encoder_path, map_location="cpu", weights_only=False)
    if saved["version"] != 3 or not state_equal(
        saved["encoder_state_dict"], pre.encoder.state_dict()
    ):
        raise ValueError("Final encoder export mismatch")
    if any("projection_head" in key for key in saved["encoder_state_dict"]):
        raise ValueError("Objective-specific projector leaked into transfer")
    transferred_parameters = sum(p.numel() for p in pre.encoder.parameters())
    if transferred_parameters != reference["encoder"]["transferred_parameters"]:
        raise ValueError("Encoder parameter count differs from reference")
    write_json(args.out / "pretrain-config.json", OmegaConf.to_container(pre_cfg, resolve=True))
    results = []
    for task in ("mortality_hospital", "los_remaining"):
        train, val, test = loaders(task)
        for protocol in ("linear_probe", "full_finetune"):
            pl.seed_everything(42, workers=True)
            folder = args.out / f"{task}-{protocol}"
            folder.mkdir()
            cfg = config("finetune", [f"tasks={task}", f"protocol={protocol}", "paradigm=ts2vec"])
            cfg.training.early_stopping_patience = None
            positive = sum(i % 2 for i in splits[0])
            cfg.training.class_weight = resolve_balanced_class_weights(
                task,
                cfg.task.task_type,
                {task: {"positive": positive, "negative": len(splits[0]) - positive}},
            )
            write_json(folder / "config.json", OmegaConf.to_container(cfg, resolve=True))
            model = FineTuneModule(cfg, checkpoint_path=str(encoder_path))
            if not state_equal(saved["encoder_state_dict"], model.encoder.state_dict()):
                raise ValueError("Transfer changed encoder weights")
            if model.encoder.config.pooling != "mean":
                raise ValueError("Downstream transfer did not select mean pooling")
            primary = cfg.task.primary_metric
            selection = ModelCheckpoint(
                dirpath=str(folder),
                monitor=f"val/{primary}",
                mode="min" if primary == "mae" else "max",
                save_top_k=1,
                filename="best",
                auto_insert_metric_name=False,
            )
            downstream = trainer(folder, [selection], True)
            downstream.fit(model, train, val)
            unchanged = state_equal(saved["encoder_state_dict"], model.encoder.state_dict())
            if unchanged != (protocol == "linear_probe"):
                raise ValueError("Encoder update scope does not match the protocol")
            if not selection.best_model_path:
                raise ValueError("No validation-selected checkpoint; refusing final-model fallback")
            # Lightning reloads the selected checkpoint. The test loader is used once.
            # This checkpoint was created above in this process. Lightning saves
            # OmegaConf configuration objects, so its safe weights-only default
            # cannot deserialize this local, trusted trainer checkpoint.
            measured = downstream.test(
                model, test, ckpt_path=selection.best_model_path, weights_only=False, verbose=False
            )[0]
            metric_value = measured[f"test/{primary}"]
            if not torch.isfinite(torch.tensor(metric_value)):
                raise ValueError("Nonfinite test primary metric")
            selected = torch.load(selection.best_model_path, map_location="cpu", weights_only=False)
            if not state_equal(selected["state_dict"], model.state_dict()):
                raise ValueError("Test model differs from selected checkpoint")
            results.append(
                {
                    "task": "synthetic_binary" if primary == "auprc" else "synthetic_regression",
                    "task_config": task,
                    "protocol": protocol,
                    "metric": primary,
                    "synthetic_test_value": metric_value,
                    "selected_epoch": selected["epoch"],
                    "validation_primary": float(selection.best_model_score),
                    "encoder_unchanged_during_fit": unchanged,
                    "transfer_exact": True,
                    "test_checkpoint_reload_exact": True,
                    "test_evaluations": 1,
                    "train_count": len(splits[0]),
                    "validation_count": len(splits[1]),
                    "test_count": len(splits[2]),
                    "checkpoint_sha256": digest(Path(selection.best_model_path)),
                    "config_sha256": digest(folder / "config.json"),
                }
            )
    write_json(
        args.out / "synthetic-results.json",
        {
            "status": "pass",
            "evidence_type": "agent synthetic software validation; not independent human reuse",
            "device": "cpu",
            "threads": 1,
            "seed": 42,
            "process_seconds": time.monotonic() - START,
            "model_operations_seconds": time.monotonic()
            - pre_trainer.lightning_module._training_start_time,
            "timeout_seconds": 120,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lightning": pl.__version__,
            "registration": "TS2VecObjective + TS2VecConfig + ssl/ts2vec.yaml",
            "encoder_parameters": transferred_parameters,
            "pretraining_trainable_parameters": sum(
                p.numel() for p in pre.parameters() if p.requires_grad
            ),
            "encoder_checkpoint_sha256": digest(encoder_path),
            "script_sha256": digest(Path(__file__)),
            "framework_files": {
                str(p.relative_to(SOURCE)): digest(p)
                for directory in ("src", "configs")
                for p in sorted((SOURCE / directory).rglob("*"))
                if p.suffix in {".py", ".yaml"}
            },
            "split_sha256": fingerprint({"indices": splits}),
            "normalization_sha256": fingerprint(
                {"mean": tensor_hash(mean), "std": tensor_hash(std)}
            ),
            "synthetic_input_sha256": fingerprint(
                {"values": tensor_hash(values), "mask": tensor_hash(mask)}
            ),
            "scientific_declaration_differences": [],
            "composed_scientific_config_checks": composed_config_checks,
            "execution_declaration_differences": actual_differences,
            "negative_controls": negatives,
            "frozen_reporting": frozen,
            "runs": results,
        },
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "synthetic_runs": len(results),
                "frozen_cells": frozen["cells"],
                "seconds": time.monotonic() - START,
            }
        )
    )
    signal.alarm(0)


if __name__ == "__main__":
    main()
