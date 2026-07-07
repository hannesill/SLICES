#!/usr/bin/env python3
"""Run random-Transformer linear probes on cached frozen-encoder embeddings.

This diagnostic avoids recomputing the frozen random encoder on every epoch. It
is diagnostic-only and does not read, write, export, or reinterpret the frozen
thesis result export.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from slices.data.datamodule import ICUDataModule
from slices.eval import MetricConfig, build_metrics
from slices.models.heads import TaskHeadConfig, build_task_head
from slices.training import FineTuneModule
from slices.training.utils import (
    build_optimizer,
    build_scheduler,
    report_and_validate_train_label_support,
    resolve_balanced_class_weights,
    train_label_support_summary,
    validate_data_prerequisites,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path("outputs/diagnostics/random_transformer_probe_cached")
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
COMPLETION_MARKER = "diagnostic_complete.json"
RESULTS_FILE = "test_results.json"

DEFAULT_DATASETS = ("miiv", "eicu", "combined")
DEFAULT_TASKS = (
    "mortality_24h",
    "mortality_hospital",
    "aki_kdigo",
    "los_remaining",
)
DEFAULT_SEEDS = (42, 123, 456, 789, 1011)


@dataclass(frozen=True)
class CachedProbeJob:
    dataset: str
    task: str
    seed: int
    output_dir: Path


class CachedLinearProbeModule(pl.LightningModule):
    """Lightning module for a linear head trained on fixed embeddings."""

    def __init__(
        self,
        cfg: DictConfig,
        input_dim: int,
        class_weight: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        self.task_type = cfg.task.get("task_type", "binary")
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.get("scheduler", None)
        self.task_head = build_task_head(
            TaskHeadConfig(
                name="linear",
                task_name=cfg.task.get("task_name", "mortality_24h"),
                task_type=self.task_type,
                n_classes=cfg.task.get("n_classes", None),
                input_dim=input_dim,
                hidden_dims=[],
                dropout=cfg.task.get("dropout", 0.0),
                activation=cfg.task.get("activation", "relu"),
                use_layer_norm=cfg.task.get("use_layer_norm", False),
            )
        )
        if self.task_type in ("binary", "multiclass"):
            label_smoothing = cfg.training.get("label_smoothing", 0.0)
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=class_weight,
                label_smoothing=label_smoothing,
            )
        elif self.task_type == "multilabel":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.task_type == "regression":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        output_dim = self.task_head.config.get_output_dim()
        eval_cfg = cfg.get("eval", {})
        metrics_cfg = eval_cfg.get("metrics", {})
        metric_config = MetricConfig(
            task_type=self.task_type,
            n_classes=output_dim,
            metrics=metrics_cfg.get("names", None),
            threshold=metrics_cfg.get("threshold", 0.5),
        )
        self.train_metrics = build_metrics(metric_config, prefix="train")
        self.val_metrics = build_metrics(metric_config, prefix="val")
        self.test_metrics = build_metrics(metric_config, prefix="test")

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        embeddings, labels = batch
        outputs = self.task_head(embeddings)
        logits = outputs["logits"]
        probs = outputs["probs"]

        if self.task_type == "regression":
            loss = self.criterion(logits.squeeze(-1), labels.float())
        elif self.task_type in ("binary", "multiclass"):
            loss = self.criterion(logits, labels.long())
        elif self.task_type == "multilabel":
            loss = self.criterion(logits, labels.float())
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage != "train"),
            sync_dist=True,
        )

        metrics = getattr(self, f"{stage}_metrics")
        if self.task_type in ("binary", "multiclass", "multilabel"):
            output_dim = self.task_head.config.get_output_dim()
            metric_input = probs[:, 1] if output_dim == 2 else probs
            metrics.update(metric_input, labels.long())
        else:
            metrics.update(logits.squeeze(-1), labels.float())

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            sync_dist=True,
        )
        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> Any:
        optimizer = build_optimizer(self.task_head.parameters(), self.optimizer_config)
        result = build_scheduler(optimizer, self.scheduler_config)
        return optimizer if result is None else result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached random Transformer frozen-encoder linear probes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=DEFAULT_TASKS,
        default=list(DEFAULT_TASKS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=("auto", "cpu", "gpu", "mps"),
        help="Lightning accelerator override. Use mps on Apple Silicon if stable locally.",
    )
    parser.add_argument("--devices", default="1", help="Lightning devices override.")
    parser.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        help="Accepted for CLI compatibility; cached diagnostics always run without W&B.",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable W&B logging.",
    )
    parser.set_defaults(wandb=False)
    parser.add_argument("--wandb-project", default="slices-diagnostics")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=f"Skip jobs whose output dir already has {COMPLETION_MARKER}.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N jobs.")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional smoke-test override. Omit for the 50-epoch diagnostic protocol.",
    )
    parser.add_argument(
        "--force-recompute-embeddings",
        action="store_true",
        help="Ignore an existing embedding cache for each job.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers for both embedding extraction and cached training.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.max_epochs is not None and args.max_epochs <= 0:
        parser.error("--max-epochs must be positive")
    return args


def _output_dir(dataset: str, task: str, seed: int) -> Path:
    return OUTPUT_ROOT / dataset / task / f"seed{seed}"


def build_jobs(args: argparse.Namespace) -> list[CachedProbeJob]:
    jobs: list[CachedProbeJob] = []
    for dataset in args.datasets:
        for task in args.tasks:
            for seed in args.seeds:
                jobs.append(
                    CachedProbeJob(
                        dataset=dataset,
                        task=task,
                        seed=seed,
                        output_dir=_output_dir(dataset, task, seed),
                    )
                )
    return jobs[: args.limit] if args.limit is not None else jobs


def _manifest_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return MANIFEST_DIR / f"random_transformer_probe_cached_{stamp}.jsonl"


def _job_complete(job: CachedProbeJob) -> bool:
    return (REPO_ROOT / job.output_dir / COMPLETION_MARKER).exists()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_manifest_record(
    handle,
    job: CachedProbeJob,
    status: str,
    returncode: int | None,
    timings: dict[str, float] | None = None,
) -> None:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "returncode": returncode,
        "dataset": job.dataset,
        "task": job.task,
        "seed": job.seed,
        "paradigm": "random_transformer",
        "protocol": "linear_probe_cached_embeddings",
        "experiment_class": "random_probe_diagnostic",
        "experiment_subtype": "random_transformer_cached_linear_probe",
        "revision": "random-probe-cached-v1",
        "label_fraction": 1.0,
        "model_size": "default",
        "output_dir": job.output_dir.as_posix(),
        "timings_sec": timings or {},
    }
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def _compose_cfg(args: argparse.Namespace, job: CachedProbeJob) -> DictConfig:
    overrides = [
        f"dataset={job.dataset}",
        f"tasks={job.task}",
        f"seed={job.seed}",
        "protocol=linear_probe",
        "project_name=slices-diagnostics",
        "experiment_name=random_transformer_probe_cached",
        "paradigm=random_transformer",
        "experiment_class=random_probe_diagnostic",
        "experiment_subtype=random_transformer_cached_linear_probe",
        "revision=random-probe-cached-v1",
        "rerun_reason=random_transformer_cached_probe_diagnostic",
        "checkpoint=null",
        "pretrain_checkpoint=null",
        "label_fraction=1.0",
        "model=transformer",
        f"training.accelerator={args.accelerator}",
        f"training.devices={args.devices}",
        "logging.use_wandb=false",
        f"logging.wandb_project={args.wandb_project}",
        f"output_dir={job.output_dir.as_posix()}",
        f"checkpoint_dir={(job.output_dir / 'checkpoints').as_posix()}",
    ]
    if args.max_epochs is not None:
        overrides.append(f"training.max_epochs={args.max_epochs}")
        overrides.append(f"scheduler.T_max={args.max_epochs}")
    if args.num_workers is not None:
        overrides.append(f"data.num_workers={args.num_workers}")

    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base=None,
    ):
        cfg = hydra.compose(config_name="finetune", overrides=overrides)
    return cfg


def _select_torch_device(accelerator: str) -> torch.device:
    if accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if accelerator == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _build_datamodule(cfg: DictConfig) -> ICUDataModule:
    task_name = cfg.task.get("task_name", "mortality_24h")
    validate_data_prerequisites(
        cfg.data.processed_dir,
        cfg.dataset,
        task_names=[task_name],
        task_configs=[cfg.task],
    )
    datamodule = ICUDataModule(
        processed_dir=cfg.data.processed_dir,
        task_name=task_name,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.get("num_workers", 4),
        seed=cfg.seed,
        label_fraction=cfg.get("label_fraction", 1.0),
    )
    datamodule.setup()
    return datamodule


def _prepare_cfg(
    cfg: DictConfig,
    datamodule: ICUDataModule,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    task_name = cfg.task.get("task_name", "mortality_24h")
    task_type = cfg.task.get("task_type", "binary")
    train_stats = datamodule.get_train_label_statistics()
    train_support_stats = report_and_validate_train_label_support(
        datamodule=datamodule,
        task_name=task_name,
        task_type=task_type,
        dataset=cfg.dataset,
        seed=cfg.seed,
        label_fraction=cfg.get("label_fraction", 1.0),
        min_train_positives=cfg.get("min_train_positives", 3),
    )

    OmegaConf.set_struct(cfg, False)
    cfg.encoder.d_input = datamodule.get_feature_dim()
    cfg.encoder.max_seq_length = datamodule.get_seq_length()
    class_weight = None
    if cfg.training.get("class_weight") == "balanced":
        resolved = resolve_balanced_class_weights(
            task_name=task_name,
            task_type=task_type,
            train_label_stats=train_stats,
        )
        cfg.training.class_weight = resolved
        if resolved is not None:
            class_weight = torch.tensor(list(resolved), dtype=torch.float32)
    OmegaConf.set_struct(cfg, True)
    return class_weight, train_label_support_summary(train_support_stats)


def _extract_split_embeddings(
    model: FineTuneModule,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    stay_ids: list[torch.Tensor] = []
    model.encoder.to(device)
    model.encoder.eval()
    with torch.inference_mode():
        for batch in loader:
            timeseries = batch["timeseries"].to(device)
            mask = batch["mask"].to(device)
            padding_mask = model._downstream_padding_mask(mask)
            encoded = model.encoder(timeseries, mask=mask, padding_mask=padding_mask)
            embeddings.append(encoded.detach().cpu())
            labels.append(batch["label"].detach().cpu())
            stay_ids.append(batch["stay_id"].detach().cpu())
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "labels": torch.cat(labels, dim=0),
        "stay_ids": torch.cat(stay_ids, dim=0),
    }


def _embedding_cache_path(job: CachedProbeJob) -> Path:
    return REPO_ROOT / job.output_dir / "embeddings.pt"


def _load_or_build_embeddings(
    args: argparse.Namespace,
    job: CachedProbeJob,
    cfg: DictConfig,
    datamodule: ICUDataModule,
    device: torch.device,
) -> tuple[dict[str, Any], float, bool]:
    cache_path = _embedding_cache_path(job)
    if cache_path.exists() and not args.force_recompute_embeddings:
        start = time.perf_counter()
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload, time.perf_counter() - start, True

    start = time.perf_counter()
    model = FineTuneModule(
        config=cfg,
        checkpoint_path=None,
        pretrain_checkpoint_path=None,
    )
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": job.dataset,
            "task": job.task,
            "seed": job.seed,
            "encoder_dim": model.encoder.get_output_dim(),
            "feature_dim": datamodule.get_feature_dim(),
            "seq_length": datamodule.get_seq_length(),
            "split_info": datamodule.get_split_info(),
        },
        "train": _extract_split_embeddings(model, datamodule.train_dataloader(), device),
        "val": _extract_split_embeddings(model, datamodule.val_dataloader(), device),
        "test": _extract_split_embeddings(model, datamodule.test_dataloader(), device),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload, time.perf_counter() - start, False


def _make_embedding_loader(
    split: dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(split["embeddings"].float(), split["labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _monitor_and_mode(cfg: DictConfig) -> tuple[str, str]:
    lower_is_better = {"loss", "mse", "mae", "rmse", "brier_score", "ece"}
    higher_is_better = {
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "r2",
    }
    task_type = cfg.task.get("task_type", "binary")
    default_metric = cfg.task.get("primary_metric", None)
    if default_metric:
        monitor = default_metric if "/" in default_metric else f"val/{default_metric}"
    elif task_type == "regression":
        monitor = "val/mse"
    else:
        monitor = "val/auprc"
    metric = monitor.split("/", 1)[-1]
    if metric in lower_is_better:
        return monitor, "min"
    if metric in higher_is_better:
        return monitor, "max"
    raise ValueError(f"Cannot infer mode for monitor '{monitor}'")


def _run_cached_probe(
    args: argparse.Namespace,
    job: CachedProbeJob,
) -> tuple[int, dict[str, float]]:
    output_dir = REPO_ROOT / job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _compose_cfg(args, job)
    pl.seed_everything(cfg.seed, workers=True)
    datamodule = _build_datamodule(cfg)
    class_weight, train_support_summary = _prepare_cfg(cfg, datamodule)
    device = _select_torch_device(args.accelerator)

    embeddings, embedding_sec, cache_hit = _load_or_build_embeddings(
        args,
        job,
        cfg,
        datamodule,
        device,
    )
    train_loader = _make_embedding_loader(
        embeddings["train"],
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = _make_embedding_loader(
        embeddings["val"],
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    test_loader = _make_embedding_loader(
        embeddings["test"],
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    monitor, mode = _monitor_and_mode(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="cached-probe-{epoch:03d}",
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
    )
    callbacks: list[Any] = [checkpoint_callback]
    if cfg.training.get("early_stopping_patience", None):
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=cfg.training.early_stopping_patience,
                mode=mode,
                verbose=True,
            )
        )

    # Keep cached runs reproducible regardless of whether the encoder cache was
    # freshly built or loaded. The encoder seed is encoded in the cache.
    pl.seed_everything(cfg.seed, workers=True)
    cached_model = CachedLinearProbeModule(
        cfg=cfg,
        input_dim=int(embeddings["train"]["embeddings"].shape[1]),
        class_weight=class_weight,
    )
    if class_weight is not None:
        cached_model.criterion.weight = cached_model.criterion.weight.to(cached_model.device)

    train_start = time.perf_counter()
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=cfg.training.get("devices", "auto"),
        precision=cfg.training.get("precision", 32),
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 10),
        callbacks=callbacks,
        logger=False,
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        deterministic=cfg.get("deterministic", False),
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    trainer.fit(cached_model, train_loader, val_loader)
    train_sec = time.perf_counter() - train_start

    if checkpoint_callback.best_model_path:
        best = torch.load(
            checkpoint_callback.best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        cached_model.load_state_dict(best["state_dict"])

    test_start = time.perf_counter()
    test_results = trainer.test(cached_model, test_loader)
    test_sec = time.perf_counter() - test_start
    timings = {
        "embedding_sec": embedding_sec,
        "train_sec": train_sec,
        "test_sec": test_sec,
        "total_sec": embedding_sec + train_sec + test_sec,
    }
    metrics = test_results[0] if test_results else {}
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": job.dataset,
        "task": job.task,
        "seed": job.seed,
        "paradigm": "random_transformer",
        "protocol": "linear_probe_cached_embeddings",
        "experiment_class": "random_probe_diagnostic",
        "experiment_subtype": "random_transformer_cached_linear_probe",
        "revision": "random-probe-cached-v1",
        "output_dir": job.output_dir.as_posix(),
        "embedding_cache": str(_embedding_cache_path(job).relative_to(REPO_ROOT)),
        "embedding_cache_hit": cache_hit,
        "best_checkpoint": checkpoint_callback.best_model_path,
        "best_monitor": monitor,
        "best_model_score": (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None
        ),
        "train_label_support": train_support_summary,
        "split_info": datamodule.get_split_info(),
        "timings_sec": timings,
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    _write_json(output_dir / RESULTS_FILE, payload)
    _write_json(
        output_dir / COMPLETION_MARKER,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": job.dataset,
            "task": job.task,
            "seed": job.seed,
            "output_dir": job.output_dir.as_posix(),
            "results_file": str((job.output_dir / RESULTS_FILE).as_posix()),
        },
    )
    return 0, timings


def run_jobs(args: argparse.Namespace, jobs: list[CachedProbeJob]) -> int:
    (REPO_ROOT / MANIFEST_DIR).mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path()
    print(f"Manifest: {manifest_path.as_posix()}", flush=True)
    with (REPO_ROOT / manifest_path).open("w", encoding="utf-8") as manifest:
        for job in jobs:
            command = [
                "uv",
                "run",
                "python",
                "scripts/diagnostics/run_cached_random_transformer_probe.py",
                "--datasets",
                job.dataset,
                "--tasks",
                job.task,
                "--seeds",
                str(job.seed),
                "--accelerator",
                args.accelerator,
                "--devices",
                str(args.devices),
            ]
            if args.max_epochs is not None:
                command.extend(["--max-epochs", str(args.max_epochs)])
            print(shlex.join(command), flush=True)

            if args.skip_existing and _job_complete(job):
                print(f"SKIP existing: {job.output_dir.as_posix()}", flush=True)
                _write_manifest_record(manifest, job, "skipped_existing", None)
                continue
            if args.dry_run:
                _write_manifest_record(manifest, job, "dry_run", None)
                continue

            try:
                returncode, timings = _run_cached_probe(args, job)
            except Exception as exc:
                _write_manifest_record(manifest, job, "failed", 1)
                print(f"FAILED {job.output_dir.as_posix()}: {exc}", file=sys.stderr, flush=True)
                return 1
            _write_manifest_record(manifest, job, "completed", returncode, timings)
            if returncode != 0:
                return returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = build_jobs(args)
    print(f"Jobs: {len(jobs)}", flush=True)
    return run_jobs(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
