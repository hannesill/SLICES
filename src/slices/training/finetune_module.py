"""Lightning module for downstream task finetuning.

This module wraps a pretrained encoder and task head for finetuning on
downstream clinical prediction tasks (mortality, LOS, AKI, sepsis, etc.).

Key features:
- Load pretrained encoder weights from SSL pretraining
- Flexible freezing strategies (freeze encoder, unfreeze for finetuning)
- Support for binary classification, multiclass, and regression tasks
- Comprehensive metrics (AUROC, AUPRC, accuracy, etc.)
"""

import logging
from typing import Any, Dict, Literal, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from slices.eval import MetricConfig, build_metrics
from slices.models.encoders import TransformerEncoder, build_encoder
from slices.models.heads import TaskHeadConfig, build_task_head
from slices.training.checkpoint_loading import (
    load_encoder_weights,
    load_from_pretrain_checkpoint,
    wrap_encoder_with_missing_token,
)
from slices.training.config_schemas import OptimizerConfig as OptimizerConfigSchema
from slices.training.config_schemas import SchedulerConfig as SchedulerConfigSchema
from slices.training.config_schemas import TaskConfig as TaskConfigSchema
from slices.training.config_schemas import TrainingConfig as TrainingConfigSchema
from slices.training.utils import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


class FineTuneModule(pl.LightningModule):
    """Lightning module for downstream task finetuning.

    This module composes a pretrained encoder with a task head and handles:
    - Loading pretrained encoder weights
    - Freezing/unfreezing encoder for different finetuning strategies
    - Task-specific losses and metrics
    - Class imbalance handling (common in clinical tasks)

    Finetuning strategies:
    - "frozen" (default): Freeze encoder, only train task head (linear probing)
    - "full": Train both encoder and task head end-to-end
    - "gradual": Start frozen, then unfreeze encoder after warmup epochs

    Example:
        >>> module = FineTuneModule(
        ...     config=cfg,
        ...     checkpoint_path="outputs/encoder.pt",  # From pretraining
        ... )
        >>> trainer.fit(module, datamodule)
    """

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: Optional[str] = None,
        pretrain_checkpoint_path: Optional[str] = None,
    ) -> None:
        """Initialize finetuning module.

        Args:
            config: Hydra configuration with encoder, task, and training settings.
            checkpoint_path: Path to encoder weights (.pt file from pretrain.py).
            pretrain_checkpoint_path: Path to full pretrain checkpoint (.ckpt).
                Either checkpoint_path or pretrain_checkpoint_path should be provided.
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Validate task and training configs (catches typos via extra="forbid")
        task_dict = OmegaConf.to_container(config.task, resolve=True)
        TaskConfigSchema(**task_dict)

        training_dict = OmegaConf.to_container(config.training, resolve=True)
        TrainingConfigSchema(**training_dict)

        optimizer_dict = OmegaConf.to_container(config.optimizer, resolve=True)
        OptimizerConfigSchema(**optimizer_dict)

        if config.get("scheduler") is not None:
            scheduler_dict = OmegaConf.to_container(config.scheduler, resolve=True)
            SchedulerConfigSchema(**scheduler_dict)

        # Build encoder from config
        encoder_name = config.encoder.name
        encoder_config_dict = {k: v for k, v in config.encoder.items() if k != "name"}
        self.encoder = build_encoder(encoder_name, encoder_config_dict)

        # Check if missing token wrapper should be used
        self.use_missing_token = config.training.get("use_missing_token", True)

        # Load pretrained weights (may wrap encoder with EncoderWithMissingToken)
        if checkpoint_path:
            self.encoder = load_encoder_weights(
                self.encoder, checkpoint_path, config, self.use_missing_token
            )
        elif pretrain_checkpoint_path:
            self.encoder = load_from_pretrain_checkpoint(
                self.encoder, pretrain_checkpoint_path, config, self.use_missing_token
            )
        elif self.use_missing_token:
            # No pretrained weights, but still wrap encoder for consistency
            self.encoder = wrap_encoder_with_missing_token(self.encoder, missing_token=None)

        # Optional projection layer for dimensionality-controlled evaluation.
        # When comparing encoders with different output dims (e.g., 64 vs 1120),
        # a shared projection ensures linear probing measures quality, not dim.
        projection_dim = config.task.get("projection_dim", None)
        if projection_dim is not None:
            encoder_dim = self.encoder.get_output_dim()
            self.projection = nn.Linear(encoder_dim, projection_dim)
            self._effective_encoder_dim = projection_dim
            logger.info(
                "Added projection layer: %d -> %d for fair evaluation",
                encoder_dim,
                projection_dim,
            )
        else:
            self.projection = None
            self._effective_encoder_dim = self.encoder.get_output_dim()

        # Build task head
        task_config = self._build_task_config(config)
        self.task_head = build_task_head(task_config)

        # Store config for optimizer and freezing
        self.optimizer_config = config.optimizer
        self.scheduler_config = config.get("scheduler", None)
        self.freeze_strategy = config.training.get("freeze_encoder", True)
        self.unfreeze_epoch = config.training.get("unfreeze_epoch", None)

        # Apply initial freezing strategy
        self._apply_freeze_strategy()

        # Parse class weights from config (null, "balanced", or list of floats).
        # "balanced" should be resolved to a list by the training script before
        # constructing the module; if it reaches here unresolved, skip it.
        class_weight = config.training.get("class_weight", None)
        if class_weight is not None and class_weight != "balanced":
            self._class_weights = torch.tensor(list(class_weight), dtype=torch.float32)
        else:
            self._class_weights = None

        # Setup loss function (use task head's task_type)
        self.task_type = self.task_head.config.task_type
        self.criterion = self._get_criterion()

        # Setup metrics
        self._setup_metrics()

        # Flag for one-time label validation on first batch
        self._labels_validated = False

    def _build_task_config(self, config: DictConfig) -> TaskHeadConfig:
        """Build task head configuration from Hydra config.

        Args:
            config: Full configuration object.

        Returns:
            TaskHeadConfig instance.
        """
        task_cfg = config.task

        # Get hidden_dims as list
        hidden_dims = task_cfg.get("hidden_dims", [64])
        if not isinstance(hidden_dims, list):
            hidden_dims = list(hidden_dims)

        # Get task_type and n_classes
        task_type = task_cfg.get("task_type", "binary")
        n_classes = task_cfg.get("n_classes", None)

        return TaskHeadConfig(
            name=task_cfg.get("head_type", "mlp"),
            task_name=task_cfg.get("task_name", "mortality_24h"),
            task_type=task_type,
            n_classes=n_classes,
            input_dim=self._effective_encoder_dim,
            hidden_dims=hidden_dims,
            dropout=task_cfg.get("dropout", 0.1),
            activation=task_cfg.get("activation", "relu"),
            use_layer_norm=task_cfg.get("use_layer_norm", False),
        )

    def _apply_freeze_strategy(self) -> None:
        """Apply freezing strategy to encoder."""
        if self.freeze_strategy:
            self._freeze_encoder()
            logger.info("Encoder frozen (training task head only)")
        else:
            self._unfreeze_encoder()
            logger.info("Encoder unfrozen (full finetuning)")

    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()

    def _get_criterion(self) -> nn.Module:
        """Get loss function based on task type.

        Returns:
            Loss module.
        """
        if self.task_type in ("binary", "multiclass"):
            label_smoothing = self.config.training.get("label_smoothing", 0.0)
            return nn.CrossEntropyLoss(weight=self._class_weights, label_smoothing=label_smoothing)
        elif self.task_type == "multilabel":
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _setup_metrics(self) -> None:
        """Setup task-specific metrics using eval module."""
        # Build metric config from task settings
        output_dim = self.task_head.config.get_output_dim()

        # Get metrics from config if specified, otherwise use defaults
        eval_cfg = self.config.get("eval", {})
        metrics_cfg = eval_cfg.get("metrics", {})
        metric_names = metrics_cfg.get("names", None)
        threshold = metrics_cfg.get("threshold", 0.5)

        metric_config = MetricConfig(
            task_type=self.task_type,
            n_classes=output_dim,
            metrics=metric_names,
            threshold=threshold,
        )

        # Build metrics for each stage
        self.train_metrics = build_metrics(metric_config, prefix="train")
        self.val_metrics = build_metrics(metric_config, prefix="val")
        self.test_metrics = build_metrics(metric_config, prefix="test")

    def forward(
        self,
        timeseries: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder and task head.

        Args:
            timeseries: Input time-series of shape (B, T, D).
            mask: Observation mask of shape (B, T, D).

        Returns:
            Dictionary with 'logits' and 'probs'.
        """
        # Encoder forward (produces pooled representation)
        padding_mask = self._downstream_padding_mask(mask)
        encoder_out = self.encoder(
            timeseries,
            mask=mask,
            padding_mask=padding_mask,
        )  # (B, d_model)

        # Optional projection to shared dimensionality
        if self.projection is not None:
            encoder_out = self.projection(encoder_out)

        # Task head forward
        return self.task_head(encoder_out)

    def _downstream_padding_mask(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Exclude fully unobserved timesteps for obs-aware Transformer pooling."""
        if isinstance(self.encoder, TransformerEncoder) and getattr(
            self.encoder.config, "obs_aware", False
        ):
            return mask.any(dim=-1)
        return None

    def _validate_labels(self, labels: torch.Tensor) -> None:
        """One-time check that labels are compatible with the task type.

        For binary/multiclass tasks, labels must be non-negative integers within
        the valid class range. Detects the common mistake of passing float
        labels into a binary CrossEntropyLoss.
        """
        if self._labels_validated:
            return
        self._labels_validated = True

        if self.task_type not in ("binary", "multiclass"):
            return

        n_classes = self.task_head.config.get_output_dim()
        unique_vals = labels.unique()

        # Check for non-integer values (e.g., continuous floats, inf)
        has_non_integer = not torch.all(labels == labels.long().float())
        has_inf = torch.any(torch.isinf(labels))
        has_out_of_range = torch.any(labels.long() < 0) or torch.any(labels.long() >= n_classes)

        if has_inf or has_non_integer or has_out_of_range:
            raise ValueError(
                f"Labels incompatible with task_type='{self.task_type}' "
                f"(n_classes={n_classes}). "
                f"Found label values: {unique_vals.tolist()[:10]}. "
                f"Expected integer labels in [0, {n_classes}). "
                f"If using a regression task, set task_type='regression' "
                f"in your config."
            )

    def _compute_loss_and_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        """Compute loss and log metrics.

        Args:
            batch: Batch dictionary with 'timeseries', 'mask', and 'label'.
            stage: Training stage for metric prefix.

        Returns:
            Loss tensor.
        """
        timeseries = batch["timeseries"]  # (B, T, D)
        mask = batch["mask"]  # (B, T, D)
        labels = batch["label"]  # (B,)

        # One-time validation: catch label/task_type mismatches early
        self._validate_labels(labels)

        # Forward pass
        outputs = self(timeseries, mask)
        logits = outputs["logits"]  # (B, n_classes) or (B,)
        probs = outputs["probs"]

        # Compute loss
        if self.task_type == "regression":
            loss = self.criterion(logits.squeeze(-1), labels.float())
        elif self.task_type in ("binary", "multiclass"):
            loss = self.criterion(logits, labels.long())
        elif self.task_type == "multilabel":
            loss = self.criterion(logits, labels.float())
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        # Log loss
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log metrics
        metrics = getattr(self, f"{stage}_metrics", None)
        if metrics is not None:
            if self.task_type in ("binary", "multiclass", "multilabel"):
                labels_long = labels.long()  # Convert to long for metrics

                # For AUROC/AUPRC, use probabilities
                output_dim = self.task_head.config.get_output_dim()
                if output_dim == 2:
                    # Binary: use probability of positive class
                    metric_input = probs[:, 1]  # (B,)
                else:
                    metric_input = probs  # (B, n_classes)

                metrics.update(metric_input, labels_long)
            elif self.task_type == "regression":
                predictions = logits.squeeze(-1)
                metrics.update(predictions, labels.float())

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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        return self._compute_loss_and_metrics(batch, "train")

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch dictionary.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        return self._compute_loss_and_metrics(batch, "val")

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Test step.

        Args:
            batch: Batch dictionary.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        return self._compute_loss_and_metrics(batch, "test")

    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch.

        Handles two concerns:
        1. Gradual unfreezing: unfreeze encoder after warmup epochs.
        2. Frozen encoder eval mode: Lightning calls model.train() before each
           epoch, recursively re-enabling dropout on frozen encoder submodules.
           We counteract this by calling encoder.eval() when frozen.
        """
        # Gradual unfreezing: unfreeze encoder after warmup
        if self.unfreeze_epoch is not None and self.current_epoch >= self.unfreeze_epoch:
            if self.freeze_strategy:  # Only if initially frozen
                self._unfreeze_encoder()
                self.freeze_strategy = False  # Don't unfreeze again
                logger.info("Epoch %d: Unfroze encoder for finetuning", self.current_epoch)

        # Keep frozen encoder in eval mode to disable dropout.
        # Lightning calls model.train() before on_train_epoch_start,
        # which recursively sets all submodules to training mode.
        if self.freeze_strategy:
            self.encoder.eval()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and optional learning rate scheduler.

        Uses param groups (encoder + head) so that all parameters are tracked
        by the optimizer from the start. When the encoder is frozen, its params
        have requires_grad=False so the optimizer skips them (grad is None).
        When gradual unfreezing sets requires_grad=True, the optimizer
        immediately starts applying updates — no need to recreate it.

        Returns:
            Dictionary with optimizer and optional scheduler.
        """
        param_groups = [
            {"params": list(self.task_head.parameters())},
            {"params": list(self.encoder.parameters())},
        ]
        if self.projection is not None:
            param_groups.append({"params": list(self.projection.parameters())})
        optimizer = build_optimizer(param_groups, self.optimizer_config)

        result = build_scheduler(optimizer, self.scheduler_config)
        if result is None:
            return optimizer
        return result

    def get_encoder(self) -> nn.Module:
        """Get the encoder module.

        Returns:
            Encoder module.
        """
        return self.encoder

    def get_task_head(self) -> nn.Module:
        """Get the task head module.

        Returns:
            Task head module.
        """
        return self.task_head
