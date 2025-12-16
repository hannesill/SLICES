"""Lightning module for downstream task finetuning.

This module wraps a pretrained encoder and task head for finetuning on
downstream clinical prediction tasks (mortality, LOS, AKI, sepsis, etc.).

Key features:
- Load pretrained encoder weights from SSL pretraining
- Flexible freezing strategies (freeze encoder, unfreeze for finetuning)
- Support for binary classification, multiclass, and regression tasks
- Comprehensive metrics (AUROC, AUPRC, accuracy, etc.)
"""

from typing import Any, Dict, Literal, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from slices.eval import MetricConfig, build_metrics
from slices.models.encoders import build_encoder
from slices.models.heads import TaskHeadConfig, build_task_head


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

        # Build encoder from config
        encoder_name = config.encoder.name
        encoder_config_dict = {k: v for k, v in config.encoder.items() if k != "name"}
        self.encoder = build_encoder(encoder_name, encoder_config_dict)

        # Load pretrained weights
        if checkpoint_path:
            self._load_encoder_weights(checkpoint_path)
        elif pretrain_checkpoint_path:
            self._load_from_pretrain_checkpoint(pretrain_checkpoint_path)

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

        # Setup loss function (use task head's task_type)
        self.task_type = self.task_head.config.task_type
        self.criterion = self._get_criterion()

        # Setup metrics
        self._setup_metrics()

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
            input_dim=self.encoder.get_output_dim(),
            hidden_dims=hidden_dims,
            dropout=task_cfg.get("dropout", 0.1),
            activation=task_cfg.get("activation", "relu"),
        )

    def _load_encoder_weights(self, path: str) -> None:
        """Load encoder weights from .pt file.

        Args:
            path: Path to encoder state dict.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If state dict keys don't match encoder architecture.
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state_dict)
        print(f"✓ Loaded encoder weights from: {path}")

    def _load_from_pretrain_checkpoint(self, path: str) -> None:
        """Load encoder from full pretraining checkpoint (.ckpt).

        Args:
            path: Path to Lightning checkpoint.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            KeyError: If checkpoint doesn't contain 'state_dict'.
            RuntimeError: If no encoder weights found in checkpoint.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if "state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint at {path} does not contain 'state_dict' key. "
                "Is this a valid PyTorch Lightning checkpoint?"
            )

        state_dict = checkpoint["state_dict"]

        # Extract encoder weights (prefixed with "encoder.")
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state_dict[key[8:]] = value  # Remove "encoder." prefix

        if not encoder_state_dict:
            raise RuntimeError(
                f"No encoder weights found in checkpoint {path}. "
                "Expected keys prefixed with 'encoder.' in state_dict."
            )

        self.encoder.load_state_dict(encoder_state_dict)
        print(f"✓ Loaded encoder from pretrain checkpoint: {path}")

    def _apply_freeze_strategy(self) -> None:
        """Apply freezing strategy to encoder."""
        if self.freeze_strategy:
            self._freeze_encoder()
            print("✓ Encoder frozen (training task head only)")
        else:
            self._unfreeze_encoder()
            print("✓ Encoder unfrozen (full finetuning)")

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
            # Use CrossEntropyLoss (works with 2+ class output)
            # Can add class weights for imbalanced data
            return nn.CrossEntropyLoss()
        elif self.task_type == "multilabel":
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _setup_metrics(self) -> None:
        """Setup task-specific metrics using eval module."""
        if self.task_type == "regression":
            # No metrics for regression yet
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
            return

        # Build metric config from task settings
        output_dim = self.task_head.config.get_output_dim()

        # Get metrics from config if specified, otherwise use defaults
        eval_cfg = self.config.get("eval", {})
        metrics_cfg = eval_cfg.get("metrics", {})
        metric_names = metrics_cfg.get("names", None)

        metric_config = MetricConfig(
            task_type=self.task_type,
            n_classes=output_dim,
            metrics=metric_names,
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
        encoder_out = self.encoder(timeseries, mask=mask)  # (B, d_model)

        # Task head forward
        return self.task_head(encoder_out)

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

        # Log classification metrics
        if self.task_type in ("binary", "multiclass", "multilabel"):
            labels_long = labels.long()  # Convert to long for metrics

            # For AUROC/AUPRC, use probabilities
            output_dim = self.task_head.config.get_output_dim()
            if output_dim == 2:
                # Binary: use probability of positive class
                metric_input = probs[:, 1]  # (B,)
            else:
                metric_input = probs  # (B, n_classes)

            # Get metrics for this stage
            metrics = getattr(self, f"{stage}_metrics", None)
            if metrics is not None:
                metrics.update(metric_input, labels_long)
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

        Used for gradual unfreezing strategy.
        """
        # Gradual unfreezing: unfreeze encoder after warmup
        if self.unfreeze_epoch is not None and self.current_epoch >= self.unfreeze_epoch:
            if self.freeze_strategy:  # Only if initially frozen
                self._unfreeze_encoder()
                self.freeze_strategy = False  # Don't unfreeze again
                print(f"✓ Epoch {self.current_epoch}: Unfroze encoder for finetuning")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and optional learning rate scheduler.

        Returns:
            Dictionary with optimizer and optional scheduler.
        """
        # Get parameters to optimize (only those with requires_grad)
        params = filter(lambda p: p.requires_grad, self.parameters())

        # Build optimizer
        optimizer_name = self.optimizer_config.name.lower()
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.get("weight_decay", 0.0)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            momentum = self.optimizer_config.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer '{optimizer_name}'")

        # Return optimizer only if no scheduler
        if self.scheduler_config is None:
            return optimizer

        # Build scheduler
        scheduler_name = self.scheduler_config.name.lower()

        if scheduler_name == "cosine":
            T_max = self.scheduler_config.get("T_max", 100)
            eta_min = self.scheduler_config.get("eta_min", 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
        elif scheduler_name == "step":
            step_size = self.scheduler_config.get("step_size", 30)
            gamma = self.scheduler_config.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        elif scheduler_name == "plateau":
            mode = self.scheduler_config.get("mode", "min")
            factor = self.scheduler_config.get("factor", 0.1)
            patience = self.scheduler_config.get("patience", 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }
        else:
            raise ValueError(f"Unknown scheduler '{scheduler_name}'")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

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
