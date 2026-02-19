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
import warnings
from typing import Any, Dict, Literal, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from slices.eval import MetricConfig, build_metrics
from slices.models.encoders import EncoderWithMissingToken, build_encoder
from slices.models.heads import TaskHeadConfig, build_task_head
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
            self._load_encoder_weights(checkpoint_path)
        elif pretrain_checkpoint_path:
            self._load_from_pretrain_checkpoint(pretrain_checkpoint_path)
        elif self.use_missing_token:
            # No pretrained weights, but still wrap encoder for consistency
            self._wrap_encoder_with_missing_token(missing_token=None)

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
            use_layer_norm=task_cfg.get("use_layer_norm", False),
        )

    def _load_encoder_weights(self, path: str) -> None:
        """Load encoder weights from .pt file.

        Handles multiple checkpoint formats:
        - v3+: Contains encoder_config for automatic architecture detection
        - v2: Contains encoder_state_dict, missing_token but no config
        - v1/legacy: Raw state_dict

        For v3+, the encoder is rebuilt from saved config, ensuring architecture
        matches between pretraining and finetuning.

        Args:
            path: Path to encoder checkpoint.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If state dict keys don't match encoder architecture.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        # Detect checkpoint format
        if isinstance(checkpoint, dict) and "version" in checkpoint:
            version = checkpoint["version"]
            state_dict = checkpoint["encoder_state_dict"]
            missing_token = checkpoint.get("missing_token", None)

            # Version 3+: Rebuild encoder from saved config
            if version >= 3 and "encoder_config" in checkpoint:
                encoder_config = dict(checkpoint["encoder_config"])
                encoder_name = encoder_config.pop("name")
                # Override pooling with finetuning config's value.
                # SSL pretraining uses pooling='none' but finetuning needs
                # pooling='mean' or 'query' to get a single representation.
                # Pooling doesn't affect learned weights, just output aggregation.
                finetune_pooling = self.config.encoder.get("pooling", "mean")
                ckpt_pooling = encoder_config.get("pooling", "none")
                if ckpt_pooling != finetune_pooling:
                    encoder_config["pooling"] = finetune_pooling
                    logger.info(
                        "Overriding pooling: %s -> %s (finetuning requires aggregated output)",
                        ckpt_pooling,
                        finetune_pooling,
                    )
                self.encoder = build_encoder(encoder_name, encoder_config)
                logger.info(
                    "Rebuilt encoder from checkpoint config: %s (d_model=%s)",
                    encoder_name,
                    encoder_config.get("d_model", "N/A"),
                )
            elif version == 2:
                # v2 checkpoint: try to infer encoder type from state_dict keys
                inferred_encoder = self._infer_encoder_type(state_dict)
                config_encoder = self.config.encoder.name

                if inferred_encoder and inferred_encoder != config_encoder:
                    raise RuntimeError(
                        f"Encoder architecture mismatch!\n"
                        f"  Checkpoint appears to be: {inferred_encoder}\n"
                        f"  Config specifies: {config_encoder}\n\n"
                        f"Fix: Add 'encoder={inferred_encoder}' to your command:\n"
                        f"  uv run python scripts/training/finetune.py \\\n"
                        f"      checkpoint={path} \\\n"
                        f"      encoder={inferred_encoder} \\\n"
                        f"      task.task_name=...\n\n"
                        f"Or re-run pretraining to create a v3 checkpoint with embedded config."
                    )

            self.encoder.load_state_dict(state_dict)
            logger.info("Loaded encoder weights from: %s (format v%d)", path, version)

            # Wrap encoder with missing token if available and enabled
            if self.use_missing_token:
                self._wrap_encoder_with_missing_token(missing_token)
        else:
            # Old format (raw state_dict)
            warnings.warn(
                f"Loading old-format checkpoint from {path}. "
                "Missing token will be randomly initialized. "
                "Re-save the encoder using SSLPretrainModule.save_encoder() "
                "to include the learned missing_token.",
                UserWarning,
            )
            self.encoder.load_state_dict(checkpoint)
            logger.info("Loaded encoder weights from: %s (legacy format)", path)

            # Wrap encoder with random missing token if enabled
            if self.use_missing_token:
                self._wrap_encoder_with_missing_token(missing_token=None)

    def _load_from_pretrain_checkpoint(self, path: str) -> None:
        """Load encoder from full pretraining checkpoint (.ckpt).

        Also extracts the missing_token from the SSL objective if present
        and wraps the encoder with EncoderWithMissingToken.

        The encoder architecture is auto-detected from the checkpoint's
        hyperparameters (saved by save_hyperparameters() during pretraining),
        and the encoder is rebuilt with the correct architecture before
        loading weights.

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

        # Auto-detect encoder architecture from checkpoint hyperparameters
        if "hyper_parameters" in checkpoint:
            hyper_params = checkpoint["hyper_parameters"]
            if "config" in hyper_params and "encoder" in hyper_params["config"]:
                ckpt_encoder_cfg = hyper_params["config"]["encoder"]
                ckpt_encoder_name = ckpt_encoder_cfg.get("name")

                # Always rebuild encoder from checkpoint config to ensure dimensions match.
                # This handles both encoder name mismatches (e.g., transformer vs smart)
                # and parameter mismatches (e.g., d_model=64 vs d_model=32).
                encoder_config_dict = {k: v for k, v in ckpt_encoder_cfg.items() if k != "name"}
                # Override pooling with finetuning config's value.
                # SSL pretraining uses pooling='none' but finetuning needs
                # pooling='mean' or 'query' to get a single representation.
                finetune_pooling = self.config.encoder.get("pooling", "mean")
                ckpt_pooling = encoder_config_dict.get("pooling", "none")
                if ckpt_pooling != finetune_pooling:
                    encoder_config_dict["pooling"] = finetune_pooling
                    logger.info(
                        "Overriding pooling: %s -> %s (finetuning requires aggregated output)",
                        ckpt_pooling,
                        finetune_pooling,
                    )
                self.encoder = build_encoder(ckpt_encoder_name, encoder_config_dict)
                logger.info(
                    "Rebuilt encoder from checkpoint config: %s (d_model=%s)",
                    ckpt_encoder_name,
                    encoder_config_dict.get("d_model", "N/A"),
                )
        else:
            # No hyper_parameters - infer encoder type from state_dict keys
            inferred_encoder = self._infer_encoder_type(encoder_state_dict)
            config_encoder = self.config.encoder.name

            if inferred_encoder and inferred_encoder != config_encoder:
                raise RuntimeError(
                    f"Encoder architecture mismatch!\n"
                    f"  Checkpoint appears to be: {inferred_encoder}\n"
                    f"  Config specifies: {config_encoder}\n\n"
                    f"Fix: Add 'encoder={inferred_encoder}' to your command:\n"
                    f"  uv run python scripts/training/finetune.py \\\n"
                    f"      pretrain_checkpoint={path} \\\n"
                    f"      encoder={inferred_encoder} \\\n"
                    f"      task.task_name=...\n"
                )

        self.encoder.load_state_dict(encoder_state_dict)
        logger.info("Loaded encoder from pretrain checkpoint: %s", path)

        # Extract missing_token from SSL objective if present
        missing_token = None
        if "ssl_objective.missing_token" in state_dict:
            missing_token = state_dict["ssl_objective.missing_token"]

        # Wrap encoder with missing token if enabled
        if self.use_missing_token:
            self._wrap_encoder_with_missing_token(missing_token)

    def _infer_encoder_type(self, state_dict: Dict[str, Any]) -> Optional[str]:
        """Infer encoder type from state_dict keys.

        Used for v2 checkpoints that don't include encoder_config.

        Args:
            state_dict: Encoder state dictionary.

        Returns:
            Inferred encoder name or None if unknown.
        """
        keys = set(state_dict.keys())

        # SMART encoder has distinctive keys
        if any("embedder" in k or "blocks" in k or "seq_att" in k for k in keys):
            return "smart"

        # Standard transformer has input_proj and layers
        if any("input_proj" in k or "layers" in k for k in keys):
            return "transformer"

        return None

    def _wrap_encoder_with_missing_token(self, missing_token: Optional[torch.Tensor]) -> None:
        """Wrap encoder with EncoderWithMissingToken.

        Args:
            missing_token: Optional pretrained missing token. If None,
                a random token will be initialized.
        """
        d_input = self.encoder.config.d_input

        if missing_token is not None:
            self.encoder = EncoderWithMissingToken(
                encoder=self.encoder,
                d_input=d_input,
                missing_token=missing_token,
                init_missing_token=False,
            )
            logger.info("Wrapped encoder with pretrained missing token")
        else:
            self.encoder = EncoderWithMissingToken(
                encoder=self.encoder,
                d_input=d_input,
                missing_token=None,
                init_missing_token=True,
            )
            logger.info("Wrapped encoder with randomly initialized missing token")

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
                logger.info("Epoch %d: Unfroze encoder for finetuning", self.current_epoch)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and optional learning rate scheduler.

        Returns:
            Dictionary with optimizer and optional scheduler.
        """
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = build_optimizer(params, self.optimizer_config)

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
