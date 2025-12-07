"""Lightning module for SSL pretraining.

This module wraps encoder and SSL objective for end-to-end training with
Lightning's training loop, logging, checkpointing, and distributed training.
"""

from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from slices.models.encoders import build_encoder
from slices.models.pretraining import build_ssl_objective, get_ssl_config_class


class SSLPretrainModule(pl.LightningModule):
    """Lightning module for SSL pretraining.
    
    This module is agnostic to:
    - Encoder architecture (transformer, RNN, CNN, etc.)
    - SSL objective (MAE, contrastive, JEPA, etc.)
    
    All models and objectives are built from configuration using factory patterns.
    
    Example:
        >>> config = {
        ...     "encoder": {
        ...         "name": "transformer",
        ...         "d_input": 35,
        ...         "d_model": 128,
        ...         "n_layers": 4,
        ...         "pooling": "none"
        ...     },
        ...     "ssl": {
        ...         "name": "mae",
        ...         "mask_ratio": 0.15,
        ...         "mask_strategy": "block"
        ...     },
        ...     "optimizer": {
        ...         "name": "adamw",
        ...         "lr": 1e-3,
        ...         "weight_decay": 0.01
        ...     }
        ... }
        >>> module = SSLPretrainModule(config)
    """

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """Initialize pretraining module.
        
        Args:
            config: Hydra configuration with encoder, ssl, and optimizer settings.
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Build encoder from config
        encoder_name = config.encoder.name
        encoder_config_dict = {
            k: v for k, v in config.encoder.items() if k != "name"
        }
        self.encoder = build_encoder(encoder_name, encoder_config_dict)
        
        # Build SSL objective from config
        ssl_name = config.ssl.name
        ssl_config_cls = get_ssl_config_class(ssl_name)
        ssl_config_dict = {
            k: v for k, v in config.ssl.items() if k != "name"
        }
        ssl_config = ssl_config_cls(name=ssl_name, **ssl_config_dict)
        self.ssl_objective = build_ssl_objective(self.encoder, ssl_config)
        
        # Store config for optimizer
        self.optimizer_config = config.optimizer
        self.scheduler_config = config.get("scheduler", None)
        
    def forward(
        self,
        timeseries: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through SSL objective.
        
        Args:
            timeseries: Input time-series of shape (B, T, D).
            mask: Observation mask of shape (B, T, D).
        
        Returns:
            Tuple of (loss, metrics_dict).
        """
        return self.ssl_objective(timeseries, mask)
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch dictionary with 'timeseries' and 'mask'.
            batch_idx: Batch index.
        
        Returns:
            Loss tensor for backpropagation.
        """
        timeseries = batch["timeseries"]  # (B, T, D)
        mask = batch["mask"]  # (B, T, D)
        
        # Forward pass
        loss, metrics = self(timeseries, mask)
        
        # Log loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        # Log all metrics from SSL objective
        for key, value in metrics.items():
            self.log(
                f"train/{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        
        return loss
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Batch dictionary with 'timeseries' and 'mask'.
            batch_idx: Batch index.
        
        Returns:
            Loss tensor.
        """
        timeseries = batch["timeseries"]  # (B, T, D)
        mask = batch["mask"]  # (B, T, D)
        
        # Forward pass
        loss, metrics = self(timeseries, mask)
        
        # Log loss
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        # Log all metrics
        for key, value in metrics.items():
            self.log(
                f"val/{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and optional learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and optional scheduler.
        """
        # Build optimizer
        optimizer_name = self.optimizer_config.name.lower()
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.get("weight_decay", 0.0)
        
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            momentum = self.optimizer_config.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. "
                f"Supported: adam, adamw, sgd"
            )
        
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
        elif scheduler_name == "warmup_cosine":
            # Warmup + cosine decay
            warmup_epochs = self.scheduler_config.get("warmup_epochs", 10)
            max_epochs = self.scheduler_config.get("max_epochs", 100)
            eta_min = self.scheduler_config.get("eta_min", 0.0)
            
            def lr_lambda(epoch: int) -> float:
                if epoch < warmup_epochs:
                    # Linear warmup: start at lr/warmup_epochs, reach lr at end
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                else:
                    # Cosine decay
                    progress = float(epoch - warmup_epochs) / float(
                        max(1, max_epochs - warmup_epochs)
                    )
                    return eta_min + (1 - eta_min) * 0.5 * (
                        1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))
                    ).item()
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(
                f"Unknown scheduler '{scheduler_name}'. "
                f"Supported: cosine, step, plateau, warmup_cosine"
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def get_encoder(self) -> nn.Module:
        """Get the encoder module for downstream tasks.
        
        Returns:
            Encoder module (can be used for fine-tuning).
        """
        return self.encoder
    
    def save_encoder(self, path: str) -> None:
        """Save encoder weights for downstream fine-tuning.
        
        Args:
            path: Path to save encoder weights.
        """
        torch.save(self.encoder.state_dict(), path)
