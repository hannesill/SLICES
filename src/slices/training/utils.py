"""Training utilities and helpers.

Shared optimizer/scheduler construction for pretrain and finetune modules,
and a shared checkpoint save helper.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn


def build_optimizer(
    params: Union[Iterator[nn.Parameter], List[Dict[str, Any]]],
    config: Any,
) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Args:
        params: Model parameters to optimize. Can be an iterator of
                parameters or a list of param group dicts.
        config: Optimizer config with 'name', 'lr', and optional
                'weight_decay', 'momentum' fields.

    Returns:
        Configured optimizer.

    Raises:
        ValueError: If optimizer name is not recognized.
    """
    name = config.name.lower()
    lr = config.lr
    weight_decay = config.get("weight_decay", 0.0)

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = config.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Supported: adam, adamw, sgd")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
) -> Optional[Dict[str, Any]]:
    """Build learning rate scheduler from config.

    Args:
        optimizer: Optimizer to schedule.
        config: Scheduler config with 'name' and scheduler-specific fields.
                If None, returns None.

    Returns:
        Lightning-compatible scheduler dict, or None if no scheduler.

    Raises:
        ValueError: If scheduler name is not recognized.
    """
    if config is None:
        return None

    name = config.name.lower()

    if name == "cosine":
        T_max = config.get("T_max", 100)
        eta_min = config.get("eta_min", 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
    elif name == "step":
        step_size = config.get("step_size", 30)
        gamma = config.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif name == "plateau":
        mode = config.get("mode", "min")
        factor = config.get("factor", 0.1)
        patience = config.get("patience", 10)
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
                "monitor": config.get("monitor", "val/loss"),
            },
        }
    elif name == "warmup_cosine":
        warmup_epochs = config.get("warmup_epochs", 10)
        max_epochs = config.get("max_epochs", 100)
        eta_min = config.get("eta_min", 0.0)

        # Get base_lr to make eta_min absolute (consistent with CosineAnnealingLR).
        # LambdaLR multiplies the lambda return by base_lr, so we divide eta_min
        # by base_lr here so the effective minimum LR equals eta_min exactly.
        base_lr = optimizer.defaults["lr"]
        eta_min_ratio = eta_min / base_lr if base_lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return (
                    eta_min_ratio
                    + (1 - eta_min_ratio)
                    * 0.5
                    * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. Supported: cosine, step, plateau, warmup_cosine"
        )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
        },
    }


def save_encoder_checkpoint(
    encoder: nn.Module,
    encoder_config: Dict[str, Any],
    path: Union[str, Path],
    missing_token: Optional[torch.Tensor] = None,
    d_input: Optional[int] = None,
) -> None:
    """Save encoder in v3 checkpoint format.

    Standardized checkpoint saving used across pretrain, finetune, and supervised
    training scripts.

    Args:
        encoder: Encoder module whose state_dict to save.
        encoder_config: Dict with 'name' and encoder architecture params.
        path: Path to save the checkpoint.
        missing_token: Optional learned missing token tensor.
        d_input: Optional input dimension (for token shape validation).
    """
    checkpoint: Dict[str, Any] = {
        "encoder_state_dict": encoder.state_dict(),
        "encoder_config": encoder_config,
        "version": 3,
    }

    if missing_token is not None:
        checkpoint["missing_token"] = missing_token.data.clone()
        if d_input is not None:
            checkpoint["d_input"] = d_input

    torch.save(checkpoint, path)
