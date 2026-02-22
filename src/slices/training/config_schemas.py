"""Pydantic validation schemas for training module configs.

These schemas validate config dicts at the Hydra -> Lightning module boundary.
They catch typos and invalid values early (e.g. ``head_tpye`` instead of
``head_type``) by using ``extra="forbid"``.

Usage::

    from omegaconf import OmegaConf
    task_cfg = TaskConfig(**OmegaConf.to_container(config.task, resolve=True))

Note: Encoder and SSL objective configs are already validated by their own
dataclass constructors (TransformerConfig, MAEConfig, etc.), so they are
not duplicated here.
"""

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, field_validator


class TaskConfig(BaseModel):
    """Validated task configuration for finetuning.

    Catches typos like ``head_tpye`` or ``hiden_dims`` that would otherwise
    silently fall back to defaults via ``.get()``.
    """

    model_config = {"extra": "forbid"}

    task_name: str
    task_type: str = "binary"
    head_type: str = "mlp"
    hidden_dims: List[int] = [64]
    dropout: float = 0.1
    activation: str = "relu"
    n_classes: Optional[int] = None
    use_layer_norm: bool = False
    projection_dim: Optional[int] = None

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        valid = {"binary", "multiclass", "multilabel", "regression"}
        if v not in valid:
            raise ValueError(f"task_type must be one of {valid}, got '{v}'")
        return v

    @field_validator("head_type")
    @classmethod
    def validate_head_type(cls, v: str) -> str:
        valid = {"mlp", "linear"}
        if v not in valid:
            raise ValueError(f"head_type must be one of {valid}, got '{v}'")
        return v

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        valid = {"relu", "gelu", "silu", "tanh"}
        if v not in valid:
            raise ValueError(f"activation must be one of {valid}, got '{v}'")
        return v


class TrainingConfig(BaseModel):
    """Validated training configuration for finetuning.

    Catches typos like ``freez_encoder`` or ``unfreez_epoch`` that would
    otherwise silently fall back to defaults.
    """

    model_config = {"extra": "forbid"}

    max_epochs: int = 50
    batch_size: int = 64
    freeze_encoder: bool = True
    unfreeze_epoch: Optional[int] = None
    use_missing_token: bool = True
    class_weight: Optional[Union[Literal["balanced"], List[float]]] = None
    accelerator: str = "auto"
    devices: Any = "auto"
    precision: Any = 32
    gradient_clip_val: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
    early_stopping_patience: Optional[int] = 10
    early_stopping_monitor: Optional[str] = None
    early_stopping_mode: Optional[str] = None
    overfit_batches: Union[int, float] = 0


class OptimizerConfig(BaseModel):
    """Validated optimizer configuration."""

    model_config = {"extra": "forbid"}

    name: str
    lr: float
    weight_decay: float = 0.0
    momentum: float = 0.9

    @field_validator("name")
    @classmethod
    def validate_optimizer_name(cls, v: str) -> str:
        valid = {"adam", "adamw", "sgd"}
        if v.lower() not in valid:
            raise ValueError(f"optimizer name must be one of {valid}, got '{v}'")
        return v


class SchedulerConfig(BaseModel):
    """Validated scheduler configuration."""

    model_config = {"extra": "forbid"}

    name: str
    T_max: Optional[Any] = None
    eta_min: float = 0.0
    step_size: Optional[int] = None
    gamma: float = 0.1
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    warmup_epochs: Optional[int] = None
    max_epochs: Optional[int] = None
    monitor: str = "val/loss"

    @field_validator("name")
    @classmethod
    def validate_scheduler_name(cls, v: str) -> str:
        valid = {"cosine", "step", "plateau", "warmup_cosine"}
        if v.lower() not in valid:
            raise ValueError(f"scheduler name must be one of {valid}, got '{v}'")
        return v
