"""Abstract base class for downstream task heads.

Task heads are lightweight classifiers/regressors that sit on top of an encoder.
The encoder-task head composition is handled by the FineTuneModule.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Literal

import torch
import torch.nn as nn


@dataclass
class TaskHeadConfig:
    """Configuration for downstream task head.

    Attributes:
        name: Task head type (e.g., 'mlp', 'linear').
        task_name: Task identifier (e.g., 'mortality_24h').
        task_type: Type of task - determines output dimension and loss:
            - 'regression': Single continuous output
            - 'binary': Binary classification (2 outputs for CrossEntropy)
            - 'multiclass': Multi-class classification (n_classes outputs)
            - 'multilabel': Multi-label classification (n_classes outputs)
        n_classes: Number of classes (required for multiclass/multilabel, None otherwise).
        input_dim: Input dimension from encoder (set automatically).
        hidden_dims: Hidden layer dimensions for MLP heads.
        dropout: Dropout probability.
        activation: Activation function for hidden layers.
    """

    name: str = "mlp"
    task_name: str = "mortality_24h"
    task_type: Literal["regression", "binary", "multiclass", "multilabel"] = "binary"
    n_classes: int | None = None
    input_dim: int = 128  # Will be set from encoder
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    dropout: float = 0.1
    activation: str = "relu"
    use_layer_norm: bool = False  # Add LayerNorm before activation (SMART paper style)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate n_classes requirement
        if self.task_type in ("multiclass", "multilabel"):
            if self.n_classes is None:
                raise ValueError(f"n_classes is required for task_type='{self.task_type}'")
            if self.n_classes < 2:
                raise ValueError(
                    f"n_classes must be >= 2 for {self.task_type}, got {self.n_classes}"
                )

        # Warn if n_classes is set for binary/regression (we'll just ignore it)
        if self.task_type in ("regression", "binary") and self.n_classes is not None:
            import warnings

            warnings.warn(
                f"n_classes={self.n_classes} will be ignored for task_type='{self.task_type}'",
                UserWarning,
            )

    def get_output_dim(self) -> int:
        """Get output dimension for this task.

        Returns:
            Number of output units for the final layer.
        """
        if self.task_type == "regression":
            return 1
        elif self.task_type == "binary":
            return 2  # Use 2 outputs for CrossEntropyLoss
        elif self.task_type in ("multiclass", "multilabel"):
            if self.n_classes is None:
                raise ValueError(f"n_classes required for {self.task_type} tasks")
            return self.n_classes
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")


class BaseTaskHead(ABC, nn.Module):
    """Abstract base class for downstream task heads.

    Task heads transform encoder outputs into task-specific predictions.
    They are lightweight and typically just a few linear layers.

    The encoder is NOT part of the task head - composition is handled
    by the FineTuneModule, which allows flexible encoder freezing strategies.
    """

    def __init__(self, config: TaskHeadConfig) -> None:
        """Initialize task head with configuration.

        Args:
            config: Task head configuration.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, encoder_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through task head.

        Args:
            encoder_output: Encoder output of shape (B, d_model) for pooled
                          encoders, or (B, T, d_model) for sequence output.

        Returns:
            Dictionary containing task outputs:
            - 'logits': Raw logits of shape (B, output_dim)
            - 'probs': Probabilities (softmax/sigmoid of logits)
        """
        pass

    def get_output_dim(self) -> int:
        """Return the output dimension of the task head.

        Returns:
            Number of output units.
        """
        return self.config.get_output_dim()
