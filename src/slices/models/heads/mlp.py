"""MLP task head for downstream classification/regression tasks.

Simple multi-layer perceptron heads suitable for most clinical prediction tasks.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTaskHead, TaskHeadConfig


class MLPTaskHead(BaseTaskHead):
    """Multi-layer perceptron task head.
    
    A flexible MLP head that can be configured for:
    - Regression (1 output)
    - Binary classification (2 outputs for CrossEntropy)
    - Multi-class classification (n_classes outputs)
    - Multi-label classification (n_classes outputs)
    
    Example:
        >>> config = TaskHeadConfig(
        ...     name="mlp",
        ...     task_name="mortality_24h",
        ...     task_type="binary",
        ...     input_dim=128,
        ...     hidden_dims=[64],
        ...     dropout=0.1,
        ... )
        >>> head = MLPTaskHead(config)
        >>> encoder_out = torch.randn(32, 128)  # (batch, d_model)
        >>> outputs = head(encoder_out)
        >>> outputs['logits'].shape
        torch.Size([32, 2])
    """
    
    def __init__(self, config: TaskHeadConfig) -> None:
        """Initialize MLP task head.
        
        Args:
            config: Task head configuration.
        """
        super().__init__(config)
        
        # Build MLP layers
        layers = []
        in_dim = config.input_dim
        
        # Hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        
        # Output layer
        output_dim = config.get_output_dim()
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.
        
        Args:
            name: Activation function name.
        
        Returns:
            Activation module.
        """
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        if name not in activations:
            raise ValueError(
                f"Unknown activation '{name}'. Choose from: {list(activations.keys())}"
            )
        return activations[name]
    
    def forward(self, encoder_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through MLP head.
        
        Args:
            encoder_output: Encoder output of shape (B, d_model).
        
        Returns:
            Dictionary with 'logits' and 'probs'.
        """
        logits = self.mlp(encoder_output)  # (B, output_dim)
        
        # Compute probabilities based on task type
        if self.config.task_type == "regression":
            probs = logits  # No activation for regression
        elif self.config.task_type in ("binary", "multiclass"):
            probs = F.softmax(logits, dim=-1)
        elif self.config.task_type == "multilabel":
            probs = torch.sigmoid(logits)  # Independent probabilities per label
        else:
            raise ValueError(f"Unknown task_type: {self.config.task_type}")
        
        return {
            "logits": logits,
            "probs": probs,
        }


class LinearTaskHead(BaseTaskHead):
    """Simple linear (single layer) task head.
    
    Minimal head with just a linear layer - useful for linear probing
    to evaluate encoder representations without additional capacity.
    
    Example:
        >>> config = TaskHeadConfig(
        ...     name="linear",
        ...     task_name="mortality_24h",
        ...     task_type="binary",
        ...     input_dim=128,
        ... )
        >>> head = LinearTaskHead(config)
    """
    
    def __init__(self, config: TaskHeadConfig) -> None:
        """Initialize linear task head.
        
        Args:
            config: Task head configuration.
        """
        super().__init__(config)
        
        # Single linear layer
        output_dim = config.get_output_dim()
        self.linear = nn.Linear(config.input_dim, output_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, encoder_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through linear head.
        
        Args:
            encoder_output: Encoder output of shape (B, d_model).
        
        Returns:
            Dictionary with 'logits' and 'probs'.
        """
        x = self.dropout(encoder_output)
        logits = self.linear(x)  # (B, output_dim)
        
        # Compute probabilities based on task type
        if self.config.task_type == "regression":
            probs = logits
        elif self.config.task_type in ("binary", "multiclass"):
            probs = F.softmax(logits, dim=-1)
        elif self.config.task_type == "multilabel":
            probs = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unknown task_type: {self.config.task_type}")
        
        return {
            "logits": logits,
            "probs": probs,
        }
