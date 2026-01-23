"""SMART SSL objective for self-supervised learning on ICU time-series.

Implements the SMART objective: momentum-based self-supervised learning
with per-sample random masking that reconstructs target encoder representations.

Reference: "SMART: Towards Pre-trained Missing-Aware Model for Patient Health
Status Prediction" (NeurIPS 2024)
Paper: https://openreview.net/forum?id=7UenF4kx4j
Code: https://github.com/yzhHoward/SMART

Key features (matching original implementation):
- Momentum encoder (EMA target) with linear momentum schedule (0.996 → 1.0)
- Per-sample random mask ratios from U[min_mask_ratio, max_mask_ratio]
- ELEMENT-WISE masking (each value masked independently, not entire timesteps)
- Simple MLP predictor (NOT transformer) to predict target representations
- Smooth L1 loss on masked positions
"""

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.smart import SMARTEncoder
from .base import BaseSSLObjective, SSLConfig


@dataclass
class SMARTSSLConfig(SSLConfig):
    """Configuration for SMART SSL objective.

    SMART uses momentum-based self-distillation where an online encoder
    learns to predict the representations from a momentum-updated target encoder.
    """

    name: str = "smart"

    # Masking parameters (per-sample random ratios)
    # Original SMART uses ELEMENT-WISE masking, not timestep-level
    min_mask_ratio: float = 0.0  # Minimum mask ratio (inclusive)
    max_mask_ratio: float = 0.75  # Maximum mask ratio (inclusive)

    # Predictor parameters (simple MLP, NOT transformer - matches original)
    # Original uses: MLP(d_model -> 4*d_model -> d_model) + Linear(d_model -> d_model)
    predictor_mlp_ratio: float = 4.0  # Hidden dimension = d_model * mlp_ratio
    predictor_dropout: float = 0.1

    # Momentum encoder parameters
    momentum_base: float = 0.996  # Initial momentum (start of training)
    momentum_final: float = 1.0  # Final momentum (end of training)

    # Loss parameters
    loss_type: str = "smooth_l1"  # "smooth_l1" or "mse"
    smooth_l1_beta: float = 1.0  # Beta for smooth L1 loss
    loss_on_observed_only: bool = True  # Only compute loss on originally observed values


class SMARTPredictor(nn.Module):
    """Simple MLP predictor for SMART (matches original implementation).

    The original SMART uses a lightweight MLP predictor, NOT a transformer.
    Architecture: MLP(d_model -> hidden -> d_model) + Linear projection

    Takes 4D representations from the online encoder and predicts
    the target encoder representations.
    """

    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize SMART predictor.

        Args:
            d_model: Encoder output dimension.
            mlp_ratio: Hidden dimension multiplier (hidden = d_model * mlp_ratio).
            dropout: Dropout probability.
        """
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)

        # MLP block (matches original Mlp class)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        # Final projection (matches original proj_out)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict target representations.

        Args:
            x: Online encoder output of shape (B, V, T, d_encoder).

        Returns:
            Predicted representations of shape (B, V, T, d_encoder).
        """
        # MLP operates element-wise on the last dimension
        x = self.mlp(x)
        x = self.proj_out(x)
        return x


class SMARTObjective(BaseSSLObjective):
    """SMART SSL objective for ICU time-series.

    Implements the SMART self-supervised learning objective:
    1. Sample per-sample mask ratios from U[min_mask_ratio, max_mask_ratio]
    2. Zero out masked timesteps in input
    3. Online encoder (with masking) produces representations
    4. Target encoder (without masking, EMA updated) produces target representations
    5. Predictor predicts target representations from online representations
    6. Smooth L1 loss on masked positions

    Requires SMARTEncoder (MART architecture) for proper 4D tensor handling.

    Example:
        >>> from slices.models.encoders import SMARTEncoder, SMARTEncoderConfig
        >>> enc_config = SMARTEncoderConfig(d_input=35, d_model=32, n_layers=2, pooling="none")
        >>> encoder = SMARTEncoder(enc_config)
        >>> smart_config = SMARTSSLConfig(max_mask_ratio=0.75)
        >>> smart = SMARTObjective(encoder, smart_config)
        >>> x = torch.randn(32, 48, 35)
        >>> obs_mask = torch.rand(32, 48, 35) > 0.3
        >>> loss, metrics = smart(x, obs_mask)
    """

    def __init__(self, encoder: nn.Module, config: SMARTSSLConfig) -> None:
        """Initialize SMART objective.

        Args:
            encoder: SMARTEncoder module (must be MART architecture).
            config: SMART SSL configuration.

        Raises:
            ValueError: If encoder is not a SMARTEncoder.
            ValueError: If encoder pooling is not "none".
        """
        super().__init__(encoder, config)
        self.config: SMARTSSLConfig = config

        # Validate encoder type
        if not isinstance(encoder, SMARTEncoder):
            raise ValueError(
                f"SMARTObjective requires a SMARTEncoder, but got {type(encoder).__name__}. "
                "SMART uses MART architecture with 4D tensor handling that requires "
                "the specialized SMARTEncoder implementation."
            )

        # Validate encoder pooling
        if encoder.config.pooling != "none":
            raise ValueError(
                f"SMARTObjective requires encoder with pooling='none' for 4D output, "
                f"but got pooling='{encoder.config.pooling}'"
            )

        # Get encoder dimensions
        self.d_model = encoder.config.d_model
        self.d_input = encoder.config.d_input  # Number of variables

        # Create target encoder as EMA copy
        self.target_encoder = copy.deepcopy(encoder)
        # Freeze target encoder - only updated via momentum
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Create predictor (simple MLP like original)
        self.predictor = SMARTPredictor(
            d_model=self.d_model,
            mlp_ratio=config.predictor_mlp_ratio,
            dropout=config.predictor_dropout,
        )

        # Track momentum for logging
        self._current_momentum = config.momentum_base

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute SMART SSL loss.

        Args:
            x: Input tensor of shape (B, T, D) in SLICES format.
            obs_mask: Observation mask of shape (B, T, D) where True = observed.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        B, T, D = x.shape

        # Step 1: Sample per-sample mask ratios from U[min, max]
        mask_ratios = (
            torch.rand(B, device=x.device)
            * (self.config.max_mask_ratio - self.config.min_mask_ratio)
            + self.config.min_mask_ratio
        )  # (B,)

        # Step 2: Create per-sample SSL masks (ELEMENT-WISE masking like original)
        # ssl_mask: True = visible, False = masked for SSL
        ssl_mask = self._create_per_sample_mask(
            shape=(B, T, D),
            mask_ratios=mask_ratios,
            obs_mask=obs_mask,
            device=x.device,
        )

        # Step 3: Create masked input for online encoder
        # Zero out masked timesteps
        x_masked = x.clone()
        x_masked = x_masked * ssl_mask.float()  # Zero where ssl_mask is False

        # Also update obs_mask for masked positions - tell encoder these are "missing"
        obs_mask_masked = obs_mask & ssl_mask  # Only keep observed+visible

        # Step 4: Online encoder forward pass
        # Returns (B, V, T, d_model) since pooling="none"
        online_repr = self.encoder(x_masked, mask=obs_mask_masked, padding_mask=None)

        # Step 5: Target encoder forward pass (no masking, no grad)
        with torch.no_grad():
            target_repr = self.target_encoder(x, mask=obs_mask, padding_mask=None)

        # Step 6: Predictor predicts target representations
        predicted_repr = self.predictor(online_repr)  # (B, V, T, d_model)

        # Step 7: Compute loss on masked positions
        loss, metrics = self._compute_loss(predicted_repr, target_repr, ssl_mask, obs_mask)

        # Add mask ratio metrics
        with torch.no_grad():
            metrics["smart_mask_ratio_mean"] = mask_ratios.mean().item()
            metrics["smart_mask_ratio_std"] = mask_ratios.std().item()
            metrics["smart_momentum"] = self._current_momentum

        return loss, metrics

    def _create_per_sample_mask(
        self,
        shape: Tuple[int, int, int],
        mask_ratios: torch.Tensor,
        obs_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Create per-sample SSL masks with different mask ratios.

        Uses ELEMENT-WISE masking (each (t, v) position masked independently).
        This matches the original SMART implementation which uses:
            mask = torch.rand_like(x) < mask_ratios.view(-1, 1, 1)

        Args:
            shape: Shape (B, T, D) of input.
            mask_ratios: Per-sample mask ratios of shape (B,).
            obs_mask: Observation mask of shape (B, T, D).
            device: Device to create mask on.

        Returns:
            SSL mask of shape (B, T, D) where False = masked.
        """
        B, T, D = shape

        # Element-wise random masking (matches original SMART)
        # Random values for each element (B, T, D)
        rand_vals = torch.rand(B, T, D, device=device)

        # Compare to per-sample mask ratios
        # mask_ratios: (B,) -> (B, 1, 1) for broadcasting
        # ssl_mask: True = visible, False = masked
        ssl_mask = rand_vals >= mask_ratios.view(B, 1, 1)

        # Only mask observed values (don't re-mask already missing values)
        # ssl_mask: False = masked for SSL
        # obs_mask: True = observed
        # Result: positions that are already missing stay visible (not penalized)
        ssl_mask = ssl_mask | (~obs_mask)

        return ssl_mask

    def _compute_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        ssl_mask: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss on masked positions.

        Args:
            predicted: Predicted representations (B, V, T, d_model).
            target: Target representations (B, V, T, d_model).
            ssl_mask: SSL mask (B, T, D) where False = masked.
            obs_mask: Observation mask (B, T, D) where True = observed.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        B, V, T, d_model = predicted.shape

        # Transpose ssl_mask and obs_mask to (B, V, T) format to match representations
        ssl_mask_t = ssl_mask.transpose(1, 2)  # (B, D, T) = (B, V, T)
        obs_mask_t = obs_mask.transpose(1, 2)  # (B, D, T) = (B, V, T)

        # Determine loss mask
        if self.config.loss_on_observed_only:
            # Loss on masked AND originally observed positions
            # ssl_mask_t: False = masked for SSL
            # obs_mask_t: True = originally observed
            loss_mask = (~ssl_mask_t) & obs_mask_t  # (B, V, T)
        else:
            # Loss on all masked positions
            loss_mask = ~ssl_mask_t  # (B, V, T)

        # Expand loss_mask to match representation shape
        loss_mask_expanded = loss_mask.unsqueeze(-1).expand_as(predicted)  # (B, V, T, d_model)

        # Compute element-wise loss
        if self.config.loss_type == "smooth_l1":
            element_loss = F.smooth_l1_loss(
                predicted, target, beta=self.config.smooth_l1_beta, reduction="none"
            )
        elif self.config.loss_type == "mse":
            element_loss = F.mse_loss(predicted, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        # Apply mask and compute mean
        masked_loss = element_loss * loss_mask_expanded.float()
        n_loss_positions = loss_mask_expanded.sum().clamp(min=1)
        loss = masked_loss.sum() / n_loss_positions

        # Compute metrics
        with torch.no_grad():
            n_total = ssl_mask.numel()
            n_masked = (~ssl_mask).sum().item()
            n_observed = obs_mask.sum().item()
            n_loss_pos = loss_mask.sum().item()

            # Loss on visible positions for monitoring
            visible_mask = ssl_mask_t & obs_mask_t
            visible_mask_expanded = visible_mask.unsqueeze(-1).expand_as(predicted)
            if visible_mask_expanded.sum() > 0:
                visible_loss = (
                    element_loss * visible_mask_expanded.float()
                ).sum() / visible_mask_expanded.sum()
            else:
                visible_loss = torch.tensor(0.0, device=loss.device)

            metrics = {
                "smart_loss": loss.detach(),
                "reconstruction_loss": loss.detach(),  # Alias for consistent naming
                "smart_loss_masked": loss.detach(),
                "smart_loss_visible": visible_loss,
                "smart_mask_ratio_actual": n_masked / n_total,
                "smart_obs_ratio": n_observed / n_total,
                "smart_loss_positions": n_loss_pos / (B * V * T),
            }

        return loss, metrics

    @torch.no_grad()
    def momentum_update(self, progress: float) -> None:
        """Update target encoder parameters with exponential moving average.

        Should be called after each optimizer step via the on_train_batch_end hook.

        Args:
            progress: Training progress as fraction in [0, 1].
                     0 = start of training, 1 = end of training.
        """
        # Linear momentum schedule: base → final
        m = (
            self.config.momentum_base
            + (self.config.momentum_final - self.config.momentum_base) * progress
        )
        self._current_momentum = m

        # EMA update: target = m * target + (1-m) * online
        for online_param, target_param in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.mul_(m).add_(online_param.data, alpha=1 - m)

    def get_encoder(self) -> nn.Module:
        """Return the online encoder for downstream use.

        Returns:
            The online encoder module.
        """
        return self.encoder
