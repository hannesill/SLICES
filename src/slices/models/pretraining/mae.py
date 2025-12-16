"""Masked Autoencoder (MAE) for self-supervised learning on ICU time-series.

Implements the MAE objective: mask portions of input, encode visible tokens,
decode to reconstruct masked tokens, and compute reconstruction loss.

Key features:
- Multiple masking strategies (random, block, timestep, feature)
- Respects observation mask (doesn't penalize truly missing values)
- Lightweight decoder with configurable architecture
- Separate reconstruction loss tracking for masked vs observed positions
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ...data.transforms import MaskingStrategy, apply_mask, create_ssl_mask
from .base import BaseSSLObjective, SSLConfig


@dataclass
class MAEConfig(SSLConfig):
    """Configuration for MAE objective.

    Note:
        - norm_target is NOT SUPPORTED and must be False. Data should be
          normalized during preprocessing instead.
        - During validation, masks are deterministic for reproducible metrics.
        - Block masking is capped to not exceed target mask_ratio significantly.
    """

    name: str = "mae"

    # Masking parameters
    # TODO: Make mask ratio dynamic during training (randomly sampled between two configured values)
    mask_ratio: float = 0.15  # Fraction of input to mask (BERT-style default)
    mask_strategy: MaskingStrategy = "random"  # random, block, timestep, feature
    min_block_size: int = 3  # For block masking
    max_block_size: int = 10
    mask_value: float = 0.0  # Value to use for masked positions

    # Decoder parameters
    decoder_d_model: int = 128  # Decoder hidden dimension
    decoder_n_layers: int = 2  # Number of decoder layers
    decoder_n_heads: int = 4  # Decoder attention heads
    decoder_d_ff: int = 512  # Decoder feedforward dimension
    decoder_dropout: float = 0.1

    # Loss parameters
    loss_on_observed_only: bool = True  # Only compute loss on originally observed values
    norm_target: bool = False  # NOT SUPPORTED - must be False, normalize data in preprocessing


class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction.

    Uses a simple transformer decoder to reconstruct masked inputs from
    encoder representations. The decoder is intentionally smaller than
    the encoder to prevent trivial solutions.
    """

    def __init__(
        self,
        d_encoder: int,
        d_input: int,
        config: MAEConfig,
    ) -> None:
        """Initialize MAE decoder.

        Args:
            d_encoder: Encoder output dimension.
            d_input: Original input dimension (reconstruction target).
            config: MAE configuration.
        """
        super().__init__()
        self.config = config

        # Project encoder output to decoder dimension
        self.encoder_proj = nn.Linear(d_encoder, config.decoder_d_model)

        # Decoder transformer layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_d_model,
            nhead=config.decoder_n_heads,
            dim_feedforward=config.decoder_d_ff,
            dropout=config.decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.decoder_n_layers,
        )

        # Project to reconstruction
        self.output_proj = nn.Linear(config.decoder_d_model, d_input)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Decode encoder output to reconstruct input.

        Args:
            encoder_output: Encoder output of shape (B, T, d_encoder).

        Returns:
            Reconstructed input of shape (B, T, d_input).
        """
        # Project encoder output
        x = self.encoder_proj(encoder_output)  # (B, T, d_decoder)

        # Pass through decoder
        x = self.decoder(x)  # (B, T, d_decoder)

        # Project to reconstruction
        x = self.output_proj(x)  # (B, T, d_input)

        return x


class MAEObjective(BaseSSLObjective):
    """Masked Autoencoder objective for ICU time-series.

    Implements the MAE SSL objective:
    1. Randomly mask portions of the input
    2. Encode the masked input
    3. Decode to reconstruct the original input
    4. Compute reconstruction loss on masked positions

    The encoder sees masked input (with mask tokens), while the decoder
    tries to reconstruct the original values. This encourages the encoder
    to learn robust representations that capture temporal structure.

    Example:
        >>> from slices.models.encoders import TransformerEncoder, TransformerConfig
        >>> enc_config = TransformerConfig(d_input=35, d_model=128, n_layers=4)
        >>> encoder = TransformerEncoder(enc_config)
        >>> mae_config = MAEConfig(mask_ratio=0.15, mask_strategy="block")
        >>> mae = MAEObjective(encoder, mae_config)
        >>> x = torch.randn(32, 48, 35)
        >>> obs_mask = torch.rand(32, 48, 35) > 0.3
        >>> loss, metrics = mae(x, obs_mask)
    """

    # Fixed seed for deterministic validation masks
    _VAL_MASK_SEED: int = 42

    def __init__(self, encoder: nn.Module, config: MAEConfig) -> None:
        """Initialize MAE objective.

        Args:
            encoder: Encoder module (e.g., TransformerEncoder).
            config: MAE configuration.
        """
        super().__init__(encoder, config)
        self.config: MAEConfig = config

        # Get encoder output dimension
        if hasattr(encoder, "get_output_dim"):
            d_encoder = encoder.get_output_dim()
        else:
            # Fallback: assume config has d_model
            d_encoder = encoder.config.d_model

        # Get input dimension
        d_input = encoder.config.d_input

        # Create decoder
        self.decoder = MAEDecoder(d_encoder, d_input, config)

        # Store encoder config for pooling checks
        self._encoder_pooling = getattr(encoder.config, "pooling", "none")
        if self._encoder_pooling != "none":
            raise ValueError(
                "MAE requires encoder with pooling='none' to get per-timestep "
                f"representations, but got pooling='{self._encoder_pooling}'"
            )

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MAE loss.

        Args:
            x: Input tensor of shape (B, T, D).
            obs_mask: Observation mask of shape (B, T, D) where True indicates
                     observed values and False indicates missing/imputed.

        Returns:
            Tuple of:
            - loss: Scalar loss tensor
            - metrics: Dict of additional metrics to log

        Note:
            During validation (self.training=False), masks are generated with a
            fixed seed for reproducibility. This ensures consistent validation
            metrics across epochs.
        """
        B, T, D = x.shape

        # Use deterministic masks for validation (reproducible metrics)
        if self.training:
            generator = None  # Random masks during training
        else:
            # Fixed seed for validation - ensures same masks each validation run
            generator = torch.Generator(device=x.device)
            generator.manual_seed(self._VAL_MASK_SEED)

        # Create SSL mask (which positions to mask for reconstruction)
        ssl_mask = create_ssl_mask(
            shape=(B, T, D),
            mask_ratio=self.config.mask_ratio,
            strategy=self.config.mask_strategy,
            obs_mask=obs_mask,
            min_block_size=self.config.min_block_size,
            max_block_size=self.config.max_block_size,
            device=x.device,
            generator=generator,
        )  # False = masked for SSL

        # Apply masking to input
        x_masked = apply_mask(x, ssl_mask, mask_value=self.config.mask_value)

        # Encode masked input
        encoder_output = self.encoder(x_masked, mask=obs_mask)  # (B, T, d_model)

        # Decode to reconstruct
        x_recon = self.decoder(encoder_output)  # (B, T, D)

        # Compute reconstruction loss
        loss, metrics = self._compute_reconstruction_loss(x_recon, x, ssl_mask, obs_mask)

        return loss, metrics

    def _compute_reconstruction_loss(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
        ssl_mask: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute reconstruction loss on masked positions.

        Args:
            x_recon: Reconstructed input of shape (B, T, D).
            x_target: Original input of shape (B, T, D).
            ssl_mask: SSL mask where False indicates masked positions.
            obs_mask: Observation mask where True indicates observed values.

        Returns:
            Tuple of (loss, metrics dict).
        """
        # Normalize targets if requested (per-sample normalization)
        if self.config.norm_target:
            # NOTE: This feature is disabled because the previous implementation
            # incorrectly normalized x_recon using target statistics after prediction,
            # creating a mismatch between training signal and model output.
            # A correct implementation would require architectural changes.
            raise NotImplementedError(
                "norm_target=True is not currently supported due to implementation "
                "issues. Use norm_target=False (default) instead. The input data "
                "should be normalized during preprocessing."
            )

        # Compute MSE
        squared_error = (x_recon - x_target) ** 2  # (B, T, D)

        # Determine which positions to include in loss
        if self.config.loss_on_observed_only:
            # Loss on masked AND originally observed values
            # ssl_mask: False = masked for SSL
            # obs_mask: True = originally observed
            # We want: masked for SSL AND originally observed
            loss_mask = (~ssl_mask) & obs_mask
        else:
            # Loss on all masked positions (even if originally missing)
            loss_mask = ~ssl_mask

        # Compute mean loss over valid positions
        loss = (squared_error * loss_mask.float()).sum() / loss_mask.float().sum().clamp(min=1)

        # Compute additional metrics
        with torch.no_grad():
            # Overall metrics
            n_total = ssl_mask.numel()
            n_masked = (~ssl_mask).sum().item()
            n_observed = obs_mask.sum().item()
            n_loss_positions = loss_mask.sum().item()

            # Loss on visible (unmasked) positions for monitoring
            visible_mask = ssl_mask & obs_mask
            if visible_mask.sum() > 0:
                visible_loss = (
                    squared_error * visible_mask.float()
                ).sum() / visible_mask.float().sum()
            else:
                visible_loss = torch.tensor(0.0, device=loss.device)

            metrics = {
                "mae_loss": loss.detach(),
                "mae_recon_loss_masked": loss.detach(),
                "mae_recon_loss_visible": visible_loss,
                "mae_mask_ratio_actual": n_masked / n_total,
                "mae_obs_ratio": n_observed / n_total,
                "mae_loss_positions": n_loss_positions / n_total,
            }

        return loss, metrics
