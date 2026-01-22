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

from ...data.transforms import MaskingStrategy, create_ssl_mask
from .base import BaseSSLObjective, SSLConfig


@dataclass
class MAEConfig(SSLConfig):
    """Configuration for MAE objective.

    Two-Token System:
        This implementation uses two learned tokens to handle missing/masked values:
        - **MISSING_TOKEN**: Replaces genuinely missing positions (obs_mask=False).
          These positions were never observed in the ICU data.
        - **MASK_TOKEN**: Replaces SSL-masked positions (obs_mask=True AND ssl_mask=False).
          These are artificially hidden for self-supervision.

        Both tokens have shape (1, 1, d_input) - one learned scalar per feature.
        The encoder receives input with these tokens substituted, and reconstruction
        loss is only computed on MASK_TOKEN positions.

    Note:
        - norm_target is NOT SUPPORTED and must be False. Data should be
          normalized during preprocessing instead.
        - Random masks are used for both training and validation (standard practice).
        - Block masking is capped to not exceed target mask_ratio significantly.
    """

    name: str = "mae"

    # Masking parameters
    # TODO: Make mask ratio dynamic during training (randomly sampled between two configured values)
    mask_ratio: float = 0.15  # Fraction of input to mask (BERT-style default)
    mask_strategy: MaskingStrategy = "random"  # random, block, timestep, feature
    min_block_size: int = 3  # For block masking
    max_block_size: int = 10

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
    1. Replace genuinely missing positions with MISSING_TOKEN
    2. Create SSL mask and replace those positions with MASK_TOKEN
    3. Encode the modified input
    4. Decode to reconstruct the original input
    5. Compute reconstruction loss only on MASK_TOKEN positions

    Two-Token System:
        This implementation uses two learned tokens to clearly distinguish
        between different types of "unknown" values:

        - **MISSING_TOKEN**: Replaces genuinely missing positions (obs_mask=False).
          These were never observed in the ICU. No reconstruction loss is computed
          on these positions since we don't know the true values.

        - **MASK_TOKEN**: Replaces SSL-masked positions (obs_mask=True AND selected
          for masking by the SSL objective). Reconstruction loss IS computed on
          these positions to train the model.

        Both tokens have shape (1, 1, d_input) with one learned scalar per feature.
        This allows each feature to have its own "missing" and "masked" representation.

    Example:
        >>> from slices.models.encoders import TransformerEncoder, TransformerConfig
        >>> enc_config = TransformerConfig(d_input=35, d_model=128, n_layers=4, pooling="none")
        >>> encoder = TransformerEncoder(enc_config)
        >>> mae_config = MAEConfig(mask_ratio=0.15)
        >>> mae = MAEObjective(encoder, mae_config)
        >>> x = torch.randn(32, 48, 35)
        >>> obs_mask = torch.rand(32, 48, 35) > 0.3
        >>> loss, metrics = mae(x, obs_mask)
    """

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

        # Two learned tokens: one per feature dimension
        # Shape (1, 1, D) broadcasts across batch and time dimensions
        # MISSING_TOKEN: replaces genuinely missing positions (obs_mask=False)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_input))
        # MASK_TOKEN: replaces SSL-masked positions (obs_mask=True AND ssl_mask=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_input))

        # Initialize with small random values
        nn.init.normal_(self.missing_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

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
        """Compute MAE loss using the two-token system.

        Args:
            x: Input tensor of shape (B, T, D). May contain imputed values at
               positions where obs_mask=False.
            obs_mask: Observation mask of shape (B, T, D) where True indicates
                     observed values and False indicates missing/imputed.

        Returns:
            Tuple of:
            - loss: Scalar loss tensor
            - metrics: Dict of additional metrics to log

        Note:
            - Random masks are used for both training and validation (standard practice).
            - MISSING_TOKEN is placed at genuinely missing positions (obs_mask=False)
            - MASK_TOKEN is placed at SSL-masked positions (obs_mask=True AND selected for masking)
            - Loss is only computed on MASK_TOKEN positions
        """
        B, T, D = x.shape

        # Use random masks for both training and validation
        # This is standard practice in MAE and makes metrics directly comparable
        generator = None

        # Step 1: Replace genuinely missing positions with MISSING_TOKEN
        # obs_mask: True = observed, False = missing
        x_input = torch.where(obs_mask, x, self.missing_token.expand(B, T, D))

        # Step 2: Create SSL mask (which observed positions to mask for reconstruction)
        # ssl_mask: True = visible, False = masked for SSL
        ssl_mask = create_ssl_mask(
            shape=(B, T, D),
            mask_ratio=self.config.mask_ratio,
            strategy=self.config.mask_strategy,
            obs_mask=obs_mask,  # Only mask observed positions
            min_block_size=self.config.min_block_size,
            max_block_size=self.config.max_block_size,
            device=x.device,
            generator=generator,
        )

        # Step 3: Replace SSL-masked positions with MASK_TOKEN
        # SSL-masked positions are:
        # - originally observed (obs_mask=True)
        # - selected for masking (ssl_mask=False)
        ssl_masked_positions = obs_mask & ~ssl_mask
        x_input = torch.where(ssl_masked_positions, self.mask_token.expand(B, T, D), x_input)

        # Step 4: Encode (no mask input needed - tokens carry the information)
        encoder_output = self.encoder(x_input, mask=None, padding_mask=None)  # (B, T, d_model)

        # Step 5: Decode to reconstruct
        x_recon = self.decoder(encoder_output)  # (B, T, D)

        # Step 6: Compute reconstruction loss (only on MASK_TOKEN positions)
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
                "reconstruction_loss": loss.detach(),  # Alias for consistent naming
                "mae_recon_loss_masked": loss.detach(),
                "mae_recon_loss_visible": visible_loss,
                "mae_mask_ratio_actual": n_masked / n_total,
                "mae_obs_ratio": n_observed / n_total,
                "mae_loss_positions": n_loss_positions / n_total,
            }

        return loss, metrics
