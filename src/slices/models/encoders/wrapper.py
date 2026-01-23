"""Encoder wrapper for consistent missing token handling.

This module provides a wrapper that applies MISSING_TOKEN substitution at
positions where obs_mask=False, ensuring consistency between pretraining
and finetuning stages.

During MAE pretraining, the encoder sees MISSING_TOKEN at positions where
obs_mask=False. Without this wrapper, finetuning would use forward-filled
imputed values instead, creating a mismatch that may degrade performance.
"""

from typing import Optional

import torch
import torch.nn as nn


class EncoderWithMissingToken(nn.Module):
    """Wraps encoder and substitutes MISSING_TOKEN at obs_mask=False positions.

    This wrapper ensures consistent input preprocessing between pretraining
    and finetuning. During MAE pretraining, the encoder sees MISSING_TOKEN
    at positions where obs_mask=False. This wrapper applies the same
    substitution during finetuning.

    The missing_token parameter can be:
    - Loaded from a pretrain checkpoint (recommended)
    - Randomly initialized for supervised baselines
    - Disabled (use_missing_token=False) for ablation studies

    Example:
        >>> encoder = TransformerEncoder(config)
        >>> wrapped = EncoderWithMissingToken(
        ...     encoder=encoder,
        ...     d_input=35,
        ...     missing_token=pretrain_missing_token,
        ... )
        >>> output = wrapped(timeseries, mask=obs_mask)

    Attributes:
        encoder: The wrapped encoder module.
        missing_token: Learned token for missing positions, shape (1, 1, D).
        use_missing_token: Whether to apply token substitution.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_input: int,
        missing_token: Optional[torch.Tensor] = None,
        init_missing_token: bool = True,
    ) -> None:
        """Initialize the encoder wrapper.

        Args:
            encoder: The encoder module to wrap.
            d_input: Input feature dimension (must match encoder's d_input).
            missing_token: Optional pretrained missing token tensor of shape
                (1, 1, D). If None and init_missing_token=True, a new token
                is randomly initialized.
            init_missing_token: Whether to initialize a random missing_token
                if none is provided. If False and missing_token is None,
                no token substitution will occur.
        """
        super().__init__()
        self.encoder = encoder
        self.d_input = d_input

        # Initialize or store missing_token
        if missing_token is not None:
            # Use provided token (from pretrain checkpoint)
            if missing_token.shape != (1, 1, d_input):
                raise ValueError(
                    f"missing_token shape must be (1, 1, {d_input}), " f"got {missing_token.shape}"
                )
            self.missing_token = nn.Parameter(missing_token.clone())
        elif init_missing_token:
            # Initialize random token (for supervised baselines)
            self.missing_token = nn.Parameter(torch.zeros(1, 1, d_input))
            nn.init.normal_(self.missing_token, std=0.02)
        else:
            # No token substitution
            self.register_parameter("missing_token", None)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with missing token substitution.

        Args:
            x: Input tensor of shape (B, T, D).
            mask: Optional observation mask of shape (B, T, D) where True
                indicates observed values and False indicates missing/imputed.
                If provided and missing_token is set, missing positions are
                replaced with the missing token.
            padding_mask: Optional padding mask of shape (B, T) where True
                indicates valid timesteps.

        Returns:
            Encoder output tensor.
        """
        # Apply missing token substitution if enabled
        if self.missing_token is not None and mask is not None:
            B, T, D = x.shape
            # Replace missing positions (mask=False) with missing_token
            x = torch.where(mask, x, self.missing_token.expand(B, T, D))

        # Forward through wrapped encoder
        return self.encoder(x, mask=mask, padding_mask=padding_mask)

    def get_output_dim(self) -> int:
        """Return the output dimension of the wrapped encoder.

        Returns:
            Output dimension from the wrapped encoder.
        """
        return self.encoder.get_output_dim()

    @property
    def config(self):
        """Access the wrapped encoder's config."""
        return self.encoder.config
