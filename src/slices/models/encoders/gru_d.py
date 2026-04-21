"""GRU-D encoder for ICU time-series data with learnable decay.

Implements GRU-D (Che et al. 2018) which handles missing data natively through
input decay (toward empirical mean) and hidden state decay. Compatible with the
existing z-normalized, zero-filled data pipeline where empirical mean = 0.

This encoder does NOT use EncoderWithMissingToken — missingness is handled
intrinsically via the decay mechanism.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseEncoder, EncoderConfig


@dataclass
class GRUDConfig(EncoderConfig):
    """Configuration for GRU-D encoder."""

    n_gru_layers: int = 1


class GRUDEncoder(BaseEncoder):
    """GRU-D encoder with learnable input and hidden state decay.

    Input at each timestep is concat(x_decayed, obs_mask) of size 2*d_input.
    Output is the final hidden state of shape (B, d_model).

    Args:
        config: GRU-D configuration.

    Example:
        >>> config = GRUDConfig(d_input=35, d_model=64, n_gru_layers=1)
        >>> encoder = GRUDEncoder(config)
        >>> x = torch.randn(4, 48, 35)
        >>> mask = torch.rand(4, 48, 35) > 0.3
        >>> out = encoder(x, mask=mask)  # (4, 64)
    """

    def __init__(self, config: GRUDConfig) -> None:
        super().__init__(config)
        self.config: GRUDConfig = config

        d_input = config.d_input
        d_model = config.d_model

        # GRU cell: input is concat(x_decayed, mask) = 2 * d_input
        self.gru_cell = nn.GRUCell(input_size=2 * d_input, hidden_size=d_model)

        # Input decay parameters
        self.W_gamma_x = nn.Parameter(torch.Tensor(d_input))
        self.b_gamma_x = nn.Parameter(torch.Tensor(d_input))

        # Hidden state decay parameters: feature-wise deltas -> hidden units.
        self.W_gamma_h = nn.Parameter(torch.Tensor(d_model, d_input))
        self.b_gamma_h = nn.Parameter(torch.Tensor(d_model))

        # Empirical mean in z-space is 0 (after z-normalization)
        self.register_buffer("x_mean", torch.zeros(d_input))

        self.dropout = nn.Dropout(config.dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.W_gamma_x, -0.1, 0.1)
        nn.init.zeros_(self.b_gamma_x)
        nn.init.uniform_(self.W_gamma_h, -0.1, 0.1)
        nn.init.zeros_(self.b_gamma_h)

    def _compute_input_decay(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Compute feature-wise input decay gamma_x for one timestep."""
        return torch.exp(-torch.relu(self.W_gamma_x * delta_t + self.b_gamma_x))

    def _compute_hidden_decay(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Compute hidden-state decay gamma_h from feature-wise deltas."""
        return torch.exp(-torch.relu(delta_t @ self.W_gamma_h.t() + self.b_gamma_h))

    def _impute_inputs(
        self,
        x_t: torch.Tensor,
        mask_t: torch.Tensor,
        gamma_x: torch.Tensor,
        x_last_observed: torch.Tensor,
    ) -> torch.Tensor:
        """Apply GRU-D input decay toward the empirical mean for missing values."""
        mask_float = mask_t.float()
        return mask_float * x_t + (1 - mask_float) * (
            gamma_x * x_last_observed + (1 - gamma_x) * self.x_mean
        )

    def _update_last_observed(
        self,
        x_t: torch.Tensor,
        mask_t: torch.Tensor,
        x_last_observed: torch.Tensor,
    ) -> torch.Tensor:
        """Update x_last only with actually observed raw values."""
        return torch.where(mask_t, x_t, x_last_observed)

    def _compute_time_deltas(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute time since last observation per feature.

        Args:
            mask: Boolean observation mask (B, T, D). True = observed.

        Returns:
            Time deltas (B, T, D) in float. delta[t=0] = 0.
            For t > 0: delta[t] = 1 if observed at t-1, else delta[t-1] + 1.
        """
        B, T, D = mask.shape
        delta = torch.zeros(B, T, D, device=mask.device)
        for t in range(1, T):
            delta[:, t, :] = torch.where(
                mask[:, t - 1, :],
                torch.ones(B, D, device=mask.device),
                delta[:, t - 1, :] + 1,
            )
        return delta

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with learnable decay for missing values.

        Args:
            x: Input tensor (B, T, D).
            mask: Observation mask (B, T, D), True = observed.
                  If None, all values treated as observed.
            padding_mask: Unused, kept for API compatibility.

        Returns:
            Final hidden state (B, d_model).
        """
        B, T, D = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones(B, T, D, dtype=torch.bool, device=device)

        mask_float = mask.float()
        delta = self._compute_time_deltas(mask)

        h = torch.zeros(B, self.config.d_model, device=device)
        x_last_observed = self.x_mean.expand(B, -1).clone()

        for t in range(T):
            # Input decay
            gamma_x = self._compute_input_decay(delta[:, t, :])
            x_decayed = self._impute_inputs(
                x[:, t, :],
                mask[:, t, :],
                gamma_x,
                x_last_observed,
            )

            # Hidden state decay
            gamma_h = self._compute_hidden_decay(delta[:, t, :])
            h = gamma_h * h

            # GRU update
            gru_input = torch.cat([x_decayed, mask_float[:, t, :]], dim=-1)
            h = self.gru_cell(gru_input, h)

            x_last_observed = self._update_last_observed(
                x[:, t, :],
                mask[:, t, :],
                x_last_observed,
            )

        h = self.dropout(h)
        return h

    def get_output_dim(self) -> int:
        return self.config.d_model
