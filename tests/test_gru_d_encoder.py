"""Tests for canonical GRU-D decay equations."""

import math

import torch
from slices.models.encoders.gru_d import GRUDConfig, GRUDEncoder


def test_gru_d_hidden_decay_uses_feature_wise_deltas():
    encoder = GRUDEncoder(GRUDConfig(d_input=2, d_model=2))
    with torch.no_grad():
        encoder.W_gamma_h.copy_(torch.eye(2))
        encoder.b_gamma_h.zero_()

    delta_t = torch.tensor([[1.0, 3.0]])

    gamma_h = encoder._compute_hidden_decay(delta_t)

    torch.testing.assert_close(gamma_h, torch.tensor([[math.exp(-1.0), math.exp(-3.0)]]))


def test_gru_d_updates_last_observed_only_from_raw_observations():
    encoder = GRUDEncoder(GRUDConfig(d_input=2, d_model=2))
    with torch.no_grad():
        encoder.x_mean.zero_()

    x_last = torch.tensor([[2.0, 10.0]])
    x_t = torch.tensor([[99.0, 20.0]])
    mask_t = torch.tensor([[False, True]])
    gamma_x = torch.tensor([[0.5, 0.5]])

    x_decayed = encoder._impute_inputs(x_t, mask_t, gamma_x, x_last)
    updated = encoder._update_last_observed(x_t, mask_t, x_last)

    torch.testing.assert_close(x_decayed, torch.tensor([[1.0, 20.0]]))
    torch.testing.assert_close(updated, torch.tensor([[2.0, 20.0]]))
