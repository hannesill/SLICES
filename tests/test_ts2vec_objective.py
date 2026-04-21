"""Tests for TS2Vec-style temporal contrastive objective."""

import pytest
import torch
from slices.models.encoders import TransformerConfig, TransformerEncoder
from slices.models.pretraining import TS2VecConfig, TS2VecObjective


@pytest.fixture
def encoder():
    config = TransformerConfig(
        d_input=10,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        pooling="none",
        obs_aware=True,
        max_seq_length=48,
    )
    return TransformerEncoder(config)


def test_empty_timesteps_are_excluded_from_ts2vec_masks(encoder):
    """TS2Vec should only count observed timesteps as eligible SSL tokens."""
    config = TS2VecConfig(
        mask_ratio=0.5,
        noise_scale=0.0,
        crop_ratio=1.0,
        proj_hidden_dim=32,
        proj_output_dim=16,
        n_hierarchical_scales=2,
    )
    objective = TS2VecObjective(encoder, config)

    B, T, D = 2, 8, 10
    x = torch.randn(B, T, D)
    obs_mask = torch.zeros(B, T, D, dtype=torch.bool)
    obs_mask[:, 1, :3] = True
    obs_mask[:, 4, :3] = True
    obs_mask[:, 7, :3] = True

    _, metrics = objective(x, obs_mask)

    assert metrics["ts2vec_n_visible_view1"].item() <= 3
    assert metrics["ts2vec_n_visible_view2"].item() <= 3
    assert metrics["ts2vec_n_overlap_per_sample"].item() <= 3


def test_ts2vec_scatter_to_full_ignores_padded_visible_tokens():
    """Uneven visible counts should not scatter padded tokens into masked slots."""
    encoded = torch.tensor(
        [
            [[10.0], [11.0]],
            [[20.0], [99.0]],
        ]
    )
    ssl_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )

    full = TS2VecObjective._scatter_to_full(encoded, ssl_mask, n_timesteps=3)

    assert torch.allclose(full[1, 0], torch.tensor([20.0]))
    assert torch.allclose(full[1, 1], torch.tensor([0.0]))
    assert torch.allclose(full[1, 2], torch.tensor([0.0]))
