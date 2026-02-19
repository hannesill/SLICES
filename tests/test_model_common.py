"""Tests for shared model utilities in slices.models.common.

Tests cover:
- apply_pooling: none, mean, max, last, cls pooling with and without masks
- PositionalEncoding: 3D/4D input, determinism, distinct positions
- get_activation: known activations, unknown raises error
- Integration: TransformerEncoder/LinearEncoder use shared pooling,
  TransformerEncoderLayer and MLPTaskHead use shared activation
"""

import pytest
import torch
import torch.nn as nn
from slices.models.common import PositionalEncoding, apply_pooling, get_activation
from slices.models.encoders import (
    LinearConfig,
    LinearEncoder,
    TransformerConfig,
    TransformerEncoder,
)
from slices.models.heads.base import TaskHeadConfig
from slices.models.heads.mlp import MLPTaskHead


class TestApplyPooling:
    """Tests for apply_pooling utility."""

    def test_none_pooling_preserves_shape(self):
        x = torch.randn(4, 10, 32)
        out = apply_pooling(x, "none")
        assert out.shape == (4, 10, 32)
        assert torch.equal(out, x)

    def test_mean_pooling_without_mask(self):
        x = torch.ones(2, 5, 8)
        out = apply_pooling(x, "mean")
        assert out.shape == (2, 8)
        assert torch.allclose(out, torch.ones(2, 8))

    def test_mean_pooling_with_mask(self):
        x = torch.ones(1, 4, 2)
        x[0, 2:] = 999.0  # padding values
        mask = torch.tensor([[True, True, False, False]])
        out = apply_pooling(x, "mean", padding_mask=mask)
        assert out.shape == (1, 2)
        assert torch.allclose(out, torch.ones(1, 2))

    def test_max_pooling_without_mask(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        out = apply_pooling(x, "max")
        assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))

    def test_max_pooling_with_mask(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
        mask = torch.tensor([[True, True, False]])
        out = apply_pooling(x, "max", padding_mask=mask)
        assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))

    def test_last_pooling_without_mask(self):
        x = torch.tensor([[[1.0], [2.0], [3.0]]])
        out = apply_pooling(x, "last")
        assert torch.allclose(out, torch.tensor([[3.0]]))

    def test_last_pooling_with_mask(self):
        x = torch.tensor([[[1.0], [2.0], [99.0]]])
        mask = torch.tensor([[True, True, False]])
        out = apply_pooling(x, "last", padding_mask=mask)
        assert torch.allclose(out, torch.tensor([[2.0]]))

    def test_cls_pooling(self):
        x = torch.tensor([[[10.0, 20.0], [1.0, 2.0]]])
        out = apply_pooling(x, "cls")
        assert torch.allclose(out, torch.tensor([[10.0, 20.0]]))

    def test_invalid_pooling_raises(self):
        x = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="Unknown pooling"):
            apply_pooling(x, "invalid")

    def test_transformer_uses_shared_pooling(self):
        """Verify TransformerEncoder._apply_pooling delegates to shared utility."""
        config = TransformerConfig(d_input=4, d_model=8, n_layers=1, n_heads=2, pooling="mean")
        encoder = TransformerEncoder(config)
        encoder.eval()
        x = torch.randn(2, 5, 4)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 8)

    def test_linear_uses_shared_pooling(self):
        """Verify LinearEncoder._apply_pooling delegates to shared utility."""
        config = LinearConfig(d_input=4, d_model=8, pooling="max")
        encoder = LinearEncoder(config)
        encoder.eval()
        x = torch.randn(2, 5, 4)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 8)


class TestPositionalEncodingShared:
    """Tests for shared PositionalEncoding module."""

    def test_3d_input(self):
        pe = PositionalEncoding(d_model=16, max_seq_length=100, dropout=0.0)
        x = torch.zeros(2, 10, 16)
        out = pe(x)
        assert out.shape == (2, 10, 16)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_4d_input(self):
        pe = PositionalEncoding(d_model=16, max_seq_length=100, dropout=0.0)
        x = torch.zeros(2, 5, 10, 16)
        out = pe(x)
        assert out.shape == (2, 5, 10, 16)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_deterministic(self):
        pe = PositionalEncoding(d_model=32, max_seq_length=100, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)

    def test_different_positions_get_different_encodings(self):
        pe = PositionalEncoding(d_model=32, max_seq_length=100, dropout=0.0)
        x = torch.zeros(1, 50, 32)
        out = pe(x)
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestGetActivation:
    """Tests for get_activation utility."""

    def test_known_activations(self):
        for name in ["relu", "gelu", "silu", "tanh"]:
            act = get_activation(name)
            assert isinstance(act, nn.Module)
            x = torch.randn(4)
            out = act(x)
            assert out.shape == (4,)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("leaky_relu_not_supported")

    def test_activation_used_in_transformer_layer(self):
        """TransformerEncoderLayer uses shared get_activation."""
        from slices.models.encoders.transformer import TransformerEncoderLayer

        for act_name in ["relu", "gelu", "silu"]:
            layer = TransformerEncoderLayer(
                d_model=16,
                n_heads=2,
                d_ff=32,
                activation=act_name,
            )
            x = torch.randn(2, 5, 16)
            out = layer(x)
            assert out.shape == (2, 5, 16)

    def test_activation_used_in_mlp_head(self):
        """MLPTaskHead uses shared get_activation."""
        config = TaskHeadConfig(
            name="mlp",
            task_name="test",
            task_type="binary",
            input_dim=16,
            hidden_dims=[8],
            activation="gelu",
        )
        head = MLPTaskHead(config)
        x = torch.randn(2, 16)
        out = head(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 2)
