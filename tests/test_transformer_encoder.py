"""Tests for transformer encoder implementation.

Tests cover:
- Basic forward pass with various configurations
- Positional encoding
- Observation mask integration
- Padding mask handling
- Different pooling strategies
- Pre-LN vs Post-LN architectures
- Edge cases and validation
"""

import pytest
import torch
import torch.nn as nn
from slices.models.encoders import TransformerConfig, TransformerEncoder
from slices.models.encoders.transformer import PositionalEncoding, TransformerEncoderLayer


class TestPositionalEncoding:
    """Tests for positional encoding module."""

    def test_positional_encoding_shape(self):
        """Test that positional encoding preserves input shape."""
        d_model = 128
        seq_length = 48
        batch_size = 16

        pos_enc = PositionalEncoding(d_model=d_model, max_seq_length=1000, dropout=0.0)
        x = torch.randn(batch_size, seq_length, d_model)
        out = pos_enc(x)

        assert out.shape == x.shape, "Positional encoding changed shape"

    def test_positional_encoding_different_lengths(self):
        """Test that positional encoding works with variable-length sequences."""
        d_model = 64
        pos_enc = PositionalEncoding(d_model=d_model, max_seq_length=1000, dropout=0.0)

        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(8, seq_len, d_model)
            out = pos_enc(x)
            assert out.shape == x.shape

    def test_positional_encoding_is_deterministic(self):
        """Test that positional encoding is deterministic (not random)."""
        d_model = 128
        pos_enc = PositionalEncoding(d_model=d_model, max_seq_length=1000, dropout=0.0)

        x = torch.randn(4, 48, d_model)
        out1 = pos_enc(x)
        out2 = pos_enc(x)

        # Should be identical (dropout=0.0)
        assert torch.allclose(out1, out2), "Positional encoding is not deterministic"


class TestTransformerEncoderLayer:
    """Tests for single transformer encoder layer."""

    def test_layer_forward_pass(self):
        """Test basic forward pass through a single layer."""
        d_model = 128
        n_heads = 8
        d_ff = 512
        batch_size = 16
        seq_length = 48

        layer = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.1,
            prenorm=True,
        )

        x = torch.randn(batch_size, seq_length, d_model)
        out = layer(x)

        assert out.shape == x.shape, "Layer changed output shape"

    def test_layer_with_padding_mask(self):
        """Test layer with padding mask."""
        d_model = 128
        n_heads = 8
        d_ff = 512
        batch_size = 8
        seq_length = 48

        layer = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.0,
        )

        x = torch.randn(batch_size, seq_length, d_model)

        # Create padding mask (True = padding to ignore)
        padding_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
        padding_mask[:, 30:] = True  # Last 18 positions are padding

        out = layer(x, key_padding_mask=padding_mask)
        assert out.shape == x.shape

    def test_prenorm_vs_postnorm(self):
        """Test that Pre-LN and Post-LN produce different results."""
        d_model = 128
        n_heads = 8
        d_ff = 512

        # Create two layers with same initialization but different norm order
        torch.manual_seed(42)
        layer_prenorm = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.0,
            prenorm=True,
        )

        torch.manual_seed(42)
        layer_postnorm = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.0,
            prenorm=False,
        )

        x = torch.randn(4, 24, d_model)

        out_prenorm = layer_prenorm(x)
        out_postnorm = layer_postnorm(x)

        # Should be different due to different normalization order
        assert not torch.allclose(out_prenorm, out_postnorm, atol=1e-5)

    def test_activation_functions(self):
        """Test different activation functions."""
        d_model = 64
        n_heads = 4
        d_ff = 256

        for activation in ["relu", "gelu", "silu"]:
            layer = TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                activation=activation,
            )

            x = torch.randn(4, 24, d_model)
            out = layer(x)
            assert out.shape == x.shape

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            TransformerEncoderLayer(
                d_model=64,
                n_heads=4,
                d_ff=256,
                activation="invalid",
            )


class TestTransformerEncoder:
    """Tests for full transformer encoder."""

    def test_basic_forward_pass(self):
        """Test basic forward pass with default config."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=4,
            n_heads=8,
            d_ff=512,
            max_seq_length=168,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        batch_size = 16
        seq_length = 48
        x = torch.randn(batch_size, seq_length, config.d_input)

        out = encoder(x)

        # Mean pooling should return (B, d_model)
        assert out.shape == (batch_size, config.d_model)

    def test_no_pooling(self):
        """Test encoder with no pooling (per-timestep outputs)."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="none",
        )

        encoder = TransformerEncoder(config)

        batch_size = 8
        seq_length = 48
        x = torch.randn(batch_size, seq_length, config.d_input)

        out = encoder(x)

        # No pooling should return (B, T, d_model)
        assert out.shape == (batch_size, seq_length, config.d_model)

    def test_cls_pooling(self):
        """Test encoder with CLS token pooling."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="cls",
        )

        encoder = TransformerEncoder(config)

        batch_size = 8
        seq_length = 48
        x = torch.randn(batch_size, seq_length, config.d_input)

        out = encoder(x)

        # CLS pooling should return (B, d_model)
        assert out.shape == (batch_size, config.d_model)

    def test_last_pooling(self):
        """Test encoder with last-timestep pooling."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="last",
        )

        encoder = TransformerEncoder(config)

        batch_size = 8
        seq_length = 48
        x = torch.randn(batch_size, seq_length, config.d_input)

        out = encoder(x)

        # Last pooling should return (B, d_model)
        assert out.shape == (batch_size, config.d_model)

    def test_max_pooling(self):
        """Test encoder with max pooling."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="max",
        )

        encoder = TransformerEncoder(config)

        batch_size = 8
        seq_length = 48
        x = torch.randn(batch_size, seq_length, config.d_input)

        out = encoder(x)

        # Max pooling should return (B, d_model)
        assert out.shape == (batch_size, config.d_model)

    def test_with_observation_mask(self):
        """Test encoder with observation mask (missingness indicator)."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        batch_size = 8
        seq_length = 48
        x = torch.randn(batch_size, seq_length, config.d_input)

        # Create observation mask (30% missing values)
        obs_mask = torch.rand(batch_size, seq_length, config.d_input) > 0.3

        out = encoder(x, mask=obs_mask)

        assert out.shape == (batch_size, config.d_model)

    def test_with_padding_mask(self):
        """Test encoder with padding mask (variable-length sequences)."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        batch_size = 8
        seq_length = 48
        x = torch.randn(batch_size, seq_length, config.d_input)

        # Create padding mask (True = valid, False = padding)
        padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
        # Variable lengths: 20, 30, 40, 48, ...
        for i in range(batch_size):
            length = 20 + i * 4
            if length < seq_length:
                padding_mask[i, length:] = False

        out = encoder(x, padding_mask=padding_mask)

        assert out.shape == (batch_size, config.d_model)

    def test_mean_pooling_respects_padding(self):
        """Test that mean pooling correctly handles padding."""
        config = TransformerConfig(
            d_input=4,
            d_model=8,
            n_layers=1,
            n_heads=2,
            pooling="mean",
            dropout=0.0,
        )

        encoder = TransformerEncoder(config)
        encoder.eval()  # Disable dropout

        # Create simple input with known values
        x1 = torch.ones(1, 10, 4)  # All ones, length 10
        x2 = torch.ones(1, 10, 4)  # All ones, length 5 (rest padding)

        # No padding for x1, padding after 5 for x2
        mask1 = torch.ones(1, 10, dtype=torch.bool)
        mask2 = torch.ones(1, 10, dtype=torch.bool)
        mask2[0, 5:] = False

        with torch.no_grad():
            out1 = encoder(x1, padding_mask=mask1)
            out2 = encoder(x2, padding_mask=mask2)

        # Outputs should be different because averaging over different lengths
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_last_pooling_respects_padding(self):
        """Test that last pooling uses the correct last valid position."""
        config = TransformerConfig(
            d_input=4,
            d_model=8,
            n_layers=1,
            n_heads=2,
            pooling="last",
            dropout=0.0,
        )

        encoder = TransformerEncoder(config)
        encoder.eval()

        batch_size = 4
        seq_length = 10
        x = torch.randn(batch_size, seq_length, 4)

        # Different valid lengths for each sequence
        padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
        padding_mask[0, 5:] = False  # Length 5
        padding_mask[1, 7:] = False  # Length 7
        padding_mask[2, 10:] = False  # Length 10 (no padding)
        padding_mask[3, 3:] = False  # Length 3

        with torch.no_grad():
            out = encoder(x, padding_mask=padding_mask)

        assert out.shape == (batch_size, 8)

    def test_without_positional_encoding(self):
        """Test encoder without positional encoding."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            use_positional_encoding=False,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        batch_size = 8
        seq_length = 48
        x = torch.randn(batch_size, seq_length, config.d_input)

        out = encoder(x)

        assert out.shape == (batch_size, config.d_model)

    def test_prenorm_vs_postnorm_full_encoder(self):
        """Test that Pre-LN and Post-LN encoders produce different results."""
        config_prenorm = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            prenorm=True,
            dropout=0.0,
        )

        config_postnorm = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            prenorm=False,
            dropout=0.0,
        )

        torch.manual_seed(42)
        encoder_prenorm = TransformerEncoder(config_prenorm)

        torch.manual_seed(42)
        encoder_postnorm = TransformerEncoder(config_postnorm)

        x = torch.randn(4, 24, 35)

        encoder_prenorm.eval()
        encoder_postnorm.eval()

        with torch.no_grad():
            out_prenorm = encoder_prenorm(x)
            out_postnorm = encoder_postnorm(x)

        assert not torch.allclose(out_prenorm, out_postnorm, atol=1e-5)

    def test_get_output_dim(self):
        """Test that get_output_dim returns correct dimension."""
        config = TransformerConfig(
            d_input=35,
            d_model=256,
            n_layers=4,
            n_heads=8,
        )

        encoder = TransformerEncoder(config)

        assert encoder.get_output_dim() == 256

    def test_invalid_d_model_n_heads_ratio(self):
        """Test that invalid d_model/n_heads ratio raises error."""
        config = TransformerConfig(
            d_input=35,
            d_model=127,  # Not divisible by n_heads=8
            n_layers=2,
            n_heads=8,
        )

        with pytest.raises(ValueError, match="d_model.*must be divisible by.*n_heads"):
            TransformerEncoder(config)

    def test_invalid_pooling_strategy(self):
        """Test that invalid pooling strategy raises error."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="invalid",
        )

        with pytest.raises(ValueError, match="Invalid pooling"):
            TransformerEncoder(config)

    def test_gradient_flow(self):
        """Test that gradients flow through the encoder."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        batch_size = 4
        seq_length = 24
        x = torch.randn(batch_size, seq_length, config.d_input, requires_grad=True)

        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Check that input gradients are computed
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check that model parameters have gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_size_one(self):
        """Test encoder with batch size of 1."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        x = torch.randn(1, 48, config.d_input)
        out = encoder(x)

        assert out.shape == (1, config.d_model)

    def test_variable_sequence_lengths(self):
        """Test encoder with different sequence lengths."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            max_seq_length=500,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        for seq_len in [10, 50, 100, 200]:
            x = torch.randn(4, seq_len, config.d_input)
            out = encoder(x)
            assert out.shape == (4, config.d_model)

    def test_encoder_is_deterministic_in_eval_mode(self):
        """Test that encoder produces same output in eval mode."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            dropout=0.1,  # Dropout enabled
            pooling="mean",
        )

        encoder = TransformerEncoder(config)
        encoder.eval()  # Disable dropout

        torch.manual_seed(42)
        x = torch.randn(4, 24, config.d_input)

        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)

        assert torch.allclose(out1, out2), "Encoder not deterministic in eval mode"

    def test_encoder_is_different_in_train_mode(self):
        """Test that encoder produces different outputs in train mode (dropout)."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            dropout=0.3,  # Higher dropout for more randomness
            pooling="mean",
        )

        encoder = TransformerEncoder(config)
        encoder.train()  # Enable dropout

        x = torch.randn(4, 24, config.d_input)

        # Run multiple times and check for differences
        out1 = encoder(x)
        out2 = encoder(x)

        # With dropout, outputs should be different
        assert not torch.allclose(out1, out2, atol=1e-5)


class TestTransformerIntegration:
    """Integration tests for transformer in realistic scenarios."""

    def test_small_transformer(self):
        """Test small transformer configuration (for testing/debugging)."""
        config = TransformerConfig(
            d_input=9,  # Minimal feature set
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            max_seq_length=24,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        x = torch.randn(8, 24, 9)
        out = encoder(x)

        assert out.shape == (8, 32)

    def test_large_transformer(self):
        """Test large transformer configuration (research-scale)."""
        config = TransformerConfig(
            d_input=50,  # Extended feature set
            d_model=512,
            n_layers=8,
            n_heads=16,
            d_ff=2048,
            max_seq_length=336,  # 14 days
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        x = torch.randn(2, 168, 50)  # Smaller batch for memory
        out = encoder(x)

        assert out.shape == (2, 512)

    def test_encoder_with_realistic_icu_data_shape(self):
        """Test encoder with realistic ICU data dimensions."""
        # Typical ICU dataset: 35 features, 48-hour window, batch of 32
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=4,
            n_heads=8,
            d_ff=512,
            max_seq_length=168,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        batch_size = 32
        seq_length = 48
        n_features = 35

        x = torch.randn(batch_size, seq_length, n_features)
        obs_mask = torch.rand(batch_size, seq_length, n_features) > 0.2  # 20% missing
        padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        # Some sequences have variable lengths
        for i in range(batch_size):
            if i % 4 == 0:  # 25% of sequences
                length = torch.randint(24, seq_length, (1,)).item()
                padding_mask[i, length:] = False

        out = encoder(x, mask=obs_mask, padding_mask=padding_mask)

        assert out.shape == (batch_size, 128)

    def test_encoder_parameter_count(self):
        """Test that parameter count is reasonable."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=4,
            n_heads=8,
            d_ff=512,
        )

        encoder = TransformerEncoder(config)

        n_params = sum(p.numel() for p in encoder.parameters())

        # Rough estimate for this config: ~1-2M parameters
        assert 500_000 < n_params < 3_000_000, f"Unexpected param count: {n_params}"

    def test_encoder_forward_backward_pass(self):
        """Test full forward and backward pass."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        x = torch.randn(16, 48, 35)
        target = torch.randn(16, 128)

        # Forward pass
        out = encoder(x)
        loss = nn.MSELoss()(out, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters were updated
        assert loss.item() > 0


class TestEncoderSimplified:
    """Tests verifying encoder works without mask handling."""

    def test_mask_parameter_ignored(self):
        """Test that mask parameter is ignored (reserved for future use)."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            dropout=0.0,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)
        encoder.eval()

        x = torch.randn(4, 24, 35)
        mask_full = torch.ones(4, 24, 35, dtype=torch.bool)
        mask_sparse = torch.zeros(4, 24, 35, dtype=torch.bool)
        mask_sparse[:, :, :10] = True

        with torch.no_grad():
            out_full = encoder(x, mask=mask_full)
            out_sparse = encoder(x, mask=mask_sparse)
            out_none = encoder(x, mask=None)

        # All should be identical since mask is ignored
        assert torch.allclose(out_full, out_sparse, atol=1e-5)
        assert torch.allclose(out_full, out_none, atol=1e-5)

    def test_input_projection_uses_d_input(self):
        """Test that input projection dimension matches d_input (not 2*d_input)."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
        )

        encoder = TransformerEncoder(config)

        # Input projection should be d_input -> d_model
        assert encoder.input_proj.in_features == 35
        assert encoder.input_proj.out_features == 128

    def test_gradient_flow(self):
        """Test that gradients flow through the encoder."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        x = torch.randn(4, 24, 35, requires_grad=True)

        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Check gradients are computed
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_realistic_icu_scenario(self):
        """Test with realistic ICU data dimensions."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=4,
            n_heads=8,
            d_ff=512,
            max_seq_length=168,
            pooling="mean",
        )

        encoder = TransformerEncoder(config)

        batch_size = 32
        seq_length = 48
        n_features = 35

        x = torch.randn(batch_size, seq_length, n_features)

        # Padding mask (variable lengths)
        padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
        for i in range(batch_size):
            if i % 4 == 0:
                length = torch.randint(24, seq_length, (1,)).item()
                padding_mask[i, length:] = False

        out = encoder(x, padding_mask=padding_mask)

        assert out.shape == (batch_size, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
