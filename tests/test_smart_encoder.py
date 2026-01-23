"""Tests for SMART (MART) encoder implementation.

Tests cover:
- MLPEmbedder: Joint (value, mask) embedding
- PositionalEncoding: Sinusoidal temporal encoding
- SeqAttentionBlock: Temporal attention with observation mask bias
- VarAttentionBlock: Cross-variable attention with query-based pooling
- SMARTEncoder: Full encoder with all components
- Pooling strategies: query, mean, none
- Edge cases and validation
"""

import pytest
import torch
import torch.nn as nn
from slices.models.encoders.smart import (
    BasicBlock,
    MLPBlock,
    MLPEmbedder,
    PositionalEncoding,
    SeqAttentionBlock,
    SMARTEncoder,
    SMARTEncoderConfig,
    VarAttentionBlock,
)


class TestMLPEmbedder:
    """Tests for MLPEmbedder module."""

    def test_mlp_embedder_output_shape(self):
        """Test that MLPEmbedder produces correct output shape."""
        d_model = 32
        embedder = MLPEmbedder(d_model)

        B, V, T = 4, 35, 48
        x = torch.randn(B, V, T)
        mask = torch.rand(B, V, T) > 0.3

        out = embedder(x, mask)

        assert out.shape == (B, V, T, d_model)

    def test_mlp_embedder_different_d_model(self):
        """Test MLPEmbedder with different d_model values."""
        for d_model in [16, 32, 64, 128]:
            embedder = MLPEmbedder(d_model)
            x = torch.randn(2, 10, 24)
            mask = torch.ones(2, 10, 24, dtype=torch.bool)

            out = embedder(x, mask)
            assert out.shape == (2, 10, 24, d_model)

    def test_mlp_embedder_gradient_flow(self):
        """Test that gradients flow through MLPEmbedder."""
        embedder = MLPEmbedder(d_model=32)
        x = torch.randn(2, 10, 24, requires_grad=True)
        mask = torch.ones(2, 10, 24, dtype=torch.bool)

        out = embedder(x, mask)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_mlp_embedder_mask_affects_output(self):
        """Test that different masks produce different outputs."""
        embedder = MLPEmbedder(d_model=32)
        embedder.eval()

        x = torch.randn(2, 10, 24)
        mask_all = torch.ones(2, 10, 24, dtype=torch.bool)
        mask_none = torch.zeros(2, 10, 24, dtype=torch.bool)

        with torch.no_grad():
            out_all = embedder(x, mask_all)
            out_none = embedder(x, mask_none)

        # Different masks should produce different outputs
        assert not torch.allclose(out_all, out_none)


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_positional_encoding_4d_shape(self):
        """Test that positional encoding preserves 4D input shape."""
        d_model = 32
        pos_enc = PositionalEncoding(d_model=d_model, max_seq_length=100, dropout=0.0)

        B, V, T = 4, 35, 48
        x = torch.randn(B, V, T, d_model)
        out = pos_enc(x)

        assert out.shape == x.shape

    def test_positional_encoding_3d_shape(self):
        """Test that positional encoding preserves 3D input shape."""
        d_model = 32
        pos_enc = PositionalEncoding(d_model=d_model, max_seq_length=100, dropout=0.0)

        BV, T = 140, 48  # B*V
        x = torch.randn(BV, T, d_model)
        out = pos_enc(x)

        assert out.shape == x.shape

    def test_positional_encoding_is_deterministic(self):
        """Test that positional encoding is deterministic (dropout=0)."""
        d_model = 32
        pos_enc = PositionalEncoding(d_model=d_model, max_seq_length=100, dropout=0.0)

        x = torch.randn(4, 35, 48, d_model)
        out1 = pos_enc(x)
        out2 = pos_enc(x)

        assert torch.allclose(out1, out2)

    def test_positional_encoding_different_lengths(self):
        """Test positional encoding with variable-length sequences."""
        d_model = 32
        pos_enc = PositionalEncoding(d_model=d_model, max_seq_length=500, dropout=0.0)

        for T in [10, 50, 100, 200]:
            x = torch.randn(4, 35, T, d_model)
            out = pos_enc(x)
            assert out.shape == x.shape


class TestSeqAttentionBlock:
    """Tests for SeqAttentionBlock (temporal attention)."""

    def test_seq_attention_output_shape(self):
        """Test that SeqAttention preserves input shape."""
        d_model = 32
        n_heads = 4
        block = SeqAttentionBlock(d_model, n_heads, dropout=0.0)

        B, V, T_plus_1 = 4, 35, 49  # T+1 includes query token
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)
        obs_mask = torch.ones(B, V, T, dtype=torch.bool)

        out = block(x, obs_mask)

        assert out.shape == x.shape

    def test_seq_attention_with_sparse_obs_mask(self):
        """Test SeqAttention with sparse observation mask."""
        d_model = 32
        n_heads = 4
        block = SeqAttentionBlock(d_model, n_heads, dropout=0.0)

        B, V, T_plus_1 = 4, 35, 49
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)
        obs_mask = torch.rand(B, V, T) > 0.5  # 50% observed

        out = block(x, obs_mask)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_seq_attention_with_padding_mask(self):
        """Test SeqAttention with padding mask."""
        d_model = 32
        n_heads = 4
        block = SeqAttentionBlock(d_model, n_heads, dropout=0.0)

        B, V, T_plus_1 = 4, 35, 49
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)
        obs_mask = torch.ones(B, V, T, dtype=torch.bool)
        padding_mask = torch.ones(B, T, dtype=torch.bool)
        padding_mask[:, 30:] = False  # Padding after position 30

        out = block(x, obs_mask, padding_mask)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_seq_attention_additive_mask_bias(self):
        """Test that different obs_masks produce different attention patterns."""
        d_model = 32
        n_heads = 4
        block = SeqAttentionBlock(d_model, n_heads, dropout=0.0)
        block.eval()

        B, V, T_plus_1 = 2, 10, 25
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)

        # All observed vs half observed
        obs_mask_all = torch.ones(B, V, T, dtype=torch.bool)
        obs_mask_half = torch.ones(B, V, T, dtype=torch.bool)
        obs_mask_half[:, :, ::2] = False  # Every other position missing

        with torch.no_grad():
            out_all = block(x, obs_mask_all)
            out_half = block(x, obs_mask_half)

        # Outputs should differ due to different attention biases
        assert not torch.allclose(out_all, out_half, atol=1e-5)


class TestVarAttentionBlock:
    """Tests for VarAttentionBlock (cross-variable attention)."""

    def test_var_attention_output_shape(self):
        """Test that VarAttention preserves input shape."""
        d_model = 32
        n_heads = 4
        block = VarAttentionBlock(d_model, n_heads, dropout=0.0)

        B, V, T_plus_1 = 4, 35, 49
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)
        obs_mask = torch.ones(B, V, T, dtype=torch.bool)

        out = block(x, obs_mask)

        assert out.shape == x.shape

    def test_var_attention_with_sparse_obs_mask(self):
        """Test VarAttention with sparse observation mask."""
        d_model = 32
        n_heads = 4
        block = VarAttentionBlock(d_model, n_heads, dropout=0.0)

        B, V, T_plus_1 = 4, 35, 49
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)
        obs_mask = torch.rand(B, V, T) > 0.5

        out = block(x, obs_mask)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_var_attention_query_based_pooling(self):
        """Test that VarAttention uses query tokens for cross-variable attention."""
        d_model = 32
        n_heads = 4
        block = VarAttentionBlock(d_model, n_heads, dropout=0.0)
        block.eval()

        B, V, T_plus_1 = 2, 10, 25
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)
        obs_mask = torch.ones(B, V, T, dtype=torch.bool)

        with torch.no_grad():
            out = block(x, obs_mask)

        # Output should have same shape but different values (attention applied)
        assert not torch.allclose(x, out, atol=1e-5)


class TestMLPBlock:
    """Tests for MLPBlock (feedforward)."""

    def test_mlp_block_output_shape(self):
        """Test that MLPBlock preserves input shape."""
        d_model = 32
        d_ff = 128
        block = MLPBlock(d_model, d_ff, dropout=0.0)

        x = torch.randn(4, 35, 48, d_model)
        out = block(x)

        assert out.shape == x.shape

    def test_mlp_block_gradient_flow(self):
        """Test that gradients flow through MLPBlock."""
        d_model = 32
        d_ff = 128
        block = MLPBlock(d_model, d_ff, dropout=0.0)

        x = torch.randn(2, 10, 24, d_model, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestBasicBlock:
    """Tests for BasicBlock (SeqAtt + VarAtt + MLP)."""

    def test_basic_block_output_shape(self):
        """Test that BasicBlock preserves input shape."""
        d_model = 32
        n_heads = 4
        d_ff = 128
        block = BasicBlock(d_model, n_heads, d_ff, dropout=0.0)

        B, V, T_plus_1 = 4, 35, 49
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)
        obs_mask = torch.ones(B, V, T, dtype=torch.bool)

        out = block(x, obs_mask)

        assert out.shape == x.shape

    def test_basic_block_with_padding_mask(self):
        """Test BasicBlock with padding mask."""
        d_model = 32
        n_heads = 4
        d_ff = 128
        block = BasicBlock(d_model, n_heads, d_ff, dropout=0.0)

        B, V, T_plus_1 = 4, 35, 49
        T = T_plus_1 - 1
        x = torch.randn(B, V, T_plus_1, d_model)
        obs_mask = torch.ones(B, V, T, dtype=torch.bool)
        padding_mask = torch.ones(B, T, dtype=torch.bool)
        padding_mask[:, 30:] = False

        out = block(x, obs_mask, padding_mask)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestSMARTEncoderInitialization:
    """Tests for SMART encoder initialization."""

    def test_smart_encoder_init_default(self):
        """Test SMART encoder initialization with default config."""
        config = SMARTEncoderConfig(d_input=35, d_model=32)
        encoder = SMARTEncoder(config)

        assert encoder.config == config
        assert hasattr(encoder, "embedder")
        assert hasattr(encoder, "query")
        assert hasattr(encoder, "pos_encoder")
        assert hasattr(encoder, "blocks")
        assert hasattr(encoder, "final_norm")

    def test_smart_encoder_query_token_shape(self):
        """Test that query tokens have correct shape."""
        config = SMARTEncoderConfig(d_input=35, d_model=32)
        encoder = SMARTEncoder(config)

        # Query tokens: one per variable
        assert encoder.query.shape == (35, 1, 32)  # (V, 1, d_model)

    def test_smart_encoder_invalid_d_model_n_heads(self):
        """Test that invalid d_model/n_heads ratio raises error."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=31,  # Not divisible by n_heads=4
            n_heads=4,
        )

        with pytest.raises(ValueError, match="d_model.*must be divisible by.*n_heads"):
            SMARTEncoder(config)

    def test_smart_encoder_invalid_pooling(self):
        """Test that invalid pooling strategy raises error."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            pooling="invalid",
        )

        with pytest.raises(ValueError, match="Invalid pooling"):
            SMARTEncoder(config)

    def test_smart_encoder_valid_pooling_options(self):
        """Test all valid pooling options."""
        for pooling in ["query", "mean", "none"]:
            config = SMARTEncoderConfig(
                d_input=35,
                d_model=32,
                pooling=pooling,
            )
            encoder = SMARTEncoder(config)
            assert encoder.config.pooling == pooling


class TestSMARTEncoderForward:
    """Tests for SMART encoder forward pass."""

    def test_basic_forward_pass(self):
        """Test basic forward pass with SLICES format input."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        B, T, D = 4, 48, 35
        x = torch.randn(B, T, D)

        out = encoder(x)

        # Query pooling: (B, V*d_model)
        assert out.shape == (B, 35 * 32)

    def test_forward_with_observation_mask(self):
        """Test forward pass with observation mask."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        B, T, D = 4, 48, 35
        x = torch.randn(B, T, D)
        mask = torch.rand(B, T, D) > 0.3  # 30% missing

        out = encoder(x, mask=mask)

        assert out.shape == (B, 35 * 32)
        assert not torch.isnan(out).any()

    def test_forward_with_padding_mask(self):
        """Test forward pass with padding mask."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        B, T, D = 4, 48, 35
        x = torch.randn(B, T, D)
        padding_mask = torch.ones(B, T, dtype=torch.bool)
        padding_mask[:, 30:] = False

        out = encoder(x, padding_mask=padding_mask)

        assert out.shape == (B, 35 * 32)
        assert not torch.isnan(out).any()

    def test_forward_creates_default_mask(self):
        """Test that forward creates all-ones mask if none provided."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",
        )
        encoder = SMARTEncoder(config)
        encoder.eval()

        B, T, D = 4, 48, 35
        x = torch.randn(B, T, D)

        # Should work without mask
        out = encoder(x)
        assert out.shape == (B, 35 * 32)


class TestSMARTEncoderPooling:
    """Tests for SMART encoder pooling strategies."""

    def test_pooling_none(self):
        """Test encoder with no pooling (for SSL)."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = SMARTEncoder(config)

        B, T, D = 4, 48, 35
        x = torch.randn(B, T, D)

        out = encoder(x)

        # No pooling: (B, V, T, d_model)
        assert out.shape == (B, 35, T, 32)

    def test_pooling_query(self):
        """Test encoder with query token pooling."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        B, T, D = 4, 48, 35
        x = torch.randn(B, T, D)

        out = encoder(x)

        # Query pooling: (B, V*d_model)
        assert out.shape == (B, 35 * 32)

    def test_pooling_mean(self):
        """Test encoder with mean pooling."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="mean",
        )
        encoder = SMARTEncoder(config)

        B, T, D = 4, 48, 35
        x = torch.randn(B, T, D)

        out = encoder(x)

        # Mean pooling: (B, d_model)
        assert out.shape == (B, 32)

    def test_mean_pooling_respects_obs_mask(self):
        """Test that mean pooling respects observation mask."""
        config = SMARTEncoderConfig(
            d_input=4,
            d_model=8,
            n_layers=1,
            n_heads=2,
            pooling="mean",
            dropout=0.0,
        )
        encoder = SMARTEncoder(config)
        encoder.eval()

        B, T, D = 2, 10, 4
        x = torch.randn(B, T, D)

        # Different observation masks
        mask_all = torch.ones(B, T, D, dtype=torch.bool)
        mask_partial = torch.ones(B, T, D, dtype=torch.bool)
        mask_partial[:, :, 0] = False  # First feature always missing

        with torch.no_grad():
            out_all = encoder(x, mask=mask_all)
            out_partial = encoder(x, mask=mask_partial)

        # Outputs should differ due to different masking
        assert not torch.allclose(out_all, out_partial, atol=1e-5)


class TestSMARTEncoderOutputDim:
    """Tests for SMART encoder output dimension calculation."""

    def test_get_output_dim_query_pooling(self):
        """Test output dimension with query pooling."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        assert encoder.get_output_dim() == 35 * 32

    def test_get_output_dim_mean_pooling(self):
        """Test output dimension with mean pooling."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            pooling="mean",
        )
        encoder = SMARTEncoder(config)

        assert encoder.get_output_dim() == 32

    def test_get_output_dim_no_pooling(self):
        """Test output dimension with no pooling."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            pooling="none",
        )
        encoder = SMARTEncoder(config)

        # No pooling returns d_model (per position)
        assert encoder.get_output_dim() == 32


class TestSMARTEncoderGradients:
    """Tests for gradient flow through SMART encoder."""

    def test_gradient_flow(self):
        """Test that gradients flow through the encoder."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="mean",
        )
        encoder = SMARTEncoder(config)

        B, T, D = 4, 24, 35
        x = torch.randn(B, T, D, requires_grad=True)

        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Check that input gradients are computed
        assert x.grad is not None
        # Gradients may be very small due to mean pooling, but should be non-zero
        assert x.grad.abs().sum() > 0, "Input gradients should be non-zero"

        # Check that model parameters have gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_query_token_gradients(self):
        """Test that query token receives gradients."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        x = torch.randn(4, 24, 35)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Query tokens should have gradients
        assert encoder.query.grad is not None
        assert encoder.query.grad.abs().sum() > 0


class TestSMARTEncoderEdgeCases:
    """Tests for edge cases."""

    def test_batch_size_one(self):
        """Test encoder with batch size of 1."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        x = torch.randn(1, 48, 35)
        out = encoder(x)

        assert out.shape == (1, 35 * 32)

    def test_variable_sequence_lengths(self):
        """Test encoder with different sequence lengths."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_length=200,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        for T in [10, 50, 100, 150]:
            x = torch.randn(4, T, 35)
            out = encoder(x)
            assert out.shape == (4, 35 * 32)

    def test_different_d_input_values(self):
        """Test encoder with different number of input variables."""
        for d_input in [9, 35, 50, 100]:
            config = SMARTEncoderConfig(
                d_input=d_input,
                d_model=32,
                n_layers=2,
                n_heads=4,
                pooling="query",
            )
            encoder = SMARTEncoder(config)

            x = torch.randn(4, 48, d_input)
            out = encoder(x)
            assert out.shape == (4, d_input * 32)

    def test_deterministic_in_eval_mode(self):
        """Test that encoder is deterministic in eval mode."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            dropout=0.1,
            pooling="query",
        )
        encoder = SMARTEncoder(config)
        encoder.eval()

        x = torch.randn(4, 24, 35)

        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)

        assert torch.allclose(out1, out2)

    def test_different_in_train_mode(self):
        """Test that encoder produces different outputs in train mode (dropout)."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            dropout=0.3,
            pooling="query",
        )
        encoder = SMARTEncoder(config)
        encoder.train()

        x = torch.randn(4, 24, 35)

        out1 = encoder(x)
        out2 = encoder(x)

        # With dropout, outputs should be different
        assert not torch.allclose(out1, out2, atol=1e-5)


class TestSMARTEncoderIntegration:
    """Integration tests for SMART encoder."""

    def test_with_realistic_icu_data_shape(self):
        """Test encoder with realistic ICU data dimensions."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_seq_length=168,
            pooling="query",
        )
        encoder = SMARTEncoder(config)

        batch_size = 32
        seq_length = 48
        n_features = 35

        x = torch.randn(batch_size, seq_length, n_features)
        obs_mask = torch.rand(batch_size, seq_length, n_features) > 0.2  # 20% missing
        padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

        # Some sequences have variable lengths
        for i in range(batch_size):
            if i % 4 == 0:
                length = torch.randint(24, seq_length, (1,)).item()
                padding_mask[i, length:] = False

        out = encoder(x, mask=obs_mask, padding_mask=padding_mask)

        assert out.shape == (batch_size, 35 * 32)
        assert not torch.isnan(out).any()

    def test_forward_backward_pass(self):
        """Test full forward and backward pass."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",
        )
        encoder = SMARTEncoder(config)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        x = torch.randn(16, 48, 35)
        target = torch.randn(16, 35 * 32)

        # Forward pass
        out = encoder(x)
        loss = nn.MSELoss()(out, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_encoder_parameter_count(self):
        """Test that parameter count is reasonable."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=256,
        )
        encoder = SMARTEncoder(config)

        n_params = sum(p.numel() for p in encoder.parameters())

        # SMART is smaller than transformer, expect ~100K-500K params
        assert 50_000 < n_params < 1_000_000, f"Unexpected param count: {n_params}"

    def test_ssl_output_shape_for_pretraining(self):
        """Test output shape when pooling=none for SSL pretraining."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = SMARTEncoder(config)

        B, T, D = 8, 48, 35
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3

        out = encoder(x, mask=obs_mask)

        # Should output (B, V, T, d_model) for SSL
        assert out.shape == (B, D, T, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
