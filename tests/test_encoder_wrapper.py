"""Tests for EncoderWithMissingToken wrapper.

Tests cover:
- Wrapper initialization (with pretrained token, random init, no token)
- Forward pass substitutes missing positions correctly
- Wrapper preserves encoder output interface
- Backward compatibility with old and new checkpoint formats
- Integration with pretrain/finetune modules
"""

import pytest
import torch
import torch.nn as nn
from slices.models.encoders import (
    EncoderWithMissingToken,
    TransformerConfig,
    TransformerEncoder,
)


class TestEncoderWithMissingTokenInit:
    """Tests for wrapper initialization."""

    def test_init_with_pretrained_token(self):
        """Test initialization with pretrained missing token."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        # Create a pretrained token
        pretrained_token = torch.randn(1, 1, 35) * 0.1

        wrapped = EncoderWithMissingToken(
            encoder=encoder,
            d_input=35,
            missing_token=pretrained_token,
        )

        # Check token was loaded
        assert wrapped.missing_token is not None
        assert wrapped.missing_token.shape == (1, 1, 35)
        assert torch.allclose(wrapped.missing_token.data, pretrained_token)

    def test_init_with_random_token(self):
        """Test initialization with random missing token."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(
            encoder=encoder,
            d_input=35,
            missing_token=None,
            init_missing_token=True,
        )

        # Check token was randomly initialized
        assert wrapped.missing_token is not None
        assert wrapped.missing_token.shape == (1, 1, 35)
        # Random initialization should have non-zero values
        assert not torch.allclose(wrapped.missing_token, torch.zeros_like(wrapped.missing_token))

    def test_init_without_token(self):
        """Test initialization without missing token (disabled)."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(
            encoder=encoder,
            d_input=35,
            missing_token=None,
            init_missing_token=False,
        )

        # Check token is None
        assert wrapped.missing_token is None

    def test_init_with_wrong_token_shape_raises_error(self):
        """Test that wrong token shape raises ValueError."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        # Wrong shape
        wrong_token = torch.randn(1, 1, 50)  # Should be 35

        with pytest.raises(ValueError, match="missing_token shape must be"):
            EncoderWithMissingToken(
                encoder=encoder,
                d_input=35,
                missing_token=wrong_token,
            )


class TestEncoderWithMissingTokenForward:
    """Tests for wrapper forward pass."""

    def test_forward_substitutes_missing_positions(self):
        """Test that forward pass substitutes missing positions with token."""
        config = TransformerConfig(
            d_input=4,  # Small for testing
            d_model=16,
            n_layers=1,
            n_heads=2,
            pooling="mean",
            dropout=0.0,
        )
        encoder = TransformerEncoder(config)

        # Create distinctive missing token
        missing_token = torch.ones(1, 1, 4) * 999.0

        wrapped = EncoderWithMissingToken(
            encoder=encoder,
            d_input=4,
            missing_token=missing_token,
        )

        # Create input with some missing positions
        batch_size = 2
        seq_length = 10
        x = torch.randn(batch_size, seq_length, 4)
        mask = torch.ones(batch_size, seq_length, 4, dtype=torch.bool)
        mask[:, 5:8, :2] = False  # Some missing positions

        # Capture input to encoder
        captured_input = []
        original_forward = wrapped.encoder.forward

        def capture_forward(x, **kwargs):
            captured_input.append(x.clone())
            return original_forward(x, **kwargs)

        wrapped.encoder.forward = capture_forward
        wrapped(x, mask=mask)

        # Check that missing positions were substituted
        actual_input = captured_input[0]
        # Missing positions should have value 999.0
        assert torch.allclose(actual_input[:, 5:8, :2], torch.ones(2, 3, 2) * 999.0)
        # Observed positions should keep original values
        assert torch.allclose(actual_input[:, :5, :], x[:, :5, :])

    def test_forward_without_mask(self):
        """Test forward pass without mask (no substitution)."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
            dropout=0.0,
        )
        encoder = TransformerEncoder(config)
        encoder.eval()

        wrapped = EncoderWithMissingToken(
            encoder=encoder,
            d_input=35,
            missing_token=torch.ones(1, 1, 35) * 999.0,
        )
        wrapped.eval()

        x = torch.randn(4, 24, 35)

        # Forward without mask
        with torch.no_grad():
            out_wrapped = wrapped(x, mask=None)
            out_original = encoder(x, mask=None)

        # Should be identical when no mask is provided
        assert torch.allclose(out_wrapped, out_original)

    def test_forward_without_token_enabled(self):
        """Test forward pass when token is disabled."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
            dropout=0.0,
        )
        encoder = TransformerEncoder(config)
        encoder.eval()

        wrapped = EncoderWithMissingToken(
            encoder=encoder,
            d_input=35,
            missing_token=None,
            init_missing_token=False,
        )
        wrapped.eval()

        x = torch.randn(4, 24, 35)
        mask = torch.ones(4, 24, 35, dtype=torch.bool)
        mask[:, :10, :] = False  # Some missing

        with torch.no_grad():
            out_wrapped = wrapped(x, mask=mask)
            out_original = encoder(x, mask=mask)

        # Should be identical when token is disabled
        assert torch.allclose(out_wrapped, out_original)


class TestEncoderWithMissingTokenInterface:
    """Tests that wrapper preserves encoder interface."""

    def test_get_output_dim(self):
        """Test that get_output_dim returns wrapped encoder's dimension."""
        config = TransformerConfig(
            d_input=35,
            d_model=256,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(
            encoder=encoder,
            d_input=35,
        )

        assert wrapped.get_output_dim() == 256
        assert wrapped.get_output_dim() == encoder.get_output_dim()

    def test_config_property(self):
        """Test that config property returns wrapped encoder's config."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=4,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(
            encoder=encoder,
            d_input=35,
        )

        assert wrapped.config is encoder.config
        assert wrapped.config.d_model == 128
        assert wrapped.config.n_layers == 4

    def test_output_shape_mean_pooling(self):
        """Test output shape with mean pooling."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(encoder=encoder, d_input=35)

        x = torch.randn(8, 48, 35)
        out = wrapped(x)

        assert out.shape == (8, 128)

    def test_output_shape_no_pooling(self):
        """Test output shape without pooling."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="none",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(encoder=encoder, d_input=35)

        x = torch.randn(8, 48, 35)
        out = wrapped(x)

        assert out.shape == (8, 48, 128)


class TestEncoderWithMissingTokenGradients:
    """Tests for gradient flow through wrapper."""

    def test_gradient_flow_through_missing_token(self):
        """Test that gradients flow through the missing token."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(encoder=encoder, d_input=35)

        x = torch.randn(4, 24, 35, requires_grad=True)
        mask = torch.ones(4, 24, 35, dtype=torch.bool)
        mask[:, :10, :] = False  # Some missing positions

        out = wrapped(x, mask=mask)
        loss = out.sum()
        loss.backward()

        # Check gradients for missing token
        assert wrapped.missing_token.grad is not None
        assert not torch.allclose(
            wrapped.missing_token.grad, torch.zeros_like(wrapped.missing_token.grad)
        )

    def test_gradient_flow_to_encoder(self):
        """Test that gradients flow to encoder parameters."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(encoder=encoder, d_input=35)

        x = torch.randn(4, 24, 35, requires_grad=True)
        mask = torch.ones(4, 24, 35, dtype=torch.bool)
        mask[:, :10, :] = False

        out = wrapped(x, mask=mask)
        loss = out.sum()
        loss.backward()

        # Check gradients for encoder parameters
        for name, param in wrapped.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestEncoderWithMissingTokenCheckpoint:
    """Tests for checkpoint saving and loading."""

    def test_state_dict_includes_missing_token(self):
        """Test that state_dict includes missing_token."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(encoder=encoder, d_input=35)

        state_dict = wrapped.state_dict()

        assert "missing_token" in state_dict
        assert state_dict["missing_token"].shape == (1, 1, 35)

    def test_load_state_dict_restores_missing_token(self):
        """Test that load_state_dict restores missing_token."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder1 = TransformerEncoder(config)
        encoder2 = TransformerEncoder(config)

        wrapped1 = EncoderWithMissingToken(encoder=encoder1, d_input=35)
        wrapped2 = EncoderWithMissingToken(encoder=encoder2, d_input=35)

        # Modify wrapped1's token
        with torch.no_grad():
            wrapped1.missing_token.fill_(42.0)

        # Save and load
        state_dict = wrapped1.state_dict()
        wrapped2.load_state_dict(state_dict)

        # Check token was restored
        assert torch.allclose(wrapped2.missing_token.data, wrapped1.missing_token.data)

    def test_round_trip_save_load(self, tmp_path):
        """Test saving and loading wrapper checkpoint."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(encoder=encoder, d_input=35)

        # Save
        checkpoint_path = tmp_path / "wrapper.pt"
        torch.save(wrapped.state_dict(), checkpoint_path)

        # Load into new wrapper
        encoder2 = TransformerEncoder(config)
        wrapped2 = EncoderWithMissingToken(encoder=encoder2, d_input=35)
        wrapped2.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Compare outputs
        wrapped.eval()
        wrapped2.eval()
        x = torch.randn(4, 24, 35)

        with torch.no_grad():
            out1 = wrapped(x)
            out2 = wrapped2(x)

        assert torch.allclose(out1, out2)


class TestEncoderWithMissingTokenIntegration:
    """Integration tests with realistic scenarios."""

    def test_with_realistic_icu_data(self):
        """Test wrapper with realistic ICU data dimensions."""
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

        wrapped = EncoderWithMissingToken(encoder=encoder, d_input=35)

        batch_size = 32
        seq_length = 48
        n_features = 35

        x = torch.randn(batch_size, seq_length, n_features)
        # 30% missing values
        obs_mask = torch.rand(batch_size, seq_length, n_features) > 0.3

        out = wrapped(x, mask=obs_mask)

        assert out.shape == (batch_size, 128)

    def test_training_step(self):
        """Test wrapper in a training loop."""
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        wrapped = EncoderWithMissingToken(encoder=encoder, d_input=35)
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-3)

        # Training step
        x = torch.randn(16, 48, 35)
        mask = torch.rand(16, 48, 35) > 0.3
        target = torch.randn(16, 128)

        optimizer.zero_grad()
        out = wrapped(x, mask=mask)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
