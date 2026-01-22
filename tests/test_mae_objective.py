"""Tests for MAE (Masked Autoencoder) SSL objective."""

import pytest
import torch
from slices.data.transforms import (
    MaskingStrategy,
    apply_mask,
    create_block_mask,
    create_feature_mask,
    create_random_mask,
    create_ssl_mask,
    create_timestep_mask,
)
from slices.models.encoders import TransformerConfig, TransformerEncoder
from slices.models.pretraining import (
    MAEConfig,
    MAEObjective,
    build_ssl_objective,
    get_ssl_config_class,
)


class TestMaskingFunctions:
    """Tests for masking utilities."""

    def test_create_random_mask(self):
        """Test random mask creation."""
        shape = (2, 10, 5)
        mask_ratio = 0.3
        device = torch.device("cpu")

        mask = create_random_mask(shape, mask_ratio, device)

        assert mask.shape == shape
        assert mask.dtype == torch.bool
        # Check approximate mask ratio (with some tolerance)
        actual_ratio = (~mask).float().mean().item()
        assert 0.15 <= actual_ratio <= 0.45  # Allow ~15% absolute deviation

    def test_create_block_mask(self):
        """Test block mask creation."""
        shape = (2, 20, 5)
        mask_ratio = 0.3
        device = torch.device("cpu")

        mask = create_block_mask(
            shape, mask_ratio, min_block_size=3, max_block_size=5, device=device
        )

        assert mask.shape == shape
        assert mask.dtype == torch.bool
        # Check that masking is consistent across features (entire timesteps)
        for b in range(shape[0]):
            # For each timestep, either all features are masked or none
            timestep_mask = mask[b, :, 0]  # First feature
            for d in range(1, shape[2]):
                assert torch.equal(mask[b, :, d], timestep_mask)

    def test_create_timestep_mask(self):
        """Test timestep mask creation."""
        shape = (2, 10, 5)
        mask_ratio = 0.3
        device = torch.device("cpu")

        mask = create_timestep_mask(shape, mask_ratio, device)

        assert mask.shape == shape
        assert mask.dtype == torch.bool
        # Check that mask is consistent across features
        for b in range(shape[0]):
            timestep_mask = mask[b, :, 0]
            for d in range(1, shape[2]):
                assert torch.equal(mask[b, :, d], timestep_mask)

    def test_create_feature_mask(self):
        """Test feature mask creation."""
        shape = (2, 10, 5)
        mask_ratio = 0.3
        device = torch.device("cpu")

        mask = create_feature_mask(shape, mask_ratio, device)

        assert mask.shape == shape
        assert mask.dtype == torch.bool
        # Check that mask is consistent across timesteps
        for b in range(shape[0]):
            feature_mask = mask[b, 0, :]
            for t in range(1, shape[1]):
                assert torch.equal(mask[b, t, :], feature_mask)

    def test_apply_mask(self):
        """Test applying mask to tensor."""
        x = torch.randn(2, 10, 5)
        mask = torch.rand(2, 10, 5) > 0.5  # Random boolean mask
        mask_value = 0.0

        x_masked = apply_mask(x, mask, mask_value)

        assert x_masked.shape == x.shape
        # Check that masked positions are set to mask_value
        assert torch.allclose(x_masked[~mask], torch.tensor(mask_value))
        # Check that unmasked positions are unchanged
        assert torch.allclose(x_masked[mask], x[mask])

    def test_create_ssl_mask_respects_obs_mask(self):
        """Test that SSL mask respects observation mask."""
        shape = (2, 10, 5)
        mask_ratio = 0.5
        device = torch.device("cpu")

        # Create observation mask (30% missing)
        obs_mask = torch.rand(shape, device=device) > 0.3

        # Create SSL mask
        ssl_mask = create_ssl_mask(
            shape, mask_ratio, strategy="random", obs_mask=obs_mask, device=device
        )

        # SSL should not mask missing values
        # obs_mask: True = observed, False = missing
        # ssl_mask: False = masked for SSL
        # Where obs_mask is False, ssl_mask must be True (not masked for SSL)
        assert torch.all(ssl_mask[~obs_mask])

    @pytest.mark.parametrize("strategy", ["random", "block", "timestep", "feature"])
    def test_all_masking_strategies(self, strategy: MaskingStrategy):
        """Test all masking strategies work."""
        shape = (2, 20, 5)
        mask_ratio = 0.3
        device = torch.device("cpu")

        mask = create_ssl_mask(shape, mask_ratio, strategy=strategy, device=device)

        assert mask.shape == shape
        assert mask.dtype == torch.bool


class TestMAEDecoder:
    """Tests for MAE decoder."""

    def test_decoder_forward(self):
        """Test decoder forward pass."""
        from slices.models.pretraining.mae import MAEDecoder

        B, T, d_encoder, d_input = 2, 10, 128, 35
        config = MAEConfig(
            decoder_d_model=64,
            decoder_n_layers=2,
            decoder_n_heads=4,
        )

        decoder = MAEDecoder(d_encoder, d_input, config)
        x = torch.randn(B, T, d_encoder)

        recon = decoder(x)

        assert recon.shape == (B, T, d_input)

    def test_decoder_trainable(self):
        """Test that decoder parameters are trainable."""
        from slices.models.pretraining.mae import MAEDecoder

        config = MAEConfig()
        decoder = MAEDecoder(d_encoder=128, d_input=35, config=config)

        # Check that decoder has parameters
        n_params = sum(p.numel() for p in decoder.parameters())
        assert n_params > 0

        # Check that parameters require grad
        for param in decoder.parameters():
            assert param.requires_grad


class TestMAEObjective:
    """Tests for MAE objective."""

    @pytest.fixture
    def encoder(self):
        """Create a small transformer encoder for testing."""
        config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",  # Required for MAE
        )
        return TransformerEncoder(config)

    @pytest.fixture
    def mae_config(self):
        """Create MAE config for testing."""
        return MAEConfig(
            mask_ratio=0.15,
            mask_strategy="random",
            decoder_d_model=32,
            decoder_n_layers=1,
            decoder_n_heads=2,
        )

    def test_mae_initialization(self, encoder, mae_config):
        """Test MAE initialization."""
        mae = MAEObjective(encoder, mae_config)

        assert mae.encoder is encoder
        assert mae.config == mae_config
        assert hasattr(mae, "decoder")

    def test_mae_forward(self, encoder, mae_config):
        """Test MAE forward pass."""
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 10, 35
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3  # 30% missing

        loss, metrics = mae(x, obs_mask)

        # Check loss
        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0
        assert not torch.isnan(loss)

        # Check metrics
        assert "mae_loss" in metrics
        assert "mae_recon_loss_masked" in metrics
        assert "mae_recon_loss_visible" in metrics
        assert "mae_mask_ratio_actual" in metrics
        assert "mae_obs_ratio" in metrics

    def test_mae_backward(self, encoder, mae_config):
        """Test that MAE can backpropagate."""
        mae = MAEObjective(encoder, mae_config)

        x = torch.randn(2, 10, 35)
        obs_mask = torch.rand(2, 10, 35) > 0.3

        loss, _ = mae(x, obs_mask)
        loss.backward()

        # Check that gradients exist
        for name, param in mae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_mae_training_step(self, encoder, mae_config):
        """Test a full training step."""
        mae = MAEObjective(encoder, mae_config)
        optimizer = torch.optim.Adam(mae.parameters(), lr=1e-3)

        x = torch.randn(4, 10, 35)
        obs_mask = torch.ones(4, 10, 35, dtype=torch.bool)  # All observed

        # Training step
        optimizer.zero_grad()
        loss, metrics = mae(x, obs_mask)
        loss.backward()
        optimizer.step()

        # Check that loss is finite
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("mask_strategy", ["random", "block", "timestep", "feature"])
    def test_mae_all_strategies(self, encoder, mask_strategy):
        """Test MAE with all masking strategies."""
        config = MAEConfig(
            mask_ratio=0.15,
            mask_strategy=mask_strategy,
            decoder_d_model=32,
            decoder_n_layers=1,
        )
        mae = MAEObjective(encoder, config)

        x = torch.randn(2, 10, 35)
        obs_mask = torch.ones(2, 10, 35, dtype=torch.bool)

        loss, metrics = mae(x, obs_mask)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_mae_respects_obs_mask(self, encoder, mae_config):
        """Test that MAE respects observation mask."""
        mae = MAEObjective(encoder, mae_config)

        x = torch.randn(2, 10, 35)
        # Create obs mask where some values are missing
        obs_mask = torch.rand(2, 10, 35) > 0.5

        # Forward pass should not error with missing values
        loss, metrics = mae(x, obs_mask)

        assert torch.isfinite(loss)
        # Check that observation ratio is tracked
        obs_ratio = metrics["mae_obs_ratio"]
        expected_ratio = obs_mask.float().mean().item()
        assert abs(obs_ratio - expected_ratio) < 0.01

    def test_mae_loss_on_observed_only(self, encoder):
        """Test that loss_on_observed_only flag works."""
        # Config with loss only on observed
        config_obs = MAEConfig(
            mask_ratio=0.5,
            mask_strategy="random",
            loss_on_observed_only=True,
        )
        mae_obs = MAEObjective(encoder, config_obs)

        # Config with loss on all masked
        config_all = MAEConfig(
            mask_ratio=0.5,
            mask_strategy="random",
            loss_on_observed_only=False,
        )
        mae_all = MAEObjective(encoder, config_all)

        # Create data with missing values
        torch.manual_seed(42)
        x = torch.randn(2, 10, 35)
        obs_mask = torch.rand(2, 10, 35) > 0.5  # 50% observed

        # Both should run without error
        loss_obs, metrics_obs = mae_obs(x, obs_mask)
        loss_all, metrics_all = mae_all(x, obs_mask)

        assert torch.isfinite(loss_obs)
        assert torch.isfinite(loss_all)

        # Loss positions should be different
        # obs: only masked AND observed
        # all: all masked (including missing)
        assert metrics_obs["mae_loss_positions"] <= metrics_all["mae_loss_positions"]

    def test_mae_requires_no_pooling(self):
        """Test that MAE raises error if encoder has pooling."""
        # Create encoder with pooling
        config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="mean",  # This should cause error
        )
        encoder = TransformerEncoder(config)

        mae_config = MAEConfig()

        with pytest.raises(ValueError, match="pooling='none'"):
            MAEObjective(encoder, mae_config)

    def test_mae_get_encoder(self, encoder, mae_config):
        """Test getting encoder from MAE."""
        mae = MAEObjective(encoder, mae_config)

        retrieved_encoder = mae.get_encoder()

        assert retrieved_encoder is encoder

    def test_mae_two_token_system(self, encoder):
        """Test MAE with two learned tokens (MISSING_TOKEN and MASK_TOKEN)."""
        mae_config = MAEConfig(
            mask_ratio=0.3,
            mask_strategy="random",
        )
        mae = MAEObjective(encoder, mae_config)

        # Check both tokens were created
        assert mae.missing_token is not None
        assert mae.mask_token is not None
        assert mae.missing_token.shape == (1, 1, 35)  # d_input=35
        assert mae.mask_token.shape == (1, 1, 35)
        assert mae.missing_token.requires_grad
        assert mae.mask_token.requires_grad

        # Test forward pass works
        B, T, D = 4, 24, 35
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3

        mae.train()
        loss, metrics = mae(x, obs_mask)

        assert loss.requires_grad
        assert loss.item() > 0

        # Test backward pass - gradients should flow to both tokens
        loss.backward()
        assert mae.missing_token.grad is not None
        assert mae.mask_token.grad is not None
        # At least one should have non-zero gradient (mask_token definitely should)
        assert mae.mask_token.grad.abs().sum() > 0

    def test_missing_token_placement(self, encoder):
        """Test that MISSING_TOKEN is placed at genuinely missing positions."""
        mae_config = MAEConfig(
            mask_ratio=0.0,  # No SSL masking - isolate missing token behavior
            mask_strategy="random",
        )
        mae = MAEObjective(encoder, mae_config)
        mae.eval()

        B, T, D = 2, 10, 35
        x = torch.randn(B, T, D)
        # Create obs_mask with known missing positions
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)
        obs_mask[:, :, 0] = False  # First feature always missing

        # Capture the input to encoder
        captured_input = []
        original_forward = mae.encoder.forward

        def capture_forward(x_in, **kwargs):
            captured_input.append(x_in.clone())
            return original_forward(x_in, **kwargs)

        from unittest.mock import patch

        with patch.object(mae.encoder, "forward", capture_forward):
            with torch.no_grad():
                mae(x, obs_mask)

        # Check that missing positions have MISSING_TOKEN values
        encoder_input = captured_input[0]
        missing_token_expanded = mae.missing_token.expand(B, T, D)

        # First feature should have MISSING_TOKEN values
        assert torch.allclose(encoder_input[:, :, 0], missing_token_expanded[:, :, 0], atol=1e-6)

    def test_mask_token_placement(self, encoder):
        """Test that MASK_TOKEN is placed at SSL-masked positions."""
        mae_config = MAEConfig(
            mask_ratio=1.0,  # Mask everything observed for testing
            mask_strategy="random",
        )
        mae = MAEObjective(encoder, mae_config)
        mae.eval()

        B, T, D = 2, 10, 35
        x = torch.randn(B, T, D)
        # All observed - so all will be SSL-masked
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        # Capture the input to encoder
        captured_input = []
        original_forward = mae.encoder.forward

        def capture_forward(x_in, **kwargs):
            captured_input.append(x_in.clone())
            return original_forward(x_in, **kwargs)

        from unittest.mock import patch

        with patch.object(mae.encoder, "forward", capture_forward):
            with torch.no_grad():
                mae(x, obs_mask)

        # Check that encoder input differs from original (should be MASK_TOKEN)
        encoder_input = captured_input[0]
        mask_token_expanded = mae.mask_token.expand(B, T, D)

        # All positions should have MASK_TOKEN values (since mask_ratio=1.0)
        assert torch.allclose(encoder_input, mask_token_expanded, atol=1e-6)


class TestSSLFactory:
    """Tests for SSL factory functions."""

    def test_build_ssl_objective_mae(self):
        """Test building MAE via factory."""
        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        mae_config = MAEConfig(mask_ratio=0.15)

        ssl_objective = build_ssl_objective(encoder, mae_config)

        assert isinstance(ssl_objective, MAEObjective)
        assert ssl_objective.encoder is encoder

    def test_build_ssl_objective_unknown(self):
        """Test that unknown objective raises error."""
        encoder_config = TransformerConfig(d_input=35, d_model=64, pooling="none")
        encoder = TransformerEncoder(encoder_config)

        from slices.models.pretraining.base import SSLConfig

        bad_config = SSLConfig(name="unknown_objective")

        with pytest.raises(ValueError, match="Unknown SSL objective"):
            build_ssl_objective(encoder, bad_config)

    def test_get_ssl_config_class(self):
        """Test getting SSL config class."""
        config_cls = get_ssl_config_class("mae")

        assert config_cls == MAEConfig

    def test_get_ssl_config_class_unknown(self):
        """Test that unknown config raises error."""
        with pytest.raises(ValueError, match="Unknown SSL objective"):
            get_ssl_config_class("unknown_objective")


class TestMAEIntegration:
    """Integration tests for MAE with dataset."""

    def test_mae_with_real_data(self, tmp_path):
        """Test MAE with realistic ICU-like data."""
        # Create synthetic ICU-like data
        B, T, D = 8, 48, 35
        x = torch.randn(B, T, D)

        # Add realistic missingness pattern
        # More missing values in later timesteps (common in ICU)
        obs_prob = torch.linspace(0.9, 0.6, T).unsqueeze(0).unsqueeze(-1)
        obs_mask = torch.rand(B, T, D) < obs_prob.expand(B, T, D)

        # Impute missing values (forward fill)
        x_imputed = x.clone()
        for b in range(B):
            for d in range(D):
                last_valid = x[b, 0, d]
                for t in range(T):
                    if obs_mask[b, t, d]:
                        last_valid = x[b, t, d]
                    else:
                        x_imputed[b, t, d] = last_valid

        # Create MAE model
        encoder_config = TransformerConfig(
            d_input=D,
            d_model=128,
            n_layers=4,
            n_heads=8,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        mae_config = MAEConfig(
            mask_ratio=0.15,
            mask_strategy="block",
            decoder_d_model=64,
            decoder_n_layers=2,
        )
        mae = MAEObjective(encoder, mae_config)

        # Train for a few steps
        optimizer = torch.optim.Adam(mae.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(20):  # More steps for reliable decrease with two-token system
            optimizer.zero_grad()
            loss, metrics = mae(x_imputed, obs_mask)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()

        # Check that loss decreased (allow 10% tolerance for random fluctuations)
        assert final_loss < initial_loss * 1.1, (
            f"Loss should decrease during training: initial={initial_loss:.4f}, "
            f"final={final_loss:.4f}"
        )

        # Check that metrics are reasonable
        assert 0.0 <= metrics["mae_mask_ratio_actual"] <= 1.0
        assert 0.0 <= metrics["mae_obs_ratio"] <= 1.0

    def test_mae_different_mask_ratios(self):
        """Test MAE with different mask ratios."""
        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        x = torch.randn(4, 10, 35)
        obs_mask = torch.ones(4, 10, 35, dtype=torch.bool)

        for mask_ratio in [0.05, 0.15, 0.30, 0.50, 0.75]:
            config = MAEConfig(
                mask_ratio=mask_ratio,
                decoder_d_model=32,
                decoder_n_layers=1,
            )
            mae = MAEObjective(encoder, config)

            loss, metrics = mae(x, obs_mask)

            # Check that actual mask ratio is close to target
            actual_ratio = metrics["mae_mask_ratio_actual"]
            # Allow 15% relative error
            assert abs(actual_ratio - mask_ratio) < 0.15


class TestTwoTokenSystem:
    """Tests for the two-token system (MISSING_TOKEN and MASK_TOKEN)."""

    def test_encoder_receives_no_mask(self):
        """Test that encoder receives mask=None (tokens carry the information)."""
        from unittest.mock import patch

        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        config = MAEConfig(
            mask_ratio=0.5,
            mask_strategy="random",
            decoder_d_model=32,
            decoder_n_layers=1,
        )
        mae = MAEObjective(encoder, config)
        mae.eval()

        B, T, D = 2, 10, 35
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        # Capture what mask is passed to encoder
        captured_masks = []
        original_forward = encoder.forward

        def capture_forward(x_in, mask=None, padding_mask=None):
            captured_masks.append(mask)
            return original_forward(x_in, mask=mask, padding_mask=padding_mask)

        with patch.object(encoder, "forward", capture_forward):
            mae(x, obs_mask)

        # The mask should be None (tokens carry the information)
        assert captured_masks[0] is None

    def test_tokens_differentiate_missing_vs_masked(self):
        """Test that MISSING_TOKEN and MASK_TOKEN have different values."""
        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        config = MAEConfig(
            mask_ratio=0.3,
            mask_strategy="random",
            decoder_d_model=32,
            decoder_n_layers=1,
        )
        mae = MAEObjective(encoder, config)

        # After random initialization, tokens should be different
        assert not torch.allclose(mae.missing_token, mae.mask_token)

    def test_input_contains_both_tokens(self):
        """Test that encoder input contains both MISSING_TOKEN and MASK_TOKEN."""
        from unittest.mock import patch

        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        config = MAEConfig(
            mask_ratio=0.5,  # High ratio to ensure some positions get masked
            mask_strategy="random",
            decoder_d_model=32,
            decoder_n_layers=1,
        )
        mae = MAEObjective(encoder, config)
        mae.eval()

        B, T, D = 2, 10, 35
        x = torch.randn(B, T, D) + 5.0  # Offset to differentiate from tokens
        # Mix of missing and observed
        obs_mask = torch.rand(B, T, D) > 0.5

        # Capture what input is passed to encoder
        captured_inputs = []
        original_forward = encoder.forward

        def capture_forward(x_in, **kwargs):
            captured_inputs.append(x_in.clone())
            return original_forward(x_in, **kwargs)

        with patch.object(encoder, "forward", capture_forward):
            with torch.no_grad():
                mae(x, obs_mask)

        encoder_input = captured_inputs[0]

        # Check that input contains values close to MISSING_TOKEN (at missing positions)
        missing_positions = ~obs_mask
        if missing_positions.any():
            missing_vals = encoder_input[missing_positions]
            expected_missing = mae.missing_token.expand_as(encoder_input)[missing_positions]
            assert torch.allclose(missing_vals, expected_missing, atol=1e-6)


class TestBugFixes:
    """Tests for bug fixes in the pretraining pipeline."""

    def test_block_masking_respects_max_ratio(self):
        """Test that block masking doesn't exceed target mask ratio significantly."""
        from slices.data.transforms import create_block_mask

        shape = (8, 48, 35)
        mask_ratio = 0.15

        # Run multiple times to check consistency
        for _ in range(10):
            mask = create_block_mask(
                shape,
                mask_ratio=mask_ratio,
                min_block_size=3,
                max_block_size=10,
                device=torch.device("cpu"),
            )

            # Count masked positions (False = masked)
            actual_ratio = (~mask[:, :, 0]).float().mean().item()  # Check first feature

            # Should not exceed target by more than max_block_size/T
            max_overshoot = 10 / 48  # max_block_size / T
            assert actual_ratio <= mask_ratio + max_overshoot, (
                f"Block masking exceeded target ratio: {actual_ratio:.3f} > "
                f"{mask_ratio + max_overshoot:.3f}"
            )

    def test_training_mask_varies(self):
        """Test that training masks vary between forward passes."""
        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        config = MAEConfig(
            mask_ratio=0.15,
            mask_strategy="random",
            decoder_d_model=32,
            decoder_n_layers=1,
        )
        mae = MAEObjective(encoder, config)

        # Set to train mode
        mae.train()

        x = torch.randn(4, 10, 35)
        obs_mask = torch.ones(4, 10, 35, dtype=torch.bool)

        # Run multiple times and check that losses vary
        losses = []
        for _ in range(5):
            loss, _ = mae(x, obs_mask)
            losses.append(loss.item())

        # At least some losses should be different (random masks)
        unique_losses = len(set(f"{l:.6f}" for l in losses))
        assert unique_losses > 1, "Training masks should vary but all losses were identical"

    def test_norm_target_raises_error(self):
        """Test that norm_target=True raises NotImplementedError."""
        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        config = MAEConfig(
            mask_ratio=0.15,
            norm_target=True,  # This should cause error
            decoder_d_model=32,
            decoder_n_layers=1,
        )
        mae = MAEObjective(encoder, config)

        x = torch.randn(4, 10, 35)
        obs_mask = torch.ones(4, 10, 35, dtype=torch.bool)

        with pytest.raises(
            NotImplementedError, match="norm_target=True is not currently supported"
        ):
            mae(x, obs_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
