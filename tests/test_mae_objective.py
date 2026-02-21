"""Tests for observation-level MAE (Masked Autoencoder) SSL objective."""

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
from slices.models.encoders import (
    ObservationTransformerConfig,
    ObservationTransformerEncoder,
    TransformerConfig,
    TransformerEncoder,
)
from slices.models.pretraining import (
    MAEConfig,
    MAEObjective,
    build_ssl_objective,
    get_ssl_config_class,
)

# =============================================================================
# Legacy masking function tests (still used by transforms.py)
# =============================================================================


class TestMaskingFunctions:
    """Tests for masking utilities (still in transforms.py for other uses)."""

    def test_create_random_mask(self):
        shape = (2, 10, 5)
        mask = create_random_mask(shape, 0.3, torch.device("cpu"))
        assert mask.shape == shape
        assert mask.dtype == torch.bool
        actual_ratio = (~mask).float().mean().item()
        assert 0.15 <= actual_ratio <= 0.45

    def test_create_block_mask(self):
        shape = (2, 20, 5)
        mask = create_block_mask(
            shape, 0.3, min_block_size=3, max_block_size=5, device=torch.device("cpu")
        )
        assert mask.shape == shape
        for b in range(shape[0]):
            timestep_mask = mask[b, :, 0]
            for d in range(1, shape[2]):
                assert torch.equal(mask[b, :, d], timestep_mask)

    def test_create_timestep_mask(self):
        shape = (2, 10, 5)
        mask = create_timestep_mask(shape, 0.3, torch.device("cpu"))
        assert mask.shape == shape
        for b in range(shape[0]):
            timestep_mask = mask[b, :, 0]
            for d in range(1, shape[2]):
                assert torch.equal(mask[b, :, d], timestep_mask)

    def test_create_feature_mask(self):
        shape = (2, 10, 5)
        mask = create_feature_mask(shape, 0.3, torch.device("cpu"))
        assert mask.shape == shape
        for b in range(shape[0]):
            feature_mask = mask[b, 0, :]
            for t in range(1, shape[1]):
                assert torch.equal(mask[b, t, :], feature_mask)

    def test_apply_mask(self):
        x = torch.randn(2, 10, 5)
        mask = torch.rand(2, 10, 5) > 0.5
        x_masked = apply_mask(x, mask, 0.0)
        assert torch.allclose(x_masked[~mask], torch.tensor(0.0))
        assert torch.allclose(x_masked[mask], x[mask])

    def test_create_ssl_mask_respects_obs_mask(self):
        shape = (2, 10, 5)
        obs_mask = torch.rand(shape) > 0.3
        ssl_mask = create_ssl_mask(
            shape, 0.5, strategy="random", obs_mask=obs_mask, device=torch.device("cpu")
        )
        assert torch.all(ssl_mask[~obs_mask])

    @pytest.mark.parametrize("strategy", ["random", "block", "timestep", "feature"])
    def test_all_masking_strategies(self, strategy: MaskingStrategy):
        shape = (2, 20, 5)
        mask = create_ssl_mask(shape, 0.3, strategy=strategy, device=torch.device("cpu"))
        assert mask.shape == shape
        assert mask.dtype == torch.bool


# =============================================================================
# Observation tokenization tests
# =============================================================================


class TestObservationTokenization:
    """Tests for observation-level tokenization."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    def test_token_count_equals_observed(self, encoder):
        """Number of valid tokens should equal number of observed values."""
        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.5

        tokens, padding_mask, token_info = encoder.tokenize(x, obs_mask)

        for b in range(B):
            expected_n = obs_mask[b].sum().item()
            actual_n = padding_mask[b].sum().item()
            assert actual_n == expected_n

    def test_token_values_match_input(self, encoder):
        """Token values should correspond to actual observed input values."""
        B, T, D = 1, 4, 3
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)
        obs_mask[0, 2, 1] = False  # One missing value

        _, _, token_info = encoder.tokenize(x, obs_mask)
        n_obs = token_info["n_obs"][0].item()

        # All observed values should be in token_info["values"]
        observed_vals = x[obs_mask].sort()[0]
        token_vals = token_info["values"][0, :n_obs].sort()[0]
        assert torch.allclose(observed_vals, token_vals)

    def test_empty_obs_mask_gives_at_least_one_token(self, encoder):
        """Even with no observations, we get at least 1 token (padded)."""
        B, T, D = 1, 4, 3
        x = torch.randn(B, T, D)
        obs_mask = torch.zeros(B, T, D, dtype=torch.bool)

        tokens, padding_mask, token_info = encoder.tokenize(x, obs_mask)

        assert tokens.shape[1] >= 1
        assert padding_mask.shape[1] >= 1
        # No valid tokens
        assert padding_mask.sum().item() == 0

    def test_all_observed_token_count(self, encoder):
        """With all observed, tokens = T * D."""
        B, T, D = 1, 4, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        _, padding_mask, token_info = encoder.tokenize(x, obs_mask)

        assert padding_mask.sum().item() == T * D

    def test_feature_and_timestep_indices_valid(self, encoder):
        """Feature and timestep indices should be within valid ranges."""
        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3

        _, padding_mask, token_info = encoder.tokenize(x, obs_mask)

        for b in range(B):
            n = padding_mask[b].sum().item()
            assert token_info["timestep_idx"][b, :n].max() < T
            assert token_info["feature_idx"][b, :n].max() < D
            assert token_info["timestep_idx"][b, :n].min() >= 0
            assert token_info["feature_idx"][b, :n].min() >= 0


# =============================================================================
# MAE Decoder tests
# =============================================================================


class TestMAEDecoder:
    """Tests for MAE decoder."""

    def test_decoder_output_shape(self):
        from slices.models.pretraining.mae import MAEDecoder

        config = MAEConfig(
            decoder_d_model=32, decoder_n_layers=1, decoder_n_heads=2, decoder_d_ff=64
        )
        decoder = MAEDecoder(d_encoder=32, n_features=10, max_seq_length=48, config=config)

        B, max_obs, n_vis = 2, 20, 5
        encoded_visible = torch.randn(B, n_vis, 32)
        ssl_mask = torch.ones(B, max_obs, dtype=torch.bool)
        ssl_mask[:, :15] = False  # First 15 are masked
        token_info = {
            "timestep_idx": torch.randint(0, 8, (B, max_obs)),
            "feature_idx": torch.randint(0, 10, (B, max_obs)),
        }
        padding_mask = torch.ones(B, max_obs, dtype=torch.bool)

        pred = decoder(encoded_visible, ssl_mask, token_info, max_obs, padding_mask)
        assert pred.shape == (B, max_obs)

    def test_decoder_mask_token_is_learnable(self):
        from slices.models.pretraining.mae import MAEDecoder

        config = MAEConfig(
            decoder_d_model=16, decoder_n_layers=1, decoder_n_heads=2, decoder_d_ff=32
        )
        decoder = MAEDecoder(d_encoder=16, n_features=5, max_seq_length=10, config=config)

        assert decoder.mask_token.requires_grad
        assert decoder.mask_token.shape == (1, 1, 16)

    def test_decoder_mask_token_in_decoder_space(self):
        """MASK_TOKEN should be in d_decoder space, not d_encoder."""
        from slices.models.pretraining.mae import MAEDecoder

        d_encoder, d_decoder = 64, 32
        config = MAEConfig(decoder_d_model=d_decoder, decoder_n_layers=1, decoder_n_heads=2)
        decoder = MAEDecoder(d_encoder=d_encoder, n_features=10, max_seq_length=48, config=config)

        assert decoder.mask_token.shape[-1] == d_decoder


# =============================================================================
# MAE Objective tests
# =============================================================================


class TestMAEObjective:
    """Tests for observation-level MAE objective."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    @pytest.fixture
    def mae_config(self):
        return MAEConfig(
            mask_ratio=0.75,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )

    def test_mae_initialization(self, encoder, mae_config):
        mae = MAEObjective(encoder, mae_config)
        assert mae.encoder is encoder
        assert mae.config == mae_config
        assert hasattr(mae, "decoder")
        # No MISSING_TOKEN in observation-level MAE
        assert mae.missing_token is None

    def test_mae_forward(self, encoder, mae_config):
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3

        loss, metrics = mae(x, obs_mask)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert "mae_loss" in metrics
        assert "mae_recon_loss_masked" in metrics
        assert "mae_recon_loss_visible" in metrics
        assert "mae_mask_ratio_actual" in metrics
        assert "mae_n_tokens_per_sample" in metrics
        assert "mae_n_visible_per_sample" in metrics
        assert "mae_n_masked_per_sample" in metrics

    def test_mae_backward(self, encoder, mae_config):
        mae = MAEObjective(encoder, mae_config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.rand(2, 8, 10) > 0.3

        loss, _ = mae(x, obs_mask)
        loss.backward()

        for name, param in mae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_encoder_sees_fewer_tokens(self, encoder, mae_config):
        """Encoder should process fewer tokens than total observations (75% masked)."""
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 4, 12, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        loss, metrics = mae(x, obs_mask)

        B = 4
        n_total = metrics["mae_n_tokens_per_sample"] * B
        n_visible = metrics["mae_n_visible_per_sample"] * B
        # Visible should be ~25% of total
        assert n_visible < n_total
        ratio = n_visible / n_total
        assert 0.10 <= ratio <= 0.50  # Allow tolerance around 0.25

    def test_loss_only_on_masked(self, encoder, mae_config):
        """Loss should only be computed on masked positions."""
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        loss, metrics = mae(x, obs_mask)

        B = 2
        n_masked = metrics["mae_n_masked_per_sample"] * B
        n_total = metrics["mae_n_tokens_per_sample"] * B
        assert n_masked > 0
        assert n_masked < n_total

    def test_mae_requires_observation_encoder(self):
        """MAE should reject encoders without tokenize/encode methods."""
        config = TransformerConfig(d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="none")
        encoder = TransformerEncoder(config)
        mae_config = MAEConfig()

        with pytest.raises(ValueError, match="tokenize.*encode"):
            MAEObjective(encoder, mae_config)

    def test_mae_requires_no_pooling(self):
        """MAE should reject encoder with pooling != 'none'."""
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="mean"
        )
        encoder = ObservationTransformerEncoder(config)
        mae_config = MAEConfig()

        with pytest.raises(ValueError, match="pooling='none'"):
            MAEObjective(encoder, mae_config)

    def test_mae_training_step(self, encoder, mae_config):
        mae = MAEObjective(encoder, mae_config)
        optimizer = torch.optim.Adam(mae.parameters(), lr=1e-3)

        x = torch.randn(4, 8, 10)
        obs_mask = torch.ones(4, 8, 10, dtype=torch.bool)

        optimizer.zero_grad()
        loss, _ = mae(x, obs_mask)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_mae_respects_obs_mask(self, encoder, mae_config):
        """With sparse data, tokens should match observed count."""
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.8  # ~20% observed

        loss, metrics = mae(x, obs_mask)
        assert torch.isfinite(loss)

        # n_tokens should reflect observed count
        expected_obs = obs_mask.sum().item()
        assert metrics["mae_n_tokens_per_sample"] * B == expected_obs

    def test_mae_get_encoder(self, encoder, mae_config):
        mae = MAEObjective(encoder, mae_config)
        assert mae.get_encoder() is encoder


# =============================================================================
# Mask ratio tests
# =============================================================================


class TestMaskRatios:
    """Test different mask ratios."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    @pytest.mark.parametrize("mask_ratio", [0.25, 0.5, 0.75, 0.9])
    def test_mask_ratios(self, encoder, mask_ratio):
        config = MAEConfig(
            mask_ratio=mask_ratio,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, config)

        x = torch.randn(4, 8, 10)
        obs_mask = torch.ones(4, 8, 10, dtype=torch.bool)

        loss, metrics = mae(x, obs_mask)
        assert torch.isfinite(loss)

        # Check actual ratio is close to target
        actual = metrics["mae_mask_ratio_actual"]
        assert abs(actual - mask_ratio) < 0.15

    def test_high_mask_ratio_at_least_one_visible(self, encoder):
        """Even with very high mask ratio, at least 1 visible token per sample."""
        config = MAEConfig(
            mask_ratio=0.99,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, config)

        x = torch.randn(4, 8, 10)
        obs_mask = torch.ones(4, 8, 10, dtype=torch.bool)

        loss, metrics = mae(x, obs_mask)
        assert torch.isfinite(loss)
        assert metrics["mae_n_visible_per_sample"] * 4 >= 4  # At least 1 per sample


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    def test_very_sparse_data(self, encoder):
        """Test with very few observations per sample."""
        mae_config = MAEConfig(
            mask_ratio=0.5,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        # Very sparse: ~5% observed
        obs_mask = torch.rand(B, T, D) > 0.95

        # Ensure at least some observations exist
        obs_mask[0, 0, 0] = True
        obs_mask[1, 0, 0] = True

        loss, metrics = mae(x, obs_mask)
        assert torch.isfinite(loss)

    def test_single_observation(self, encoder):
        """Test with exactly 1 observed value per sample."""
        mae_config = MAEConfig(
            mask_ratio=0.75,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 4, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.zeros(B, T, D, dtype=torch.bool)
        obs_mask[0, 0, 0] = True
        obs_mask[1, 1, 3] = True

        loss, metrics = mae(x, obs_mask)
        assert torch.isfinite(loss)
        # With 1 obs, at least 1 must be visible, so 0 masked → loss should be 0
        # (no masked tokens to compute loss on)
        assert loss.item() < 1e-7

    def test_batch_with_varying_sparsity(self, encoder):
        """Test batch where samples have different observation counts."""
        mae_config = MAEConfig(
            mask_ratio=0.75,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 3, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.zeros(B, T, D, dtype=torch.bool)
        obs_mask[0] = True  # All observed
        obs_mask[1, :4, :5] = True  # Half observed
        obs_mask[2, 0, 0] = True  # Almost nothing

        loss, metrics = mae(x, obs_mask)
        assert torch.isfinite(loss)


# =============================================================================
# Gradient flow tests
# =============================================================================


class TestGradientFlow:
    """Test gradient flow through all components."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    def test_gradients_flow_to_encoder(self, encoder):
        mae_config = MAEConfig(
            mask_ratio=0.75,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, mae_config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.ones(2, 8, 10, dtype=torch.bool)

        loss, _ = mae(x, obs_mask)
        loss.backward()

        # Encoder parameters should have gradients
        encoder_has_grad = False
        for name, param in encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                encoder_has_grad = True
                break
        assert encoder_has_grad, "No gradients flowed to encoder"

    def test_gradients_flow_to_decoder(self, encoder):
        mae_config = MAEConfig(
            mask_ratio=0.75,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, mae_config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.ones(2, 8, 10, dtype=torch.bool)

        loss, _ = mae(x, obs_mask)
        loss.backward()

        # Decoder mask_token should have gradient
        assert mae.decoder.mask_token.grad is not None
        assert mae.decoder.mask_token.grad.abs().sum() > 0

    def test_gradients_flow_to_decoder_feature_embed(self, encoder):
        mae_config = MAEConfig(
            mask_ratio=0.75,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, mae_config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.ones(2, 8, 10, dtype=torch.bool)

        loss, _ = mae(x, obs_mask)
        loss.backward()

        assert mae.decoder.feature_embed.weight.grad is not None


# =============================================================================
# Training convergence test
# =============================================================================


class TestTrainingConvergence:
    """Test that loss decreases during training."""

    def test_loss_decreases(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=2, n_heads=4, d_ff=64, pooling="none"
        )
        encoder = ObservationTransformerEncoder(config)

        mae_config = MAEConfig(
            mask_ratio=0.75,
            decoder_d_model=32,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=64,
        )
        mae = MAEObjective(encoder, mae_config)
        optimizer = torch.optim.Adam(mae.parameters(), lr=1e-3)

        # Fixed data for reproducible convergence
        torch.manual_seed(42)
        B, T, D = 4, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        initial_loss = None
        for step in range(30):
            optimizer.zero_grad()
            loss, _ = mae(x, obs_mask)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert (
            final_loss < initial_loss
        ), f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_training_masks_vary(self):
        """Training masks should vary between forward passes."""
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        encoder = ObservationTransformerEncoder(config)

        mae_config = MAEConfig(
            mask_ratio=0.5,
            decoder_d_model=16,
            decoder_n_layers=1,
            decoder_n_heads=2,
            decoder_d_ff=32,
        )
        mae = MAEObjective(encoder, mae_config)
        mae.train()

        x = torch.randn(4, 8, 10)
        obs_mask = torch.ones(4, 8, 10, dtype=torch.bool)

        losses = []
        for _ in range(5):
            loss, _ = mae(x, obs_mask)
            losses.append(loss.item())

        unique = len(set(f"{l:.6f}" for l in losses))
        assert unique > 1, "Training masks should vary but all losses were identical"


# =============================================================================
# SSL Factory tests
# =============================================================================


class TestSSLFactory:
    """Tests for SSL factory with observation-level MAE."""

    def test_build_ssl_objective_mae(self):
        encoder_config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="none"
        )
        encoder = ObservationTransformerEncoder(encoder_config)
        mae_config = MAEConfig(mask_ratio=0.75)

        ssl_objective = build_ssl_objective(encoder, mae_config)

        assert isinstance(ssl_objective, MAEObjective)
        assert ssl_objective.encoder is encoder

    def test_build_ssl_objective_unknown(self):
        encoder_config = ObservationTransformerConfig(d_input=10, d_model=32, pooling="none")
        encoder = ObservationTransformerEncoder(encoder_config)

        from slices.models.pretraining.base import SSLConfig

        bad_config = SSLConfig(name="unknown_objective")

        with pytest.raises(ValueError, match="Unknown SSL objective"):
            build_ssl_objective(encoder, bad_config)

    def test_get_ssl_config_class(self):
        config_cls = get_ssl_config_class("mae")
        assert config_cls == MAEConfig

    def test_get_ssl_config_class_unknown(self):
        with pytest.raises(ValueError, match="Unknown SSL objective"):
            get_ssl_config_class("unknown_objective")


# =============================================================================
# Observation encoder forward (finetuning mode) tests
# =============================================================================


class TestObservationEncoderForward:
    """Test observation encoder in forward/finetuning mode."""

    def test_forward_with_mean_pooling(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="mean"
        )
        encoder = ObservationTransformerEncoder(config)

        B, T, D = 4, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3

        out = encoder(x, mask=obs_mask)
        assert out.shape == (B, 32)  # (B, d_model)

    def test_forward_with_no_pooling(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        encoder = ObservationTransformerEncoder(config)

        B, T, D = 2, 4, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        out = encoder(x, mask=obs_mask)
        assert out.shape == (B, T * D, 32)  # (B, max_obs, d_model)

    def test_forward_without_mask(self):
        """Without mask, all values treated as observed."""
        config = ObservationTransformerConfig(
            d_input=5, d_model=16, n_layers=1, n_heads=2, d_ff=32, pooling="mean"
        )
        encoder = ObservationTransformerEncoder(config)

        x = torch.randn(2, 4, 5)
        out = encoder(x)
        assert out.shape == (2, 16)

    def test_encoder_output_dim(self):
        config = ObservationTransformerConfig(d_input=10, d_model=64)
        encoder = ObservationTransformerEncoder(config)
        assert encoder.get_output_dim() == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
