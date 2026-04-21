"""Tests for timestep-level MAE (Masked Autoencoder) SSL objective."""

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

        B, T, n_vis = 2, 8, 3
        encoded_visible = torch.randn(B, n_vis, 32)
        ssl_mask = torch.ones(B, T, dtype=torch.bool)
        ssl_mask[:, :5] = False  # First 5 are masked
        token_info = {
            "timestep_idx": torch.arange(T).unsqueeze(0).expand(B, -1),
        }

        pred = decoder(encoded_visible, ssl_mask, token_info, T)
        assert pred.shape == (B, T, 10)  # (B, T, D)

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

    def test_decoder_ignores_padded_visible_tokens(self):
        """Padded visible-token rows must not overwrite real masked positions."""
        from slices.models.pretraining.mae import MAEDecoder

        config = MAEConfig(decoder_d_model=2, decoder_n_layers=1, decoder_n_heads=1)
        decoder = MAEDecoder(d_encoder=2, n_features=2, max_seq_length=3, config=config)
        decoder.encoder_proj = torch.nn.Identity()
        decoder.decoder = torch.nn.Identity()
        decoder.embed_dropout = torch.nn.Identity()
        decoder.output_proj = torch.nn.Identity()
        decoder.time_pe.zero_()
        with torch.no_grad():
            decoder.mask_token.fill_(-1.0)

        encoded_visible = torch.tensor(
            [
                [[10.0, 10.0], [11.0, 11.0]],
                [[20.0, 20.0], [99.0, 99.0]],
            ]
        )
        ssl_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ]
        )
        token_info = {
            "timestep_idx": torch.arange(3).unsqueeze(0).expand(2, -1),
        }

        pred = decoder(encoded_visible, ssl_mask, token_info, n_timesteps=3)

        assert torch.allclose(pred[1, 0], torch.tensor([20.0, 20.0]))
        assert torch.allclose(pred[1, 1], torch.tensor([-1.0, -1.0]))
        assert torch.allclose(pred[1, 2], torch.tensor([-1.0, -1.0]))


# =============================================================================
# MAE Objective tests
# =============================================================================


class TestMAEObjective:
    """Tests for timestep-level MAE objective."""

    @pytest.fixture
    def encoder(self):
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
        assert "mae_n_timesteps" in metrics
        assert "mae_n_visible_per_sample" in metrics
        assert "mae_n_masked_per_sample" in metrics
        assert "mae_n_loss_positions" in metrics

    def test_mae_backward(self, encoder, mae_config):
        mae = MAEObjective(encoder, mae_config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.rand(2, 8, 10) > 0.3

        loss, _ = mae(x, obs_mask)
        loss.backward()

        for name, param in mae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_encoder_sees_fewer_timesteps(self, encoder, mae_config):
        """Encoder should process fewer timesteps than total (75% masked)."""
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 4, 12, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        loss, metrics = mae(x, obs_mask)

        n_visible = metrics["mae_n_visible_per_sample"]
        n_masked = metrics["mae_n_masked_per_sample"]
        # Visible should be ~25% of total
        ratio = n_visible / (n_visible + n_masked)
        assert 0.10 <= ratio <= 0.50

    def test_loss_only_on_masked_and_observed(self, encoder, mae_config):
        """Loss should only be computed on observed features at masked timesteps."""
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        loss, metrics = mae(x, obs_mask)

        n_masked = metrics["mae_n_masked_per_sample"] * B
        assert n_masked > 0
        assert metrics["mae_n_loss_positions"] > 0

    def test_mae_requires_obs_aware(self):
        """MAE should reject encoders without obs_aware=True."""
        config = TransformerConfig(d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="none")
        encoder = TransformerEncoder(config)
        mae_config = MAEConfig()

        with pytest.raises(ValueError, match="obs_aware=True"):
            MAEObjective(encoder, mae_config)

    def test_mae_requires_no_pooling(self):
        """MAE should reject encoder with pooling != 'none'."""
        config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=1,
            n_heads=4,
            pooling="mean",
            obs_aware=True,
        )
        encoder = TransformerEncoder(config)
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
        """With sparse data, loss positions should only be on observed features."""
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.8  # ~20% observed

        loss, metrics = mae(x, obs_mask)
        assert torch.isfinite(loss)

        # Loss positions should be <= observed features at masked timesteps
        n_loss = metrics["mae_n_loss_positions"]
        n_observed = obs_mask.sum().item()
        assert n_loss <= n_observed

    def test_empty_timesteps_are_excluded_from_mae_counts(self, encoder, mae_config):
        """MAE masking metrics should only count timesteps with observations."""
        mae = MAEObjective(encoder, mae_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.zeros(B, T, D, dtype=torch.bool)
        obs_mask[:, 1, :3] = True
        obs_mask[:, 5, :3] = True
        obs_mask[:, 6, :3] = True

        _, metrics = mae(x, obs_mask)

        assert metrics["mae_n_visible_per_sample"] + metrics[
            "mae_n_masked_per_sample"
        ] == pytest.approx(3.0)

    def test_mae_passes_valid_timestep_mask_to_masking(self, encoder, mae_config, monkeypatch):
        """Fully unobserved timesteps should be excluded from MAE masking."""
        mae = MAEObjective(encoder, mae_config)
        captured = {}

        def fake_create_timestep_mask(
            batch_size,
            n_timesteps,
            mask_ratio,
            device,
            valid_timestep_mask=None,
        ):
            captured["batch_size"] = batch_size
            captured["n_timesteps"] = n_timesteps
            captured["mask_ratio"] = mask_ratio
            captured["device"] = device
            captured["masking_valid_timestep_mask"] = valid_timestep_mask.clone()
            return torch.ones(batch_size, n_timesteps, dtype=torch.bool, device=device)

        def fake_extract_visible_timesteps(
            tokens,
            ssl_mask,
            valid_timestep_mask=None,
        ):
            captured["extract_valid_timestep_mask"] = valid_timestep_mask.clone()
            visible_tokens = tokens[:, :1, :]
            vis_padding = torch.ones(
                tokens.shape[0],
                1,
                dtype=torch.bool,
                device=tokens.device,
            )
            return visible_tokens, vis_padding

        monkeypatch.setattr(
            "slices.models.pretraining.mae.create_timestep_mask",
            fake_create_timestep_mask,
        )
        monkeypatch.setattr(
            "slices.models.pretraining.mae.extract_visible_timesteps",
            fake_extract_visible_timesteps,
        )

        x = torch.randn(2, 4, 10)
        obs_mask = torch.tensor(
            [
                [
                    [True] * 10,
                    [False] * 10,
                    [True] * 10,
                    [False] * 10,
                ],
                [
                    [False] * 10,
                    [True] * 10,
                    [True] * 10,
                    [False] * 10,
                ],
            ],
            dtype=torch.bool,
        )

        mae(x, obs_mask)

        expected = obs_mask.any(dim=-1)
        assert torch.equal(captured["masking_valid_timestep_mask"], expected)
        assert torch.equal(captured["extract_valid_timestep_mask"], expected)

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

        x = torch.randn(4, 32, 10)
        obs_mask = torch.ones(4, 32, 10, dtype=torch.bool)

        loss, metrics = mae(x, obs_mask)
        assert torch.isfinite(loss)

        # Check actual ratio is close to target (32 timesteps → low discretization noise)
        actual = metrics["mae_mask_ratio_actual"]
        assert abs(actual - mask_ratio) < 0.15

    def test_high_mask_ratio_at_least_one_visible(self, encoder):
        """Even with very high mask ratio, at least 1 visible timestep per sample."""
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
        assert metrics["mae_n_visible_per_sample"] >= 1  # At least 1 per sample


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def encoder(self):
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


# =============================================================================
# Training convergence test
# =============================================================================


class TestTrainingConvergence:
    """Test that loss decreases during training."""

    def test_loss_decreases(self):
        config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            pooling="none",
            obs_aware=True,
            max_seq_length=48,
        )
        encoder = TransformerEncoder(config)

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
        encoder = TransformerEncoder(config)

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
    """Tests for SSL factory with timestep-level MAE."""

    def test_build_ssl_objective_mae(self):
        encoder_config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=1,
            n_heads=4,
            pooling="none",
            obs_aware=True,
        )
        encoder = TransformerEncoder(encoder_config)
        mae_config = MAEConfig(mask_ratio=0.75)

        ssl_objective = build_ssl_objective(encoder, mae_config)

        assert isinstance(ssl_objective, MAEObjective)
        assert ssl_objective.encoder is encoder

    def test_build_ssl_objective_unknown(self):
        encoder_config = TransformerConfig(
            d_input=10,
            d_model=32,
            pooling="none",
            obs_aware=True,
        )
        encoder = TransformerEncoder(encoder_config)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
