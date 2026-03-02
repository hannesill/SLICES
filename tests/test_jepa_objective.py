"""Tests for timestep-level JEPA (Joint-Embedding Predictive Architecture) SSL objective."""

import pytest
import torch
from slices.models.encoders import (
    TransformerConfig,
    TransformerEncoder,
)
from slices.models.pretraining import (
    JEPAConfig,
    JEPAObjective,
    build_ssl_objective,
    get_ssl_config_class,
)

# =============================================================================
# Predictor tests
# =============================================================================


class TestJEPAPredictor:
    """Tests for JEPA predictor."""

    def test_predictor_output_shape(self):
        from slices.models.pretraining.jepa import JEPAPredictor

        config = JEPAConfig(
            predictor_d_model=32,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=64,
        )
        predictor = JEPAPredictor(d_encoder=32, max_seq_length=48, config=config)

        B, T, n_vis = 2, 8, 3
        encoded_visible = torch.randn(B, n_vis, 32)
        ssl_mask = torch.ones(B, T, dtype=torch.bool)
        ssl_mask[:, :5] = False  # First 5 masked
        token_info = {
            "timestep_idx": torch.arange(T).unsqueeze(0).expand(B, -1),
        }

        pred = predictor(encoded_visible, ssl_mask, token_info, T)
        # Output should be (B, T, d_encoder)
        assert pred.shape == (B, T, 32)

    def test_predictor_mask_token_is_learnable(self):
        from slices.models.pretraining.jepa import JEPAPredictor

        config = JEPAConfig(
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )
        predictor = JEPAPredictor(d_encoder=16, max_seq_length=10, config=config)

        assert predictor.mask_token.requires_grad
        assert predictor.mask_token.shape == (1, 1, 16)

    def test_predictor_output_in_encoder_space(self):
        """Output should be in d_encoder space, not d_predictor."""
        from slices.models.pretraining.jepa import JEPAPredictor

        d_encoder, d_predictor = 64, 32
        config = JEPAConfig(
            predictor_d_model=d_predictor,
            predictor_n_layers=1,
            predictor_n_heads=2,
        )
        predictor = JEPAPredictor(d_encoder=d_encoder, max_seq_length=48, config=config)

        B, T, n_vis = 2, 8, 3
        encoded_visible = torch.randn(B, n_vis, d_encoder)
        ssl_mask = torch.ones(B, T, dtype=torch.bool)
        ssl_mask[:, :5] = False
        token_info = {
            "timestep_idx": torch.arange(T).unsqueeze(0).expand(B, -1),
        }

        pred = predictor(encoded_visible, ssl_mask, token_info, T)
        assert pred.shape[-1] == d_encoder


# =============================================================================
# Init validation tests
# =============================================================================


class TestJEPAInit:
    """Tests for JEPA initialization and validation."""

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
    def jepa_config(self):
        return JEPAConfig(
            mask_ratio=0.75,
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )

    def test_initialization(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)
        assert jepa.encoder is encoder
        assert jepa.config == jepa_config
        assert hasattr(jepa, "predictor")
        assert hasattr(jepa, "target_encoder")
        assert jepa.missing_token is None

    def test_requires_obs_aware(self):
        config = TransformerConfig(d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="none")
        encoder = TransformerEncoder(config)
        jepa_config = JEPAConfig()

        with pytest.raises(ValueError, match="obs_aware=True"):
            JEPAObjective(encoder, jepa_config)

    def test_requires_no_pooling(self):
        config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=1,
            n_heads=4,
            pooling="mean",
            obs_aware=True,
        )
        encoder = TransformerEncoder(config)
        jepa_config = JEPAConfig()

        with pytest.raises(ValueError, match="pooling='none'"):
            JEPAObjective(encoder, jepa_config)

    def test_target_encoder_is_frozen(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)
        for param in jepa.target_encoder.parameters():
            assert not param.requires_grad

    def test_target_encoder_is_copy(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)
        assert jepa.target_encoder is not jepa.encoder
        for p_online, p_target in zip(encoder.parameters(), jepa.target_encoder.parameters()):
            assert torch.allclose(p_online, p_target)


# =============================================================================
# Forward pass tests
# =============================================================================


class TestJEPAForward:
    """Tests for JEPA forward pass."""

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
    def jepa_config(self):
        return JEPAConfig(
            mask_ratio=0.75,
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )

    def test_forward_returns_loss_and_metrics(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3

        loss, metrics = jepa(x, obs_mask)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert "jepa_loss" in metrics
        assert "ssl_loss" in metrics
        assert "jepa_mask_ratio_actual" in metrics
        assert "jepa_n_timesteps" in metrics
        assert "jepa_n_visible_per_sample" in metrics
        assert "jepa_n_masked_per_sample" in metrics
        assert "jepa_momentum" in metrics

    def test_backward(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.rand(2, 8, 10) > 0.3

        loss, _ = jepa(x, obs_mask)
        loss.backward()

        has_encoder_grad = False
        for param in jepa.encoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_encoder_grad = True
                break
        assert has_encoder_grad

        has_pred_grad = False
        for param in jepa.predictor.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_pred_grad = True
                break
        assert has_pred_grad

    def test_encoder_sees_fewer_timesteps(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)

        B, T, D = 4, 12, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        _, metrics = jepa(x, obs_mask)

        n_visible = metrics["jepa_n_visible_per_sample"]
        n_masked = metrics["jepa_n_masked_per_sample"]
        ratio = n_visible / (n_visible + n_masked)
        assert 0.10 <= ratio <= 0.50  # ~25% visible


# =============================================================================
# Momentum tests
# =============================================================================


class TestJEPAMomentum:
    """Tests for JEPA momentum update."""

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
    def jepa_config(self):
        return JEPAConfig(
            mask_ratio=0.75,
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
            momentum_base=0.996,
            momentum_final=1.0,
        )

    def test_momentum_update_changes_target(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)

        original_params = [p.clone() for p in jepa.target_encoder.parameters()]

        x = torch.randn(2, 8, 10)
        obs_mask = torch.ones(2, 8, 10, dtype=torch.bool)
        loss, _ = jepa(x, obs_mask)
        loss.backward()
        optimizer = torch.optim.Adam([p for p in jepa.parameters() if p.requires_grad], lr=1e-2)
        optimizer.step()

        jepa.momentum_update(progress=0.5)

        changed = False
        for orig, new in zip(original_params, jepa.target_encoder.parameters()):
            if not torch.allclose(orig, new):
                changed = True
                break
        assert changed

    def test_momentum_schedule(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)

        jepa.momentum_update(progress=0.0)
        assert abs(jepa._current_momentum - 0.996) < 1e-6

        jepa.momentum_update(progress=1.0)
        assert abs(jepa._current_momentum - 1.0) < 1e-6

        jepa.momentum_update(progress=0.5)
        expected = 0.996 + (1.0 - 0.996) * 0.5
        assert abs(jepa._current_momentum - expected) < 1e-6

    def test_ema_formula(self, encoder, jepa_config):
        jepa = JEPAObjective(encoder, jepa_config)

        with torch.no_grad():
            for p in jepa.encoder.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        online_params = [p.clone() for p in jepa.encoder.parameters()]
        target_params = [p.clone() for p in jepa.target_encoder.parameters()]

        m = 0.996
        jepa.momentum_update(progress=0.0)

        for online, old_target, new_target in zip(
            online_params,
            target_params,
            jepa.target_encoder.parameters(),
        ):
            expected = m * old_target + (1 - m) * online
            assert torch.allclose(new_target, expected, atol=1e-6)
            assert not torch.allclose(new_target, old_target, atol=1e-6)


# =============================================================================
# Edge cases
# =============================================================================


class TestJEPAEdgeCases:
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
        config = JEPAConfig(
            mask_ratio=0.5,
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )
        jepa = JEPAObjective(encoder, config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.95
        obs_mask[0, 0, 0] = True
        obs_mask[1, 0, 0] = True

        loss, metrics = jepa(x, obs_mask)
        assert torch.isfinite(loss)

    def test_batch_with_varying_sparsity(self, encoder):
        config = JEPAConfig(
            mask_ratio=0.75,
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )
        jepa = JEPAObjective(encoder, config)

        B, T, D = 3, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.zeros(B, T, D, dtype=torch.bool)
        obs_mask[0] = True
        obs_mask[1, :4, :5] = True
        obs_mask[2, 0, 0] = True

        loss, metrics = jepa(x, obs_mask)
        assert torch.isfinite(loss)


# =============================================================================
# Gradient flow tests
# =============================================================================


class TestJEPAGradientFlow:
    """Test gradient flow."""

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

    def test_gradients_to_encoder_and_predictor(self, encoder):
        config = JEPAConfig(
            mask_ratio=0.75,
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )
        jepa = JEPAObjective(encoder, config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.ones(2, 8, 10, dtype=torch.bool)

        loss, _ = jepa(x, obs_mask)
        loss.backward()

        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in jepa.encoder.parameters()
        )
        assert encoder_has_grad

        pred_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in jepa.predictor.parameters()
        )
        assert pred_has_grad

    def test_no_gradients_to_target(self, encoder):
        config = JEPAConfig(
            mask_ratio=0.75,
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )
        jepa = JEPAObjective(encoder, config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.ones(2, 8, 10, dtype=torch.bool)

        loss, _ = jepa(x, obs_mask)
        loss.backward()

        for param in jepa.target_encoder.parameters():
            assert param.grad is None


# =============================================================================
# Training convergence test
# =============================================================================


class TestJEPAConvergence:
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

        jepa_config = JEPAConfig(
            mask_ratio=0.75,
            predictor_d_model=32,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=64,
        )
        jepa = JEPAObjective(encoder, jepa_config)
        optimizer = torch.optim.Adam([p for p in jepa.parameters() if p.requires_grad], lr=1e-3)

        torch.manual_seed(42)
        B, T, D = 4, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        initial_loss = None
        for step in range(30):
            optimizer.zero_grad()
            loss, _ = jepa(x, obs_mask)
            loss.backward()
            optimizer.step()
            jepa.momentum_update(progress=step / 30)

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert (
            final_loss < initial_loss
        ), f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"


# =============================================================================
# Loss type tests
# =============================================================================


class TestJEPALossTypes:
    """Test different loss types."""

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

    def test_mse_loss(self, encoder):
        config = JEPAConfig(
            loss_type="mse",
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )
        jepa = JEPAObjective(encoder, config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.ones(2, 8, 10, dtype=torch.bool)

        loss, _ = jepa(x, obs_mask)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_cosine_loss(self, encoder):
        config = JEPAConfig(
            loss_type="cosine",
            predictor_d_model=16,
            predictor_n_layers=1,
            predictor_n_heads=2,
            predictor_d_ff=32,
        )
        jepa = JEPAObjective(encoder, config)

        x = torch.randn(2, 8, 10)
        obs_mask = torch.ones(2, 8, 10, dtype=torch.bool)

        loss, _ = jepa(x, obs_mask)
        assert torch.isfinite(loss)
        assert loss.item() >= 0


# =============================================================================
# Factory integration tests
# =============================================================================


class TestJEPAFactory:
    """Tests for JEPA factory integration."""

    def test_in_registry(self):
        from slices.models.pretraining.factory import CONFIG_REGISTRY, SSL_REGISTRY

        assert SSL_REGISTRY["jepa"] is JEPAObjective
        assert CONFIG_REGISTRY["jepa"] is JEPAConfig
        assert get_ssl_config_class("jepa") == JEPAConfig

    def test_build_works(self):
        encoder_config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=1,
            n_heads=4,
            pooling="none",
            obs_aware=True,
        )
        encoder = TransformerEncoder(encoder_config)
        jepa_config = JEPAConfig(mask_ratio=0.75)

        ssl_objective = build_ssl_objective(encoder, jepa_config)

        assert isinstance(ssl_objective, JEPAObjective)
        assert ssl_objective.encoder is encoder

    def test_get_encoder(self):
        encoder_config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=1,
            n_heads=4,
            pooling="none",
            obs_aware=True,
        )
        encoder = TransformerEncoder(encoder_config)
        jepa_config = JEPAConfig()

        jepa = JEPAObjective(encoder, jepa_config)
        assert jepa.get_encoder() is encoder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
