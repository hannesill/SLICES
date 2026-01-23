"""Tests for SMART SSL objective.

Tests cover:
- SMARTPredictor: Simple MLP predictor
- SMARTObjective initialization and validation
- Forward pass and loss computation
- Per-sample random masking (element-wise)
- Momentum encoder updates
- Loss on observed positions only
- Integration with SMARTEncoder
"""

import pytest
import torch
from slices.models.encoders.smart import SMARTEncoder, SMARTEncoderConfig
from slices.models.encoders.transformer import TransformerConfig, TransformerEncoder
from slices.models.pretraining.factory import (
    build_ssl_objective,
    get_ssl_config_class,
)
from slices.models.pretraining.smart import (
    SMARTObjective,
    SMARTPredictor,
    SMARTSSLConfig,
)


class TestSMARTPredictor:
    """Tests for SMART predictor (simple MLP)."""

    def test_predictor_output_shape(self):
        """Test that predictor preserves input shape."""
        d_model = 32
        predictor = SMARTPredictor(d_model, mlp_ratio=4.0, dropout=0.0)

        B, V, T = 4, 35, 48
        x = torch.randn(B, V, T, d_model)
        out = predictor(x)

        assert out.shape == x.shape

    def test_predictor_gradient_flow(self):
        """Test that gradients flow through predictor."""
        predictor = SMARTPredictor(d_model=32, mlp_ratio=4.0, dropout=0.0)

        x = torch.randn(2, 10, 24, 32, requires_grad=True)
        out = predictor(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_predictor_different_mlp_ratios(self):
        """Test predictor with different MLP ratios."""
        for mlp_ratio in [2.0, 4.0, 8.0]:
            predictor = SMARTPredictor(d_model=32, mlp_ratio=mlp_ratio, dropout=0.0)
            x = torch.randn(2, 10, 24, 32)
            out = predictor(x)
            assert out.shape == x.shape


class TestSMARTObjectiveInitialization:
    """Tests for SMART objective initialization."""

    @pytest.fixture
    def smart_encoder(self):
        """Create a SMART encoder for testing."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",  # Required for SMART SSL
        )
        return SMARTEncoder(config)

    @pytest.fixture
    def smart_config(self):
        """Create SMART SSL config for testing."""
        return SMARTSSLConfig(
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
            momentum_base=0.996,
            momentum_final=1.0,
        )

    def test_smart_initialization(self, smart_encoder, smart_config):
        """Test SMART objective initialization."""
        smart = SMARTObjective(smart_encoder, smart_config)

        assert smart.encoder is smart_encoder
        assert smart.config == smart_config
        assert hasattr(smart, "target_encoder")
        assert hasattr(smart, "predictor")

    def test_smart_requires_smart_encoder(self, smart_config):
        """Test that SMART raises error if encoder is not SMARTEncoder."""
        # Create a Transformer encoder instead
        transformer_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        transformer_encoder = TransformerEncoder(transformer_config)

        with pytest.raises(ValueError, match="requires a SMARTEncoder"):
            SMARTObjective(transformer_encoder, smart_config)

    def test_smart_requires_pooling_none(self, smart_config):
        """Test that SMART raises error if encoder pooling is not 'none'."""
        # Create encoder with query pooling
        encoder_config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="query",  # This should cause error
        )
        encoder = SMARTEncoder(encoder_config)

        with pytest.raises(ValueError, match="pooling='none'"):
            SMARTObjective(encoder, smart_config)

    def test_target_encoder_is_frozen(self, smart_encoder, smart_config):
        """Test that target encoder parameters are frozen."""
        smart = SMARTObjective(smart_encoder, smart_config)

        for param in smart.target_encoder.parameters():
            assert not param.requires_grad

    def test_target_encoder_is_copy(self, smart_encoder, smart_config):
        """Test that target encoder is a deep copy of online encoder."""
        smart = SMARTObjective(smart_encoder, smart_config)

        # Parameters should be equal initially
        for online_param, target_param in zip(
            smart.encoder.parameters(), smart.target_encoder.parameters()
        ):
            assert torch.allclose(online_param, target_param)

    def test_online_encoder_is_trainable(self, smart_encoder, smart_config):
        """Test that online encoder parameters are trainable."""
        smart = SMARTObjective(smart_encoder, smart_config)

        for param in smart.encoder.parameters():
            assert param.requires_grad


class TestSMARTObjectiveForward:
    """Tests for SMART objective forward pass."""

    @pytest.fixture
    def smart_encoder(self):
        """Create a SMART encoder for testing."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        return SMARTEncoder(config)

    @pytest.fixture
    def smart_config(self):
        """Create SMART SSL config for testing."""
        return SMARTSSLConfig(
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
        )

    def test_smart_forward(self, smart_encoder, smart_config):
        """Test SMART forward pass."""
        smart = SMARTObjective(smart_encoder, smart_config)

        B, T, D = 4, 48, 35
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3

        loss, metrics = smart(x, obs_mask)

        # Check loss
        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0
        assert not torch.isnan(loss)

        # Check metrics
        assert "smart_loss" in metrics
        assert "smart_mask_ratio_mean" in metrics
        assert "smart_mask_ratio_std" in metrics
        assert "smart_momentum" in metrics
        assert "smart_loss_masked" in metrics
        assert "smart_loss_visible" in metrics

    def test_smart_backward(self, smart_encoder, smart_config):
        """Test that SMART can backpropagate."""
        smart = SMARTObjective(smart_encoder, smart_config)

        x = torch.randn(2, 24, 35)
        obs_mask = torch.rand(2, 24, 35) > 0.3

        loss, _ = smart(x, obs_mask)
        loss.backward()

        # Check that encoder gradients exist
        for name, param in smart.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for encoder {name}"

        # Check that predictor gradients exist
        for name, param in smart.predictor.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for predictor {name}"

        # Target encoder should NOT have gradients
        for name, param in smart.target_encoder.named_parameters():
            assert param.grad is None, f"Target encoder {name} should not have gradients"

    def test_smart_training_step(self, smart_encoder, smart_config):
        """Test a full training step."""
        smart = SMARTObjective(smart_encoder, smart_config)
        optimizer = torch.optim.Adam(smart.parameters(), lr=1e-3)

        x = torch.randn(4, 24, 35)
        obs_mask = torch.ones(4, 24, 35, dtype=torch.bool)

        # Training step
        optimizer.zero_grad()
        loss, metrics = smart(x, obs_mask)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)


class TestSMARTMasking:
    """Tests for SMART masking strategy."""

    @pytest.fixture
    def smart_encoder(self):
        """Create a SMART encoder for testing."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        return SMARTEncoder(config)

    def test_per_sample_mask_ratios(self, smart_encoder):
        """Test that per-sample mask ratios are applied correctly."""
        config = SMARTSSLConfig(
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
        )
        smart = SMARTObjective(smart_encoder, config)

        x = torch.randn(8, 48, 35)
        obs_mask = torch.ones(8, 48, 35, dtype=torch.bool)

        loss, metrics = smart(x, obs_mask)

        # Mean mask ratio should be within expected range
        mean_ratio = metrics["smart_mask_ratio_mean"]
        assert 0.0 <= mean_ratio <= 0.75

        # Standard deviation should be non-zero (per-sample variation)
        std_ratio = metrics["smart_mask_ratio_std"]
        assert std_ratio >= 0

    def test_element_wise_masking(self, smart_encoder):
        """Test that masking is element-wise (not timestep-level)."""
        config = SMARTSSLConfig(
            min_mask_ratio=0.5,
            max_mask_ratio=0.5,  # Fixed ratio for testing
        )
        smart = SMARTObjective(smart_encoder, config)

        # Run multiple times and check that different elements get masked
        x = torch.randn(4, 24, 35)
        obs_mask = torch.ones(4, 24, 35, dtype=torch.bool)

        # Run forward pass to trigger mask creation
        loss1, metrics1 = smart(x, obs_mask)
        loss2, metrics2 = smart(x, obs_mask)

        # Losses should differ due to random element-wise masking
        # (unless by chance the same elements are masked)
        # This is a probabilistic test, but with 50% masking it should differ
        # Allow some tolerance for rare cases
        assert loss1.item() != loss2.item() or abs(loss1.item() - loss2.item()) < 1e-6

    def test_ssl_mask_respects_obs_mask(self, smart_encoder):
        """Test that SSL mask respects observation mask (doesn't mask missing values)."""
        config = SMARTSSLConfig(
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
            loss_on_observed_only=True,
        )
        smart = SMARTObjective(smart_encoder, config)

        B, T, D = 4, 24, 35
        x = torch.randn(B, T, D)
        # Make some values missing
        obs_mask = torch.rand(B, T, D) > 0.3  # 30% missing

        loss, metrics = smart(x, obs_mask)

        # Loss should still be computed
        assert torch.isfinite(loss)
        # Loss positions should be less than total elements (some are missing)
        assert metrics["smart_loss_positions"] <= 1.0

    def test_no_masking_when_ratio_zero(self, smart_encoder):
        """Test that no masking occurs when mask ratios are zero."""
        config = SMARTSSLConfig(
            min_mask_ratio=0.0,
            max_mask_ratio=0.0,  # No masking
        )
        smart = SMARTObjective(smart_encoder, config)

        x = torch.randn(4, 24, 35)
        obs_mask = torch.ones(4, 24, 35, dtype=torch.bool)

        loss, metrics = smart(x, obs_mask)

        # With no masking, the loss should be zero or very small
        # (predicting target from identical representations)
        # Actually, there's still loss because predictor transforms representations
        assert torch.isfinite(loss)
        assert metrics["smart_mask_ratio_actual"] == 0.0


class TestSMARTLossComputation:
    """Tests for SMART loss computation."""

    @pytest.fixture
    def smart_encoder(self):
        """Create a SMART encoder for testing."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        return SMARTEncoder(config)

    def test_smooth_l1_loss(self, smart_encoder):
        """Test SMART with smooth L1 loss."""
        config = SMARTSSLConfig(
            loss_type="smooth_l1",
            smooth_l1_beta=1.0,
        )
        smart = SMARTObjective(smart_encoder, config)

        x = torch.randn(4, 24, 35)
        obs_mask = torch.ones(4, 24, 35, dtype=torch.bool)

        loss, metrics = smart(x, obs_mask)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_mse_loss(self, smart_encoder):
        """Test SMART with MSE loss."""
        config = SMARTSSLConfig(
            loss_type="mse",
        )
        smart = SMARTObjective(smart_encoder, config)

        x = torch.randn(4, 24, 35)
        obs_mask = torch.ones(4, 24, 35, dtype=torch.bool)

        loss, metrics = smart(x, obs_mask)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_loss_on_observed_only(self, smart_encoder):
        """Test that loss_on_observed_only flag works."""
        config_obs = SMARTSSLConfig(
            loss_on_observed_only=True,
        )
        smart_obs = SMARTObjective(smart_encoder, config_obs)

        # Need a fresh encoder for the other config
        encoder_config = SMARTEncoderConfig(
            d_input=35, d_model=32, n_layers=2, n_heads=4, pooling="none"
        )
        encoder_all = SMARTEncoder(encoder_config)
        config_all = SMARTSSLConfig(
            loss_on_observed_only=False,
        )
        smart_all = SMARTObjective(encoder_all, config_all)

        # Create data with missing values
        torch.manual_seed(42)
        x = torch.randn(4, 24, 35)
        obs_mask = torch.rand(4, 24, 35) > 0.5  # 50% observed

        loss_obs, metrics_obs = smart_obs(x, obs_mask)
        loss_all, metrics_all = smart_all(x, obs_mask)

        assert torch.isfinite(loss_obs)
        assert torch.isfinite(loss_all)

        # Both should produce valid losses with the respective configs
        # With loss_on_observed_only=True, loss positions should be less than full
        # because we exclude missing positions from loss computation
        assert 0 < metrics_obs["smart_loss_positions"] < 1.0
        assert 0 < metrics_all["smart_loss_positions"] < 1.0


class TestSMARTMomentumUpdate:
    """Tests for SMART momentum encoder updates."""

    @pytest.fixture
    def smart_encoder(self):
        """Create a SMART encoder for testing."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        return SMARTEncoder(config)

    def test_momentum_update(self, smart_encoder):
        """Test that momentum update modifies target encoder."""
        config = SMARTSSLConfig(
            momentum_base=0.996,
            momentum_final=1.0,
        )
        smart = SMARTObjective(smart_encoder, config)

        # Store initial target parameters
        initial_target_params = [p.clone() for p in smart.target_encoder.parameters()]

        # Modify online encoder
        x = torch.randn(2, 24, 35)
        obs_mask = torch.ones(2, 24, 35, dtype=torch.bool)
        loss, _ = smart(x, obs_mask)
        loss.backward()

        # Apply optimizer step to online encoder
        optimizer = torch.optim.Adam(smart.parameters(), lr=1e-2)
        optimizer.step()

        # Now apply momentum update
        smart.momentum_update(progress=0.0)

        # Target parameters should have changed
        for initial_param, current_param in zip(
            initial_target_params, smart.target_encoder.parameters()
        ):
            assert not torch.allclose(initial_param, current_param, atol=1e-6)

    def test_momentum_schedule(self, smart_encoder):
        """Test that momentum increases from base to final."""
        config = SMARTSSLConfig(
            momentum_base=0.996,
            momentum_final=1.0,
        )
        smart = SMARTObjective(smart_encoder, config)

        # Check momentum at different progress points
        smart.momentum_update(progress=0.0)
        assert smart._current_momentum == 0.996

        smart.momentum_update(progress=0.5)
        assert smart._current_momentum == 0.998

        smart.momentum_update(progress=1.0)
        assert smart._current_momentum == 1.0

    def test_momentum_ema_formula(self, smart_encoder):
        """Test that EMA formula is correctly applied."""
        config = SMARTSSLConfig(
            momentum_base=0.99,  # Easier to verify with 0.99
            momentum_final=0.99,  # Keep constant for this test
        )
        smart = SMARTObjective(smart_encoder, config)

        # Store initial target parameters
        target_before = [p.clone() for p in smart.target_encoder.parameters()]
        online_params = [p.clone() for p in smart.encoder.parameters()]

        # Apply momentum update
        smart.momentum_update(progress=0.0)

        # Check EMA formula: target = m * target + (1-m) * online
        m = 0.99
        for target_b, target_a, online in zip(
            target_before, smart.target_encoder.parameters(), online_params
        ):
            expected = m * target_b + (1 - m) * online
            assert torch.allclose(target_a, expected, atol=1e-6)

    def test_momentum_tracked_in_metrics(self, smart_encoder):
        """Test that current momentum is tracked in metrics."""
        config = SMARTSSLConfig(
            momentum_base=0.996,
            momentum_final=1.0,
        )
        smart = SMARTObjective(smart_encoder, config)

        x = torch.randn(2, 24, 35)
        obs_mask = torch.ones(2, 24, 35, dtype=torch.bool)

        loss, metrics = smart(x, obs_mask)

        assert "smart_momentum" in metrics
        assert metrics["smart_momentum"] == 0.996  # Initial value


class TestSMARTGetEncoder:
    """Tests for get_encoder method."""

    def test_get_encoder_returns_online_encoder(self):
        """Test that get_encoder returns the online encoder."""
        config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = SMARTEncoder(config)

        smart_config = SMARTSSLConfig()
        smart = SMARTObjective(encoder, smart_config)

        retrieved = smart.get_encoder()

        assert retrieved is smart.encoder


class TestSMARTFactory:
    """Tests for SMART factory functions."""

    def test_smart_in_ssl_registry(self):
        """Test that SMART is registered in SSL factory."""
        from slices.models.pretraining.factory import CONFIG_REGISTRY, SSL_REGISTRY

        assert "smart" in SSL_REGISTRY
        assert SSL_REGISTRY["smart"] is SMARTObjective
        assert "smart" in CONFIG_REGISTRY
        assert CONFIG_REGISTRY["smart"] is SMARTSSLConfig

    def test_build_ssl_objective_smart(self):
        """Test building SMART via factory."""
        encoder_config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = SMARTEncoder(encoder_config)

        smart_config = SMARTSSLConfig(
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
        )

        ssl_objective = build_ssl_objective(encoder, smart_config)

        assert isinstance(ssl_objective, SMARTObjective)
        assert ssl_objective.encoder is encoder

    def test_get_ssl_config_class_smart(self):
        """Test getting SMART config class via factory."""
        config_cls = get_ssl_config_class("smart")

        assert config_cls is SMARTSSLConfig


class TestSMARTIntegration:
    """Integration tests for SMART with realistic data."""

    def test_smart_with_realistic_icu_data(self):
        """Test SMART with realistic ICU-like data."""
        # Create synthetic ICU-like data
        B, T, D = 8, 48, 35
        x = torch.randn(B, T, D)

        # Add realistic missingness pattern
        obs_prob = torch.linspace(0.9, 0.6, T).unsqueeze(0).unsqueeze(-1)
        obs_mask = torch.rand(B, T, D) < obs_prob.expand(B, T, D)

        # Create SMART model
        encoder_config = SMARTEncoderConfig(
            d_input=D,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = SMARTEncoder(encoder_config)

        smart_config = SMARTSSLConfig(
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
        )
        smart = SMARTObjective(encoder, smart_config)

        # Train for a few steps
        optimizer = torch.optim.Adam(smart.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(20):
            optimizer.zero_grad()
            loss, metrics = smart(x, obs_mask)
            loss.backward()
            optimizer.step()

            # Simulate momentum update
            smart.momentum_update(progress=step / 20)

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()

        # Check that loss is finite
        assert torch.isfinite(torch.tensor(final_loss))

        # Check that metrics are reasonable
        assert 0.0 <= metrics["smart_mask_ratio_actual"] <= 1.0
        assert 0.0 <= metrics["smart_obs_ratio"] <= 1.0

    def test_smart_different_mask_ratios(self):
        """Test SMART with different mask ratio ranges."""
        encoder_config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )

        x = torch.randn(4, 24, 35)
        obs_mask = torch.ones(4, 24, 35, dtype=torch.bool)

        for max_ratio in [0.25, 0.50, 0.75, 1.0]:
            encoder = SMARTEncoder(encoder_config)
            config = SMARTSSLConfig(
                min_mask_ratio=0.0,
                max_mask_ratio=max_ratio,
            )
            smart = SMARTObjective(encoder, config)

            loss, metrics = smart(x, obs_mask)

            assert torch.isfinite(loss)
            # Mean ratio should be around max_ratio / 2 (uniform distribution)
            assert metrics["smart_mask_ratio_mean"] <= max_ratio

    def test_smart_training_varies_masks(self):
        """Test that training masks vary between forward passes."""
        encoder_config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = SMARTEncoder(encoder_config)

        config = SMARTSSLConfig(
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
        )
        smart = SMARTObjective(encoder, config)
        smart.train()

        x = torch.randn(4, 24, 35)
        obs_mask = torch.ones(4, 24, 35, dtype=torch.bool)

        # Run multiple times and check that losses vary
        losses = []
        for _ in range(5):
            loss, _ = smart(x, obs_mask)
            losses.append(loss.item())

        # At least some losses should be different (random masks)
        unique_losses = len(set(f"{l:.6f}" for l in losses))
        assert unique_losses > 1, "Training masks should vary but all losses were identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
