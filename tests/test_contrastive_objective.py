"""Tests for observation-level Contrastive (SimCLR-style) SSL objective."""

import pytest
import torch
from slices.models.encoders import (
    ObservationTransformerConfig,
    ObservationTransformerEncoder,
    TransformerConfig,
    TransformerEncoder,
)
from slices.models.pretraining import (
    ContrastiveConfig,
    ContrastiveObjective,
    build_ssl_objective,
    get_ssl_config_class,
)

# =============================================================================
# Projection head tests
# =============================================================================


class TestProjectionHead:
    """Tests for the projection head."""

    def test_output_shape(self):
        from slices.models.pretraining.contrastive import ProjectionHead

        head = ProjectionHead(d_input=32, hidden_dim=64, output_dim=16)
        x = torch.randn(4, 32)
        z = head(x)
        assert z.shape == (4, 16)

    def test_l2_normalized(self):
        from slices.models.pretraining.contrastive import ProjectionHead

        head = ProjectionHead(d_input=32, hidden_dim=64, output_dim=16)
        x = torch.randn(4, 32)
        z = head(x)

        # Each row should have unit norm
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# =============================================================================
# Init validation tests
# =============================================================================


class TestContrastiveInit:
    """Tests for Contrastive initialization and validation."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    @pytest.fixture
    def contrastive_config(self):
        return ContrastiveConfig(
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=0.1,
        )

    def test_initialization(self, encoder, contrastive_config):
        obj = ContrastiveObjective(encoder, contrastive_config)
        assert obj.encoder is encoder
        assert obj.config == contrastive_config
        assert hasattr(obj, "projection_head")
        assert obj.missing_token is None
        # No target encoder in SimCLR-style
        assert not hasattr(obj, "target_encoder")

    def test_requires_observation_encoder(self):
        config = TransformerConfig(d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="none")
        encoder = TransformerEncoder(config)
        cont_config = ContrastiveConfig()

        with pytest.raises(ValueError, match="tokenize.*encode"):
            ContrastiveObjective(encoder, cont_config)

    def test_requires_no_pooling(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="mean"
        )
        encoder = ObservationTransformerEncoder(config)
        cont_config = ContrastiveConfig()

        with pytest.raises(ValueError, match="pooling='none'"):
            ContrastiveObjective(encoder, cont_config)


# =============================================================================
# Forward pass tests
# =============================================================================


class TestContrastiveForward:
    """Tests for Contrastive forward pass."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    @pytest.fixture
    def contrastive_config(self):
        return ContrastiveConfig(
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=0.1,
        )

    def test_forward_returns_loss_and_metrics(self, encoder, contrastive_config):
        obj = ContrastiveObjective(encoder, contrastive_config)

        B, T, D = 4, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.3

        loss, metrics = obj(x, obs_mask)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert "contrastive_loss" in metrics
        assert "reconstruction_loss" in metrics
        assert "contrastive_accuracy" in metrics
        assert "contrastive_pos_similarity" in metrics
        assert "contrastive_temperature" in metrics
        assert "contrastive_n_tokens_per_sample" in metrics
        assert "contrastive_n_visible_view1" in metrics
        assert "contrastive_n_visible_view2" in metrics

    def test_backward(self, encoder, contrastive_config):
        obj = ContrastiveObjective(encoder, contrastive_config)

        x = torch.randn(4, 8, 10)
        obs_mask = torch.rand(4, 8, 10) > 0.3

        loss, _ = obj(x, obs_mask)
        loss.backward()

        for name, param in obj.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_two_views_encoded(self, encoder, contrastive_config):
        """Forward pass should use two different views."""
        obj = ContrastiveObjective(encoder, contrastive_config)

        B, T, D = 4, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        _, metrics = obj(x, obs_mask)

        # Both views should have visible tokens
        assert metrics["contrastive_n_visible_view1"] > 0
        assert metrics["contrastive_n_visible_view2"] > 0


# =============================================================================
# NT-Xent loss tests
# =============================================================================


class TestNTXentLoss:
    """Tests for NT-Xent loss behavior."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    def test_accuracy_in_range(self, encoder):
        config = ContrastiveConfig(
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
        )
        obj = ContrastiveObjective(encoder, config)

        x = torch.randn(8, 8, 10)
        obs_mask = torch.ones(8, 8, 10, dtype=torch.bool)

        _, metrics = obj(x, obs_mask)
        assert 0.0 <= metrics["contrastive_accuracy"] <= 1.0

    def test_perfect_alignment_low_loss(self, encoder):
        """When z1 == z2, NT-Xent loss should be near-minimal."""

        B, proj_dim = 8, 16
        temperature = 0.1

        # Create identical, L2-normalized embeddings for both views
        z = torch.randn(B, proj_dim)
        z = torch.nn.functional.normalize(z, dim=-1)
        z1, z2 = z.clone(), z.clone()

        # Compute NT-Xent manually
        z_cat = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z_cat, z_cat.t()) / temperature
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)])
        mask = torch.eye(2 * B, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))
        loss = torch.nn.functional.cross_entropy(sim_matrix, labels)

        # With perfect alignment, positive sim = 1/tau and negatives < 1/tau
        # Loss should be low (near zero for distinct samples)
        assert loss.item() < 2.0  # Much less than log(2B-1) ≈ 2.7

        # Also: accuracy should be 1.0 with perfect alignment
        preds = sim_matrix.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        assert accuracy.item() == 1.0

    def test_temperature_effect_on_loss(self, encoder):
        """Lower temperature should sharpen the distribution, increasing loss magnitude."""
        config_low_temp = ContrastiveConfig(
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=0.01,
        )
        config_high_temp = ContrastiveConfig(
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=1.0,
        )

        torch.manual_seed(123)
        obj_low = ContrastiveObjective(encoder, config_low_temp)
        # Need a separate encoder copy for fair comparison
        from copy import deepcopy

        encoder2 = deepcopy(encoder)
        obj_high = ContrastiveObjective(encoder2, config_high_temp)
        # Copy weights so both objectives are identical except temperature
        obj_high.load_state_dict(obj_low.state_dict())

        x = torch.randn(8, 8, 10)
        obs_mask = torch.ones(8, 8, 10, dtype=torch.bool)

        torch.manual_seed(99)
        loss_low, metrics_low = obj_low(x, obs_mask)
        torch.manual_seed(99)
        loss_high, metrics_high = obj_high(x, obs_mask)

        # Lower temperature produces higher loss (sharper distribution, harder task)
        assert loss_low.item() > loss_high.item()

    def test_temperature_in_metrics(self, encoder):
        temp = 0.07
        config = ContrastiveConfig(
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=temp,
        )
        obj = ContrastiveObjective(encoder, config)

        x = torch.randn(4, 8, 10)
        obs_mask = torch.ones(4, 8, 10, dtype=torch.bool)

        _, metrics = obj(x, obs_mask)
        assert metrics["contrastive_temperature"] == temp


# =============================================================================
# Edge cases
# =============================================================================


class TestContrastiveEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    def test_sparse_data(self, encoder):
        config = ContrastiveConfig(
            mask_ratio=0.5,
            proj_hidden_dim=64,
            proj_output_dim=16,
        )
        obj = ContrastiveObjective(encoder, config)

        B, T, D = 4, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.9  # ~10% observed
        # Ensure at least some observations per sample
        for b in range(B):
            obs_mask[b, 0, 0] = True

        loss, metrics = obj(x, obs_mask)
        assert torch.isfinite(loss)

    def test_single_observation(self, encoder):
        config = ContrastiveConfig(
            mask_ratio=0.5,
            proj_hidden_dim=64,
            proj_output_dim=16,
        )
        obj = ContrastiveObjective(encoder, config)

        B, T, D = 2, 4, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.zeros(B, T, D, dtype=torch.bool)
        obs_mask[0, 0, 0] = True
        obs_mask[1, 1, 3] = True

        loss, metrics = obj(x, obs_mask)
        assert torch.isfinite(loss)


# =============================================================================
# Gradient flow tests
# =============================================================================


class TestContrastiveGradientFlow:
    """Test gradient flow."""

    @pytest.fixture
    def encoder(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, d_ff=64, pooling="none"
        )
        return ObservationTransformerEncoder(config)

    def test_gradients_to_encoder_and_projection(self, encoder):
        config = ContrastiveConfig(
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
        )
        obj = ContrastiveObjective(encoder, config)

        x = torch.randn(4, 8, 10)
        obs_mask = torch.ones(4, 8, 10, dtype=torch.bool)

        loss, _ = obj(x, obs_mask)
        loss.backward()

        # Encoder should have gradients
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in obj.encoder.parameters()
        )
        assert encoder_has_grad

        # Projection head should have gradients
        proj_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in obj.projection_head.parameters()
        )
        assert proj_has_grad


# =============================================================================
# Training convergence test
# =============================================================================


class TestContrastiveConvergence:
    """Test that loss decreases during training."""

    def test_loss_decreases(self):
        config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=2, n_heads=4, d_ff=64, pooling="none"
        )
        encoder = ObservationTransformerEncoder(config)

        cont_config = ContrastiveConfig(
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=0.1,
        )
        obj = ContrastiveObjective(encoder, cont_config)
        optimizer = torch.optim.Adam(obj.parameters(), lr=1e-3)

        torch.manual_seed(42)
        B, T, D = 8, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        initial_loss = None
        for step in range(30):
            optimizer.zero_grad()
            loss, _ = obj(x, obs_mask)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert (
            final_loss < initial_loss
        ), f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"


# =============================================================================
# Factory integration tests
# =============================================================================


class TestContrastiveFactory:
    """Tests for Contrastive factory integration."""

    def test_in_registry(self):
        from slices.models.pretraining.factory import CONFIG_REGISTRY, SSL_REGISTRY

        assert SSL_REGISTRY["contrastive"] is ContrastiveObjective
        assert CONFIG_REGISTRY["contrastive"] is ContrastiveConfig
        assert get_ssl_config_class("contrastive") == ContrastiveConfig

    def test_build_works(self):
        encoder_config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="none"
        )
        encoder = ObservationTransformerEncoder(encoder_config)
        cont_config = ContrastiveConfig(mask_ratio=0.75)

        ssl_objective = build_ssl_objective(encoder, cont_config)

        assert isinstance(ssl_objective, ContrastiveObjective)
        assert ssl_objective.encoder is encoder

    def test_get_encoder(self):
        encoder_config = ObservationTransformerConfig(
            d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="none"
        )
        encoder = ObservationTransformerEncoder(encoder_config)
        cont_config = ContrastiveConfig()

        obj = ContrastiveObjective(encoder, cont_config)
        assert obj.get_encoder() is encoder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
