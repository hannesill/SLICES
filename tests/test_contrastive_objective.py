"""Tests for timestep-level Contrastive SSL objective (temporal + instance modes)."""

import pytest
import torch
from slices.models.encoders import (
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

        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# =============================================================================
# Init validation tests
# =============================================================================


class TestContrastiveInit:
    """Tests for Contrastive initialization and validation."""

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
    def contrastive_config(self):
        return ContrastiveConfig(
            mode="instance",
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
        assert not hasattr(obj, "target_encoder")

    def test_requires_obs_aware(self):
        config = TransformerConfig(d_input=10, d_model=32, n_layers=1, n_heads=4, pooling="none")
        encoder = TransformerEncoder(config)
        cont_config = ContrastiveConfig()

        with pytest.raises(ValueError, match="obs_aware=True"):
            ContrastiveObjective(encoder, cont_config)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="must be 'temporal' or 'instance'"):
            ContrastiveConfig(mode="bad")

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
    def contrastive_config(self):
        return ContrastiveConfig(
            mode="instance",
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
        assert "ssl_loss" in metrics
        assert "contrastive_accuracy" in metrics
        assert "contrastive_pos_similarity" in metrics
        assert "contrastive_temperature" in metrics
        assert "contrastive_n_timesteps" in metrics
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

        assert metrics["contrastive_n_visible_view1"] > 0
        assert metrics["contrastive_n_visible_view2"] > 0

    def test_empty_timesteps_are_excluded_from_visible_counts(self, encoder, contrastive_config):
        """Contrastive views should ignore hours with no observed variables."""
        obj = ContrastiveObjective(encoder, contrastive_config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.zeros(B, T, D, dtype=torch.bool)
        obs_mask[:, 2, :4] = True
        obs_mask[:, 6, :4] = True

        _, metrics = obj(x, obs_mask)

        assert metrics["contrastive_n_visible_view1"] <= 2
        assert metrics["contrastive_n_visible_view2"] <= 2

    def test_single_eligible_timestep_falls_back_to_shared_view(self, encoder):
        """Complementary masks should not create an all-padding view on sparse samples."""
        config = ContrastiveConfig(
            mode="instance",
            mask_ratio=0.5,
            complementary_masks=True,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=0.1,
        )
        obj = ContrastiveObjective(encoder, config)

        B, T, D = 2, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.zeros(B, T, D, dtype=torch.bool)
        obs_mask[:, 3, :4] = True

        loss, metrics = obj(x, obs_mask)

        assert torch.isfinite(loss)
        assert torch.isfinite(metrics["contrastive_loss"])
        assert metrics["contrastive_n_visible_view1"] > 0
        assert metrics["contrastive_n_visible_view2"] > 0


# =============================================================================
# NT-Xent loss tests
# =============================================================================


class TestNTXentLoss:
    """Tests for NT-Xent loss behavior."""

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

    def test_accuracy_in_range(self, encoder):
        config = ContrastiveConfig(
            mode="instance",
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

        z = torch.randn(B, proj_dim)
        z = torch.nn.functional.normalize(z, dim=-1)
        z1, z2 = z.clone(), z.clone()

        z_cat = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z_cat, z_cat.t()) / temperature
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)])
        mask = torch.eye(2 * B, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))
        loss = torch.nn.functional.cross_entropy(sim_matrix, labels)

        assert loss.item() < 2.0

        preds = sim_matrix.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        assert accuracy.item() == 1.0

    def test_temperature_effect_on_loss(self, encoder):
        """Lower temperature should sharpen the distribution, increasing loss magnitude."""
        config_low_temp = ContrastiveConfig(
            mode="instance",
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=0.01,
        )
        config_high_temp = ContrastiveConfig(
            mode="instance",
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=1.0,
        )

        torch.manual_seed(123)
        obj_low = ContrastiveObjective(encoder, config_low_temp)
        from copy import deepcopy

        encoder2 = deepcopy(encoder)
        obj_high = ContrastiveObjective(encoder2, config_high_temp)
        obj_high.load_state_dict(obj_low.state_dict())

        x = torch.randn(8, 8, 10)
        obs_mask = torch.ones(8, 8, 10, dtype=torch.bool)

        torch.manual_seed(99)
        loss_low, metrics_low = obj_low(x, obs_mask)
        torch.manual_seed(99)
        loss_high, metrics_high = obj_high(x, obs_mask)

        assert loss_low.item() > loss_high.item()

    def test_temperature_in_metrics(self, encoder):
        temp = 0.07
        config = ContrastiveConfig(
            mode="instance",
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

    def test_sparse_data(self, encoder):
        config = ContrastiveConfig(
            mode="instance",
            mask_ratio=0.5,
            proj_hidden_dim=64,
            proj_output_dim=16,
        )
        obj = ContrastiveObjective(encoder, config)

        B, T, D = 4, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.rand(B, T, D) > 0.9
        for b in range(B):
            obs_mask[b, 0, 0] = True

        loss, metrics = obj(x, obs_mask)
        assert torch.isfinite(loss)


# =============================================================================
# Gradient flow tests
# =============================================================================


class TestContrastiveGradientFlow:
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

    def test_gradients_to_encoder_and_projection(self, encoder):
        config = ContrastiveConfig(
            mode="instance",
            mask_ratio=0.75,
            proj_hidden_dim=64,
            proj_output_dim=16,
        )
        obj = ContrastiveObjective(encoder, config)

        x = torch.randn(4, 8, 10)
        obs_mask = torch.ones(4, 8, 10, dtype=torch.bool)

        loss, _ = obj(x, obs_mask)
        loss.backward()

        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in obj.encoder.parameters()
        )
        assert encoder_has_grad

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

        cont_config = ContrastiveConfig(
            mode="instance",
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
# Temporal mode tests
# =============================================================================


class TestTemporalContrastive:
    """Tests for temporal contrastive mode (per-timestep overlap pairs)."""

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
    def temporal_config(self):
        return ContrastiveConfig(
            mode="temporal",
            mask_ratio=0.5,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=0.1,
            complementary_masks=False,
        )

    def test_scatter_to_full_shape_and_zeros(self, encoder, temporal_config):
        """_scatter_to_full produces correct shape with zeros at masked positions."""
        from slices.models.pretraining.contrastive import ContrastiveObjective

        B, T, d_enc = 4, 8, 32
        # Create a mask with exactly 4 visible per sample
        ssl_mask = torch.zeros(B, T, dtype=torch.bool)
        ssl_mask[:, :4] = True  # first 4 timesteps visible

        encoded = torch.randn(B, 4, d_enc)
        full = ContrastiveObjective._scatter_to_full(encoded, ssl_mask, T)

        assert full.shape == (B, T, d_enc)
        # Masked positions (indices 4-7) should be zero
        assert (full[:, 4:, :] == 0).all()
        # Visible positions should be non-zero (with high probability)
        assert full[:, :4, :].abs().sum() > 0

    def test_scatter_to_full_ignores_padded_visible_tokens(self, encoder, temporal_config):
        """Uneven visible counts should not scatter padded tokens into masked slots."""
        from slices.models.pretraining.contrastive import ContrastiveObjective

        encoded = torch.tensor(
            [
                [[10.0], [11.0]],
                [[20.0], [99.0]],
            ]
        )
        ssl_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ]
        )

        full = ContrastiveObjective._scatter_to_full(encoded, ssl_mask, n_timesteps=3)

        assert torch.allclose(full[1, 0], torch.tensor([20.0]))
        assert torch.allclose(full[1, 1], torch.tensor([0.0]))
        assert torch.allclose(full[1, 2], torch.tensor([0.0]))

    def test_scatter_to_full_gradient_flow(self):
        """Gradients flow through scatter into torch.zeros back to the source tensor."""
        from slices.models.pretraining.contrastive import ContrastiveObjective

        B, T, d = 2, 6, 4
        ssl_mask = torch.zeros(B, T, dtype=torch.bool)
        ssl_mask[:, :3] = True  # 3 visible per sample

        encoded = torch.randn(B, 3, d, requires_grad=True)
        full = ContrastiveObjective._scatter_to_full(encoded, ssl_mask, T)

        # Sum visible positions only — gradient should reach encoded
        loss = full[ssl_mask].sum()
        loss.backward()

        assert encoded.grad is not None
        assert encoded.grad.abs().sum() > 0
        # Each visible encoded token contributes d elements to the sum,
        # so each gradient element should be 1.0
        assert torch.allclose(encoded.grad, torch.ones_like(encoded.grad))

    def test_scatter_to_full_roundtrip(self, encoder, temporal_config):
        """Scatter should place tokens back at the correct original positions."""
        from slices.models.pretraining.contrastive import ContrastiveObjective
        from slices.models.pretraining.masking import (
            create_timestep_mask,
            extract_visible_timesteps,
        )

        B, T, d = 4, 8, 32
        tokens = torch.randn(B, T, d)

        torch.manual_seed(42)
        ssl_mask = create_timestep_mask(B, T, 0.5, tokens.device)

        vis_tokens, vis_padding = extract_visible_timesteps(tokens, ssl_mask)
        full = ContrastiveObjective._scatter_to_full(vis_tokens, ssl_mask, T)

        # At visible positions, scattered values should match original tokens
        for b in range(B):
            for t in range(T):
                if ssl_mask[b, t]:
                    assert torch.allclose(full[b, t], tokens[b, t], atol=1e-6)

    def test_forward_returns_loss_and_overlap_metrics(self, encoder, temporal_config):
        obj = ContrastiveObjective(encoder, temporal_config)

        B, T, D = 8, 8, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        loss, metrics = obj(x, obs_mask)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)

        # Standard metrics
        assert "contrastive_loss" in metrics
        assert "ssl_loss" in metrics
        assert "contrastive_accuracy" in metrics
        assert "contrastive_pos_similarity" in metrics

        # Temporal-specific overlap metrics
        assert "contrastive_n_overlap_tokens" in metrics
        assert "contrastive_n_overlap_per_sample" in metrics
        assert metrics["contrastive_n_overlap_tokens"] > 0

    def test_gradient_flow_temporal(self, encoder, temporal_config):
        """Gradients flow to encoder and projection head in temporal mode."""
        obj = ContrastiveObjective(encoder, temporal_config)

        x = torch.randn(8, 8, 10)
        obs_mask = torch.ones(8, 8, 10, dtype=torch.bool)

        loss, _ = obj(x, obs_mask)
        loss.backward()

        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in obj.encoder.parameters()
        )
        assert encoder_has_grad

        proj_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in obj.projection_head.parameters()
        )
        assert proj_has_grad

    def test_convergence_temporal(self):
        """Loss decreases during training in temporal mode."""
        enc_config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            pooling="none",
            obs_aware=True,
            max_seq_length=48,
        )
        encoder = TransformerEncoder(enc_config)

        cont_config = ContrastiveConfig(
            mode="temporal",
            mask_ratio=0.5,
            proj_hidden_dim=64,
            proj_output_dim=16,
            temperature=0.1,
            complementary_masks=False,
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

    def test_overlap_count_reasonable(self, encoder, temporal_config):
        """With mask_ratio=0.5, ~25% of timesteps should overlap."""
        obj = ContrastiveObjective(encoder, temporal_config)

        B, T, D = 64, 48, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        _, metrics = obj(x, obs_mask)

        # Expected overlap per sample ≈ 0.5 * 0.5 * 48 = 12
        avg_overlap = metrics["contrastive_n_overlap_per_sample"]
        assert 4 < avg_overlap < 24, f"Unexpected overlap: {avg_overlap}"


# =============================================================================
# Complementary masks tests
# =============================================================================


class TestComplementaryMasks:
    """Tests for complementary mask behavior."""

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

    def test_complementary_masks_zero_overlap(self, encoder):
        """Complementary masks should have zero overlap."""
        from slices.models.pretraining.masking import create_timestep_mask

        B, T = 8, 48
        ssl_mask_1 = create_timestep_mask(B, T, 0.5, torch.device("cpu"))
        ssl_mask_2 = ~ssl_mask_1

        overlap = ssl_mask_1 & ssl_mask_2
        assert overlap.sum() == 0

    def test_complementary_masks_full_coverage(self, encoder):
        """Complementary masks should cover all timesteps."""
        from slices.models.pretraining.masking import create_timestep_mask

        B, T = 8, 48
        ssl_mask_1 = create_timestep_mask(B, T, 0.5, torch.device("cpu"))
        ssl_mask_2 = ~ssl_mask_1

        coverage = ssl_mask_1 | ssl_mask_2
        assert coverage.all()

    def test_complementary_temporal_raises(self):
        """complementary_masks=True + mode='temporal' should raise ValueError."""
        with pytest.raises(ValueError, match="incompatible with mode='temporal'"):
            ContrastiveConfig(
                mode="temporal",
                complementary_masks=True,
            )

    def test_forward_with_complementary_masks(self, encoder):
        """Forward pass works with complementary masks (default)."""
        config = ContrastiveConfig(
            mode="instance",
            mask_ratio=0.5,
            proj_hidden_dim=64,
            proj_output_dim=16,
            complementary_masks=True,
        )
        obj = ContrastiveObjective(encoder, config)

        B, T, D = 8, 16, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        loss, metrics = obj(x, obs_mask)
        assert torch.isfinite(loss)
        assert "contrastive_loss" in metrics

    def test_forward_with_independent_masks(self, encoder):
        """Forward pass works with independent masks (complementary_masks=False)."""
        config = ContrastiveConfig(
            mode="instance",
            mask_ratio=0.5,
            proj_hidden_dim=64,
            proj_output_dim=16,
            complementary_masks=False,
        )
        obj = ContrastiveObjective(encoder, config)

        B, T, D = 8, 16, 10
        x = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, D, dtype=torch.bool)

        loss, metrics = obj(x, obs_mask)
        assert torch.isfinite(loss)


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
        encoder_config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=1,
            n_heads=4,
            pooling="none",
            obs_aware=True,
        )
        encoder = TransformerEncoder(encoder_config)
        cont_config = ContrastiveConfig(mask_ratio=0.75)

        ssl_objective = build_ssl_objective(encoder, cont_config)

        assert isinstance(ssl_objective, ContrastiveObjective)
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
        cont_config = ContrastiveConfig()

        obj = ContrastiveObjective(encoder, cont_config)
        assert obj.get_encoder() is encoder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
