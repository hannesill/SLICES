"""Tests for factory modules in slices.models.

Tests the factory pattern implementations for encoders, task heads, and SSL objectives.
"""

import pytest
import torch
import torch.nn as nn
from slices.models.encoders.factory import (
    ENCODER_CONFIG_REGISTRY,
    ENCODER_REGISTRY,
    build_encoder,
    get_encoder_config_class,
)
from slices.models.encoders.smart import SMARTEncoder, SMARTEncoderConfig
from slices.models.encoders.transformer import TransformerConfig, TransformerEncoder
from slices.models.heads.base import TaskHeadConfig
from slices.models.heads.factory import (
    TASK_HEAD_REGISTRY,
    build_task_head,
    build_task_head_from_dict,
    get_available_task_heads,
)
from slices.models.heads.mlp import LinearTaskHead, MLPTaskHead
from slices.models.pretraining.factory import (
    CONFIG_REGISTRY,
    SSL_REGISTRY,
    build_ssl_objective,
    get_ssl_config_class,
)
from slices.models.pretraining.mae import MAEConfig, MAEObjective
from slices.models.pretraining.smart import SMARTObjective, SMARTSSLConfig


class TestEncoderFactory:
    """Tests for encoder factory functions."""

    def test_encoder_registry_contains_transformer(self):
        """Encoder registry should contain transformer encoder."""
        assert "transformer" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY["transformer"] is TransformerEncoder

    def test_encoder_config_registry_contains_transformer(self):
        """Config registry should contain transformer config."""
        assert "transformer" in ENCODER_CONFIG_REGISTRY
        assert ENCODER_CONFIG_REGISTRY["transformer"] is TransformerConfig

    def test_build_encoder_creates_transformer(self):
        """build_encoder should create TransformerEncoder correctly."""
        config_dict = {
            "d_input": 35,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "pooling": "mean",
        }

        encoder = build_encoder("transformer", config_dict)

        assert isinstance(encoder, TransformerEncoder)
        assert encoder.config.d_input == 35
        assert encoder.config.d_model == 64

    def test_build_encoder_unknown_raises(self):
        """build_encoder should raise for unknown encoder name."""
        with pytest.raises(ValueError, match="Unknown encoder 'invalid'"):
            build_encoder("invalid", {})

    def test_build_encoder_with_defaults(self):
        """build_encoder should work with minimal config using defaults."""
        # TransformerConfig has defaults for most fields
        config_dict = {
            "d_input": 35,
        }

        encoder = build_encoder("transformer", config_dict)
        assert isinstance(encoder, TransformerEncoder)
        assert encoder.config.d_input == 35
        # Should use default d_model
        assert encoder.config.d_model == 128

    def test_get_encoder_config_class(self):
        """get_encoder_config_class should return correct config class."""
        config_cls = get_encoder_config_class("transformer")
        assert config_cls is TransformerConfig

    def test_get_encoder_config_class_unknown_raises(self):
        """get_encoder_config_class should raise for unknown encoder."""
        with pytest.raises(ValueError, match="Unknown encoder"):
            get_encoder_config_class("nonexistent")

    def test_built_encoder_produces_output(self):
        """Built encoder should produce output of correct shape."""
        config_dict = {
            "d_input": 35,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "pooling": "mean",
        }

        encoder = build_encoder("transformer", config_dict)

        # Test forward pass
        x = torch.randn(2, 48, 35)
        mask = torch.ones(2, 48, dtype=torch.bool)

        output = encoder(x, mask)

        # With mean pooling, output should be (B, d_model)
        assert output.shape == (2, 64)

    def test_encoder_registry_contains_smart(self):
        """Encoder registry should contain SMART encoder."""
        assert "smart" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY["smart"] is SMARTEncoder

    def test_encoder_config_registry_contains_smart(self):
        """Config registry should contain SMART config."""
        assert "smart" in ENCODER_CONFIG_REGISTRY
        assert ENCODER_CONFIG_REGISTRY["smart"] is SMARTEncoderConfig

    def test_build_encoder_creates_smart(self):
        """build_encoder should create SMARTEncoder correctly."""
        config_dict = {
            "d_input": 35,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "pooling": "query",
        }

        encoder = build_encoder("smart", config_dict)

        assert isinstance(encoder, SMARTEncoder)
        assert encoder.config.d_input == 35
        assert encoder.config.d_model == 32

    def test_built_smart_encoder_produces_output(self):
        """Built SMART encoder should produce output of correct shape."""
        config_dict = {
            "d_input": 35,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "pooling": "query",
        }

        encoder = build_encoder("smart", config_dict)

        # Test forward pass
        x = torch.randn(2, 48, 35)
        obs_mask = torch.ones(2, 48, 35, dtype=torch.bool)

        output = encoder(x, mask=obs_mask)

        # With query pooling, output should be (B, V*d_model)
        assert output.shape == (2, 35 * 32)


class TestTaskHeadFactory:
    """Tests for task head factory functions."""

    def test_task_head_registry_contains_mlp(self):
        """Task head registry should contain mlp and linear."""
        assert "mlp" in TASK_HEAD_REGISTRY
        assert "linear" in TASK_HEAD_REGISTRY
        assert TASK_HEAD_REGISTRY["mlp"] is MLPTaskHead
        assert TASK_HEAD_REGISTRY["linear"] is LinearTaskHead

    def test_get_available_task_heads(self):
        """get_available_task_heads should return list of names."""
        heads = get_available_task_heads()
        assert isinstance(heads, list)
        assert "mlp" in heads
        assert "linear" in heads

    def test_build_task_head_mlp(self):
        """build_task_head should create MLPTaskHead correctly."""
        config = TaskHeadConfig(
            name="mlp",
            task_name="mortality_24h",
            task_type="binary",
            n_classes=None,
            input_dim=64,
            hidden_dims=[32],
        )

        head = build_task_head(config)

        assert isinstance(head, MLPTaskHead)
        assert head.config.task_name == "mortality_24h"

    def test_build_task_head_linear(self):
        """build_task_head should create LinearTaskHead correctly."""
        config = TaskHeadConfig(
            name="linear",
            task_name="mortality_24h",
            task_type="binary",
            n_classes=None,
            input_dim=64,
            hidden_dims=[],  # Not used for linear
        )

        head = build_task_head(config)

        assert isinstance(head, LinearTaskHead)

    def test_build_task_head_unknown_raises(self):
        """build_task_head should raise for unknown head name."""
        config = TaskHeadConfig(
            name="unknown_head",
            task_name="test",
            task_type="binary",
            n_classes=None,
            input_dim=64,
            hidden_dims=[],
        )

        with pytest.raises(ValueError, match="Unknown task head"):
            build_task_head(config)

    def test_build_task_head_from_dict(self):
        """build_task_head_from_dict should create head from dictionary."""
        config_dict = {
            "name": "mlp",
            "task_name": "mortality_24h",
            "task_type": "binary",
            "n_classes": None,
            "input_dim": 64,
            "hidden_dims": [32],
        }

        head = build_task_head_from_dict(config_dict)

        assert isinstance(head, MLPTaskHead)

    def test_build_task_head_from_dict_handles_list_config(self):
        """build_task_head_from_dict should handle non-list hidden_dims."""
        # Simulate OmegaConf ListConfig by using tuple
        config_dict = {
            "name": "mlp",
            "task_name": "test",
            "task_type": "binary",
            "n_classes": None,
            "input_dim": 64,
            "hidden_dims": (32, 16),  # Tuple instead of list
        }

        head = build_task_head_from_dict(config_dict)
        assert isinstance(head, MLPTaskHead)

    def test_built_task_head_produces_output(self):
        """Built task head should produce output of correct shape."""
        config = TaskHeadConfig(
            name="mlp",
            task_name="mortality_24h",
            task_type="binary",
            n_classes=None,
            input_dim=64,
            hidden_dims=[32],
        )

        head = build_task_head(config)

        # Test forward pass
        x = torch.randn(4, 64)
        output = head(x)

        # Task heads return dict with 'logits' and 'probs'
        assert isinstance(output, dict)
        assert "logits" in output
        assert "probs" in output
        # Binary classification uses 2 outputs for CrossEntropy
        assert output["logits"].shape == (4, 2)

    def test_build_multiclass_task_head(self):
        """Building multiclass task head should output correct dimensions."""
        config = TaskHeadConfig(
            name="mlp",
            task_name="phenotype",
            task_type="multiclass",
            n_classes=10,
            input_dim=64,
            hidden_dims=[32],
        )

        head = build_task_head(config)
        x = torch.randn(4, 64)
        output = head(x)

        # Task heads return dict with 'logits' and 'probs'
        assert isinstance(output, dict)
        assert output["logits"].shape == (4, 10)


class TestSSLObjectiveFactory:
    """Tests for SSL objective factory functions."""

    def test_ssl_registry_contains_mae(self):
        """SSL registry should contain MAE objective."""
        assert "mae" in SSL_REGISTRY
        assert SSL_REGISTRY["mae"] is MAEObjective

    def test_config_registry_contains_mae(self):
        """Config registry should contain MAE config."""
        assert "mae" in CONFIG_REGISTRY
        assert CONFIG_REGISTRY["mae"] is MAEConfig

    def test_build_ssl_objective_mae(self):
        """build_ssl_objective should create MAEObjective correctly."""
        # Create encoder first
        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        # Create MAE config
        mae_config = MAEConfig(
            name="mae",
            mask_ratio=0.15,
            mask_strategy="random",
        )

        ssl_objective = build_ssl_objective(encoder, mae_config)

        assert isinstance(ssl_objective, MAEObjective)
        assert ssl_objective.config.mask_ratio == 0.15

    def test_build_ssl_objective_unknown_raises(self):
        """build_ssl_objective should raise for unknown objective name."""

        class FakeConfig:
            name = "unknown_objective"

        encoder = nn.Linear(10, 10)

        with pytest.raises(ValueError, match="Unknown SSL objective"):
            build_ssl_objective(encoder, FakeConfig())

    def test_get_ssl_config_class(self):
        """get_ssl_config_class should return correct config class."""
        config_cls = get_ssl_config_class("mae")
        assert config_cls is MAEConfig

    def test_get_ssl_config_class_unknown_raises(self):
        """get_ssl_config_class should raise for unknown objective."""
        with pytest.raises(ValueError, match="Unknown SSL objective"):
            get_ssl_config_class("nonexistent")

    def test_built_ssl_objective_forward_pass(self):
        """Built SSL objective should perform forward pass correctly."""
        encoder_config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = TransformerEncoder(encoder_config)

        mae_config = MAEConfig(
            name="mae",
            mask_ratio=0.15,
            mask_strategy="random",
        )

        ssl_objective = build_ssl_objective(encoder, mae_config)

        # Test forward pass - MAEObjective takes x and obs_mask as separate args
        x = torch.randn(2, 48, 35)
        obs_mask = torch.ones(2, 48, 35, dtype=torch.bool)

        loss, metrics = ssl_objective(x, obs_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert isinstance(metrics, dict)

    def test_ssl_registry_contains_smart(self):
        """SSL registry should contain SMART objective."""
        assert "smart" in SSL_REGISTRY
        assert SSL_REGISTRY["smart"] is SMARTObjective

    def test_config_registry_contains_smart(self):
        """Config registry should contain SMART config."""
        assert "smart" in CONFIG_REGISTRY
        assert CONFIG_REGISTRY["smart"] is SMARTSSLConfig

    def test_build_ssl_objective_smart(self):
        """build_ssl_objective should create SMARTObjective correctly."""
        # SMART requires SMARTEncoder with pooling=none
        encoder_config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = SMARTEncoder(encoder_config)

        smart_config = SMARTSSLConfig(
            name="smart",
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
        )

        ssl_objective = build_ssl_objective(encoder, smart_config)

        assert isinstance(ssl_objective, SMARTObjective)
        assert ssl_objective.config.max_mask_ratio == 0.75

    def test_get_ssl_config_class_smart(self):
        """get_ssl_config_class should return SMART config class."""
        config_cls = get_ssl_config_class("smart")
        assert config_cls is SMARTSSLConfig

    def test_built_smart_ssl_objective_forward_pass(self):
        """Built SMART SSL objective should perform forward pass correctly."""
        encoder_config = SMARTEncoderConfig(
            d_input=35,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pooling="none",
        )
        encoder = SMARTEncoder(encoder_config)

        smart_config = SMARTSSLConfig(
            name="smart",
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
        )

        ssl_objective = build_ssl_objective(encoder, smart_config)

        # Test forward pass
        x = torch.randn(2, 48, 35)
        obs_mask = torch.ones(2, 48, 35, dtype=torch.bool)

        loss, metrics = ssl_objective(x, obs_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert isinstance(metrics, dict)
        assert "smart_loss" in metrics


class TestFactoryIntegration:
    """Integration tests combining multiple factories."""

    def test_build_complete_model_pipeline(self):
        """Test building a complete encoder + task head pipeline."""
        # Build encoder
        encoder = build_encoder(
            "transformer",
            {
                "d_input": 35,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "pooling": "mean",
            },
        )

        # Build task head
        head = build_task_head_from_dict(
            {
                "name": "mlp",
                "task_name": "mortality_24h",
                "task_type": "binary",
                "n_classes": None,
                "input_dim": 64,  # Matches encoder d_model
                "hidden_dims": [32],
            }
        )

        # Test forward pass
        x = torch.randn(4, 48, 35)
        mask = torch.ones(4, 48, dtype=torch.bool)

        embeddings = encoder(x, mask)
        output = head(embeddings)

        assert embeddings.shape == (4, 64)
        # Task heads return dict with 'logits'
        assert output["logits"].shape == (4, 2)  # Binary uses 2 outputs

    def test_build_complete_smart_ssl_pipeline(self):
        """Test building a complete SMART encoder + SSL objective pipeline."""
        # Build SMART encoder with pooling=none for SSL
        encoder = build_encoder(
            "smart",
            {
                "d_input": 35,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "pooling": "none",
            },
        )

        # Build SMART SSL objective
        smart_config = SMARTSSLConfig(
            name="smart",
            min_mask_ratio=0.0,
            max_mask_ratio=0.75,
        )
        ssl_objective = build_ssl_objective(encoder, smart_config)

        # Test forward pass
        x = torch.randn(4, 48, 35)
        obs_mask = torch.rand(4, 48, 35) > 0.3

        loss, metrics = ssl_objective(x, obs_mask)

        assert torch.isfinite(loss)
        assert "smart_loss" in metrics
        assert "smart_mask_ratio_mean" in metrics

    def test_build_smart_encoder_for_downstream(self):
        """Test building SMART encoder for downstream task with query pooling."""
        # Build SMART encoder with query pooling for downstream
        encoder = build_encoder(
            "smart",
            {
                "d_input": 35,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "pooling": "query",
            },
        )

        # Build task head
        head = build_task_head_from_dict(
            {
                "name": "mlp",
                "task_name": "mortality_24h",
                "task_type": "binary",
                "n_classes": None,
                "input_dim": 35 * 32,  # V * d_model for query pooling
                "hidden_dims": [128],
            }
        )

        # Test forward pass
        x = torch.randn(4, 48, 35)
        obs_mask = torch.rand(4, 48, 35) > 0.3

        embeddings = encoder(x, mask=obs_mask)
        output = head(embeddings)

        assert embeddings.shape == (4, 35 * 32)
        assert output["logits"].shape == (4, 2)
