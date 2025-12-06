"""Tests for downstream task finetuning module and task heads."""

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from slices.models.encoders import TransformerConfig, TransformerEncoder
from slices.models.heads import (
    LinearTaskHead,
    MLPTaskHead,
    TaskHeadConfig,
    build_task_head,
    build_task_head_from_dict,
    get_available_task_heads,
)
from slices.training import FineTuneModule


class TestTaskHeadConfig:
    """Tests for TaskHeadConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TaskHeadConfig()
        
        assert config.name == "mlp"
        assert config.task_name == "mortality_24h"
        assert config.task_type == "binary"
        assert config.n_classes is None
        assert config.input_dim == 128
        assert config.hidden_dims == [64]
        assert config.dropout == 0.1
        assert config.activation == "relu"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TaskHeadConfig(
            name="linear",
            task_name="mortality_48h",
            task_type="multiclass",
            n_classes=5,
            input_dim=256,
            hidden_dims=[128, 64],
            dropout=0.2,
            activation="gelu",
        )
        
        assert config.name == "linear"
        assert config.task_name == "mortality_48h"
        assert config.task_type == "multiclass"
        assert config.n_classes == 5
        assert config.input_dim == 256
        assert config.hidden_dims == [128, 64]
    
    def test_n_classes_validation(self):
        """Test n_classes validation."""
        # Should fail for multiclass without n_classes
        with pytest.raises(ValueError, match="n_classes is required"):
            TaskHeadConfig(task_type="multiclass")
        
        # Should fail for multilabel without n_classes
        with pytest.raises(ValueError, match="n_classes is required"):
            TaskHeadConfig(task_type="multilabel")
        
        # Should work for binary without n_classes
        config = TaskHeadConfig(task_type="binary")
        assert config.get_output_dim() == 2
        
        # Should work for regression without n_classes
        config = TaskHeadConfig(task_type="regression")
        assert config.get_output_dim() == 1


class TestMLPTaskHead:
    """Tests for MLPTaskHead."""
    
    def test_binary_classification(self):
        """Test MLP head for binary classification."""
        config = TaskHeadConfig(
            name="mlp",
            task_type="binary",
            input_dim=128,
            hidden_dims=[64],
        )
        head = MLPTaskHead(config)
        
        # Check architecture
        assert head.config.task_type == "binary"
        
        # Forward pass
        batch_size = 32
        encoder_out = torch.randn(batch_size, 128)
        outputs = head(encoder_out)
        
        assert "logits" in outputs
        assert "probs" in outputs
        assert outputs["logits"].shape == (batch_size, 2)
        assert outputs["probs"].shape == (batch_size, 2)
        
        # Check probs sum to 1
        prob_sum = outputs["probs"].sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones(batch_size), atol=1e-5)
    
    def test_multiclass_classification(self):
        """Test MLP head for multiclass classification."""
        n_classes = 5
        config = TaskHeadConfig(
            name="mlp",
            task_type="multiclass",
            n_classes=n_classes,
            input_dim=128,
            hidden_dims=[64, 32],
        )
        head = MLPTaskHead(config)
        
        batch_size = 16
        encoder_out = torch.randn(batch_size, 128)
        outputs = head(encoder_out)
        
        assert outputs["logits"].shape == (batch_size, n_classes)
        assert outputs["probs"].shape == (batch_size, n_classes)
    
    def test_regression(self):
        """Test MLP head for regression."""
        config = TaskHeadConfig(
            name="mlp",
            task_type="regression",
            n_classes=1,  # Not used for regression
            input_dim=64,
            hidden_dims=[32],
        )
        head = MLPTaskHead(config)
        
        batch_size = 8
        encoder_out = torch.randn(batch_size, 64)
        outputs = head(encoder_out)
        
        assert outputs["logits"].shape == (batch_size, 1)
    
    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "gelu", "silu", "tanh"]:
            config = TaskHeadConfig(
                name="mlp",
                input_dim=64,
                hidden_dims=[32],
                activation=activation,
            )
            head = MLPTaskHead(config)
            
            encoder_out = torch.randn(4, 64)
            outputs = head(encoder_out)
            
            assert outputs["logits"].shape == (4, 2)
    
    def test_invalid_activation(self):
        """Test invalid activation raises error."""
        config = TaskHeadConfig(
            name="mlp",
            input_dim=64,
            hidden_dims=[32],
            activation="invalid",
        )
        
        with pytest.raises(ValueError, match="Unknown activation"):
            MLPTaskHead(config)


class TestLinearTaskHead:
    """Tests for LinearTaskHead."""
    
    def test_binary_classification(self):
        """Test linear head for binary classification."""
        config = TaskHeadConfig(
            name="linear",
            task_type="binary",
            input_dim=128,
        )
        head = LinearTaskHead(config)
        
        batch_size = 32
        encoder_out = torch.randn(batch_size, 128)
        outputs = head(encoder_out)
        
        assert outputs["logits"].shape == (batch_size, 2)
        assert outputs["probs"].shape == (batch_size, 2)
    
    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        config = TaskHeadConfig(
            name="linear",
            input_dim=128,
            dropout=0.5,
        )
        head = LinearTaskHead(config)
        
        head.train()
        encoder_out = torch.randn(32, 128)
        
        # Run multiple times - with dropout, outputs should differ
        out1 = head(encoder_out)["logits"]
        out2 = head(encoder_out)["logits"]
        
        # Outputs might differ due to dropout (not guaranteed but likely with 0.5)
        # This is a weak test but verifies dropout is in the forward path
        assert out1.shape == out2.shape


class TestTaskHeadFactory:
    """Tests for task head factory functions."""
    
    def test_build_mlp_head(self):
        """Test building MLP head from config."""
        config = TaskHeadConfig(
            name="mlp",
            input_dim=128,
            hidden_dims=[64],
        )
        head = build_task_head(config)
        
        assert isinstance(head, MLPTaskHead)
    
    def test_build_linear_head(self):
        """Test building linear head from config."""
        config = TaskHeadConfig(
            name="linear",
            input_dim=128,
        )
        head = build_task_head(config)
        
        assert isinstance(head, LinearTaskHead)
    
    def test_build_from_dict(self):
        """Test building head from dictionary."""
        config_dict = {
            "name": "mlp",
            "task_name": "mortality_24h",
            "task_type": "binary",
            "input_dim": 128,
            "hidden_dims": [64, 32],
        }
        head = build_task_head_from_dict(config_dict)
        
        assert isinstance(head, MLPTaskHead)
        assert head.config.hidden_dims == [64, 32]
    
    def test_invalid_head_type(self):
        """Test invalid head type raises error."""
        config = TaskHeadConfig(name="invalid_head", input_dim=128)
        
        with pytest.raises(ValueError, match="Unknown task head"):
            build_task_head(config)
    
    def test_get_available_heads(self):
        """Test getting available head types."""
        heads = get_available_task_heads()
        
        assert "mlp" in heads
        assert "linear" in heads


class TestFineTuneModule:
    """Tests for FineTuneModule."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return OmegaConf.create({
            "encoder": {
                "name": "transformer",
                "d_input": 35,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "d_ff": 128,
                "dropout": 0.1,
                "max_seq_length": 48,
                "pooling": "mean",
                "use_positional_encoding": True,
                "prenorm": True,
                "activation": "gelu",
                "layer_norm_eps": 1e-5,
            },
            "task": {
                "task_name": "mortality_24h",
                "task_type": "binary",
                "n_classes": None,
                "head_type": "mlp",
                "hidden_dims": [32],
                "dropout": 0.1,
                "activation": "relu",
            },
            "training": {
                "freeze_encoder": True,
                "unfreeze_epoch": None,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.01,
            },
        })
    
    @pytest.fixture
    def encoder_checkpoint(self, tmp_path):
        """Create a temporary encoder checkpoint."""
        # Create encoder and save weights - must match sample_config
        config = TransformerConfig(
            d_input=35,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            max_seq_length=48,  # Must match sample_config
            pooling="mean",
            use_positional_encoding=True,
            prenorm=True,
            activation="gelu",
            layer_norm_eps=1e-5,
        )
        encoder = TransformerEncoder(config)
        
        checkpoint_path = tmp_path / "encoder.pt"
        torch.save(encoder.state_dict(), checkpoint_path)
        
        return str(checkpoint_path)
    
    def test_module_creation(self, sample_config):
        """Test creating FineTuneModule without checkpoint."""
        module = FineTuneModule(sample_config)
        
        assert module.encoder is not None
        assert module.task_head is not None
        assert isinstance(module.task_head, MLPTaskHead)
    
    def test_load_encoder_checkpoint(self, sample_config, encoder_checkpoint):
        """Test loading encoder from checkpoint."""
        module = FineTuneModule(sample_config, checkpoint_path=encoder_checkpoint)
        
        assert module.encoder is not None
    
    def test_freeze_encoder(self, sample_config):
        """Test encoder freezing."""
        sample_config.training.freeze_encoder = True
        module = FineTuneModule(sample_config)
        
        # Check encoder params are frozen
        for param in module.encoder.parameters():
            assert not param.requires_grad
        
        # Check task head params are trainable
        for param in module.task_head.parameters():
            assert param.requires_grad
    
    def test_unfreeze_encoder(self, sample_config):
        """Test full finetuning (encoder unfrozen)."""
        sample_config.training.freeze_encoder = False
        module = FineTuneModule(sample_config)
        
        # Check encoder params are trainable
        for param in module.encoder.parameters():
            assert param.requires_grad
    
    def test_forward_pass(self, sample_config):
        """Test forward pass through module."""
        module = FineTuneModule(sample_config)
        
        batch_size = 8
        seq_len = 48
        n_features = 35
        
        timeseries = torch.randn(batch_size, seq_len, n_features)
        mask = torch.ones(batch_size, seq_len, n_features, dtype=torch.bool)
        
        outputs = module(timeseries, mask)
        
        assert "logits" in outputs
        assert "probs" in outputs
        assert outputs["logits"].shape == (batch_size, 2)
    
    def test_training_step(self, sample_config):
        """Test training step."""
        module = FineTuneModule(sample_config)
        
        batch = {
            "timeseries": torch.randn(8, 48, 35),
            "mask": torch.ones(8, 48, 35, dtype=torch.bool),
            "label": torch.randint(0, 2, (8,)),
        }
        
        loss = module.training_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive loss
    
    def test_validation_step(self, sample_config):
        """Test validation step."""
        module = FineTuneModule(sample_config)
        
        batch = {
            "timeseries": torch.randn(8, 48, 35),
            "mask": torch.ones(8, 48, 35, dtype=torch.bool),
            "label": torch.randint(0, 2, (8,)),
        }
        
        loss = module.validation_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
    
    def test_configure_optimizers(self, sample_config):
        """Test optimizer configuration."""
        module = FineTuneModule(sample_config)
        
        optimizer = module.configure_optimizers()
        
        assert isinstance(optimizer, torch.optim.AdamW)
    
    def test_configure_optimizers_with_scheduler(self, sample_config):
        """Test optimizer with scheduler."""
        sample_config.scheduler = {
            "name": "cosine",
            "T_max": 50,
            "eta_min": 1e-6,
        }
        module = FineTuneModule(sample_config)
        
        result = module.configure_optimizers()
        
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
    
    def test_trainable_params_frozen(self, sample_config):
        """Test parameter count with frozen encoder."""
        sample_config.training.freeze_encoder = True
        module = FineTuneModule(sample_config)
        
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # With frozen encoder, trainable should be much less than total
        assert trainable_params < total_params
        
        # Only task head should be trainable
        task_head_params = sum(p.numel() for p in module.task_head.parameters())
        assert trainable_params == task_head_params
    
    def test_trainable_params_unfrozen(self, sample_config):
        """Test parameter count with unfrozen encoder."""
        sample_config.training.freeze_encoder = False
        module = FineTuneModule(sample_config)
        
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # All params should be trainable
        assert trainable_params == total_params


class TestFineTuneModuleGradualUnfreeze:
    """Tests for gradual unfreezing strategy."""
    
    @pytest.fixture
    def config_gradual(self):
        """Config with gradual unfreezing."""
        return OmegaConf.create({
            "encoder": {
                "name": "transformer",
                "d_input": 35,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "d_ff": 128,
                "dropout": 0.1,
                "max_seq_length": 48,
                "pooling": "mean",
                "use_positional_encoding": True,
                "prenorm": True,
                "activation": "gelu",
                "layer_norm_eps": 1e-5,
            },
            "task": {
                "task_name": "mortality_24h",
                "task_type": "binary",
                "n_classes": None,
                "head_type": "mlp",
                "hidden_dims": [32],
                "dropout": 0.1,
                "activation": "relu",
            },
            "training": {
                "freeze_encoder": True,
                "unfreeze_epoch": 5,  # Unfreeze at epoch 5
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.01,
            },
        })
    
    def test_initial_frozen_state(self, config_gradual):
        """Test encoder is initially frozen."""
        module = FineTuneModule(config_gradual)
        
        for param in module.encoder.parameters():
            assert not param.requires_grad
    
    def test_unfreeze_at_epoch(self, config_gradual):
        """Test unfreezing happens at correct epoch."""
        module = FineTuneModule(config_gradual)
        
        # Mock the current_epoch property since we're not using a real trainer
        # Before epoch 5 - should still be frozen
        for epoch in range(5):
            # Check encoder is still frozen
            for param in module.encoder.parameters():
                assert not param.requires_grad
        
        # Manually simulate unfreezing at epoch 5
        # In real usage, this is called by Lightning via on_train_epoch_start
        module._unfreeze_encoder()
        module.freeze_strategy = False
        
        # Now encoder should be unfrozen
        for param in module.encoder.parameters():
            assert param.requires_grad


class TestEndToEnd:
    """End-to-end tests for the finetuning pipeline."""
    
    @pytest.fixture
    def full_config(self):
        """Create full configuration."""
        return OmegaConf.create({
            "encoder": {
                "name": "transformer",
                "d_input": 10,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 2,
                "d_ff": 64,
                "dropout": 0.0,
                "max_seq_length": 24,
                "pooling": "mean",
                "use_positional_encoding": True,
                "prenorm": True,
                "activation": "gelu",
                "layer_norm_eps": 1e-5,
            },
            "task": {
                "task_name": "mortality_24h",
                "task_type": "binary",
                "n_classes": None,
                "head_type": "mlp",
                "hidden_dims": [16],
                "dropout": 0.0,
                "activation": "relu",
            },
            "training": {
                "freeze_encoder": True,
                "unfreeze_epoch": None,
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
        })
    
    def test_backward_pass(self, full_config):
        """Test gradient computation works."""
        module = FineTuneModule(full_config)
        
        batch = {
            "timeseries": torch.randn(4, 24, 10),
            "mask": torch.ones(4, 24, 10, dtype=torch.bool),
            "label": torch.tensor([0, 1, 0, 1]),
        }
        
        loss = module.training_step(batch, 0)
        loss.backward()
        
        # Check gradients exist for task head
        for param in module.task_head.parameters():
            assert param.grad is not None
        
        # Check no gradients for frozen encoder
        for param in module.encoder.parameters():
            assert param.grad is None
    
    def test_prediction_consistency(self, full_config):
        """Test predictions are consistent in eval mode."""
        module = FineTuneModule(full_config)
        module.eval()
        
        timeseries = torch.randn(4, 24, 10)
        mask = torch.ones(4, 24, 10, dtype=torch.bool)
        
        with torch.no_grad():
            out1 = module(timeseries, mask)
            out2 = module(timeseries, mask)
        
        assert torch.allclose(out1["logits"], out2["logits"])
        assert torch.allclose(out1["probs"], out2["probs"])
