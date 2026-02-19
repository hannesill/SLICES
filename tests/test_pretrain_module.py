"""Tests for the SSL pretraining module.

Tests cover:
- Module initialization and configuration
- Forward pass and training/validation steps
- Optimizer configuration (adam, adamw, sgd)
- Scheduler configuration (cosine, step, plateau, warmup_cosine)
- Error handling for invalid configurations
- Encoder access and checkpoint saving
"""

import pytest
import torch
from omegaconf import OmegaConf
from slices.training.pretrain_module import SSLPretrainModule


@pytest.fixture
def minimal_config():
    """Create minimal config for testing."""
    return OmegaConf.create(
        {
            "encoder": {
                "name": "transformer",
                "d_input": 9,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 4,
                "d_ff": 64,
                "max_seq_length": 48,
                "pooling": "none",
                "dropout": 0.1,
                "use_positional_encoding": True,
                "prenorm": True,
                "activation": "gelu",
                "layer_norm_eps": 1e-5,
            },
            "ssl": {
                "name": "mae",
                "mask_ratio": 0.15,
                "mask_strategy": "random",
                "decoder_d_model": 16,
                "decoder_n_layers": 1,
                "decoder_n_heads": 2,
                "decoder_d_ff": 32,
                "decoder_dropout": 0.1,
                "loss_on_observed_only": True,
                "norm_target": False,
                "min_block_size": 3,
                "max_block_size": 10,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.01,
            },
        }
    )


class TestSSLPretrainModuleInitialization:
    """Tests for module initialization."""

    def test_module_initialization(self, minimal_config):
        """Test that module initializes correctly."""
        module = SSLPretrainModule(minimal_config)

        assert module.encoder is not None
        assert module.ssl_objective is not None
        assert module.config == minimal_config

    def test_encoder_type(self, minimal_config):
        """Test that encoder is correct type."""
        from slices.models.encoders import TransformerEncoder

        module = SSLPretrainModule(minimal_config)
        assert isinstance(module.encoder, TransformerEncoder)

    def test_ssl_objective_type(self, minimal_config):
        """Test that SSL objective is correct type."""
        from slices.models.pretraining import MAEObjective

        module = SSLPretrainModule(minimal_config)
        assert isinstance(module.ssl_objective, MAEObjective)

    def test_hyperparameters_saved(self, minimal_config):
        """Test that hyperparameters are saved."""
        module = SSLPretrainModule(minimal_config)
        # Lightning's save_hyperparameters should save the config
        assert hasattr(module, "hparams")


class TestSSLPretrainModuleForward:
    """Tests for forward pass."""

    def test_forward_pass(self, minimal_config):
        """Test forward pass returns loss and metrics."""
        module = SSLPretrainModule(minimal_config)

        batch_size, seq_len, n_features = 4, 48, 9
        timeseries = torch.randn(batch_size, seq_len, n_features)
        mask = torch.ones(batch_size, seq_len, n_features, dtype=torch.bool)

        loss, metrics = module(timeseries, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert isinstance(metrics, dict)
        assert "mae_loss" in metrics

    def test_forward_with_missing_values(self, minimal_config):
        """Test forward pass with missing values in observation mask."""
        module = SSLPretrainModule(minimal_config)

        batch_size, seq_len, n_features = 4, 48, 9
        timeseries = torch.randn(batch_size, seq_len, n_features)
        # 30% missing values
        mask = torch.rand(batch_size, seq_len, n_features) > 0.3

        loss, metrics = module(timeseries, mask)

        assert torch.isfinite(loss)
        assert "mae_obs_ratio" in metrics


class TestSSLPretrainModuleTraining:
    """Tests for training and validation steps."""

    def test_training_step(self, minimal_config):
        """Test training step returns loss."""
        module = SSLPretrainModule(minimal_config)

        batch = {
            "timeseries": torch.randn(4, 48, 9),
            "mask": torch.ones(4, 48, 9, dtype=torch.bool),
        }

        loss = module.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_validation_step(self, minimal_config):
        """Test validation step returns loss."""
        module = SSLPretrainModule(minimal_config)

        batch = {
            "timeseries": torch.randn(4, 48, 9),
            "mask": torch.ones(4, 48, 9, dtype=torch.bool),
        }

        loss = module.validation_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)

    def test_backward_pass(self, minimal_config):
        """Test that gradients are computed correctly."""
        module = SSLPretrainModule(minimal_config)

        batch = {
            "timeseries": torch.randn(4, 48, 9),
            "mask": torch.ones(4, 48, 9, dtype=torch.bool),
        }

        loss = module.training_step(batch, 0)
        loss.backward()

        # Check gradients exist for encoder and decoder
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestOptimizerConfiguration:
    """Tests for optimizer configuration."""

    def test_adam_optimizer(self, minimal_config):
        """Test Adam optimizer configuration."""
        minimal_config.optimizer.name = "adam"
        module = SSLPretrainModule(minimal_config)

        optimizer = module.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)

    def test_adamw_optimizer(self, minimal_config):
        """Test AdamW optimizer configuration."""
        minimal_config.optimizer.name = "adamw"
        module = SSLPretrainModule(minimal_config)

        optimizer = module.configure_optimizers()

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_sgd_optimizer(self, minimal_config):
        """Test SGD optimizer configuration."""
        minimal_config.optimizer.name = "sgd"
        minimal_config.optimizer.momentum = 0.9
        module = SSLPretrainModule(minimal_config)

        optimizer = module.configure_optimizers()

        assert isinstance(optimizer, torch.optim.SGD)

    def test_invalid_optimizer(self, minimal_config):
        """Test that invalid optimizer raises error at init."""
        from pydantic import ValidationError

        minimal_config.optimizer.name = "invalid"
        with pytest.raises(ValidationError, match="optimizer name"):
            SSLPretrainModule(minimal_config)


class TestSchedulerConfiguration:
    """Tests for scheduler configuration."""

    def test_cosine_scheduler(self, minimal_config):
        """Test cosine annealing scheduler."""
        minimal_config.scheduler = {"name": "cosine", "T_max": 100, "eta_min": 1e-6}
        module = SSLPretrainModule(minimal_config)

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )

    def test_step_scheduler(self, minimal_config):
        """Test step scheduler."""
        minimal_config.scheduler = {"name": "step", "step_size": 30, "gamma": 0.1}
        module = SSLPretrainModule(minimal_config)

        result = module.configure_optimizers()

        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.StepLR,
        )

    def test_plateau_scheduler(self, minimal_config):
        """Test reduce on plateau scheduler."""
        minimal_config.scheduler = {
            "name": "plateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
        }
        module = SSLPretrainModule(minimal_config)

        result = module.configure_optimizers()

        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        )
        assert result["lr_scheduler"]["monitor"] == "val/loss"

    def test_no_scheduler(self, minimal_config):
        """Test that no scheduler returns just optimizer."""
        module = SSLPretrainModule(minimal_config)

        result = module.configure_optimizers()

        # Should return optimizer directly, not dict
        assert isinstance(result, torch.optim.Optimizer)

    def test_invalid_scheduler(self, minimal_config):
        """Test that invalid scheduler raises error at init."""
        from pydantic import ValidationError

        minimal_config.scheduler = {"name": "invalid"}
        with pytest.raises(ValidationError, match="scheduler name"):
            SSLPretrainModule(minimal_config)


class TestEncoderAccess:
    """Tests for encoder access methods."""

    def test_get_encoder(self, minimal_config):
        """Test get_encoder returns encoder."""
        module = SSLPretrainModule(minimal_config)

        encoder = module.get_encoder()

        assert encoder is module.encoder

    def test_save_encoder(self, minimal_config, tmp_path):
        """Test save_encoder saves weights."""
        module = SSLPretrainModule(minimal_config)

        save_path = tmp_path / "encoder.pt"
        module.save_encoder(str(save_path))

        assert save_path.exists()

        # Verify saved weights can be loaded
        state_dict = torch.load(save_path, weights_only=True)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0


class TestLRWarmup:
    """Tests for learning rate warmup schedule."""

    def test_warmup_starts_nonzero(self):
        """Test that LR warmup doesn't start at zero."""
        from slices.training.pretrain_module import SSLPretrainModule

        # Create minimal config
        config = OmegaConf.create(
            {
                "encoder": {
                    "name": "transformer",
                    "d_input": 9,
                    "d_model": 32,
                    "n_layers": 1,
                    "n_heads": 4,
                    "d_ff": 64,
                    "max_seq_length": 48,
                    "pooling": "none",
                    "dropout": 0.1,
                    "use_positional_encoding": True,
                    "prenorm": True,
                    "activation": "gelu",
                    "layer_norm_eps": 1e-5,
                },
                "ssl": {
                    "name": "mae",
                    "mask_ratio": 0.15,
                    "mask_strategy": "random",
                    "decoder_d_model": 16,
                    "decoder_n_layers": 1,
                    "decoder_n_heads": 2,
                    "decoder_d_ff": 32,
                    "decoder_dropout": 0.1,
                    "loss_on_observed_only": True,
                    "norm_target": False,
                    "min_block_size": 3,
                    "max_block_size": 10,
                },
                "optimizer": {
                    "name": "adamw",
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                },
                "scheduler": {
                    "name": "warmup_cosine",
                    "warmup_epochs": 10,
                    "max_epochs": 100,
                    "eta_min": 1e-6,
                },
            }
        )

        module = SSLPretrainModule(config)
        opt_config = module.configure_optimizers()

        # Get the scheduler
        scheduler = opt_config["lr_scheduler"]["scheduler"]

        # Check LR at epoch 0 (should NOT be zero)
        epoch_0_lr = scheduler.get_last_lr()[0]

        # With warmup_epochs=10, epoch 0 should have lr = base_lr * (1/10) = 1e-4
        expected_lr = 1e-3 * (1 / 10)
        assert epoch_0_lr == pytest.approx(
            expected_lr, rel=1e-6
        ), f"Epoch 0 LR should be {expected_lr}, got {epoch_0_lr}"

    def test_warmup_schedule_values(self):
        """Test warmup schedule produces expected LR values."""
        from slices.training.pretrain_module import SSLPretrainModule

        # Create minimal config
        config = OmegaConf.create(
            {
                "encoder": {
                    "name": "transformer",
                    "d_input": 9,
                    "d_model": 32,
                    "n_layers": 1,
                    "n_heads": 4,
                    "d_ff": 64,
                    "max_seq_length": 48,
                    "pooling": "none",
                    "dropout": 0.1,
                    "use_positional_encoding": True,
                    "prenorm": True,
                    "activation": "gelu",
                    "layer_norm_eps": 1e-5,
                },
                "ssl": {
                    "name": "mae",
                    "mask_ratio": 0.15,
                    "mask_strategy": "random",
                    "decoder_d_model": 16,
                    "decoder_n_layers": 1,
                    "decoder_n_heads": 2,
                    "decoder_d_ff": 32,
                    "decoder_dropout": 0.1,
                    "loss_on_observed_only": True,
                    "norm_target": False,
                    "min_block_size": 3,
                    "max_block_size": 10,
                },
                "optimizer": {
                    "name": "adamw",
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                },
                "scheduler": {
                    "name": "warmup_cosine",
                    "warmup_epochs": 10,
                    "max_epochs": 100,
                    "eta_min": 0.0,
                },
            }
        )

        module = SSLPretrainModule(config)
        opt_config = module.configure_optimizers()

        _optimizer = opt_config["optimizer"]  # Configured but scheduler is tested
        scheduler = opt_config["lr_scheduler"]["scheduler"]

        base_lr = 1e-3
        warmup_epochs = 10

        # Test warmup phase
        for epoch in range(warmup_epochs):
            expected_lr = base_lr * (epoch + 1) / warmup_epochs
            actual_lr = scheduler.get_last_lr()[0]

            assert actual_lr == pytest.approx(
                expected_lr, rel=1e-6
            ), f"Epoch {epoch}: expected LR {expected_lr}, got {actual_lr}"

            # Step to next epoch
            scheduler.step()

        # At epoch=warmup_epochs, should be at full LR (start of cosine decay)
        actual_lr = scheduler.get_last_lr()[0]
        assert actual_lr == pytest.approx(
            base_lr, rel=1e-6
        ), f"Epoch {warmup_epochs}: expected full LR {base_lr}, got {actual_lr}"

    def test_warmup_reaches_full_lr(self):
        """Test that warmup reaches full learning rate at end of warmup phase."""
        from slices.training.pretrain_module import SSLPretrainModule

        config = OmegaConf.create(
            {
                "encoder": {
                    "name": "transformer",
                    "d_input": 9,
                    "d_model": 32,
                    "n_layers": 1,
                    "n_heads": 4,
                    "d_ff": 64,
                    "max_seq_length": 48,
                    "pooling": "none",
                    "dropout": 0.1,
                    "use_positional_encoding": True,
                    "prenorm": True,
                    "activation": "gelu",
                    "layer_norm_eps": 1e-5,
                },
                "ssl": {
                    "name": "mae",
                    "mask_ratio": 0.15,
                    "mask_strategy": "random",
                    "decoder_d_model": 16,
                    "decoder_n_layers": 1,
                    "decoder_n_heads": 2,
                    "decoder_d_ff": 32,
                    "decoder_dropout": 0.1,
                    "loss_on_observed_only": True,
                    "norm_target": False,
                    "min_block_size": 3,
                    "max_block_size": 10,
                },
                "optimizer": {
                    "name": "adamw",
                    "lr": 0.001,
                    "weight_decay": 0.01,
                },
                "scheduler": {
                    "name": "warmup_cosine",
                    "warmup_epochs": 5,
                    "max_epochs": 50,
                    "eta_min": 0.0,
                },
            }
        )

        module = SSLPretrainModule(config)
        opt_config = module.configure_optimizers()

        scheduler = opt_config["lr_scheduler"]["scheduler"]

        # Step through warmup
        for _ in range(5):
            scheduler.step()

        # Should be at full LR now
        actual_lr = scheduler.get_last_lr()[0]
        assert actual_lr == pytest.approx(
            0.001, rel=1e-6
        ), f"After warmup: expected LR 0.001, got {actual_lr}"


class TestPretrainModuleConfigValidation:
    """Test that SSLPretrainModule rejects invalid optimizer/scheduler configs."""

    def test_typo_in_optimizer_config_rejected(self, minimal_config):
        from pydantic import ValidationError

        cfg = OmegaConf.to_container(minimal_config, resolve=True)
        cfg["optimizer"]["lerning_rate"] = 1e-3
        with pytest.raises(ValidationError, match="lerning_rate"):
            SSLPretrainModule(OmegaConf.create(cfg))

    def test_typo_in_scheduler_config_rejected(self, minimal_config):
        from pydantic import ValidationError

        cfg = OmegaConf.to_container(minimal_config, resolve=True)
        cfg["scheduler"] = {"name": "cosine", "t_max": 100}
        with pytest.raises(ValidationError, match="t_max"):
            SSLPretrainModule(OmegaConf.create(cfg))

    def test_no_scheduler_passes(self, minimal_config):
        """Module without scheduler should still pass validation."""
        module = SSLPretrainModule(minimal_config)
        assert module is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
