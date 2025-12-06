"""Tests for the SSL pretraining module.

Tests cover:
- Learning rate warmup schedule
- Optimizer and scheduler configuration
"""

import math

import pytest
import torch
from omegaconf import DictConfig, OmegaConf


class TestLRWarmup:
    """Tests for learning rate warmup schedule."""
    
    def test_warmup_starts_nonzero(self):
        """Test that LR warmup doesn't start at zero."""
        from slices.training.pretrain_module import SSLPretrainModule
        
        # Create minimal config
        config = OmegaConf.create({
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
                "mask_value": 0.0,
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
        })
        
        module = SSLPretrainModule(config)
        opt_config = module.configure_optimizers()
        
        # Get the scheduler
        scheduler = opt_config["lr_scheduler"]["scheduler"]
        
        # Check LR at epoch 0 (should NOT be zero)
        epoch_0_lr = scheduler.get_last_lr()[0]
        
        # With warmup_epochs=10, epoch 0 should have lr = base_lr * (1/10) = 1e-4
        expected_lr = 1e-3 * (1 / 10)
        assert epoch_0_lr == pytest.approx(expected_lr, rel=1e-6), (
            f"Epoch 0 LR should be {expected_lr}, got {epoch_0_lr}"
        )
    
    def test_warmup_schedule_values(self):
        """Test warmup schedule produces expected LR values."""
        from slices.training.pretrain_module import SSLPretrainModule
        
        # Create minimal config
        config = OmegaConf.create({
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
                "mask_value": 0.0,
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
        })
        
        module = SSLPretrainModule(config)
        opt_config = module.configure_optimizers()
        
        optimizer = opt_config["optimizer"]
        scheduler = opt_config["lr_scheduler"]["scheduler"]
        
        base_lr = 1e-3
        warmup_epochs = 10
        
        # Test warmup phase
        for epoch in range(warmup_epochs):
            expected_lr = base_lr * (epoch + 1) / warmup_epochs
            actual_lr = scheduler.get_last_lr()[0]
            
            assert actual_lr == pytest.approx(expected_lr, rel=1e-6), (
                f"Epoch {epoch}: expected LR {expected_lr}, got {actual_lr}"
            )
            
            # Step to next epoch
            scheduler.step()
        
        # At epoch=warmup_epochs, should be at full LR (start of cosine decay)
        actual_lr = scheduler.get_last_lr()[0]
        assert actual_lr == pytest.approx(base_lr, rel=1e-6), (
            f"Epoch {warmup_epochs}: expected full LR {base_lr}, got {actual_lr}"
        )
    
    def test_warmup_reaches_full_lr(self):
        """Test that warmup reaches full learning rate at end of warmup phase."""
        from slices.training.pretrain_module import SSLPretrainModule
        
        config = OmegaConf.create({
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
                "mask_value": 0.0,
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
        })
        
        module = SSLPretrainModule(config)
        opt_config = module.configure_optimizers()
        
        scheduler = opt_config["lr_scheduler"]["scheduler"]
        
        # Step through warmup
        for _ in range(5):
            scheduler.step()
        
        # Should be at full LR now
        actual_lr = scheduler.get_last_lr()[0]
        assert actual_lr == pytest.approx(0.001, rel=1e-6), (
            f"After warmup: expected LR 0.001, got {actual_lr}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
