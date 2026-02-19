"""Tests for training utilities: optimizer/scheduler builders and checkpoint saving.

Tests cover:
- build_optimizer: Adam, AdamW, SGD, unknown raises error
- build_scheduler: cosine, step, plateau, warmup_cosine, none, unknown raises error
- save_encoder_checkpoint: v3 format, missing token, state dict loadability
- Supervised script save_encoder_weights: v3 format, loadable by FineTuneModule
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from slices.training import FineTuneModule
from slices.training.utils import (
    build_optimizer,
    build_scheduler,
    save_encoder_checkpoint,
)


class TestBuildOptimizer:
    """Tests for shared build_optimizer."""

    def _make_config(self, **kwargs):
        return OmegaConf.create(kwargs)

    def test_adam(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="adam", lr=0.001, weight_decay=0.0)
        opt = build_optimizer(model.parameters(), cfg)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="adamw", lr=0.001, weight_decay=0.01)
        opt = build_optimizer(model.parameters(), cfg)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="sgd", lr=0.01, momentum=0.9)
        opt = build_optimizer(model.parameters(), cfg)
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_optimizer_raises(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="rmsprop", lr=0.001)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(model.parameters(), cfg)


class TestBuildScheduler:
    """Tests for shared build_scheduler."""

    def _make_config(self, **kwargs):
        return OmegaConf.create(kwargs)

    def _make_optimizer(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="adam", lr=0.001)
        return build_optimizer(model.parameters(), cfg)

    def test_cosine_scheduler(self):
        optimizer = self._make_optimizer()
        sched_cfg = self._make_config(name="cosine", T_max=50, eta_min=1e-6)
        result = build_scheduler(optimizer, sched_cfg)
        assert result is not None
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )

    def test_step_scheduler(self):
        optimizer = self._make_optimizer()
        result = build_scheduler(
            optimizer,
            self._make_config(name="step", step_size=10, gamma=0.5),
        )
        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.StepLR,
        )

    def test_plateau_scheduler_has_monitor(self):
        optimizer = self._make_optimizer()
        result = build_scheduler(
            optimizer,
            self._make_config(name="plateau", patience=5),
        )
        assert result["lr_scheduler"]["monitor"] == "val/loss"

    def test_warmup_cosine_scheduler(self):
        optimizer = self._make_optimizer()
        result = build_scheduler(
            optimizer,
            self._make_config(name="warmup_cosine", warmup_epochs=5, max_epochs=50),
        )
        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.LambdaLR,
        )

    def test_none_scheduler(self):
        optimizer = self._make_optimizer()
        assert build_scheduler(optimizer, None) is None

    def test_unknown_scheduler_raises(self):
        optimizer = self._make_optimizer()
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(
                optimizer,
                self._make_config(name="exponential"),
            )


class TestSaveEncoderCheckpoint:
    """Tests for shared save_encoder_checkpoint."""

    def test_saves_and_loads_v3_format(self, tmp_path):
        """Checkpoint contains required v3 fields."""
        encoder = nn.Linear(4, 8)
        config = {"name": "linear", "d_input": 4, "d_model": 8}
        path = tmp_path / "encoder.pt"

        save_encoder_checkpoint(encoder, config, path)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert ckpt["version"] == 3
        assert ckpt["encoder_config"]["name"] == "linear"
        assert "encoder_state_dict" in ckpt
        assert "weight" in ckpt["encoder_state_dict"]

    def test_saves_missing_token(self, tmp_path):
        """Optional missing token is saved when provided."""
        encoder = nn.Linear(4, 8)
        config = {"name": "linear", "d_input": 4, "d_model": 8}
        missing_token = torch.randn(4)
        path = tmp_path / "encoder.pt"

        save_encoder_checkpoint(encoder, config, path, missing_token=missing_token, d_input=4)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert "missing_token" in ckpt
        assert torch.allclose(ckpt["missing_token"], missing_token)
        assert ckpt["d_input"] == 4

    def test_no_missing_token_when_none(self, tmp_path):
        encoder = nn.Linear(4, 8)
        config = {"name": "test"}
        path = tmp_path / "encoder.pt"

        save_encoder_checkpoint(encoder, config, path)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert "missing_token" not in ckpt

    def test_state_dict_is_loadable(self, tmp_path):
        """Saved state dict can be loaded into a fresh encoder."""
        encoder = nn.Linear(4, 8)
        config = {"name": "linear"}
        path = tmp_path / "encoder.pt"

        with torch.no_grad():
            encoder.weight.fill_(0.42)
            encoder.bias.fill_(0.07)

        save_encoder_checkpoint(encoder, config, path)

        new_encoder = nn.Linear(4, 8)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        new_encoder.load_state_dict(ckpt["encoder_state_dict"])

        assert torch.allclose(new_encoder.weight, encoder.weight)
        assert torch.allclose(new_encoder.bias, encoder.bias)


class TestSupervisedCheckpointFormat:
    """Test that save_encoder_weights from supervised script produces v3 format."""

    @pytest.fixture
    def finetune_config(self):
        return OmegaConf.create(
            {
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
                    "freeze_encoder": False,
                    "unfreeze_epoch": None,
                },
                "optimizer": {
                    "name": "adam",
                    "lr": 1e-3,
                },
            }
        )

    def test_save_encoder_weights_v3_format(self, finetune_config, tmp_path):
        """save_encoder_weights should produce a v3 checkpoint with encoder_config."""
        from scripts.training.supervised import save_encoder_weights

        model = FineTuneModule(finetune_config)
        encoder_path = save_encoder_weights(model, finetune_config, str(tmp_path))

        checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)

        assert isinstance(checkpoint, dict), "Checkpoint should be a dict, not raw state_dict"
        assert "version" in checkpoint
        assert checkpoint["version"] == 3
        assert "encoder_state_dict" in checkpoint
        assert "encoder_config" in checkpoint
        assert "name" in checkpoint["encoder_config"]
        assert checkpoint["encoder_config"]["name"] == "transformer"

    def test_saved_checkpoint_loadable_by_finetune_module(self, finetune_config, tmp_path):
        """v3 checkpoint from supervised save should be loadable by FineTuneModule."""
        from scripts.training.supervised import save_encoder_weights

        model = FineTuneModule(finetune_config)
        encoder_path = save_encoder_weights(model, finetune_config, str(tmp_path))

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded_module = FineTuneModule(
                finetune_config,
                checkpoint_path=str(encoder_path),
            )
            user_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "old-format" in str(x.message).lower()
            ]
            assert len(user_warnings) == 0, "Loading saved checkpoint triggered old-format warning"

        x = torch.randn(2, 24, 10)
        mask = torch.ones(2, 24, 10, dtype=torch.bool)
        out = loaded_module(x, mask)
        assert "logits" in out
