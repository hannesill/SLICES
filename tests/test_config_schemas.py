"""Tests for Pydantic config validation schemas and YAML config files.

Tests cover:
- TaskConfig, TrainingConfig, OptimizerConfig, SchedulerConfig validation
- Typo/extra key rejection, default values, invalid enum values
- YAML config file correctness (e.g., finetune.yaml has expected keys)
"""

import pytest
import yaml
from pydantic import ValidationError
from slices.training.config_schemas import (
    OptimizerConfig,
    SchedulerConfig,
    TaskConfig,
    TrainingConfig,
)


class TestTaskConfigValidation:
    """Tests that TaskConfig catches invalid/misspelled keys."""

    def test_valid_config_passes(self):
        """Valid task config should pass validation."""
        cfg = TaskConfig(
            task_name="mortality_24h",
            task_type="binary",
            head_type="mlp",
            hidden_dims=[64],
            dropout=0.1,
            activation="relu",
        )
        assert cfg.task_name == "mortality_24h"
        assert cfg.head_type == "mlp"

    def test_extra_key_rejected(self):
        """Typo in config key should raise ValidationError."""
        with pytest.raises(ValidationError, match="head_tpye"):
            TaskConfig(
                task_name="mortality_24h",
                head_tpye="mlp",  # Typo!
            )

    def test_misspelled_hidden_dims_rejected(self):
        """Misspelled hidden_dims should raise ValidationError."""
        with pytest.raises(ValidationError, match="hiden_dims"):
            TaskConfig(
                task_name="mortality_24h",
                hiden_dims=[64],  # Typo!
            )

    def test_invalid_task_type_rejected(self):
        """Invalid task_type value should raise ValidationError."""
        with pytest.raises(ValidationError, match="task_type"):
            TaskConfig(
                task_name="mortality_24h",
                task_type="classification",  # Not a valid type
            )

    def test_invalid_head_type_rejected(self):
        """Invalid head_type value should raise ValidationError."""
        with pytest.raises(ValidationError, match="head_type"):
            TaskConfig(
                task_name="mortality_24h",
                head_type="transformer",  # Not a valid head
            )

    def test_invalid_activation_rejected(self):
        """Invalid activation value should raise ValidationError."""
        with pytest.raises(ValidationError, match="activation"):
            TaskConfig(
                task_name="mortality_24h",
                activation="leaky_relu",  # Not in valid set
            )

    def test_defaults_applied(self):
        """Missing optional fields should use defaults."""
        cfg = TaskConfig(task_name="mortality_24h")
        assert cfg.task_type == "binary"
        assert cfg.head_type == "mlp"
        assert cfg.hidden_dims == [64]
        assert cfg.dropout == 0.1
        assert cfg.activation == "relu"
        assert cfg.n_classes is None
        assert cfg.use_layer_norm is False


class TestTrainingConfigValidation:
    """Tests that TrainingConfig catches invalid/misspelled keys."""

    def test_valid_config_passes(self):
        """Valid training config should pass validation."""
        cfg = TrainingConfig(
            freeze_encoder=True,
            unfreeze_epoch=5,
        )
        assert cfg.freeze_encoder is True
        assert cfg.unfreeze_epoch == 5

    def test_extra_key_rejected(self):
        """Typo in training config key should raise ValidationError."""
        with pytest.raises(ValidationError, match="freez_encoder"):
            TrainingConfig(freez_encoder=True)

    def test_misspelled_unfreeze_rejected(self):
        """Misspelled unfreeze_epoch should raise ValidationError."""
        with pytest.raises(ValidationError, match="unfreez_epoch"):
            TrainingConfig(unfreez_epoch=5)

    def test_unknown_key_rejected(self):
        """Completely unknown key should raise ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.001)

    def test_defaults_applied(self):
        """Missing optional fields should use defaults."""
        cfg = TrainingConfig()
        assert cfg.max_epochs == 50
        assert cfg.batch_size == 64
        assert cfg.freeze_encoder is True
        assert cfg.unfreeze_epoch is None
        assert cfg.use_missing_token is True

    def test_use_missing_token_field_exists(self):
        """use_missing_token should be a valid field with default True."""
        cfg = TrainingConfig(use_missing_token=False)
        assert cfg.use_missing_token is False


class TestOptimizerConfigValidation:
    """Tests that OptimizerConfig catches invalid configs."""

    def test_valid_config_passes(self):
        cfg = OptimizerConfig(name="adamw", lr=1e-4)
        assert cfg.name == "adamw"

    def test_extra_key_rejected(self):
        with pytest.raises(ValidationError, match="lerning_rate"):
            OptimizerConfig(name="adam", lr=1e-3, lerning_rate=1e-3)

    def test_invalid_optimizer_name(self):
        with pytest.raises(ValidationError, match="optimizer name"):
            OptimizerConfig(name="rmsprop", lr=1e-3)


class TestSchedulerConfigValidation:
    """Tests that SchedulerConfig catches invalid configs."""

    def test_valid_config_passes(self):
        cfg = SchedulerConfig(name="cosine", T_max=100)
        assert cfg.name == "cosine"

    def test_extra_key_rejected(self):
        with pytest.raises(ValidationError, match="t_max"):
            SchedulerConfig(name="cosine", t_max=100)  # Wrong case

    def test_invalid_scheduler_name(self):
        with pytest.raises(ValidationError, match="scheduler name"):
            SchedulerConfig(name="linear_warmup", T_max=100)


class TestUseMissingTokenYAML:
    """Test that use_missing_token is defined in finetune.yaml."""

    def test_finetune_yaml_has_use_missing_token(self):
        """finetune.yaml should define use_missing_token under training."""
        import pathlib

        yaml_path = pathlib.Path(__file__).parent.parent / "configs" / "finetune.yaml"
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        assert "training" in raw
        assert "use_missing_token" in raw["training"], (
            "use_missing_token should be explicitly defined in finetune.yaml "
            "so users can discover and override it"
        )
        assert raw["training"]["use_missing_token"] is True
