"""Tests for Phase 2 improvements: config validation, code quality, and cleanup.

Tests cover:
- 2.1: Pydantic config validation at module boundary (TaskConfig, TrainingConfig, etc.)
- 2.2: use_missing_token exposed in finetune.yaml
- 2.5: logger used instead of print in finetune_module
- 2.6: import warnings at module level in dataset.py
"""

import inspect
import logging

import pytest
import yaml
from omegaconf import OmegaConf
from pydantic import ValidationError
from slices.training import FineTuneModule
from slices.training.config_schemas import (
    OptimizerConfig,
    SchedulerConfig,
    TaskConfig,
    TrainingConfig,
)

# =========================================================================
# 2.1: Pydantic Config Validation
# =========================================================================


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


# =========================================================================
# 2.1: Integration test - validation at FineTuneModule boundary
# =========================================================================


class TestFineTuneModuleConfigValidation:
    """Test that FineTuneModule rejects invalid configs at init time."""

    @pytest.fixture
    def base_config(self):
        """Valid base config."""
        return {
            "encoder": {
                "name": "transformer",
                "d_input": 10,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 2,
                "d_ff": 64,
                "dropout": 0.1,
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
                "dropout": 0.1,
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
        }

    def test_valid_config_creates_module(self, base_config):
        """Valid config should create module without error."""
        cfg = OmegaConf.create(base_config)
        module = FineTuneModule(cfg)
        assert module is not None

    def test_typo_in_task_config_rejected(self, base_config):
        """Typo in task config should be caught at module creation."""
        base_config["task"]["head_tpye"] = base_config["task"].pop("head_type")
        cfg = OmegaConf.create(base_config)
        with pytest.raises(ValidationError, match="head_tpye"):
            FineTuneModule(cfg)

    def test_typo_in_training_config_rejected(self, base_config):
        """Typo in training config should be caught at module creation."""
        base_config["training"]["freez_encoder"] = base_config["training"].pop("freeze_encoder")
        cfg = OmegaConf.create(base_config)
        with pytest.raises(ValidationError, match="freez_encoder"):
            FineTuneModule(cfg)

    def test_extra_task_key_rejected(self, base_config):
        """Unknown task config key should be caught at module creation."""
        base_config["task"]["warmup_steps"] = 100
        cfg = OmegaConf.create(base_config)
        with pytest.raises(ValidationError):
            FineTuneModule(cfg)


# =========================================================================
# 2.2: use_missing_token exposed in finetune.yaml
# =========================================================================


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


# =========================================================================
# 2.5: logger used instead of print in finetune_module
# =========================================================================


class TestFineTuneModuleUsesLogger:
    """Test that finetune_module uses logger instead of print."""

    def test_no_print_calls_in_module(self):
        """finetune_module.py should not contain bare print() calls."""
        import slices.training.finetune_module as mod

        source = inspect.getsource(mod)

        # Find print( calls that aren't in comments or strings
        import ast

        tree = ast.parse(source)
        print_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    print_calls.append(node.lineno)

        assert len(print_calls) == 0, (
            f"Found print() calls at lines {print_calls} in finetune_module.py. "
            "Use logger.info() instead."
        )

    def test_module_has_logger(self):
        """finetune_module should define a module-level logger."""
        import slices.training.finetune_module as mod

        assert hasattr(mod, "logger"), "finetune_module should have a module-level logger"
        assert isinstance(mod.logger, logging.Logger)


# =========================================================================
# 2.6: import warnings at module level in dataset.py
# =========================================================================


class TestDatasetImportWarnings:
    """Test that dataset.py has module-level import warnings."""

    def test_no_inline_import_warnings(self):
        """dataset.py should not have 'import warnings' inside function bodies."""
        import slices.data.dataset as mod

        source = inspect.getsource(mod)

        import ast

        tree = ast.parse(source)

        inline_imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Import):
                        for alias in child.names:
                            if alias.name == "warnings":
                                inline_imports.append(node.name)
                    elif isinstance(child, ast.ImportFrom):
                        if child.module == "warnings":
                            inline_imports.append(node.name)

        assert len(inline_imports) == 0, (
            f"Found 'import warnings' inside functions: {inline_imports}. " "Move to module level."
        )

    def test_module_level_import_exists(self):
        """dataset.py should have warnings imported at module level."""
        import slices.data.dataset as mod

        # Check that warnings is accessible at module scope
        assert hasattr(mod, "warnings") or "warnings" in dir(
            mod
        ), "dataset.py should import warnings at module level"
