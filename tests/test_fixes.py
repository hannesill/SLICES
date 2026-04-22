"""Regression tests for FIXES.md Issues 2-9.

Tests label manifest validation, normalization stats integrity,
sequence-length override, checkpoint provenance, mortality precision,
and exporter dedup logic.
"""

import importlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest
import torch
import yaml
from omegaconf import OmegaConf
from slices.data.labels import LabelBuilder, LabelBuilderFactory, LabelConfig
from slices.data.labels.aki import AKILabelBuilder
from slices.data.labels.los import LOSLabelBuilder
from slices.data.labels.mortality import MortalityLabelBuilder
from slices.data.tensor_cache import (
    _compute_split_hash,
    get_data_fingerprint,
    get_preprocessing_fingerprint,
    load_cached_tensors,
    load_normalization_stats,
    save_cached_tensors,
    save_normalization_stats,
)
from slices.data.tensor_preprocessing import extract_tensors_from_dataframe


class DummyWandbRun:
    """Minimal W&B run stub for export pipeline tests."""

    def __init__(self, run_id, config, tags, name, created_at="2026-04-07T00:00:00"):
        self.config = config
        self.summary_metrics = {}
        self.id = run_id
        self.url = f"https://example.com/{run_id}"
        self.name = name
        self.tags = tags
        self.group = "g"
        self.created_at = created_at


# ============================================================================
# Issue 3: Label manifest freshness checking
# ============================================================================


class TestLabelManifest:
    """Tests for label builder versioning and config hashing."""

    def test_all_builders_have_semantic_version(self):
        """Every label builder subclass must define SEMANTIC_VERSION."""
        assert hasattr(MortalityLabelBuilder, "SEMANTIC_VERSION")
        assert hasattr(AKILabelBuilder, "SEMANTIC_VERSION")
        assert hasattr(LOSLabelBuilder, "SEMANTIC_VERSION")

    def test_config_hash_deterministic(self):
        """Same config produces same hash."""
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
        )
        h1 = LabelBuilder.config_hash(config)
        h2 = LabelBuilder.config_hash(config)
        assert h1 == h2
        assert len(h1) == 16

    def test_config_hash_changes_on_window_change(self):
        """Different prediction_window_hours produces different hash."""
        config_24 = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=24,
        )
        config_48 = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=48,
            observation_window_hours=24,
        )
        assert LabelBuilder.config_hash(config_24) != LabelBuilder.config_hash(config_48)

    def test_config_hash_changes_on_gap_change(self):
        """Different gap_hours produces different hash."""
        config_0 = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            gap_hours=0,
        )
        config_6 = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            gap_hours=6,
        )
        assert LabelBuilder.config_hash(config_0) != LabelBuilder.config_hash(config_6)

    def test_config_hash_changes_on_label_sources_change(self):
        """Different label_sources must invalidate stale emitted labels."""
        config_stays = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            label_sources=["stays"],
        )
        config_mortality = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            label_sources=["stays", "mortality_info"],
        )

        assert LabelBuilder.config_hash(config_stays) != LabelBuilder.config_hash(config_mortality)

    def test_config_hash_changes_on_supported_datasets_change(self):
        """Dataset support affects whether a task emits labels."""
        config_miiv = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            supported_datasets=["miiv"],
        )
        config_all = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            supported_datasets=None,
        )

        assert LabelBuilder.config_hash(config_miiv) != LabelBuilder.config_hash(config_all)

    def test_validate_data_prerequisites_version_mismatch(self, tmp_path):
        """Builder version mismatch should raise RuntimeError."""
        from slices.training.utils import validate_data_prerequisites

        # Create metadata with old version
        metadata = {
            "label_manifest": {
                "mortality_24h": {
                    "builder_version": "0.0.1",  # old version
                    "config_hash": "will_not_match",
                }
            }
        }
        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        with open(tmp_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with pytest.raises(RuntimeError, match="Label builder version mismatch"):
            validate_data_prerequisites(str(tmp_path), "miiv", task_names=["mortality_24h"])

    def test_validate_data_prerequisites_missing_manifest_raises(self, tmp_path):
        """Missing label_manifest should abort training."""
        from slices.training.utils import validate_data_prerequisites

        metadata = {"task_names": ["mortality_24h"]}  # no label_manifest
        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        with open(tmp_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with pytest.raises(RuntimeError, match="no label_manifest"):
            validate_data_prerequisites(str(tmp_path), "miiv", task_names=["mortality_24h"])

    def test_validate_data_prerequisites_missing_metadata_raises(self, tmp_path):
        """Missing metadata.yaml should abort training."""
        from slices.training.utils import validate_data_prerequisites

        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        # No metadata.yaml at all

        with pytest.raises(FileNotFoundError, match="metadata.yaml not found"):
            validate_data_prerequisites(str(tmp_path), "miiv", task_names=["mortality_24h"])

    def test_validate_data_prerequisites_missing_task_in_manifest_raises(self, tmp_path):
        """Task missing from manifest should abort training."""
        from slices.training.utils import validate_data_prerequisites

        metadata = {
            "label_manifest": {
                "mortality_hospital": {
                    "builder_version": "1.0.0",
                    "config_hash": "abc123",
                }
                # mortality_24h is NOT in the manifest
            }
        }
        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        with open(tmp_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with pytest.raises(RuntimeError, match="not found in label manifest"):
            validate_data_prerequisites(str(tmp_path), "miiv", task_names=["mortality_24h"])

    def test_validate_data_prerequisites_matching_passes(self, tmp_path):
        """Matching manifest should pass cleanly."""
        from slices.data.utils import get_package_data_dir
        from slices.training.utils import validate_data_prerequisites

        with open(get_package_data_dir() / "tasks" / "mortality_24h.yaml") as f:
            config = LabelConfig(**yaml.safe_load(f))
        builder = LabelBuilderFactory.create(config)
        metadata = {
            "label_manifest": {
                "mortality_24h": {
                    "builder_version": builder.SEMANTIC_VERSION,
                    "config_hash": LabelBuilder.config_hash(config),
                }
            }
        }
        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        with open(tmp_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Should not raise when the manifest matches the checked task config.
        validate_data_prerequisites(str(tmp_path), "miiv", task_names=["mortality_24h"])

    def test_validate_data_prerequisites_rejects_wrong_benchmark_window(self, tmp_path):
        """Declared benchmark windows must match the fixed thesis contract."""
        from slices.training.utils import validate_data_prerequisites

        metadata = {"seq_length_hours": 48}
        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        with open(tmp_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with pytest.raises(RuntimeError, match="seq_length_hours"):
            validate_data_prerequisites(str(tmp_path), "miiv")

    def test_validate_data_prerequisites_rejects_blocked_features(self, tmp_path):
        """Processed tensors must not contain leakage or future-derived features."""
        from slices.training.utils import validate_data_prerequisites

        metadata = {"feature_names": ["hr", "los_icu"]}
        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        with open(tmp_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with pytest.raises(RuntimeError, match="blocked leakage"):
            validate_data_prerequisites(str(tmp_path), "miiv")

    def test_validate_data_prerequisites_uses_provided_task_config(self, tmp_path):
        """Active Hydra task configs should override the fallback task-definition tree."""
        from slices.training.utils import validate_data_prerequisites

        active_task = {
            "task_name": "mortality_24h",
            "task_type": "binary",
            "prediction_window_hours": 24,
            "observation_window_hours": 6,
            "gap_hours": 6,
            "label_sources": ["stays", "mortality_info"],
            "label_params": {},
            "head_type": "mlp",
            "hidden_dims": [64],
            "dropout": 0.1,
            "activation": "relu",
        }
        label_config = LabelConfig(
            task_name=active_task["task_name"],
            task_type=active_task["task_type"],
            prediction_window_hours=active_task["prediction_window_hours"],
            observation_window_hours=active_task["observation_window_hours"],
            gap_hours=active_task["gap_hours"],
            label_sources=active_task["label_sources"],
            label_params=active_task["label_params"],
        )
        builder = LabelBuilderFactory.create(label_config)
        metadata = {
            "label_manifest": {
                "mortality_24h": {
                    "builder_version": builder.SEMANTIC_VERSION,
                    "config_hash": LabelBuilder.config_hash(label_config),
                }
            }
        }
        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        with open(tmp_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        validate_data_prerequisites(
            str(tmp_path),
            "miiv",
            task_names=["mortality_24h"],
            task_configs=[active_task],
        )

    def test_validate_data_prerequisites_missing_task_config_raises(self, tmp_path):
        """Missing task config should fail closed instead of skipping validation."""
        from slices.training.utils import validate_data_prerequisites

        metadata = {
            "label_manifest": {
                "definitely_missing_task": {
                    "builder_version": "1.0.0",
                    "config_hash": "abc123",
                }
            }
        }
        (tmp_path / "splits.yaml").write_text("seed: 42\n")
        with open(tmp_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with pytest.raises(FileNotFoundError, match="Task config not found"):
            validate_data_prerequisites(
                str(tmp_path),
                "miiv",
                task_names=["definitely_missing_task"],
            )


# ============================================================================
# Issue 9: Hash-keyed normalization stats cache
# ============================================================================


class TestNormalizationStatsCache:
    """Tests for hash-keyed normalization stats files."""

    def test_split_hash_deterministic(self):
        """Same indices produce same hash."""
        indices = [0, 5, 10, 15]
        h1 = _compute_split_hash(indices, normalize=True)
        h2 = _compute_split_hash(indices, normalize=True)
        assert h1 == h2

    def test_split_hash_differs_for_different_indices(self):
        """Different indices produce different hash."""
        h1 = _compute_split_hash([0, 1, 2], normalize=True)
        h2 = _compute_split_hash([0, 1, 3], normalize=True)
        assert h1 != h2

    def test_split_hash_differs_for_normalize_flag(self):
        """Same indices but different normalize flag produce different hash."""
        h1 = _compute_split_hash([0, 1, 2], normalize=True)
        h2 = _compute_split_hash([0, 1, 2], normalize=False)
        assert h1 != h2

    def test_split_hash_order_invariant(self):
        """Hash is the same regardless of input order (sorted internally)."""
        h1 = _compute_split_hash([5, 0, 10], normalize=True)
        h2 = _compute_split_hash([0, 5, 10], normalize=True)
        assert h1 == h2

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load stats with hash-keyed filenames."""
        means = torch.tensor([1.0, 2.0, 3.0])
        stds = torch.tensor([0.5, 1.0, 1.5])
        names = ["a", "b", "c"]
        indices = [0, 1, 2, 3, 4]

        save_normalization_stats(tmp_path, means, stds, names, indices, normalize=True)

        loaded = load_normalization_stats(tmp_path, indices, normalize=True)
        assert loaded is not None
        assert loaded["feature_means"] == means.tolist()
        assert loaded["feature_stds"] == stds.tolist()

    def test_different_splits_different_files(self, tmp_path):
        """Different splits write to different files."""
        means = torch.tensor([1.0, 2.0])
        stds = torch.tensor([0.5, 1.0])
        names = ["a", "b"]

        save_normalization_stats(tmp_path, means, stds, names, [0, 1, 2], normalize=True)
        save_normalization_stats(tmp_path, means * 2, stds * 2, names, [3, 4, 5], normalize=True)

        stats_files = list(tmp_path.glob("normalization_stats_*.yaml"))
        assert len(stats_files) == 2

        loaded1 = load_normalization_stats(tmp_path, [0, 1, 2], normalize=True)
        loaded2 = load_normalization_stats(tmp_path, [3, 4, 5], normalize=True)
        assert loaded1["feature_means"] != loaded2["feature_means"]

    def test_legacy_fallback(self, tmp_path):
        """Legacy normalization_stats.yaml is loaded when hash-keyed file doesn't exist."""
        indices = [0, 1, 2]
        legacy_stats = {
            "feature_means": [1.0, 2.0],
            "feature_stds": [0.5, 1.0],
            "feature_names": ["a", "b"],
            "train_indices": indices,
            "normalize": True,
            "data_fingerprint": get_data_fingerprint(tmp_path),
            "preprocessing_fingerprint": get_preprocessing_fingerprint(),
        }
        with open(tmp_path / "normalization_stats.yaml", "w") as f:
            yaml.dump(legacy_stats, f)

        loaded = load_normalization_stats(tmp_path, indices, normalize=True)
        assert loaded is not None
        assert loaded["feature_means"] == [1.0, 2.0]

    def test_legacy_stats_without_fingerprints_are_ignored(self, tmp_path):
        """Legacy stats without freshness fingerprints should not be trusted."""
        indices = [0, 1, 2]
        legacy_stats = {
            "feature_means": [1.0, 2.0],
            "feature_stds": [0.5, 1.0],
            "feature_names": ["a", "b"],
            "train_indices": indices,
            "normalize": True,
        }
        with open(tmp_path / "normalization_stats.yaml", "w") as f:
            yaml.dump(legacy_stats, f)

        loaded = load_normalization_stats(tmp_path, indices, normalize=True)
        assert loaded is None

    def test_hash_keyed_stats_invalidated_on_fingerprint_mismatch(self, tmp_path, monkeypatch):
        """Hash-keyed normalization stats should be invalidated when fingerprints change."""
        import slices.data.tensor_cache as cache_mod

        monkeypatch.setattr(cache_mod, "get_data_fingerprint", lambda data_dir: "data-v1")
        monkeypatch.setattr(cache_mod, "get_preprocessing_fingerprint", lambda: "prep-v1")

        means = torch.tensor([1.0, 2.0])
        stds = torch.tensor([0.5, 1.0])
        save_normalization_stats(tmp_path, means, stds, ["a", "b"], [0, 1, 2], normalize=True)
        assert load_normalization_stats(tmp_path, [0, 1, 2], normalize=True) is not None

        monkeypatch.setattr(cache_mod, "get_preprocessing_fingerprint", lambda: "prep-v2")
        assert load_normalization_stats(tmp_path, [0, 1, 2], normalize=True) is None


class TestRawTensorCacheFreshness:
    """Tests for raw tensor cache invalidation."""

    def test_tensor_cache_invalidated_on_data_fingerprint_mismatch(self, tmp_path, monkeypatch):
        """Raw tensor cache should be ignored when processed data changes."""
        import slices.data.tensor_cache as cache_mod

        monkeypatch.setattr(cache_mod, "get_data_fingerprint", lambda data_dir: "data-v1")
        monkeypatch.setattr(cache_mod, "get_preprocessing_fingerprint", lambda: "prep-v1")

        timeseries = torch.zeros((2, 4, 3), dtype=torch.float32)
        masks = torch.ones((2, 4, 3), dtype=torch.bool)
        save_cached_tensors(tmp_path, timeseries, masks, seq_length=4, n_features=3)
        loaded = load_cached_tensors(tmp_path, seq_length=4, n_features=3)
        assert loaded is not None

        monkeypatch.setattr(cache_mod, "get_data_fingerprint", lambda data_dir: "data-v2")
        assert load_cached_tensors(tmp_path, seq_length=4, n_features=3) is None

    def test_failed_tensor_cache_save_removes_temp_file(self, tmp_path, monkeypatch):
        """Failed atomic writes should not leave a target cache or stale temp file."""
        import slices.data.tensor_cache as cache_mod

        monkeypatch.setattr(cache_mod, "get_data_fingerprint", lambda data_dir: "data-v1")
        monkeypatch.setattr(cache_mod, "get_preprocessing_fingerprint", lambda: "prep-v1")

        def fail_save(cache_data, path):
            raise RuntimeError("simulated save failure")

        monkeypatch.setattr(cache_mod.torch, "save", fail_save)

        timeseries = torch.zeros((2, 4, 3), dtype=torch.float32)
        masks = torch.ones((2, 4, 3), dtype=torch.bool)
        save_cached_tensors(tmp_path, timeseries, masks, seq_length=4, n_features=3)

        cache_path = cache_mod.get_tensor_cache_path(tmp_path, seq_length=4, n_features=3)
        assert not cache_path.exists()
        assert list(cache_path.parent.glob("*.tmp")) == []


class TestCombinedDatasetValidation:
    """Tests for combined-dataset compatibility checks."""

    def test_feature_order_mismatch_raises(self):
        """Same feature set in different order should fail closed."""
        import importlib

        mod = importlib.import_module("scripts.preprocessing.create_combined_dataset")

        meta_a = {"feature_names": ["hr", "map"], "seq_length_hours": 48, "min_stay_hours": 48}
        meta_b = {"feature_names": ["map", "hr"], "seq_length_hours": 48, "min_stay_hours": 48}

        with pytest.raises(ValueError, match="Feature order mismatch"):
            mod.validate_feature_compatibility(meta_a, meta_b)

    def test_invariant_mismatch_raises(self):
        """Different preprocessing invariants should fail before merge."""
        import importlib

        mod = importlib.import_module("scripts.preprocessing.create_combined_dataset")

        meta_a = {"feature_names": ["hr", "map"], "seq_length_hours": 48, "min_stay_hours": 48}
        meta_b = {"feature_names": ["hr", "map"], "seq_length_hours": 72, "min_stay_hours": 48}

        with pytest.raises(ValueError, match="preprocessing invariants"):
            mod.validate_feature_compatibility(meta_a, meta_b)

    def test_patient_ids_are_namespaced_even_without_stay_collision(self):
        """Combined setup should namespace patient IDs independently of stay_id overlap."""
        import importlib

        mod = importlib.import_module("scripts.preprocessing.create_combined_dataset")

        static_a = pl.DataFrame({"stay_id": [1], "patient_id": [10]})
        static_b = pl.DataFrame({"stay_id": [2], "patient_id": [10]})

        namespaced_a = mod.namespace_patient_ids(static_a, "miiv")
        namespaced_b = mod.namespace_patient_ids(static_b, "eicu")

        mod.validate_no_id_collision(namespaced_a, namespaced_b)
        assert namespaced_a["patient_id"].item() == "miiv:10"
        assert namespaced_b["patient_id"].item() == "eicu:10"


class TestCombinedSetupPath:
    """Regression tests for the combined-dataset setup flow."""

    def _write_processed_dataset(
        self,
        processed_dir,
        dataset_name: str,
        stay_id: int,
        patient_id: int,
    ) -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)

        static_df = pl.DataFrame(
            {
                "stay_id": [stay_id],
                "patient_id": [patient_id],
                "age": [65],
                "gender": ["M"],
            }
        )
        static_df.write_parquet(processed_dir / "static.parquet")

        timeseries_df = pl.DataFrame(
            {
                "stay_id": [stay_id],
                "timeseries": [[[[1.0, 2.0], [3.0, 4.0]]]],
                "mask": [[[[True, True], [True, True]]]],
            }
        )
        timeseries_df.write_parquet(processed_dir / "timeseries.parquet")

        labels_df = pl.DataFrame(
            {
                "stay_id": [stay_id],
                "mortality_24h": [0],
            }
        )
        labels_df.write_parquet(processed_dir / "labels.parquet")

        metadata = {
            "dataset": dataset_name,
            "feature_set": "core",
            "feature_names": ["hr", "map"],
            "n_features": 2,
            "seq_length_hours": 2,
            "min_stay_hours": 2,
            "label_horizon_hours": 24,
            "task_names": ["mortality_24h"],
            "label_manifest": {
                "mortality_24h": {
                    "builder_version": "1.0.0",
                    "config_hash": "abc123",
                }
            },
        }
        with open(processed_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

    def test_create_combined_dataset_prepares_by_default(self, tmp_path, monkeypatch):
        """Combined creation should trigger splits/stat prep unless explicitly skipped."""
        mod = importlib.import_module("scripts.preprocessing.create_combined_dataset")
        source_a = tmp_path / "miiv"
        source_b = tmp_path / "eicu"
        output_dir = tmp_path / "combined"

        self._write_processed_dataset(source_a, "miiv", stay_id=1, patient_id=11)
        self._write_processed_dataset(source_b, "eicu", stay_id=2, patient_id=22)

        captured = {}

        def fake_prepare_processed_dataset(processed_dir, seed, dataset_name=None):
            captured["processed_dir"] = processed_dir
            captured["seed"] = seed
            captured["dataset_name"] = dataset_name
            return {}, {}

        monkeypatch.setattr(mod, "prepare_processed_dataset", fake_prepare_processed_dataset)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "create_combined_dataset.py",
                "--source",
                str(source_a),
                str(source_b),
                "--output",
                str(output_dir),
            ],
        )

        mod.main()

        assert captured["processed_dir"] == output_dir
        assert captured["seed"] == 42
        assert captured["dataset_name"] == "combined"

    def test_setup_and_extract_default_path_includes_combined(self, tmp_path):
        """The default setup path should build the combined dataset as part of readiness prep."""
        repo_root = Path(__file__).resolve().parents[1]
        temp_repo = tmp_path / "repo"
        (temp_repo / "scripts" / "internal").mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / "scripts" / "setup_and_extract.sh", temp_repo / "scripts")

        for raw_dir in ("data/raw/mimiciv", "data/raw/eicu-crd"):
            (temp_repo / raw_dir).mkdir(parents=True, exist_ok=True)

        for ds in ("miiv", "eicu"):
            ricu_output = temp_repo / "data" / "ricu_output" / ds
            ricu_output.mkdir(parents=True, exist_ok=True)
            (ricu_output / "done.txt").write_text("ok\n")

            processed_dir = temp_repo / "data" / "processed" / ds
            processed_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "timeseries.parquet",
                "static.parquet",
                "labels.parquet",
                "metadata.yaml",
                "splits.yaml",
                "normalization_stats.yaml",
            ):
                (processed_dir / name).write_text("")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        uv_log = tmp_path / "uv.log"
        uv_script = bin_dir / "uv"
        uv_script.write_text(
            "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$*" >> "$UV_LOG"\n' "exit 0\n"
        )
        uv_script.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["UV_LOG"] = str(uv_log)

        result = subprocess.run(
            ["bash", "scripts/setup_and_extract.sh", "--skip-deps"],
            cwd=temp_repo,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        log_text = uv_log.read_text()
        assert "scripts/preprocessing/create_combined_dataset.py" in log_text
        assert "--source data/processed/miiv data/processed/eicu" in log_text

    def test_build_processed_data_commands_include_combined_flow(self):
        """The lightweight wrapper should orchestrate the existing Python steps."""
        mod = importlib.import_module("scripts.preprocessing.build_processed_data")

        commands = mod.build_commands(
            ["miiv", "eicu"],
            build_combined=True,
            seed=7,
            python_executable="python",
        )

        assert commands == [
            ["python", "scripts/preprocessing/extract_ricu.py", "dataset=miiv"],
            ["python", "scripts/preprocessing/prepare_dataset.py", "dataset=miiv", "seed=7"],
            ["python", "scripts/preprocessing/extract_ricu.py", "dataset=eicu"],
            ["python", "scripts/preprocessing/prepare_dataset.py", "dataset=eicu", "seed=7"],
            [
                "python",
                "scripts/preprocessing/create_combined_dataset.py",
                "--source",
                "data/processed/miiv",
                "data/processed/eicu",
                "--names",
                "miiv",
                "eicu",
                "--output",
                "data/processed/combined",
                "--seed",
                "7",
            ],
        ]

    def test_build_processed_data_combined_adds_missing_source(self):
        """Requesting combined should ensure both source datasets are processed."""
        mod = importlib.import_module("scripts.preprocessing.build_processed_data")

        commands = mod.build_commands(
            ["miiv"],
            build_combined=True,
            seed=42,
            python_executable="python",
        )

        assert ["python", "scripts/preprocessing/extract_ricu.py", "dataset=eicu"] in commands


class TestFairnessRevisionScoping:
    """Regression tests for revision-scoped standalone fairness evaluation."""

    def test_parse_args_accepts_revision_filter(self, monkeypatch):
        """The fairness CLI should expose a revision filter for rerun scoping."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")
        monkeypatch.delenv("REVISION", raising=False)
        monkeypatch.delenv("WANDB_REVISION", raising=False)

        monkeypatch.setattr(
            sys,
            "argv",
            ["evaluate_fairness.py", "--revision", "thesis-v1"],
        )
        args = mod.parse_args()

        assert args.revision == ["thesis-v1"]

    def test_parse_args_uses_revision_env_when_cli_omits_it(self, monkeypatch):
        """The standalone script may inherit an explicit revision from the environment."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")
        monkeypatch.setenv("REVISION", "thesis-v1")
        monkeypatch.delenv("WANDB_REVISION", raising=False)
        monkeypatch.setattr(sys, "argv", ["evaluate_fairness.py"])

        args = mod.parse_args()

        assert args.revision == ["thesis-v1"]

    def test_parse_args_requires_revision_when_no_scope_is_available(self, monkeypatch):
        """Fail closed instead of running an unscoped fairness sweep."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")
        monkeypatch.delenv("REVISION", raising=False)
        monkeypatch.delenv("WANDB_REVISION", raising=False)
        monkeypatch.setattr(sys, "argv", ["evaluate_fairness.py"])

        with pytest.raises(SystemExit) as excinfo:
            mod.parse_args()

        assert excinfo.value.code == 2

    def test_default_experiment_classes_include_all_downstream_training_classes(self):
        """The standalone thesis fairness corpus should include every downstream class."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")

        expected = {
            "core_ssl_benchmark",
            "label_efficiency",
            "cross_dataset_transfer",
            "hp_robustness",
            "capacity_study",
            "classical_baselines",
            "ts2vec_extension",
            "smart_external_reference",
        }
        assert expected.issubset(set(mod.DEFAULT_EXPERIMENT_CLASSES))

    def test_fetch_eval_runs_single_revision_adds_server_side_filter(self, monkeypatch):
        """A single revision should be sent as a W&B tag filter."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")
        captured = {}

        class DummyApi:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            def runs(self, path, filters, order):
                captured["path"] = path
                captured["filters"] = filters
                captured["order"] = order
                return []

        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=DummyApi))

        mod.fetch_eval_runs(
            project="proj",
            entity="entity",
            experiment_classes=["core_ssl_benchmark"],
            paradigms=["mae"],
            datasets=["miiv"],
            phases=["finetune"],
            revisions=["thesis-v1"],
        )

        assert captured["path"] == "entity/proj"
        assert captured["filters"]["$and"] == [
            {"tags": "experiment_class:core_ssl_benchmark"},
            {"tags": "paradigm:mae"},
            {"tags": "dataset:miiv"},
            {"tags": "phase:finetune"},
            {"tags": "revision:thesis-v1"},
        ]

    def test_fetch_eval_runs_multi_revision_filters_client_side(self, monkeypatch):
        """Multiple revisions should be filtered client-side without mixing reruns."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")

        runs = [
            SimpleNamespace(id="keep_v1", tags=["phase:finetune", "revision:v1"]),
            SimpleNamespace(id="keep_v2", tags=["phase:supervised", "revision:v2"]),
            SimpleNamespace(id="drop_phase", tags=["phase:baseline", "revision:v1"]),
            SimpleNamespace(id="drop_revision", tags=["phase:finetune", "revision:v3"]),
        ]

        class DummyApi:
            def __init__(self, timeout):
                pass

            def runs(self, path, filters, order):
                return runs

        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=DummyApi))

        filtered = mod.fetch_eval_runs(
            project="proj",
            entity=None,
            experiment_classes=None,
            paradigms=None,
            datasets=None,
            phases=["finetune", "supervised"],
            revisions=["v1", "v2"],
        )

        assert [run.id for run in filtered] == ["keep_v1", "keep_v2"]

    def test_recorded_best_checkpoint_is_used_for_fairness(self, tmp_path):
        """Standalone fairness should honor the run's recorded best checkpoint."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")

        run_dir = tmp_path / "run123" / "checkpoints"
        run_dir.mkdir(parents=True)
        best_ckpt = run_dir / "best.ckpt"
        last_ckpt = run_dir / "last.ckpt"
        best_ckpt.write_text("best")
        last_ckpt.write_text("last")

        run = SimpleNamespace(
            id="run123",
            config={"output_dir": "outputs/run123"},
            summary_metrics={
                "_eval_checkpoint_source": "best",
                "_best_ckpt_path": "outputs/run123/checkpoints/best.ckpt",
            },
        )

        ckpt_path, source = mod.resolve_evaluation_checkpoint(
            run,
            outputs_root=str(tmp_path),
            task_type="binary",
        )

        assert ckpt_path == best_ckpt
        assert source == "recorded_best"

    def test_recorded_final_checkpoint_uses_last_ckpt(self, tmp_path):
        """Standalone fairness should use last.ckpt when test metrics used the final model."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")

        run_dir = tmp_path / "run456" / "checkpoints"
        run_dir.mkdir(parents=True)
        (run_dir / "last.ckpt").write_text("last")

        run = SimpleNamespace(
            id="run456",
            config={"output_dir": "outputs/run456"},
            summary_metrics={"_eval_checkpoint_source": "final"},
        )

        ckpt_path, source = mod.resolve_evaluation_checkpoint(
            run,
            outputs_root=str(tmp_path),
            task_type="binary",
        )

        assert ckpt_path == run_dir / "last.ckpt"
        assert source == "recorded_final"

    def test_missing_checkpoint_provenance_raises(self, tmp_path):
        """Runs without recorded evaluation provenance must not use heuristic checkpoint lookup."""
        mod = importlib.import_module("scripts.eval.evaluate_fairness")

        run_dir = tmp_path / "run789" / "checkpoints"
        run_dir.mkdir(parents=True)
        (run_dir / "best.ckpt").write_text("best")
        (run_dir / "last.ckpt").write_text("last")

        run = SimpleNamespace(
            id="run789",
            config={"output_dir": "outputs/run789"},
            summary_metrics={},
        )

        with pytest.raises(RuntimeError, match="lacks recorded checkpoint provenance"):
            mod.resolve_evaluation_checkpoint(
                run,
                outputs_root=str(tmp_path),
                task_type="binary",
            )

    def test_tmux_launcher_passes_revision_to_fairness(self, tmp_path):
        """The thesis launcher should thread revision tags through the fairness sweep."""
        repo_root = Path(__file__).resolve().parents[1]
        temp_repo = tmp_path / "repo"
        (temp_repo / "scripts" / "internal").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            repo_root / "scripts" / "internal" / "launch_thesis_tmux.sh",
            temp_repo / "scripts" / "internal",
        )

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tmux_log = tmp_path / "tmux.log"
        tmux_script = bin_dir / "tmux"
        tmux_script.write_text(
            "#!/usr/bin/env bash\n"
            'printf \'%s\\n\' "$*" >> "$TMUX_LOG"\n'
            'if [ "$1" = "has-session" ]; then\n'
            "  exit 1\n"
            "fi\n"
            "exit 0\n"
        )
        tmux_script.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["TMUX_LOG"] = str(tmux_log)
        env["SESSION_NAME"] = "test-session"
        env["WANDB_ENTITY"] = "test-entity"
        env["REVISION"] = "thesis-v2"
        env["RUN_EXPORT"] = "0"
        env["SKIP_LAUNCH_GIT_CHECK"] = "1"
        env["VALIDATE_PROCESSED_ARTIFACTS"] = "0"
        env["PURGE_RUNTIME_CACHES"] = "0"

        result = subprocess.run(
            ["bash", "scripts/internal/launch_thesis_tmux.sh"],
            cwd=temp_repo,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        runner_scripts = list((temp_repo / "logs" / "runner").glob("thesis-run-*.sh"))
        assert len(runner_scripts) == 1
        runner_text = runner_scripts[0].read_text()

        assert "scripts/eval/evaluate_fairness.py" in runner_text
        assert "--revision thesis-v2" in runner_text
        assert "--project slices-thesis" in runner_text
        assert "--entity test-entity" in runner_text

    def test_tmux_launcher_includes_classical_baselines_in_fairness(self, tmp_path):
        """The fairness sweep should include the canonical baseline class by default."""
        repo_root = Path(__file__).resolve().parents[1]
        temp_repo = tmp_path / "repo"
        (temp_repo / "scripts" / "internal").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            repo_root / "scripts" / "internal" / "launch_thesis_tmux.sh",
            temp_repo / "scripts" / "internal",
        )

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tmux_log = tmp_path / "tmux.log"
        tmux_script = bin_dir / "tmux"
        tmux_script.write_text(
            "#!/usr/bin/env bash\n"
            'printf \'%s\\n\' "$*" >> "$TMUX_LOG"\n'
            'if [ "$1" = "has-session" ]; then\n'
            "  exit 1\n"
            "fi\n"
            "exit 0\n"
        )
        tmux_script.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["TMUX_LOG"] = str(tmux_log)
        env["SESSION_NAME"] = "test-session"
        env["WANDB_ENTITY"] = "test-entity"
        env["RUN_EXPORT"] = "0"
        env["SKIP_LAUNCH_GIT_CHECK"] = "1"
        env["VALIDATE_PROCESSED_ARTIFACTS"] = "0"
        env["PURGE_RUNTIME_CACHES"] = "0"

        result = subprocess.run(
            ["bash", "scripts/internal/launch_thesis_tmux.sh"],
            cwd=temp_repo,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        runner_scripts = list((temp_repo / "logs" / "runner").glob("thesis-run-*.sh"))
        assert len(runner_scripts) == 1
        runner_text = runner_scripts[0].read_text()

        assert "--experiment-class" in runner_text
        assert "classical_baselines" in runner_text
        assert "core_ssl_benchmark" in runner_text

    def test_tmux_status_pane_uses_uv_run_python(self, tmp_path):
        """The status pane should use uv-managed Python."""
        repo_root = Path(__file__).resolve().parents[1]
        temp_repo = tmp_path / "repo"
        (temp_repo / "scripts" / "internal").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            repo_root / "scripts" / "internal" / "launch_thesis_tmux.sh",
            temp_repo / "scripts" / "internal",
        )

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tmux_log = tmp_path / "tmux.log"
        tmux_script = bin_dir / "tmux"
        tmux_script.write_text(
            "#!/usr/bin/env bash\n"
            'printf \'%s\\n\' "$*" >> "$TMUX_LOG"\n'
            'if [ "$1" = "has-session" ]; then\n'
            "  exit 1\n"
            "fi\n"
            "exit 0\n"
        )
        tmux_script.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["TMUX_LOG"] = str(tmux_log)
        env["SESSION_NAME"] = "test-session"
        env["WANDB_ENTITY"] = "test-entity"
        env["RUN_EXPORT"] = "0"
        env["SKIP_LAUNCH_GIT_CHECK"] = "1"
        env["VALIDATE_PROCESSED_ARTIFACTS"] = "0"
        env["PURGE_RUNTIME_CACHES"] = "0"

        result = subprocess.run(
            ["bash", "scripts/internal/launch_thesis_tmux.sh"],
            cwd=temp_repo,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            "uv\\ run\\ python\\ scripts/internal/run_experiments.py\\ status"
            in tmux_log.read_text()
        )


# ============================================================================
# Issue 4: Sequence-length override
# ============================================================================


class TestSeqLengthOverride:
    """Tests for sequence-length mismatch handling in tensor extraction."""

    def _make_df(self, n_samples: int, stored_len: int, n_features: int) -> pl.DataFrame:
        """Create a test DataFrame with nested list columns."""
        rows = []
        for i in range(n_samples):
            ts_row = [[float(j + i * 0.1) for _ in range(n_features)] for j in range(stored_len)]
            mask_row = [[True for _ in range(n_features)] for _ in range(stored_len)]
            rows.append({"timeseries": ts_row, "mask": mask_row})
        return pl.DataFrame(rows)

    def test_override_shorter_than_stored(self):
        """Requesting shorter seq_length should truncate."""
        df = self._make_df(n_samples=3, stored_len=48, n_features=2)
        ts, masks = extract_tensors_from_dataframe(df, seq_length=24, n_features=2)
        assert ts.shape == (3, 24, 2)
        assert masks.shape == (3, 24, 2)

    def test_override_longer_than_stored(self):
        """Requesting longer seq_length should pad."""
        df = self._make_df(n_samples=3, stored_len=24, n_features=2)
        ts, masks = extract_tensors_from_dataframe(df, seq_length=48, n_features=2)
        assert ts.shape == (3, 48, 2)
        assert masks.shape == (3, 48, 2)
        # Padded positions should have NaN in timeseries and False in masks
        assert torch.isnan(ts[0, 24, 0])
        assert not masks[0, 24, 0]

    def test_override_equal_to_stored(self):
        """Requesting same seq_length should use fast path."""
        df = self._make_df(n_samples=3, stored_len=48, n_features=2)
        ts, masks = extract_tensors_from_dataframe(df, seq_length=48, n_features=2)
        assert ts.shape == (3, 48, 2)
        assert masks.shape == (3, 48, 2)


# ============================================================================
# Issue 2: Mortality timestamp precision
# ============================================================================


class TestMortalityPrecision:
    """Tests for precision-aware mortality label building."""

    def _make_mortality_data(self, stays, deaths):
        """Helper to create raw_data dict for mortality builder.

        stays: list of (stay_id, intime, outtime)
        deaths: list of (stay_id, death_time, death_date, precision, source,
                         hospital_expire_flag)
        """
        stays_df = pl.DataFrame(
            {
                "stay_id": [s[0] for s in stays],
                "intime": [s[1] for s in stays],
                "outtime": [s[2] for s in stays],
            }
        )
        mort_df = pl.DataFrame(
            {
                "stay_id": [d[0] for d in deaths],
                "death_time": [d[1] for d in deaths],
                "death_date": [d[2] for d in deaths],
                "death_time_precision": [d[3] for d in deaths],
                "death_source": [d[4] for d in deaths],
                "hospital_expire_flag": [d[5] for d in deaths],
                "dischtime": [None for _ in deaths],
                "discharge_location": [None for _ in deaths],
                "date_of_death": [d[1] if d[1] else None for d in deaths],
            },
            schema={
                "stay_id": pl.Int64,
                "death_time": pl.Datetime("us"),
                "death_date": pl.Date,
                "death_time_precision": pl.Utf8,
                "death_source": pl.Utf8,
                "hospital_expire_flag": pl.Int32,
                "dischtime": pl.Datetime("us"),
                "discharge_location": pl.Utf8,
                "date_of_death": pl.Datetime("us"),
            },
        )
        return {"stays": stays_df, "mortality_info": mort_df}

    def test_timestamp_precision_exact(self):
        """Exact timestamp within prediction window → positive."""
        base = datetime(2020, 1, 1, 0, 0, 0)
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        # Death at hour 60 (within 48-72 prediction window)
        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=96))],
            deaths=[(1, base + timedelta(hours=60), None, "timestamp", "admissions.deathtime", 1)],
        )
        labels = builder.build_labels(raw_data)
        assert labels["label"][0] == 1

    def test_timestamp_death_during_observation_excluded_even_if_outtime_after_obs(self):
        """Observation-window timestamp deaths do not require outtime < obs_end."""
        base = datetime(2020, 1, 1, 0, 0, 0)
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=96))],
            deaths=[(1, base + timedelta(hours=12), None, "timestamp", "admissions.deathtime", 1)],
        )

        labels = builder.build_labels(raw_data)

        assert labels["label"][0] is None

    def test_hospital_timestamp_death_during_observation_excluded_even_if_outtime_after_obs(self):
        """Hospital mortality with obs window must exclude early observed deaths."""
        base = datetime(2020, 1, 1, 0, 0, 0)
        config = LabelConfig(
            task_name="mortality_hospital",
            task_type="binary",
            prediction_window_hours=None,
            observation_window_hours=24,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=96))],
            deaths=[(1, base + timedelta(hours=12), None, "timestamp", "admissions.deathtime", 1)],
        )

        labels = builder.build_labels(raw_data)

        assert labels["label"][0] is None

    def test_timestamp_precision_outside_window(self):
        """Exact timestamp after prediction window → negative."""
        base = datetime(2020, 1, 1, 0, 0, 0)
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        # Death at hour 80 (after 72h prediction end)
        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=96))],
            deaths=[(1, base + timedelta(hours=80), None, "timestamp", "admissions.deathtime", 1)],
        )
        labels = builder.build_labels(raw_data)
        assert labels["label"][0] == 0

    def test_date_only_fully_inside_window(self):
        """Date-only death where entire day falls inside prediction window → positive."""
        base = datetime(2020, 1, 1, 0, 0, 0)
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        # Prediction window: hour 48-72 = Jan 3 00:00 to Jan 4 00:00
        # Death date: Jan 3 (00:00-23:59) — entirely within window
        from datetime import date

        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=96))],
            deaths=[(1, None, date(2020, 1, 3), "date", "patients.dod", 1)],
        )
        labels = builder.build_labels(raw_data)
        assert labels["label"][0] == 1

    def test_date_only_overlapping_boundary_is_null(self):
        """Date-only death overlapping prediction window boundary → null."""
        base = datetime(2020, 1, 1, 0, 0, 0)
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        # Prediction window: hour 48-72 = Jan 3 00:00 to Jan 4 00:00
        # Death date: Jan 4 (00:00-23:59) — overlaps pred_end boundary
        from datetime import date

        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=120))],
            deaths=[(1, None, date(2020, 1, 4), "date", "patients.dod", 1)],
        )
        labels = builder.build_labels(raw_data)
        assert labels["label"][0] is None

    def test_date_only_before_observation_end(self):
        """Date-only death before observation window end → null (excluded)."""
        base = datetime(2020, 1, 1, 0, 0, 0)
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        # Death date: Jan 1 — before obs_end (Jan 3 00:00).
        # Patient also left ICU during obs (outtime < obs_end).
        from datetime import date

        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=20))],  # short stay
            deaths=[(1, None, date(2020, 1, 1), "date", "patients.dod", 1)],
        )
        labels = builder.build_labels(raw_data)
        assert labels["label"][0] is None

    def test_timezone_aware_stays_with_date_only_death(self):
        """UTC-aware stay times should not break date-only mortality comparisons."""
        base = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        from datetime import date

        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=96))],
            deaths=[(1, None, date(2020, 1, 3), "date", "patients.dod", 1)],
        )

        assert raw_data["stays"]["intime"].dtype == pl.Datetime("us", "UTC")

        labels = builder.build_labels(raw_data)
        assert labels["label"][0] == 1

    def test_survived_patient(self):
        """Patient who survived → label 0."""
        base = datetime(2020, 1, 1, 0, 0, 0)
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )
        builder = MortalityLabelBuilder(config)

        raw_data = self._make_mortality_data(
            stays=[(1, base, base + timedelta(hours=96))],
            deaths=[(1, None, None, None, None, 0)],
        )
        labels = builder.build_labels(raw_data)
        assert labels["label"][0] == 0

    def test_legacy_schema_migration(self):
        """Legacy datetime date_of_death should be migrated conservatively."""
        merged = pl.DataFrame(
            {
                "stay_id": [1],
                "intime": [datetime(2020, 1, 1)],
                "outtime": [datetime(2020, 1, 5)],
                "date_of_death": [datetime(2020, 1, 3, 12, 0, 0)],
                "hospital_expire_flag": [1],
            }
        )
        result = MortalityLabelBuilder._ensure_precision_columns(merged)
        assert "death_time_precision" in result.columns
        assert "death_time" in result.columns
        assert result["death_time_precision"][0] == "date"
        assert result["death_time"][0] is None
        assert result["death_date"][0].isoformat() == "2020-01-03"


# ============================================================================
# Issue 6: Exporter LR/mask_ratio decode and historical recovery
# ============================================================================


class TestExporterClassMetadata:
    """Tests for class-based exporter metadata recovery."""

    def test_lr_decode_matches_run_experiments_encoding(self):
        """Decode table must match str(v).replace('.','') encoding from run_experiments.py."""
        # These are the exact values from LR_ABLATION = [2e-4, 5e-4, 2e-3]
        assert str(2e-4).replace(".", "") == "00002"
        assert str(5e-4).replace(".", "") == "00005"
        assert str(2e-3).replace(".", "") == "0002"

        from scripts.export_results import _LR_DECODE

        assert _LR_DECODE["00002"] == 2e-4
        assert _LR_DECODE["00005"] == 5e-4
        assert _LR_DECODE["0002"] == 2e-3

    def test_mr_decode_matches_run_experiments_encoding(self):
        """Decode table must match str(v).replace('.','') encoding."""
        assert str(0.3).replace(".", "") == "03"
        assert str(0.75).replace(".", "") == "075"

        from scripts.export_results import _MR_DECODE

        assert _MR_DECODE["03"] == 0.3
        assert _MR_DECODE["075"] == 0.75

    def test_recover_lr_from_output_dir(self):
        """LR sensitivity finetunes recover upstream LR from output_dir pattern."""
        from scripts.export_results import _recover_pretrain_metadata

        config = {
            "output_dir": "outputs/hp_robustness/finetune_mae_mortality_24h_miiv_seed789_lr00002"
        }
        up_lr, up_mr, subtype = _recover_pretrain_metadata("some_run", config)
        assert up_lr == 2e-4
        assert subtype == "lr_sensitivity"

    def test_recover_mask_ratio_from_output_dir(self):
        """Mask-ratio sensitivity finetunes recover upstream mask ratio from output_dir."""
        from scripts.export_results import _recover_pretrain_metadata

        for suffix in ["mask_ratio03", "maskratio03"]:
            config = {
                "output_dir": (
                    "outputs/hp_robustness/" f"finetune_jepa_mortality_24h_miiv_seed42_{suffix}"
                )
            }
            up_lr, up_mr, subtype = _recover_pretrain_metadata("some_run", config)
            assert up_mr == 0.3
            assert subtype == "mask_ratio_sensitivity"

    def test_new_runs_use_config_directly(self):
        """New runs with explicit upstream fields skip name parsing."""
        from scripts.export_results import _recover_pretrain_metadata

        config = {"upstream_pretrain_lr": 5e-4, "experiment_subtype": "lr_sensitivity"}
        up_lr, up_mr, subtype = _recover_pretrain_metadata("irrelevant_name", config)
        assert up_lr == 5e-4
        assert subtype == "lr_sensitivity"

    def test_core_run_returns_none(self):
        """Core runs (no HP ablation) return None for all fields."""
        from scripts.export_results import _recover_pretrain_metadata

        config = {"output_dir": "outputs/core_ssl_benchmark/finetune_mae_mortality_24h_miiv_seed42"}
        up_lr, up_mr, subtype = _recover_pretrain_metadata("some_run", config)
        assert up_lr is None
        assert up_mr is None
        assert subtype is None

    def test_infer_model_size_for_capacity_variants(self):
        """Exporter should preserve medium/large labels."""
        from scripts.export_results import _infer_model_size

        assert _infer_model_size({"encoder": {"d_model": 64, "n_layers": 2}}) == "default"
        assert _infer_model_size({"encoder": {"d_model": 128, "n_layers": 4}}) == "medium"
        assert _infer_model_size({"encoder": {"d_model": 256, "n_layers": 4}}) == "large"
        assert (
            _infer_model_size(
                {"paradigm": "smart", "encoder": {"name": "smart", "d_model": 32, "n_layers": 2}}
            )
            == "default"
        )

    def test_extract_run_uses_experiment_class_tags(self):
        """Runs should export with explicit final experiment classes."""
        from scripts.export_results import extract_run

        transfer = DummyWandbRun(
            run_id="transfer",
            config={
                "experiment_class": "cross_dataset_transfer",
                "dataset": "eicu",
                "paradigm": "mae",
                "source_dataset": "miiv",
                "seed": 789,
                "encoder": {"d_model": 64, "n_layers": 2},
                "training": {"freeze_encoder": False},
                "task": {"task_name": "mortality_24h"},
            },
            tags=["phase:finetune"],
            name="finetune_mae_mortality_24h_eicu_from_miiv_seed789",
        )
        assert extract_run(transfer, [])["experiment_class"] == "cross_dataset_transfer"

        ts2vec = DummyWandbRun(
            run_id="ts2vec",
            config={
                "experiment_class": "ts2vec_extension",
                "dataset": "miiv",
                "paradigm": "ts2vec",
                "seed": 42,
                "encoder": {"d_model": 64, "n_layers": 2},
                "training": {"freeze_encoder": False},
                "task": {"task_name": "mortality_24h"},
            },
            tags=["phase:finetune"],
            name="finetune_ts2vec_mortality_24h_miiv_seed42",
        )
        assert extract_run(ts2vec, [])["experiment_class"] == "ts2vec_extension"

    def test_extract_run_recovers_transfer_from_output_dir(self):
        """Transfer runs should recover source_dataset from output_dir."""
        from scripts.export_results import extract_run

        transfer = DummyWandbRun(
            run_id="transfer_output_dir",
            config={
                "experiment_class": "cross_dataset_transfer",
                "dataset": "eicu",
                "paradigm": "mae",
                "seed": 789,
                "output_dir": (
                    "outputs/cross_dataset_transfer/"
                    "finetune_mae_mortality_24h_eicu_seed789_from_miiv"
                ),
                "encoder": {"d_model": 64, "n_layers": 2},
                "training": {"freeze_encoder": False},
                "task": {"task_name": "mortality_24h"},
            },
            tags=["phase:finetune"],
            name="s10_finetune_eicu_mae_mortality_24h_seed789",
        )

        row = extract_run(transfer, [])
        assert row["source_dataset"] == "miiv"
        assert row["experiment_class"] == "cross_dataset_transfer"

    def test_extract_run_assigns_protocol_b_to_xgboost_baselines(self):
        """XGBoost baselines must align with the Protocol-B comparison family."""
        from scripts.export_results import extract_run

        run = DummyWandbRun(
            run_id="xgb",
            config={
                "experiment_class": "classical_baselines",
                "dataset": "miiv",
                "paradigm": "xgboost",
                "seed": 42,
                "task": {"task_name": "mortality_24h"},
            },
            tags=["phase:baseline"],
            name="s11_xgboost_miiv_mortality_24h_seed42",
        )

        assert extract_run(run, [])["protocol"] == "B"

    def test_xgboost_wandb_tags_include_revision_scope(self):
        """XGBoost runs must be visible to revision-scoped export and fairness jobs."""
        from scripts.training.xgboost_baseline import _build_wandb_tags

        cfg = OmegaConf.create(
            {
                "logging": {"wandb_tags": ["phase:baseline"]},
                "experiment_class": "classical_baselines",
                "revision": "thesis-v1",
                "rerun_reason": "rerun canonical thesis baseline sweep with fixed tags",
                "label_fraction": 0.1,
            }
        )

        tags = _build_wandb_tags(cfg)

        assert "phase:baseline" in tags
        assert "experiment_class:classical_baselines" in tags
        assert "revision:thesis-v1" in tags
        assert "label_fraction:0.1" in tags
        assert "ablation:label-efficiency" in tags
        assert any(tag.startswith("rerun-reason:") and len(tag) <= 64 for tag in tags)

    def test_build_per_seed_df_keeps_distinct_hp_ablation_configs(self):
        """Distinct upstream HP-ablation configs must not deduplicate together."""
        from scripts.export_results import build_per_seed_df

        runs = [
            DummyWandbRun(
                run_id="hp_lr_2e4",
                config={
                    "experiment_class": "hp_robustness",
                    "dataset": "miiv",
                    "paradigm": "mae",
                    "seed": 42,
                    "output_dir": (
                        "outputs/hp_robustness/" "finetune_mae_mortality_24h_miiv_seed42_lr00002"
                    ),
                    "encoder": {"d_model": 64, "n_layers": 2},
                    "training": {"freeze_encoder": False},
                    "task": {"task_name": "mortality_24h"},
                },
                tags=["phase:finetune"],
                name="finetune_mae_mortality_24h_miiv_seed42_lr00002",
                created_at="2026-04-07T00:00:01",
            ),
            DummyWandbRun(
                run_id="hp_lr_5e4",
                config={
                    "experiment_class": "hp_robustness",
                    "dataset": "miiv",
                    "paradigm": "mae",
                    "seed": 42,
                    "output_dir": (
                        "outputs/hp_robustness/" "finetune_mae_mortality_24h_miiv_seed42_lr00005"
                    ),
                    "encoder": {"d_model": 64, "n_layers": 2},
                    "training": {"freeze_encoder": False},
                    "task": {"task_name": "mortality_24h"},
                },
                tags=["phase:finetune"],
                name="finetune_mae_mortality_24h_miiv_seed42_lr00005",
                created_at="2026-04-07T00:00:02",
            ),
        ]

        df = build_per_seed_df(runs)

        assert len(df) == 2
        assert set(df["experiment_class"]) == {"hp_robustness"}
        assert sorted(df["upstream_pretrain_lr"].tolist()) == [2e-4, 5e-4]

    def test_build_aggregated_df_merges_hp_robustness_across_seeds(self):
        """HP robustness rows should merge across seeds by scientific fingerprint."""
        from scripts.export_results import build_aggregated_df

        per_seed_df = pd.DataFrame(
            [
                {
                    "wandb_run_id": f"run_{i}",
                    "experiment_class": "hp_robustness",
                    "experiment_type": "hp_robustness",
                    "experiment_subtype": "lr_sensitivity",
                    "paradigm": "mae",
                    "dataset": "miiv",
                    "task": "mortality_24h",
                    "seed": seed,
                    "protocol": "B",
                    "label_fraction": 1.0,
                    "model_size": "default",
                    "source_dataset": None,
                    "phase": "finetune",
                    "upstream_pretrain_lr": 2e-4,
                    "upstream_pretrain_mask_ratio": None,
                }
                for i, seed in enumerate([42, 123, 456, 789, 1011])
            ]
        )

        agg = build_aggregated_df(per_seed_df)

        assert len(agg) == 1
        assert agg.iloc[0]["experiment_class"] == "hp_robustness"
        assert agg.iloc[0]["experiment_subtype"] == "lr_sensitivity"
        assert agg.iloc[0]["n_seeds"] == 5
        assert set(yaml.safe_load(agg.iloc[0]["experiment_class_list"])) == {"hp_robustness"}

    def test_build_per_seed_df_raises_on_ambiguous_core_collision(self):
        """Distinct configs that a fingerprint would collapse should fail closed."""
        from scripts.export_results import build_per_seed_df

        runs = [
            DummyWandbRun(
                run_id="core_lr_1e4",
                config={
                    "experiment_class": "core_ssl_benchmark",
                    "dataset": "miiv",
                    "paradigm": "mae",
                    "seed": 42,
                    "encoder": {"d_model": 64, "n_layers": 2},
                    "training": {"freeze_encoder": False},
                    "optimizer": {"lr": 1e-4},
                    "task": {"task_name": "mortality_24h"},
                },
                tags=["phase:finetune"],
                name="finetune_mae_mortality_24h_miiv_seed42_lr1e4",
                created_at="2026-04-07T00:00:01",
            ),
            DummyWandbRun(
                run_id="core_lr_3e4",
                config={
                    "experiment_class": "core_ssl_benchmark",
                    "dataset": "miiv",
                    "paradigm": "mae",
                    "seed": 42,
                    "encoder": {"d_model": 64, "n_layers": 2},
                    "training": {"freeze_encoder": False},
                    "optimizer": {"lr": 3e-4},
                    "task": {"task_name": "mortality_24h"},
                },
                tags=["phase:finetune"],
                name="finetune_mae_mortality_24h_miiv_seed42_lr3e4",
                created_at="2026-04-07T00:00:02",
            ),
        ]

        with pytest.raises(RuntimeError, match="Ambiguous export fingerprint"):
            build_per_seed_df(runs)


class TestExperimentRunnerWandbOverrides:
    """Tests for clean project/entity overrides in the experiment runner."""

    def test_apply_wandb_target_injects_project_and_entity(self):
        from scripts.internal.run_experiments import Run, apply_wandb_target

        run = Run(
            id="core_ssl_benchmark_supervised_mortality_24h_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/supervised_mortality_24h_miiv_seed42",
        )

        result = apply_wandb_target([run], project="slices-thesis", entity="hannes-ill")
        assert result[0].extra_overrides["project_name"] == "slices-thesis"
        assert result[0].extra_overrides["logging.wandb_project"] == "slices-thesis"
        assert result[0].extra_overrides["logging.wandb_entity"] == "hannes-ill"

    def test_launch_commit_mismatch_resets_stale_completed_state(self):
        from scripts.internal.run_experiments import Run, reset_state_for_launch_commit_mismatch

        run = Run(
            id="core_ssl_benchmark_rev-thesis-v1_supervised_mortality_24h_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/supervised_mortality_24h_miiv_seed42",
            task="mortality_24h",
            extra_overrides={"+launch_commit": "newcommit"},
        )
        state = {
            "version": 1,
            "runs": {
                run.id: {
                    "status": "completed",
                    "launch_commit": "oldcommit",
                    "command": "uv run python scripts/training/supervised.py",
                }
            },
        }

        reset_state_for_launch_commit_mismatch([run], state)

        assert state["runs"][run.id]["status"] == "pending"
        assert "launch_commit changed" in state["runs"][run.id]["reset_reason"]

    def test_launch_commit_mismatch_quarantines_stale_resume_artifacts(self, tmp_path):
        from scripts.internal.run_experiments import Run, reset_state_for_launch_commit_mismatch

        output_dir = tmp_path / "supervised_mortality_24h_miiv_seed42"
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "last.ckpt").write_text("old checkpoint")

        run = Run(
            id="core_ssl_benchmark_rev-thesis-v1_supervised_mortality_24h_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir=str(output_dir),
            task="mortality_24h",
            extra_overrides={"+launch_commit": "newcommit"},
        )
        state = {
            "version": 1,
            "runs": {
                run.id: {
                    "status": "completed",
                    "launch_commit": "oldcommit",
                    "command": "uv run python scripts/training/supervised.py",
                }
            },
        }

        reset_state_for_launch_commit_mismatch([run], state)

        state_entry = state["runs"][run.id]
        quarantined_output_dir = Path(state_entry["quarantined_output_dir"])
        assert state_entry["status"] == "pending"
        assert not output_dir.exists()
        assert (quarantined_output_dir / "checkpoints" / "last.ckpt").read_text() == (
            "old checkpoint"
        )
        assert not any(part.startswith("ckpt_path=") for part in run.build_command({}))

    def test_wandb_target_mismatch_resets_completed_state_and_output(self, tmp_path):
        from scripts.internal.run_experiments import (
            Run,
            apply_wandb_target,
            reset_state_for_launch_identity_mismatch,
        )

        output_dir = tmp_path / "supervised_mortality_24h_miiv_seed42"
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "last.ckpt").write_text("old checkpoint")

        run = Run(
            id="core_ssl_benchmark_rev-thesis-v1_supervised_mortality_24h_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir=str(output_dir),
            task="mortality_24h",
            extra_overrides={"revision": "thesis-v1"},
        )
        apply_wandb_target([run], project="new-project", entity="team-a")
        state = {
            "version": 1,
            "runs": {
                run.id: {
                    "status": "completed",
                    "command": (
                        "uv run python scripts/training/supervised.py "
                        "revision=thesis-v1 project_name=old-project "
                        "logging.wandb_project=old-project logging.wandb_entity=team-a"
                    ),
                }
            },
        }

        reset_state_for_launch_identity_mismatch([run], state)

        state_entry = state["runs"][run.id]
        assert state_entry["status"] == "pending"
        assert "launch identity changed" in state_entry["reset_reason"]
        assert not output_dir.exists()
        assert Path(state_entry["quarantined_output_dir"]).exists()

    def test_scoped_run_does_not_resume_unmarked_checkpoint(self, tmp_path):
        from scripts.internal.run_experiments import Run

        output_dir = tmp_path / "pretrain_mae_miiv_seed42"
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "last.ckpt").write_text("old checkpoint")

        run = Run(
            id="core_ssl_benchmark_rev-thesis-v1_pretrain_mae_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir=str(output_dir),
            extra_overrides={"revision": "thesis-v1"},
        )

        assert not any(part.startswith("ckpt_path=") for part in run.build_command({}))

    def test_scoped_run_resumes_matching_marked_checkpoint(self, tmp_path):
        from scripts.internal.run_experiments import Run, _write_output_launch_identity

        output_dir = tmp_path / "pretrain_mae_miiv_seed42"
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        last_ckpt = checkpoint_dir / "last.ckpt"
        last_ckpt.write_text("checkpoint")

        run = Run(
            id="core_ssl_benchmark_rev-thesis-v1_pretrain_mae_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir=str(output_dir),
            extra_overrides={"revision": "thesis-v1"},
        )
        _write_output_launch_identity(run)

        assert f"ckpt_path={last_ckpt}" in run.build_command({})

    def test_revised_status_handles_base_and_revision_groups(self, monkeypatch, capsys):
        import scripts.internal.run_experiments as runner

        base = runner.Run(
            id="core_ssl_benchmark_supervised_mortality_24h_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/supervised_mortality_24h_miiv_seed42",
            task="mortality_24h",
        )
        revised_id = "core_ssl_benchmark_rev-thesis-v1_supervised_mortality_24h_miiv_seed42"

        monkeypatch.setattr(runner, "generate_all_runs", lambda: [base])
        monkeypatch.setattr(
            runner,
            "load_state",
            lambda: {
                "version": 1,
                "runs": {
                    revised_id: {"status": "completed"},
                },
            },
        )

        runner.print_status()

        output = capsys.readouterr().out
        assert "core_ssl_benchmark" in output
        assert "core_ssl_benchmark/thesis-v1" in output

    def test_revised_status_filter_uses_revisioned_generated_ids(self, monkeypatch, capsys):
        import scripts.internal.run_experiments as runner

        base = runner.Run(
            id="core_ssl_benchmark_supervised_mortality_24h_miiv_seed42",
            run_key="supervised_mortality_24h_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/supervised_mortality_24h_miiv_seed42",
            task="mortality_24h",
        )
        revised_id = "core_ssl_benchmark_rev-thesis-v1_supervised_mortality_24h_miiv_seed42"

        monkeypatch.setattr(runner, "generate_all_runs", lambda: [base])
        monkeypatch.setattr(
            runner,
            "load_state",
            lambda: {"version": 1, "runs": {revised_id: {"status": "completed"}}},
        )

        runner.print_status(revision_filter="thesis-v1")

        output = capsys.readouterr().out
        assert "core_ssl_benchmark/thesis-v1" in output
        assert "    1 |    1 |" in output
        assert "core_ssl_benchmark       |" not in output

    def test_main_run_requires_revision_and_project(self, monkeypatch):
        import scripts.internal.run_experiments as runner

        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_experiments.py", "run", "--experiment-class", "core_ssl_benchmark"],
        )

        with pytest.raises(SystemExit) as excinfo:
            runner.main()

        assert excinfo.value.code == 2

    def test_main_retry_requires_revision_scope_and_project(self, monkeypatch):
        import scripts.internal.run_experiments as runner

        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        monkeypatch.setattr(sys, "argv", ["run_experiments.py", "retry", "--failed"])

        with pytest.raises(SystemExit) as excinfo:
            runner.main()

        assert excinfo.value.code == 2

    def test_cmd_run_respects_requested_class_order(self, monkeypatch):
        import scripts.internal.run_experiments as runner

        capacity = runner.Run(
            id="capacity_study_supervised_mortality_24h_miiv_seed42",
            experiment_class="capacity_study",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir="outputs/capacity_study/supervised_mortality_24h_miiv_seed42",
            task="mortality_24h",
        )
        classical = runner.Run(
            id="classical_baselines_supervised_mortality_24h_miiv_seed789",
            experiment_class="classical_baselines",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=789,
            output_dir="outputs/classical_baselines/supervised_mortality_24h_miiv_seed789",
            task="mortality_24h",
        )
        scheduled = {}

        monkeypatch.setattr(runner, "generate_all_runs", lambda: [capacity, classical])
        monkeypatch.setattr(runner, "load_state", lambda: {"version": 1, "runs": {}})
        monkeypatch.setattr(runner, "recover_stale_running", lambda state: None)
        monkeypatch.setattr(
            runner,
            "run_scheduler",
            lambda runs, state, parallel, dry_run: scheduled.setdefault("runs", runs),
        )

        args = SimpleNamespace(
            experiment_class=["classical_baselines", "capacity_study"],
            revision=None,
            reason=None,
            project=None,
            entity=None,
            parallel=1,
            dry_run=True,
        )

        runner.cmd_run(args)

        assert [run.experiment_class for run in scheduled["runs"]] == [
            "classical_baselines",
            "capacity_study",
        ]

    def test_pretrain_and_finetune_commands_resume_from_last_checkpoint(self, tmp_path):
        from scripts.internal.run_experiments import Run

        pretrain_dir = tmp_path / "pretrain"
        finetune_dir = tmp_path / "finetune"
        (pretrain_dir / "checkpoints").mkdir(parents=True)
        (finetune_dir / "checkpoints").mkdir(parents=True)
        (pretrain_dir / "checkpoints" / "last.ckpt").write_text("checkpoint")
        (finetune_dir / "checkpoints" / "last.ckpt").write_text("checkpoint")

        pretrain = Run(
            id="pretrain",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir=str(pretrain_dir),
        )
        finetune = Run(
            id="finetune",
            experiment_class="core_ssl_benchmark",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir=str(finetune_dir),
            depends_on=[pretrain.id],
            task="mortality_24h",
            freeze_encoder=False,
        )

        pretrain_cmd = pretrain.build_command({})
        assert f"ckpt_path={pretrain_dir / 'checkpoints' / 'last.ckpt'}" in pretrain_cmd
        assert f"ckpt_path={finetune_dir / 'checkpoints' / 'last.ckpt'}" in finetune.build_command(
            {pretrain.id: pretrain}
        )

    def test_recover_stale_running_resets_pid_reuse(self, monkeypatch):
        import scripts.internal.run_experiments as runner

        state = {
            "version": 1,
            "runs": {"run-a": {"status": "running", "pid": 123, "command": "uv run expected"}},
        }

        monkeypatch.setattr(runner, "is_pid_alive", lambda pid: True)
        monkeypatch.setattr(runner, "_pid_matches_command", lambda pid, command: False)

        runner.recover_stale_running(state)

        assert state["runs"]["run-a"]["status"] == "pending"
        assert "pid" not in state["runs"]["run-a"]
        assert "command" not in state["runs"]["run-a"]

    def test_scheduler_exit_code_flags_pending_and_running_runs(self):
        import scripts.internal.run_experiments as runner

        runs = [
            runner.Run(
                id="pending-run",
                experiment_class="core_ssl_benchmark",
                run_type="pretrain",
                paradigm="mae",
                dataset="miiv",
                seed=42,
                output_dir="outputs/pending",
            ),
            runner.Run(
                id="running-run",
                experiment_class="core_ssl_benchmark",
                run_type="pretrain",
                paradigm="mae",
                dataset="miiv",
                seed=123,
                output_dir="outputs/running",
            ),
        ]
        state = {
            "version": 1,
            "runs": {
                "pending-run": {"status": "pending"},
                "running-run": {"status": "running"},
            },
        }

        assert runner._scheduler_exit_code(runs, state) == 1

    def test_runner_no_longer_exposes_inherited_tag_command(self):
        import scripts.internal.run_experiments as runner

        assert not hasattr(runner, "cmd_tag")

    def test_wandb_logger_includes_model_size_metadata(self, monkeypatch):
        import slices.training.utils as training_utils

        captured = {}

        class DummyConfig:
            def update(self, cfg):
                captured["config_update"] = cfg

        class DummyExperiment:
            config = DummyConfig()

        class DummyWandbLogger:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                self.experiment = DummyExperiment()

        monkeypatch.setattr(training_utils, "WandbLogger", DummyWandbLogger)

        cfg = OmegaConf.create(
            {
                "output_dir": "outputs/capacity_study/example",
                "experiment_class": "capacity_study",
                "model_size": "medium",
                "source_dataset": "eicu",
                "label_fraction": 0.1,
                "training": {"freeze_encoder": False},
                "logging": {
                    "use_wandb": True,
                    "wandb_project": "slices",
                    "wandb_entity": None,
                    "run_name": "capacity_study_finetune_miiv_mae_mortality_24h_seed42",
                    "wandb_group": "finetune_miiv_mae_mortality_24h",
                    "wandb_tags": ["phase:finetune"],
                },
            }
        )

        logger = training_utils.setup_wandb_logger(cfg)

        assert isinstance(logger, DummyWandbLogger)
        assert captured["kwargs"]["name"] == (
            "capacity_study_finetune_miiv_mae_mortality_24h_seed42_from_eicu_medium_frac01"
        )
        assert (
            captured["kwargs"]["group"] == "finetune_miiv_mae_mortality_24h_from_eicu_medium_frac01"
        )
        assert "experiment_class:capacity_study" in captured["kwargs"]["tags"]
        assert "model_size:medium" in captured["kwargs"]["tags"]
        assert "source_dataset:eicu" in captured["kwargs"]["tags"]
        assert "ablation:label-efficiency" in captured["kwargs"]["tags"]
        assert "ablation:transfer" in captured["kwargs"]["tags"]

    def test_transfer_finetune_command_propagates_source_dataset(self):
        from scripts.internal.run_experiments import Run

        pretrain = Run(
            id="pretrain_mae_miiv_seed42",
            experiment_class="cross_dataset_transfer",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/cross_dataset_transfer/pretrain_mae_miiv_seed42",
        )
        finetune = Run(
            id="finetune_mae_eicu_seed42",
            experiment_class="cross_dataset_transfer",
            run_type="finetune",
            paradigm="mae",
            dataset="eicu",
            seed=42,
            output_dir="outputs/cross_dataset_transfer/finetune_mae_mortality_24h_eicu_seed42_from_miiv",
            depends_on=[pretrain.id],
            task="mortality_24h",
            freeze_encoder=False,
            source_dataset="miiv",
        )

        cmd = finetune.build_command({pretrain.id: pretrain, finetune.id: finetune})
        assert "+source_dataset=miiv" in cmd

    def test_protocol_a_finetune_command_uses_strict_linear_probe(self):
        from scripts.internal.run_experiments import Run

        pretrain = Run(
            id="pretrain_mae_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/pretrain_mae_miiv_seed42",
        )
        probe = Run(
            id="probe_mae_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/probe_mae_mortality_24h_miiv_seed42",
            depends_on=[pretrain.id],
            task="mortality_24h",
            freeze_encoder=True,
        )

        cmd = probe.build_command({pretrain.id: pretrain, probe.id: probe})

        assert "training.freeze_encoder=true" in cmd
        assert "task.head_type=linear" in cmd
        assert "task.hidden_dims=[]" in cmd
        assert "task.dropout=0.0" in cmd
        assert "protocol=A" in cmd
        assert "+protocol=A" not in cmd

    def test_protocol_b_finetune_command_keeps_task_mlp_head(self):
        from scripts.internal.run_experiments import Run

        pretrain = Run(
            id="pretrain_mae_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/pretrain_mae_miiv_seed42",
        )
        finetune = Run(
            id="finetune_mae_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/finetune_mae_mortality_24h_miiv_seed42",
            depends_on=[pretrain.id],
            task="mortality_24h",
            freeze_encoder=False,
        )

        cmd = finetune.build_command({pretrain.id: pretrain, finetune.id: finetune})

        assert "training.freeze_encoder=false" in cmd
        assert "task.head_type=linear" not in cmd
        assert "protocol=B" in cmd
        assert "+protocol=B" not in cmd

    def test_xgboost_command_does_not_load_protocol_config_group(self):
        from scripts.internal.run_experiments import Run

        run = Run(
            id="xgboost_miiv_seed42",
            experiment_class="classical_baselines",
            run_type="xgboost",
            paradigm="xgboost",
            dataset="miiv",
            seed=42,
            output_dir="outputs/classical_baselines/xgboost_mortality_24h_miiv_seed42",
            task="mortality_24h",
        )

        cmd = run.build_command({run.id: run})

        assert "+protocol=B" not in cmd
        assert "protocol=B" not in cmd

    def test_manual_finetune_protocol_configs_encode_safe_defaults(self):
        protocol_a = OmegaConf.load("configs/protocol/a.yaml")
        protocol_b = OmegaConf.load("configs/protocol/b.yaml")

        assert protocol_a.protocol == "A"
        assert protocol_a.training.freeze_encoder is True
        assert protocol_a.training.max_epochs == 50
        assert protocol_a.training.early_stopping_patience == 10
        assert protocol_a.optimizer.lr == pytest.approx(1.0e-4)
        assert protocol_a.task.head_type == "linear"
        assert list(protocol_a.task.hidden_dims) == []
        assert protocol_a.task.dropout == 0.0

        assert protocol_b.protocol == "B"
        assert protocol_b.training.freeze_encoder is False
        assert protocol_b.training.max_epochs == 100
        assert protocol_b.training.early_stopping_patience == 10
        assert protocol_b.optimizer.lr == pytest.approx(3.0e-4)
        assert protocol_b.task.head_type == "mlp"
        assert list(protocol_b.task.hidden_dims) == [64]
        assert protocol_b.task.dropout == 0.3


class TestExperimentRunnerRetry:
    """Tests for retry scoping and dependency preservation."""

    def test_cmd_retry_respects_class_filter_but_keeps_dependencies(self, monkeypatch):
        import scripts.internal.run_experiments as runner

        dependency = runner.Run(
            id="dep_pretrain",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/pretrain_mae_miiv_seed42",
        )
        target = runner.Run(
            id="target_finetune",
            experiment_class="label_efficiency",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/label_efficiency/finetune_mae_miiv_seed42",
            depends_on=[dependency.id],
            task="mortality_24h",
        )
        other = runner.Run(
            id="other_failed",
            experiment_class="classical_baselines",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir="outputs/classical_baselines/supervised_miiv_seed42",
            task="mortality_24h",
        )

        state = {
            "version": 1,
            "runs": {
                dependency.id: {"status": "completed"},
                target.id: {"status": "failed"},
                other.id: {"status": "failed"},
            },
        }
        scheduled = {}

        monkeypatch.setattr(runner, "generate_all_runs", lambda: [dependency, target, other])
        monkeypatch.setattr(runner, "load_state", lambda: state)
        monkeypatch.setattr(runner, "recover_stale_running", lambda state: None)
        monkeypatch.setattr(runner, "save_state", lambda state: None)
        monkeypatch.setattr(
            runner,
            "run_scheduler",
            lambda runs, state, parallel, dry_run: scheduled.setdefault("runs", runs),
        )

        args = SimpleNamespace(
            experiment_class=["label_efficiency"],
            failed=True,
            skipped=False,
            revision=None,
            reason=None,
            project=None,
            entity=None,
            parallel=1,
        )

        runner.cmd_retry(args)

        scheduled_ids = [run.id for run in scheduled["runs"]]
        assert target.id in scheduled_ids
        assert dependency.id in scheduled_ids
        assert other.id not in scheduled_ids

    def test_cmd_retry_skipped_reports_failed_dependencies(self, monkeypatch, capsys):
        import scripts.internal.run_experiments as runner

        dependency = runner.Run(
            id="dep_pretrain",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/pretrain_mae_miiv_seed42",
        )
        target = runner.Run(
            id="target_finetune",
            experiment_class="label_efficiency",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/label_efficiency/finetune_mae_miiv_seed42",
            depends_on=[dependency.id],
            task="mortality_24h",
        )

        state = {
            "version": 1,
            "runs": {
                dependency.id: {"status": "failed"},
                target.id: {"status": "skipped"},
            },
        }

        monkeypatch.setattr(runner, "generate_all_runs", lambda: [dependency, target])
        monkeypatch.setattr(runner, "load_state", lambda: state)
        monkeypatch.setattr(runner, "recover_stale_running", lambda state: None)
        monkeypatch.setattr(
            runner,
            "run_scheduler",
            lambda runs, state, parallel, dry_run: pytest.fail("scheduler should not run"),
        )

        args = SimpleNamespace(
            experiment_class=["label_efficiency"],
            failed=False,
            skipped=True,
            revision=None,
            reason=None,
            project=None,
            entity=None,
            parallel=1,
        )

        with pytest.raises(SystemExit):
            runner.cmd_retry(args)

        output = capsys.readouterr().out
        assert "dep_pretrain [failed]" in output
        assert (
            "uv run python scripts/internal/run_experiments.py retry --failed --skipped" in output
        )
        assert state["runs"][target.id]["status"] == "skipped"
        assert state["runs"][dependency.id]["status"] == "failed"

    def test_cmd_retry_skipped_resets_failed_dependencies_when_requested(self, monkeypatch):
        import scripts.internal.run_experiments as runner

        dependency = runner.Run(
            id="dep_pretrain",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/pretrain_mae_miiv_seed42",
        )
        target = runner.Run(
            id="target_finetune",
            experiment_class="label_efficiency",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/label_efficiency/finetune_mae_miiv_seed42",
            depends_on=[dependency.id],
            task="mortality_24h",
        )

        state = {
            "version": 1,
            "runs": {
                dependency.id: {"status": "failed"},
                target.id: {"status": "skipped"},
            },
        }
        scheduled = {}

        monkeypatch.setattr(runner, "generate_all_runs", lambda: [dependency, target])
        monkeypatch.setattr(runner, "load_state", lambda: state)
        monkeypatch.setattr(runner, "recover_stale_running", lambda state: None)
        monkeypatch.setattr(runner, "save_state", lambda state: None)
        monkeypatch.setattr(
            runner,
            "run_scheduler",
            lambda runs, state, parallel, dry_run: scheduled.setdefault("runs", runs),
        )

        args = SimpleNamespace(
            experiment_class=["label_efficiency"],
            failed=True,
            skipped=True,
            revision=None,
            reason=None,
            project=None,
            entity=None,
            parallel=1,
        )

        runner.cmd_retry(args)

        scheduled_ids = [run.id for run in scheduled["runs"]]
        assert dependency.id in scheduled_ids
        assert target.id in scheduled_ids
        assert state["runs"][dependency.id]["status"] == "pending"
        assert state["runs"][target.id]["status"] == "pending"

    def test_cmd_retry_revises_dependencies_for_revision_scoped_retry(self, monkeypatch):
        import scripts.internal.run_experiments as runner

        dependency = runner.Run(
            id="core_ssl_benchmark_pretrain_mae_miiv_seed42",
            run_key="pretrain_mae_miiv_seed42",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/pretrain_mae_miiv_seed42",
        )
        target = runner.Run(
            id="label_efficiency_finetune_mae_mortality_24h_miiv_seed42",
            run_key="finetune_mae_mortality_24h_miiv_seed42",
            experiment_class="label_efficiency",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/label_efficiency/finetune_mae_mortality_24h_miiv_seed42",
            depends_on=[dependency.id],
            task="mortality_24h",
        )

        revised_dep_id = "core_ssl_benchmark_rev-thesis-v1_pretrain_mae_miiv_seed42"
        revised_target_id = "label_efficiency_rev-thesis-v1_finetune_mae_mortality_24h_miiv_seed42"
        state = {
            "version": 1,
            "runs": {
                revised_dep_id: {"status": "completed"},
                revised_target_id: {"status": "failed"},
            },
        }
        scheduled = {}

        monkeypatch.setattr(runner, "generate_all_runs", lambda: [dependency, target])
        monkeypatch.setattr(runner, "load_state", lambda: state)
        monkeypatch.setattr(runner, "recover_stale_running", lambda state: None)
        monkeypatch.setattr(runner, "save_state", lambda state: None)
        monkeypatch.setattr(
            runner,
            "run_scheduler",
            lambda runs, state, parallel, dry_run: scheduled.setdefault("runs", runs),
        )

        args = SimpleNamespace(
            experiment_class=["label_efficiency"],
            failed=True,
            skipped=False,
            revision="thesis-v1",
            reason="retry downstream failure",
            project=None,
            entity=None,
            parallel=1,
        )

        runner.cmd_retry(args)

        scheduled_by_id = {run.id: run for run in scheduled["runs"]}
        assert revised_dep_id in scheduled_by_id
        assert revised_target_id in scheduled_by_id
        assert scheduled_by_id[revised_target_id].depends_on == [revised_dep_id]
        assert scheduled_by_id[revised_dep_id].extra_overrides["revision"] == "thesis-v1"
        assert scheduled_by_id[revised_target_id].extra_overrides["revision"] == "thesis-v1"

    def test_select_ready_runs_prioritizes_pretrains_with_slot_budget(self):
        import scripts.internal.run_experiments as runner

        pretrain = runner.Run(
            id="pretrain",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/pretrain_mae_miiv_seed42",
        )
        finetune = runner.Run(
            id="finetune",
            experiment_class="core_ssl_benchmark",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/finetune_mae_miiv_seed42",
            task="mortality_24h",
        )

        selected = runner._select_ready_runs(
            [finetune, pretrain],
            active_run_ids=set(),
            runs_by_id={pretrain.id: pretrain, finetune.id: finetune},
            slot_budget=4,
        )

        assert [run.id for run in selected] == [pretrain.id]

    def test_select_ready_runs_does_not_mix_pretrain_with_active_gpu_work(self):
        import scripts.internal.run_experiments as runner

        pretrain = runner.Run(
            id="pretrain",
            experiment_class="core_ssl_benchmark",
            run_type="pretrain",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/pretrain_mae_miiv_seed42",
        )
        finetune = runner.Run(
            id="finetune",
            experiment_class="core_ssl_benchmark",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/finetune_mae_miiv_seed42",
            task="mortality_24h",
        )

        selected = runner._select_ready_runs(
            [pretrain],
            active_run_ids={finetune.id},
            runs_by_id={pretrain.id: pretrain, finetune.id: finetune},
            slot_budget=4,
        )

        assert selected == []

    def test_run_scheduler_returns_nonzero_for_failed_run(self, tmp_path, monkeypatch):
        import scripts.internal.run_experiments as runner

        failed_run = runner.Run(
            id="failed",
            experiment_class="core_ssl_benchmark",
            run_type="supervised",
            paradigm="supervised",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/supervised_miiv_seed42",
            task="mortality_24h",
        )
        state = {"version": 1, "runs": {failed_run.id: {"status": "failed"}}}

        monkeypatch.setattr(runner, "LOG_DIR", tmp_path)
        monkeypatch.setattr(runner, "save_state", lambda state: None)

        exit_code = runner.run_scheduler([failed_run], state, parallel=1, dry_run=False)

        assert exit_code == 1

    def test_scheduler_exit_code_nonzero_for_dependency_skips(self):
        import scripts.internal.run_experiments as runner

        skipped_run = runner.Run(
            id="skipped",
            experiment_class="core_ssl_benchmark",
            run_type="finetune",
            paradigm="mae",
            dataset="miiv",
            seed=42,
            output_dir="outputs/core_ssl_benchmark/finetune_mae_miiv_seed42",
            task="mortality_24h",
        )
        state = {"version": 1, "runs": {skipped_run.id: {"status": "skipped"}}}

        assert runner._scheduler_exit_code([skipped_run], state) == 1


class TestExportStatisticalScope:
    """Regression tests for classical-baseline significance comparisons."""

    def test_build_statistical_tests_df_includes_xgboost_comparisons(self):
        from scripts.export_results import build_statistical_tests_df

        rows = []
        for paradigm, phase, protocol, offset in [
            ("xgboost", "baseline", "B", 0.08),
            ("gru_d", "baseline", "B", 0.03),
            ("supervised", "supervised", "B", 0.0),
        ]:
            for seed in [42, 123]:
                for task, base in [("mortality_24h", 0.30), ("aki_kdigo", 0.25)]:
                    rows.append(
                        {
                            "experiment_class": "classical_baselines",
                            "experiment_type": "classical_baselines",
                            "paradigm": paradigm,
                            "dataset": "miiv",
                            "task": task,
                            "seed": seed,
                            "protocol": protocol,
                            "label_fraction": 1.0,
                            "model_size": "default",
                            "source_dataset": None,
                            "phase": phase,
                            "test/auprc": base + offset + (seed / 100000.0),
                        }
                    )

        stats_df = build_statistical_tests_df(pd.DataFrame(rows))
        pairs = {
            tuple(sorted((row["paradigm_a"], row["paradigm_b"]))) for _, row in stats_df.iterrows()
        }

        assert ("gru_d", "supervised") in pairs
        assert ("supervised", "xgboost") in pairs
        assert ("gru_d", "xgboost") in pairs
        per_task = stats_df[stats_df["comparison_scope"] == "per_task"]
        assert set(per_task["task"]) == {"mortality_24h", "aki_kdigo"}


class TestXGBoostBaseline:
    """Regression tests for XGBoost configuration choices."""

    def test_xgboost_uses_primary_metric_for_early_stopping(self):
        from scripts.training.xgboost_baseline import _xgboost_eval_metric

        assert _xgboost_eval_metric("binary") == "aucpr"
        assert _xgboost_eval_metric("regression") == "mae"

    def test_xgboost_binary_ece_matches_uniform_bin_definition(self):
        from scripts.training.xgboost_baseline import _binary_ece

        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.3, 0.7, 0.9]

        assert _binary_ece(y_true, y_prob, n_bins=2) == pytest.approx(0.2)

    def test_xgboost_config_caps_cpu_parallelism(self):
        cfg = OmegaConf.load("configs/xgboost.yaml")

        assert cfg.xgboost.n_jobs == 4

    def test_xgboost_balanced_scale_pos_weight_uses_native_ratio(self):
        from scripts.training.xgboost_baseline import _resolve_scale_pos_weight

        labels = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)

        assert _resolve_scale_pos_weight("balanced", labels.numpy()) == pytest.approx(4.0)


class TestTrainingScriptClassWeighting:
    """Regression tests for entrypoint-level class weight resolution."""

    @pytest.mark.parametrize(
        "module_name",
        ["scripts.training.finetune", "scripts.training.supervised"],
    )
    def test_balanced_class_weight_uses_train_split_stats(self, monkeypatch, module_name):
        """'balanced' must resolve from train-split stats before module construction."""
        module = importlib.import_module(module_name)
        captured = {}

        class DummyDataModule:
            def __init__(self, *args, **kwargs):
                pass

            def setup(self):
                return None

            def get_feature_dim(self):
                return 3

            def get_seq_length(self):
                return 24

            def get_split_info(self):
                return {
                    "train_patients": 5,
                    "train_stays": 10,
                    "val_patients": 2,
                    "val_stays": 4,
                    "test_patients": 2,
                    "test_stays": 4,
                }

            def get_label_statistics(self):
                return {
                    "mortality_24h": {
                        "total": 100,
                        "positive": 20,
                        "negative": 80,
                        "prevalence": 0.2,
                    }
                }

            def get_train_label_statistics(self, use_full_train: bool = False):
                return {
                    "mortality_24h": {
                        "total": 10,
                        "positive": 1,
                        "negative": 9,
                        "prevalence": 0.1,
                    }
                }

        class StopAfterCaptureError(Exception):
            pass

        def fake_finetune_module(config, checkpoint_path=None, pretrain_checkpoint_path=None):
            captured["class_weight"] = list(config.training.class_weight)
            captured["d_input"] = config.encoder.d_input
            captured["max_seq_length"] = config.encoder.max_seq_length
            raise StopAfterCaptureError

        monkeypatch.setattr(module, "validate_data_prerequisites", lambda *args, **kwargs: None)
        monkeypatch.setattr(module.pl, "seed_everything", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "ICUDataModule", DummyDataModule)
        monkeypatch.setattr(module, "FineTuneModule", fake_finetune_module)
        if hasattr(module, "_detect_paradigm_from_checkpoint"):
            monkeypatch.setattr(module, "_detect_paradigm_from_checkpoint", lambda *a, **k: None)

        cfg = OmegaConf.create(
            {
                "dataset": "miiv",
                "seed": 42,
                "paradigm": "mae",
                "checkpoint": None,
                "pretrain_checkpoint": "dummy.ckpt",
                "data": {
                    "processed_dir": "/tmp/processed",
                    "num_workers": 0,
                },
                "task": {
                    "task_name": "mortality_24h",
                    "task_type": "binary",
                    "head_type": "mlp",
                    "hidden_dims": [16],
                    "dropout": 0.0,
                    "activation": "relu",
                },
                "encoder": {
                    "name": "transformer",
                    "d_input": 0,
                    "max_seq_length": 0,
                },
                "training": {
                    "batch_size": 8,
                    "class_weight": "balanced",
                    "freeze_encoder": False,
                },
                "optimizer": {
                    "name": "adam",
                    "lr": 1e-3,
                },
                "logging": {
                    "use_wandb": False,
                },
            }
        )

        with pytest.raises(StopAfterCaptureError):
            module.main.__wrapped__(cfg)

        expected = [(10 / (2 * 9)) ** 0.5, (10 / (2 * 1)) ** 0.5]
        assert captured["class_weight"] == pytest.approx(expected)
        assert captured["d_input"] == 3
        assert captured["max_seq_length"] == 24

    @pytest.mark.parametrize(
        "module_name",
        ["scripts.training.finetune", "scripts.training.supervised"],
    )
    def test_regression_task_disables_balanced_class_weight(self, monkeypatch, module_name):
        """Regression runs should not derive binary class weights."""
        module = importlib.import_module(module_name)
        captured = {}

        class DummyDataModule:
            def __init__(self, *args, **kwargs):
                pass

            def setup(self):
                return None

            def get_feature_dim(self):
                return 3

            def get_seq_length(self):
                return 24

            def get_split_info(self):
                return {
                    "train_patients": 5,
                    "train_stays": 10,
                    "val_patients": 2,
                    "val_stays": 4,
                    "test_patients": 2,
                    "test_stays": 4,
                }

            def get_label_statistics(self):
                return {
                    "los_remaining": {
                        "task_type": "regression",
                        "total": 10,
                        "mean": 4.2,
                        "std": 1.1,
                        "min": 1.0,
                        "max": 8.0,
                    }
                }

            def get_train_label_statistics(self, use_full_train: bool = False):
                return {
                    "los_remaining": {
                        "task_type": "regression",
                        "total": 10,
                        "mean": 4.2,
                        "std": 1.1,
                        "min": 1.0,
                        "max": 8.0,
                    }
                }

        class StopAfterCaptureError(Exception):
            pass

        def fake_finetune_module(config, checkpoint_path=None, pretrain_checkpoint_path=None):
            captured["class_weight"] = config.training.class_weight
            raise StopAfterCaptureError

        monkeypatch.setattr(module, "validate_data_prerequisites", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "report_and_validate_train_label_support", lambda *a, **k: None)
        monkeypatch.setattr(module.pl, "seed_everything", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "ICUDataModule", DummyDataModule)
        monkeypatch.setattr(module, "FineTuneModule", fake_finetune_module)
        if hasattr(module, "_detect_paradigm_from_checkpoint"):
            monkeypatch.setattr(module, "_detect_paradigm_from_checkpoint", lambda *a, **k: None)

        cfg = OmegaConf.create(
            {
                "dataset": "miiv",
                "seed": 42,
                "paradigm": "mae",
                "checkpoint": None,
                "pretrain_checkpoint": "dummy.ckpt",
                "data": {
                    "processed_dir": "/tmp/processed",
                    "num_workers": 0,
                },
                "task": {
                    "task_name": "los_remaining",
                    "task_type": "regression",
                    "head_type": "mlp",
                    "hidden_dims": [16],
                    "dropout": 0.0,
                    "activation": "relu",
                },
                "encoder": {
                    "name": "transformer",
                    "d_input": 0,
                    "max_seq_length": 0,
                },
                "training": {
                    "batch_size": 8,
                    "class_weight": "balanced",
                    "freeze_encoder": False,
                },
                "optimizer": {
                    "name": "adam",
                    "lr": 1e-3,
                },
                "logging": {
                    "use_wandb": False,
                },
            }
        )

        with pytest.raises(StopAfterCaptureError):
            module.main.__wrapped__(cfg)

        assert captured["class_weight"] is None


class TestSupervisedBaselineMetrics:
    """Regression tests for supervised trivial baselines."""

    def test_classification_baseline_uses_train_majority_class(self):
        module = importlib.import_module("scripts.training.supervised")

        class DummyDataset:
            def __init__(self, labels):
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {"label": float(self.labels[idx])}

        class DummyLoader:
            def __init__(self, dataset):
                self.dataset = dataset

        class DummyDataModule:
            def train_dataloader(self):
                return DummyLoader(DummyDataset([1, 1, 1, 0]))

            def test_dataloader(self):
                return DummyLoader(DummyDataset([0, 0, 0]))

        metrics = module.compute_baseline_metrics(
            DummyDataModule(),
            task_name="mortality_24h",
            task_type="binary",
        )

        assert metrics["baseline/train_positive_ratio"] == pytest.approx(0.75)
        assert metrics["baseline/majority_accuracy"] == pytest.approx(0.0)

    def test_regression_baseline_fits_on_train_labels_only(self):
        module = importlib.import_module("scripts.training.supervised")

        class DummyDataset:
            def __init__(self, labels):
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {"label": float(self.labels[idx])}

        class DummyLoader:
            def __init__(self, dataset):
                self.dataset = dataset

        class DummyDataModule:
            def train_dataloader(self):
                return DummyLoader(DummyDataset([10.0, 10.0]))

            def test_dataloader(self):
                return DummyLoader(DummyDataset([0.0, 0.0]))

        metrics = module.compute_baseline_metrics(
            DummyDataModule(),
            task_name="los_remaining",
            task_type="regression",
        )

        assert metrics["baseline/train_label_mean"] == pytest.approx(10.0)
        assert metrics["baseline/mean_predictor_mse"] == pytest.approx(100.0)
        assert metrics["baseline/median_predictor_mae"] == pytest.approx(10.0)


class TestFinetuneCheckpointParadigmDetection:
    """Regression tests for checkpoint-driven finetune metadata."""

    def test_entrypoint_rejects_ambiguous_checkpoint_sources(self):
        module = importlib.import_module("scripts.training.finetune")
        cfg = OmegaConf.create(
            {
                "checkpoint": "outputs/pretrain/encoder.pt",
                "pretrain_checkpoint": "outputs/pretrain/last.ckpt",
            }
        )

        with pytest.raises(ValueError, match="exactly one checkpoint source"):
            module.main.__wrapped__(cfg)

    def test_pretrain_checkpoint_auto_detects_paradigm(self, monkeypatch, tmp_path):
        module = importlib.import_module("scripts.training.finetune")
        captured = {}

        class DummyDataModule:
            def __init__(self, *args, **kwargs):
                pass

            def setup(self):
                return None

            def get_feature_dim(self):
                return 3

            def get_seq_length(self):
                return 24

            def get_split_info(self):
                return {
                    "train_patients": 5,
                    "train_stays": 10,
                    "val_patients": 2,
                    "val_stays": 4,
                    "test_patients": 2,
                    "test_stays": 4,
                }

            def get_label_statistics(self):
                return {
                    "mortality_24h": {
                        "total": 100,
                        "positive": 20,
                        "negative": 80,
                        "prevalence": 0.2,
                    }
                }

            def get_train_label_statistics(self, use_full_train: bool = False):
                return {
                    "mortality_24h": {
                        "total": 10,
                        "positive": 2,
                        "negative": 8,
                        "prevalence": 0.2,
                    }
                }

        class StopAfterCaptureError(Exception):
            pass

        def fake_finetune_module(config, checkpoint_path=None, pretrain_checkpoint_path=None):
            captured["paradigm"] = config.paradigm
            captured["pretrain_checkpoint_path"] = pretrain_checkpoint_path
            raise StopAfterCaptureError

        pretrain_checkpoint = tmp_path / "pretrain.ckpt"
        torch.save(
            {
                "state_dict": {"encoder.weight": torch.zeros(1)},
                "hyper_parameters": {
                    "config": {
                        "ssl": {"name": "jepa"},
                        "paradigm": "jepa",
                    }
                },
            },
            pretrain_checkpoint,
        )

        monkeypatch.setattr(module, "validate_data_prerequisites", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "report_and_validate_train_label_support", lambda *a, **k: None)
        monkeypatch.setattr(module.pl, "seed_everything", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "ICUDataModule", DummyDataModule)
        monkeypatch.setattr(module, "FineTuneModule", fake_finetune_module)

        cfg = OmegaConf.create(
            {
                "dataset": "miiv",
                "seed": 42,
                "paradigm": "mae",
                "checkpoint": None,
                "pretrain_checkpoint": str(pretrain_checkpoint),
                "data": {
                    "processed_dir": "/tmp/processed",
                    "num_workers": 0,
                },
                "task": {
                    "task_name": "mortality_24h",
                    "task_type": "binary",
                    "prediction_window_hours": 24,
                    "observation_window_hours": 24,
                    "gap_hours": 0,
                    "label_sources": ["stays", "mortality_info"],
                    "label_params": {},
                    "head_type": "mlp",
                    "hidden_dims": [16],
                    "dropout": 0.0,
                    "activation": "relu",
                },
                "encoder": {
                    "name": "transformer",
                    "d_input": 0,
                    "max_seq_length": 0,
                },
                "training": {
                    "batch_size": 8,
                    "class_weight": None,
                    "freeze_encoder": False,
                },
                "optimizer": {
                    "name": "adam",
                    "lr": 1e-3,
                },
                "logging": {
                    "use_wandb": False,
                },
            }
        )

        with pytest.raises(StopAfterCaptureError):
            module.main.__wrapped__(cfg)

        assert captured["paradigm"] == "jepa"
        assert captured["pretrain_checkpoint_path"] == str(pretrain_checkpoint)


class TestExperimentRunnerMatrix:
    """Tests for thesis-final experiment-class matrix coverage."""

    def test_matrix_counts_match_final_experiment_classes(self):
        from collections import Counter

        from scripts.internal.run_experiments import (
            EXPECTED_CLASS_COUNTS,
            generate_all_runs,
            scientific_fingerprint,
        )

        runs = generate_all_runs()
        by_class = Counter(run.experiment_class for run in runs)

        assert len(runs) == 2305
        assert sum(run.experiment_class != "smart_external_reference" for run in runs) == 2170
        assert by_class == EXPECTED_CLASS_COUNTS
        assert len({scientific_fingerprint(run) for run in runs}) == len(runs)
        assert not any(run.task == "mortality" for run in runs)

        non_classical_baselines = [
            run
            for run in runs
            if run.run_type in ("gru_d", "xgboost")
            and run.experiment_class != "classical_baselines"
        ]
        assert non_classical_baselines == []

        contrastive_mask_sweep = [
            run
            for run in runs
            if run.experiment_class == "hp_robustness"
            and run.paradigm == "contrastive"
            and run.experiment_subtype == "mask_ratio_sensitivity"
            and run.run_type == "pretrain"
        ]
        assert contrastive_mask_sweep
        assert all(
            run.extra_overrides["ssl.complementary_masks"] is False
            for run in contrastive_mask_sweep
        )

    def test_capacity_study_expanded_to_five_seeds(self):
        from scripts.internal.run_experiments import SEEDS_EXTENDED, MatrixBuilder

        builder = MatrixBuilder()
        builder.build_capacity_study()

        assert len(builder.runs) == 100
        assert sorted({run.seed for run in builder.runs}) == SEEDS_EXTENDED
        assert {run.model_size for run in builder.runs} == {"medium", "large"}

    def test_ts2vec_extension_includes_both_protocols(self):
        from scripts.internal.run_experiments import MatrixBuilder

        builder = MatrixBuilder()
        builder.build_ts2vec_extension()

        assert len(builder.runs) == 135
        pretrains = [run for run in builder.runs if run.run_type == "pretrain"]
        finetunes = [run for run in builder.runs if run.run_type == "finetune"]

        assert len(pretrains) == 15
        assert sum(run.freeze_encoder is True for run in finetunes) == 60
        assert sum(run.freeze_encoder is False for run in finetunes) == 60

    def test_smart_external_reference_uses_stable_default_model_size(self):
        from scripts.internal.run_experiments import MatrixBuilder

        builder = MatrixBuilder()
        builder.build_smart_external_reference()

        assert len(builder.runs) == 135
        assert {run.model_size for run in builder.runs} == {"default"}
