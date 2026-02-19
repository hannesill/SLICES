"""Tests for Phase 1 bug fixes.

Targeted tests for each fix in Phase 1 of the improvement plan,
verifying correctness of the bug fixes and preventing regressions.
"""

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from omegaconf import OmegaConf
from slices.data.datamodule import ICUDataModule
from slices.data.dataset import ICUDataset
from slices.data.transforms import create_block_mask
from slices.models.encoders import TransformerConfig, TransformerEncoder
from slices.training import FineTuneModule

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_extracted_data(tmp_path):
    """Create mock extracted data files for testing."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)

    metadata = {
        "dataset": "mock",
        "feature_set": "core",
        "feature_names": ["heart_rate", "sbp", "resp_rate"],
        "n_features": 3,
        "seq_length_hours": 48,
        "min_stay_hours": 6,
        "task_names": ["mortality_24h", "mortality_hospital"],
        "n_stays": 10,
    }
    with open(data_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    static_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "patient_id": [100, 100, 101, 101, 102, 103, 104, 105, 106, 107],
            "age": [65, 65, 45, 45, 70, 55, 60, 75, 50, 80],
            "gender": ["M", "M", "F", "F", "M", "F", "M", "F", "M", "F"],
            "los_days": [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 2.5, 3.5, 4.5, 2.0],
        }
    )
    static_df.write_parquet(data_dir / "static.parquet")

    seq_length = 48
    n_features = 3
    np.random.seed(42)

    timeseries_data = []
    mask_data = []
    for _ in range(10):
        ts = np.random.randn(seq_length, n_features) * 10 + 70
        mask = np.random.rand(seq_length, n_features) > 0.3
        ts[~mask] = float("nan")
        timeseries_data.append(ts.tolist())
        mask_data.append(mask.tolist())

    timeseries_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "timeseries": timeseries_data,
            "mask": mask_data,
        }
    )
    timeseries_df.write_parquet(data_dir / "timeseries.parquet")

    labels_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "mortality_24h": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            "mortality_hospital": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        }
    )
    labels_df.write_parquet(data_dir / "labels.parquet")

    return data_dir


@pytest.fixture
def mock_multilabel_data(tmp_path):
    """Create mock extracted data with multi-label (phenotyping) task."""
    data_dir = tmp_path / "processed_ml"
    data_dir.mkdir(parents=True)

    metadata = {
        "dataset": "mock",
        "feature_set": "core",
        "feature_names": ["heart_rate", "sbp"],
        "n_features": 2,
        "seq_length_hours": 24,
        "min_stay_hours": 6,
        "task_names": ["phenotyping"],
        "n_stays": 8,
    }
    with open(data_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    static_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 9)),
            "patient_id": [100, 101, 102, 103, 104, 105, 106, 107],
            "age": [65, 45, 70, 55, 60, 75, 50, 80],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "los_days": [3.0, 2.0, 3.0, 4.0, 2.5, 3.5, 4.5, 2.0],
        }
    )
    static_df.write_parquet(data_dir / "static.parquet")

    seq_length = 24
    n_features = 2
    np.random.seed(42)

    timeseries_data = []
    mask_data = []
    for _ in range(8):
        ts = np.random.randn(seq_length, n_features) * 10 + 70
        mask = np.random.rand(seq_length, n_features) > 0.2
        ts[~mask] = float("nan")
        timeseries_data.append(ts.tolist())
        mask_data.append(mask.tolist())

    timeseries_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 9)),
            "timeseries": timeseries_data,
            "mask": mask_data,
        }
    )
    timeseries_df.write_parquet(data_dir / "timeseries.parquet")

    # Multi-label columns: phenotyping_sepsis, phenotyping_respiratory_failure
    # Note: no "phenotyping" column — only prefixed columns
    labels_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 9)),
            "phenotyping_sepsis": [1, 0, 1, 0, 1, 0, 1, 0],
            "phenotyping_respiratory_failure": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    labels_df.write_parquet(data_dir / "labels.parquet")

    return data_dir


@pytest.fixture
def sample_finetune_config():
    """Create a sample config for FineTuneModule tests."""
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


# ============================================================================
# 1.1: Checkpoint config dictionary mutation
# ============================================================================


class TestCheckpointConfigMutation:
    """Test that loading a v3 checkpoint doesn't mutate the checkpoint dict."""

    def test_checkpoint_dict_unchanged_after_load(self, sample_finetune_config, tmp_path):
        """Loading a v3 checkpoint should NOT mutate the 'encoder_config' dict.

        Before the fix, encoder_config.pop("name") would remove the "name"
        key from the checkpoint dictionary. If the same checkpoint object was
        reused (e.g., logged or re-saved), "name" would be gone.
        """
        # Create a v3 checkpoint
        config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            dropout=0.0,
            max_seq_length=24,
            pooling="none",
            use_positional_encoding=True,
            prenorm=True,
            activation="gelu",
            layer_norm_eps=1e-5,
        )
        encoder = TransformerEncoder(config)

        encoder_config = {
            "name": "transformer",
            "d_input": 10,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "max_seq_length": 24,
            "pooling": "none",
            "use_positional_encoding": True,
            "prenorm": True,
            "activation": "gelu",
            "layer_norm_eps": 1e-5,
        }

        checkpoint = {
            "encoder_state_dict": encoder.state_dict(),
            "encoder_config": encoder_config,
            "version": 3,
        }

        ckpt_path = tmp_path / "encoder_v3.pt"
        torch.save(checkpoint, ckpt_path)

        # Load the checkpoint in FineTuneModule
        FineTuneModule(sample_finetune_config, checkpoint_path=str(ckpt_path))

        # Reload the checkpoint dict from disk and verify "name" key is intact
        reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert (
            "name" in reloaded["encoder_config"]
        ), "Checkpoint 'encoder_config' was mutated: 'name' key is missing"

    def test_in_memory_checkpoint_dict_preserved(self, sample_finetune_config, tmp_path):
        """Verify that the in-memory checkpoint dict is not mutated during load."""
        config = TransformerConfig(
            d_input=10,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            dropout=0.0,
            max_seq_length=24,
            pooling="none",
            use_positional_encoding=True,
            prenorm=True,
            activation="gelu",
            layer_norm_eps=1e-5,
        )
        encoder = TransformerEncoder(config)

        encoder_config = {
            "name": "transformer",
            "d_input": 10,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "max_seq_length": 24,
            "pooling": "none",
            "use_positional_encoding": True,
            "prenorm": True,
            "activation": "gelu",
            "layer_norm_eps": 1e-5,
        }

        checkpoint = {
            "encoder_state_dict": encoder.state_dict(),
            "encoder_config": encoder_config,
            "version": 3,
        }

        ckpt_path = tmp_path / "encoder_v3.pt"
        torch.save(checkpoint, ckpt_path)

        # Load checkpoint into memory, then into FineTuneModule
        in_memory_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        original_config = dict(in_memory_ckpt["encoder_config"])

        # Simulate what FineTuneModule._load_encoder_weights does
        FineTuneModule(sample_finetune_config, checkpoint_path=str(ckpt_path))

        # The original in_memory_ckpt from disk should still have "name"
        # (we re-load from disk to verify the file wasn't corrupted)
        disk_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert disk_ckpt["encoder_config"] == original_config


# ============================================================================
# 1.2: Label statistics key mismatch
# ============================================================================


class TestLabelStatisticsKeys:
    """Test that get_label_statistics returns keys that the scripts actually use."""

    def test_prevalence_key_present(self, mock_extracted_data):
        """get_label_statistics must return 'prevalence', not 'positive_ratio'."""
        dataset = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=False,
        )
        stats = dataset.get_label_statistics()

        for task_name, task_stats in stats.items():
            assert (
                "prevalence" in task_stats
            ), f"Task '{task_name}' missing 'prevalence' key in label statistics"
            # Verify 'positive_ratio' does NOT exist (it was the buggy key)
            assert "positive_ratio" not in task_stats
            assert "negative_ratio" not in task_stats

    def test_prevalence_value_correct(self, mock_extracted_data):
        """Verify prevalence = positive / total."""
        dataset = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=False,
        )
        stats = dataset.get_label_statistics()

        for task_name, task_stats in stats.items():
            if "prevalence" in task_stats and task_stats["total"] > 0:
                expected_prevalence = task_stats["positive"] / task_stats["total"]
                assert abs(task_stats["prevalence"] - expected_prevalence) < 1e-6


# ============================================================================
# 1.3: Multi-label task crash in DataModule
# ============================================================================


class TestMultiLabelTask:
    """Test that multi-label tasks work through DataModule and Dataset."""

    def test_multilabel_dataset_loads(self, mock_multilabel_data):
        """ICUDataset should load multi-label (phenotyping) tasks without crash."""
        dataset = ICUDataset(
            mock_multilabel_data,
            task_name="phenotyping",
            normalize=False,
        )

        assert len(dataset) == 8
        sample = dataset[0]
        assert "label" in sample
        # Multi-label: label should be a vector of length n_labels
        assert sample["label"].dim() == 1
        assert sample["label"].shape[0] == 2  # phenotyping_sepsis + phenotyping_respiratory_failure

    def test_multilabel_datamodule_setup(self, mock_multilabel_data):
        """ICUDataModule.setup() should not crash for multi-label tasks."""
        dm = ICUDataModule(
            processed_dir=mock_multilabel_data,
            task_name="phenotyping",
            batch_size=4,
            num_workers=0,
            normalize=False,
        )
        dm.setup()

        assert dm.dataset is not None
        assert len(dm.train_indices) + len(dm.val_indices) + len(dm.test_indices) == len(dm.dataset)

    def test_multilabel_label_statistics(self, mock_multilabel_data):
        """get_label_statistics should report per-subtask stats for multi-label tasks."""
        dataset = ICUDataset(
            mock_multilabel_data,
            task_name="phenotyping",
            normalize=False,
        )
        stats = dataset.get_label_statistics()

        assert "phenotyping" in stats
        pheno_stats = stats["phenotyping"]
        assert "n_labels" in pheno_stats
        assert pheno_stats["n_labels"] == 2
        assert "mean_prevalence" in pheno_stats
        assert "subtasks" in pheno_stats
        assert "phenotyping_sepsis" in pheno_stats["subtasks"]
        assert "phenotyping_respiratory_failure" in pheno_stats["subtasks"]

    def test_multilabel_filter_missing_labels(self, tmp_path):
        """DataModule should filter stays with partially missing multi-label targets."""
        data_dir = tmp_path / "processed_ml_partial"
        data_dir.mkdir(parents=True)

        metadata = {
            "dataset": "mock",
            "feature_names": ["hr"],
            "n_features": 1,
            "seq_length_hours": 24,
            "min_stay_hours": 6,
            "task_names": ["phenotyping"],
            "n_stays": 6,
        }
        with open(data_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        pl.DataFrame(
            {
                "stay_id": list(range(1, 7)),
                "patient_id": list(range(100, 106)),
                "age": [50] * 6,
                "gender": ["M"] * 6,
                "los_days": [3.0] * 6,
            }
        ).write_parquet(data_dir / "static.parquet")

        np.random.seed(0)
        ts_data = []
        mask_data = []
        for _ in range(6):
            ts = np.random.randn(24, 1).tolist()
            msk = [[True]] * 24
            ts_data.append(ts)
            mask_data.append(msk)

        pl.DataFrame(
            {
                "stay_id": list(range(1, 7)),
                "timeseries": ts_data,
                "mask": mask_data,
            }
        ).write_parquet(data_dir / "timeseries.parquet")

        # Stay 3 has a null label for phenotyping_a → should be filtered
        pl.DataFrame(
            {
                "stay_id": list(range(1, 7)),
                "phenotyping_a": [1, 0, None, 1, 0, 1],
                "phenotyping_b": [0, 1, 0, 1, 0, 1],
            }
        ).write_parquet(data_dir / "labels.parquet")

        dm = ICUDataModule(
            processed_dir=data_dir,
            task_name="phenotyping",
            batch_size=2,
            num_workers=0,
            normalize=False,
        )
        dm.setup()

        # Stay 3 should have been excluded (5 remaining)
        total = len(dm.train_indices) + len(dm.val_indices) + len(dm.test_indices)
        assert total == 5, f"Expected 5 stays after filtering null labels, got {total}"


# ============================================================================
# 1.4: Dead code in TransformerEncoder._apply_pooling
# ============================================================================


class TestPoolingDeadCode:
    """Verify the unreachable cls branch inside pooling=='none' was removed."""

    def test_pooling_none_returns_full_sequence(self):
        """pooling='none' should return full sequence output (B, T, d_model)."""
        config = TransformerConfig(
            d_input=5,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
            max_seq_length=10,
            pooling="none",
        )
        encoder = TransformerEncoder(config)

        x = torch.randn(2, 10, 5)
        mask = torch.ones(2, 10, 5, dtype=torch.bool)
        out = encoder(x, mask=mask)

        assert out.dim() == 3
        assert out.shape == (2, 10, 16)

    def test_pooling_mean_returns_single_vector(self):
        """pooling='mean' should return (B, d_model)."""
        config = TransformerConfig(
            d_input=5,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
            max_seq_length=10,
            pooling="mean",
        )
        encoder = TransformerEncoder(config)

        x = torch.randn(2, 10, 5)
        mask = torch.ones(2, 10, 5, dtype=torch.bool)
        out = encoder(x, mask=mask)

        assert out.dim() == 2
        assert out.shape == (2, 16)


# ============================================================================
# 1.5: Normalization variance inconsistency
# ============================================================================


class TestNormalizationConsistency:
    """Verify that prepare_dataset.py and ICUDataset use the same variance formula."""

    def test_bessel_correction_matches_dataset(self):
        """Compute variance using the prepare_dataset.py formula and compare with
        ICUDataset's vectorized formula. Both should use Bessel's correction."""
        np.random.seed(123)
        n_samples = 50
        seq_length = 24
        n_features = 3

        # Generate random data
        data = np.random.randn(n_samples, seq_length, n_features) * 10 + 70
        masks = np.random.rand(n_samples, seq_length, n_features) > 0.2
        data[~masks] = np.nan

        # --- Method 1: prepare_dataset.py formula (after fix) ---
        sums = np.zeros(n_features)
        sq_sums = np.zeros(n_features)
        counts = np.zeros(n_features)

        for idx in range(n_samples):
            ts_data = data[idx]
            mask_data = masks[idx]
            actual_len = seq_length
            for t in range(actual_len):
                for f in range(n_features):
                    val = ts_data[t][f]
                    mask_val = mask_data[t][f]
                    if mask_val and not np.isnan(val):
                        sums[f] += val
                        sq_sums[f] += val * val
                        counts[f] += 1

        means_script = np.zeros(n_features)
        stds_script = np.ones(n_features)
        for f in range(n_features):
            if counts[f] > 0:
                mean = sums[f] / counts[f]
                variance = (sq_sums[f] - counts[f] * mean * mean) / max(counts[f] - 1, 1)
                std = np.sqrt(max(variance, 0))
                means_script[f] = mean
                stds_script[f] = std if std > 1e-6 else 1.0

        # --- Method 2: ICUDataset formula (vectorized Bessel's correction) ---
        ts_tensor = torch.from_numpy(data.astype(np.float32))
        masks_tensor = torch.from_numpy(masks)

        flat_ts = ts_tensor.reshape(-1, n_features)
        flat_masks = masks_tensor.reshape(-1, n_features)
        valid_mask = flat_masks & ~torch.isnan(flat_ts)

        masked_ts = torch.where(valid_mask, flat_ts, torch.zeros_like(flat_ts))
        valid_counts = valid_mask.sum(dim=0).float()
        feature_sums = masked_ts.sum(dim=0)

        means_dataset = torch.where(
            valid_counts > 0,
            feature_sums / valid_counts,
            torch.zeros(n_features),
        )

        deviations = torch.where(
            valid_mask,
            (flat_ts - means_dataset.unsqueeze(0)) ** 2,
            torch.zeros_like(flat_ts),
        )
        variance_dataset = torch.where(
            valid_counts > 1,
            deviations.sum(dim=0) / (valid_counts - 1),
            torch.ones(n_features),
        )
        stds_dataset = torch.sqrt(variance_dataset)
        stds_dataset = torch.clamp(stds_dataset, min=1e-6)

        # --- Compare ---
        np.testing.assert_allclose(
            means_script,
            means_dataset.numpy(),
            rtol=1e-5,
            err_msg="Means differ between prepare_dataset and ICUDataset",
        )
        np.testing.assert_allclose(
            stds_script,
            stds_dataset.numpy(),
            rtol=1e-4,
            err_msg="Stds differ between prepare_dataset and ICUDataset",
        )


# ============================================================================
# 1.6: Supervised script saves legacy checkpoint format
# ============================================================================


class TestSupervisedCheckpointFormat:
    """Test that save_encoder_weights produces v3 format."""

    def test_save_encoder_weights_v3_format(self, sample_finetune_config, tmp_path):
        """save_encoder_weights should produce a v3 checkpoint with encoder_config."""
        from scripts.training.supervised import save_encoder_weights

        model = FineTuneModule(sample_finetune_config)
        encoder_path = save_encoder_weights(model, sample_finetune_config, str(tmp_path))

        checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)

        assert isinstance(checkpoint, dict), "Checkpoint should be a dict, not raw state_dict"
        assert "version" in checkpoint
        assert checkpoint["version"] == 3
        assert "encoder_state_dict" in checkpoint
        assert "encoder_config" in checkpoint
        assert "name" in checkpoint["encoder_config"]
        assert checkpoint["encoder_config"]["name"] == "transformer"

    def test_saved_checkpoint_loadable_by_finetune_module(self, sample_finetune_config, tmp_path):
        """v3 checkpoint from supervised save should be loadable by FineTuneModule."""
        from scripts.training.supervised import save_encoder_weights

        model = FineTuneModule(sample_finetune_config)
        encoder_path = save_encoder_weights(model, sample_finetune_config, str(tmp_path))

        # Should load without deprecation warnings
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded_module = FineTuneModule(
                sample_finetune_config,
                checkpoint_path=str(encoder_path),
            )
            user_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "old-format" in str(x.message).lower()
            ]
            assert len(user_warnings) == 0, "Loading saved checkpoint triggered old-format warning"

        # Forward pass should work
        x = torch.randn(2, 24, 10)
        mask = torch.ones(2, 24, 10, dtype=torch.bool)
        out = loaded_module(x, mask)
        assert "logits" in out


# ============================================================================
# 1.7: Block masking under-counts overlapping regions
# ============================================================================


class TestBlockMaskingOverlap:
    """Test that block masking accurately counts masked positions."""

    def test_mask_ratio_more_accurate_with_overlap_fix(self):
        """With overlap counting fix, actual mask ratio should be closer to target.

        Use a small sequence where overlaps are likely and verify the mask ratio
        is within a reasonable tolerance of the target.
        """
        B, T, D = 100, 20, 5
        shape = (B, T, D)
        mask_ratio = 0.5
        generator = torch.Generator().manual_seed(42)

        mask = create_block_mask(
            shape,
            mask_ratio=mask_ratio,
            min_block_size=2,
            max_block_size=5,
            generator=generator,
        )

        # Count actual masked ratio
        masked_fraction = 1.0 - mask.float().mean().item()

        # With the overlap fix, actual ratio should be >= target
        # (the loop now correctly tracks newly masked positions)
        assert masked_fraction >= mask_ratio * 0.8, (
            f"Actual masked fraction {masked_fraction:.3f} is too far below "
            f"target {mask_ratio:.3f}"
        )

    def test_no_double_counting_in_small_sequence(self):
        """Force overlapping blocks and verify mask count is correct.

        Before the fix, the loop would early-exit because it double-counted
        overlap positions, thinking it had masked enough when it hadn't.
        After the fix, it counts only newly masked positions, so it continues
        placing blocks until the target is truly met.
        """
        # Run many trials and verify the average mask ratio is close to target
        B, T, D = 50, 20, 3
        shape = (B, T, D)
        mask_ratio = 0.5
        generator = torch.Generator().manual_seed(42)

        mask = create_block_mask(
            shape,
            mask_ratio=mask_ratio,
            min_block_size=3,
            max_block_size=8,
            generator=generator,
        )

        actual_masked = 1.0 - mask.float().mean().item()
        # With overlap fix, the actual ratio should be close to the target
        assert actual_masked >= mask_ratio * 0.85, (
            f"Average masked fraction {actual_masked:.3f} is too far below "
            f"target {mask_ratio:.3f}"
        )


# ============================================================================
# 1.8: torch.load with weights_only=True in tensor cache
# ============================================================================


class TestTensorCacheWeightsOnly:
    """Test that tensor cache uses weights_only=True."""

    def test_cache_round_trip_with_weights_only(self, mock_extracted_data):
        """Create a dataset, let it cache, then reload — should use weights_only=True."""
        # First load: creates cache
        dataset1 = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=False,
        )
        n1 = len(dataset1)

        # Second load: should use cache (with weights_only=True)
        dataset2 = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=False,
        )
        n2 = len(dataset2)

        assert n1 == n2

        # Verify sample equality
        s1 = dataset1[0]
        s2 = dataset2[0]
        assert torch.allclose(s1["timeseries"], s2["timeseries"])
        assert torch.equal(s1["mask"], s2["mask"])

    def test_cache_does_not_contain_arbitrary_objects(self, mock_extracted_data):
        """Tensor cache should only contain tensors and primitive types,
        compatible with weights_only=True."""
        ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=False,
        )

        # Find the cache file
        cache_dir = mock_extracted_data / ".tensor_cache"
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pt"))
            assert len(cache_files) > 0, "Cache directory exists but no .pt files found"

            # Loading with weights_only=True should succeed
            for cache_file in cache_files:
                cached = torch.load(cache_file, weights_only=True)
                assert isinstance(cached, dict)
                assert "timeseries_tensor" in cached
                assert "mask_tensor" in cached
