"""Prepare dataset splits and normalization statistics.

This script computes patient-level train/val/test splits and normalization
statistics on the training set only. Run this ONCE after extraction, before
training.

The output files (splits.yaml, normalization_stats.yaml) are required for
reproducible training runs and prevent data leakage from val/test sets.

Usage:
    uv run python scripts/preprocessing/prepare_dataset.py dataset=miiv
    uv run python scripts/preprocessing/prepare_dataset.py dataset=eicu
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import hydra
import numpy as np
import polars as pl
import yaml
from omegaconf import DictConfig
from slices.constants import TEST_RATIO, TRAIN_RATIO, VAL_RATIO


def _atomic_yaml_write(path: Path, data: dict) -> None:
    """Write YAML atomically using temp file + rename."""
    with NamedTemporaryFile(dir=path.parent, suffix=".yaml", mode="w", delete=False) as tmp:
        tmp_path = tmp.name
        yaml.dump(data, tmp, default_flow_style=False)
    os.replace(tmp_path, path)


def compute_patient_splits(
    static_df: pl.DataFrame,
    timeseries_df: pl.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    """Compute patient-level splits.

    All stays from the same patient go to the same split.

    Returns:
        Dictionary with split information including indices and patient lists.
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Get stay_ids in timeseries order (this is the canonical ordering)
    stay_ids = timeseries_df["stay_id"].to_list()

    # Get stay_id -> patient_id mapping
    stay_to_patient = dict(zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list()))

    # Get unique patients (sorted for deterministic ordering across Python runs)
    unique_patients = sorted(set(stay_to_patient.values()))
    n_patients = len(unique_patients)

    print(f"  Total patients: {n_patients:,}")
    print(f"  Total stays: {len(stay_ids):,}")

    # Warn if patient_id == stay_id for all stays (e.g. HiRID, SICdb)
    if n_patients == len(stay_ids) and n_patients > 0:
        patient_set = set(stay_to_patient.values())
        stay_set = set(stay_ids)
        if patient_set == stay_set:
            print(
                "  WARNING: patient_id == stay_id for all stays. This dataset likely "
                "lacks true patient-level identifiers (e.g. HiRID, SICdb). "
                "Patient-level split cannot prevent leakage from repeat ICU admissions."
            )

    # Shuffle patients deterministically
    rng = np.random.RandomState(seed)
    patient_indices = np.arange(n_patients)
    rng.shuffle(patient_indices)
    shuffled_patients = [unique_patients[i] for i in patient_indices]

    # Split patients
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train : n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val :])

    # Verify no overlap
    assert train_patients.isdisjoint(val_patients), "Train/val overlap!"
    assert train_patients.isdisjoint(test_patients), "Train/test overlap!"
    assert val_patients.isdisjoint(test_patients), "Val/test overlap!"

    # Map to stay indices
    train_indices = []
    val_indices = []
    test_indices = []

    for idx, stay_id in enumerate(stay_ids):
        patient_id = stay_to_patient[stay_id]
        if patient_id in train_patients:
            train_indices.append(idx)
        elif patient_id in val_patients:
            val_indices.append(idx)
        else:
            test_indices.append(idx)

    return {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_patients": sorted(train_patients),
        "val_patients": sorted(val_patients),
        "test_patients": sorted(test_patients),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_stays": len(train_indices),
        "val_stays": len(val_indices),
        "test_stays": len(test_indices),
    }


def compute_normalization_stats(
    timeseries_df: pl.DataFrame,
    train_indices: list,
    feature_names: list,
    seq_length: int,
) -> dict:
    """Compute mean and std for each feature on training set only.

    Uses vectorized numpy operations for efficiency.
    """
    n_features = len(feature_names)
    n_train = len(train_indices)

    # Extract training data into numpy arrays for vectorized computation
    raw_timeseries = timeseries_df["timeseries"].to_list()
    raw_masks = timeseries_df["mask"].to_list()

    # Accumulators
    sums = np.zeros(n_features)
    sq_sums = np.zeros(n_features)
    counts = np.zeros(n_features)

    for progress, idx in enumerate(train_indices):
        if (progress + 1) % 5000 == 0:
            print(f"  Processing {progress + 1:,}/{n_train:,} training samples...")

        # Convert to numpy arrays for vectorized ops
        ts_arr = np.array(raw_timeseries[idx][:seq_length], dtype=np.float64)
        mask_arr = np.array(raw_masks[idx][:seq_length], dtype=bool)

        # Valid = mask is True AND value is finite
        valid = mask_arr & np.isfinite(ts_arr)

        # Zero out invalid entries so they don't affect sums
        ts_valid = np.where(valid, ts_arr, 0.0)

        # Accumulate per-feature sums along time axis
        counts += valid.sum(axis=0)
        sums += ts_valid.sum(axis=0)
        sq_sums += (ts_valid**2).sum(axis=0)

    # Compute mean and std
    means = np.zeros(n_features)
    stds = np.ones(n_features)

    safe_counts = np.maximum(counts, 1)
    means = sums / safe_counts
    variance = (sq_sums - counts * means**2) / np.maximum(safe_counts - 1, 1)
    stds = np.sqrt(np.maximum(variance, 0.0))
    stds = np.where(stds > 1e-6, stds, 1.0)
    # Zero out means for features with no observations
    means = np.where(counts > 0, means, 0.0)

    return {
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "feature_names": feature_names,
        "train_indices": train_indices,
        "observation_counts": counts.tolist(),
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Prepare dataset splits and normalization statistics.

    Usage:
        uv run python scripts/preprocessing/prepare_dataset.py dataset=miiv
        uv run python scripts/preprocessing/prepare_dataset.py dataset=eicu
    """
    print("=" * 70)
    print(f"Dataset Preparation — {cfg.dataset}")
    print("=" * 70)

    processed_dir = Path(cfg.data.processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {processed_dir}\n"
            f"Run first: uv run python scripts/preprocessing/extract_ricu.py dataset={cfg.dataset}"
        )

    # Load metadata
    print("\n1. Loading metadata...")
    with open(processed_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    feature_names = metadata["feature_names"]
    seq_length = metadata["seq_length_hours"]
    print(f"  Features: {len(feature_names)}")
    print(f"  Sequence length: {seq_length} hours")

    # Load data
    print("\n2. Loading parquet files...")
    static_df = pl.read_parquet(processed_dir / "static.parquet")
    timeseries_df = pl.read_parquet(processed_dir / "timeseries.parquet")
    print(f"  Loaded {len(timeseries_df):,} stays")

    # Compute splits (ratios are benchmark invariants from constants.py)
    print("\n3. Computing patient-level splits...")
    splits = compute_patient_splits(
        static_df=static_df,
        timeseries_df=timeseries_df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=cfg.seed,
    )

    print(f"  Train: {splits['train_stays']:,} stays ({len(splits['train_patients']):,} patients)")
    print(f"  Val:   {splits['val_stays']:,} stays ({len(splits['val_patients']):,} patients)")
    print(f"  Test:  {splits['test_stays']:,} stays ({len(splits['test_patients']):,} patients)")

    # Save splits FIRST
    splits_path = processed_dir / "splits.yaml"
    # Save full patient lists to enable cache validation in datamodule
    # (patient lists are required to verify split consistency)
    splits_to_save = {
        "seed": splits["seed"],
        "train_ratio": splits["train_ratio"],
        "val_ratio": splits["val_ratio"],
        "test_ratio": splits["test_ratio"],
        "train_indices": splits["train_indices"],
        "val_indices": splits["val_indices"],
        "test_indices": splits["test_indices"],
        "train_stays": splits["train_stays"],
        "val_stays": splits["val_stays"],
        "test_stays": splits["test_stays"],
        # Full patient lists for datamodule cache validation
        "train_patients": splits["train_patients"],
        "val_patients": splits["val_patients"],
        "test_patients": splits["test_patients"],
    }
    _atomic_yaml_write(splits_path, splits_to_save)
    print(f"\n  Saved: {splits_path}")

    # Compute normalization stats on train set only
    print("\n4. Computing normalization statistics (train set only)...")
    stats = compute_normalization_stats(
        timeseries_df=timeseries_df,
        train_indices=splits["train_indices"],
        feature_names=feature_names,
        seq_length=seq_length,
    )

    # Save normalization stats
    stats_path = processed_dir / "normalization_stats.yaml"
    _atomic_yaml_write(stats_path, stats)
    print(f"  Saved: {stats_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Preparation Complete")
    print("=" * 70)
    print(f"\nOutput directory: {processed_dir}")
    print("  - splits.yaml")
    print("  - normalization_stats.yaml")
    print("\nYou can now run training:")
    print(f"  uv run python scripts/training/pretrain.py dataset={cfg.dataset} ssl=mae")


if __name__ == "__main__":
    main()
