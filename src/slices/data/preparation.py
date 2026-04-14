"""Utilities for preparing processed datasets for training.

This module owns the canonical split-generation and normalization-stat logic
used by both standalone scripts and combined-dataset assembly.
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import polars as pl
import yaml

from slices.constants import TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from slices.data.tensor_cache import (
    get_data_fingerprint,
    get_preprocessing_fingerprint,
)


def _atomic_yaml_write(path: Path, data: dict) -> None:
    """Write YAML atomically using a temp file and rename."""
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
    """Compute patient-level train/val/test splits."""
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    stay_ids = timeseries_df["stay_id"].to_list()
    stay_to_patient = dict(zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list()))

    unique_patients = sorted(set(stay_to_patient.values()))
    n_patients = len(unique_patients)

    print(f"  Total patients: {n_patients:,}")
    print(f"  Total stays: {len(stay_ids):,}")

    if n_patients == len(stay_ids) and n_patients > 0:
        patient_set = set(stay_to_patient.values())
        stay_set = set(stay_ids)
        if patient_set == stay_set:
            print(
                "  WARNING: patient_id == stay_id for all stays. This dataset likely "
                "lacks true patient-level identifiers (e.g. HiRID, SICdb). "
                "Patient-level split cannot prevent leakage from repeat ICU admissions."
            )

    rng = np.random.RandomState(seed)
    patient_indices = np.arange(n_patients)
    rng.shuffle(patient_indices)
    shuffled_patients = [unique_patients[i] for i in patient_indices]

    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train : n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val :])

    assert train_patients.isdisjoint(val_patients), "Train/val overlap!"
    assert train_patients.isdisjoint(test_patients), "Train/test overlap!"
    assert val_patients.isdisjoint(test_patients), "Val/test overlap!"

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
    """Compute per-feature normalization stats on the training split only."""
    n_features = len(feature_names)
    n_train = len(train_indices)

    raw_timeseries = timeseries_df["timeseries"].to_list()
    raw_masks = timeseries_df["mask"].to_list()

    sums = np.zeros(n_features)
    sq_sums = np.zeros(n_features)
    counts = np.zeros(n_features)

    for progress, idx in enumerate(train_indices):
        if (progress + 1) % 5000 == 0:
            print(f"  Processing {progress + 1:,}/{n_train:,} training samples...")

        ts_arr = np.array(raw_timeseries[idx][:seq_length], dtype=np.float64)
        mask_arr = np.array(raw_masks[idx][:seq_length], dtype=bool)

        valid = mask_arr & np.isfinite(ts_arr)
        ts_valid = np.where(valid, ts_arr, 0.0)

        counts += valid.sum(axis=0)
        sums += ts_valid.sum(axis=0)
        sq_sums += (ts_valid**2).sum(axis=0)

    safe_counts = np.maximum(counts, 1)
    means = sums / safe_counts
    variance = (sq_sums - counts * means**2) / np.maximum(safe_counts - 1, 1)
    stds = np.sqrt(np.maximum(variance, 0.0))
    stds = np.where(stds > 1e-6, stds, 1.0)
    means = np.where(counts > 0, means, 0.0)

    return {
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "feature_names": feature_names,
        "train_indices": train_indices,
        "observation_counts": counts.tolist(),
    }


def prepare_processed_dataset(
    processed_dir: Path,
    seed: int,
    dataset_name: str | None = None,
) -> tuple[dict, dict]:
    """Generate and save splits plus normalization stats for a processed dataset."""
    dataset_label = dataset_name or processed_dir.name

    print("=" * 70)
    print(f"Dataset Preparation — {dataset_label}")
    print("=" * 70)

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {processed_dir}\n" "Run extraction before preparation."
        )

    print("\n1. Loading metadata...")
    with open(processed_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    feature_names = metadata["feature_names"]
    seq_length = metadata["seq_length_hours"]
    print(f"  Features: {len(feature_names)}")
    print(f"  Sequence length: {seq_length} hours")

    print("\n2. Loading parquet files...")
    static_df = pl.read_parquet(processed_dir / "static.parquet")
    timeseries_df = pl.read_parquet(processed_dir / "timeseries.parquet")
    print(f"  Loaded {len(timeseries_df):,} stays")

    print("\n3. Computing patient-level splits...")
    splits = compute_patient_splits(
        static_df=static_df,
        timeseries_df=timeseries_df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=seed,
    )

    print(f"  Train: {splits['train_stays']:,} stays ({len(splits['train_patients']):,} patients)")
    print(f"  Val:   {splits['val_stays']:,} stays ({len(splits['val_patients']):,} patients)")
    print(f"  Test:  {splits['test_stays']:,} stays ({len(splits['test_patients']):,} patients)")

    splits_path = processed_dir / "splits.yaml"
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
        "train_patients": splits["train_patients"],
        "val_patients": splits["val_patients"],
        "test_patients": splits["test_patients"],
    }
    _atomic_yaml_write(splits_path, splits_to_save)
    print(f"\n  Saved: {splits_path}")

    print("\n4. Computing normalization statistics (train set only)...")
    stats = compute_normalization_stats(
        timeseries_df=timeseries_df,
        train_indices=splits["train_indices"],
        feature_names=feature_names,
        seq_length=seq_length,
    )
    stats["normalize"] = True
    stats["split_hash"] = None
    stats["data_fingerprint"] = get_data_fingerprint(processed_dir)
    stats["preprocessing_fingerprint"] = get_preprocessing_fingerprint()

    stats_path = processed_dir / "normalization_stats.yaml"
    _atomic_yaml_write(stats_path, stats)
    print(f"  Saved: {stats_path}")

    print("\n" + "=" * 70)
    print("Preparation Complete")
    print("=" * 70)
    print(f"\nOutput directory: {processed_dir}")
    print("  - splits.yaml")
    print("  - normalization_stats.yaml")
    print("\nYou can now run training:")
    print(f"  uv run python scripts/training/pretrain.py dataset={dataset_label} ssl=mae")

    return splits_to_save, stats
