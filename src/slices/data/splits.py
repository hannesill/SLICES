"""Patient-level split logic for ICU datasets.

Handles loading/computing/caching of train/val/test splits, label filtering,
and label-efficiency subsampling. Extracted from ICUDataModule for modularity.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import yaml

logger = logging.getLogger(__name__)


def load_cached_splits(
    processed_dir: Path,
    static_df: pl.DataFrame,
    stay_ids: List[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """Load cached splits from splits.yaml if valid.

    Validates that cached splits match current parameters (seed, ratios) and
    that the patient lists are consistent with the current task-filtered cohort.

    Args:
        processed_dir: Path to processed data directory containing splits.yaml.
        static_df: Static dataframe with stay_id -> patient_id mapping.
        stay_ids: List of stay_ids in order from timeseries parquet.
        seed: Random seed for split computation.
        train_ratio: Fraction of patients for training.
        val_ratio: Fraction of patients for validation.
        test_ratio: Fraction of patients for testing.

    Returns:
        Tuple of (train_indices, val_indices, test_indices) if cache is valid,
        None otherwise.
    """
    split_path = processed_dir / "splits.yaml"
    if not split_path.exists():
        return None

    try:
        with open(split_path) as f:
            cached = yaml.safe_load(f)

        # Validate parameters match
        if (
            cached.get("seed") != seed
            or not np.isclose(cached.get("train_ratio", 0), train_ratio)
            or not np.isclose(cached.get("val_ratio", 0), val_ratio)
            or not np.isclose(cached.get("test_ratio", 0), test_ratio)
        ):
            logger.debug("Cached splits have different parameters, recomputing")
            return None

        # Get patient lists from cache
        train_patients = set(cached.get("train_patients", []))
        val_patients = set(cached.get("val_patients", []))
        test_patients = set(cached.get("test_patients", []))

        if not train_patients or not val_patients or not test_patients:
            logger.debug("Cached splits missing patient lists, recomputing")
            return None

        # Validate patient sets are disjoint
        if not (
            train_patients.isdisjoint(val_patients)
            and train_patients.isdisjoint(test_patients)
            and val_patients.isdisjoint(test_patients)
        ):
            logger.debug("Cached splits have overlapping patients, recomputing")
            return None

        # Get stay_id -> patient_id mapping
        stay_to_patient = dict(
            zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list())
        )

        # Validate cached patient lists against the current cohort represented by
        # stay_ids. For supervised tasks this may be a label-filtered subset, so
        # validating against the full static table would incorrectly reuse
        # full-cohort cached splits for a smaller task-specific cohort.
        current_patients = set()
        for stay_id in stay_ids:
            patient_id = stay_to_patient.get(stay_id)
            if patient_id is None:
                logger.debug(
                    f"Stay {stay_id} missing from static stay->patient mapping, recomputing"
                )
                return None
            current_patients.add(patient_id)
        cached_patients = train_patients | val_patients | test_patients

        if current_patients != cached_patients:
            logger.debug(
                f"Cached splits have different patients "
                f"(cached: {len(cached_patients)}, current: {len(current_patients)}), "
                f"recomputing"
            )
            return None

        # Reconstruct indices from patient lists
        train_indices = []
        val_indices = []
        test_indices = []

        for idx, stay_id in enumerate(stay_ids):
            patient_id = stay_to_patient.get(stay_id)
            if patient_id in train_patients:
                train_indices.append(idx)
            elif patient_id in val_patients:
                val_indices.append(idx)
            elif patient_id in test_patients:
                test_indices.append(idx)
            else:
                logger.debug(f"Stay {stay_id} has unknown patient {patient_id}, recomputing")
                return None

        logger.info(
            f"Loaded cached splits: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )
        return train_indices, val_indices, test_indices

    except Exception as e:
        logger.debug(f"Failed to load cached splits: {e}, recomputing")
        return None


def filter_stays_with_missing_labels(
    stay_ids: List[int],
    labels_df: pl.DataFrame,
    task_name: Optional[str],
) -> Tuple[List[int], Set[int]]:
    """Filter out stays with missing labels for the configured task.

    This MUST be called BEFORE computing splits to ensure indices remain consistent
    between DataModule and Dataset.

    Args:
        stay_ids: List of all stay_ids from timeseries.
        labels_df: Labels dataframe with task columns.
        task_name: Task name to filter for, or None for unsupervised.

    Returns:
        Tuple of (filtered_stay_ids, excluded_stay_ids_set).
    """
    if task_name is None:
        return stay_ids, set()

    logger.debug(f"Checking labels for task '{task_name}'")

    # Detect multi-label tasks: columns prefixed with "{task_name}_"
    multilabel_cols = [c for c in labels_df.columns if c.startswith(f"{task_name}_")]
    is_multilabel = len(multilabel_cols) > 0 and task_name not in labels_df.columns

    # Get stays with valid labels for this task
    if not is_multilabel and task_name not in labels_df.columns:
        raise ValueError(
            f"Task '{task_name}' not found in labels. " f"Available: {labels_df.columns}"
        )

    # Find stays with non-null labels
    if is_multilabel:
        valid_labels_df = labels_df.filter(
            pl.all_horizontal([pl.col(c).is_not_null() for c in multilabel_cols])
        )
    else:
        valid_labels_df = labels_df.filter(pl.col(task_name).is_not_null())
    valid_stay_ids = set(valid_labels_df["stay_id"].to_list())

    # Filter stay_ids maintaining order
    filtered_stay_ids = [sid for sid in stay_ids if sid in valid_stay_ids]
    excluded_stay_ids = set(stay_ids) - valid_stay_ids

    if excluded_stay_ids:
        pct_excluded = len(excluded_stay_ids) / len(stay_ids) * 100
        logger.warning(
            f"Excluding {len(excluded_stay_ids):,} stays ({pct_excluded:.1f}%) "
            f"with missing '{task_name}' labels. Remaining: {len(filtered_stay_ids):,}"
        )

    return filtered_stay_ids, excluded_stay_ids


def compute_patient_level_splits(
    processed_dir: Path,
    task_name: Optional[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[
    List[int], List[int], List[int], pl.DataFrame, pl.DataFrame, List[int], List[int], Set[int]
]:
    """Compute patient-level train/val/test splits from parquet files.

    First checks for cached splits in splits.yaml. If found and parameters match,
    loads splits from cache. Otherwise, computes splits from scratch.

    Loads static and timeseries data directly without requiring the full dataset
    to be initialized. This is called BEFORE dataset creation to ensure
    normalization statistics use only training data (prevents data leakage).

    IMPORTANT: Filters out stays with missing labels BEFORE computing splits
    to ensure indices remain consistent with the Dataset after it loads.

    Uses deterministic shuffling of patient_id to assign patients to splits.
    All stays from a patient go to the same split.

    Args:
        processed_dir: Path to processed data directory.
        task_name: Task name for label filtering, or None for unsupervised.
        seed: Random seed for split computation.
        train_ratio: Fraction of patients for training.
        val_ratio: Fraction of patients for validation.
        test_ratio: Fraction of patients for testing.

    Returns:
        Tuple of (train_indices, val_indices, test_indices, static_df, labels_df,
                  all_stay_ids, filtered_stay_ids, excluded_stay_ids).
    """
    logger.info("Loading data for split computation...")

    # Load only needed columns for efficiency
    static_path = processed_dir / "static.parquet"
    logger.debug(f"Loading static data from {static_path.name}")
    static_df = pl.read_parquet(static_path, columns=["stay_id", "patient_id"])

    # Load only stay_id column from timeseries (much faster than full load)
    timeseries_path = processed_dir / "timeseries.parquet"
    logger.debug(f"Loading stay_ids from {timeseries_path.name}")
    timeseries_df = pl.read_parquet(timeseries_path, columns=["stay_id"])
    all_stay_ids = timeseries_df["stay_id"].to_list()

    # Load labels to filter out stays with missing labels
    labels_path = processed_dir / "labels.parquet"
    logger.debug(f"Loading labels from {labels_path.name}")
    labels_df = pl.read_parquet(labels_path)

    # CRITICAL: Filter stays with missing labels BEFORE computing splits
    # This ensures indices computed here match the filtered Dataset
    stay_ids, excluded_stay_ids = filter_stays_with_missing_labels(
        all_stay_ids, labels_df, task_name
    )

    # Try to load cached splits first
    logger.debug("Checking for cached splits")
    cached_splits = load_cached_splits(
        processed_dir, static_df, stay_ids, seed, train_ratio, val_ratio, test_ratio
    )
    if cached_splits is not None:
        train_indices, val_indices, test_indices = cached_splits
        return (
            train_indices,
            val_indices,
            test_indices,
            static_df,
            labels_df,
            all_stay_ids,
            stay_ids,
            excluded_stay_ids,
        )

    logger.info("Computing patient-level splits...")

    # Get stay_id -> patient_id mapping (only for filtered stays)
    logger.debug("Building stay-to-patient mapping")
    stay_to_patient = dict(zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list()))

    # Get unique patients (sorted for deterministic ordering across Python runs)
    filtered_patient_ids = {stay_to_patient[sid] for sid in stay_ids if sid in stay_to_patient}
    unique_patients = sorted(filtered_patient_ids)
    n_patients = len(unique_patients)
    logger.debug(f"Found {n_patients:,} unique patients")

    # Warn if patient_id == stay_id for all stays (e.g. HiRID, SICdb)
    # This means the dataset lacks true patient-level IDs, so multiple ICU
    # stays from the same real patient may leak across splits.
    if n_patients == len(stay_ids) and n_patients > 0:
        filtered_stay_set = set(stay_ids)
        if filtered_patient_ids == filtered_stay_set:
            logger.warning(
                "patient_id == stay_id for all stays. This dataset likely lacks "
                "true patient-level identifiers (e.g. HiRID, SICdb). "
                "Patient-level split cannot prevent leakage from repeat ICU admissions."
            )

    # Shuffle patients deterministically using seed
    logger.debug(f"Shuffling patients (seed={seed})")
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
    logger.debug(
        f"Split: {len(train_patients):,} train, "
        f"{len(val_patients):,} val, {len(test_patients):,} test patients"
    )

    # Verify no patient overlap between splits (data leakage check)
    assert train_patients.isdisjoint(
        val_patients
    ), "Patient leakage detected: train/val splits have overlapping patients"
    assert train_patients.isdisjoint(
        test_patients
    ), "Patient leakage detected: train/test splits have overlapping patients"
    assert val_patients.isdisjoint(
        test_patients
    ), "Patient leakage detected: val/test splits have overlapping patients"

    # Verify all patients are accounted for in exactly one split
    all_patients_in_splits = train_patients | val_patients | test_patients
    all_unique_patients = set(unique_patients)
    missing_patients = all_unique_patients - all_patients_in_splits
    extra_patients = all_patients_in_splits - all_unique_patients

    assert not missing_patients, (
        f"Patient split validation failed: {len(missing_patients)} patients "
        f"not assigned to any split. First 5: {list(missing_patients)[:5]}"
    )
    assert not extra_patients, (
        f"Patient split validation failed: {len(extra_patients)} patients "
        f"in splits but not in data. First 5: {list(extra_patients)[:5]}"
    )

    # Map back to stay indices
    logger.debug("Mapping patients to stay indices")
    train_indices = []
    val_indices = []
    test_indices = []

    for idx, stay_id in enumerate(stay_ids):
        patient_id = stay_to_patient.get(stay_id)
        if patient_id is None:
            raise ValueError(
                f"Stay {stay_id} has no patient_id mapping. "
                "Patient IDs are required for patient-level splits."
            )
        if patient_id in train_patients:
            train_indices.append(idx)
        elif patient_id in val_patients:
            val_indices.append(idx)
        elif patient_id in test_patients:
            test_indices.append(idx)
        else:
            raise ValueError(
                f"Stay {stay_id} has patient_id {patient_id} which is not "
                "in any split. This indicates a bug in split computation."
            )

    # Final validation: all stays should be accounted for
    total_assigned = len(train_indices) + len(val_indices) + len(test_indices)
    assert total_assigned == len(stay_ids), (
        f"Split validation failed: {total_assigned} stays assigned but "
        f"{len(stay_ids)} total stays in dataset"
    )

    logger.info(
        f"Splits computed: {len(train_indices):,} train, "
        f"{len(val_indices):,} val, {len(test_indices):,} test stays"
    )

    return (
        train_indices,
        val_indices,
        test_indices,
        static_df,
        labels_df,
        all_stay_ids,
        stay_ids,
        excluded_stay_ids,
    )


def subsample_train_indices(
    train_indices: List[int],
    label_fraction: float,
    seed: int,
) -> List[int]:
    """Subsample training indices for label-efficiency ablations.

    Uses a separate RNG seeded with the provided seed to ensure deterministic
    subsampling. Always selects at least 1 sample.

    Args:
        train_indices: Full list of training indices.
        label_fraction: Fraction of training data to use (0, 1].
        seed: Random seed for reproducibility.

    Returns:
        Subsampled list of training indices.
    """
    n_full = len(train_indices)
    n_subsample = max(1, int(n_full * label_fraction))

    rng = np.random.RandomState(seed)
    subsample_idx = rng.choice(n_full, size=n_subsample, replace=False)
    subsample_idx.sort()  # Maintain original order
    subsampled = [train_indices[i] for i in subsample_idx]

    logger.info(
        f"Label fraction={label_fraction}: using {n_subsample:,}/{n_full:,} "
        f"training samples ({label_fraction * 100:.1f}%)"
    )

    return subsampled


def save_split_info(
    processed_dir: Path,
    dataset: Any,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_subset_indices: Optional[List[int]] = None,
    label_fraction: float = 1.0,
) -> None:
    """Save split information to file for reproducibility.

    Args:
        processed_dir: Path to processed data directory.
        dataset: ICUDataset instance with stay_ids and static_df.
        train_indices: Training split indices.
        val_indices: Validation split indices.
        test_indices: Test split indices.
        seed: Random seed used.
        train_ratio: Training ratio used.
        val_ratio: Validation ratio used.
        test_ratio: Test ratio used.
        train_subset_indices: Optional optimization subset indices for
            label-efficiency runs. When provided and different from
            train_indices, persisted separately from the full split provenance.
        label_fraction: Label fraction used for the optimization subset.
    """
    static_df = dataset.static_df
    stay_to_patient = dict(zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list()))

    train_patients = sorted({stay_to_patient[dataset.stay_ids[i]] for i in train_indices})
    val_patients = sorted({stay_to_patient[dataset.stay_ids[i]] for i in val_indices})
    test_patients = sorted({stay_to_patient[dataset.stay_ids[i]] for i in test_indices})

    split_info = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "label_fraction": label_fraction,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "train_stays": len(train_indices),
        "val_stays": len(val_indices),
        "test_stays": len(test_indices),
    }

    if train_subset_indices is not None and train_subset_indices != train_indices:
        train_subset_stay_ids = [dataset.stay_ids[i] for i in train_subset_indices]
        train_subset_patients = sorted(
            {stay_to_patient[dataset.stay_ids[i]] for i in train_subset_indices}
        )
        split_info.update(
            {
                "train_subset_patients": train_subset_patients,
                "train_subset_stays": len(train_subset_indices),
                "train_subset_stay_ids": train_subset_stay_ids,
            }
        )

    split_path = processed_dir / "splits.yaml"
    with open(split_path, "w") as f:
        yaml.dump(split_info, f, default_flow_style=False)
