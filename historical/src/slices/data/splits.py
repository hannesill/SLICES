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
    cohort_stay_ids: Optional[List[int]] = None,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """Load cached splits from splits.yaml if valid.

    Validates that cached splits match current parameters (seed, ratios) and
    that the patient lists are consistent with the full cohort. The returned
    indices are always mapped over ``stay_ids``, which may be task-filtered.

    Args:
        processed_dir: Path to processed data directory containing splits.yaml.
        static_df: Static dataframe with stay_id -> patient_id mapping.
        stay_ids: List of stay_ids to map into split indices. For supervised
            tasks this may already be label-filtered.
        seed: Random seed for split computation.
        train_ratio: Fraction of patients for training.
        val_ratio: Fraction of patients for validation.
        test_ratio: Fraction of patients for testing.
        cohort_stay_ids: Optional full-cohort stay_ids used to validate that
            cached patient lists are global rather than task-filtered. When not
            provided, ``stay_ids`` is treated as the full cohort for backward
            compatibility.

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

        # Validate cached patient lists against the full cohort, not the
        # possibly task-filtered stay_ids. Supervised runs should reuse the
        # global patient assignment and then filter inside those sets.
        full_stay_ids = cohort_stay_ids if cohort_stay_ids is not None else stay_ids
        full_patients = set()
        for stay_id in full_stay_ids:
            patient_id = stay_to_patient.get(stay_id)
            if patient_id is None:
                logger.debug(
                    f"Stay {stay_id} missing from static stay->patient mapping, recomputing"
                )
                return None
            full_patients.add(patient_id)
        cached_patients = train_patients | val_patients | test_patients

        if full_patients != cached_patients:
            logger.debug(
                f"Cached splits have different full-cohort patients "
                f"(cached: {len(cached_patients)}, full cohort: {len(full_patients)}), "
                f"recomputing"
            )
            return None

        # The current mapped cohort may be task-filtered, but every remaining
        # patient must come from the cached full-cohort split.
        current_patients = set()
        for stay_id in stay_ids:
            patient_id = stay_to_patient.get(stay_id)
            if patient_id is None:
                logger.debug(
                    f"Stay {stay_id} missing from static stay->patient mapping, recomputing"
                )
                return None
            current_patients.add(patient_id)
        if not current_patients.issubset(cached_patients):
            logger.debug(
                f"Cached splits do not cover current patients "
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

    This is applied after the global patient split is defined. Split indices are
    then mapped onto the filtered stay list so DataModule and Dataset stay
    index-consistent without changing patient assignment per task.

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
        raise ValueError(f"Task '{task_name}' not found in labels. Available: {labels_df.columns}")

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


def _split_patient_sets(
    patient_ids: Set[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Split patient IDs deterministically into train/val/test sets."""
    unique_patients = sorted(patient_ids)
    n_patients = len(unique_patients)

    rng = np.random.RandomState(seed)
    patient_indices = np.arange(n_patients)
    rng.shuffle(patient_indices)
    shuffled_patients = [unique_patients[i] for i in patient_indices]

    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train : n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val :])

    return train_patients, val_patients, test_patients


def _indices_from_patient_sets(
    stay_ids: List[int],
    stay_to_patient: dict[int, int],
    train_patients: Set[int],
    val_patients: Set[int],
    test_patients: Set[int],
) -> Tuple[List[int], List[int], List[int]]:
    """Map stay IDs to split indices according to precomputed patient sets."""
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

    return train_indices, val_indices, test_indices


def save_global_split_info(
    processed_dir: Path,
    static_df: pl.DataFrame,
    stay_ids: List[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    dataset: Optional[Any] = None,
    train_subset_indices: Optional[List[int]] = None,
    label_fraction: float = 1.0,
) -> None:
    """Save canonical full-cohort patient split information."""
    stay_to_patient = dict(zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list()))
    full_patient_ids = {stay_to_patient[sid] for sid in stay_ids if sid in stay_to_patient}
    train_patients, val_patients, test_patients = _split_patient_sets(
        full_patient_ids,
        seed,
        train_ratio,
        val_ratio,
    )
    train_indices, val_indices, test_indices = _indices_from_patient_sets(
        stay_ids,
        stay_to_patient,
        train_patients,
        val_patients,
        test_patients,
    )

    split_info = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "label_fraction": label_fraction,
        "train_patients": sorted(train_patients),
        "val_patients": sorted(val_patients),
        "test_patients": sorted(test_patients),
        "train_stays": len(train_indices),
        "val_stays": len(val_indices),
        "test_stays": len(test_indices),
    }

    if dataset is not None and train_subset_indices is not None:
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


def compute_patient_level_splits(
    processed_dir: Path,
    task_name: Optional[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[
    List[int],
    List[int],
    List[int],
    pl.DataFrame,
    pl.DataFrame,
    List[int],
    List[int],
    Set[int],
    List[int],
]:
    """Compute patient-level train/val/test splits from parquet files.

    First checks for cached full-cohort splits in splits.yaml. If found and
    parameters match, loads the global patient assignments from cache. Otherwise,
    computes global patient assignments from scratch.

    Loads static and timeseries data directly without requiring the full dataset
    to be initialized. This is called BEFORE dataset creation to ensure
    normalization statistics use only training data (prevents data leakage).

    IMPORTANT: Patient assignment is computed on the full cohort first. Stays
    with missing task labels are filtered only after that assignment, so SSL
    pretraining and downstream task splits share one global patient partition.

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
                  all_stay_ids, filtered_stay_ids, excluded_stay_ids,
                  normalization_train_indices). The first three index lists are
                  mapped over ``filtered_stay_ids`` for supervised tasks. The
                  normalization index list is always mapped over ``all_stay_ids``
                  so downstream tasks share the same dataset/seed normalizer as
                  SSL pretraining.
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

    # Filter task labels before mapping indices, but do not let this change the
    # patient assignment. SSL and every downstream task must share the same
    # full-cohort patient split.
    stay_ids, excluded_stay_ids = filter_stays_with_missing_labels(
        all_stay_ids, labels_df, task_name
    )

    # Build the full-cohort stay_id -> patient_id mapping once. Downstream
    # optimization indices may be task-filtered, but normalization must remain
    # anchored to this unfiltered index space.
    logger.debug("Building stay-to-patient mapping")
    stay_to_patient = dict(zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list()))

    # Try to load cached splits first
    logger.debug("Checking for cached splits")
    cached_splits = load_cached_splits(
        processed_dir,
        static_df,
        stay_ids,
        seed,
        train_ratio,
        val_ratio,
        test_ratio,
        cohort_stay_ids=all_stay_ids,
    )
    if cached_splits is not None:
        train_indices, val_indices, test_indices = cached_splits
        with open(processed_dir / "splits.yaml") as f:
            cached_split_info = yaml.safe_load(f) or {}
        train_patients = set(cached_split_info.get("train_patients", []))
        normalization_train_indices = [
            idx
            for idx, stay_id in enumerate(all_stay_ids)
            if stay_to_patient[stay_id] in train_patients
        ]
        return (
            train_indices,
            val_indices,
            test_indices,
            static_df,
            labels_df,
            all_stay_ids,
            stay_ids,
            excluded_stay_ids,
            normalization_train_indices,
        )

    logger.info("Computing full-cohort patient-level splits...")

    # Get unique full-cohort patients (sorted for deterministic ordering across
    # Python runs). Task filtering is applied later when mapping stay indices.
    full_patient_ids = {stay_to_patient[sid] for sid in all_stay_ids if sid in stay_to_patient}
    unique_patients = sorted(full_patient_ids)
    n_patients = len(unique_patients)
    logger.debug(f"Found {n_patients:,} unique patients")

    # Warn if patient_id == stay_id for all stays (e.g. HiRID, SICdb)
    # This means the dataset lacks true patient-level IDs, so multiple ICU
    # stays from the same real patient may leak across splits.
    if n_patients == len(all_stay_ids) and n_patients > 0:
        full_stay_set = set(all_stay_ids)
        if full_patient_ids == full_stay_set:
            logger.warning(
                "patient_id == stay_id for all stays. This dataset likely lacks "
                "true patient-level identifiers (e.g. HiRID, SICdb). "
                "Patient-level split cannot prevent leakage from repeat ICU admissions."
            )

    # Shuffle full-cohort patients deterministically using seed
    logger.debug(f"Shuffling patients (seed={seed})")
    train_patients, val_patients, test_patients = _split_patient_sets(
        full_patient_ids,
        seed,
        train_ratio,
        val_ratio,
    )
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

    # Map global patient assignments back to the current stay list. For
    # supervised tasks, stay_ids has already been label-filtered, so this
    # applies task filtering inside the global train/val/test patient sets.
    logger.debug("Mapping global patient split to current stay indices")
    train_indices, val_indices, test_indices = _indices_from_patient_sets(
        stay_ids,
        stay_to_patient,
        train_patients,
        val_patients,
        test_patients,
    )
    normalization_train_indices, _, _ = _indices_from_patient_sets(
        all_stay_ids,
        stay_to_patient,
        train_patients,
        val_patients,
        test_patients,
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
        normalization_train_indices,
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
