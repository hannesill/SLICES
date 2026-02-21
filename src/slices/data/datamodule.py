"""Lightning DataModule for ICU data.

Handles patient-level splits, data loading, and batching for training.
Ensures no patient appears in multiple splits (prevents data leakage).
Supports sliding windows for SSL pretraining with longer sequences.
Auto-detects decompensation tasks and configures sliding windows with
on-the-fly per-window label computation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import lightning.pytorch as L
import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from slices.constants import (
    NORMALIZE,
    PIN_MEMORY,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from slices.data.dataset import ICUDataset
from slices.data.sliding_window import SlidingWindowDataset
from slices.data.utils import get_package_data_dir

# Module-level logger
logger = logging.getLogger(__name__)

# Constants
TQDM_MIN_ITEMS = 1000  # Only show progress bar for collections larger than this


def icu_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching ICU samples.

    Handles both regular ICUDataset samples and SlidingWindowDataset samples
    which include additional window_start and window_idx fields.

    Args:
        batch: List of sample dictionaries from ICUDataset or SlidingWindowDataset.

    Returns:
        Batched dictionary with stacked tensors.
    """
    # Stack tensors
    timeseries = torch.stack([s["timeseries"] for s in batch])  # (B, T, D)
    mask = torch.stack([s["mask"] for s in batch])  # (B, T, D)
    stay_ids = torch.tensor([s["stay_id"] for s in batch])  # (B,)

    result = {
        "timeseries": timeseries,
        "mask": mask,
        "stay_id": stay_ids,
    }

    # Stack labels if present
    if "label" in batch[0]:
        labels = torch.stack([s["label"] for s in batch])  # (B,)
        result["label"] = labels

    # Collate static features if present
    if "static" in batch[0]:
        static_keys = batch[0]["static"].keys()
        result["static"] = {key: [s["static"].get(key) for s in batch] for key in static_keys}

    # Stack sliding window metadata if present
    if "window_start" in batch[0]:
        result["window_start"] = torch.tensor([s["window_start"] for s in batch])  # (B,)
    if "window_idx" in batch[0]:
        result["window_idx"] = torch.tensor([s["window_idx"] for s in batch])  # (B,)

    return result


class ICUDataModule(L.LightningDataModule):
    """Lightning DataModule for ICU data.

    Implements patient-level splits to prevent data leakage.
    Uses hashing of patient_id for deterministic, reproducible splits.

    Auto-detects decompensation tasks (``window_label_mode: death_hours`` in
    task YAML) and configures sliding windows with per-window label computation.

    Example:
        >>> dm = ICUDataModule(
        ...     processed_dir="data/processed/mimic-iv-demo",
        ...     task_name="mortality_24h",
        ...     batch_size=32,
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        task_name: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        seq_length: Optional[int] = None,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        seed: int = 42,
        normalize: bool = NORMALIZE,
        pin_memory: bool = PIN_MEMORY,
        # Sliding window parameters for SSL pretraining
        enable_sliding_windows: bool = False,
        window_size: Optional[int] = None,
        window_stride: Optional[int] = None,
    ) -> None:
        """Initialize DataModule.

        Args:
            processed_dir: Directory containing extracted parquet files.
            task_name: Task for label extraction (e.g., 'mortality_24h').
            batch_size: Batch size for training.
            num_workers: Number of data loading workers.
            seq_length: Override sequence length (uses metadata default if None).
            train_ratio: Fraction of patients for training.
            val_ratio: Fraction of patients for validation.
            test_ratio: Fraction of patients for testing.
            seed: Random seed for reproducible splits.
            normalize: Whether to normalize features.
            pin_memory: Whether to pin memory for faster GPU transfer.
            enable_sliding_windows: Whether to use sliding windows for training.
                Useful for SSL pretraining with longer sequences (e.g., 168h).
                When enabled, train_dataloader uses overlapping windows and
                val_dataloader uses non-overlapping windows.
            window_size: Size of sliding windows in timesteps (hours).
                Defaults to seq_length if None (no windowing effect).
            window_stride: Step between consecutive windows.
                Defaults to window_size // 2 (50% overlap) for training.
                Validation always uses stride=window_size (non-overlapping).
        """
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.task_name = task_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.normalize = normalize
        self.pin_memory = pin_memory

        # Sliding window parameters
        self.enable_sliding_windows = enable_sliding_windows
        self.window_size = window_size
        self.window_stride = window_stride

        # Decompensation-specific parameters (auto-detected in setup)
        self._decompensation_mode = False
        self._decompensation_pred_hours: Optional[int] = None
        self._decompensation_train_stride: Optional[int] = None
        self._decompensation_eval_stride: Optional[int] = None

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Will be set in setup()
        self.dataset: Optional[ICUDataset] = None
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []

    def _detect_decompensation_mode(self) -> None:
        """Auto-detect decompensation task from task YAML config.

        Checks for ``window_label_mode: death_hours`` in the task's label_params.
        When detected, configures sliding windows and decompensation parameters.
        """
        if self.task_name is None:
            return

        tasks_dir = get_package_data_dir() / "tasks"
        task_yaml = tasks_dir / f"{self.task_name}.yaml"
        if not task_yaml.exists():
            return

        with open(task_yaml) as f:
            task_config = yaml.safe_load(f)

        label_params = task_config.get("label_params", {})
        if label_params.get("window_label_mode") != "death_hours":
            return

        # Enable decompensation mode
        self._decompensation_mode = True
        self._decompensation_pred_hours = task_config.get("prediction_window_hours", 24)

        # Use observation_window_hours as the window size
        obs_window = task_config.get("observation_window_hours", 48)
        self.window_size = obs_window

        # Get strides from task config
        self._decompensation_train_stride = label_params.get("stride_hours", 6)
        self._decompensation_eval_stride = label_params.get("eval_stride_hours", 1)

        # Auto-enable sliding windows
        self.enable_sliding_windows = True
        self.window_stride = self._decompensation_train_stride

        logger.info(
            f"Decompensation mode detected: obs_window={obs_window}h, "
            f"pred_hours={self._decompensation_pred_hours}h, "
            f"train_stride={self._decompensation_train_stride}h, "
            f"eval_stride={self._decompensation_eval_stride}h"
        )

    def _load_cached_splits(
        self, static_df: pl.DataFrame, stay_ids: List[int]
    ) -> Optional[Tuple[List[int], List[int], List[int]]]:
        """Load cached splits from splits.yaml if valid.

        Validates that cached splits match current parameters (seed, ratios) and
        that the patient lists are consistent with current data.

        Args:
            static_df: Static dataframe with stay_id -> patient_id mapping.
            stay_ids: List of stay_ids in order from timeseries parquet.

        Returns:
            Tuple of (train_indices, val_indices, test_indices) if cache is valid,
            None otherwise.
        """
        split_path = self.processed_dir / "splits.yaml"
        if not split_path.exists():
            return None

        try:
            with open(split_path) as f:
                cached = yaml.safe_load(f)

            # Validate parameters match
            if (
                cached.get("seed") != self.seed
                or not np.isclose(cached.get("train_ratio", 0), self.train_ratio)
                or not np.isclose(cached.get("val_ratio", 0), self.val_ratio)
                or not np.isclose(cached.get("test_ratio", 0), self.test_ratio)
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

            # Validate all patients in data are accounted for
            current_patients = set(stay_to_patient.values())
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

    def _filter_stays_with_missing_labels(
        self, stay_ids: List[int], labels_df: pl.DataFrame
    ) -> Tuple[List[int], Set[int]]:
        """Filter out stays with missing labels for the configured task.

        This MUST be called BEFORE computing splits to ensure indices remain consistent
        between DataModule and Dataset.

        Args:
            stay_ids: List of all stay_ids from timeseries.
            labels_df: Labels dataframe with task columns.

        Returns:
            Tuple of (filtered_stay_ids, excluded_stay_ids_set).
        """
        if self.task_name is None:
            return stay_ids, set()

        logger.debug(f"Checking labels for task '{self.task_name}'")

        # Detect multi-label tasks: columns prefixed with "{task_name}_"
        multilabel_cols = [c for c in labels_df.columns if c.startswith(f"{self.task_name}_")]
        is_multilabel = len(multilabel_cols) > 0 and self.task_name not in labels_df.columns

        # Get stays with valid labels for this task
        if not is_multilabel and self.task_name not in labels_df.columns:
            raise ValueError(
                f"Task '{self.task_name}' not found in labels. " f"Available: {labels_df.columns}"
            )

        # Find stays with non-null labels
        if is_multilabel:
            valid_labels_df = labels_df.filter(
                pl.all_horizontal([pl.col(c).is_not_null() for c in multilabel_cols])
            )
        else:
            valid_labels_df = labels_df.filter(pl.col(self.task_name).is_not_null())
        valid_stay_ids = set(valid_labels_df["stay_id"].to_list())

        # Filter stay_ids maintaining order
        filtered_stay_ids = [sid for sid in stay_ids if sid in valid_stay_ids]
        excluded_stay_ids = set(stay_ids) - valid_stay_ids

        if excluded_stay_ids:
            pct_excluded = len(excluded_stay_ids) / len(stay_ids) * 100
            logger.warning(
                f"Excluding {len(excluded_stay_ids):,} stays ({pct_excluded:.1f}%) "
                f"with missing '{self.task_name}' labels. Remaining: {len(filtered_stay_ids):,}"
            )

        return filtered_stay_ids, excluded_stay_ids

    def _get_patient_level_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """Create patient-level train/val/test splits from parquet files.

        First checks for cached splits in splits.yaml. If found and parameters match,
        loads splits from cache. Otherwise, computes splits from scratch.

        Loads static and timeseries data directly without requiring the full dataset
        to be initialized. This is called BEFORE dataset creation to ensure
        normalization statistics use only training data (prevents data leakage).

        IMPORTANT: Filters out stays with missing labels BEFORE computing splits
        to ensure indices remain consistent with the Dataset after it loads.

        Uses deterministic shuffling of patient_id to assign patients to splits.
        All stays from a patient go to the same split.

        Returns:
            Tuple of (train_indices, val_indices, test_indices).
        """
        logger.info("Loading data for split computation...")

        # Load only needed columns for efficiency
        static_path = self.processed_dir / "static.parquet"
        logger.debug(f"Loading static data from {static_path.name}")
        self._static_df = pl.read_parquet(static_path, columns=["stay_id", "patient_id"])

        # Load only stay_id column from timeseries (much faster than full load)
        timeseries_path = self.processed_dir / "timeseries.parquet"
        logger.debug(f"Loading stay_ids from {timeseries_path.name}")
        timeseries_df = pl.read_parquet(timeseries_path, columns=["stay_id"])
        all_stay_ids = timeseries_df["stay_id"].to_list()

        # Load labels to filter out stays with missing labels
        labels_path = self.processed_dir / "labels.parquet"
        logger.debug(f"Loading labels from {labels_path.name}")
        self._labels_df = pl.read_parquet(labels_path)

        # CRITICAL: Filter stays with missing labels BEFORE computing splits
        # This ensures indices computed here match the filtered Dataset
        stay_ids, self._excluded_stay_ids = self._filter_stays_with_missing_labels(
            all_stay_ids, self._labels_df
        )

        # Store for later use
        self._all_stay_ids = all_stay_ids
        self._filtered_stay_ids = stay_ids

        # Try to load cached splits first
        logger.debug("Checking for cached splits")
        cached_splits = self._load_cached_splits(self._static_df, stay_ids)
        if cached_splits is not None:
            return cached_splits

        logger.info("Computing patient-level splits...")

        # Get stay_id -> patient_id mapping (only for filtered stays)
        logger.debug("Building stay-to-patient mapping")
        stay_to_patient = dict(
            zip(self._static_df["stay_id"].to_list(), self._static_df["patient_id"].to_list())
        )

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
        logger.debug(f"Shuffling patients (seed={self.seed})")
        rng = np.random.RandomState(self.seed)
        patient_indices = np.arange(n_patients)
        rng.shuffle(patient_indices)
        shuffled_patients = [unique_patients[i] for i in patient_indices]

        # Split patients
        n_train = int(n_patients * self.train_ratio)
        n_val = int(n_patients * self.val_ratio)

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

        return train_indices, val_indices, test_indices

    def _save_split_info(self) -> None:
        """Save split information to file for reproducibility."""
        if self.dataset is None:
            return

        static_df = self.dataset.static_df
        stay_to_patient = dict(
            zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list())
        )

        train_patients = sorted(
            {stay_to_patient[self.dataset.stay_ids[i]] for i in self.train_indices}
        )
        val_patients = sorted({stay_to_patient[self.dataset.stay_ids[i]] for i in self.val_indices})
        test_patients = sorted(
            {stay_to_patient[self.dataset.stay_ids[i]] for i in self.test_indices}
        )

        split_info = {
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "train_patients": train_patients,
            "val_patients": val_patients,
            "test_patients": test_patients,
            "train_stays": len(self.train_indices),
            "val_stays": len(self.val_indices),
            "test_stays": len(self.test_indices),
        }

        split_path = self.processed_dir / "splits.yaml"
        with open(split_path, "w") as f:
            yaml.dump(split_info, f, default_flow_style=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for train/val/test.

        Creates patient-level splits BEFORE initializing the dataset to ensure
        normalization statistics are computed only on training data (no leakage).

        Auto-detects decompensation tasks and configures sliding windows.

        Args:
            stage: Stage name ('fit', 'validate', 'test', or None).
        """
        # Skip if already set up (Lightning calls setup() again in trainer.fit())
        if self.dataset is not None:
            logger.debug("DataModule already set up, skipping")
            return

        logger.info("Setting up ICUDataModule")

        # Auto-detect decompensation mode from task YAML
        self._detect_decompensation_mode()

        # CRITICAL: Get splits FIRST before creating dataset
        # This also filters stays with missing labels to ensure index consistency
        logger.debug("[Step 1/3] Computing patient-level splits")
        self.train_indices, self.val_indices, self.test_indices = self._get_patient_level_splits()

        # Create dataset with training indices for normalization
        # IMPORTANT: Use handle_missing_labels='raise' because we already filtered
        # missing labels in _get_patient_level_splits. If any are still missing,
        # it indicates a bug.
        logger.debug("[Step 2/3] Creating ICUDataset")
        self.dataset = ICUDataset(
            data_dir=self.processed_dir,
            task_name=self.task_name,
            seq_length=self.seq_length,
            normalize=self.normalize,
            train_indices=self.train_indices,
            # Use 'raise' since we pre-filtered - any missing labels now is a bug
            handle_missing_labels="raise" if self.task_name else "filter",
            # Pass excluded stays so Dataset can validate consistency
            _excluded_stay_ids=getattr(self, "_excluded_stay_ids", None),
        )

        # Validate dataset size matches expected filtered size
        expected_size = (
            len(self._filtered_stay_ids) if hasattr(self, "_filtered_stay_ids") else None
        )
        if expected_size is not None and len(self.dataset) != expected_size:
            raise RuntimeError(
                f"Dataset size mismatch! Expected {expected_size} stays "
                f"(after filtering), got {len(self.dataset)}. "
                "This indicates an index consistency bug."
            )

        # Save split information for reproducibility
        logger.debug("[Step 3/3] Saving split information")
        self._save_split_info()

        logger.info(
            f"DataModule setup complete: "
            f"Train={len(self.train_indices):,}, Val={len(self.val_indices):,}, "
            f"Test={len(self.test_indices):,} stays"
        )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader.

        If sliding windows are enabled, wraps the base dataset with
        SlidingWindowDataset using overlapping windows.
        For decompensation, uses task-specific stride and pred_hours.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")

        if self.enable_sliding_windows:
            window_size = self.window_size or self.dataset.seq_length
            window_stride = self.window_stride or window_size // 2

            train_dataset = SlidingWindowDataset(
                self.dataset,
                window_size=window_size,
                stride=window_stride,
                stay_indices=self.train_indices,
                decompensation_pred_hours=self._decompensation_pred_hours,
            )
            logger.info(
                f"Train dataloader using sliding windows: {len(train_dataset)} windows "
                f"(window_size={window_size}, stride={window_stride})"
            )
        else:
            train_dataset = Subset(self.dataset, self.train_indices)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=icu_collate_fn,
            drop_last=True,  # Avoid small batches
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader.

        If sliding windows are enabled, uses NON-overlapping windows
        (stride=window_size) to avoid inflated validation metrics.
        For decompensation, uses eval_stride for finer-grained evaluation.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before val_dataloader()")

        if self.enable_sliding_windows:
            window_size = self.window_size or self.dataset.seq_length

            # Decompensation uses eval_stride; default SSL uses non-overlapping
            if self._decompensation_mode and self._decompensation_eval_stride:
                val_stride = self._decompensation_eval_stride
            else:
                val_stride = window_size  # Non-overlapping

            val_dataset = SlidingWindowDataset(
                self.dataset,
                window_size=window_size,
                stride=val_stride,
                stay_indices=self.val_indices,
                decompensation_pred_hours=self._decompensation_pred_hours,
            )
            logger.info(
                f"Val dataloader using sliding windows: {len(val_dataset)} windows "
                f"(window_size={window_size}, stride={val_stride})"
            )
        else:
            val_dataset = Subset(self.dataset, self.val_indices)

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=icu_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader.

        For decompensation, uses sliding windows with eval_stride.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before test_dataloader()")

        if self._decompensation_mode:
            window_size = self.window_size or self.dataset.seq_length
            eval_stride = self._decompensation_eval_stride or window_size

            test_dataset = SlidingWindowDataset(
                self.dataset,
                window_size=window_size,
                stride=eval_stride,
                stay_indices=self.test_indices,
                decompensation_pred_hours=self._decompensation_pred_hours,
            )
            logger.info(
                f"Test dataloader using sliding windows: {len(test_dataset)} windows "
                f"(window_size={window_size}, stride={eval_stride})"
            )
        else:
            test_dataset = Subset(self.dataset, self.test_indices)

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=icu_collate_fn,
        )

    def get_feature_dim(self) -> int:
        """Return number of input features."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_feature_dim()")
        return self.dataset.n_features

    def get_seq_length(self) -> int:
        """Return sequence length.

        For decompensation, returns the observation window size.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_seq_length()")
        if self._decompensation_mode and self.window_size:
            return self.window_size
        return self.dataset.seq_length

    def get_split_info(self) -> Dict[str, Any]:
        """Return information about data splits.

        Returns:
            Dict with stay counts, patient counts, and actual ratios.
            Use this to verify splits are reasonable.
        """
        total_stays = len(self.dataset) if self.dataset else 0

        # Count unique patients per split
        if self.dataset is not None:
            static_df = self.dataset.static_df
            stay_to_patient = dict(
                zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list())
            )

            train_patients = {stay_to_patient[self.dataset.stay_ids[i]] for i in self.train_indices}
            val_patients = {stay_to_patient[self.dataset.stay_ids[i]] for i in self.val_indices}
            test_patients = {stay_to_patient[self.dataset.stay_ids[i]] for i in self.test_indices}

            n_train_patients = len(train_patients)
            n_val_patients = len(val_patients)
            n_test_patients = len(test_patients)
            total_patients = n_train_patients + n_val_patients + n_test_patients
        else:
            n_train_patients = n_val_patients = n_test_patients = total_patients = 0

        return {
            # Stay counts
            "train_stays": len(self.train_indices),
            "val_stays": len(self.val_indices),
            "test_stays": len(self.test_indices),
            "total_stays": total_stays,
            # Patient counts
            "train_patients": n_train_patients,
            "val_patients": n_val_patients,
            "test_patients": n_test_patients,
            "total_patients": total_patients,
            # Actual ratios (for verification)
            "actual_train_ratio": len(self.train_indices) / total_stays if total_stays > 0 else 0,
            "actual_val_ratio": len(self.val_indices) / total_stays if total_stays > 0 else 0,
            "actual_test_ratio": len(self.test_indices) / total_stays if total_stays > 0 else 0,
        }

    def get_label_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Return label statistics for all tasks."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_label_statistics()")
        return self.dataset.get_label_statistics()
