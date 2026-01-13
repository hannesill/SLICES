"""Sentinel patient selection for pipeline debugging.

Selects representative patients by clinical/demographic criteria for
targeted inspection of data extraction and model outputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import polars as pl


class SelectionStrategy(str, Enum):
    """Strategy for selecting patients within a stratum."""

    RANDOM = "random"
    EXTREME = "extreme"
    PERCENTILE = "percentile"


@dataclass
class StratificationCriterion:
    """Single criterion for stratifying patients.

    Attributes:
        column: Column name in static.parquet (e.g., 'los_days', 'age').
        values: For categorical columns, list of values to stratify by.
        bins: For numeric columns, bin edges (e.g., [0, 1, 3, 7, inf] for LOS).
        bin_labels: Optional labels for numeric bins.
    """

    column: str
    values: Optional[List[Union[str, int]]] = None
    bins: Optional[List[float]] = None
    bin_labels: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate criterion configuration."""
        if self.values is None and self.bins is None:
            raise ValueError(f"Criterion for '{self.column}' must have either 'values' or 'bins'")
        if self.values is not None and self.bins is not None:
            raise ValueError(f"Criterion for '{self.column}' cannot have both 'values' and 'bins'")


@dataclass
class SentinelSlot:
    """A single slot defining criteria for one sentinel patient.

    Attributes:
        name: Descriptive name for this slot (e.g., "short_stay_clean").
        filters: Dict of column -> (operator, value) filters.
            Operators: "==", "!=", "<", ">", "<=", ">=", "in"
        sort_by: Optional column to sort by before picking.
        sort_descending: Whether to sort descending.
    """

    name: str
    filters: dict = field(default_factory=dict)
    sort_by: Optional[str] = None
    sort_descending: bool = False


@dataclass
class SentinelConfig:
    """Configuration for sentinel patient selection.

    Attributes:
        n_per_stratum: Number of patients to select per stratum (legacy mode).
        criteria: List of stratification criteria (legacy mode).
        strategy: Selection strategy within each stratum.
        seed: Random seed for reproducibility.
        slots: List of SentinelSlot for targeted selection (preferred).
    """

    n_per_stratum: int = 1
    criteria: List[StratificationCriterion] = field(default_factory=list)
    strategy: SelectionStrategy = SelectionStrategy.RANDOM
    seed: int = 42
    slots: Optional[List[SentinelSlot]] = None


# ---------------------------------------------------------------------------
# Default Criteria Presets
# ---------------------------------------------------------------------------


def default_los_criterion() -> StratificationCriterion:
    """LOS criterion: short (<1 day), medium (1-3 days), long (>3 days)."""
    return StratificationCriterion(
        column="los_days",
        bins=[0, 1, 3, 7, float("inf")],
        bin_labels=["<1d", "1-3d", "3-7d", ">7d"],
    )


def default_age_criterion() -> StratificationCriterion:
    """Age criterion: young, middle, elderly."""
    return StratificationCriterion(
        column="age",
        bins=[0, 40, 65, float("inf")],
        bin_labels=["<40", "40-65", ">65"],
    )


def default_mortality_criterion() -> StratificationCriterion:
    """Mortality criterion (requires label extraction)."""
    return StratificationCriterion(
        column="mortality_24h",
        values=[0, 1],
    )


def default_unit_criterion() -> StratificationCriterion:
    """Care unit criterion."""
    return StratificationCriterion(
        column="first_careunit",
        values=[
            "Medical Intensive Care Unit (MICU)",
            "Surgical Intensive Care Unit (SICU)",
            "Coronary Care Unit (CCU)",
        ],
    )


def get_default_criteria() -> List[StratificationCriterion]:
    """Get default stratification criteria (LOS and age)."""
    return [
        default_los_criterion(),
        default_age_criterion(),
    ]


def get_default_sentinel_slots() -> List[SentinelSlot]:
    """Get default 8 sentinel slots covering key edge cases.

    Returns 8 slots targeting:
    1. Short stay (<1 day) - boundary condition, survived
    2. Short stay + died - label alignment edge case
    3. Medium stay + low missingness - "clean" baseline
    4. Medium stay + high missingness - test imputation
    5. Long stay (>5 days) - test truncation to 48h window
    6. Young patient (<40) - demographic edge
    7. Old patient (>80) - demographic edge
    8. Random sample - catch unexpected issues
    """
    return [
        SentinelSlot(
            name="short_stay_survived",
            filters={"los_days": ("<", 1), "mortality_24h": ("==", 0)},
            sort_by="los_days",
            sort_descending=False,  # shortest first
        ),
        SentinelSlot(
            name="short_stay_died",
            filters={"los_days": ("<", 1), "mortality_24h": ("==", 1)},
            sort_by="los_days",
            sort_descending=False,
        ),
        SentinelSlot(
            name="medium_stay_clean",
            filters={"los_days": (">=", 1), "los_days_max": ("<", 3)},
            sort_by="missingness_pct",
            sort_descending=False,  # lowest missingness first
        ),
        SentinelSlot(
            name="medium_stay_sparse",
            filters={"los_days": (">=", 1), "los_days_max": ("<", 3)},
            sort_by="missingness_pct",
            sort_descending=True,  # highest missingness first
        ),
        SentinelSlot(
            name="long_stay",
            filters={"los_days": (">", 5)},
            sort_by="los_days",
            sort_descending=True,  # longest first
        ),
        SentinelSlot(
            name="young_patient",
            filters={"age": ("<", 40)},
            sort_by="age",
            sort_descending=False,  # youngest first
        ),
        SentinelSlot(
            name="elderly_patient",
            filters={"age": (">", 80)},
            sort_by="age",
            sort_descending=True,  # oldest first
        ),
        SentinelSlot(
            name="random_sample",
            filters={},  # No filters, random selection
        ),
    ]


# ---------------------------------------------------------------------------
# Missingness Computation
# ---------------------------------------------------------------------------


def compute_missingness(
    timeseries_df: pl.DataFrame,
    mask_col: str = "mask",
) -> pl.DataFrame:
    """Compute per-stay missingness percentage from observation masks.

    Args:
        timeseries_df: Dense timeseries DataFrame with mask column.
            The mask column contains nested lists: list[list[bool]].
        mask_col: Column name containing observation masks.

    Returns:
        DataFrame with columns:
            - stay_id: int64
            - missingness_pct: float (0-100, percentage of missing values)
            - n_observed: int (total observed values)
            - n_total: int (total possible values)
    """
    if mask_col not in timeseries_df.columns:
        raise ValueError(f"Column '{mask_col}' not found in DataFrame")

    def _compute_mask_stats(mask: list) -> dict:
        """Compute missingness from nested mask list."""
        arr = np.array(mask, dtype=bool)
        n_total = arr.size
        n_observed = arr.sum()
        missingness_pct = 100.0 * (1 - n_observed / n_total) if n_total > 0 else 100.0
        return {
            "missingness_pct": missingness_pct,
            "n_observed": int(n_observed),
            "n_total": int(n_total),
        }

    # Process each row
    results = []
    for row in timeseries_df.iter_rows(named=True):
        stats = _compute_mask_stats(row[mask_col])
        stats["stay_id"] = row["stay_id"]
        results.append(stats)

    return pl.DataFrame(results).select("stay_id", "missingness_pct", "n_observed", "n_total")


# ---------------------------------------------------------------------------
# Selection Functions
# ---------------------------------------------------------------------------


def _apply_criterion(
    df: pl.DataFrame,
    criterion: StratificationCriterion,
) -> pl.DataFrame:
    """Apply a single stratification criterion to create stratum labels.

    Args:
        df: DataFrame to stratify.
        criterion: Stratification criterion.

    Returns:
        DataFrame with additional stratum column named '{column}_stratum'.
    """
    col = criterion.column
    stratum_col = f"{col}_stratum"

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    if criterion.values is not None:
        # Categorical stratification
        df = df.with_columns(
            pl.when(pl.col(col).is_in(criterion.values))
            .then(pl.col(col).cast(pl.Utf8))
            .otherwise(pl.lit(None))
            .alias(stratum_col)
        )
    else:
        # Numeric binning
        bins = criterion.bins
        labels = criterion.bin_labels or [f"bin_{i}" for i in range(len(bins) - 1)]

        # Create bin labels using cut
        df = df.with_columns(pl.col(col).cut(bins[1:-1], labels=labels).alias(stratum_col))

    return df


def _apply_slot_filter(
    df: pl.DataFrame,
    column: str,
    operator: str,
    value: Union[int, float, str, list],
) -> pl.DataFrame:
    """Apply a single filter condition to a DataFrame.

    Args:
        df: DataFrame to filter.
        column: Column name.
        operator: Comparison operator.
        value: Value to compare against.

    Returns:
        Filtered DataFrame.
    """
    if column not in df.columns:
        return df.filter(pl.lit(False))  # Return empty if column missing

    col = pl.col(column)
    if operator == "==":
        return df.filter(col == value)
    elif operator == "!=":
        return df.filter(col != value)
    elif operator == "<":
        return df.filter(col < value)
    elif operator == ">":
        return df.filter(col > value)
    elif operator == "<=":
        return df.filter(col <= value)
    elif operator == ">=":
        return df.filter(col >= value)
    elif operator == "in":
        return df.filter(col.is_in(value))
    else:
        raise ValueError(f"Unknown operator: {operator}")


def _select_from_slot(
    df: pl.DataFrame,
    slot: SentinelSlot,
    already_selected: set,
    rng: np.random.Generator,
) -> Optional[int]:
    """Select one patient matching a slot's criteria.

    Args:
        df: DataFrame with all patient data.
        slot: SentinelSlot defining selection criteria.
        already_selected: Set of already selected stay_ids to avoid duplicates.
        rng: Random number generator.

    Returns:
        Selected stay_id or None if no match found.
    """
    filtered = df.filter(~pl.col("stay_id").is_in(list(already_selected)))

    # Apply all filters
    for col, (op, val) in slot.filters.items():
        # Handle special case for "column_max" suffix (upper bound on same column)
        if col.endswith("_max"):
            actual_col = col[:-4]  # Remove "_max" suffix
            filtered = _apply_slot_filter(filtered, actual_col, op, val)
        else:
            filtered = _apply_slot_filter(filtered, col, op, val)

    if len(filtered) == 0:
        return None

    # Sort if specified
    if slot.sort_by and slot.sort_by in filtered.columns:
        filtered = filtered.sort(slot.sort_by, descending=slot.sort_descending)
        return int(filtered["stay_id"][0])
    else:
        # Random selection
        idx = int(rng.integers(0, len(filtered)))
        return int(filtered["stay_id"][idx])


def select_sentinel_patients(
    static_df: pl.DataFrame,
    config: Optional[SentinelConfig] = None,
    labels_df: Optional[pl.DataFrame] = None,
    timeseries_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Select sentinel patients for targeted debugging.

    Uses slot-based selection (default: 8 patients covering key edge cases)
    or legacy stratified sampling if config.criteria is provided.

    Args:
        static_df: Static features DataFrame (from static.parquet).
        config: Sentinel selection configuration. If None, uses default 8 slots.
        labels_df: Optional labels DataFrame for mortality-based selection.
        timeseries_df: Optional timeseries for missingness-based selection.

    Returns:
        DataFrame with columns:
            - stay_id: Selected patient stay IDs
            - slot_name: Name of the slot this patient fills
            - All original static columns for reference
    """
    if config is None:
        config = SentinelConfig(slots=get_default_sentinel_slots())

    # Prepare data: join all available info
    df = static_df.clone()

    if labels_df is not None:
        label_cols = [c for c in labels_df.columns if c != "stay_id"]
        df = df.join(labels_df.select(["stay_id"] + label_cols), on="stay_id", how="left")

    if timeseries_df is not None:
        missingness_df = compute_missingness(timeseries_df)
        df = df.join(missingness_df, on="stay_id", how="left")

    rng = np.random.default_rng(config.seed)

    # Use slot-based selection if slots are provided
    if config.slots:
        selected_rows = []
        already_selected: set = set()

        for slot in config.slots:
            stay_id = _select_from_slot(df, slot, already_selected, rng)
            if stay_id is not None:
                selected_rows.append({"stay_id": stay_id, "slot_name": slot.name})
                already_selected.add(stay_id)
            else:
                print(f"Warning: No patient found for slot '{slot.name}'")

        if not selected_rows:
            # Fallback to random if no slots matched
            sample_size = min(8, len(df))
            sampled_ids = rng.choice(df["stay_id"].to_numpy(), size=sample_size, replace=False)
            selected_rows = [{"stay_id": sid, "slot_name": "fallback"} for sid in sampled_ids]

        selected = pl.DataFrame(selected_rows)
        result = selected.join(static_df, on="stay_id", how="left")
        return result

    # Legacy: stratified sampling based on criteria
    stratum_cols = []
    for criterion in config.criteria:
        try:
            df = _apply_criterion(df, criterion)
            stratum_cols.append(f"{criterion.column}_stratum")
        except ValueError as e:
            print(f"Warning: Skipping criterion - {e}")

    if not stratum_cols:
        sample_size = min(config.n_per_stratum * 8, len(df))
        return df.sample(n=sample_size, seed=config.seed).with_columns(
            pl.lit("random").alias("slot_name")
        )

    df = df.with_columns(pl.concat_str(stratum_cols, separator="_").alias("stratum"))
    df = df.filter(pl.col("stratum").is_not_null())

    grouped = df.group_by("stratum").agg(pl.all())
    sampled_rows = []

    for row in grouped.iter_rows(named=True):
        stratum = row["stratum"]
        stay_ids = row["stay_id"]
        n_sample = min(config.n_per_stratum, len(stay_ids))
        sampled = rng.choice(stay_ids, size=n_sample, replace=False)
        for sid in sampled:
            sampled_rows.append({"stay_id": sid, "slot_name": stratum})

    selected = pl.DataFrame(sampled_rows)
    result = selected.join(static_df, on="stay_id", how="left")
    return result


def select_by_missingness(
    timeseries_df: pl.DataFrame,
    n_per_bin: int = 3,
    bins: Optional[List[float]] = None,
    bin_labels: Optional[List[str]] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Select patients stratified by missingness level.

    Args:
        timeseries_df: Dense timeseries DataFrame with mask column.
        n_per_bin: Number of patients per missingness bin.
        bins: Missingness percentage bin edges (default: [0, 10, 30, 50, 70, 100]).
        bin_labels: Labels for bins.
        seed: Random seed.

    Returns:
        DataFrame with columns:
            - stay_id
            - missingness_pct
            - missingness_bin
    """
    if bins is None:
        bins = [0, 10, 30, 50, 70, 100]
    if bin_labels is None:
        bin_labels = ["<10%", "10-30%", "30-50%", "50-70%", ">70%"]

    # Compute missingness
    miss_df = compute_missingness(timeseries_df)

    # Bin missingness
    miss_df = miss_df.with_columns(
        pl.col("missingness_pct").cut(bins[1:-1], labels=bin_labels).alias("missingness_bin")
    )

    # Sample from each bin (handle case where bin has fewer than n_per_bin)
    miss_df_filtered = miss_df.filter(pl.col("missingness_bin").is_not_null())
    grouped = miss_df_filtered.group_by("missingness_bin").agg(pl.all())

    sampled_rows = []
    rng = np.random.default_rng(seed)

    for row in grouped.iter_rows(named=True):
        bin_label = row["missingness_bin"]
        stay_ids = row["stay_id"]
        n_sample = min(n_per_bin, len(stay_ids))
        if n_sample > 0:
            sampled = rng.choice(stay_ids, size=n_sample, replace=False)
            for sid in sampled:
                sampled_rows.append({"stay_id": sid, "missingness_bin": bin_label})

    selected = pl.DataFrame(sampled_rows)

    # Join back missingness values
    result = selected.join(
        miss_df.select("stay_id", "missingness_pct"),
        on="stay_id",
        how="left",
    )

    return result


def select_extreme_stays(
    static_df: pl.DataFrame,
    column: str,
    n_each: int = 3,
) -> pl.DataFrame:
    """Select patients with extreme values for a numeric column.

    Args:
        static_df: Static features DataFrame.
        column: Column name to select extremes for.
        n_each: Number of min and max examples each.

    Returns:
        DataFrame with columns:
            - stay_id
            - {column}: The column value
            - extreme_type: 'min' or 'max'
    """
    if column not in static_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Get min and max values
    min_stays = (
        static_df.filter(pl.col(column).is_not_null())
        .sort(column)
        .head(n_each)
        .with_columns(pl.lit("min").alias("extreme_type"))
    )

    max_stays = (
        static_df.filter(pl.col(column).is_not_null())
        .sort(column, descending=True)
        .head(n_each)
        .with_columns(pl.lit("max").alias("extreme_type"))
    )

    result = pl.concat([min_stays, max_stays])

    return result.select("stay_id", column, "extreme_type")
