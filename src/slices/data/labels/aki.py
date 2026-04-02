"""AKI (Acute Kidney Injury) label builder using KDIGO criteria."""

import logging
from typing import Dict

import polars as pl

from .base import LabelBuilder

logger = logging.getLogger(__name__)


class AKILabelBuilder(LabelBuilder):
    """AKI detection using KDIGO Stage 1+ criteria from creatinine trajectory.

    KDIGO criteria (ANY triggers AKI = 1):
    - Creatinine rise >= 0.3 mg/dL within any 48h window
    - Creatinine >= 1.5x baseline within 7 days of baseline measurement

    Baseline: minimum creatinine in first baseline_window_hours (default 48h).
    Detection window: hours [observation_window_hours, observation_window_hours +
    prediction_window_hours) — forward-looking to avoid data leakage.
    Label = null if no creatinine in the prediction window.
    """

    SEMANTIC_VERSION = "1.0.0"

    def build_labels(self, raw_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Build AKI labels from stays and timeseries data.

        Args:
            raw_data: Dict with "stays" and "timeseries" DataFrames.

        Returns:
            DataFrame with stay_id (Int64) and label (Int32, nullable).
        """
        self.validate_inputs(raw_data)
        stays = raw_data["stays"]
        timeseries = raw_data["timeseries"]

        if len(stays) == 0:
            return pl.DataFrame(
                {
                    "stay_id": pl.Series([], dtype=pl.Int64),
                    "label": pl.Series([], dtype=pl.Int32),
                }
            )

        crea_col = self.config.label_params.get("creatinine_col", "crea")
        baseline_hours = self.config.label_params.get("baseline_window_hours", 48)
        abs_threshold = self.config.label_params.get("absolute_rise_threshold", 0.3)
        rel_threshold = self.config.label_params.get("relative_rise_threshold", 1.5)
        rel_window_hours = self.config.label_params.get("relative_window_hours", 168)

        # Forward-looking prediction window
        obs_hours = self.config.observation_window_hours
        prediction_hours = self.config.prediction_window_hours
        if not prediction_hours:
            raise ValueError(
                "AKI task requires prediction_window_hours to define the forward-looking "
                "detection window. Set it in the task YAML (e.g. prediction_window_hours: 24)."
            )
        detection_start = obs_hours
        detection_end = obs_hours + prediction_hours

        stay_ids = stays["stay_id"].to_list()

        # Validate creatinine column exists
        if crea_col not in timeseries.columns:
            logger.warning(f"Creatinine column '{crea_col}' not in timeseries; all labels null")
            return pl.DataFrame(
                {
                    "stay_id": pl.Series(stay_ids, dtype=pl.Int64),
                    "label": pl.Series([None] * len(stay_ids), dtype=pl.Int32),
                }
            )

        # Filter to relevant columns and non-null creatinine rows
        crea_ts = (
            timeseries.select("stay_id", "hour", crea_col)
            .drop_nulls(subset=[crea_col])
            .sort("stay_id", "hour")
        )

        # Pre-compute baselines per stay: min creatinine in first baseline_hours
        baselines = (
            crea_ts.filter(pl.col("hour") < baseline_hours)
            .group_by("stay_id")
            .agg(pl.col(crea_col).min().alias("baseline"))
        )

        # Prediction window creatinine values (detection window)
        post_obs = crea_ts.filter(
            (pl.col("hour") >= detection_start) & (pl.col("hour") < detection_end)
        )

        # Join baselines with post-baseline data
        post_with_baseline = post_obs.join(baselines, on="stay_id", how="inner")

        # --- KDIGO Criterion 1: Relative rise (>= 1.5x baseline within 7 days) ---
        # post_with_baseline already filtered to [detection_start, detection_end)
        rel_aki = (
            post_with_baseline.filter(pl.col("hour") < rel_window_hours)
            .filter(pl.col(crea_col) >= pl.col("baseline") * rel_threshold)
            .select("stay_id")
            .unique()
            .with_columns(pl.lit(True).alias("rel_aki"))
        )

        # --- KDIGO Criterion 2: Absolute rise (>= 0.3 mg/dL within 48h window) ---
        # Use all creatinine data but only flag rises where the later measurement
        # falls in the detection window [detection_start, detection_end).
        abs_aki = self._check_absolute_criterion_vectorized(
            crea_ts, crea_col, abs_threshold, detection_start, detection_end
        )

        # Combine criteria: AKI = 1 if either criterion is met
        stay_df = pl.DataFrame({"stay_id": pl.Series(stay_ids, dtype=pl.Int64)})

        # Stays with any creatinine data (for null vs 0 distinction)
        stays_with_crea = crea_ts.select("stay_id").unique()

        # Stays with baseline (needed for evaluation)
        stays_with_baseline = baselines.select("stay_id").unique()

        # Stays with post-baseline data
        stays_with_post_obs = post_obs.select("stay_id").unique()

        result = (
            stay_df.join(
                stays_with_crea.with_columns(pl.lit(True).alias("has_crea")),
                on="stay_id",
                how="left",
            )
            .join(
                stays_with_baseline.with_columns(pl.lit(True).alias("has_baseline")),
                on="stay_id",
                how="left",
            )
            .join(
                stays_with_post_obs.with_columns(pl.lit(True).alias("has_post_obs")),
                on="stay_id",
                how="left",
            )
            .join(rel_aki, on="stay_id", how="left")
            .join(abs_aki, on="stay_id", how="left")
        )

        # Build label:
        # - null if no creatinine data, no baseline, or no data in prediction window
        # - 1 if either AKI criterion met
        # - 0 otherwise
        result = result.with_columns(
            pl.when(pl.col("has_crea").is_null() | pl.col("has_baseline").is_null())
            .then(None)
            .when(pl.col("has_post_obs").is_null())
            .then(None)
            .when(pl.col("rel_aki").is_not_null() | pl.col("abs_aki").is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("label")
        )

        result_df = result.select(
            pl.col("stay_id").cast(pl.Int64),
            "label",
        )

        n_aki = result_df.filter(pl.col("label") == 1).height
        n_no_aki = result_df.filter(pl.col("label") == 0).height
        n_null = result_df["label"].null_count()
        logger.info(
            f"AKI labels: {n_aki} AKI, {n_no_aki} no AKI, "
            f"{n_null} excluded (no creatinine/baseline or no data in prediction window)"
        )

        return result_df

    def _check_absolute_criterion_vectorized(
        self,
        crea_ts: pl.DataFrame,
        crea_col: str,
        abs_threshold: float,
        detection_start_hour: int,
        detection_end_hour: int,
    ) -> pl.DataFrame:
        """Check KDIGO absolute rise criterion using vectorized operations.

        Finds any pair of creatinine values within a 48h window where the later
        value exceeds the earlier by >= abs_threshold and the later measurement
        falls in the detection window [detection_start_hour, detection_end_hour).

        Returns:
            DataFrame with stay_id column for stays meeting the criterion.
        """
        if len(crea_ts) == 0:
            return pl.DataFrame(
                {
                    "stay_id": pl.Series([], dtype=pl.Int64),
                    "abs_aki": pl.Series([], dtype=pl.Boolean),
                }
            )

        # Self-join within each stay: pair each measurement with later ones
        left = crea_ts.select(
            pl.col("stay_id"),
            pl.col("hour").alias("hour_i"),
            pl.col(crea_col).alias("crea_i"),
        )
        right = crea_ts.select(
            pl.col("stay_id"),
            pl.col("hour").alias("hour_j"),
            pl.col(crea_col).alias("crea_j"),
        )

        # Join on stay_id and filter: j > i, within 48h, rise >= threshold,
        # and the later measurement is in the detection window [start, end)
        pairs = left.join(right, on="stay_id", how="inner").filter(
            (pl.col("hour_j") > pl.col("hour_i"))
            & (pl.col("hour_j") - pl.col("hour_i") <= 48)
            & (pl.col("hour_j") >= detection_start_hour)
            & (pl.col("hour_j") < detection_end_hour)
            & (pl.col("crea_j") - pl.col("crea_i") >= abs_threshold)
        )

        return pairs.select("stay_id").unique().with_columns(pl.lit(True).alias("abs_aki"))
