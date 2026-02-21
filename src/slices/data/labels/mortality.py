"""Mortality prediction label builders."""

import logging
from typing import Dict

import polars as pl

from .base import LabelBuilder

logger = logging.getLogger(__name__)


class MortalityLabelBuilder(LabelBuilder):
    """Build mortality prediction labels with configurable time windows.

    Supports multiple prediction windows:
    - ICU mortality: Death during ICU stay (window_hours=-1)
    - Hospital mortality: Death before hospital discharge (window_hours=None)
    - Time-bounded mortality: Death within prediction_window after observation ends

    When observation_window_hours is set (recommended):
        - Observation period: [intime, intime + observation_window_hours]
        - Gap period: [obs_end, obs_end + gap_hours] (optional buffer)
        - Prediction period: [gap_end, gap_end + prediction_window_hours]
        - Label = 1 if death occurred during prediction period
        - Label = null if death occurred during observation period (excluded)

    When observation_window_hours is None (legacy behavior):
        - Label = 1 if death occurred within prediction_window_hours of admission
    """

    def build_labels(self, raw_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Build mortality labels from stay and mortality data.

        Expected raw_data sources:
        - 'stays': stay_id, intime, outtime
        - 'mortality_info': stay_id, date_of_death, hospital_expire_flag,
                           dischtime, discharge_location

        Args:
            raw_data: Dictionary with 'stays' and 'mortality_info' DataFrames.

        Returns:
            DataFrame with stay_id and binary label (1=died, 0=survived).
            Label is null for stays where death occurred during observation window.
        """
        self.validate_inputs(raw_data)

        stays = raw_data["stays"]
        mortality = raw_data["mortality_info"]

        # Handle empty DataFrames
        if len(stays) == 0:
            return pl.DataFrame(
                {
                    "stay_id": pl.Series([], dtype=pl.Int64),
                    "label": pl.Series([], dtype=pl.Int32),
                }
            )

        # Join mortality info with stays
        merged = stays.join(mortality, on="stay_id", how="left")

        # Compute label based on prediction window
        window_hours = self.config.prediction_window_hours
        obs_hours = self.config.observation_window_hours
        gap_hours = self.config.gap_hours

        # Validate required temporal fields for windowed tasks
        needs_intime = obs_hours is not None or (
            window_hours is not None and window_hours != -1 and obs_hours is None
        )
        needs_outtime = window_hours == -1
        needs_date_of_death = window_hours is not None

        if needs_intime and merged["intime"].null_count() == len(merged):
            raise ValueError(
                f"Task '{self.config.task_name}' requires 'intime' for windowed labels, "
                "but all values are null. This dataset (e.g., eICU) may not provide "
                "admission timestamps. Use a hospital-mortality task with "
                "prediction_window_hours=null and observation_window_hours=null instead."
            )

        if needs_outtime and merged["outtime"].null_count() == len(merged):
            raise ValueError(
                f"Task '{self.config.task_name}' requires 'outtime' for ICU mortality labels, "
                "but all values are null. This dataset may not provide discharge timestamps."
            )

        if needs_date_of_death and merged["date_of_death"].null_count() == len(merged):
            logger.warning(
                f"Task '{self.config.task_name}': 'date_of_death' is null for all stays. "
                "This dataset (e.g., eICU) may not provide death timestamps. "
                "All windowed mortality labels will be null (excluded). "
                "Consider using hospital-mortality (hospital_expire_flag) instead."
            )

        # Check dtype of date_of_death to handle DATE vs DATETIME properly
        # This avoids edge cases where DATE type is cast to DATETIME with 00:00:00
        date_of_death_dtype = mortality["date_of_death"].dtype
        is_date_type = date_of_death_dtype == pl.Date

        if window_hours is None and obs_hours is None:
            # Hospital mortality, no observation window (legacy)
            # Null hospital_expire_flag means outcome is unknown — keep as null.
            # ICUDataset's handle_missing_labels='filter' will exclude these stays.
            labels = merged.select(
                [
                    "stay_id",
                    pl.col("hospital_expire_flag").cast(pl.Int32).alias("label"),
                ]
            )

        elif window_hours is None and obs_hours is not None:
            # Hospital mortality with observation window exclusion
            # Patients who died during observation are excluded (label=null)
            obs_end_datetime = pl.col("intime") + pl.duration(hours=obs_hours)
            left_icu_during_obs = pl.col("outtime") < obs_end_datetime

            if is_date_type:
                obs_end = obs_end_datetime.cast(pl.Date)
                died_during_obs = (
                    pl.col("date_of_death").is_not_null()
                    & (pl.col("date_of_death") <= obs_end)
                    & left_icu_during_obs
                )
            else:
                died_during_obs = (
                    pl.col("date_of_death").is_not_null()
                    & (pl.col("date_of_death") <= obs_end_datetime)
                    & left_icu_during_obs
                )

            labels = merged.select(
                [
                    "stay_id",
                    pl.when(died_during_obs)
                    .then(None)
                    .when(pl.col("hospital_expire_flag") == 1)
                    .then(1)
                    .otherwise(0)
                    .cast(pl.Int32)
                    .alias("label"),
                ]
            )

        elif window_hours == -1 and obs_hours is None:
            # ICU mortality (died during or at ICU discharge), no observation window
            if is_date_type:
                # For DATE type, compare dates only (ignores time component)
                comparison = pl.col("date_of_death") <= pl.col("outtime").cast(pl.Date)
            else:
                # For DATETIME type, compare full datetime for precision
                comparison = pl.col("date_of_death") <= pl.col("outtime")

            labels = merged.select(
                [
                    "stay_id",
                    pl.when(pl.col("date_of_death").is_not_null() & comparison)
                    .then(1)
                    .otherwise(0)
                    .alias("label"),
                ]
            )

        elif obs_hours is not None:
            # Time-bounded mortality with observation window (recommended)
            # Prediction window starts after observation + gap ends
            labels = self._build_windowed_mortality_labels(
                merged, obs_hours, gap_hours, window_hours, is_date_type
            )

        else:
            # Legacy: Time-bounded mortality from admission (e.g., 24h, 48h from intime)
            if is_date_type:
                # For DATE type, compare dates only
                comparison = pl.col("date_of_death") <= (
                    pl.col("intime") + pl.duration(hours=window_hours)
                ).cast(pl.Date)
            else:
                # For DATETIME type, compare full datetime for precision
                comparison = pl.col("date_of_death") <= (
                    pl.col("intime") + pl.duration(hours=window_hours)
                )

            labels = merged.select(
                [
                    "stay_id",
                    pl.when(pl.col("date_of_death").is_not_null() & comparison)
                    .then(1)
                    .otherwise(0)
                    .alias("label"),
                ]
            )

        return labels

    def _build_windowed_mortality_labels(
        self,
        merged: pl.DataFrame,
        obs_hours: int,
        gap_hours: int,
        prediction_hours: int,
        is_date_type: bool,
    ) -> pl.DataFrame:
        """Build mortality labels with explicit observation and prediction windows.

        Timeline (for bounded prediction window):
            |---- observation ----|-- gap --|---- prediction ----|
            intime            obs_end    gap_end              pred_end

        Timeline (for prediction_hours == -1, until ICU discharge):
            |---- observation ----|-- gap --|---- prediction (until discharge) ----|
            intime            obs_end    gap_end                               outtime

        Args:
            merged: DataFrame with stays and mortality info joined.
            obs_hours: Hours of observation window from admission.
            gap_hours: Hours of gap between observation and prediction.
            prediction_hours: Hours of prediction window, or -1 for "until ICU discharge".
            is_date_type: Whether date_of_death is DATE (vs DATETIME).

        Returns:
            DataFrame with stay_id and label:
            - 1: Death occurred during prediction window
            - 0: Survived prediction window (or died after)
            - null: Death occurred during observation or gap window (excluded)

        Note:
            Uses outtime (ICU discharge time) in addition to date_of_death to determine
            if death occurred during observation. This fixes false positives when
            date_of_death is a DATE type (day-level precision) - if outtime >= obs_end,
            the patient was still in ICU at observation end and could not have died
            during observation.
        """
        # Calculate window boundaries
        # obs_end = intime + obs_hours
        # gap_end = obs_end + gap_hours = intime + obs_hours + gap_hours
        prediction_start_hours = obs_hours + gap_hours

        # Check if prediction window extends until ICU discharge
        until_icu_discharge = prediction_hours == -1

        # Use outtime to determine if patient was still in ICU at end of observation.
        # This is more reliable than date_of_death for DATE types (day-level precision).
        # If outtime >= intime + obs_hours, patient was alive at observation end.
        obs_end_datetime = pl.col("intime") + pl.duration(hours=obs_hours)
        left_icu_during_obs = pl.col("outtime") < obs_end_datetime

        if is_date_type:
            # For DATE type, cast boundaries to Date for comparison
            obs_end = obs_end_datetime.cast(pl.Date)
            pred_start = (pl.col("intime") + pl.duration(hours=prediction_start_hours)).cast(
                pl.Date
            )

            # Death during observation window (exclude these stays)
            # Must check BOTH date_of_death AND outtime because date_of_death has only
            # day-level precision which causes false positives (see issue with DATE vs DATETIME).
            # If outtime >= obs_end, patient was still in ICU at observation end, so they
            # could not have died during observation regardless of what date_of_death says.
            died_during_obs = (
                pl.col("date_of_death").is_not_null()
                & (pl.col("date_of_death") <= obs_end)
                & left_icu_during_obs
            )

            # Death during gap period (exclude — ambiguous whether model could predict)
            died_during_gap = (
                (
                    pl.col("date_of_death").is_not_null()
                    & (pl.col("date_of_death") > obs_end)
                    & (pl.col("date_of_death") < pred_start)
                )
                if gap_hours > 0
                else pl.lit(False)
            )

            if until_icu_discharge:
                # Prediction window ends at ICU discharge (outtime)
                pred_end = pl.col("outtime").cast(pl.Date)
            else:
                # Prediction window ends at fixed time after observation
                prediction_end_hours = prediction_start_hours + prediction_hours
                pred_end = (pl.col("intime") + pl.duration(hours=prediction_end_hours)).cast(
                    pl.Date
                )

            # Death during prediction window
            # Use >= pred_start to include deaths exactly at prediction start
            died_during_pred = (
                pl.col("date_of_death").is_not_null()
                & (pl.col("date_of_death") >= pred_start)
                & (pl.col("date_of_death") <= pred_end)
            )
        else:
            # For DATETIME type, use full precision
            obs_end = obs_end_datetime
            pred_start = pl.col("intime") + pl.duration(hours=prediction_start_hours)

            # Death during observation window (exclude these stays)
            # Also check outtime for consistency with DATE type logic
            died_during_obs = (
                pl.col("date_of_death").is_not_null()
                & (pl.col("date_of_death") <= obs_end)
                & left_icu_during_obs
            )

            # Death during gap period (exclude — ambiguous whether model could predict)
            died_during_gap = (
                (
                    pl.col("date_of_death").is_not_null()
                    & (pl.col("date_of_death") > obs_end)
                    & (pl.col("date_of_death") < pred_start)
                )
                if gap_hours > 0
                else pl.lit(False)
            )

            if until_icu_discharge:
                # Prediction window ends at ICU discharge (outtime)
                pred_end = pl.col("outtime")
            else:
                # Prediction window ends at fixed time after observation
                prediction_end_hours = prediction_start_hours + prediction_hours
                pred_end = pl.col("intime") + pl.duration(hours=prediction_end_hours)

            # Death during prediction window
            # Use >= pred_start to include deaths exactly at prediction start
            died_during_pred = (
                pl.col("date_of_death").is_not_null()
                & (pl.col("date_of_death") >= pred_start)
                & (pl.col("date_of_death") <= pred_end)
            )

        # Build labels:
        # - null if died during observation (can't predict the past)
        # - null if died during gap period (ambiguous, exclude)
        # - 1 if died during prediction window
        # - 0 otherwise (survived or died after prediction window)
        labels = merged.select(
            [
                "stay_id",
                pl.when(died_during_obs)
                .then(None)
                .when(died_during_gap)
                .then(None)
                .when(died_during_pred)
                .then(1)
                .otherwise(0)
                .cast(pl.Int32)
                .alias("label"),
            ]
        )

        return labels
