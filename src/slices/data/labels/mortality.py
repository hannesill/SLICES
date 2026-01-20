"""Mortality prediction label builders."""

from typing import Dict

import polars as pl

from .base import LabelBuilder


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

        # Check dtype of date_of_death to handle DATE vs DATETIME properly
        # This avoids edge cases where DATE type is cast to DATETIME with 00:00:00
        date_of_death_dtype = mortality["date_of_death"].dtype
        is_date_type = date_of_death_dtype == pl.Date

        # Compute label based on prediction window
        window_hours = self.config.prediction_window_hours
        obs_hours = self.config.observation_window_hours
        gap_hours = self.config.gap_hours

        if window_hours is None:
            # Hospital mortality (default)
            labels = merged.select(
                [
                    "stay_id",
                    pl.col("hospital_expire_flag").fill_null(0).cast(pl.Int32).alias("label"),
                ]
            )

        elif window_hours == -1:
            # ICU mortality (died during or at ICU discharge)
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

        Timeline:
            |---- observation ----|-- gap --|---- prediction ----|
            intime            obs_end    gap_end              pred_end

        Args:
            merged: DataFrame with stays and mortality info joined.
            obs_hours: Hours of observation window from admission.
            gap_hours: Hours of gap between observation and prediction.
            prediction_hours: Hours of prediction window.
            is_date_type: Whether date_of_death is DATE (vs DATETIME).

        Returns:
            DataFrame with stay_id and label:
            - 1: Death occurred during prediction window
            - 0: Survived prediction window (or died after)
            - null: Death occurred during observation window (excluded)
        """
        # Calculate window boundaries
        # obs_end = intime + obs_hours
        # gap_end = obs_end + gap_hours = intime + obs_hours + gap_hours
        # pred_end = gap_end + prediction_hours = intime + obs_hours + gap_hours + prediction_hours
        prediction_start_hours = obs_hours + gap_hours
        prediction_end_hours = prediction_start_hours + prediction_hours

        if is_date_type:
            # For DATE type, cast boundaries to Date for comparison
            obs_end = (pl.col("intime") + pl.duration(hours=obs_hours)).cast(pl.Date)
            pred_start = (pl.col("intime") + pl.duration(hours=prediction_start_hours)).cast(
                pl.Date
            )
            pred_end = (pl.col("intime") + pl.duration(hours=prediction_end_hours)).cast(pl.Date)

            # Death during observation window (exclude these stays)
            died_during_obs = pl.col("date_of_death").is_not_null() & (
                pl.col("date_of_death") <= obs_end
            )

            # Death during prediction window
            died_during_pred = (
                pl.col("date_of_death").is_not_null()
                & (pl.col("date_of_death") > pred_start)
                & (pl.col("date_of_death") <= pred_end)
            )
        else:
            # For DATETIME type, use full precision
            obs_end = pl.col("intime") + pl.duration(hours=obs_hours)
            pred_start = pl.col("intime") + pl.duration(hours=prediction_start_hours)
            pred_end = pl.col("intime") + pl.duration(hours=prediction_end_hours)

            # Death during observation window (exclude these stays)
            died_during_obs = pl.col("date_of_death").is_not_null() & (
                pl.col("date_of_death") <= obs_end
            )

            # Death during prediction window
            died_during_pred = (
                pl.col("date_of_death").is_not_null()
                & (pl.col("date_of_death") > pred_start)
                & (pl.col("date_of_death") <= pred_end)
            )

        # Build labels:
        # - null if died during observation (can't predict the past)
        # - 1 if died during prediction window
        # - 0 otherwise (survived or died after prediction window)
        labels = merged.select(
            [
                "stay_id",
                pl.when(died_during_obs)
                .then(None)
                .when(died_during_pred)
                .then(1)
                .otherwise(0)
                .cast(pl.Int32)
                .alias("label"),
            ]
        )

        return labels
