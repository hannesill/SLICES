"""Mortality prediction label builders."""

from typing import Dict

import polars as pl

from .base import LabelBuilder


class MortalityLabelBuilder(LabelBuilder):
    """Build mortality prediction labels with configurable time windows.

    Supports multiple prediction windows:
    - ICU mortality: Death during ICU stay (window_hours=-1)
    - 24h/48h mortality: Death within N hours of ICU admission
    - Hospital mortality: Death before hospital discharge (window_hours=None)
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

        else:
            # Time-bounded mortality (e.g., 24h, 48h)
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
