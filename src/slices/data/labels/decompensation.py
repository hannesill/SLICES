"""Decompensation prediction label builder (sliding-window)."""

import logging
from typing import Dict, List, Optional

import polars as pl

from .base import LabelBuilder

logger = logging.getLogger(__name__)


class DecompensationLabelBuilder(LabelBuilder):
    """Build decompensation labels using a sliding-window approach.

    Unlike single-label tasks, this produces multiple labels per stay — one
    per valid window position. The output DataFrame contains a ``window_start``
    column to identify each window.

    Label definition for a window starting at hour *t*:
        - Observation period: [t, t + obs_window)
        - Prediction period:  [t + obs_window, t + obs_window + pred_window)
        - Label = 1 if death falls within the prediction period
        - Label = 0 if survived past prediction period (or no death)
        - EXCLUDED if death occurs during observation period
        - EXCLUDED if stay is shorter than t + obs_window
    """

    def build_labels(self, raw_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Build sliding-window decompensation labels.

        Args:
            raw_data: Dict with ``"stays"`` and ``"mortality_info"`` DataFrames.

        Returns:
            DataFrame with columns: stay_id (Int64), window_start (Int64),
            label (Int32).
        """
        self.validate_inputs(raw_data)

        stays = raw_data["stays"]
        mortality = raw_data["mortality_info"]

        empty = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "window_start": pl.Series([], dtype=pl.Int64),
                "label": pl.Series([], dtype=pl.Int32),
            }
        )

        if len(stays) == 0:
            return empty

        obs_window = self.config.observation_window_hours or 48
        pred_window = self.config.prediction_window_hours or 24
        stride = self.config.label_params.get("stride_hours", 6)

        death_hours = self._compute_death_hours(stays, mortality)

        # Compute stay lengths in hours
        stay_lengths: Dict[int, float] = dict(
            zip(
                stays["stay_id"].to_list(),
                (stays["los_days"] * 24.0).cast(pl.Float64).to_list(),
            )
        )

        rows: List[Dict] = []
        for stay_id, stay_length in stay_lengths.items():
            death_hour = death_hours.get(stay_id)
            t_start = 0

            while True:
                obs_end = t_start + obs_window
                pred_end = obs_end + pred_window

                # Stay too short for this window
                if obs_end > stay_length:
                    break

                # Death during observation [t_start, obs_end) — exclude
                if death_hour is not None and t_start <= death_hour < obs_end:
                    t_start += stride
                    continue

                # Determine label
                if death_hour is not None and obs_end <= death_hour < pred_end:
                    label = 1
                else:
                    label = 0

                rows.append({"stay_id": stay_id, "window_start": t_start, "label": label})
                t_start += stride

        if not rows:
            return empty

        result = pl.DataFrame(rows).cast(
            {"stay_id": pl.Int64, "window_start": pl.Int64, "label": pl.Int32}
        )

        n_pos = result.filter(pl.col("label") == 1).height
        n_neg = result.filter(pl.col("label") == 0).height
        n_stays = result["stay_id"].n_unique()
        logger.info(
            f"Decompensation labels: {n_pos} positive, {n_neg} negative "
            f"from {n_stays} stays ({len(result)} total windows)"
        )

        return result

    @staticmethod
    def _compute_death_hours(
        stays: pl.DataFrame, mortality: pl.DataFrame
    ) -> Dict[int, Optional[float]]:
        """Compute hours from admission to death for each stay.

        Args:
            stays: DataFrame with stay_id, intime columns.
            mortality: DataFrame with stay_id, date_of_death columns.

        Returns:
            Dict mapping stay_id to hours until death (None if survived).
        """
        joined = stays.select("stay_id", "intime").join(
            mortality.select("stay_id", "date_of_death"),
            on="stay_id",
            how="left",
        )

        death_hours: Dict[int, Optional[float]] = {}
        for row in joined.iter_rows(named=True):
            sid = row["stay_id"]
            if row["date_of_death"] is None:
                death_hours[sid] = None
            else:
                delta = row["date_of_death"] - row["intime"]
                death_hours[sid] = delta.total_seconds() / 3600.0

        return death_hours
