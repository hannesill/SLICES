"""AKI (Acute Kidney Injury) label builder using KDIGO criteria."""

import logging
from typing import Dict, List

import polars as pl

from .base import LabelBuilder

logger = logging.getLogger(__name__)


class AKILabelBuilder(LabelBuilder):
    """AKI detection using KDIGO Stage 1+ criteria from creatinine trajectory.

    KDIGO criteria (ANY triggers AKI = 1):
    - Creatinine rise >= 0.3 mg/dL within any 48h window
    - Creatinine >= 1.5x baseline within 7 days

    Baseline: minimum creatinine in first 24h (standard clinical proxy).
    Only evaluates criteria AFTER the observation window ends.
    Label = null if no creatinine measurements available.
    """

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

        obs_hours = self.config.observation_window_hours or 48
        crea_col = self.config.label_params.get("creatinine_col", "crea")
        baseline_hours = self.config.label_params.get("baseline_window_hours", 24)
        abs_threshold = self.config.label_params.get("absolute_rise_threshold", 0.3)
        rel_threshold = self.config.label_params.get("relative_rise_threshold", 1.5)

        results: List[Dict] = []
        for stay_id in stays["stay_id"].to_list():
            stay_ts = timeseries.filter(pl.col("stay_id") == stay_id)

            # Check creatinine column exists and has data
            if crea_col not in stay_ts.columns:
                results.append({"stay_id": stay_id, "label": None})
                continue

            crea_data = stay_ts.select("hour", crea_col).drop_nulls().sort("hour")
            if len(crea_data) == 0:
                results.append({"stay_id": stay_id, "label": None})
                continue

            # Baseline: min creatinine in first baseline_hours
            baseline_crea = crea_data.filter(pl.col("hour") < baseline_hours)[crea_col]
            if len(baseline_crea) == 0:
                results.append({"stay_id": stay_id, "label": None})
                continue
            baseline = baseline_crea.min()

            # Only evaluate AFTER observation window
            post_obs = crea_data.filter(pl.col("hour") >= obs_hours)
            if len(post_obs) == 0:
                results.append({"stay_id": stay_id, "label": 0})
                continue

            aki = self._check_kdigo_criteria(
                post_obs, crea_col, baseline, abs_threshold, rel_threshold
            )
            results.append({"stay_id": stay_id, "label": int(aki)})

        result_df = pl.DataFrame(results)
        # Cast types explicitly
        result_df = result_df.cast({"stay_id": pl.Int64, "label": pl.Int32})

        n_aki = result_df.filter(pl.col("label") == 1).height
        n_no_aki = result_df.filter(pl.col("label") == 0).height
        n_null = result_df["label"].null_count()
        logger.info(
            f"AKI labels: {n_aki} AKI, {n_no_aki} no AKI, {n_null} excluded (no creatinine)"
        )

        return result_df

    def _check_kdigo_criteria(
        self,
        crea_df: pl.DataFrame,
        crea_col: str,
        baseline: float,
        abs_threshold: float,
        rel_threshold: float,
    ) -> bool:
        """Check KDIGO Stage 1+ criteria.

        Returns True if:
        - Any creatinine rise >= abs_threshold within 48h sliding window, OR
        - Any creatinine >= rel_threshold * baseline
        """
        values = crea_df[crea_col].to_list()
        hours = crea_df["hour"].to_list()

        # Check relative rise (any value >= 1.5x baseline)
        for v in values:
            if v >= rel_threshold * baseline:
                return True

        # Check absolute rise (>= 0.3 within any 48h window)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if hours[j] - hours[i] <= 48:
                    if values[j] - values[i] >= abs_threshold:
                        return True
                else:
                    break  # hours are sorted, no need to check further

        return False
