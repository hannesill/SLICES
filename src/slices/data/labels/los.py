"""Remaining length of stay label builder."""

import logging
from typing import Dict

import polars as pl

from .base import LabelBuilder

logger = logging.getLogger(__name__)


class LOSLabelBuilder(LabelBuilder):
    """Remaining length of stay = los_days - (obs_window_hours / 24).

    Only needs 'stays' data source. Returns Float64 label.
    Clips at 0.0 (stays exactly at boundary get 0).
    """

    def build_labels(self, raw_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        self.validate_inputs(raw_data)
        stays = raw_data["stays"]

        if len(stays) == 0:
            return pl.DataFrame(
                {
                    "stay_id": pl.Series([], dtype=pl.Int64),
                    "label": pl.Series([], dtype=pl.Float64),
                }
            )

        obs_hours = self.config.observation_window_hours or 0
        obs_days = obs_hours / 24.0

        labels = stays.select(
            [
                "stay_id",
                pl.when(pl.col("los_days") - obs_days < 0.0)
                .then(0.0)
                .otherwise(pl.col("los_days") - obs_days)
                .cast(pl.Float64)
                .alias("label"),
            ]
        )

        # Log distribution stats
        non_null = labels["label"].drop_nulls()
        if len(non_null) > 0:
            mean = float(non_null.mean())  # type: ignore[arg-type]
            median = float(non_null.median())  # type: ignore[arg-type]
            std = non_null.std()
            std_str = f"{std:.1f}" if std is not None else "n/a"
            min_val = float(non_null.min())  # type: ignore[arg-type]
            max_val = float(non_null.max())  # type: ignore[arg-type]
            logger.info(
                f"LOS label distribution: mean={mean:.1f}d, "
                f"median={median:.1f}d, std={std_str}d, "
                f"min={min_val:.1f}d, max={max_val:.1f}d"
            )

        return labels
