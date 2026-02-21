"""Death-hours label builder for decompensation prediction.

Stores hours from admission to death as a per-stay Float64 column.
Binary per-window labels are computed on-the-fly by SlidingWindowDataset.
"""

import logging
from typing import Dict, Optional

import polars as pl

from .base import LabelBuilder

logger = logging.getLogger(__name__)


class DeathHoursLabelBuilder(LabelBuilder):
    """Build death_hours labels: one row per stay.

    Output column ``label`` is Float64:
    - Hours from admission to death for deceased patients
    - ``float('inf')`` for survivors (avoids null-based filtering in ICUDataset)
    """

    def build_labels(self, raw_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Build per-stay death_hours labels.

        Args:
            raw_data: Dict with ``"stays"`` and ``"mortality_info"`` DataFrames.

        Returns:
            DataFrame with columns: stay_id (Int64), label (Float64).
        """
        self.validate_inputs(raw_data)

        stays = raw_data["stays"]
        mortality = raw_data["mortality_info"]

        if len(stays) == 0:
            return pl.DataFrame(
                {
                    "stay_id": pl.Series([], dtype=pl.Int64),
                    "label": pl.Series([], dtype=pl.Float64),
                }
            )

        death_hours = self._compute_death_hours(stays, mortality)

        rows = []
        for stay_id in stays["stay_id"].to_list():
            dh = death_hours.get(stay_id)
            rows.append(
                {
                    "stay_id": stay_id,
                    "label": dh if dh is not None else float("inf"),
                }
            )

        result = pl.DataFrame(rows).cast({"stay_id": pl.Int64, "label": pl.Float64})

        n_deceased = sum(1 for v in death_hours.values() if v is not None)
        n_survivors = len(result) - n_deceased
        logger.info(
            f"Death-hours labels: {n_deceased} deceased, {n_survivors} survivors "
            f"({len(result)} total stays)"
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
