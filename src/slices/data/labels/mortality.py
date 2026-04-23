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

    Precision-aware labeling:
        When death_time_precision is available (new schema), uses it to determine
        comparison logic per row:
        - "timestamp": exact datetime comparison using death_time
        - "date": interval-based conservative comparison using death_date
          (treats as [00:00:00, 23:59:59.999] interval; boundary overlaps → null)
        - "unknown" or missing: uses explicit death evidence and preserves
          unknown outcomes as null
    """

    SEMANTIC_VERSION = "2.2.0"

    def build_labels(self, raw_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Build mortality labels from stay and mortality data.

        Expected raw_data sources:
        - 'stays': stay_id, intime, outtime
        - 'mortality_info': stay_id, death_time, death_date, death_time_precision,
                           death_source, hospital_expire_flag, dischtime,
                           discharge_location (and optionally legacy date_of_death)

        Args:
            raw_data: Dictionary with 'stays' and 'mortality_info' DataFrames.

        Returns:
            DataFrame with stay_id and binary label (1=died, 0=survived).
            Label is null for stays where death occurred during observation window,
            or where date-only precision cannot resolve boundary cases.
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

        # Ensure precision-aware columns and nullable outcome evidence exist
        # (backward compat with legacy data).
        merged = self._ensure_precision_columns(merged)

        # Compute label based on prediction window
        window_hours = self.config.prediction_window_hours
        obs_hours = self.config.observation_window_hours
        gap_hours = self.config.gap_hours

        # Validate required temporal fields for windowed tasks
        needs_intime = obs_hours is not None or (
            window_hours is not None and window_hours != -1 and obs_hours is None
        )
        needs_outtime = window_hours == -1

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

        if window_hours is None and obs_hours is None:
            # Hospital mortality, no observation window (legacy)
            labels = merged.select(
                [
                    "stay_id",
                    pl.when(self._death_evidence_expr())
                    .then(1)
                    .when(self._known_survivor_expr())
                    .then(0)
                    .otherwise(None)
                    .cast(pl.Int32)
                    .alias("label"),
                ]
            )

        elif window_hours is None and obs_hours is not None:
            # Hospital mortality with observation window exclusion
            labels = self._build_hospital_mortality_with_obs(merged, obs_hours)

        elif window_hours == -1 and obs_hours is None:
            # ICU mortality (died during or at ICU discharge), no observation window
            labels = self._build_icu_mortality(merged)

        elif obs_hours is not None:
            # Time-bounded mortality with observation window (recommended path)
            labels = self._build_windowed_mortality_labels(
                merged, obs_hours, gap_hours, window_hours
            )

        else:
            # Legacy: Time-bounded mortality from admission
            labels = self._build_legacy_windowed(merged, window_hours)

        return labels

    @staticmethod
    def _ensure_precision_columns(merged: pl.DataFrame) -> pl.DataFrame:
        """Ensure precision-aware columns exist, migrating legacy data if needed."""
        merged = MortalityLabelBuilder._normalize_datetime_columns(merged)
        merged = MortalityLabelBuilder._ensure_outcome_columns(merged)

        if "death_time_precision" in merged.columns:
            # Keep death_time in the repo's canonical tz-naive datetime form.
            if "death_time" in merged.columns:
                dt_dtype = merged["death_time"].dtype
                if dt_dtype not in (
                    pl.Datetime,
                    pl.Datetime("us"),
                    pl.Datetime("ns"),
                    pl.Datetime("ms"),
                ):
                    # Strip timezone if present, or cast to Datetime
                    merged = merged.with_columns(
                        pl.col("death_time").cast(pl.Datetime("us")).alias("death_time")
                    )
            else:
                merged = merged.with_columns(
                    pl.lit(None).cast(pl.Datetime("us")).alias("death_time")
                )
            if "death_date" not in merged.columns:
                merged = merged.with_columns(pl.lit(None).cast(pl.Date).alias("death_date"))
            if "death_source" not in merged.columns:
                merged = merged.with_columns(pl.lit(None).cast(pl.Utf8).alias("death_source"))
            return merged

        # Legacy schema: only date_of_death exists
        if "date_of_death" not in merged.columns:
            # No death info at all — add empty precision columns
            return merged.with_columns(
                pl.lit(None).cast(pl.Datetime("us")).alias("death_time"),
                pl.lit(None).cast(pl.Date).alias("death_date"),
                pl.lit(None).cast(pl.Utf8).alias("death_time_precision"),
                pl.lit(None).cast(pl.Utf8).alias("death_source"),
            )

        # Infer precision from dtype
        dod_dtype = merged["date_of_death"].dtype
        if dod_dtype == pl.Date:
            return merged.with_columns(
                pl.lit(None).cast(pl.Datetime("us")).alias("death_time"),
                pl.col("date_of_death").cast(pl.Date).alias("death_date"),
                pl.when(pl.col("date_of_death").is_not_null())
                .then(pl.lit("date"))
                .otherwise(pl.lit(None))
                .alias("death_time_precision"),
                pl.lit("legacy").alias("death_source"),
            )
        else:
            # Legacy datetimes are ambiguous: older exports may have stored
            # date-only values as midnight-cast timestamps. Treat them as
            # date precision unless the upstream schema explicitly provides
            # a death_time_precision column.
            if dod_dtype not in (
                pl.Datetime,
                pl.Datetime("us"),
                pl.Datetime("ns"),
                pl.Datetime("ms"),
                pl.Datetime("us"),
            ):
                merged = merged.with_columns(
                    pl.col("date_of_death").cast(pl.Datetime("us")).alias("date_of_death")
                )
            return merged.with_columns(
                pl.lit(None).cast(pl.Datetime("us")).alias("death_time"),
                pl.col("date_of_death").cast(pl.Date).alias("death_date"),
                pl.when(pl.col("date_of_death").is_not_null())
                .then(pl.lit("date"))
                .otherwise(pl.lit(None))
                .alias("death_time_precision"),
                pl.lit("legacy").alias("death_source"),
            )

    @staticmethod
    def _ensure_outcome_columns(merged: pl.DataFrame) -> pl.DataFrame:
        """Ensure nullable outcome/evidence columns exist with stable dtypes."""
        exprs = []

        if "hospital_expire_flag" in merged.columns:
            exprs.append(
                pl.col("hospital_expire_flag").cast(pl.Int32).alias("hospital_expire_flag")
            )
        else:
            exprs.append(pl.lit(None).cast(pl.Int32).alias("hospital_expire_flag"))

        if "dischtime" in merged.columns:
            dtype = merged["dischtime"].dtype
            if dtype not in (
                pl.Datetime,
                pl.Datetime("us"),
                pl.Datetime("ns"),
                pl.Datetime("ms"),
            ):
                exprs.append(pl.col("dischtime").cast(pl.Datetime("us")).alias("dischtime"))
        else:
            exprs.append(pl.lit(None).cast(pl.Datetime("us")).alias("dischtime"))

        if "discharge_location" in merged.columns:
            exprs.append(pl.col("discharge_location").cast(pl.Utf8).alias("discharge_location"))
        else:
            exprs.append(pl.lit(None).cast(pl.Utf8).alias("discharge_location"))

        if exprs:
            merged = merged.with_columns(exprs)

        return merged

    @staticmethod
    def _normalize_datetime_columns(merged: pl.DataFrame) -> pl.DataFrame:
        """Normalize timestamp columns to tz-naive microsecond datetimes.

        RICU parquet exports use UTC-aware timestamps. Downstream label logic
        also compares against date-only fields, so all temporal columns must
        share the same tz-naive dtype before any comparisons run.
        """
        exprs = []
        for column in ("intime", "outtime", "dischtime", "death_time", "date_of_death"):
            if column not in merged.columns:
                continue

            dtype = merged[column].dtype
            if isinstance(dtype, pl.datatypes.Datetime):
                exprs.append(pl.col(column).cast(pl.Datetime("us")).alias(column))

        if exprs:
            return merged.with_columns(exprs)

        return merged

    @staticmethod
    def _death_location_evidence_expr() -> pl.Expr:
        """Return true for discharge destinations/statuses that directly encode death."""
        normalized_location = pl.col("discharge_location").str.strip_chars().str.to_lowercase()
        return normalized_location.is_in(["death", "died", "expired", "deceased", "dead"])

    @staticmethod
    def _death_evidence_expr() -> pl.Expr:
        """Return true when any reliable mortality evidence is present."""
        flag_death = (pl.col("hospital_expire_flag") == 1).fill_null(False)
        location_death = MortalityLabelBuilder._death_location_evidence_expr().fill_null(False)
        timed_death = pl.col("death_time").is_not_null() | pl.col("death_date").is_not_null()
        return flag_death | location_death | timed_death

    @staticmethod
    def _known_survivor_expr() -> pl.Expr:
        """Return true only for explicit survivor outcomes with no conflicting death evidence."""
        return (
            (pl.col("hospital_expire_flag") == 0) & ~MortalityLabelBuilder._death_evidence_expr()
        ).fill_null(False)

    @staticmethod
    def _effective_death_time_expr() -> pl.Expr:
        """Use exact death_time, or death-coded discharge time when no death_date exists."""
        discharge_death_time = (
            pl.when(
                MortalityLabelBuilder._death_evidence_expr()
                & pl.col("death_date").is_null()
                & pl.col("dischtime").is_not_null()
            )
            .then(pl.col("dischtime"))
            .otherwise(pl.lit(None).cast(pl.Datetime("us")))
        )
        return pl.coalesce([pl.col("death_time"), discharge_death_time])

    @staticmethod
    def _effective_death_precision_expr() -> pl.Expr:
        """Return the precision available after discharge-time death evidence fallback."""
        effective_death_time = MortalityLabelBuilder._effective_death_time_expr()
        return (
            pl.when(effective_death_time.is_not_null())
            .then(pl.lit("timestamp"))
            .when(pl.col("death_date").is_not_null())
            .then(pl.lit("date"))
            .when(MortalityLabelBuilder._death_evidence_expr())
            .then(pl.lit("unknown"))
            .otherwise(pl.lit(None))
        )

    def _build_hospital_mortality_with_obs(
        self, merged: pl.DataFrame, obs_hours: int
    ) -> pl.DataFrame:
        """Hospital mortality with observation window exclusion."""
        obs_end = pl.col("intime") + pl.duration(hours=obs_hours)
        effective_death_time = self._effective_death_time_expr()
        effective_precision = self._effective_death_precision_expr()

        # Observation windows are half-open: [intime, obs_end). Deaths exactly
        # at obs_end belong to the prediction period when gap_hours == 0.
        ts_died_during_obs = effective_death_time.is_not_null() & (effective_death_time < obs_end)

        # For date precision: conservative interval logic. A date-only death is
        # [00:00, 23:59:59]; if it is fully before obs_end or overlaps the
        # boundary from the observation side, exclude.
        date_start = pl.col("death_date").cast(pl.Datetime("us"))
        date_end = date_start + pl.duration(hours=23, minutes=59, seconds=59)
        date_died_during_obs = pl.col("death_date").is_not_null() & (date_end < obs_end)
        date_obs_boundary = (
            pl.col("death_date").is_not_null() & (date_start < obs_end) & (date_end >= obs_end)
        )

        died_during_obs = (
            pl.when(effective_precision == "timestamp")
            .then(ts_died_during_obs)
            .when(effective_precision == "date")
            .then(date_died_during_obs | date_obs_boundary)
            .otherwise(pl.lit(False))
        )

        return merged.select(
            [
                "stay_id",
                pl.when(died_during_obs)
                .then(None)
                .when(self._death_evidence_expr())
                .then(1)
                .when(self._known_survivor_expr())
                .then(0)
                .otherwise(None)
                .cast(pl.Int32)
                .alias("label"),
            ]
        )

    def _build_icu_mortality(self, merged: pl.DataFrame) -> pl.DataFrame:
        """ICU mortality (died during or at ICU discharge), no observation window."""
        effective_death_time = self._effective_death_time_expr()
        effective_precision = self._effective_death_precision_expr()

        ts_died = effective_death_time.is_not_null() & (effective_death_time <= pl.col("outtime"))
        date_died = pl.col("death_date").is_not_null() & (
            pl.col("death_date") <= pl.col("outtime").cast(pl.Date)
        )

        died_in_icu = (
            pl.when(effective_precision == "timestamp")
            .then(ts_died)
            .when(effective_precision == "date")
            .then(date_died)
            .otherwise(pl.lit(None))
        )
        known_not_icu_death = self._death_evidence_expr() & died_in_icu.is_not_null() & ~died_in_icu

        return merged.select(
            [
                "stay_id",
                pl.when(died_in_icu)
                .then(1)
                .when(self._known_survivor_expr() | known_not_icu_death)
                .then(0)
                .otherwise(None)
                .cast(pl.Int32)
                .alias("label"),
            ]
        )

    def _build_legacy_windowed(self, merged: pl.DataFrame, window_hours: int) -> pl.DataFrame:
        """Legacy: Time-bounded mortality from admission."""
        boundary = pl.col("intime") + pl.duration(hours=window_hours)
        effective_death_time = self._effective_death_time_expr()
        effective_precision = self._effective_death_precision_expr()

        ts_died = effective_death_time.is_not_null() & (effective_death_time <= boundary)
        # Date-only: death_date end of day <= boundary
        date_died_definite = pl.col("death_date").is_not_null() & (
            (
                pl.col("death_date").cast(pl.Datetime("us"))
                + pl.duration(hours=23, minutes=59, seconds=59)
            )
            <= boundary
        )
        # Date-only: boundary overlap (death_date start-of-day <= boundary < end-of-day)
        date_boundary_overlap = pl.col("death_date").is_not_null() & (
            (pl.col("death_date").cast(pl.Datetime("us")) <= boundary)
            & (
                boundary
                < (
                    pl.col("death_date").cast(pl.Datetime("us"))
                    + pl.duration(hours=23, minutes=59, seconds=59)
                )
            )
        )

        died = (
            pl.when(effective_precision == "timestamp")
            .then(ts_died)
            .when(effective_precision == "date")
            .then(
                pl.when(date_died_definite)
                .then(pl.lit(True))
                .when(date_boundary_overlap)
                .then(pl.lit(None))  # null = ambiguous
                .otherwise(pl.lit(False))
            )
            .otherwise(pl.lit(None))
        )

        return merged.select(
            [
                "stay_id",
                pl.when(died.is_null() & self._known_survivor_expr())
                .then(0)
                .when(died.is_null())
                .then(None)
                .when(died)
                .then(1)
                .otherwise(0)
                .cast(pl.Int32)
                .alias("label"),
            ]
        )

    def _build_windowed_mortality_labels(
        self,
        merged: pl.DataFrame,
        obs_hours: int,
        gap_hours: int,
        prediction_hours: int,
    ) -> pl.DataFrame:
        """Build mortality labels with explicit observation and prediction windows.

        Timeline (for bounded prediction window):
            |---- observation ----|-- gap --|---- prediction ----|
            intime            obs_end    gap_end              pred_end

        Uses precision-aware per-row logic:
        - "timestamp": exact datetime comparison using death_time
        - "date": interval-based [00:00:00, 23:59:59.999] comparison using death_date;
          boundary overlaps produce null (conservative exclusion)
        - "unknown"/missing: true survivors remain 0, unknown outcomes are null
        """
        prediction_start_hours = obs_hours + gap_hours
        until_icu_discharge = prediction_hours == -1
        effective_death_time = self._effective_death_time_expr()
        effective_precision = self._effective_death_precision_expr()

        obs_end = pl.col("intime") + pl.duration(hours=obs_hours)
        pred_start = pl.col("intime") + pl.duration(hours=prediction_start_hours)

        if until_icu_discharge:
            pred_end = pl.col("outtime")
        else:
            prediction_end_hours = prediction_start_hours + prediction_hours
            pred_end = pl.col("intime") + pl.duration(hours=prediction_end_hours)

        # --- Timestamp precision: exact comparisons ---
        ts_died_during_obs = effective_death_time.is_not_null() & (effective_death_time < obs_end)
        ts_died_during_gap = (
            (
                effective_death_time.is_not_null()
                & (effective_death_time > obs_end)
                & (effective_death_time < pred_start)
            )
            if gap_hours > 0
            else pl.lit(False)
        )
        ts_died_during_pred = (
            effective_death_time.is_not_null()
            & (effective_death_time >= pred_start)
            & (effective_death_time <= pred_end)
        )

        # --- Date precision: interval-based conservative logic ---
        # A date-only death represents [date 00:00:00, date 23:59:59.999].
        # "Definite" means the entire interval falls within the region.
        # "Overlap" means the interval straddles a boundary → null.
        date_start = pl.col("death_date").cast(pl.Datetime("us"))
        date_end = date_start + pl.duration(hours=23, minutes=59, seconds=59)

        # Died before obs_end? The entire date interval is strictly before the
        # half-open observation boundary.
        date_died_during_obs = pl.col("death_date").is_not_null() & (date_end < obs_end)

        # Date interval overlaps obs_end boundary (part before, part after)
        date_obs_boundary = (
            pl.col("death_date").is_not_null() & (date_start < obs_end) & (date_end >= obs_end)
        )

        # Definite during prediction: entire interval within [pred_start, pred_end]
        date_died_during_pred_definite = (
            pl.col("death_date").is_not_null() & (date_start >= pred_start) & (date_end <= pred_end)
        )

        # Date interval overlaps pred_start boundary
        date_pred_start_overlap = (
            pl.col("death_date").is_not_null()
            & (date_start < pred_start)
            & (date_end >= pred_start)
        )

        # Date interval overlaps pred_end boundary
        date_pred_end_overlap = (
            pl.col("death_date").is_not_null() & (date_start <= pred_end) & (date_end > pred_end)
        )

        # Definite during gap: entire interval within (obs_end, pred_start)
        date_died_during_gap = (
            (pl.col("death_date").is_not_null() & (date_start > obs_end) & (date_end < pred_start))
            if gap_hours > 0
            else pl.lit(False)
        )

        # Any boundary overlap → ambiguous → null
        date_any_boundary = date_obs_boundary | date_pred_start_overlap | date_pred_end_overlap

        # --- Per-row label computation ---
        # Timestamp precision path
        ts_label = (
            pl.when(ts_died_during_obs)
            .then(None)
            .when(ts_died_during_gap)
            .then(None)
            .when(ts_died_during_pred)
            .then(1)
            .otherwise(0)
        )

        # Date precision path
        date_label = (
            pl.when(date_any_boundary)
            .then(None)  # ambiguous boundary → exclude
            .when(date_died_during_obs)
            .then(None)  # definite during obs → exclude
            .when(date_died_during_gap)
            .then(None)  # definite during gap → exclude
            .when(date_died_during_pred_definite)
            .then(1)  # definite during prediction → positive
            .otherwise(0)  # definite outside → negative
        )

        # Combine by precision
        label_expr = (
            pl.when(effective_precision == "timestamp")
            .then(ts_label)
            .when(effective_precision == "date")
            .then(date_label)
            .otherwise(
                # Unknown death timing or unknown outcome → exclude. Explicit
                # survivors remain negatives.
                pl.when(self._known_survivor_expr())
                .then(0)
                .otherwise(None)
            )
        )

        return merged.select(
            [
                "stay_id",
                label_expr.cast(pl.Int32).alias("label"),
            ]
        )
