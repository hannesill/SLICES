"""Benchmark invariants for SLICES.

These constants define the fixed parameters of the SLICES benchmark.
They should NOT be varied between experiments — they are part of the
benchmark definition (cohort criteria, evaluation protocol, preprocessing).

Experiment knobs (model architecture, SSL objective, learning rate, etc.)
remain in Hydra configs under configs/.
"""

# =============================================================================
# Observation Window
# =============================================================================
SEQ_LENGTH_HOURS: int = 24  # Model input window
MIN_STAY_HOURS: int = 24
# Benchmark label horizon: 24h observation + up to 24h forward prediction
# (e.g. mortality_24h / AKI from hours 24-48).
LABEL_HORIZON_HOURS: int = 48

# =============================================================================
# Extraction
# =============================================================================
EXTRACTION_BATCH_SIZE: int = 5000

# =============================================================================
# Patient-Level Splits
# =============================================================================
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# =============================================================================
# Preprocessing
# =============================================================================
NORMALIZE: bool = True
PIN_MEMORY: bool = True

# =============================================================================
# Feature Blocklist
# =============================================================================
# RICU concepts that must be excluded from model input because they either
# directly leak labels, encode future treatment duration, or are high-level
# summaries derived from lower-level features already present in the benchmark.
DIRECT_LEAKAGE_FEATURES: frozenset[str] = frozenset(
    {
        # Updated hourly by RICU and directly reveal length-of-stay targets.
        "los_hosp",
        "los_icu",
        # Internal ricu helper variable, not a clinical concept.
        "dur_var",
    }
)

FUTURE_DERIVED_FEATURES: frozenset[str] = frozenset(
    {
        # Medication duration concepts use stop/end times and can encode
        # post-observation-window treatment duration in rows timestamped <24h.
        "dobu_dur",
        "dopa_dur",
        "epi_dur",
        "norepi_dur",
        # Derived from vasopressor rate + duration concepts.
        "dobu60",
        "dopa60",
        "epi60",
        "norepi60",
        "vaso_ind",
        # Derived sepsis/cardio-SOFA features downstream of the duration concepts
        # or suspected-infection windows; keep raw ingredients instead.
        "sofa_cardio",
        "sofa",
        "susp_inf",
        "sep3",
    }
)

DERIVED_SUMMARY_FEATURES: frozenset[str] = frozenset(
    {
        # Ratios or dose-equivalent summaries; raw components/rates are retained.
        "pafi",
        "safi",
        "norepi_equiv",
        # Windowed or composite clinical indicators/scores; raw components are
        # retained where available.
        "vent_ind",
        "gcs",
        "urine24",
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cns",
        "sofa_renal",
        "qsofa",
        "sirs",
        "news",
        "mews",
    }
)

FEATURE_BLOCKLIST: frozenset[str] = (
    DIRECT_LEAKAGE_FEATURES | FUTURE_DERIVED_FEATURES | DERIVED_SUMMARY_FEATURES
)
