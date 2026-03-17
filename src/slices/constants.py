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
SEQ_LENGTH_HOURS: int = 48  # Model input window
MIN_STAY_HOURS: int = 48
# RICU extraction horizon — longer than SEQ_LENGTH_HOURS to enable
# forward-looking labels (e.g. AKI prediction from hours 48-72).
LABEL_HORIZON_HOURS: int = 72

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
# RICU concepts that must be excluded from model input:
# - los_hosp / los_icu: leak downstream task labels (updated hourly by RICU,
#   directly reveal the length-of-stay answer)
# - dur_var: internal ricu helper variable, not a clinical concept
FEATURE_BLOCKLIST: frozenset = frozenset({"los_hosp", "los_icu", "dur_var"})
