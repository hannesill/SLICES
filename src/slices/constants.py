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
SEQ_LENGTH_HOURS: int = 48
MIN_STAY_HOURS: int = 48

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
