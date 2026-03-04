#!/usr/bin/env bash
# =============================================================================
# Sprint 1: Sanity Check
# MIMIC-IV | All 4 paradigms | mortality_24h | seed=42
#
# Runs 3 SSL pretrain jobs in parallel + 1 supervised baseline.
# After each pretrain finishes, automatically launches finetuning.
# =============================================================================
set -euo pipefail

SPRINT=1
DATASET=miiv
SEED=42
TASK=mortality_24h
LOG_DIR="logs/sprint1"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "SLICES Sprint 1: Sanity Check"
echo "  Dataset:  $DATASET"
echo "  Task:     $TASK"
echo "  Seed:     $SEED"
echo "  Paradigms: MAE, JEPA, Contrastive, Supervised"
echo "============================================================"

# Use predictable output dirs (override Hydra's timestamped dirs)
MAE_DIR="outputs/sprint1/pretrain_mae"
JEPA_DIR="outputs/sprint1/pretrain_jepa"
CONTRASTIVE_DIR="outputs/sprint1/pretrain_contrastive"
SUPERVISED_DIR="outputs/sprint1/supervised"
FT_MAE_DIR="outputs/sprint1/finetune_mae_${TASK}"
FT_JEPA_DIR="outputs/sprint1/finetune_jepa_${TASK}"
FT_CONTRASTIVE_DIR="outputs/sprint1/finetune_contrastive_${TASK}"

# -----------------------------------------------------------------------------
# Helper: run pretrain then finetune
# -----------------------------------------------------------------------------
pretrain_and_finetune() {
    local ssl_name="$1"
    local pretrain_dir="$2"
    local finetune_dir="$3"
    local log_prefix="$LOG_DIR/${ssl_name}"

    echo "[$(date '+%H:%M:%S')] Starting $ssl_name pretraining..."

    uv run python scripts/training/pretrain.py \
        dataset=$DATASET \
        ssl=$ssl_name \
        seed=$SEED \
        sprint=$SPRINT \
        hydra.run.dir=$pretrain_dir \
        2>&1 | tee "${log_prefix}_pretrain.log"

    echo "[$(date '+%H:%M:%S')] $ssl_name pretraining complete."

    # Check encoder was saved
    if [ ! -f "${pretrain_dir}/encoder.pt" ]; then
        echo "ERROR: ${pretrain_dir}/encoder.pt not found!"
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] Starting $ssl_name finetuning on $TASK..."

    uv run python scripts/training/finetune.py \
        dataset=$DATASET \
        checkpoint=${pretrain_dir}/encoder.pt \
        tasks=$TASK \
        seed=$SEED \
        sprint=$SPRINT \
        hydra.run.dir=$finetune_dir \
        2>&1 | tee "${log_prefix}_finetune.log"

    echo "[$(date '+%H:%M:%S')] $ssl_name finetuning complete."
}

# -----------------------------------------------------------------------------
# Launch all 4 runs in parallel
# -----------------------------------------------------------------------------

# MAE: pretrain -> finetune
pretrain_and_finetune mae "$MAE_DIR" "$FT_MAE_DIR" &
PID_MAE=$!

# JEPA: pretrain -> finetune
pretrain_and_finetune jepa "$JEPA_DIR" "$FT_JEPA_DIR" &
PID_JEPA=$!

# Contrastive: pretrain -> finetune
pretrain_and_finetune contrastive "$CONTRASTIVE_DIR" "$FT_CONTRASTIVE_DIR" &
PID_CONTRASTIVE=$!

# Supervised: single run (no pretrain step)
(
    echo "[$(date '+%H:%M:%S')] Starting supervised baseline..."

    uv run python scripts/training/supervised.py \
        dataset=$DATASET \
        tasks=$TASK \
        seed=$SEED \
        sprint=$SPRINT \
        hydra.run.dir=$SUPERVISED_DIR \
        2>&1 | tee "${LOG_DIR}/supervised.log"

    echo "[$(date '+%H:%M:%S')] Supervised baseline complete."
) &
PID_SUPERVISED=$!

echo ""
echo "All 4 jobs launched in parallel:"
echo "  MAE:          PID=$PID_MAE"
echo "  JEPA:         PID=$PID_JEPA"
echo "  Contrastive:  PID=$PID_CONTRASTIVE"
echo "  Supervised:   PID=$PID_SUPERVISED"
echo ""
echo "Logs: $LOG_DIR/"
echo "  mae_pretrain.log / mae_finetune.log"
echo "  jepa_pretrain.log / jepa_finetune.log"
echo "  contrastive_pretrain.log / contrastive_finetune.log"
echo "  supervised.log"
echo ""
echo "Waiting for all jobs to finish..."

# Wait and report results
FAILED=0

wait $PID_MAE || { echo "FAILED: MAE pipeline"; FAILED=$((FAILED + 1)); }
wait $PID_JEPA || { echo "FAILED: JEPA pipeline"; FAILED=$((FAILED + 1)); }
wait $PID_CONTRASTIVE || { echo "FAILED: Contrastive pipeline"; FAILED=$((FAILED + 1)); }
wait $PID_SUPERVISED || { echo "FAILED: Supervised baseline"; FAILED=$((FAILED + 1)); }

echo ""
echo "============================================================"
if [ $FAILED -eq 0 ]; then
    echo "Sprint 1 COMPLETE — all 4 paradigms finished successfully."
else
    echo "Sprint 1 FINISHED WITH $FAILED FAILURE(S). Check logs."
fi
echo "============================================================"
echo ""
echo "Output directories:"
echo "  MAE pretrain:          $MAE_DIR"
echo "  MAE finetune:          $FT_MAE_DIR"
echo "  JEPA pretrain:         $JEPA_DIR"
echo "  JEPA finetune:         $FT_JEPA_DIR"
echo "  Contrastive pretrain:  $CONTRASTIVE_DIR"
echo "  Contrastive finetune:  $FT_CONTRASTIVE_DIR"
echo "  Supervised:            $SUPERVISED_DIR"
echo ""
echo "Check W&B for training curves: https://wandb.ai/slices"

exit $FAILED
