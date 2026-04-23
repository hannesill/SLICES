#!/usr/bin/env bash
# =============================================================================
# Auto-shutdown: powers off the VM after the configured idle threshold.
# Install as a cron job on the GCP VM (see bottom of script).
# =============================================================================
set -euo pipefail

IDLE_THRESHOLD_MIN=59
STAMP_FILE="/tmp/slices_last_training_activity"

# Check if any final-run process or Claude Code session is running
if pgrep -f "scripts/(training/(pretrain|finetune|supervised|xgboost_baseline)|internal/run_experiments|preprocessing/prepare_dataset|eval/evaluate_fairness|export_results)\.py" > /dev/null 2>&1 \
   || pgrep -f "claude" > /dev/null 2>&1; then
    # Activity detected — update timestamp and exit
    date +%s > "$STAMP_FILE"
    exit 0
fi

# No training running. Check when the last one was seen.
if [ ! -f "$STAMP_FILE" ]; then
    # No stamp file = script just installed, start the clock now
    date +%s > "$STAMP_FILE"
    exit 0
fi

LAST_ACTIVE=$(cat "$STAMP_FILE")
NOW=$(date +%s)
IDLE_SEC=$(( NOW - LAST_ACTIVE ))
IDLE_MIN=$(( IDLE_SEC / 60 ))

if [ "$IDLE_MIN" -ge "$IDLE_THRESHOLD_MIN" ]; then
    echo "$(date): No training activity for ${IDLE_MIN}m. Shutting down."
    logger "slices-auto-shutdown: idle ${IDLE_MIN}m, shutting down"
    sudo shutdown -h now
else
    echo "$(date): Idle ${IDLE_MIN}m / ${IDLE_THRESHOLD_MIN}m threshold. Staying up."
fi
