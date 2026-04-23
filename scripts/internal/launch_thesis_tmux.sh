#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="${SESSION_NAME:-slices-thesis}"
WANDB_PROJECT="${WANDB_PROJECT:-slices-thesis}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
REVISION="${REVISION:-thesis-v1}"
REASON="${REASON:-thesis-final class-based rerun corpus}"
LAUNCH_COMMIT="${LAUNCH_COMMIT:-}"
SKIP_LAUNCH_GIT_CHECK="${SKIP_LAUNCH_GIT_CHECK:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
EXPECTED_FEATURES="${EXPECTED_FEATURES:-84}"
VALIDATE_PROCESSED_ARTIFACTS="${VALIDATE_PROCESSED_ARTIFACTS:-1}"
PURGE_RUNTIME_CACHES="${PURGE_RUNTIME_CACHES:-1}"
PARALLEL_MAIN="${PARALLEL_MAIN:-4}"
PARALLEL_APPENDIX="${PARALLEL_APPENDIX:-4}"
BATCH_SIZE_FAIRNESS="${BATCH_SIZE_FAIRNESS:-64}"
DEVICE_FAIRNESS="${DEVICE_FAIRNESS:-auto}"
INCLUDE_SMART_REFERENCE="${INCLUDE_SMART_REFERENCE:-1}"
INCLUDE_TS2VEC_EXTENSION="${INCLUDE_TS2VEC_EXTENSION:-1}"
INCLUDE_CAPACITY_STUDY="${INCLUDE_CAPACITY_STUDY:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"
STATUS_INTERVAL="${STATUS_INTERVAL:-60}"
RESULTS_DIR="${RESULTS_DIR:-results/${WANDB_PROJECT}_${REVISION}}"
LOG_DIR="${LOG_DIR:-logs/runner}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists." >&2
  exit 1
fi

if [[ -z "$WANDB_ENTITY" ]]; then
  echo "WANDB_ENTITY must be set for thesis runs." >&2
  exit 1
fi

if [[ "$SKIP_LAUNCH_GIT_CHECK" == "1" ]]; then
  LAUNCH_COMMIT="${LAUNCH_COMMIT:-unchecked}"
else
  if [[ -z "$LAUNCH_COMMIT" ]]; then
    LAUNCH_COMMIT="$(git -C "$REPO_ROOT" rev-parse --verify HEAD)"
  fi

  if ! git -C "$REPO_ROOT" cat-file -e "$LAUNCH_COMMIT^{commit}" 2>/dev/null; then
    echo "LAUNCH_COMMIT is not available in this checkout: $LAUNCH_COMMIT" >&2
    exit 1
  fi

  CURRENT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --verify HEAD)"
  if [[ "$CURRENT_COMMIT" != "$LAUNCH_COMMIT" ]]; then
    echo "Refusing to launch from a different commit." >&2
    echo "  current: $CURRENT_COMMIT" >&2
    echo "  expected: $LAUNCH_COMMIT" >&2
    exit 1
  fi

  if [[ "$ALLOW_DIRTY" != "1" ]]; then
    if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --cached --quiet; then
      echo "Refusing to launch with tracked uncommitted changes." >&2
      echo "Commit the reviewed state, or set ALLOW_DIRTY=1 for an explicit local dry run." >&2
      exit 1
    fi
  fi
fi

validate_processed_artifacts() {
  uv run python - "$EXPECTED_FEATURES" <<'PY'
from pathlib import Path
import sys

import polars as pl
import yaml

expected_features = int(sys.argv[1])
base = Path("data/processed")
datasets = ("miiv", "eicu", "combined")
required_files = (
    "metadata.yaml",
    "static.parquet",
    "timeseries.parquet",
    "labels.parquet",
    "splits.yaml",
    "normalization_stats.yaml",
)
errors = []

for dataset in datasets:
    path = base / dataset
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        errors.append(f"{dataset}: missing {', '.join(missing)}")
        continue

    with open(path / "metadata.yaml") as f:
        metadata = yaml.safe_load(f) or {}

    feature_names = metadata.get("feature_names") or []
    n_features = metadata.get("n_features", len(feature_names))
    if n_features != expected_features:
        errors.append(f"{dataset}: expected {expected_features} features, found {n_features}")

    n_stays = metadata.get("n_stays")
    for filename in ("static.parquet", "timeseries.parquet", "labels.parquet"):
        height = pl.scan_parquet(path / filename).select(pl.len()).collect().item()
        if n_stays is not None and height != n_stays:
            errors.append(f"{dataset}: {filename} has {height} rows, metadata has {n_stays}")

    if dataset in {"miiv", "eicu"} and not metadata.get("zero_observation_stays_excluded"):
        errors.append(f"{dataset}: metadata does not confirm zero-observation exclusion")

    print(f"{dataset}: {n_stays:,} stays, {n_features} features")

if errors:
    print("\nProcessed artifact readiness check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

purge_runtime_caches() {
  if [[ "$PURGE_RUNTIME_CACHES" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "$REPO_ROOT/data/processed" ]]; then
    return 0
  fi

  find "$REPO_ROOT/data/processed" -maxdepth 2 -type d -name ".tensor_cache" -prune -exec rm -rf {} +
  find "$REPO_ROOT/data/processed" -maxdepth 2 -type f -name "normalization_stats_*.yaml" -delete
}

echo "Launch commit: $LAUNCH_COMMIT"
if [[ "$VALIDATE_PROCESSED_ARTIFACTS" == "1" ]]; then
  echo "Validating processed artifacts..."
  (
    cd "$REPO_ROOT"
    validate_processed_artifacts
  )
  echo "Processed artifacts are ready."
fi
purge_runtime_caches

mkdir -p "$LOG_DIR"

main_classes=(
  core_ssl_benchmark
  label_efficiency
  cross_dataset_transfer
  hp_robustness
  classical_baselines
)
fairness_classes=("${main_classes[@]}")
appendix_classes=()

if [[ "$INCLUDE_TS2VEC_EXTENSION" == "1" ]]; then
  main_classes+=(ts2vec_extension)
  fairness_classes+=(ts2vec_extension)
fi

if [[ "$INCLUDE_CAPACITY_STUDY" == "1" ]]; then
  main_classes+=(capacity_study)
  fairness_classes+=(capacity_study)
fi

if [[ "$INCLUDE_SMART_REFERENCE" == "1" ]]; then
  appendix_classes+=(smart_external_reference)
  fairness_classes+=(smart_external_reference)
fi

warmup_classes=("${main_classes[@]}")
if [[ "$INCLUDE_SMART_REFERENCE" == "1" ]]; then
  warmup_classes+=(smart_external_reference)
fi

quote_cmd() {
  printf "%q " "$@"
}

run_args=(--project "$WANDB_PROJECT" --entity "$WANDB_ENTITY")
export_args=(--project "$WANDB_PROJECT" --entity "$WANDB_ENTITY")
fairness_args=(--project "$WANDB_PROJECT" --entity "$WANDB_ENTITY")

run_args+=(--revision "$REVISION" --reason "$REASON" --launch-commit "$LAUNCH_COMMIT")
export_args+=(--revision "$REVISION" --output-dir "$RESULTS_DIR")
if [[ -n "$REVISION" ]]; then
  fairness_args+=(--revision "$REVISION")
fi
export_args+=(--experiment-class "${fairness_classes[@]}")
fairness_args+=(--experiment-class "${fairness_classes[@]}" --batch-size "$BATCH_SIZE_FAIRNESS" --device "$DEVICE_FAIRNESS")

warmup_cmd=(uv run python scripts/internal/run_experiments.py warmup --experiment-class "${warmup_classes[@]}")
main_cmd=(uv run python scripts/internal/run_experiments.py run --experiment-class "${main_classes[@]}" --parallel "$PARALLEL_MAIN" "${run_args[@]}")
fairness_cmd=(uv run python scripts/eval/evaluate_fairness.py "${fairness_args[@]}")
export_cmd=(uv run python scripts/export_results.py "${export_args[@]}")

if ((${#appendix_classes[@]} > 0)); then
  appendix_cmd=(uv run python scripts/internal/run_experiments.py run --experiment-class "${appendix_classes[@]}" --parallel "$PARALLEL_APPENDIX" "${run_args[@]}")
else
  appendix_cmd=()
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
runner_script="$LOG_DIR/thesis-run-${timestamp}.sh"
runner_log="$LOG_DIR/thesis-run-${timestamp}.log"
status_log="$LOG_DIR/thesis-status-${timestamp}.log"

printf -v warmup_line "%q " "${warmup_cmd[@]}"
printf -v main_line "%q " "${main_cmd[@]}"
printf -v fairness_line "%q " "${fairness_cmd[@]}"
printf -v export_line "%q " "${export_cmd[@]}"

appendix_block=""
if ((${#appendix_cmd[@]} > 0)); then
  printf -v appendix_line "%q " "${appendix_cmd[@]}"
  appendix_block="${appendix_line}"$'\n'
fi

export_block=""
if [[ "$RUN_EXPORT" == "1" ]]; then
  export_block="${export_line}"$'\n'
fi

cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf "%q" "$REPO_ROOT")

echo "Start: \$(date -Iseconds)"
echo "W&B project: $(printf "%q" "$WANDB_PROJECT")"
echo "W&B entity: $(printf "%q" "$WANDB_ENTITY")"
echo "Revision: $(printf "%q" "$REVISION")"
echo "Launch commit: $(printf "%q" "$LAUNCH_COMMIT")"
echo "Main classes: ${main_classes[*]}"
echo "Fairness classes: ${fairness_classes[*]}"
echo "Appendix classes: ${appendix_classes[*]:-none}"

if [[ "$(printf "%q" "$SKIP_LAUNCH_GIT_CHECK")" != "1" ]]; then
  current_commit="\$(git rev-parse --verify HEAD)"
  if [[ "\$current_commit" != "$(printf "%q" "$LAUNCH_COMMIT")" ]]; then
    echo "Launch commit mismatch: current=\$current_commit expected=$(printf "%q" "$LAUNCH_COMMIT")" >&2
    exit 1
  fi
fi

uv sync --dev --locked
${warmup_line}
${main_line}
${appendix_block}${fairness_line}
${export_block}echo "Finished: \$(date -Iseconds)"
EOF

chmod +x "$runner_script"

status_loop="cd $(printf "%q" "$REPO_ROOT"); while true; do clear; date; echo; "
status_loop+="uv run python scripts/internal/run_experiments.py status "
status_loop+="--revision $(printf "%q" "$REVISION"); "
status_loop+="sleep $(printf "%q" "$STATUS_INTERVAL"); done"
status_cmd=(bash -lc "$status_loop")
printf -v status_line "%q " "${status_cmd[@]}"

tmux new-session -d -s "$SESSION_NAME" -n run "bash $(printf "%q" "$runner_script") 2>&1 | tee $(printf "%q" "$runner_log")"
tmux new-window -t "$SESSION_NAME" -n status "$status_line"
tmux pipe-pane -o -t "$SESSION_NAME:status" "cat >> $(printf "%q" "$status_log")"

echo "Created tmux session: $SESSION_NAME"
echo "Run log: $runner_log"
echo "Status log: $status_log"
echo "Attach with: tmux attach -t $SESSION_NAME"
