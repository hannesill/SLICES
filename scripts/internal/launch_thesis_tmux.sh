#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="${SESSION_NAME:-slices-thesis}"
WANDB_PROJECT="${WANDB_PROJECT:-slices-thesis}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
REVISION="${REVISION:-thesis-v1}"
REASON="${REASON:-thesis-final canonical-sprint11-baselines}"
PARALLEL_MAIN="${PARALLEL_MAIN:-4}"
PARALLEL_APPENDIX="${PARALLEL_APPENDIX:-4}"
BATCH_SIZE_FAIRNESS="${BATCH_SIZE_FAIRNESS:-64}"
DEVICE_FAIRNESS="${DEVICE_FAIRNESS:-auto}"
INCLUDE_SPRINT12="${INCLUDE_SPRINT12:-1}"
INCLUDE_SPRINT13="${INCLUDE_SPRINT13:-1}"
INCLUDE_SPRINT7P="${INCLUDE_SPRINT7P:-1}"
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

mkdir -p "$LOG_DIR"

main_sprints=(1 1b 1c 2 3 4 5 6 7 8 10 11)
fairness_sprints=(1 2 3 4 5 6 7 8 10 11)
tag_sprints=(1b 1c 2 5 6 7 8)
appendix_sprints=()

if [[ "$INCLUDE_SPRINT13" == "1" ]]; then
  main_sprints+=(13)
  fairness_sprints+=(13)
fi

if [[ "$INCLUDE_SPRINT7P" == "1" ]]; then
  main_sprints+=(7p)
  fairness_sprints+=(7p)
  tag_sprints+=(7p)
fi

if [[ "$INCLUDE_SPRINT12" == "1" ]]; then
  appendix_sprints+=(12)
  fairness_sprints+=(12)
fi

warmup_sprints=("${main_sprints[@]}")
if [[ "$INCLUDE_SPRINT12" == "1" ]]; then
  warmup_sprints+=(12)
fi

quote_cmd() {
  printf "%q " "$@"
}

run_args=(--project "$WANDB_PROJECT" --entity "$WANDB_ENTITY")
export_args=(--project "$WANDB_PROJECT" --entity "$WANDB_ENTITY")
fairness_args=(--project "$WANDB_PROJECT" --entity "$WANDB_ENTITY")
tag_args=(--project "$WANDB_PROJECT" --entity "$WANDB_ENTITY")

run_args+=(--revision "$REVISION" --reason "$REASON")
export_args+=(--revision "$REVISION" --output-dir "$RESULTS_DIR")
if [[ -n "$REVISION" ]]; then
  tag_args+=(--revision "$REVISION")
fi
if [[ -n "$REVISION" ]]; then
  fairness_args+=(--revision "$REVISION")
fi
fairness_args+=(--sprint "${fairness_sprints[@]}" --batch-size "$BATCH_SIZE_FAIRNESS" --device "$DEVICE_FAIRNESS")

warmup_cmd=(uv run python scripts/internal/run_experiments.py warmup --sprint "${warmup_sprints[@]}")
main_cmd=(uv run python scripts/internal/run_experiments.py run --sprint "${main_sprints[@]}" --parallel "$PARALLEL_MAIN" "${run_args[@]}")
tag_cmd=(uv run python scripts/internal/run_experiments.py tag --sprint "${tag_sprints[@]}" "${tag_args[@]}")
fairness_cmd=(uv run python scripts/eval/evaluate_fairness.py "${fairness_args[@]}")
export_cmd=(uv run python scripts/export_results.py "${export_args[@]}")

if ((${#appendix_sprints[@]} > 0)); then
  appendix_cmd=(uv run python scripts/internal/run_experiments.py run --sprint "${appendix_sprints[@]}" --parallel "$PARALLEL_APPENDIX" "${run_args[@]}")
else
  appendix_cmd=()
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
runner_script="$LOG_DIR/thesis-run-${timestamp}.sh"
runner_log="$LOG_DIR/thesis-run-${timestamp}.log"
status_log="$LOG_DIR/thesis-status-${timestamp}.log"

printf -v warmup_line "%q " "${warmup_cmd[@]}"
printf -v main_line "%q " "${main_cmd[@]}"
printf -v tag_line "%q " "${tag_cmd[@]}"
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
echo "Main sprints: ${main_sprints[*]}"
echo "Fairness sprints: ${fairness_sprints[*]}"
echo "Appendix sprints: ${appendix_sprints[*]:-none}"

uv sync --dev
${warmup_line}
${main_line}
${tag_line}
${appendix_block}${fairness_line}
${export_block}echo "Finished: \$(date -Iseconds)"
EOF

chmod +x "$runner_script"

status_cmd=(
  bash
  -lc
  "cd $(printf "%q" "$REPO_ROOT"); while true; do clear; date; echo; uv run python scripts/internal/run_experiments.py status; sleep $(printf "%q" "$STATUS_INTERVAL"); done"
)
printf -v status_line "%q " "${status_cmd[@]}"

tmux new-session -d -s "$SESSION_NAME" -n run "bash $(printf "%q" "$runner_script") 2>&1 | tee $(printf "%q" "$runner_log")"
tmux new-window -t "$SESSION_NAME" -n status "$status_line"
tmux pipe-pane -o -t "$SESSION_NAME:status" "cat >> $(printf "%q" "$status_log")"

echo "Created tmux session: $SESSION_NAME"
echo "Run log: $runner_log"
echo "Status log: $status_log"
echo "Attach with: tmux attach -t $SESSION_NAME"
