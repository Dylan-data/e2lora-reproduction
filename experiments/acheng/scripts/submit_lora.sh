#!/bin/bash
set -eo pipefail

REPO=/home/msds/weicheng001/e2lora-reproduction
CONFIG="$REPO/experiments/acheng/configs/eurosat_lora_seed2026.json"
SCRIPT="$REPO/experiments/acheng/scripts/train_lora.sbatch"
RESUME_DIR=
cd "$REPO"

case "$(realpath "$CONFIG")" in
    "$REPO/experiments/acheng/configs/"*) ;;
    *) echo "Config must be under experiments/acheng/configs/." >&2; exit 2 ;;
esac
case "$(realpath "$SCRIPT")" in
    "$REPO/experiments/acheng/scripts/"*) ;;
    *) echo "Slurm script must be under experiments/acheng/scripts/." >&2; exit 2 ;;
esac
if rg -n 'experiments/kathy/' "$CONFIG" "$SCRIPT"; then
    echo "Active configuration or Slurm script still references experiments/kathy/." >&2
    exit 2
fi

if [[ "${1:-}" == "--resume" ]]; then
    if [[ -z "${2:-}" || $# -ne 2 ]]; then
        echo "Usage: $0 [--resume <existing-run-directory>]" >&2
        exit 2
    fi
    RESUME_DIR="$2"
fi

if squeue -h -u "$USER" -o "%j %T" | awk '$1 ~ /^acheng_/ { found=1 } END { exit !found }'; then
    echo "An acheng experiment is already pending or running; refusing to submit another." >&2
    exit 3
fi

if [[ -n "$RESUME_DIR" ]]; then
    RUN_DIR=$(realpath "$RESUME_DIR")
    case "$RUN_DIR/" in
        "$REPO/experiments/acheng/outputs/"*) ;;
        *) echo "Resume directory must be inside experiments/acheng/outputs/." >&2; exit 2 ;;
    esac
    if [[ -e "$RUN_DIR/result.json" ]]; then
        echo "A completed result already exists; refusing to overwrite $RUN_DIR/result.json." >&2
        exit 2
    fi
    if [[ ! -f "$RUN_DIR/checkpoint_latest.pt" || ! -f "$RUN_DIR/config.json" ]]; then
        echo "Resume requires checkpoint_latest.pt and the immutable config.json." >&2
        exit 2
    fi
    if ! cmp -s "$CONFIG" "$RUN_DIR/config.json"; then
        echo "Resume config differs from the existing run config." >&2
        exit 2
    fi
    export ACHENG_RESUME=1
else
    RUN_ID="$(date +%Y%m%dT%H%M%S%z)"
    RUN_DIR="$REPO/experiments/acheng/outputs/EuroSAT/LoRA-r8/2026/$RUN_ID"
    mkdir -p "$(dirname "$RUN_DIR")"
    mkdir "$RUN_DIR"
    cp "$CONFIG" "$RUN_DIR/config.json"
    export ACHENG_RESUME=0
fi

export ACHENG_RUN_DIR="$RUN_DIR"
export ACHENG_CONFIG="$RUN_DIR/config.json"

SBATCH_SEGMENT_ARGS=()
if [[ "$ACHENG_RESUME" == "1" ]]; then
    SBATCH_SEGMENT_ARGS+=(--time=00:03:00)
fi

case "$RUN_DIR/" in
    "$REPO/experiments/acheng/outputs/"*) ;;
    *) echo "Run directory must be inside experiments/acheng/outputs/." >&2; exit 2 ;;
esac
LOG_DIR=$(dirname "$RUN_DIR/slurm-%j.out")
for path in "$RUN_DIR" "$LOG_DIR" "$(dirname "$RUN_DIR/result.json")" \
    "$(dirname "$RUN_DIR/predictions.npz")" "$(dirname "$RUN_DIR/checkpoint_latest.pt")"; do
    resolved=$(realpath "$path")
    case "$resolved/" in
        "$REPO/experiments/acheng/outputs/"*) ;;
        *) echo "Run/log/result/checkpoint path escapes acheng outputs: $resolved" >&2; exit 2 ;;
    esac
done
if [[ ! -d "$RUN_DIR" || ! -f "$ACHENG_CONFIG" ]]; then
    echo "Run directory and immutable config must exist before sbatch." >&2
    exit 2
fi

JOB_ID=$(sbatch --parsable \
    "${SBATCH_SEGMENT_ARGS[@]}" \
    --job-name=acheng_eu_lora2026 \
    --output="$RUN_DIR/slurm-%j.out" \
    --error="$RUN_DIR/slurm-%j.err" \
    --export=ALL,ACHENG_RUN_DIR,ACHENG_CONFIG,ACHENG_RESUME \
    "$SCRIPT")
printf '%s\n' "$JOB_ID" > "$RUN_DIR/job-$JOB_ID.txt"
printf 'Submitted job %s\nRun directory: %s\n' "$JOB_ID" "$RUN_DIR"
