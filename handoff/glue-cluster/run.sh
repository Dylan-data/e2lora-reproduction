#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false
python_bin="${PYTHON_BIN:-python3}"
task="${1:?Usage: bash run.sh TASK METHOD}"
method="${2:?Usage: bash run.sh TASK METHOD}"
case "$task" in sst2|qnli) ;; *) exit 2 ;; esac
grep -Fxq "$method" methods.txt || { echo "Unknown method: $method" >&2; exit 2; }
out="$PWD/outputs/$task-transfer-seed42"
mkdir -p "$out/checkpoints" "$out/results"
exec 9>"$out/checkpoints/$method.lock"
flock -n 9 || { echo "$method already running" >&2; exit 1; }
if compgen -G "$out/results/*/glue-$task-$method-original-*/result.json" >/dev/null; then
    echo "$method already completed"
    exit 0
fi
"$python_bin" experiments/dylan/glue_full.py --task "$task" --method "$method" \
    --seed 42 --device "${DEVICE:-cuda:0}" --output-root "$out/results" \
    --checkpoint "$out/checkpoints/$method.pt"
