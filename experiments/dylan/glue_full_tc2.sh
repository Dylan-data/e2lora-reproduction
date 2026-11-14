#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src"
out="$PWD/outputs/dylan/glue-full-seed42"
mkdir -p "$out/checkpoints"
for run in "$@"; do
    task=${run%%:*}
    method=${run#*:}
    checkpoint="$out/checkpoints/$task-$method.pt"
    if compgen -G "$out/results/text-understanding/glue-$task-$method-original-*/result.json" >/dev/null; then
        continue
    fi
    .cache/glue-env/bin/python experiments/dylan/glue_full.py \
        --task "$task" --method "$method" --seed 42 --device cuda:0 \
        --output-root "$out/results" --checkpoint "$checkpoint"
    rm -f -- "$checkpoint"
done
