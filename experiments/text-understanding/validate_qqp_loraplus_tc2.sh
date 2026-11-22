#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src"
out="$PWD/outputs/dylan/glue-full-seed42/lr-validation-$SLURM_JOB_ID"
for rate in 0.0005 0.00005; do
    .cache/glue-env/bin/python experiments/dylan/glue_full.py \
        --task qqp --method loraplus --seed 42 --device cuda:0 \
        --validation-run --validation-learning-rate "$rate" --output-root "$out"
done
