#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src"
out="$PWD/outputs/dylan/glue-full-seed42/lr-validation-$SLURM_JOB_ID"
for task in sst2 qnli cola; do
    for method in fft loraplus; do
        .cache/glue-env/bin/python experiments/dylan/glue_full.py \
            --task "$task" --method "$method" --seed 42 --device cuda:0 \
            --validation-run --output-root "$out"
    done
done
