#!/usr/bin/env bash
#SBATCH --job-name=chloe-he-fig5
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=outputs/chloe/slurm/humaneval-%A_%a.out
#SBATCH --error=outputs/chloe/slurm/humaneval-%A_%a.err
set -euo pipefail

: "${CHLOE_CODE_JSONL:?Set path to the verified code-only Code-Feedback JSONL}"
: "${CHLOE_CODE_ONLY_CONFIRMED:?Set to 1 only after verifying source provenance}"
: "${CHLOE_ALLOW_UNPINNED_HUMANEVAL:?Set to 1 after reviewing the unpinned evaluation revision}"
[[ "$CHLOE_CODE_ONLY_CONFIRMED" == 1 ]]
[[ "$CHLOE_ALLOW_UNPINNED_HUMANEVAL" == 1 ]]

PYTHON_BIN="${CHLOE_PYTHON:-python}"
exec "$PYTHON_BIN" experiments/chloe/humaneval_ablations.py run   --index "${SLURM_ARRAY_TASK_ID:?Submit as a SLURM array job}"   --train-jsonl "$CHLOE_CODE_JSONL"   --confirm-code-only-source   --allow-unpinned-humaneval
