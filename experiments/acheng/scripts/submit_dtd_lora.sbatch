#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:45:00
#SBATCH --signal=USR1@120

set -eo pipefail

if [[ -z "${ACHENG_RUN_DIR:-}" || -z "${ACHENG_CONFIG:-}" ]]; then
    echo "ACHENG_RUN_DIR and ACHENG_CONFIG must be supplied by submit_dtd_lora.sh" >&2
    exit 2
fi

REPO=/home/msds/weicheng001/e2lora-reproduction
PYTHON=/home/msds/weicheng001/.conda/envs/e2image/bin/python
export PYTHONPATH="$REPO/src"
export HF_HOME="$REPO/experiments/acheng/outputs/cache/huggingface"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$REPO"
hostname
nvidia-smi
"$PYTHON" -m pip freeze > "$ACHENG_RUN_DIR/environment-$SLURM_JOB_ID.txt"
printf 'CUBLAS_WORKSPACE_CONFIG=%s\nHF_HOME=%s\n' \
    "$CUBLAS_WORKSPACE_CONFIG" "$HF_HOME" \
    > "$ACHENG_RUN_DIR/runtime-environment-$SLURM_JOB_ID.txt"
git rev-parse HEAD > "$ACHENG_RUN_DIR/git-commit-$SLURM_JOB_ID.txt"
sha256sum "$ACHENG_CONFIG" > "$ACHENG_RUN_DIR/config-$SLURM_JOB_ID.sha256"
sha256sum \
    experiments/acheng/scripts/train_lora.py \
    src/e2lora/adapters/lora_layers.py \
    src/e2lora/model/clip_classification.py \
    > "$ACHENG_RUN_DIR/source-$SLURM_JOB_ID.sha256"

RESUME_ARGS=()
if [[ "${ACHENG_RESUME:-0}" == "1" ]]; then
    RESUME_ARGS=(--resume)
fi

srun --ntasks=1 --cpus-per-task=4 "$PYTHON" experiments/acheng/scripts/train_lora.py \
    --config "$ACHENG_CONFIG" \
    --run-dir "$ACHENG_RUN_DIR" \
    "${RESUME_ARGS[@]}"
