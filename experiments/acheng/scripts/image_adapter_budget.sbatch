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
: "${ACHENG_RUN_DIR:?Missing run directory}"
: "${ACHENG_CONFIG:?Missing config}"
: "${ACHENG_PHASE:?Missing phase}"
: "${ACHENG_RESUME:?Missing resume flag}"
REPO=/home/msds/weicheng001/e2lora-reproduction
PYTHON=/home/msds/weicheng001/.conda/envs/e2image/bin/python
export PYTHONPATH="$REPO/src:$REPO/experiments/acheng/scripts"
export HF_HOME="$REPO/experiments/acheng/outputs/cache/huggingface"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
hostname
nvidia-smi
"$PYTHON" -m pip freeze > "$ACHENG_RUN_DIR/environment-$SLURM_JOB_ID.txt"
sha256sum "$ACHENG_CONFIG" \
    experiments/acheng/scripts/train_image_adapter.py \
    experiments/acheng/scripts/audit_sharelora_batch.py \
    src/e2lora/adapters/lora_layers.py \
    src/e2lora/model/clip_classification.py > "$ACHENG_RUN_DIR/source-$SLURM_JOB_ID.sha256"
if [[ "$ACHENG_PHASE" == pilot ]]; then
    [[ "$ACHENG_RESUME" == 0 ]] || { echo "Pilot cannot resume" >&2; exit 2; }
    SCRIPT=experiments/acheng/scripts/audit_sharelora_batch.py
elif [[ "$ACHENG_PHASE" == formal ]]; then
    SCRIPT=experiments/acheng/scripts/train_image_adapter.py
else
    echo "Unknown phase: $ACHENG_PHASE" >&2
    exit 2
fi
RESUME=()
if [[ "$ACHENG_RESUME" == 1 ]]; then RESUME=(--resume); fi
srun --ntasks=1 --cpus-per-task=4 "$PYTHON" "$SCRIPT" \
    --config "$ACHENG_CONFIG" --run-dir "$ACHENG_RUN_DIR" "${RESUME[@]}"
