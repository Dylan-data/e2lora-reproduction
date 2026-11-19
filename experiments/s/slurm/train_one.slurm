#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@300
#SBATCH --job-name=e2lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Usage:
# sbatch slurm/train_one.slurm table2 math dora_r4 llama31_8b 42 1 2048 512
# Resume an existing run with:
# sbatch --export=ALL,RESUME=1 slurm/train_one.slurm table2 math dora_r4 llama31_8b 42 1 2048 512

if [[ "$#" -ne 8 ]]; then
  echo "Usage: $0 <suite> <task> <method> <model_key> <seed> <epochs> <max_seq_length> <max_new_tokens>" >&2
  exit 2
fi

SUITE="$1"
TASK="$2"
METHOD="$3"
MODEL_KEY="$4"
SEED="$5"
EPOCHS="$6"
MAX_SEQ_LENGTH="$7"
MAX_NEW_TOKENS="$8"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

# SLURM opens stdout/stderr before the script starts. Create logs/ before sbatch
# submission, e.g. `mkdir -p logs` in the login shell.
mkdir -p outputs

source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate /home/msds/jins0013/.conda/envs/e2lora-nlg

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

OUT_ROOT="${OUT_ROOT:-outputs}"
OUTPUT_DIR="${OUT_ROOT}/${SUITE}/${TASK}/${METHOD}/seed_${SEED}"

ARGS=(
  --suite "$SUITE"
  --task "$TASK"
  --method "$METHOD"
  --model-key "$MODEL_KEY"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --output-dir "$OUTPUT_DIR"
  --per-device-batch-size 1
  --gradient-accumulation-steps 64
  --gradient-examples 64
  --weight-decay 5e-4
  --warmup-ratio 0.03
  --save-steps 5
  --gradient-checkpointing
  --num-workers 0
  --attn-implementation sdpa
  --device cuda
  --do-final-eval
)

if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

echo "Job ID: $SLURM_JOB_ID"
echo "GPU:"
nvidia-smi

echo "Starting: $SUITE $TASK $METHOD $MODEL_KEY seed=$SEED"

printf 'Run directory: %s\n' "$OUTPUT_DIR"
printf 'Effective batch size: %s\n' "$((1 * 64))"
printf 'Checkpoint interval: every %s optimizer updates\n' "20"
printf 'Resume: %s\n' "${RESUME:-0}"

python -u experiments/s/suite_train.py "${ARGS[@]}"

echo "Training and evaluation finished: $(date)"
