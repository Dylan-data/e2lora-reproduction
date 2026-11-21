#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=mmqa-check
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Edit this path according to your server's Conda installation.
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate /home/msds/jins0013/.conda/envs/e2lora-nlg

export HF_HOME="$HOME/hf-cache"
export TOKENIZERS_PARALLELISM=false

MODEL_KEY="${1:-llama31_8b}"
LIMIT="${2:-}"

mkdir -p outputs/preflight

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARGS+=(--limit "$LIMIT")
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: $MODEL_KEY"
echo "Start: $(date)"

nvidia-smi

python -u experiments/s/inspect_token_lengths.py \
    --task math \
    --model-key "$MODEL_KEY" \
    "${LIMIT_ARGS[@]}" \
    --candidates 512,1024,2048,4096 \
    --output "outputs/preflight/math_${MODEL_KEY}_${LIMIT:-full}.json"

echo "Finished: $(date)"
