#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=e2lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate /home/msds/jins0013/.conda/envs/e2lora-nlg

export HF_HOME="$HOME/hf-cache"
export TOKENIZERS_PARALLELISM=false

# Arguments:
# suite task method model_key seed epochs max_seq_length max_new_tokens

SUITE="${1:?Missing suite}"
TASK="${2:?Missing task}"
METHOD="${3:?Missing method}"
MODEL_KEY="${4:?Missing model key}"
SEED="${5:?Missing seed}"
EPOCHS="${6:?Missing epochs}"
MAX_SEQ="${7:?Missing max sequence length}"
MAX_NEW="${8:?Missing max new tokens}"

if [[ "$SUITE" == "table15" ]]; then
    OUT="outputs/table15/$MODEL_KEY/$METHOD/seed_$SEED"
else
    OUT="outputs/$SUITE/$TASK/$METHOD/seed_$SEED"
fi

mkdir -p "$OUT"

ARGS=(
    --suite "$SUITE"
    --task "$TASK"
    --method "$METHOD"
    --model-key "$MODEL_KEY"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --max-seq-length "$MAX_SEQ"
    --max-new-tokens "$MAX_NEW"
    --per-device-batch-size 1
    --gradient-accumulation-steps 64
    --gradient-examples 64
    --gradient-checkpointing
    --attn-implementation sdpa
    --device cuda
    --do-final-eval
    --output-dir "$OUT"
)

# Sources that are not pinned in the current repository.
case "$TASK" in
    code|piqa|siqa)
        ARGS+=(--allow-unpinned-datasets)
        ;;
esac

# Code-Feedback needs a verified local 100K code-only JSONL file.
if [[ "$TASK" == "code" ]]; then
    : "${CODE_JSONL:?Set CODE_JSONL to the remote JSONL path}"
    ARGS+=(--train-jsonl "$CODE_JSONL")
    ARGS+=(--code-only-source-confirmed)
fi

# Optional resume from the last saved epoch checkpoint.
if [[ "${RESUME:-0}" == "1" ]]; then
    ARGS+=(--resume)
fi

echo "Job ID: $SLURM_JOB_ID"
echo "GPU:"
nvidia-smi

echo "Starting: $SUITE $TASK $METHOD $MODEL_KEY seed=$SEED"
echo "Output: $OUT"

python -u experiments/s/suite_train.py "${ARGS[@]}"

echo "Training and evaluation finished: $(date)"

