#!/bin/bash
#SBATCH --job-name=chloe-ablation
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/chloe/slurm/formal-%A_%a.out
#SBATCH --error=outputs/chloe/slurm/formal-%A_%a.err

# TEMPLATE ONLY. Confirm time limit, array range/concurrency, GPU quota and
# long-sampling runtime before submitting. No array/time defaults are assumed.
set -euo pipefail
cd /home/msds/yahui003/e2lora-reproduction
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate e2lora
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/home/msds/yahui003/e2lora-storage/hf-home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
if [[ $# -ne 2 || "$1" != --batch ]]; then
  printf '%s\n' 'Usage after authorization: formal-array.sbatch --batch <compiled batch.json>' >&2
  exit 2
fi
python experiments/chloe/run_array.py --batch "$2" --execute
