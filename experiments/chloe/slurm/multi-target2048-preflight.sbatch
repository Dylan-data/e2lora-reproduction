#!/bin/bash
#SBATCH --job-name=chloe-gradient-multi
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/chloe/slurm/multi-target2048-%j.out
#SBATCH --error=outputs/chloe/slurm/multi-target2048-%j.err

# Prepared only: CLI rejects full128 and more than two samples.
set -euo pipefail
cd /home/msds/yahui003/e2lora-reproduction
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate e2lora
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/home/msds/yahui003/e2lora-storage/hf-home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
python experiments/chloe/multi_target_preflight.py "$@" --execute
