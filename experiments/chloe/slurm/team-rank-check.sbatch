#!/usr/bin/env bash
# Selected after live scontrol/sacctmgr queries on 2026-11-20.
# normal: <=10 CPU, 1 GPU, 30G memory per user, 6h, 2 jobs/submissions.
#SBATCH --job-name=chloe-team-rank-check
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/chloe/slurm/team-rank-check-%j.out
#SBATCH --error=outputs/chloe/slurm/team-rank-check-%j.err
set -euo pipefail
cd /home/msds/yahui003/e2lora-reproduction
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate e2lora
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
python -c 'import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory, flush=True); assert torch.cuda.get_device_properties(0).total_memory >= 40*1024**3, "Use a separate plan for a smaller GPU"'
python experiments/chloe/team_rank_check.py --config experiments/chloe/configs/team-partial-rank-short.json
