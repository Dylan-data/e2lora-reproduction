#!/usr/bin/env bash
#SBATCH --job-name=e2lora-glue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=05:45:00
#SBATCH --array=0-5%1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted package root}"
method=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" methods.txt)
bash run.sh "${1:?Pass sst2 or qnli}" "$method"
