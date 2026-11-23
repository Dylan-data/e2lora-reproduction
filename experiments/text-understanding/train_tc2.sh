#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --job-name=e2lora-glue
#SBATCH --output=outputs/text-understanding/slurm-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src
exec .cache/glue-env/bin/python experiments/dylan/glue_full.py --device cuda:0 "$@"
