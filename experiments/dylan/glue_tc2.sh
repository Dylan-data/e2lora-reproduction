#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --job-name=e2lora-glue
#SBATCH --output=outputs/dylan/glue-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src
python_bin="${GLUE_PYTHON:-$HOME/e2lora-reproduction/.cache/glue-env/bin/python}"
exec "$python_bin" experiments/dylan/train_glue.py --device cuda:0 "$@"
