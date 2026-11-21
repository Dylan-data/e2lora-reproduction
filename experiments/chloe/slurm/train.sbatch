#!/usr/bin/env bash
# Resource flags are supplied from inspected JSON by run_ablations.py.
#SBATCH --job-name=chloe-suite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@120
#SBATCH --output=outputs/chloe/slurm/suite-%j.out
#SBATCH --error=outputs/chloe/slurm/suite-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# Activate the intended environment before sbatch. No embedded home path/token.
nvidia-smi --query-gpu=name,memory.total --format=csv
exec python -u experiments/chloe/run_ablations.py run --batch "${1:?batch.json required}" --index "${2:?index required}"
