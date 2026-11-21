#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=eurosat_train_pilot
#SBATCH --output=outputs/image-classification/logs/%x-%j.out
#SBATCH --error=outputs/image-classification/logs/%x-%j.err
set -eo pipefail
module load anaconda
eval "$(conda shell.bash hook)"
conda activate e2image
cd "$HOME/e2lora-reproduction"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/outputs/image-classification/cache/huggingface"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
method="${1:?Supply exactly one method: lora, e2lora, manual}"
run_dir="outputs/image-classification/eurosat-${method}-pilot-${SLURM_JOB_ID}"
mkdir -p "$run_dir/source"
cp experiments/kathy/train_pilot.py experiments/kathy/train_pilot.sh experiments/kathy/PLAN.md "$run_dir/source/"
cp src/e2lora/model/clip_classification.py src/e2lora/adapters/adaptive.py src/e2lora/adapters/lora_layers.py "$run_dir/source/"
git rev-parse HEAD > "$run_dir/git-commit.txt"
sha256sum "$run_dir"/source/* > "$run_dir/source.sha256"
python -m pip freeze > "$run_dir/environment.txt"
python -m pip check
hostname
nvidia-smi
python experiments/kathy/train_pilot.py --method "$method" --output "$run_dir"
