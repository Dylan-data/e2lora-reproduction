#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=eurosat_zero_full
#SBATCH --output=outputs/image-classification/logs/%x-%j.out
#SBATCH --error=outputs/image-classification/logs/%x-%j.err

set -eo pipefail
module load anaconda
eval "$(conda shell.bash hook)"
conda activate e2image
cd "$HOME/e2lora-reproduction"

export PYTHONUNBUFFERED=1
export HF_HOME="$PWD/outputs/image-classification/cache/huggingface"

run_dir="outputs/image-classification/eurosat-zero-shot-full-${SLURM_JOB_ID}"
mkdir -p "$run_dir"

hostname
nvidia-smi
python -m pip freeze > "$run_dir/environment-before.txt"
python -m pip install --only-binary=:all: --no-deps "pyarrow==21.0.0" > "$run_dir/dependency-repair.log" 2>&1
python -m pip check > "$run_dir/pip-check.txt"
python -c 'import numpy, pyarrow, datasets, torch; print("numpy", numpy.__version__, "pyarrow", pyarrow.__version__, "datasets", datasets.__version__, "torch", torch.__version__); assert torch.cuda.is_available()' > "$run_dir/import-check.txt"
python -m pip freeze > "$run_dir/environment.txt"
sha256sum experiments/kathy/zero_shot.py > "$run_dir/script.sha256"

python experiments/kathy/zero_shot.py --output "$run_dir" --limit 0
