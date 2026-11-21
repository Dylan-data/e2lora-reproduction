#!/bin/bash
#SBATCH --job-name=e2lora-torch-check
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/torch-check-%j.out
#SBATCH --error=logs/torch-check-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate e2lora

echo "Node: $(hostname)"
nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA unavailable"

print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

a = torch.randn(1024, 1024, device="cuda")
b = a @ a

print("Matrix test:", b.shape)
print("GPU computation successful!")
PY
