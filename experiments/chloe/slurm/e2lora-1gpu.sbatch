#!/bin/bash
#SBATCH --job-name=chloe-e2lora-smoke
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/chloe/slurm/smoke-e2lora8-%j.out
#SBATCH --error=outputs/chloe/slurm/smoke-e2lora8-%j.err

# Submit only after explicit user approval; this script never submits other jobs.
set -euo pipefail
cd /home/msds/yahui003/e2lora-reproduction
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate e2lora
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/home/msds/yahui003/e2lora-storage/hf-home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

if [[ $# -ne 4 || "$1" != --stage || "$3" != --config ]]; then
  printf '%s\n' 'Usage: sbatch e2lora-1gpu.sbatch --stage prepare|smoke --config experiments/chloe/configs/smoke-e2lora-r8.json' >&2
  exit 2
fi
stage_name="$2"
config_path="$4"
if [[ "$stage_name" != prepare && "$stage_name" != smoke ]]; then
  printf '%s\n' 'stage must be prepare or smoke' >&2
  exit 2
fi
python experiments/chloe/prepare_e2lora.py --config "$config_path" --dry-run
python - "$config_path" "$stage_name" <<'PYCODE'
import sys
sys.path.insert(0, 'experiments/chloe')
from common import read_json
from adaptive_smoke import prepared_artifact
config = read_json(sys.argv[1])
if config['method'] != 'e2lora' or config['training']['max_steps'] != 20:
    raise SystemExit('This acceptance script requires the independent 20-update E2LoRA Smoke config.')
if sys.argv[2] == 'smoke':
    artifact = prepared_artifact(config)
    print('Validated prepared plan:', len(artifact['plan']), 'groups;', artifact['parameters']['total_trainable_parameters'], 'trainable parameters')
PYCODE
# GRES is registered as gpu:nvidia on this cluster; verify the assigned model
# instead of inventing an unregistered gpu:a40 type or node constraint.
gpu_uuid="$(python - <<'PY'
import os, sys
sys.path.insert(0, 'experiments/chloe')
import torch
from common import write_json
from train import require_gpu_allocation

require_gpu_allocation()
properties = torch.cuda.get_device_properties(0)
if 'A40' not in properties.name:
    raise SystemExit(f'Expected NVIDIA A40 for this Smoke, received {properties.name}; no training started.')
record = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'gpu_name': properties.name,
          'gpu_uuid': str(properties.uuid), 'gpu_total_memory_bytes': properties.total_memory,
          'bf16_supported': torch.cuda.is_bf16_supported(), 'torch_version': torch.__version__,
          'cuda_version': torch.version.cuda}
write_json(f"outputs/chloe/slurm/device-{os.environ['SLURM_JOB_ID']}.json", record)
print(record['gpu_uuid'])
PY
)"
nvidia-smi --id="GPU-${gpu_uuid#GPU-}"

# Sample only this allocation's GPU. Stop the logger on success or failure while
# preserving the training/evaluation exit code for Slurm accounting.
gpu_monitor_pid=''
cleanup_gpu_monitor() {
  local task_exit_code=$?
  trap - EXIT
  if [[ -n "$gpu_monitor_pid" ]]; then
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
  fi
  exit "$task_exit_code"
}
trap cleanup_gpu_monitor EXIT
nvidia-smi --id="GPU-${gpu_uuid#GPU-}" \
  --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv --loop=10 > "outputs/chloe/slurm/gpu-${SLURM_JOB_ID}.csv" &
gpu_monitor_pid=$!

if [[ "$stage_name" == prepare ]]; then
  python experiments/chloe/prepare_e2lora.py --config "$config_path"
  printf '%s\n' 'E2LoRA preparation finished: sampling, LSS/HRA, parameter count and gradient probe; zero optimizer updates.'
  exit 0
fi
python experiments/chloe/train.py --config "$config_path" --stop-after-step 10
checkpoint_path="$(python - "$config_path" <<'PYCODE'
import json, sys
from pathlib import Path
print(Path(json.load(open(sys.argv[1]))['output_dir']) / 'checkpoint.pt')
PYCODE
)"
python experiments/chloe/train.py --config "$config_path" --resume "$checkpoint_path"
python experiments/chloe/evaluate.py --config "$config_path"
printf '%s\n' 'E2LoRA Smoke finished: 20 total optimizer updates, restored shared plan at step 10, 32 GSM8K cases scored.'
