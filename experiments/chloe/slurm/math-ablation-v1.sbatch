#!/usr/bin/env bash
#SBATCH --job-name=chloe-math-ablation
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=outputs/chloe/slurm/math-ablation-%j.out
#SBATCH --error=outputs/chloe/slurm/math-ablation-%j.err
set -euo pipefail
cd /home/msds/yahui003/e2lora-reproduction
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate e2lora
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
method="${1:?method lora or e2lora required}"
rank="${2:?base rank required}"
seed="${3:?seed required}"
config=experiments/chloe/configs/math-ablation-v1.json
smoke_result="outputs/chloe/ablations-math-v1/smoke/${method}-r${rank}/seed-${seed}/result.json"
python -c 'import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory,flush=True); assert torch.cuda.get_device_properties(0).total_memory >= 40*1024**3'
if [[ ! -f "$smoke_result" ]]; then
  python -u experiments/chloe/formal_math.py --config "$config" --method "$method" --rank "$rank" --seed "$seed" --smoke
fi
python - "$smoke_result" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['status']=='completed' and r['smoke'] and r['steps']==2
assert r['source_training_manifest_verified'] and r['fixed_components_unchanged']
assert r['evaluation']['evaluation_examples']==8 and r['evaluation']['ordered_team_manifest_verified']
print('Same-protocol smoke passed:',r['parameters']['trainable_adapter_parameters'],r['evaluation'],flush=True)
PY
resume_args=()
if [[ "${4:-}" == --resume ]]; then resume_args+=(--resume); fi
# USR1 requests a checkpoint in the Python child at the next optimizer boundary.
train_pid=''
trap 'if [[ -n "$train_pid" ]]; then kill -USR1 "$train_pid"; fi' USR1
python -u experiments/chloe/formal_math.py --config "$config" --method "$method" --rank "$rank" --seed "$seed" "${resume_args[@]}" &
train_pid=$!
while true; do
  if wait "$train_pid"; then exit 0; else
    child_status=$?
    if kill -0 "$train_pid" 2>/dev/null; then continue; fi
    exit "$child_status"
  fi
done
