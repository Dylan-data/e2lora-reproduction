#!/bin/bash
set -euo pipefail

REPO=/home/msds/weicheng001/e2lora-reproduction
CONFIG="$REPO/experiments/acheng/configs/eurosat_lora_seed2027.json"
SBATCH_SCRIPT="$REPO/experiments/acheng/scripts/train_lora.sbatch"
REGISTRY="$REPO/experiments/acheng/runs.json"
EXPERIMENT_ID=table3/eurosat/lora-r8/seed-2027
RESUME_DIR=

if [[ "${1:-}" == "--resume" && $# -eq 2 ]]; then
    RESUME_DIR=$2
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--resume <existing-run-directory>]" >&2
    exit 2
fi

cd "$REPO"

for path in "$CONFIG" "$SBATCH_SCRIPT" "$REGISTRY"; do
    resolved=$(realpath "$path")
    case "$resolved" in
        "$REPO"/experiments/acheng/*) ;;
        *) echo "Refusing path outside experiments/acheng: $resolved" >&2; exit 2 ;;
    esac
done
if rg -n 'experiments/kathy/' "$CONFIG" "$SBATCH_SCRIPT"; then
    echo "Active configuration or Slurm script references experiments/kathy/." >&2
    exit 2
fi

if squeue -h -u "$USER" -o "%j %T" | awk '$1 ~ /^acheng_/ { found=1 } END { exit !found }'; then
    echo "An acheng experiment is already pending or running; refusing to submit another." >&2
    exit 3
fi

assoc=$(sacctmgr -n -P show assoc "user=$USER" format=User,Account,Partition,QOS)
if ! awk -F'|' -v user="$USER" '$1 == user && $4 == "normal" { found=1 } END { exit !found }' <<<"$assoc"; then
    echo "Could not verify the user's normal QoS association." >&2
    exit 2
fi
qos=$(sacctmgr -n -P show qos format=Name,MaxTRESPerUser,MaxJobsPerUser,MaxWall)
normal=$(awk -F'|' '$1 == "normal" { print; exit }' <<<"$qos")
if [[ -z "$normal" ]] || ! grep -q 'cpu=10' <<<"$normal" || \
   ! grep -q 'gres/gpu=1' <<<"$normal" || ! grep -q 'mem=30G' <<<"$normal" || \
   ! grep -Eq '\|(06:00:00|6:00:00)$' <<<"$normal"; then
    echo "Current normal QoS limits are not the verified 1 GPU/10 CPU/30G/6-hour limits." >&2
    exit 2
fi

python - "$REGISTRY" "$EXPERIMENT_ID" "$RESUME_DIR" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [x for x in d["task_jobs"] if x.get("experiment_id") == sys.argv[2]]
if not sys.argv[3] and matches:
    raise SystemExit(f"Experiment unit is already registered; refusing duplicate: {matches[0].get('status')}")
if sys.argv[3] and (
    len(matches) != 1
    or matches[0].get("status") not in {
        "CANCELLED_CHECKPOINTED",
        "CANCELLED_BY_UID_CHECKPOINTED",
        "INTERRUPTED_CHECKPOINTED",
        "FAILED_RESUME_SUBMISSION",
    }
):
    raise SystemExit("Resume requires one interrupted, checkpointed seed-2027 run in the registry.")
PY

if [[ -n "$RESUME_DIR" ]]; then
    RUN_DIR=$(realpath "$RESUME_DIR")
    case "$RUN_DIR/" in
        "$REPO/experiments/acheng/outputs/EuroSAT/LoRA-r8/2027/"*) ;;
        *) echo "Resume directory must be under acheng's seed-2027 output path." >&2; exit 2 ;;
    esac
    if [[ -e "$RUN_DIR/result.json" || ! -f "$RUN_DIR/checkpoint_latest.pt" || ! -f "$RUN_DIR/config.json" ]]; then
        echo "Resume requires an incomplete run with immutable config and a checkpoint." >&2
        exit 2
    fi
    if ! cmp -s "$CONFIG" "$RUN_DIR/config.json"; then
        echo "Resume config differs from the original run config." >&2
        exit 2
    fi
    ACHENG_RESUME=1
else
    RUN_ID=$(date +%Y%m%dT%H%M%S%z)
    RUN_DIR="$REPO/experiments/acheng/outputs/EuroSAT/LoRA-r8/2027/$RUN_ID"
    case "$(realpath -m "$RUN_DIR")/" in
        "$REPO/experiments/acheng/outputs/"*) ;;
        *) echo "Run path escapes experiments/acheng/outputs." >&2; exit 2 ;;
    esac
    if [[ -e "$RUN_DIR" ]]; then
        echo "Run directory already exists; refusing to overwrite: $RUN_DIR" >&2
        exit 2
    fi
    mkdir -p "$(dirname "$RUN_DIR")"
    mkdir "$RUN_DIR"
    cp -- "$CONFIG" "$RUN_DIR/config.json"
    ACHENG_RESUME=0
fi

export ACHENG_RUN_DIR="$RUN_DIR"
export ACHENG_CONFIG="$RUN_DIR/config.json"
export ACHENG_RESUME

for path in "$RUN_DIR" "$(dirname "$RUN_DIR/slurm-%j.out")" \
    "$(dirname "$RUN_DIR/result.json")" "$(dirname "$RUN_DIR/checkpoint_latest.pt")"; do
    resolved=$(realpath "$path")
    case "$resolved/" in
        "$REPO/experiments/acheng/outputs/"*) ;;
        *) echo "Run/log/result/checkpoint path escapes acheng outputs: $resolved" >&2; exit 2 ;;
    esac
done
cmp -s "$CONFIG" "$RUN_DIR/config.json"

JOB_ID=$(sbatch --parsable \
    --job-name=acheng_eu_lora2027 \
    --output="$RUN_DIR/slurm-%j.out" \
    --error="$RUN_DIR/slurm-%j.err" \
    --export=ALL,ACHENG_RUN_DIR,ACHENG_CONFIG,ACHENG_RESUME \
    "$SBATCH_SCRIPT")
printf '%s\n' "$JOB_ID" > "$RUN_DIR/job-$JOB_ID.txt"

python - "$REGISTRY" "$RUN_DIR" "$JOB_ID" "$EXPERIMENT_ID" "$ACHENG_RESUME" <<'PY'
from datetime import datetime
from pathlib import Path
import hashlib, json, os, sys
registry, run_dir, job_id, experiment_id, resume = sys.argv[1:]
run_dir = Path(run_dir)
config_bytes = (run_dir / "config.json").read_bytes()
config = json.loads(config_bytes)
canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
segment = {"job_id": job_id, "state": "SUBMITTED", "requested_time_limit": "05:45:00"}
d = json.loads(Path(registry).read_text(encoding="utf-8"))
matches = [x for x in d["task_jobs"] if x.get("experiment_id") == experiment_id]
if resume == "1":
    if len(matches) != 1:
        raise SystemExit("Expected exactly one registered run during continuation.")
    record = matches[0]
    record.setdefault("segments", []).append(segment)
    record.update({
        "status": "SUBMITTED",
        "scheduler_state": "SUBMITTED",
        "job_id": job_id,
        "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "failure_reason": None,
        "retry_count": record.get("retry_count", 0) + 1,
        "next_step": f"Verify checkpoint restoration in continuation job {job_id}.",
    })
else:
    if matches:
        raise SystemExit("Unit appeared in the registry during submission; reconcile job manually.")
    record = {
    "experiment_id": experiment_id,
    "model": f"{config['model']['repo']}@{config['model']['revision']}",
    "dataset": f"{config['dataset']['repo']}@{config['dataset']['revision']}",
    "method": "LoRA-r8-alpha16",
    "seed": 2027,
    "validation_seed": 2026,
    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    "canonical_config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    "attempt": 1,
    "status": "SUBMITTED",
    "scheduler_state": "SUBMITTED",
    "job_id": job_id,
    "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "started_at": None,
    "ended_at": None,
    "exit_code": None,
    "failure_reason": None,
    "retry_count": 0,
    "partition": "MGPU-TC2",
    "qos": "normal",
    "resources": {"gpus": 1, "cpus": 4, "memory": "24G", "time_limit": "05:45:00"},
    "output_dir": str(run_dir.relative_to(Path.cwd())) + "/",
    "stdout": str((run_dir / f"slurm-{job_id}.out").relative_to(Path.cwd())),
    "stderr": str((run_dir / f"slurm-{job_id}.err").relative_to(Path.cwd())),
    "checkpoint": str((run_dir / "checkpoint_latest.pt").relative_to(Path.cwd())),
    "segments": [segment],
    "next_step": f"Check Slurm job {job_id}; validate full 30-epoch result before any next unit.",
    "result": None,
}
    d["task_jobs"].append(record)
temp = Path(registry).with_suffix(".json.tmp")
temp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temp, registry)
PY

printf 'Submitted job %s\nRun directory: %s\n' "$JOB_ID" "$RUN_DIR"
