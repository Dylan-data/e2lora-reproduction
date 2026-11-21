#!/bin/bash
set -eo pipefail

REPO=/home/msds/weicheng001/e2lora-reproduction
cd "$REPO"
CONFIG="$REPO/experiments/acheng/configs/dtd_lora_seed2026.json"
SBATCH_SCRIPT="$REPO/experiments/acheng/scripts/submit_dtd_lora.sbatch"
REGISTRY="$REPO/experiments/acheng/runs.json"
EXPERIMENT_ID=table3/dtd/lora-r8/seed-2026
RESUME_DIR=

if [[ "${1:-}" == "--resume" ]]; then
    if [[ -z "${2:-}" || $# -ne 2 ]]; then
        echo "Usage: $0 [--resume <existing-run-directory>]" >&2
        exit 2
    fi
    RESUME_DIR="$2"
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--resume <existing-run-directory>]" >&2
    exit 2
fi

for path in "$CONFIG" "$SBATCH_SCRIPT"; do
    case "$(realpath "$path")" in
        "$REPO/experiments/acheng/"*) ;;
        *) echo "Config and script must be owned by experiments/acheng: $path" >&2; exit 2 ;;
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

if [[ -n "$RESUME_DIR" ]]; then
    RUN_DIR=$(realpath "$RESUME_DIR")
    case "$RUN_DIR/" in
        "$REPO/experiments/acheng/outputs/DTD/LoRA-r8/2026/"*) ;;
        *) echo "Resume directory must be under the DTD LoRA output tree." >&2; exit 2 ;;
    esac
    if [[ -e "$RUN_DIR/result.json" || ! -f "$RUN_DIR/checkpoint_latest.pt" || ! -f "$RUN_DIR/config.json" ]] || \
       ! cmp -s "$CONFIG" "$RUN_DIR/config.json"; then
        echo "Resume requires the immutable matching config, checkpoint, and no existing result." >&2
        exit 2
    fi
    python - "$REGISTRY" "$EXPERIMENT_ID" "$RUN_DIR" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
matches=[r for r in d["task_jobs"] if r.get("experiment_id")==sys.argv[2]]
relative=sys.argv[3].removeprefix("/home/msds/weicheng001/e2lora-reproduction/")
if len(matches)!=1 or matches[0].get("output_dir","").rstrip("/")!=relative:
    raise SystemExit("Resume directory does not match exactly one registered run.")
if matches[0].get("status") not in {"CANCELLED_BY_UID_CHECKPOINTED","PAUSED_CHECKPOINTED","INTERRUPTED_CHECKPOINTED"}:
    raise SystemExit("Only a checkpointed run may be resumed.")
PY
    export ACHENG_RESUME=1
else
    if python - "$REGISTRY" "$EXPERIMENT_ID" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
if any(r.get("experiment_id")==sys.argv[2] for r in d["task_jobs"]):
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "This exact experiment unit is already registered; reconcile it instead of resubmitting." >&2
        exit 3
    fi
    RUN_ROOT="$REPO/experiments/acheng/outputs/DTD/LoRA-r8/2026"
    RUN_ID="$(date +%Y%m%dT%H%M%S%z)"
    RUN_DIR="$RUN_ROOT/$RUN_ID"
    mkdir -p "$RUN_ROOT"
    mkdir "$RUN_DIR"
    cp "$CONFIG" "$RUN_DIR/config.json"
    export ACHENG_RESUME=0
fi

ACHENG_CONFIG="$RUN_DIR/config.json"
ACHENG_RUN_DIR="$RUN_DIR"
export ACHENG_CONFIG ACHENG_RUN_DIR
for path in "$RUN_DIR" "$(dirname "$RUN_DIR/slurm-%j.out")" \
    "$(dirname "$RUN_DIR/result.json")" "$(dirname "$RUN_DIR/predictions.npz")" \
    "$(dirname "$RUN_DIR/checkpoint_latest.pt")"; do
    resolved=$(realpath "$path")
    case "$resolved/" in
        "$REPO/experiments/acheng/outputs/"*) ;;
        *) echo "Run/log/result/checkpoint path escapes acheng outputs: $resolved" >&2; exit 2 ;;
    esac
done

JOB_ID=$(sbatch --parsable \
    --job-name=acheng_dtd_lora2026 \
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
run_dir=Path(run_dir); config_bytes=(run_dir/"config.json").read_bytes()
config=json.loads(config_bytes); d=json.loads(Path(registry).read_text(encoding="utf-8"))
matches=[x for x in d["task_jobs"] if x.get("experiment_id")==experiment_id]
segment={"job_id":job_id,"state":"SUBMITTED","requested_time_limit":"00:03:00" if resume=="1" else "05:45:00"}
if resume=="1":
    if len(matches)!=1: raise SystemExit("Expected exactly one registered run during resume.")
    record=matches[0]; record.setdefault("segments",[]).append(segment)
    record.update({"status":"SUBMITTED","scheduler_state":"SUBMITTED","job_id":job_id,
                   "submitted_at":datetime.now().astimezone().isoformat(timespec="seconds"),
                   "started_at":None,"ended_at":None,"exit_code":None,"failure_reason":None,
                   "retry_count":record.get("retry_count",0)+1,
                   "stdout":str((run_dir/f"slurm-{job_id}.out").relative_to(Path.cwd())),
                   "stderr":str((run_dir/f"slurm-{job_id}.err").relative_to(Path.cwd())),
                   "next_step":f"Verify checkpoint restoration and training progress in job {job_id}."})
else:
    if matches: raise SystemExit("Experiment was registered concurrently; reconcile manually.")
    record={
        "experiment_id":experiment_id,
        "model":f"{config['model']['repo']}@{config['model']['revision']}",
        "dataset":f"{config['dataset']['repo']}@{config['dataset']['revision']}",
        "method":"LoRA-r8-alpha16","seed":config["training"]["seed"],
        "validation_seed":config["dataset"]["validation_seed"],
        "config_sha256":hashlib.sha256(config_bytes).hexdigest(),
        "canonical_config_sha256":hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest(),
        "status":"SUBMITTED","scheduler_state":"SUBMITTED","job_id":job_id,
        "submitted_at":datetime.now().astimezone().isoformat(timespec="seconds"),
        "attempt":1,"retry_count":0,"partition":"MGPU-TC2","qos":"normal",
        "resources":{"gpus":1,"cpus":4,"memory":"24G","time_limit":"05:45:00"},
        "output_dir":str(run_dir.relative_to(Path.cwd()))+"/",
        "stdout":str((run_dir/f"slurm-{job_id}.out").relative_to(Path.cwd())),
        "stderr":str((run_dir/f"slurm-{job_id}.err").relative_to(Path.cwd())),
        "checkpoint":str((run_dir/"checkpoint_latest.pt").relative_to(Path.cwd())),
        "checkpoint_path":str((run_dir/"checkpoint_latest.pt").relative_to(Path.cwd())),
        "segments":[segment],"result_path":None,
        "next_step":f"Validate 30-epoch DTD LoRA run, test sample count, metrics, parameters and checkpoint after job {job_id}."
    }
    d["task_jobs"].append(record)
temp=Path(registry).with_suffix(".json.tmp")
temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
os.replace(temp,registry)
PY
printf 'Submitted job %s\nRun directory: %s\n' "$JOB_ID" "$RUN_DIR"
