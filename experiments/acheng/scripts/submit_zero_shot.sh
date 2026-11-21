#!/bin/bash
set -eo pipefail

REPO=/home/msds/weicheng001/e2lora-reproduction
cd "$REPO"

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 <config.json> [--retry <existing-run-directory>]" >&2
    exit 2
fi
CONFIG=$(realpath "$1")
RETRY_DIR=
if [[ "${2:-}" == "--retry" && -n "${3:-}" ]]; then
    RETRY_DIR="$3"
elif [[ $# -ne 1 ]]; then
    echo "Usage: $0 <config.json> [--retry <existing-run-directory>]" >&2
    exit 2
fi
case "$CONFIG" in
    "$REPO/experiments/acheng/configs/"*.json) ;;
    *) echo "Config must be under experiments/acheng/configs/." >&2; exit 2 ;;
esac
SBATCH_SCRIPT="$REPO/experiments/acheng/scripts/evaluate_zero_shot.sbatch"
REGISTRY="$REPO/experiments/acheng/runs.json"
if rg -n 'experiments/kathy/' "$CONFIG" "$SBATCH_SCRIPT"; then
    echo "Active configuration or Slurm script references experiments/kathy/." >&2
    exit 2
fi
if squeue -h -u "$USER" -o "%j %T" | awk '$1 ~ /^acheng_/ { found=1 } END { exit !found }'; then
    echo "An acheng experiment is already pending or running; refusing to submit another." >&2
    exit 3
fi

mapfile -t META < <(python - "$CONFIG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1],encoding="utf-8"))
for x in (c["experiment_id"],c["dataset"]["name"],c["method"]["name"],
          str(c["evaluation"]["seed"]),c["execution"]["run_root"]):
 print(x)
PY
)
if [[ "${#META[@]}" -ne 5 ]]; then
    echo "Failed to read zero-shot config metadata." >&2
    exit 2
fi
EXPERIMENT_ID="${META[0]}"
DATASET="${META[1]}"
METHOD="${META[2]}"
SEED="${META[3]}"
RUN_ROOT_REL="${META[4]}"
RUN_ROOT="$REPO/$RUN_ROOT_REL"
case "$(realpath -m "$RUN_ROOT")/" in
    "$REPO/experiments/acheng/outputs/"*) ;;
    *) echo "Configured output root must be under experiments/acheng/outputs/." >&2; exit 2 ;;
esac
SAFE_DATASET=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_')
JOB_NAME="acheng_${SAFE_DATASET}_zs${SEED}"

if [[ -n "$RETRY_DIR" ]]; then
    RUN_DIR=$(realpath "$RETRY_DIR")
    case "$RUN_DIR/" in
        "$(realpath -m "$RUN_ROOT")/"*) ;;
        *) echo "Retry directory must be under the configured dataset/method output root." >&2; exit 2 ;;
    esac
    if [[ ! -f "$RUN_DIR/config.json" || -e "$RUN_DIR/result.json" ]] || \
       ! cmp -s "$CONFIG" "$RUN_DIR/config.json"; then
        echo "Retry requires the immutable matching config and no existing result." >&2
        exit 2
    fi
    python - "$REGISTRY" "$EXPERIMENT_ID" "$RUN_DIR" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); rows=[r for r in d["task_jobs"] if r.get("experiment_id")==sys.argv[2]]
relative=sys.argv[3].removeprefix("/home/msds/weicheng001/e2lora-reproduction/")
if len(rows)!=1 or rows[0].get("output_dir","").rstrip("/")!=relative:
 raise SystemExit("Retry directory does not match exactly one registered zero-shot unit.")
if rows[0].get("status") not in {"FAILED","FAILED_RETRYABLE","CANCELLED_BY_UID_CHECKPOINTED"}:
 raise SystemExit("Only a failed zero-shot unit without result.json may be retried.")
PY
else
    if python - "$REGISTRY" "$EXPERIMENT_ID" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
raise SystemExit(0 if any(r.get("experiment_id")==sys.argv[2] for r in d["task_jobs"]) else 1)
PY
    then
        echo "This exact experiment unit is already registered; reconcile it instead of resubmitting." >&2
        exit 3
    fi
    RUN_ID="$(date +%Y%m%dT%H%M%S%z)"
    RUN_DIR="$(realpath -m "$RUN_ROOT")/$RUN_ID"
    mkdir -p "$RUN_ROOT"
    mkdir "$RUN_DIR"
    cp "$CONFIG" "$RUN_DIR/config.json"
fi

ACHENG_CONFIG="$RUN_DIR/config.json"
ACHENG_RUN_DIR="$RUN_DIR"
export ACHENG_CONFIG ACHENG_RUN_DIR
for path in "$RUN_DIR" "$(dirname "$RUN_DIR/slurm-%j.out")" \
    "$(dirname "$RUN_DIR/result.json")" "$(dirname "$RUN_DIR/predictions.json")"; do
    resolved=$(realpath "$path")
    case "$resolved/" in
        "$REPO/experiments/acheng/outputs/"*) ;;
        *) echo "Run/log/result path escapes acheng outputs: $resolved" >&2; exit 2 ;;
    esac
done

JOB_ID=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --output="$RUN_DIR/slurm-%j.out" \
    --error="$RUN_DIR/slurm-%j.err" \
    --export=ALL,ACHENG_RUN_DIR,ACHENG_CONFIG \
    "$SBATCH_SCRIPT")
printf '%s\n' "$JOB_ID" > "$RUN_DIR/job-$JOB_ID.txt"

python - "$REGISTRY" "$RUN_DIR" "$JOB_ID" "$EXPERIMENT_ID" "$DATASET" "$METHOD" "$SEED" "$RETRY_DIR" <<'PY'
from datetime import datetime
from pathlib import Path
import hashlib,json,os,sys
registry,run_dir,job_id,experiment_id,dataset,method,seed,retry_dir=sys.argv[1:]
run_dir=Path(run_dir);b=(run_dir/"config.json").read_bytes();config=json.loads(b);d=json.loads(Path(registry).read_text(encoding="utf-8"))
rows=[r for r in d["task_jobs"] if r.get("experiment_id")==experiment_id]
segment={"job_id":job_id,"state":"SUBMITTED"}
if retry_dir:
 if len(rows)!=1: raise SystemExit("Expected exactly one registered zero-shot unit during retry.")
 record=rows[0];record.setdefault("segments",[]).append(segment)
 record.update({"status":"SUBMITTED","scheduler_state":"SUBMITTED","job_id":job_id,
                "submitted_at":datetime.now().astimezone().isoformat(timespec="seconds"),
                "failure_reason":None,"retry_count":record.get("retry_count",0)+1,
                "stdout":str((run_dir/f"slurm-{job_id}.out").relative_to(Path.cwd())),
                "stderr":str((run_dir/f"slurm-{job_id}.err").relative_to(Path.cwd())),
                "next_step":f"Validate zero-shot accuracy and full test sample count for job {job_id}."})
else:
 if rows: raise SystemExit("Experiment was registered concurrently; reconcile manually.")
 record={"experiment_id":experiment_id,
         "model":f"{config['model']['repo']}@{config['model']['revision']}",
         "dataset":f"{config['dataset']['repo']}@{config['dataset']['revision']}",
         "method":method,"seed":int(seed),"config_sha256":hashlib.sha256(b).hexdigest(),
         "status":"SUBMITTED","scheduler_state":"SUBMITTED","job_id":job_id,
         "submitted_at":datetime.now().astimezone().isoformat(timespec="seconds"),
         "partition":config["execution"]["partition"],"qos":config["execution"]["qos"],
         "resources":{"gpus":config["execution"]["gpus"],"cpus":config["execution"]["cpus"],
                      "memory":config["execution"]["memory"],"time_limit":config["execution"]["time_limit"]},
         "output_dir":str(run_dir.relative_to(Path.cwd()))+"/",
         "stdout":str((run_dir/f"slurm-{job_id}.out").relative_to(Path.cwd())),
         "stderr":str((run_dir/f"slurm-{job_id}.err").relative_to(Path.cwd())),
         "segments":[segment],"result_path":None,
         "next_step":f"Verify the pinned {dataset} split and class map, score all test samples, and validate result.json for job {job_id}."}
 d["task_jobs"].append(record)
tmp=Path(registry).with_suffix(".json.tmp");tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");os.replace(tmp,registry)
PY
printf 'Submitted job %s\nRun directory: %s\n' "$JOB_ID" "$RUN_DIR"
