#!/bin/bash
set -eo pipefail

REPO=/home/msds/weicheng001/e2lora-reproduction
cd "$REPO"
CONFIG="$REPO/experiments/acheng/configs/dtd_zero_shot_seed2026.json"
SBATCH_SCRIPT="$REPO/experiments/acheng/scripts/evaluate_zero_shot.sbatch"
REGISTRY="$REPO/experiments/acheng/runs.json"
EXPERIMENT_ID=table3/dtd/zero-shot/seed-2026
RETRY_DIR=

if [[ "${1:-}" == "--retry" ]]; then
    if [[ -z "${2:-}" || $# -ne 2 ]]; then
        echo "Usage: $0 [--retry <existing-run-directory>]" >&2
        exit 2
    fi
    RETRY_DIR="$2"
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--retry <existing-run-directory>]" >&2
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
if [[ -z "$RETRY_DIR" ]] && python - "$REGISTRY" "$EXPERIMENT_ID" <<'PY'
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

if [[ -n "$RETRY_DIR" ]]; then
    RUN_DIR=$(realpath "$RETRY_DIR")
    case "$RUN_DIR/" in
        "$REPO/experiments/acheng/outputs/DTD/zero-shot/2026/"*) ;;
        *) echo "Retry directory must be under the DTD zero-shot output tree." >&2; exit 2 ;;
    esac
    if [[ ! -f "$RUN_DIR/config.json" || -e "$RUN_DIR/result.json" ]] || \
       ! cmp -s "$CONFIG" "$RUN_DIR/config.json"; then
        echo "Retry requires the immutable matching config and no existing result." >&2
        exit 2
    fi
    python - "$REGISTRY" "$EXPERIMENT_ID" "$RUN_DIR" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
matches=[r for r in d["task_jobs"] if r.get("experiment_id")==sys.argv[2]]
if len(matches)!=1 or matches[0].get("output_dir","").rstrip("/")!=sys.argv[3].removeprefix("/home/msds/weicheng001/e2lora-reproduction/"):
    raise SystemExit("Retry directory does not match exactly one registered experiment.")
if matches[0].get("status") not in {"FAILED_RETRYABLE", "FAILED"}:
    raise SystemExit("Only a recorded failed unit may be retried.")
PY
else
    RUN_ROOT="$REPO/experiments/acheng/outputs/DTD/zero-shot/2026"
    RUN_ID="$(date +%Y%m%dT%H%M%S%z)"
    RUN_DIR="$RUN_ROOT/$RUN_ID"
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
    --job-name=acheng_dtd_zs2026 \
    --output="$RUN_DIR/slurm-%j.out" \
    --error="$RUN_DIR/slurm-%j.err" \
    --export=ALL,ACHENG_RUN_DIR,ACHENG_CONFIG \
    "$SBATCH_SCRIPT")
printf '%s\n' "$JOB_ID" > "$RUN_DIR/job-$JOB_ID.txt"

python - "$REGISTRY" "$RUN_DIR" "$JOB_ID" "$EXPERIMENT_ID" "$RETRY_DIR" <<'PY'
from datetime import datetime
from pathlib import Path
import hashlib, json, os, sys

registry, run_dir, job_id, experiment_id, retry_dir = sys.argv[1:]
run_dir = Path(run_dir)
config_bytes = (run_dir / "config.json").read_bytes()
config = json.loads(config_bytes)
d = json.loads(Path(registry).read_text(encoding="utf-8"))
matches = [x for x in d["task_jobs"] if x.get("experiment_id") == experiment_id]
if retry_dir:
    if len(matches) != 1:
        raise SystemExit("Expected exactly one registered unit for retry.")
    record = matches[0]
    record.setdefault("segments", []).append({"job_id": job_id, "state": "SUBMITTED", "retry": True})
    record.update({
        "status": "SUBMITTED",
        "scheduler_state": "SUBMITTED",
        "job_id": job_id,
        "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "failure_reason": None,
        "retry_count": record.get("retry_count", 0) + 1,
        "stdout": str((run_dir / f"slurm-{job_id}.out").relative_to(Path.cwd())),
        "stderr": str((run_dir / f"slurm-{job_id}.err").relative_to(Path.cwd())),
        "next_step": f"Validate DTD zero-shot output and result.json for retry job {job_id}.",
    })
else:
    if matches:
        raise SystemExit("Experiment was registered concurrently; inspect the submitted job manually.")
    record = {
        "experiment_id": experiment_id,
        "model": f"{config['model']['repo']}@{config['model']['revision']}",
        "dataset": f"{config['dataset']['repo']}@{config['dataset']['revision']}",
        "method": "Zero-shot",
        "seed": config["evaluation"]["seed"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "status": "SUBMITTED",
        "scheduler_state": "SUBMITTED",
        "job_id": job_id,
        "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "partition": "MGPU-TC2",
        "qos": "normal",
        "resources": {"gpus": 1, "cpus": 4, "memory": "24G", "time_limit": "02:00:00"},
        "output_dir": str(run_dir.relative_to(Path.cwd())) + "/",
        "stdout": str((run_dir / f"slurm-{job_id}.out").relative_to(Path.cwd())),
        "stderr": str((run_dir / f"slurm-{job_id}.err").relative_to(Path.cwd())),
        "segments": [{"job_id": job_id, "state": "SUBMITTED"}],
        "result_path": None,
        "next_step": f"Verify pinned DTD split/class map, complete 1880-example zero-shot scoring, and validate result.json for job {job_id}.",
    }
    d["task_jobs"].append(record)
temp = Path(registry).with_suffix(".json.tmp")
temp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temp, registry)
PY

printf 'Submitted job %s\nRun directory: %s\n' "$JOB_ID" "$RUN_DIR"
