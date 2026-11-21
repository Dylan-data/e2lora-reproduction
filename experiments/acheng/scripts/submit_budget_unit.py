#!/usr/bin/env python3
"""Register and submit one serial image experiment within current Slurm limits."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments" / "acheng"
REGISTRY = TASK / "runs.json"


def persist(data):
    temp = REGISTRY.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, REGISTRY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--estimate-minutes", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    config_path = args.config.resolve()
    config_path.relative_to(TASK / "configs")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != ("one_batch_pilot" if args.phase == "pilot" else "formal"):
        raise ValueError("Phase and configuration disagree.")
    if args.estimate_minutes <= 0:
        raise ValueError("The runtime estimate must be positive.")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    previous = [x for x in registry["task_jobs"] if x.get("experiment_id") == config["experiment_id"]]
    if previous:
        latest = max(previous, key=lambda x: x.get("attempt", 1))
        if latest.get("status") != "FAILED" or latest.get("result_path"):
            raise ValueError("Experiment is already active, checkpointed, or validated; reconcile/resume it instead.")
        if latest.get("attempt", 1) >= 3:
            raise ValueError("Two retries have already been used for this configuration.")
        attempt = latest.get("attempt", 1) + 1
    else:
        attempt = 1
    if any(x.get("status") in ("SUBMITTED", "PENDING", "RUNNING") for x in registry["task_jobs"]):
        raise ValueError("An active task unit is already registered.")
    active = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%j|%T|%R"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if active:
        raise RuntimeError(f"User queue is not empty; inspect before submission:\n{active}")
    assoc = subprocess.run(
        ["sacctmgr", "-n", "-P", "show", "assoc", f"user={os.environ['USER']}",
         "format=Account,Partition,QOS,MaxJobs,MaxTRES,GrpTRES"],
        text=True, capture_output=True, check=True,
    ).stdout
    if not any("normal" in line.split("|")[2].split(",") for line in assoc.splitlines() if len(line.split("|")) >= 3):
        raise RuntimeError(f"normal QoS not present in the current association:\n{assoc}")
    now = datetime.now().astimezone()
    run_root = (ROOT / config["execution"]["run_root"]).resolve()
    run_root.relative_to(TASK / "outputs")
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / now.strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir()
    run_config = run_dir / "config.json"
    run_config.write_bytes(config_bytes)
    relative = str(run_dir.relative_to(ROOT))
    record = {
        "experiment_id": config["experiment_id"],
        "attempt": attempt,
        "model": f"{config['model']['repo']}@{config['model']['revision']}",
        "dataset": f"{config['dataset']['repo']}@{config['dataset']['revision']}",
        "method": f"{config['method']['name']}-r{config['method']['rank']}",
        "seed": config["training"]["seed"],
        "validation_seed": config["dataset"].get("validation_seed", config["training"]["seed"]),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "canonical_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "status": "PREPARED", "job_id": None, "phase": args.phase,
        "estimated_gpu_minutes": args.estimate_minutes,
        "purpose": "Verify shared A identities, distinct B identities, exact parameter count and one real batch update"
        if args.phase == "pilot" else "Complete 30 epochs, validation-only checkpoint selection and full test evaluation",
        "acceptance": "Finite loss and gradients, updated B, unchanged frozen weights, correct shared object counts and parameter count"
        if args.phase == "pilot" else "Matching split and config, 30 epochs, finite training, updated adapters, frozen base, full prediction and metric audit",
        "output_dir": relative + "/", "checkpoint": relative + "/checkpoint_latest.pt",
        "result_path": None, "segments": [], "partition": "MGPU-TC2", "qos": "normal",
        "next_step": "Submit this single registered GPU unit and validate before any other submission.",
    }
    registry["task_jobs"].append(record)
    persist(registry)
    env = os.environ.copy()
    env.update(ACHENG_RUN_DIR=str(run_dir), ACHENG_CONFIG=str(run_config), ACHENG_PHASE=args.phase, ACHENG_RESUME="0")
    limit = "00:15:00" if args.phase == "pilot" else config["execution"]["time_limit"]
    dataset_slug = config["dataset"]["name"].lower().replace("-", "")
    method_slug = config["method"]["name"].lower().replace("²", "2")
    command = [
        "sbatch", "--parsable", f"--time={limit}",
        f"--job-name=acheng_{dataset_slug}_{'sharepilot' if args.phase == 'pilot' else method_slug}",
        f"--output={run_dir}/slurm-%j.out", f"--error={run_dir}/slurm-%j.err",
        "--export=ALL,ACHENG_RUN_DIR,ACHENG_CONFIG,ACHENG_PHASE,ACHENG_RESUME",
        str(TASK / "scripts/image_adapter_budget.sbatch"),
    ]
    submitted = subprocess.run(command, env=env, text=True, capture_output=True)
    if submitted.returncode:
        record.update(status="FAILED", failure_reason=f"sbatch failed: {submitted.stderr.strip()}")
        persist(registry)
        raise RuntimeError(record["failure_reason"])
    job = submitted.stdout.strip().split(";", 1)[0]
    if not job.isdigit():
        raise RuntimeError(f"Cannot interpret sbatch response; reconcile before retry: {submitted.stdout!r}")
    record.update(status="SUBMITTED", scheduler_state="SUBMITTED", job_id=job,
                  stdout=f"{relative}/slurm-{job}.out", stderr=f"{relative}/slurm-{job}.err",
                  submitted_at=now.isoformat(timespec="seconds"),
                  segments=[{"job_id": job, "state": "SUBMITTED", "time_limit": limit}],
                  next_step=f"Monitor job {job}; validate before submitting the next unit.")
    persist(registry)
    print(f"job={job} phase={args.phase} run={run_dir} estimate_minutes={args.estimate_minutes}")


if __name__ == "__main__":
    main()
