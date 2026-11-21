#!/usr/bin/env python3
"""Verify and serially resume an image run after a Slurm interruption."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments" / "acheng"
REGISTRY = TASK / "runs.json"


def persist(data):
    temp = REGISTRY.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, REGISTRY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--estimate-minutes", required=True, type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    matches = [r for r in registry["task_jobs"] if r.get("experiment_id") == args.experiment_id]
    if not matches:
        raise ValueError("Expected a registered unit.")
    record = max(matches, key=lambda x: x.get("attempt", 1))
    if record.get("phase") != "formal" or record.get("status") not in ("SUBMITTED", "RUNNING", "INTERRUPTED_CHECKPOINTED"):
        raise ValueError("Unit is not an interrupted formal run.")
    queue = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%j|%T|%R"],
                           text=True, capture_output=True, check=True).stdout.strip()
    if queue:
        raise RuntimeError(f"User queue not empty:\n{queue}")
    prior_job = record["job_id"]
    accounting = subprocess.run(
        ["sacct", "-n", "-P", "-j", prior_job, "--format=JobID,State,ExitCode,ElapsedRaw"],
        text=True, capture_output=True, check=True,
    ).stdout
    entries = [line.split("|") for line in accounting.splitlines()]
    top = next((fields for fields in entries if fields[0] == prior_job), None)
    step = next((fields for fields in entries if fields[0] == f"{prior_job}.0"), None)
    if not top or not step or not (step[1].startswith("CANCELLED") or top[1].startswith("TIMEOUT")):
        raise RuntimeError(f"Not a verified cancellation/timeout; diagnose first:\n{accounting}")
    segment_seconds = int(top[3])
    run_dir = (ROOT / record["output_dir"]).resolve()
    run_dir.relative_to(TASK / "outputs")
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    if (run_dir / "result.json").exists() or not checkpoint_path.is_file():
        raise ValueError("Result already exists or checkpoint is missing.")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if hashlib.sha256(config_bytes).hexdigest() != record["config_sha256"]:
        raise ValueError("Config bytes differ from registered config.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if checkpoint["config_sha256"] != canonical or canonical != record["canonical_config_sha256"]:
        raise ValueError("Checkpoint config does not match registered config.")
    for path, digest in checkpoint["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source differs from checkpoint: {path}")
    for key in ("optimizer", "scheduler", "python_rng", "torch_rng", "cuda_rng", "epoch_order", "best_adapter", "history"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing {key}")
    if checkpoint["split"] != json.loads((run_dir / "splits.json").read_text(encoding="utf-8")):
        raise ValueError("Checkpoint and saved split disagree.")
    segment = record["segments"][-1]
    if "allocated_gpu_seconds" not in segment:
        segment.update(state=top[1], exit_code=top[2], allocated_gpu_seconds=segment_seconds)
    now = datetime.now().astimezone()
    if args.estimate_minutes <= 0:
        raise ValueError("The continuation estimate must be positive.")
    assoc = subprocess.run(
        ["sacctmgr", "-n", "-P", "show", "assoc", f"user={os.environ['USER']}",
         "format=Account,Partition,QOS,MaxJobs,MaxTRES,GrpTRES"], text=True, capture_output=True, check=True,
    ).stdout
    if not any("normal" in line.split("|")[2].split(",") for line in assoc.splitlines() if len(line.split("|")) >= 3):
        raise RuntimeError(f"normal QoS not available:\n{assoc}")
    record.update(status="INTERRUPTED_CHECKPOINTED", scheduler_state=top[1], exit_code=top[2],
                  checkpoint_progress={"epoch": checkpoint["epoch"], "next_batch": checkpoint["next_batch"],
                                       "global_step": checkpoint["global_step"],
                                       "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
                  next_step="Submit one continuation using the validated checkpoint.")
    persist(registry)
    limit = config["execution"]["time_limit"]
    env = os.environ.copy()
    env.update(ACHENG_RUN_DIR=str(run_dir), ACHENG_CONFIG=str(config_path),
               ACHENG_PHASE="formal", ACHENG_RESUME="1")
    dataset_slug = config["dataset"]["name"].lower().replace("-", "")
    method_slug = config["method"]["name"].lower().replace("²", "2")
    command = [
        "sbatch", "--parsable", f"--time={limit}",
        f"--job-name=acheng_{dataset_slug}_{method_slug}",
        f"--output={run_dir}/slurm-%j.out", f"--error={run_dir}/slurm-%j.err",
        "--export=ALL,ACHENG_RUN_DIR,ACHENG_CONFIG,ACHENG_PHASE,ACHENG_RESUME",
        str(TASK / "scripts/image_adapter_budget.sbatch"),
    ]
    submitted = subprocess.run(command, env=env, text=True, capture_output=True)
    if submitted.returncode:
        record["next_step"] = f"Submission failed; preserve checkpoint: {submitted.stderr.strip()}"
        persist(registry)
        raise RuntimeError(record["next_step"])
    job = submitted.stdout.strip().split(";", 1)[0]
    if not job.isdigit():
        raise RuntimeError(f"Unrecognized sbatch response; reconcile before retry: {submitted.stdout!r}")
    relative = str(run_dir.relative_to(ROOT))
    record["segments"].append({"job_id": job, "state": "SUBMITTED", "time_limit": limit})
    record.update(status="SUBMITTED", scheduler_state="SUBMITTED", job_id=job, exit_code=None,
                  stdout=f"{relative}/slurm-{job}.out", stderr=f"{relative}/slurm-{job}.err",
                  submitted_at=now.isoformat(timespec="seconds"),
                  next_step=f"Verify job {job} restored the checkpoint, then monitor and validate.")
    persist(registry)
    print(f"resumed job={job} previous={prior_job} step={checkpoint['global_step']} limit={limit}")


if __name__ == "__main__":
    main()
