#!/usr/bin/env python3
"""Serially checkpoint-resume one registered EuroSAT LoRA run until its result is valid."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


POLL_SECONDS = 30
REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "experiments/acheng/scripts/submit_lora.sh"
REGISTRY = REPO / "experiments/acheng/runs.json"
PROGRESS = REPO / "experiments/acheng/PROGRESS.md"
EXPERIMENT_ID = "table3/eurosat/lora-r8/seed-2026"


def command(args, *, check=True):
    return subprocess.run(
        args, cwd=REPO, check=check, text=True, capture_output=True
    )


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_config_hash(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_accounting(job_id):
    result = command([
        "sacct", "-n", "-P", "-j", job_id,
        "--format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,End",
    ])
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    parent = next((row for row in rows if row[0] == job_id), None)
    if parent is None:
        return None, rows
    fields = ("job_id", "job_name", "state", "exit_code", "elapsed", "time_limit", "end")
    return dict(zip(fields, parent)), rows


def active_acheng_jobs():
    result = command(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i %j %T"])
    return [
        line.split()
        for line in result.stdout.splitlines()
        if line.split() and line.split()[1].startswith("acheng_")
    ]


def check_current_qos():
    user = os.environ["USER"]
    assoc = command([
        "sacctmgr", "-n", "-P", "show", "assoc", f"user={user}",
        "format=User,Account,Partition,QOS",
    ])
    if not any(row.split("|")[0] == user and row.split("|")[3] == "normal"
               for row in assoc.stdout.splitlines() if row.strip()):
        raise RuntimeError("The user's normal QoS association could not be verified.")
    qos = command([
        "sacctmgr", "-n", "-P", "show", "qos",
        "format=Name,MaxTRESPerUser,MaxJobsPerUser,MaxWall",
    ])
    normal = next(
        (row.split("|") for row in qos.stdout.splitlines()
         if row.strip() and row.split("|")[0] == "normal"),
        None,
    )
    if normal is None or len(normal) < 4:
        raise RuntimeError("Could not verify current normal QoS limits.")
    limits = normal[1]
    required = ("cpu=10", "gres/gpu=1", "mem=30G")
    if any(token not in limits for token in required):
        raise RuntimeError(f"normal QoS limits changed or are insufficient: {normal}")
    if normal[3] not in {"06:00:00", "6:00:00"}:
        raise RuntimeError(f"Unexpected normal QoS wall-time limit: {normal[3]}")


def load_run_record():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    matches = [
        item for item in data["task_jobs"]
        if item.get("experiment_id") == EXPERIMENT_ID
        and item.get("attempt") == 3
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one registered attempt-3 unit, found {len(matches)}.")
    return data, matches[0]


def write_registry(data):
    temporary = REGISTRY.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, REGISTRY)


def set_status_line(text):
    lines = PROGRESS.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("**State:**"):
            lines[index] = f"**State:** {text}"
            break
    PROGRESS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_final_result(run_dir):
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    split = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    expected_hash = canonical_config_hash(run_dir / "config.json")
    if result.get("status") != "completed" or result.get("config_sha256") != expected_hash:
        raise RuntimeError("Final result status/config hash is invalid.")
    if result.get("max_epochs") != 30 or not 1 <= result.get("best_epoch", 0) <= 30:
        raise RuntimeError("Final result does not represent the registered full training.")
    if result.get("dataset_revision") != split.get("revision"):
        raise RuntimeError("Final result and split dataset revisions differ.")
    if result.get("classes") != split.get("classes"):
        raise RuntimeError("Final result class order differs from the audited split.")
    expected_prompts = [f"a photo of a {name}" for name in result["classes"]]
    if result.get("prompts") != expected_prompts:
        raise RuntimeError("Final result prompts do not match the class order.")
    test = result["test"]
    expected_indices = split["test_indices"]
    if test.get("indices") != expected_indices or test.get("count") != len(expected_indices):
        raise RuntimeError("Test predictions do not match the complete pinned test split.")
    labels = test["labels"]
    predictions = test["predictions"]
    count = test["count"]
    if len(labels) != count or len(predictions) != count or len(set(test["indices"])) != count:
        raise RuntimeError("Test arrays have missing or duplicate examples.")
    if any(type(x) is not int or not 0 <= x < len(result["classes"]) for x in labels + predictions):
        raise RuntimeError("A test label or prediction is outside the class range.")
    correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    accuracy = 100.0 * correct / count
    if correct != test["correct"] or not math.isclose(
        accuracy, test["accuracy"], rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError("Stored test score does not match labels and predictions.")
    if result.get("frozen_parameters_unchanged") is not True:
        raise RuntimeError("Frozen backbone verification did not pass.")
    if not result.get("updated_parameter_names"):
        raise RuntimeError("No adapter parameter update was recorded.")
    for key in ("training_seconds", "evaluation_seconds", "elapsed_seconds", "gpu_hours"):
        if not math.isfinite(float(result[key])):
            raise RuntimeError(f"Non-finite resource metric: {key}")
    return result


def checkpoint_status(run_dir):
    path = run_dir / "checkpoint_status.json"
    if not path.is_file():
        return None
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "checkpointed_for_time_limit":
        raise RuntimeError("Unexpected checkpoint status.")
    if status.get("config_sha256") != canonical_config_hash(run_dir / "config.json"):
        raise RuntimeError("Checkpoint status config hash does not match immutable config.")
    checkpoint = run_dir / "checkpoint_latest.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("Latest checkpoint is missing or empty.")
    if Path(status.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise RuntimeError("Checkpoint status points outside the current run.")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return status, digest


def verify_source(run_dir, job_id):
    manifest = run_dir / f"source-{job_id}.sha256"
    if not manifest.is_file():
        raise RuntimeError(f"Source hash manifest is missing for job {job_id}.")
    command(["sha256sum", "--check", str(manifest)])


def update_segment(job_id, accounting, rows, run_dir, *, state, checkpoint=None):
    data, record = load_run_record()
    segment = next(
        (item for item in record.setdefault("segments", [])
         if item.get("job_id") == job_id),
        None,
    )
    if segment is None:
        segment = {"job_id": job_id}
        record["segments"].append(segment)
    segment.update({
        "state": state,
        "ended_at": accounting.get("end") or now(),
        "elapsed": accounting.get("elapsed"),
        "exit_code": accounting.get("exit_code"),
        "requested_time_limit": accounting.get("time_limit"),
        "log": f"experiments/acheng/outputs/EuroSAT/LoRA-r8/2026/"
               f"{run_dir.name}/slurm-{job_id}.out",
        "stderr_bytes": (run_dir / f"slurm-{job_id}.err").stat().st_size,
    })
    if checkpoint:
        checkpoint_json, digest = checkpoint
        segment.update({
            "checkpoint_confirmation": (
                f"checkpointed_for_time_limit; next_epoch "
                f"{checkpoint_json['next_epoch']}, next_batch "
                f"{checkpoint_json['next_batch']}, global_step "
                f"{checkpoint_json['global_step']}"
            ),
            "checkpoint_sha256": digest,
        })
    log_path = run_dir / f"slurm-{job_id}.out"
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        resume_match = re.search(
            r"Resuming .* at epoch (\d+), batch (\d+), global step (\d+)\.",
            log_text,
        )
        if resume_match:
            segment["resume_confirmation"] = (
                f"Restored epoch {resume_match.group(1)}, batch "
                f"{resume_match.group(2)}, global step {resume_match.group(3)}."
            )
        history = record.setdefault("validation_history", [])
        known_epochs = {entry["epoch"] for entry in history}
        for line in log_text.splitlines():
            if line.startswith("{"):
                try:
                    metric = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "validation_accuracy" not in metric or metric.get("epoch") in known_epochs:
                    continue
                history.append({
                    "epoch": metric["epoch"],
                    "accuracy_percent": metric["validation_accuracy"],
                    "correct": metric["validation_correct"],
                    "count": metric["validation_count"],
                })
                known_epochs.add(metric["epoch"])
    child_states = [
        row[2] for row in rows
        if len(row) > 2 and row[0].startswith(job_id + ".")
    ]
    if any("CANCELLED by" in item for item in child_states):
        segment["step_states"] = child_states
    record["job_id"] = job_id
    record["scheduler_state"] = accounting["state"]
    record["exit_code"] = accounting["exit_code"]
    record["ended_at"] = accounting.get("end") or now()
    if state == "COMPLETED_CHECKPOINTED":
        record["status"] = "CHECKPOINTED"
        record["next_step"] = (
            f"Continue from epoch{checkpoint[0]['next_epoch']}/"
            f"batch{checkpoint[0]['next_batch']}/"
            f"step{checkpoint[0]['global_step']} in the same run."
        )
    elif state == "CANCELLED_BY_UID_CHECKPOINT_PENDING_VERIFY":
        record["status"] = "INTERRUPTED_CHECKPOINT_PENDING_VERIFY"
        record["next_step"] = "Resume same run; inspect exact checkpoint restoration before proceeding."
    elif state == "COMPLETED":
        record["status"] = "COMPLETED_VALIDATED"
        record["next_step"] = "Add validated EuroSAT LoRA result to the phase report."
        result = validate_final_result(run_dir)
        record["result"] = {
            "best_epoch": result["best_epoch"],
            "test_count": result["test"]["count"],
            "test_correct": result["test"]["correct"],
            "test_accuracy_percent": result["test"]["accuracy"],
            "adapter_trainable_parameters": result["adapter_trainable_parameters"],
            "total_trainable_parameters": result["total_trainable_parameters"],
            "training_seconds": result["training_seconds"],
            "evaluation_seconds": result["evaluation_seconds"],
            "elapsed_seconds": result["elapsed_seconds"],
            "gpu_hours": result["gpu_hours"],
            "peak_cuda_bytes": result["peak_cuda_bytes"],
            "gpu": result["gpu"],
        }
    write_registry(data)
    current = (
        f"EuroSAT LoRA seed 2026: {state}, last job {job_id}; "
        + (
            f"checkpoint epoch{checkpoint[0]['next_epoch']}/"
            f"batch{checkpoint[0]['next_batch']}/step{checkpoint[0]['global_step']}."
            if checkpoint else "final result validation pending."
        )
    )
    set_status_line(current)


def submit_resume(run_dir):
    check_current_qos()
    active = active_acheng_jobs()
    if active:
        raise RuntimeError(f"Refusing duplicate submission while acheng jobs are active: {active}")
    text = command(["bash", str(WRAPPER), "--resume", str(run_dir)]).stdout
    match = re.search(r"Submitted job (\d+)", text)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job ID from wrapper output: {text}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True, help="Already-submitted current continuation.")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_root = (REPO / "experiments/acheng/outputs/EuroSAT/LoRA-r8/2026").resolve()
    if not run_dir.is_relative_to(output_root):
        raise RuntimeError("Run directory must be under acheng EuroSAT LoRA-r8 outputs.")
    if not (run_dir / "config.json").is_file():
        raise FileNotFoundError("Immutable run config is missing.")
    if not (run_dir / "checkpoint_latest.pt").is_file():
        raise FileNotFoundError("Run checkpoint is missing.")
    current_job = args.job_id
    data, record = load_run_record()
    existing = next(
        (segment for segment in record.setdefault("segments", [])
         if segment.get("job_id") == current_job),
        None,
    )
    if existing is None:
        record["segments"].append({
            "job_id": current_job,
            "state": "RUNNING",
            "submitted_at": now(),
            "requested_time_limit": "00:03:00",
            "resume_confirmation": "pending startup log",
            "log": f"experiments/acheng/outputs/EuroSAT/LoRA-r8/2026/"
                   f"{run_dir.name}/slurm-{current_job}.out",
        })
    record["job_id"] = current_job
    record["status"] = "RUNNING"
    record["scheduler_state"] = "RUNNING"
    record["next_step"] = f"Monitor sequential continuation job {current_job}."
    write_registry(data)

    while True:
        log_path = run_dir / "continuation-control.log"
        accounting, rows = get_accounting(current_job)
        if accounting is None or accounting["state"] in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{now()} waiting job={current_job}\n")
            time.sleep(POLL_SECONDS)
            continue

        verify_source(run_dir, current_job)
        result_path = run_dir / "result.json"
        checkpoint = None
        if result_path.is_file():
            result = validate_final_result(run_dir)
            update_segment(current_job, accounting, rows, run_dir, state="COMPLETED", checkpoint=None)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{now()} complete job={current_job} "
                    f"best_epoch={result['best_epoch']} "
                    f"test={result['test']['correct']}/{result['test']['count']} "
                    f"accuracy={result['test']['accuracy']:.12f}\n"
                )
            return 0

        if accounting["state"] == "COMPLETED" and accounting["exit_code"] == "0:0":
            checkpoint = checkpoint_status(run_dir)
            if checkpoint is None:
                raise RuntimeError(
                    f"Job {current_job} exited cleanly but produced neither final result nor checkpoint status."
                )
            update_segment(
                current_job, accounting, rows, run_dir,
                state="COMPLETED_CHECKPOINTED", checkpoint=checkpoint
            )
        else:
            child_cancelled_by_uid = any(
                len(row) > 2 and row[0].startswith(current_job + ".")
                and "CANCELLED by" in row[2]
                for row in rows
            )
            if not child_cancelled_by_uid:
                raise RuntimeError(
                    f"Job {current_job} ended unexpectedly: "
                    f"{accounting['state']} {accounting['exit_code']}; inspect its logs."
                )
            if not (run_dir / "checkpoint_latest.pt").is_file():
                raise RuntimeError("Externally interrupted run has no resumable checkpoint.")
            update_segment(
                current_job, accounting, rows, run_dir,
                state="CANCELLED_BY_UID_CHECKPOINT_PENDING_VERIFY"
            )

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{now()} segment={current_job} state={accounting['state']} "
                f"exit={accounting['exit_code']} next_action=resume_same_run\n"
            )
        current_job = submit_resume(run_dir)
        data, record = load_run_record()
        record["job_id"] = current_job
        record["status"] = "RUNNING"
        record["scheduler_state"] = "RUNNING"
        record["submitted_at"] = now()
        record["started_at"] = None
        record["ended_at"] = None
        record["exit_code"] = None
        record["next_step"] = f"Verify sequential continuation job {current_job}."
        record.setdefault("segments", []).append({
            "job_id": current_job,
            "state": "RUNNING",
            "submitted_at": record["submitted_at"],
            "requested_time_limit": "00:03:00",
            "resume_confirmation": "pending startup log",
            "log": f"experiments/acheng/outputs/EuroSAT/LoRA-r8/2026/"
                   f"{run_dir.name}/slurm-{current_job}.out",
        })
        write_registry(data)
        set_status_line(
            f"EuroSAT LoRA seed 2026: job {current_job} running as a 3-minute "
            "same-run checkpoint continuation; no other experiment is scheduled."
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{now()} submitted job={current_job}\n")


if __name__ == "__main__":
    sys.exit(main())
