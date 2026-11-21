#!/usr/bin/env python3
"""Validate one completed image adapter run and account allocated GPU time."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments" / "acheng"
REGISTRY = TASK / "runs.json"
BASELINES = {
    "DTD": TASK / "outputs/DTD/LoRA-r8/2026/20261119T173117+0800",
    "EuroSAT": TASK / "outputs/EuroSAT/LoRA-r8/2026/20261117T140537+0800",
    "GTSRB": TASK / "outputs/GTSRB/LoRA-r8/2026/20261119T194115+0800",
}
TEST_COUNTS = {"DTD": 1880, "EuroSAT": 2700, "GTSRB": 12630}
STEPS = {"DTD": 1590, "EuroSAT": 9120, "GTSRB": 11250}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    matches = [r for r in registry["task_jobs"] if r.get("experiment_id") == args.experiment_id]
    if not matches:
        raise ValueError("Expected a registered run.")
    record = max(matches, key=lambda x: x.get("attempt", 1))
    if record.get("phase") != "formal" or record.get("status") not in ("SUBMITTED", "RUNNING"):
        raise ValueError("Expected a submitted formal run.")
    run = (ROOT / record["output_dir"]).resolve()
    run.relative_to(TASK / "outputs")
    config_bytes = (run / "config.json").read_bytes()
    config = json.loads(config_bytes)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    predictions = json.loads((run / "predictions.json").read_text(encoding="utf-8"))
    dataset_name = config["dataset"]["name"]
    baseline_dir = BASELINES[dataset_name]
    baseline_result = json.loads((baseline_dir / "result.json").read_text(encoding="utf-8"))
    if hashlib.sha256(config_bytes).hexdigest() != record["config_sha256"]:
        raise ValueError("Config bytes differ from registered config.")
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if canonical != result["config_sha256"] or canonical != record["canonical_config_sha256"]:
        raise ValueError("Result config hash mismatch.")
    for path, digest in result["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Result source hash differs: {path}")
    rank = config["method"]["rank"]
    if config["method"]["name"] == "ShareLoRA":
        family_shapes = {
            "self_attn.q_proj": (768, 768),
            "self_attn.k_proj": (768, 768),
            "self_attn.v_proj": (768, 768),
            "self_attn.out_proj": (768, 768),
            "mlp.fc1": (768, 3072),
            "mlp.fc2": (3072, 768),
        }
        expected_count = sum(rank * in_features for in_features, _ in family_shapes.values())
        expected_count += 12 * sum(rank * out_features for _, out_features in family_shapes.values())
    elif config["method"]["name"] == "LoRA":
        expected_count = 663552 if rank == 4 else 1327104 if rank == 8 else None
    else:
        expected_count = None
    if expected_count is None:
        raise ValueError(f"No parameter formula registered for {config['method']['name']} rank {rank}.")
    expected = (config["experiment_id"], config["method"]["name"], config["method"]["rank"], 2026,
                30, expected_count, expected_count)
    observed = (result["experiment_id"], result["method"], result["rank"], result["seed"],
                result["max_epochs"], result["adapter_trainable_parameters"], result["total_trainable_parameters"])
    if observed != expected:
        raise ValueError(f"Unexpected result configuration: {observed}")
    if result["model_revision"] != baseline_result["model_revision"] or result["dataset_revision"] != baseline_result["dataset_revision"]:
        raise ValueError("Model or data revision differs from baseline.")
    if result["preprocessing"] != baseline_result["preprocessing"] or result["classes"] != baseline_result["classes"]:
        raise ValueError("Preprocessing or label ordering differs from baseline.")
    if result["prompts"] != baseline_result["prompts"] or set(result["target_layers"]) != set(baseline_result["target_layers"]):
        raise ValueError("Prompt or target layer differs from baseline.")
    split = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    if split != json.loads((baseline_dir / "splits.json").read_text(encoding="utf-8")):
        raise ValueError("Train/validation/test sample identities differ from baseline.")
    history = result["history"]
    if len(history) != 30 or [x["epoch"] for x in history] != list(range(1, 31)) or history[-1]["global_step"] != STEPS[dataset_name]:
        raise ValueError(f"Training did not finish 30 epochs and {STEPS[dataset_name]} steps.")
    if any(not math.isfinite(x["train_loss"]) or not math.isfinite(x["validation_accuracy"]) for x in history):
        raise ValueError("Non-finite training or validation metrics.")
    maximum = max(x["validation_accuracy"] for x in history)
    expected_best = next(x["epoch"] for x in history if x["validation_accuracy"] == maximum)
    if result["best_epoch"] != expected_best or result["validation"]["accuracy"] != maximum:
        raise ValueError("Best epoch was not selected from validation accuracy.")
    test_count = TEST_COUNTS[dataset_name]
    validation_count = baseline_result["validation"]["count"]
    for key, count in (("validation", validation_count), ("test", test_count)):
        current = result[key]
        if current != predictions[key] or current["count"] != count:
            raise ValueError(f"Prediction data/count mismatch for {key}.")
        if len(current["labels"]) != count or len(current["predictions"]) != count:
            raise ValueError(f"Incomplete prediction data for {key}.")
        correct = sum(a == b for a, b in zip(current["labels"], current["predictions"]))
        if correct != current["correct"] or abs(100 * correct / count - current["accuracy"]) > 1e-8:
            raise ValueError(f"Prediction accuracy mismatch for {key}.")
    if result["test"]["indices"] != list(range(test_count)) or result["test"]["labels"] != baseline_result["test"]["labels"]:
        raise ValueError("Test order or labels differ from baseline.")
    if result["validation"]["indices"] != split["validation_indices"]:
        raise ValueError("Validation prediction identities differ from split.")
    changed = result["updated_parameter_names"]
    if not result["frozen_parameters_unchanged"] or not any(".down.weight" in s for s in changed) or not any(".up.weight" in s for s in changed):
        raise ValueError("Frozen weights or adapter update audit failed.")
    accounting = subprocess.run(
        ["sacct", "-n", "-P", "-j", record["job_id"], "--format=JobID,State,ExitCode,ElapsedRaw"],
        text=True, capture_output=True, check=True,
    ).stdout
    line = next((x.split("|") for x in accounting.splitlines() if x.split("|")[0] == record["job_id"]), None)
    if not line or line[1:3] != ["COMPLETED", "0:0"]:
        raise ValueError(f"Final Slurm job did not complete successfully:\n{accounting}")
    segment = record["segments"][-1]
    if "allocated_gpu_seconds" in segment:
        raise ValueError("This segment has already been accounted.")
    segment.update(state="COMPLETED", exit_code="0:0", allocated_gpu_seconds=int(line[3]))
    total = sum(s.get("allocated_gpu_seconds", 0) for s in record["segments"])
    record.update(status="COMPLETED_VALIDATED", scheduler_state="COMPLETED", exit_code="0:0",
                  result_path=str((run / "result.json").relative_to(ROOT)),
                  result={"test_correct": result["test"]["correct"], "test_count": result["test"]["count"],
                          "accuracy_percent": result["test"]["accuracy"], "best_epoch": result["best_epoch"],
                          "epochs": 30, "global_step": STEPS[dataset_name], "trainable_parameters": expected_count,
                          "allocated_gpu_seconds": total},
                  next_step="Update the supplied Word report with this validated result before the next unit.")
    temp = REGISTRY.with_suffix(".json.tmp")
    temp.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, REGISTRY)
    print("VALIDATED", args.experiment_id, "job", record["job_id"],
          "test", result["test"]["correct"], "/", result["test"]["count"],
          "accuracy", result["test"]["accuracy"], "best epoch", result["best_epoch"],
          "unit allocated GPU seconds", total)


if __name__ == "__main__":
    main()
