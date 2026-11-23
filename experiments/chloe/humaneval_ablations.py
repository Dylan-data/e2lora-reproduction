#!/usr/bin/env python3
"""Run and summarize the HumanEval rank/sample sweeps for report Figure 5."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/chloe/configs/humaneval-figure5.json"
DEFAULT_PLAN = ROOT / "outputs/chloe/humaneval-ablations/plan.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("task") != "code" or config.get("method") != "e2lora":
        raise ValueError("This runner is for E2LoRA HumanEval ablations only")
    if not config.get("seeds") or any(type(x) is not int for x in config["seeds"]):
        raise ValueError("List one or more explicit integer seeds")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seed values must be unique")

    rank = config["rank_sweep"]
    steps = config["sampling_sweep"]

    if rank["fixed_sampling_steps"] not in steps["values"]:
        raise ValueError("The shared rank-8 point must be present in the sampling sweep")
    if rank["values"].count(steps["fixed_rank"]) != 1:
        raise ValueError("The shared fixed-rank point must occur exactly once in the rank sweep")
    for values in (rank["values"], steps["values"]):
        if not values or any(type(x) is not int or x < 1 for x in values):
            raise ValueError("Ranks and sample counts must be positive integers")

    t = config["training"]
    required = ("epochs", "max_seq_length", "max_new_tokens", "per_device_batch_size",
                "gradient_accumulation_steps", "learning_rate", "weight_decay", "warmup_ratio",
                "alpha", "gradient_checkpointing")
                
    missing = [key for key in required if key not in t or t[key] is None]
    if missing:
        raise ValueError("Resolve these training settings explicitly: " + ", ".join(missing))
    return config


def specifications(config):
    rows = {}

    def add(rank, count, study, x_value):
        for seed in config["seeds"]:
            key = (rank, count, seed)
            row = rows.setdefault(key, {"method": "e2lora", "rank": rank,
                "gradient_examples": count, "seed": seed, "studies": [], "x_values": {}})
            if study not in row["studies"]:
                row["studies"].append(study)
            row["x_values"][study] = x_value

    for rank in config["rank_sweep"]["values"]:
        add(rank, config["rank_sweep"]["fixed_sampling_steps"], "figure5_rank", rank)
    for count in config["sampling_sweep"]["values"]:
        add(config["sampling_sweep"]["fixed_rank"], count, "figure5_sampling", count)
    result = []

    for (rank, count, seed), row in sorted(rows.items()):
        row["run_id"] = f"humaneval-e2lora-r{rank}-n{count}-seed{seed}"
        row["output_dir"] = str(Path(config["output_root"]) / "figure5" /
            f"e2lora-r{rank}-n{count}" / f"seed-{seed}")
        result.append(row)
    return result


def write_plan(config, plan_path=DEFAULT_PLAN):

    plan_path = Path(plan_path)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path

    plan_path.parent.mkdir(parents=True, exist_ok=True)

    plan = {"schema_version": 1, "task": "HumanEval", "model_key": config["model_key"],
        "config": str(Path("experiments/chloe/configs/humaneval-figure5.json")),
        "rank_fixed_sampling_steps": config["rank_sweep"]["fixed_sampling_steps"],
        "sampling_fixed_rank": config["sampling_sweep"]["fixed_rank"],
        "seeds": config["seeds"], "entries": specifications(config)}
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def count_jsonl_rows(path, minimum):

    count = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for _ in stream:
            count += 1
            if count >= minimum:
                return count
    return count


def command_for(entry, config, train_jsonl, resume=False):
    t = config["training"]

    argv = [sys.executable, "-u", "experiments/s/suite_train.py",
        "--suite", "table2", "--task", "code", "--method", "e2lora",
        "--model-key", config["model_key"], "--seed", str(entry["seed"]),
        "--data-seed", str(config["data_seed"]), "--validation-seed", str(config["validation_seed"]),
        "--output-dir", entry["output_dir"], "--epochs", str(t["epochs"]),
        "--max-seq-length", str(t["max_seq_length"]), "--max-new-tokens", str(t["max_new_tokens"]),
        "--per-device-batch-size", str(t["per_device_batch_size"]),
        "--gradient-accumulation-steps", str(t["gradient_accumulation_steps"]),
        "--gradient-examples", str(entry["gradient_examples"]),
        "--e2-base-rank", str(entry["rank"]), "--learning-rate", str(t["learning_rate"]),
        "--weight-decay", str(t["weight_decay"]), "--warmup-ratio", str(t["warmup_ratio"]),
        "--alpha", str(t["alpha"]), "--train-jsonl", str(Path(train_jsonl).resolve()),
        "--code-only-source-confirmed", "--allow-unpinned-datasets", "--do-final-eval"]

    if t["gradient_checkpointing"]:
        argv.append("--gradient-checkpointing")

    if resume:
        argv.append("--resume")

    return argv


def run_one(index, train_jsonl, confirm_source, allow_unpinned):

    config = load_config()
    plan = write_plan(config)
    entries = plan["entries"]

    if not 0 <= index < len(entries):
        raise IndexError(f"Index must be between 0 and {len(entries)-1}")
    if not confirm_source:
        raise ValueError("Pass --confirm-code-only-source after verifying this file contains code-only training rows")
    if not allow_unpinned:
        raise ValueError("Pass --allow-unpinned-humaneval after reviewing the recorded dataset revision caveat")
    source = Path(train_jsonl).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Code-Feedback JSONL not found: {source}")
    minimum = config["data"]["train_count"] + config["data"]["validation_count"]
    if count_jsonl_rows(source, minimum) < minimum:
        raise ValueError(f"Need at least {minimum} usable code-only rows; do not pad the public snapshot")

    entry = entries[index]
    output = ROOT / entry["output_dir"]
    result = output / "result.json"

    if result.is_file():
        saved = json.loads(result.read_text(encoding="utf-8"))
        if saved.get("epochs_completed", 0) >= config["training"]["epochs"]:
            print(f"Run already has a completed training record: {result}")
            return

    checkpoint = output / "checkpoint_last.pt"
    if output.exists() and any(output.iterdir()) and not checkpoint.is_file():
        raise FileExistsError(f"Output directory is nonempty without a resumable checkpoint: {output}")

    argv = command_for(entry, config, source, resume=checkpoint.is_file())
    print(json.dumps({"run_id": entry["run_id"], "studies": entry["studies"],
        "argv": argv}, indent=2), flush=True)

    subprocess.run(argv, cwd=ROOT, check=True)


def read_result(entry):

    path = ROOT / entry["output_dir"] / "result.json"
    if not path.is_file():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def pass_at_one(result):

    final = result.get("final_evaluation") or {}
    harness = final.get("official_harness") or {}
    raw = harness.get("official_harness_result") if isinstance(harness, dict) else None

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    score = None

    if isinstance(raw, dict):
        score = raw.get("pass@1", raw.get("pass_at_1"))

    if score is None:
        score = final.get("pass@1", final.get("pass_at_1"))

    return None if score is None else 100.0 * float(score)


def summarize(output_dir=None):
    config = load_config()
    entries = specifications(config)
    out = Path(output_dir or (Path(config["output_root"]) / "summary"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        result = read_result(entry)
        final = (result or {}).get("final_evaluation") or {}
        rows.append({"run_id": entry["run_id"], "studies": ",".join(entry["studies"]),
            "rank": entry["rank"], "gradient_examples": entry["gradient_examples"],
            "seed": entry["seed"],
            "parameters": ((result or {}).get("parameter_counts") or {}).get("trainable_total_params"),
            "humaneval_pass_at_1_percent": pass_at_one(result) if result else None,
            "scoring_status": final.get("pass_at_1_status", "pending" if not result else "score pending"),
            "output_dir": entry["output_dir"]})
    if rows:
        with (out / "figure5_humaneval.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for study, xkey, title, filename in [
            ("figure5_rank", "rank", "HumanEval: E2LoRA rank sweep", "figure5_humaneval_rank.png"),
            ("figure5_sampling", "gradient_examples", "HumanEval: gradient-sampling sweep", "figure5_humaneval_sampling.png")]:
            selected = [r for r in rows if study in r["studies"] and r["humaneval_pass_at_1_percent"] is not None]
            if not selected:
                continue
            grouped = {}
            for row in selected:
                grouped.setdefault(row[xkey], []).append(row["humaneval_pass_at_1_percent"])
            xs = sorted(grouped)
            means = [sum(grouped[x])/len(grouped[x]) for x in xs]
            plt.figure()
            plt.plot(xs, means, marker="o")
            plt.xlabel("E2LoRA rank" if xkey == "rank" else "Gradient-sampling examples")
            plt.ylabel("HumanEval pass@1 (%)")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out / filename, dpi=180)
            plt.close()
    except ImportError:
        pass
    print(json.dumps({"runs": len(rows), "scored": sum(r["humaneval_pass_at_1_percent"] is not None for r in rows),
        "summary": str(out / "figure5_humaneval.csv")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help="write the 11-point rank/sampling plan; submit no GPU jobs")
    run = sub.add_parser("run", help="run one preplanned point inside an allocated GPU job")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--train-jsonl", required=True)
    run.add_argument("--confirm-code-only-source", action="store_true")
    run.add_argument("--allow-unpinned-humaneval", action="store_true")
    status = sub.add_parser("status", help="print status for each planned point")
    score = sub.add_parser("score", help="run the official scorer; use only in an isolated sandbox")
    score.add_argument("--index", type=int, required=True)
    score.add_argument("--timeout", type=int, default=3)
    score.add_argument("--confirm-isolated-sandbox", action="store_true")
    sub.add_parser("summarize", help="write CSV and pass@1 plots for completed scored runs")
    args = parser.parse_args()

    if args.action == "plan":
        plan = write_plan(load_config())
        print(json.dumps({"planned_runs": len(plan["entries"]), "gpu_jobs_submitted": 0,
            "plan": str(DEFAULT_PLAN)}, indent=2))

    elif args.action == "run":
        run_one(args.index, args.train_jsonl, args.confirm_code_only_source, args.allow_unpinned_humaneval)

    elif args.action == "status":
        config = load_config(); plan = write_plan(config)
        for i, entry in enumerate(plan["entries"]):
            result = read_result(entry)
            print(json.dumps({"index": i, "run_id": entry["run_id"],
                "training": "not started" if result is None else "training record exists",
                "pass_at_1": None if result is None else pass_at_one(result)}))

    elif args.action == "score":
        if not args.confirm_isolated_sandbox or os.environ.get("HUMANEVAL_ISOLATED_SANDBOX") != "1":
            raise ValueError("Run this command only in a hardened, network-isolated sandbox with HUMANEVAL_ISOLATED_SANDBOX=1 and --confirm-isolated-sandbox")
        config = load_config(); entries = specifications(config)
        if not 0 <= args.index < len(entries):
            raise IndexError("Run index is outside the compiled plan")
        output = ROOT / entries[args.index]["output_dir"]
        samples = output / "final-evaluation" / "samples.jsonl"
        if not samples.is_file():
            raise FileNotFoundError(f"Generated HumanEval samples not found: {samples}")
        sys.path.insert(0, str(ROOT / "src"))
        from e2lora.tasks.code_metrics import run_official_humaneval
        score_result = run_official_humaneval(samples, timeout=args.timeout)
        result_path = output / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.setdefault("final_evaluation", {})["official_harness"] = score_result
        result["final_evaluation"].pop("pass_at_1_status", None)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        eval_path = output / "final-evaluation" / "result.json"
        if eval_path.is_file():
            eval_result = json.loads(eval_path.read_text(encoding="utf-8"))
            eval_result["official_harness"] = score_result
            eval_result.pop("pass_at_1_status", None)
            eval_path.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
        print(json.dumps(score_result, indent=2))

    else:
        summarize()


if __name__ == "__main__":
    main()
