"""Build and audit the complete formal matrix. This CLI never submits GPU jobs."""
import argparse
from copy import deepcopy
import csv
import json

from alignment import protocol_issues
from common import ROOT, digest, read_json, write_json
from experiment_scope import active_tasks, current_scope, require_active_task

CONFIG_ROOT = ROOT / "experiments/chloe/configs/formal"
PROTOCOL = "experiments/chloe/configs/formal-protocol.pending.json"
METHODS = {
    "lora": (False, False, "rmi"),
    "sharing_only": (True, False, "rmi"),
    "rank_allocation_only": (False, True, "rmi"),
    "cosine_with_rank_allocation": (True, True, "cosine"),
    "l1_with_rank_allocation": (True, True, "l1"),
    "l2_with_rank_allocation": (True, True, "l2"),
    "kl_with_rank_allocation": (True, True, "kl"),
    "e2lora": (True, True, "rmi"),
}


def specifications():
    setup = read_json("references/ablation-setup.json")
    scope = current_scope()
    specs = []

    def add(run_id, study, task, method, rank=None, steps=None, **extra):
        lss, hra, metric = METHODS[method]
        specs.append(dict(schema_version=2, run_id=run_id, study=study, task=task,
                          method=method, run_type="formal", protocol_file=PROTOCOL,
                          base_rank=rank, sampling_steps=steps, lss=lss, hra=hra,
                          similarity=metric, rank_policy="paper", grouping="greedy",
                          independent_runs=scope["planned_independent_runs"],
                          training_seeds=scope["known_member4_table2_seeds"],
                          eligible_for_formal_tables=False, status="BLOCKED_PROTOCOL_ALIGNMENT",
                          **extra))

    for method in METHODS:
        add(f"t4-{method}", "table4", "math", method)
    for pair in setup["parameter_budget_comparisons"]:
        budget = f"lora{pair['lora_rank']}-e2lora{pair['e2lora_rank']}"
        for method, rank in [("lora", pair["lora_rank"]), ("e2lora", pair["e2lora_rank"])]:
            add(f"t5-{method}-r{rank}", "table5", "math", method, rank, budget_pair=budget)
    for task in active_tasks():
        for rank in setup["rank_values"]:
            add(f"f5-rank-{task}-r{rank}", "figure5_rank", task, "e2lora", rank)
        for steps in setup["gradient_sampling_steps"]:
            add(f"f5-steps-{task}-n{steps}", "figure5_steps", task, "e2lora", steps=steps)
    return specs


def materialize(spec, protocol, repeat):
    """Resolve only supplied confirmed settings. No seeds/ranks/steps are guessed."""
    require_active_task(spec["task"])
    issues = protocol_issues(protocol, spec["task"])
    if issues:
        raise ValueError("Protocol blocked: " + "; ".join(issues))
    if type(repeat) is not int or not 0 <= repeat < 3:
        raise ValueError("repeat must be 0, 1 or 2")
    if repeat >= len(protocol["training"]["seeds"]):
        raise ValueError("This repeat's training seed has not been confirmed.")
    result = deepcopy(spec)
    mechanism = protocol["mechanisms"]
    rank = spec["base_rank"]
    if rank is None:
        rank = (mechanism["figure5_steps_fixed_rank"] if spec["study"] == "figure5_steps"
                else mechanism["table4_base_rank"])
    steps = spec["sampling_steps"]
    if steps is None:
        steps = (mechanism["figure5_rank_fixed_sampling_steps"] if spec["study"] == "figure5_rank"
                 else mechanism["main_sampling_steps"])
    seed = protocol["training"]["seeds"][repeat]
    protocol_sha256 = digest(protocol)
    result.update(base_rank=rank, sampling_steps=steps, seed=seed, repeat=repeat,
                  protocol_sha256=protocol_sha256,
                  output_dir=f"outputs/chloe/formal/{spec['task']}/protocol-{protocol_sha256[:12]}/{spec['run_id']}/seed{seed}",
                  status="CONFIGURED_REQUIRES_GPU_AUTHORIZATION", resolved_protocol=deepcopy(protocol))
    if spec["method"] == "lora":
        result["sampling_steps"] = 0
    for key in ["base_rank", "sampling_steps"]:
        if type(result[key]) is not int or result[key] < (0 if key == "sampling_steps" else 1):
            raise ValueError(f"Invalid confirmed {key}")
    return result


def generate():
    specs = specifications()
    rows = []
    for spec in specs:
        path = CONFIG_ROOT / (spec["run_id"] + ".json")
        write_json(path, spec)
        for repeat in range(3):
            seeds = spec["training_seeds"]
            seed = seeds[repeat] if repeat < len(seeds) else "PENDING_MEMBER4"
            rows.append(dict(run_id=spec["run_id"], repeat=repeat, seed=seed,
                             task=spec["task"], study=spec["study"], method=spec["method"],
                             base_rank=spec["base_rank"], sampling_steps=spec["sampling_steps"],
                             config=str(path.relative_to(ROOT)), status=spec["status"],
                             output_dir=f"outputs/chloe/formal/{spec['task']}/protocol-<PENDING>/{spec['run_id']}/seed{seed}",
                             gpu_count=1, eligible_for_formal_tables=False))
    with (ROOT / "experiments/chloe/formal-matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overview = [dict(run_id=s["run_id"], study=s["study"], task=s["task"], method=s["method"],
                     base_rank=s["base_rank"], sampling_steps=s["sampling_steps"],
                     known_training_seeds="42", planned_repeats=s["independent_runs"], gpu_count=1,
                     config=f"experiments/chloe/configs/formal/{s['run_id']}.json", status=s["status"])
                for s in specs]
    overview.append(dict(run_id="table10-dp-greedy", study="table10", task="cpu_grouping", method="dp/greedy",
                         base_rank=None, sampling_steps=None, known_training_seeds="not_applicable",
                         planned_repeats=0, gpu_count=0, config="", status="CPU_METHOD_DELIVERY_TASK_SCORES_PENDING"))
    with (ROOT / "experiments/chloe/experiment-matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(overview[0]))
        writer.writeheader()
        writer.writerows(overview)
    # Keep old Code specifications for reference, but make their inactive state explicit.
    retired = []
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        spec = read_json(path)
        if spec.get("task") not in active_tasks():
            spec.update(status="EXCLUDED_BY_GROUP_APPROVAL_MATH_ONLY", independent_runs=0,
                        training_seeds=[], eligible_for_formal_tables=False,
                        exclusion_scope="math-only-2026-11-18")
            write_json(path, spec)
            retired.append(str(path.relative_to(ROOT)))
    summary = {"scope": current_scope(), "configurations": len(specs),
               "planned_runs": len(rows), "known_seed_rows": sum(row["seed"] != "PENDING_MEMBER4" for row in rows),
               "code_gpu_runs": 0, "inactive_retained_configs": retired, "gpu_jobs_submitted": 0}
    write_json(ROOT / "experiments/chloe/report/math-only-matrix-summary.json", summary)
    return summary


def compile_batch(protocol, *, task, directory):
    from common import output_path
    directory = output_path(directory)
    if task != "all":
        require_active_task(task)
    selected = [s for s in specifications() if task == "all" or s["task"] == task]
    # Resolve every row before writing anything: a blocked task produces no partial batch.
    # Compile only supplied seeds. Three repetitions are still required for the final aggregate.
    repeats = protocol.get("training", {}).get("seeds")
    if not isinstance(repeats, list) or not 1 <= len(repeats) <= 3:
        raise ValueError("Supply one to three confirmed training seeds before compilation.")
    runs = [materialize(s, protocol, repeat) for s in selected for repeat in range(len(repeats))]
    entries = []
    for run in runs:
        path = directory / f"{run['run_id']}-seed{run['seed']}.json"
        if path.exists() and read_json(path) != run:
            raise ValueError(f"Existing compiled config differs: {path}")
        entries.append({"index": len(entries), "config": str(path), "run_id": run["run_id"], "seed": run["seed"]})
    for run, entry in zip(runs, entries):
        write_json(entry["config"], run)
    manifest = {"schema_version": 1, "run_type": "formal_batch_plan", "gpu_jobs_submitted": 0,
                "count": len(entries), "entries": entries}
    write_json(directory / "batch.json", manifest)
    return {"batch_manifest": str(directory / "batch.json"), "runs": len(entries), "gpu_jobs_submitted": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "preflight", "compile", "compile-batch"])
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--config")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", default="outputs/chloe/formal-preflight.json")
    parser.add_argument("--task", choices=["math", "code", "all"], default="all")
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(generate(), indent=2))
        return 0
    protocol = read_json(args.protocol)
    if args.command == "compile-batch":
        print(json.dumps(compile_batch(protocol, task=args.task, directory=args.output), indent=2))
        return 0
    if args.command == "compile":
        if not args.config:
            parser.error("compile needs --config")
        result = materialize(read_json(args.config), protocol, args.repeat)
        from common import output_path
        write_json(output_path(args.output), result)
        print(args.output)
        return 0
    reports = {task: protocol_issues(protocol, task) for task in active_tasks()}
    result = {"gpu_jobs_submitted": 0, "issues": reports,
              "ready_tasks": [task for task, issues in reports.items() if not issues],
              "planned_runs": sum(s["independent_runs"] for s in specifications()),
              "excluded_tasks": current_scope()["excluded_tasks"], "formal_results": 0}
    from common import output_path
    write_json(output_path(args.output), result)
    print(json.dumps({**result, "issues": {t: len(v) for t, v in reports.items()}, "report": args.output}, indent=2))
    return 1 if any(reports.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
