"""Audit a compiled batch; reject Smoke/partial results and check Table 5 budgets."""
import argparse
import json
import math
import statistics

from common import digest, file_sha256, output_path, read_json, write_json
from formal_runner import validate_resolved


def audit_run(spec):
    issues = validate_resolved(spec)
    directory = output_path(spec["output_dir"])
    paths = [directory / name for name in ["result.json", "evaluation.json", "checkpoint.pt"]]
    if not all(path.is_file() for path in paths):
        return {"run_id": spec["run_id"], "seed": spec["seed"], "eligible": False,
                "issues": issues + ["missing training, checkpoint or evaluation artifact"], "score": None}
    result, evaluation = read_json(paths[0]), read_json(paths[1])
    checkpoint_sha256 = file_sha256(paths[2])
    expected = spec["resolved_protocol"]["tasks"][spec["task"]]["evaluation"]
    for name, value in [("result", result), ("evaluation", evaluation)]:
        if value.get("run_type") != "formal" or value.get("task") != spec["task"]:
            issues.append(f"{name}: Smoke or wrong task")
        if value.get("protocol_sha256") != spec["protocol_sha256"]:
            issues.append(f"{name}: protocol mismatch")
        if value.get("checkpoint_sha256") != checkpoint_sha256:
            issues.append(f"{name}: checkpoint mismatch")
    if result.get("run_id") != spec["run_id"] or result.get("seed") != spec["seed"]:
        issues.append("run ID or training seed mismatch")
    training = result.get("training", {})
    planned = training.get("planned_optimizer_steps")
    if (training.get("status") != "completed" or type(planned) is not int or planned < 1
            or training.get("progress", {}).get("global_step") != planned):
        issues.append("training budget incomplete")
    if evaluation.get("metric") != expected["metric"] or evaluation.get("evaluation_examples") != expected["count"] or evaluation.get("task_ids_sha256") != expected["task_ids_sha256"]:
        issues.append("evaluation metric/count/IDs mismatch")
    score = evaluation.get("score")
    if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
        issues.append("invalid evaluation score")
    return {"run_id": spec["run_id"], "seed": spec["seed"], "study": spec["study"], "task": spec["task"],
            "budget_pair": spec.get("budget_pair"), "method": spec["method"], "eligible": not issues,
            "issues": issues, "score": score if not issues else None,
            "trainable_adapter_parameters": result.get("parameters", {}).get("trainable_adapter_parameters")}


def audit_budget_pairs(rows, tolerance):
    if not isinstance(tolerance, (float, int)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Table 5 needs an explicit nonnegative relative budget tolerance.")
    pairs = {}
    for row in rows:
        if row.get("study") == "table5":
            pairs.setdefault((row.get("budget_pair"), row["seed"]), {})[row["method"]] = row
    checks = []
    for (pair, seed), methods in pairs.items():
        lora, e2 = [methods.get(method, {}).get("trainable_adapter_parameters") for method in ["lora", "e2lora"]]
        gap = abs(e2 - lora) / lora if isinstance(lora, int) and lora > 0 and isinstance(e2, int) else None
        matched = gap is not None and gap <= tolerance
        if not matched:
            for row in methods.values():
                row["eligible"] = False
                row["score"] = None
                row["issues"].append("Table 5 parameter budgets not verified as comparable")
        checks.append(dict(pair=pair, seed=seed, lora_parameters=lora, e2lora_parameters=e2,
                           relative_gap=gap, tolerance=tolerance, matched=matched))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", default="outputs/chloe/formal-summary.json")
    args = parser.parse_args()
    batch = read_json(args.batch)
    specs = [read_json(entry["config"]) for entry in batch["entries"]]
    if not specs or batch.get("count") != len(specs):
        raise ValueError("Empty or inconsistent compiled batch")
    if len({digest(spec["resolved_protocol"]) for spec in specs}) != 1:
        raise ValueError("Batch combines different protocols")
    if len({(spec["run_id"], spec["seed"]) for spec in specs}) != len(specs):
        raise ValueError("Batch contains duplicate run/seed entries")
    rows = [audit_run(spec) for spec in specs]
    tolerance = specs[0]["resolved_protocol"]["mechanisms"]["table5_max_relative_budget_gap"]
    budgets = audit_budget_pairs(rows, tolerance)
    groups = {}
    for row in rows:
        groups.setdefault(row["run_id"], []).append(row)
    aggregates = {}
    for run_id, group in groups.items():
        valid = len(group) == 3 and len({r["seed"] for r in group}) == 3 and all(r["eligible"] for r in group)
        values = [r["score"] for r in group] if valid else []
        aggregates[run_id] = dict(eligible_for_formal_tables=valid, completed_repeats=sum(r["eligible"] for r in group),
                                  mean=statistics.mean(values) if values else None,
                                  sample_std=statistics.stdev(values) if values else None)
    write_json(output_path(args.output), dict(rows=rows, table5_budget_checks=budgets, aggregates=aggregates))
    print(json.dumps({"audited_runs": len(rows), "eligible_experiments": sum(r["eligible_for_formal_tables"] for r in aggregates.values())}))


if __name__ == "__main__":
    main()
