"""Strict, CPU-only member-4 protocol validation and configuration comparison."""
import argparse
import json
import math

from common import digest, file_sha256, read_json, resolve
from experiment_scope import active_tasks


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def compare_configs(left, right, *, allowed=()):
    """Exact field comparison; missing fields differ from explicit nulls."""
    a, b = flatten(left), flatten(right)
    return [{"field": key, "left": a.get(key), "right": b.get(key),
             "left_present": key in a, "right_present": key in b}
            for key in sorted(a.keys() | b.keys())
            if (key not in a or key not in b or a[key] != b[key]) and key not in allowed]


def protocol_issues(protocol, task, *, verify_files=True):
    if task not in active_tasks():
        return [f"task {task} excluded by group-approved Math-only scope"]
    template = read_json("experiments/chloe/configs/formal-protocol.pending.json")
    # Template declares required fields. Unknown additional fields are compared too.
    selected = {k: protocol.get(k) for k in ["confirmation", "model", "training", "adapter", "sampling", "mechanisms"]}
    required = {k: template[k] for k in selected}
    selected["task"] = protocol.get("tasks", {}).get(task, {})
    required["task"] = template["tasks"][task]
    fields = flatten(required)

    def get_path(value, path):
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    nullable = {"task.training.max_steps"}
    if task == "math":
        # GSM8K numeric scoring executes no generated programs.
        nullable.update({"task.evaluation.timeout_seconds", "task.evaluation.isolation"})
    if protocol.get("sampling", {}).get("seed_policy") == "training_seed":
        nullable.add("sampling.fixed_seed")
    if (selected["task"] or {}).get("generation", {}).get("do_sample") is False:
        nullable.update(f"task.generation.{key}" for key in ["temperature", "top_p", "top_k"])
    issues = [f"unconfirmed:{key}" for key in fields
              if key not in nullable and (get_path(selected, key) is None or get_path(selected, key) == "")]
    if protocol.get("status") != "confirmed":
        issues.append("protocol.status must be confirmed by member 4 and member 5")
    seeds = protocol.get("training", {}).get("seeds")
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(type(x) is not int for x in seeds) or len(set(seeds)) != len(seeds)):
        issues.append("one to three distinct confirmed integer training seeds are required; final aggregate needs three")
    known = read_json("references/ablation-setup.json")
    generation = read_json("references/text-generation-setup.json")
    for key in ["model", "training", "rank_policy", "independent_runs"]:
        if key in {"model", "training"}:
            if compare_configs(known[key], generation[key]):
                issues.append(f"team references disagree:{key}")
        elif known.get(key) != generation.get(key):
            issues.append(f"team references disagree:{key}")
    # Pin already agreed facts; changes need an updated shared reference, not a local override.
    if protocol.get("model", {}).get("repo") != known["model"]["repo"] or protocol.get("model", {}).get("revision") != known["model"]["revision"]:
        issues.append("model repo/revision differs from team reference")
    training = protocol.get("training", {})
    for key in ["checkpoint_every", "log_every"]:
        if type(training.get(key)) is not int or training[key] < 1:
            issues.append(f"training.{key} must be a positive integer")
    if training.get("max_grad_norm") is not None and (not isinstance(training["max_grad_norm"], (float, int)) or not math.isfinite(training["max_grad_norm"]) or training["max_grad_norm"] <= 0):
        issues.append("invalid max_grad_norm")
    for key in ["optimizer", "betas", "weight_decay", "learning_rate", "precision", "scheduler", "warmup_ratio"]:
        if training.get(key) != known["training"][key]:
            issues.append(f"training.{key} differs from team reference")
    if training.get("global_batch_size") != known["training"]["batch_size"]:
        issues.append("global batch size differs from team reference")
    if protocol.get("mechanisms", {}).get("paper_parameter_resolution_confirmed") is not True:
        issues.append("paper parameter-count/partial-rank ambiguity is unresolved")
    if protocol.get("mechanisms", {}).get("rank_policy") != "paper":
        issues.append("rank_policy must remain paper")
    mechanism = protocol.get("mechanisms", {})
    for key in ["table4_base_rank", "main_sampling_steps", "figure5_rank_fixed_sampling_steps", "figure5_steps_fixed_rank"]:
        if type(mechanism.get(key)) is not int or mechanism[key] < 1:
            issues.append(f"mechanisms.{key} must be a confirmed positive integer")
    gap = mechanism.get("table5_max_relative_budget_gap")
    if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0:
        issues.append("Table 5 relative budget tolerance must be a finite nonnegative number")
    adapter = protocol.get("adapter", {})
    if adapter.get("alpha") is not None and (not isinstance(adapter["alpha"], (int, float)) or not math.isfinite(adapter["alpha"]) or adapter["alpha"] <= 0):
        issues.append("invalid alpha")
    supported = {"loss": "response_only_token_weighted", "packing": False,
                 "truncation": "keep_BOS_and_prompt_tail_then_truncate_response_right",
                 "dropout": 0, "adapter_dtype": "float32", "scaling": "alpha_over_group_rank",
                 "initialization": "kaiming_A_zero_trainable_B"}
    for key, value in supported.items():
        if adapter.get(key) is not None and adapter.get(key) != value:
            issues.append(f"unsupported adapter.{key}; implement the confirmed policy before running")
    task_protocol = protocol.get("tasks", {}).get(task, {})
    if task_protocol.get("smoke_validation", {}).get("passed") is not True:
        issues.append(f"{task} GPU smoke has not been recorded as passed")
    if task == "math" and task_protocol.get("formal_preflight", {}).get("passed") is not True:
        issues.append("Math formal-length/adapter/evaluator preflight has not passed; short Smoke alone is insufficient")
    if task == "math" and task_protocol.get("manifest", {}).get("count") != 100000:
        issues.append("Math formal manifest must have exactly 100000 examples")
    if task_protocol.get("evaluation", {}).get("metric") != ("accuracy" if task == "math" else "pass@1"):
        issues.append("evaluation metric does not match the task")
    if task == "math" and task_protocol.get("evaluation", {}).get("count") != 1319:
        issues.append("Math formal evaluation must cover exactly 1319 examples")
    sampling = protocol.get("sampling", {})
    for key in ["batch_size", "micro_batch_size"]:
        if type(sampling.get(key)) is not int or sampling[key] < 1:
            issues.append(f"sampling.{key} must be a confirmed positive integer")
    if sampling.get("order") not in {None, "manifest_shuffle_prefix"} or sampling.get("loss_weighting") not in {None, "supervised_response_tokens"}:
        issues.append("unsupported sampling order or loss weighting")
    if sampling.get("seed_policy") not in {None, "training_seed", "fixed"}:
        issues.append("unsupported sampling seed policy")
    if sampling.get("seed_policy") == "fixed" and type(sampling.get("fixed_seed")) is not int:
        issues.append("fixed sampling needs an integer seed")
    train = task_protocol.get("training", {})
    for key in ["epochs", "max_length", "max_prompt_length", "micro_batch_size", "gradient_accumulation"]:
        if type(train.get(key)) is not int or train[key] <= 0:
            issues.append(f"task.training.{key} must be a confirmed positive integer")
    if all(type(train.get(k)) is int for k in ["micro_batch_size", "gradient_accumulation"]):
        if train["micro_batch_size"] * train["gradient_accumulation"] != training.get("global_batch_size"):
            issues.append("single-GPU microbatch * accumulation must equal global batch")
    if all(type(train.get(k)) is int for k in ["max_length", "max_prompt_length"]):
        if not 2 <= train["max_prompt_length"] < train["max_length"]:
            issues.append("prompt length must leave response tokens")
    if train.get("max_steps") is not None and (type(train["max_steps"]) is not int or train["max_steps"] < 1):
        issues.append("max_steps must be null or a positive integer")
    if task_protocol.get("prompt", {}).get("eos_policy") not in {None, "append_eos_before_right_truncation"}:
        issues.append("unsupported EOS policy")
    from e2lora.adapters.similarity import CONVENTIONS
    conventions = protocol.get("mechanisms", {}).get("metric_conventions", {})
    for metric, convention in CONVENTIONS.items():
        if conventions.get(metric) is not None and conventions[metric] != convention:
            issues.append(f"unsupported confirmed convention:{metric}")
    for method, policy in protocol.get("mechanisms", {}).get("adapter_policies", {}).items():
        if policy is None:
            continue
        if not isinstance(policy, dict) or policy.get("mode") not in {"all_trainable", "partial_group_ab"}:
            issues.append(f"unsupported adapter policy:{method}")
        elif policy["mode"] == "partial_group_ab" and (not isinstance(policy.get("active_rank_per_projection_by_base_rank"), dict)
                                                      or not isinstance(policy.get("frozen_b_init_scale"), (int, float))):
            issues.append(f"incomplete partial-rank policy:{method}")
        elif method == "lora" and policy["mode"] != "all_trainable":
            issues.append("vanilla LoRA baseline must train both factors at full rank")
    policies = protocol.get("mechanisms", {}).get("adapter_policies", {})
    controls = [policy for method, policy in policies.items() if method != "lora" and policy is not None]
    if controls and any(policy != controls[0] for policy in controls[1:]):
        issues.append("Table 4 mechanisms must use the same freezing/initialization policy")
    argv = task_protocol.get("evaluator", {}).get("argv")
    if argv is not None and (not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv)):
        issues.append("evaluator.argv must be an explicit nonempty argument list")
    elif argv:
        command_text = " ".join(argv)
        for field in ["script", "config", "checkpoint", "output"]:
            if "{" + field + "}" not in command_text:
                issues.append(f"evaluator.argv must use {{{field}}}")
        try:
            [item.format_map({k: k for k in ["python", "script", "config", "checkpoint", "output"]}) for item in argv]
        except (KeyError, ValueError):
            issues.append("unsupported evaluator argument placeholder")
    if task_protocol.get("data", {}).get("format") not in {None, "normalized_query_response_json"}:
        issues.append("unsupported data format; member 4 must supply normalized query/response data or an adapter")
    if verify_files:
        artifacts = {label: task_protocol.get(label, {}) for label in ["data", "manifest", "evaluator"]}
        evaluation = task_protocol.get("evaluation", {})
        for label in ["task_ids", "environment_lock"]:
            artifacts[label] = {"path": evaluation.get(label + "_path"), "sha256": evaluation.get(label + "_sha256")}
        artifacts["smoke_validation"] = {"path": task_protocol.get("smoke_validation", {}).get("report_path"),
                                         "sha256": task_protocol.get("smoke_validation", {}).get("report_sha256")}
        artifacts["formal_preflight"] = {"path": task_protocol.get("formal_preflight", {}).get("report_path"),
                                         "sha256": task_protocol.get("formal_preflight", {}).get("report_sha256")}
        for label, artifact in artifacts.items():
            path, expected = artifact.get("path"), artifact.get("sha256")
            if path is None or expected is None:
                continue
            path = resolve(path)
            if not path.is_file():
                issues.append(f"missing artifact:{label}:{path}")
            elif file_sha256(path) != expected:
                issues.append(f"artifact digest mismatch:{label}")
        if not protocol.get("model", {}).get("path") or not resolve(protocol["model"]["path"]).is_dir():
            issues.append("local model directory is missing")
    return sorted(set(issues))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="experiments/chloe/configs/formal-protocol.pending.json")
    parser.add_argument("--task", choices=["math", "code"], default="math")
    parser.add_argument("--compare", help="Member 4's copy of the same protocol schema")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    issues = protocol_issues(protocol, args.task)
    differences = compare_configs(protocol, read_json(args.compare)) if args.compare else []
    print(json.dumps({"task": args.task, "protocol_sha256": digest(protocol), "issues": issues,
                      "differences": differences, "ready": not issues and not differences}, indent=2))
    return 1 if issues or differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
