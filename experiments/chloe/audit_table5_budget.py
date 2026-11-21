"""Static Table 5 audit: cached LSS/HRA statistics and actual meta model counts.

No weights, gradients, optimizer, GPU, or training are loaded/run. Existing
outputs are protected. Cached length-256 Smoke groups are not formal results.
"""
import argparse
import json
import math
from pathlib import Path

from common import digest, file_sha256, offline, read_json, resolve
from e2lora.adapters.adaptive import allocate_ranks, greedy_intervals

PAPER_URL = "https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf"
FAMILIES = ("q_proj", "k_proj", "v_proj", "o_proj")
SOURCE_PLAN = "outputs/chloe/smoke/math-e2lora-r8-seed42/prepared-plan.json"


def literal_intervals(matrix):
    """Independent transcription of Algorithm 2, using zero-based half-open bounds."""
    n = len(matrix)
    thresholds = [math.fsum(row[j] for j in range(n) if j != i) / (n - 1)
                  for i, row in enumerate(matrix)]
    boundaries = [0]
    while boundaries[-1] < n:
        start = boundaries[-1]
        rejected = next((j for j in range(start + 1, n)
                         if matrix[j][start] < thresholds[start]), n)
        boundaries.append(rejected)
    return list(zip(boundaries, boundaries[1:])), thresholds


def literal_allocation(entropies, intervals, base_rank):
    """Equations 5-7, independently evaluated; nearest integer uses Python ties-to-even."""
    scores = [max(entropies[first:last]) for first, last in intervals]
    total = math.fsum(scores)
    if abs(total) < 1e-12:
        raise ValueError("Undefined allocation for zero interval-entropy sum.")
    quotas = [score / total * len(entropies) * base_rank for score in scores]
    ranks = [round(value) for value in quotas]
    if min(ranks) < 1:
        raise ValueError("Literal allocation has a nonpositive rank.")
    return scores, quotas, ranks


def cached_plan(smoke, base_rank):
    """Recompute both algorithms; reject stale cache intervals/ranks, without resampling."""
    plan, details = [], {}
    for family in FAMILIES:
        cached = smoke["diagnostics"][family]
        names, entropy, matrix = cached["layers"], cached["entropy"], cached["rmi"]
        expected_names = [f"model.layers.{i}.self_attn.{family}" for i in range(32)]
        if names != expected_names:
            raise ValueError("This audit requires all 32 ordered Llama projection layers.")
        intervals, thresholds = literal_intervals(matrix)
        implemented_intervals, implemented_thresholds = greedy_intervals(matrix)
        if intervals != implemented_intervals or intervals != [tuple(x) for x in cached["intervals"]]:
            raise ValueError("Cached LSS intervals differ from Algorithm 2 or current code.")
        if any(not math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)
               for a, b in zip(thresholds, implemented_thresholds)):
            raise ValueError("LSS thresholds differ from the independent evaluation.")
        scores, quotas, ranks = literal_allocation(entropy, intervals, base_rank)
        if ranks != allocate_ranks(entropy, intervals, base_rank, rank_policy="paper"):
            raise ValueError("Current HRA differs from the independent formula evaluation.")
        if base_rank == 8 and ranks != cached["ranks"]:
            raise ValueError("Stored Smoke ranks changed.")
        for (first, last), rank in zip(intervals, ranks):
            plan.append(dict(family=family, layers=names[first:last], rank=rank))
        details[family] = dict(intervals=intervals, thresholds=thresholds,
                               interval_entropies=scores, unrounded_ranks=quotas,
                               allocated_ranks=ranks, allocated_rank_sum=sum(ranks),
                               nominal_rank_sum=32 * base_rank,
                               rank_rounding_delta=sum(ranks) - 32 * base_rank,
                               all_interval_entropies_negative=all(value < 0 for value in scores))
    if base_rank == 8 and plan != smoke["plan"]:
        raise ValueError("Independently reconstructed plan differs from the source Smoke plan.")
    return plan, details


def installed_counts(model, plan):
    """Inspect Parameter identities and actual shapes, never inferring counts from labels."""
    from common import parameter_report

    shared_ids, rows = set(), []
    for group in plan:
        layers = [model.get_submodule(name) for name in group["layers"]]
        first = layers[0]
        if not all(layer.down is first.down and layer.up is first.up for layer in layers):
            raise ValueError("A/B are not actually shared within the declared interval.")
        a, b = first.down.weight, first.up.weight
        if shared_ids.intersection({id(a), id(b)}):
            raise ValueError("Unexpected cross-interval/family sharing.")
        shared_ids.update((id(a), id(b)))
        rank, width, output = group["rank"], first.base.in_features, first.base.out_features
        if list(a.shape) != [rank, width] or list(b.shape) != [output, rank]:
            raise ValueError("Installed factor shapes differ from the allocated rank.")
        if not a.requires_grad or not b.requires_grad or first.up.in_features != rank:
            raise ValueError("Unexpected freezing or sliced rank.")
        count = a.numel() + b.numel()
        rows.append(dict(family=group["family"], targets=group["layers"],
                         layer_indices=[int(name.split(".")[2]) for name in group["layers"]],
                         rank=rank, active_forward_rank=first.up.in_features,
                         A_shape=list(a.shape), B_shape=list(b.shape),
                         A_trainable_elements=a.numel(), B_trainable_elements=b.numel(),
                         physical_unique_trainable_parameters=count,
                         active_trainable_coordinate_parameters=count,
                         frozen_adapter_elements=0, actual_shared_parameter_identity_verified=True,
                         cross_group_parameters_distinct_verified=True,
                         A_dtype=str(a.dtype), B_dtype=str(b.dtype)))
    actual = {id(p): p for p in model.parameters() if p.requires_grad}
    if set(actual) != shared_ids:
        raise ValueError("Unexpected trainable base/head parameters or missing adapters.")
    physical = sum(p.numel() for p in actual.values())
    logical = sum(len(row["targets"]) * row["physical_unique_trainable_parameters"] for row in rows)
    repeated = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False) if p.requires_grad)
    if physical != sum(row["physical_unique_trainable_parameters"] for row in rows) or repeated != logical:
        raise ValueError("Independent identity and layer-use counts do not reconcile.")
    reported = parameter_report(model, plan)
    if reported["total_trainable_parameters"] != physical:
        raise ValueError("Existing parameter_report disagrees with direct Parameter inspection.")
    return dict(physical_unique_trainable_parameters=physical,
                active_trainable_coordinate_parameters=physical,
                unique_trainable_tensors=len(actual), frozen_adapter_elements=0,
                logical_layer_uses_parameter_elements_not_unique=logical, groups=rows)


def build_audit():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from common import attach_trainable_plan

    offline()
    smoke = read_json(SOURCE_PLAN)
    config_path = "experiments/chloe/configs/smoke-e2lora-r8.json"
    config = read_json(config_path)
    if smoke["config_sha256"] != digest(config) or smoke["partial_rank_sharing_enabled"]:
        raise ValueError("Source plan/config identity or freezing policy changed.")
    model_config_path = resolve(config["model"]["path"]) / "config.json"
    architecture = AutoConfig.from_pretrained(model_config_path.parent, local_files_only=True)
    cases = {}
    for method, rank, displayed in [("lora", 4, 3410000), ("lora", 8, 6820000),
                                     ("e2lora", 8, 3610000), ("e2lora", 16, 7220000)]:
        if method == "e2lora":
            plan, hra = cached_plan(smoke, rank)
        else:
            plan = [dict(family=family, layers=[f"model.layers.{i}.self_attn.{family}"], rank=rank)
                    for family in FAMILIES for i in range(32)]
            hra = {}
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(architecture, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa")
            attach_trainable_plan(model, plan, alpha=config["adapter"]["alpha"])
        measured = installed_counts(model, plan)
        if any(p.device.type != "meta" for p in model.parameters()):
            raise ValueError("The architecture audit must remain on meta.")
        summaries = {}
        for family in FAMILIES:
            rows = [row for row in measured["groups"] if row["family"] == family]
            coefficient = rows[0]["A_shape"][1] + rows[0]["B_shape"][0]
            nominal = 32 * rank * coefficient
            summaries[family] = dict(hra.get(family, {}), group_count=len(rows),
                allocated_rank_sum=sum(row["rank"] for row in rows),
                parameters_per_rank=coefficient,
                physical_unique_trainable_parameters=sum(row["physical_unique_trainable_parameters"] for row in rows),
                active_trainable_coordinate_parameters=sum(row["active_trainable_coordinate_parameters"] for row in rows),
                same_nominal_rank_independent_lora_parameters=nominal,
                parameter_rounding_delta=sum(row["physical_unique_trainable_parameters"] for row in rows) - nominal,
                paper_family_parameter_target=None)
        nominal_total = sum(row["same_nominal_rank_independent_lora_parameters"] for row in summaries.values())
        actual = measured["physical_unique_trainable_parameters"]
        measured.update(families=summaries, displayed_paper_center_parameters=displayed,
            displayed_paper_center_is_not_an_exact_author_count=True,
            conventional_two_decimal_M_display_bin=[displayed - 5000, displayed + 5000],
            actual_minus_displayed_center=actual - displayed,
            same_nominal_rank_independent_lora_parameters=nominal_total,
            parameter_rounding_delta=actual - nominal_total,
            maximum_rounding_only_absolute_delta=sum(row["parameters_per_rank"] * row["group_count"] / 2
                                                    for row in summaries.values()))
        cases[f"{method}_r{rank}"] = measured
        del model
    r8, r16 = cases["e2lora_r8"], cases["e2lora_r16"]
    if r8["physical_unique_trainable_parameters"] != smoke["parameters"]["total_trainable_parameters"]:
        raise ValueError("Meta measurement disagrees with the stored Stage A count.")
    stage_b = read_json("outputs/chloe/smoke/math-e2lora-r8-seed42/result.json")
    if r8["physical_unique_trainable_parameters"] != stage_b["parameters"]["total_trainable_parameters"]:
        raise ValueError("Meta measurement disagrees with the stored Stage B result.")
    sources = [SOURCE_PLAN, config_path, str(model_config_path),
               "outputs/chloe/smoke/math-e2lora-r8-seed42/result.json",
               "src/e2lora/adapters/adaptive.py", "src/e2lora/adapters/preparation.py",
               "src/e2lora/adapters/lora_layers.py", "src/e2lora/adapters/partial_rank.py",
               "experiments/chloe/common.py", "experiments/chloe/formal_runner.py",
               "experiments/chloe/train.py", "experiments/chloe/audit_table5_budget.py"]
    return dict(run_type="static_cpu_meta_table5_budget_audit",
        status="AUDIT_COMPLETED_TABLE5_PARAMETER_RULE_UNRESOLVED", eligible_for_formal_tables=False,
        base_weights_loaded=False, gpu_operations=False, gradient_sampling_run=False,
        optimizer_updates=0, gpu_jobs_submitted=0, formal_experiments_started=0,
        algorithm_hra_or_freezing_rules_changed=False,
        evidence_scope="Cached length-256 two-example Smoke diagnostics, not author or formal 2048 groups",
        source_plan=SOURCE_PLAN, paper_url=PAPER_URL, paper_pages=[6, 9, 27],
        source_sha256={path: file_sha256(resolve(path)) for path in sources},
        cases=cases,
        maximum_rounding_delta_for_any_32_layer_partition=32 / 2 * (8192 + 5120 + 5120 + 8192),
        count_of_unique_trainable_tensors_matches_stored_stage_A_B=r8["unique_trainable_tensors"] == 48,
        existing_active139_candidate_parameters=139 * (8192 + 5120 + 5120 + 8192),
        existing_active139_candidate_minus_table5_r8_display_center=139 * 26624 - 3610000,
        A_only_trainable_counterfactual_r8=sum(row["A_trainable_elements"] for row in r8["groups"]),
        B_only_trainable_counterfactual_r8=sum(row["B_trainable_elements"] for row in r8["groups"]),
        requires_author_clarification=["Meaning of Table 5 rank label versus Algorithm 3 r_vanilla",
            "Total rank budget and normalization domain across projection families",
            "Exact per-family intervals, allocated/used ranks and A/B sharing map",
            "Any max-rank storage, slicing, masking, freezing and optimizer counting policy",
            "Exact unique trainable counts, target modules, seeds and reduction across runs",
            "Treatment of negative/zero entropies and rounding ties"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/chloe/report/table5-budget-audit-v1.json")
    args = parser.parse_args()
    destination = resolve(args.output)
    if destination.exists():
        raise FileExistsError("Existing audit is protected; choose a fresh output name.")
    report = build_audit()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(destination), "status": report["status"],
        "parameters": {name: row["physical_unique_trainable_parameters"] for name, row in report["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
