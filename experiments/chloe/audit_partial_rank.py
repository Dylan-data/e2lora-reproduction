"""CPU meta-device parameter audit using actual Llama architecture and Smoke groups.

The 139-active-rank case is explicitly a team candidate, never a formal default.
No base weight tensors are loaded and no GPU or optimizer is used.
"""
import argparse
import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from common import offline, output_path, read_json, resolve, write_json
from e2lora.adapters.lora_layers import apply_plan
from e2lora.adapters.partial_rank import active_rank_plan, apply_partial_plan, count_adapters
from e2lora.adapters.adaptive import allocate_ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="outputs/chloe/smoke/math-e2lora-r8-seed42/prepared-plan.json")
    parser.add_argument("--output", default="outputs/chloe/cpu/partial-rank-audit.json")
    args = parser.parse_args()
    offline()
    smoke = read_json(args.plan)
    cfg = read_json("experiments/chloe/configs/smoke-e2lora-r8.json")
    architecture = AutoConfig.from_pretrained(resolve(cfg["model"]["path"]), local_files_only=True)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    results = {}

    def measure(label, plan, active=None, global_down=False):
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(architecture, dtype=torch.bfloat16, attn_implementation="sdpa")
            if active is None:
                apply_plan(model, plan, alpha=16)
            else:
                plan = active_rank_plan(plan, {p: active for p in targets})
                apply_partial_plan(model, plan, alpha=16, frozen_b_init_scale=.01, global_down=global_down)
        counts = count_adapters(model, plan)
        results[label] = {"parameters": counts, "allocated_ranks": {p: sum(g["rank"] for g in plan if g["family"] == p) for p in targets},
                          "active_ranks": {p: sum(g.get("trainable_rank", g["rank"]) for g in plan if g["family"] == p) for p in targets}}

    for rank in [4, 8, 16, 32, 64]:
        plan = [dict(family=p, layers=[f"model.layers.{i}.self_attn.{p}"], rank=rank)
                for p in targets for i in range(architecture.num_hidden_layers)]
        measure(f"lora-r{rank}", plan)
    measure("smoke_actual_groups_all_trainable", smoke["plan"])
    sharing = [dict(g, rank=8) for g in smoke["plan"]]
    measure("table4_sharing_only_r8_cached_smoke_groups_not_formal", sharing)
    hra_only = []
    for family, detail in smoke["diagnostics"].items():
        intervals = [(i, i + 1) for i in range(len(detail["layers"]))]
        ranks = allocate_ranks(detail["entropy"], intervals, 8, rank_policy="paper")
        hra_only.extend(dict(family=family, layers=[name], rank=rank)
                        for name, rank in zip(detail["layers"], ranks))
    measure("table4_hra_only_r8_cached_smoke_entropy_not_formal", hra_only)
    for base_rank in [4, 8, 16, 32, 64]:
        plan = []
        for family, detail in smoke["diagnostics"].items():
            ranks = allocate_ranks(detail["entropy"], detail["intervals"], base_rank, rank_policy="paper")
            plan.extend(dict(family=family, layers=detail["layers"][start:end], rank=rank)
                        for (start, end), rank in zip(detail["intervals"], ranks))
        measure(f"e2_r{base_rank}_cached_smoke_rmi_all_trainable_not_formal", plan)
    measure("candidate_active139_group_AB_not_paper_confirmed", smoke["plan"], active=139)
    measure("candidate_active139_global_frozen_A_not_paper_confirmed", smoke["plan"], active=139, global_down=True)
    result = {"run_type": "cpu_meta_parameter_audit", "eligible_for_formal_tables": False,
              "base_weights_loaded": False, "gpu_operations": False,
              "source_plan": args.plan, "measurement": "unique installed Parameters and buffers on actual local model architecture",
              "candidates_are_not_confirmed_for_chloe_formal_runs": True,
              "groups_are_short_smoke_diagnostics_not_formal_sampling": True,
              "alternative_similarity_counts": "pending real gradients and agreed conventions; not inferred from cached RMI",
              "results": results}
    write_json(output_path(args.output), result)
    print(json.dumps({name: row["parameters"]["trainable_adapter_parameters"] for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
