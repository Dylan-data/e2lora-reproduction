"""Run one full GLUE experiment with the selected reproduction settings."""
from fractions import Fraction
import importlib.util
import sys
from pathlib import Path

from sst2_partial_rank import SplitLinear


def allocate(weights, budget):
    quotas = [Fraction(weight) * budget / sum(weights) for weight in weights]
    ranks = [int(quota // 1) for quota in quotas]
    order = sorted(range(len(ranks)), key=lambda i: quotas[i] - ranks[i], reverse=True)
    for index in order[:budget - sum(ranks)]:
        ranks[index] += 1
    if sum(ranks) != budget or min(ranks) < 1:
        raise ValueError(f"Invalid allocation for budget {budget}: {ranks}")
    return ranks


def main():
    spec = importlib.util.spec_from_file_location(
        "train_glue", Path(__file__).with_name("train_glue.py")
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    method = sys.argv[sys.argv.index("--method") + 1]
    cfg = runner.SETUP["partial_rank_sharing"]
    original_plan = runner.build_plan
    original_attach = runner.apply_plan
    if method == "vera":
        from peft import VeraConfig, get_peft_model

        def attach_vera(model, method, rank, total_steps):
            return get_peft_model(model, VeraConfig(r=1024, target_modules=cfg["target_projections"]))

        runner.attach_baseline = attach_vera

    if method in {"e2lora", "e2sharelora"}:
        def plan_with_fixed_budget(gradients, families, **kwargs):
            plan, diagnostics = original_plan(gradients, families, **kwargs)
            for family, data in diagnostics.items():
                groups = [group for group in plan if group["family"] == family]
                scores = [max(data["entropy"][first:last]) for first, last in data["intervals"]]
                ranks = allocate(scores, cfg["total_rank_per_projection"])
                active = allocate(ranks, cfg["trainable_rank_per_projection"])
                for group, rank, trainable in zip(groups, ranks, active):
                    group.update(rank=rank, trainable_rank=trainable)
                data.update(ranks=ranks, allocated_rank=sum(ranks), trainable_ranks=active)
            return plan, diagnostics

        def attach_partial(model, plan, **kwargs):
            original_attach(model, plan, **kwargs)
            for group in plan:
                layers = [model.get_submodule(name) for name in group["layers"]]
                rank = group["trainable_rank"]
                down = layers[0].down
                if method == "e2sharelora":
                    down.requires_grad_(False)
                else:
                    down = SplitLinear(down.weight, rank, axis=0)
                weight = layers[0].up.weight.detach().clone()
                weight[:, rank:].uniform_(-0.01, 0.01)
                up = SplitLinear(weight, rank, axis=1)
                for layer in layers:
                    layer.down, layer.up = down, up
            actual = runner.trainable_count(model)
            expected = cfg["trainable_adapter_parameters"][method]
            if actual != expected:
                raise ValueError(f"Expected {expected} trainable adapter parameters, got {actual}")
            print(f"{method} trainable adapter parameters: {actual}", flush=True)

        runner.build_plan = plan_with_fixed_budget
        runner.apply_plan = attach_partial

    runner.main()


if __name__ == "__main__":
    main()
