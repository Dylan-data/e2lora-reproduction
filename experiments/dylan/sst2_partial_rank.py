"""Run the SST-2 candidate with 96 total and 52 trainable ranks per projection."""
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn


class SplitLinear(nn.Module):
    def __init__(self, weight, active_rank, axis):
        super().__init__()
        self.axis = axis
        self.in_features = weight.shape[1]
        active, fixed = weight.detach().split(
            [active_rank, weight.shape[axis] - active_rank], dim=axis
        )
        self.active = nn.Parameter(active.clone())
        self.register_buffer("fixed", fixed.clone())

    @property
    def weight(self):
        return torch.cat((self.active, self.fixed), dim=self.axis)

    def forward(self, inputs):
        return nn.functional.linear(inputs, self.weight)


def main():
    spec = importlib.util.spec_from_file_location(
        "train_glue", Path(__file__).with_name("train_glue.py")
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = Path(os.environ["PARTIAL_RANK_OUTPUT"])
    source = json.loads((output / "pilot-plan.json").read_text())
    plan = source["plan"]
    for family in ("query", "value"):
        groups = [group for group in plan if group["family"] == family]
        if sum(group["rank"] for group in groups) != 96:
            raise ValueError(f"Expected rank 96 for {family}")
        quotas = [Fraction(group["rank"] * 52, 96) for group in groups]
        active = [int(quota) for quota in quotas]
        order = sorted(range(len(active)), key=lambda i: quotas[i] - active[i], reverse=True)
        for index in order[:52 - sum(active)]:
            active[index] += 1
        for group, rank in zip(groups, active):
            group["trainable_rank"] = rank

    apply_plan = runner.apply_plan

    def attach(model, plan, **kwargs):
        apply_plan(model, plan, **kwargs)
        for group in plan:
            layers = [model.get_submodule(name) for name in group["layers"]]
            first = layers[0]
            rank = group["trainable_rank"]
            down = SplitLinear(first.down.weight, rank, axis=0)
            up_weight = first.up.weight.detach().clone()
            up_weight[:, rank:].uniform_(-0.01, 0.01)
            up = SplitLinear(up_weight, rank, axis=1)
            for layer in layers:
                layer.down = down
                layer.up = up
        actual = runner.trainable_count(model)
        if actual != 159744:
            raise ValueError(f"Expected 159744 trainable adapter parameters, got {actual}")
        print(f"E²LoRA trainable adapter parameters: {actual}", flush=True)

    runner.collect_gradients = lambda *args, **kwargs: None
    runner.build_plan = lambda *args, **kwargs: (plan, source["diagnostics"])
    runner.apply_plan = attach
    sys.argv = [
        sys.argv[0], "--task", "sst2", "--method", "e2lora", "--seed", "42",
        "--device", "cuda:0", "--output-root", str(output / "results"),
        "--checkpoint", str(output / "e2lora.pt"),
    ]
    runner.main()


if __name__ == "__main__":
    main()
