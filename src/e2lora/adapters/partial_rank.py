"""Explicit candidate partial-rank policy; never enabled by a default.

Frozen ranks are buffers and have a saved, potentially nonzero forward branch.
This is an implementable team candidate, not a claim about the paper's code.
"""
from copy import deepcopy
from fractions import Fraction
import math

import torch
from torch import nn
from torch.nn import functional as F

from .lora_layers import apply_plan


def active_rank_plan(plan, budgets):
    """Exact largest-remainder allocation, capped by each group's total rank."""
    if not plan or any(not g.get("family") or type(g.get("rank")) is not int or g["rank"] < 1 for g in plan):
        raise ValueError("Provide nonempty groups with a family and positive total ranks.")
    result = deepcopy(plan)
    families = {g["family"] for g in result}
    if set(budgets) != families:
        raise ValueError("Supply an active rank budget for exactly every projection family.")
    for family in sorted(families):
        groups = [g for g in result if g["family"] == family]
        total = sum(g["rank"] for g in groups)
        budget = budgets[family]
        if type(budget) is not int or not 0 <= budget <= total:
            raise ValueError("Active rank budget must lie between zero and the allocated total.")
        quotas = [Fraction(g["rank"] * budget, total) for g in groups]
        active = [int(q) for q in quotas]
        order = sorted(range(len(groups)), key=lambda i: (-(quotas[i] - active[i]), i))
        for i in order[:budget - sum(active)]:
            active[i] += 1
        for group, count in zip(groups, active):
            group["trainable_rank"] = count
    return result


class SplitLinear(nn.Module):
    def __init__(self, weight, active_rank, *, axis):
        super().__init__()
        if axis not in {0, 1} or type(active_rank) is not int or not 0 <= active_rank <= weight.shape[axis]:
            raise ValueError("Invalid active rank or factor axis.")
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.axis = axis
        active, fixed = weight.detach().split([active_rank, weight.shape[axis] - active_rank], dim=axis)
        self.active = nn.Parameter(active.clone(), requires_grad=active.numel() > 0)
        self.register_buffer("fixed", fixed.clone())

    @property
    def weight(self):
        return torch.cat([self.active, self.fixed], dim=self.axis)

    def forward(self, inputs):
        return F.linear(inputs, self.weight)


def apply_partial_plan(model, plan, *, alpha, frozen_b_init_scale, global_down=False):
    if not isinstance(frozen_b_init_scale, (float, int)) or not math.isfinite(frozen_b_init_scale) or frozen_b_init_scale < 0:
        raise ValueError("Supply an explicit finite nonnegative frozen B initialization scale.")
    for group in plan:
        count = group.get("trainable_rank")
        if type(count) is not int or not 0 <= count <= group["rank"]:
            raise ValueError("Each group needs a valid trainable_rank.")
    apply_plan(model, plan, alpha=alpha, global_down=global_down)
    for group in plan:
        layers = [model.get_submodule(name) for name in group["layers"]]
        first, count = layers[0], group["trainable_rank"]
        if global_down:
            down = first.down
            down.requires_grad_(False)
        else:
            down = SplitLinear(first.down.weight, count, axis=0)
        weight = first.up.weight.detach().clone()
        if frozen_b_init_scale:
            weight[:, count:].uniform_(-frozen_b_init_scale, frozen_b_init_scale)
        up = SplitLinear(weight, count, axis=1)
        for layer in layers:
            layer.down, layer.up = down, up


def adapter_state(model):
    """Unique adapter Parameters AND buffers; exclude base weights and alias copies."""
    state, seen = {}, set()
    for name, layer in model.named_modules():
        if not all(hasattr(layer, key) for key in ("base", "down", "up")):
            continue
        for factor in ("down", "up"):
            for key, value in getattr(layer, factor).state_dict(keep_vars=True).items():
                if id(value) not in seen:
                    state[f"{name}.{factor}.{key}"] = value
                    seen.add(id(value))
    return state


def count_adapters(model, plan):
    values = adapter_state(model)
    unique_trainable = {id(p): p for p in model.parameters() if p.requires_grad}
    adapters = {id(x): x for x in values.values()}
    if not set(unique_trainable).issubset(adapters):
        raise ValueError("Trainable parameters outside the adapters (base or head).")
    for group in plan:
        layers = [model.get_submodule(name) for name in group["layers"]]
        if any(x.down is not layers[0].down or x.up is not layers[0].up for x in layers):
            raise ValueError("Group sharing identity was lost.")
    total = sum(x.numel() for x in adapters.values())
    trainable = sum(x.numel() for x in unique_trainable.values())
    return {"adapter_elements_including_frozen_buffers": total,
            "trainable_adapter_parameters": trainable, "total_trainable_parameters": trainable,
            "frozen_adapter_elements": total - trainable,
            "unique_trainable_tensors": len(unique_trainable)}
