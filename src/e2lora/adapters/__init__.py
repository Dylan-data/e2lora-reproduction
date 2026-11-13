"""Ordinary LoRA and manual weight sharing for the first integration run."""
from collections import defaultdict

import torch
from torch import nn


class AdaptedLinear(nn.Module):
    def __init__(self, base, down, up, alpha):
        super().__init__()
        self.base = base
        self.down = down
        self.up = up
        self.scale = alpha / down.out_features

    def forward(self, inputs):
        return self.base(inputs) + self.up(self.down(inputs)) * self.scale


def apply_plan(model, plan, *, alpha):
    """Each group specifies layer names and rank. Both factors are shared."""
    layers_by_group = []
    seen = set()
    for group in plan:
        names, rank = group["layers"], group["rank"]
        if not names or not isinstance(rank, int) or rank < 1:
            raise ValueError("Every group needs at least one layer and a positive rank.")
        layers = [model.get_submodule(name) for name in names]
        if any(not isinstance(layer, nn.Linear) for layer in layers):
            raise ValueError("Adapters require torch.nn.Linear layers.")
        if len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("Each target layer must appear only once.")
        signatures = {
            (layer.in_features, layer.out_features, layer.weight.device, layer.weight.dtype)
            for layer in layers
        }
        if len(signatures) != 1:
            raise ValueError("Layers in one group must have the same shape, device and dtype.")
        seen.update(names)
        layers_by_group.append((names, rank, layers))
    if not layers_by_group:
        raise ValueError("The plan must include at least one target layer.")

    model.requires_grad_(False)
    for names, rank, layers in layers_by_group:
        first = layers[0]
        kwargs = {"device": first.weight.device, "dtype": first.weight.dtype, "bias": False}
        down = nn.Linear(first.in_features, rank, **kwargs)
        up = nn.Linear(rank, first.out_features, **kwargs)
        nn.init.zeros_(up.weight)
        for name, layer in zip(names, layers):
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, AdaptedLinear(layer, down, up, alpha))


def manual_plan(targets, *, rank, share):
    groups = defaultdict(list)
    for name in targets:
        # Keep different projections separate in the shared pilot.
        groups[name.rsplit(".", 1)[-1] if share else name].append(name)
    return [{"layers": names, "rank": rank} for names in groups.values()]


def trainable_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
