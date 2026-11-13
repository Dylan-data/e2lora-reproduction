"""Gradient analysis and adaptive plans from E²LoRA §3 and Appendix M."""
import math
from time import perf_counter

import torch
from torch import nn


def collect_gradients(model, targets, batches, loss_for_batch):
    """Average gradients on CPU. The callback returns (mean loss, averaging weight)."""
    layers = [model.get_submodule(name) for name in targets]
    if not layers or any(not isinstance(layer, nn.Linear) for layer in layers):
        raise ValueError("Provide one or more linear target layers.")
    weights = [layer.weight for layer in layers]
    if len({id(weight) for weight in weights}) != len(weights):
        raise ValueError("Gradient targets must have distinct weights.")
    flags = [weight.requires_grad for weight in weights]
    modes = [(module, module.training) for module in model.modules()]
    totals = {}
    count = 0
    try:
        model.eval()
        for weight in weights:
            weight.requires_grad_(True)
        with torch.enable_grad():
            for batch in batches:
                loss, size = loss_for_batch(model, batch)
                if size <= 0 or not math.isfinite(size) or not torch.isfinite(loss):
                    raise ValueError("Each batch needs a finite mean loss and positive weight.")
                gradients = torch.autograd.grad(loss, weights)
                for name, grad in zip(targets, gradients):
                    cpu = grad.detach().to(device="cpu", dtype=torch.float32)
                    if name not in totals:
                        totals[name] = torch.zeros_like(cpu)
                    totals[name].add_(cpu, alpha=size)
                count += size
    finally:
        for weight, flag in zip(weights, flags):
            weight.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
    if not count:
        raise ValueError("Provide at least one training batch.")
    for grad in totals.values():
        grad.div_(count)
    return totals


def _entropy(variance, epsilon):
    return math.log(max(variance, epsilon ** 2)) / 2 + math.log(2 * math.pi * math.e) / 2


def gradient_similarity(gradients, *, epsilon=1e-12):
    """Equations 1–3, using population variance and the actual concatenated distribution."""
    if not gradients or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Provide gradients and a finite positive epsilon.")
    moments = []
    for grad in gradients:
        flat = grad.detach().flatten().to(dtype=torch.float64)
        if not flat.numel() or not torch.isfinite(flat).all():
            raise ValueError("Each gradient must be nonempty and finite.")
        var, mean = torch.var_mean(flat, correction=0)
        moments.append((flat.numel(), mean.item(), var.item()))

    entropies = [_entropy(var, epsilon) for _, _, var in moments]
    matrix = [[1.0 for _ in moments] for _ in moments]
    for i, (n, mean_i, var_i) in enumerate(moments):
        for j in range(i):
            m, mean_j, var_j = moments[j]
            p = n / (n + m)
            q = 1 - p
            pooled = p * var_i + q * var_j + p * q * (mean_i - mean_j) ** 2
            denominator = min(entropies[i], entropies[j])
            if abs(denominator) < epsilon:
                denominator = math.copysign(epsilon, denominator)
            similarity = (entropies[i] + entropies[j] - _entropy(pooled, epsilon)) / denominator
            matrix[i][j] = matrix[j][i] = similarity
    return entropies, matrix


def greedy_intervals(matrix):
    """Return half-open consecutive intervals and each layer's mean similarity to other layers."""
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("Expected a nonempty square similarity matrix.")
    thresholds = [
        sum(value for j, value in enumerate(row) if j != i) / max(1, n - 1)
        for i, row in enumerate(matrix)
    ]
    intervals = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and matrix[end][start] >= thresholds[start]:
            end += 1
        intervals.append((start, end))
        start = end
    return intervals, thresholds


def allocate_ranks(entropies, intervals, base_rank, *, rank_policy="shift_nonpositive"):
    """Equations 5–7 with an explicit policy for nonpositive differential entropy."""
    if not isinstance(base_rank, int) or base_rank < 1:
        raise ValueError("Base rank must be a positive integer.")
    scores = [max(entropies[start:end]) for start, end in intervals]
    if rank_policy == "shift_nonpositive":
        low = min(scores)
        if low <= 0:
            scores = [score - low + 1.0 for score in scores]
    elif rank_policy != "paper":
        raise ValueError("Rank policy must be 'paper' or 'shift_nonpositive'.")
    total = sum(scores)
    if abs(total) < 1e-12:
        raise ValueError("Interval entropies sum to zero; select a rank policy for this case.")
    ranks = [round(score / total * len(entropies) * base_rank) for score in scores]
    if rank_policy == "paper" and min(ranks) < 1:
        raise ValueError("Literal rank allocation produced a nonpositive rank.")
    return [max(1, rank) for rank in ranks]


def build_plan(gradients, families, *, base_rank, rank_policy="shift_nonpositive"):
    """Each family lists compatible layer names in depth order, e.g. all query projections."""
    plan = []
    diagnostics = {}
    for family, names in families.items():
        start = perf_counter()
        entropies, matrix = gradient_similarity([gradients[name] for name in names])
        similarity_seconds = perf_counter() - start
        start = perf_counter()
        intervals, thresholds = greedy_intervals(matrix)
        grouping_seconds = perf_counter() - start
        start = perf_counter()
        ranks = allocate_ranks(entropies, intervals, base_rank, rank_policy=rank_policy)
        allocation_seconds = perf_counter() - start
        for (first, last), rank in zip(intervals, ranks):
            plan.append({"layers": list(names[first:last]), "rank": rank, "family": family})
        diagnostics[family] = {
            "layers": list(names), "entropy": entropies, "rmi": matrix,
            "thresholds": thresholds, "intervals": intervals, "ranks": ranks,
            "nominal_rank_budget": len(names) * base_rank, "allocated_rank": sum(ranks),
            "similarity_seconds": similarity_seconds, "grouping_seconds": grouping_seconds,
            "allocation_seconds": allocation_seconds,
        }
    return plan, diagnostics
