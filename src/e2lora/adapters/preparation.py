"""Bound dense gradient memory to one target; reuse the existing LSS/HRA kernels."""
import math
from time import perf_counter

import torch
from torch import nn

from .adaptive import allocate_ranks, collect_gradients, gradient_similarity, greedy_intervals


def collect_family_gradients(model, targets, batches, loss_for_batch, *, on_target_complete=None):
    """One target at a time; return detached CPU means for one projection family.

    An optional observer receives scalar metadata only, after GPU temporaries
    are released. CUDA synchronization is added only for observed probes.
    The caller must release this dictionary before starting the next family.
    """
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze the base before sequential gradient collection.")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Provide distinct nonempty gradient targets.")
    batches = list(batches)
    if not batches:
        raise ValueError("Provide fixed sampling batches.")
    device = next(model.parameters()).device
    gradients = {}
    for index, name in enumerate(targets):
        if on_target_complete is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        current = collect_gradients(model, [name], batches, loss_for_batch)
        gradients.update(current)
        del current
        if on_target_complete is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            detail = {"target": name, "target_index_in_family": index,
                      "backward_passes": len(batches), "gradient_seconds": perf_counter() - started,
                      "cpu_gradient_cache_bytes": sum(g.numel() * g.element_size() for g in gradients.values())}
            if device.type == "cuda":
                detail.update(cuda_allocated_after_bytes=torch.cuda.memory_allocated(device),
                              cuda_reserved_after_bytes=torch.cuda.memory_reserved(device),
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            on_target_complete(detail)
    return gradients


def collect_plan(model, families, batches, loss_for_batch, *, base_rank,
                 rank_policy, diagnostics_callback=None, mechanism_options=None):
    """Replay fixed batches per target, offload means, and plan one family at a time.

    The base must be frozen. No optimizer is created and no weights are updated.
    CPU FP32 mean gradients are retained for one projection family, then released.
    """
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze the base before adaptive gradient collection.")
    names = [name for family in families.values() for name in family]
    if not names or len(names) != len(set(names)) or any(not family for family in families.values()):
        raise ValueError("Provide nonempty families with distinct targets in depth order.")
    for family in families.values():
        layers = [model.get_submodule(name) for name in family]
        if any(not isinstance(layer, nn.Linear) for layer in layers):
            raise ValueError("Gradient targets must be original linear layers.")
        if len({(x.in_features, x.out_features) for x in layers}) != 1:
            raise ValueError("Projection families must have compatible shapes.")
    batches = list(batches)
    if not batches:
        raise ValueError("Provide fixed sampling batches.")
    device = next(model.parameters()).device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    plan, diagnostics = [], {}
    timings = {"gradient_seconds": 0.0, "similarity_seconds": 0.0,
               "grouping_seconds": 0.0, "allocation_seconds": 0.0,
               "target_block_size": 1, "backward_passes": len(names) * len(batches)}
    for family, targets in families.items():
        synchronize()
        started = perf_counter()
        gradients = collect_family_gradients(model, targets, batches, loss_for_batch)
        synchronize()
        gradient_seconds = perf_counter() - started
        if mechanism_options is not None:
            from .ablation import family_plan
            try:
                groups, detail = family_plan(gradients, targets, family=family,
                                             base_rank=base_rank, rank_policy=rank_policy,
                                             **mechanism_options)
            except ValueError as error:
                if diagnostics_callback:
                    diagnostics_callback(family, {"status": "planning_failed", "error": str(error),
                                                   "gradient_seconds": gradient_seconds})
                raise
            del gradients
            detail.update(gradient_seconds=gradient_seconds, status="planned")
            plan.extend(groups)
            diagnostics[family] = detail
            for key in ["gradient_seconds", "similarity_seconds", "grouping_seconds", "allocation_seconds"]:
                timings[key] += detail[key]
            if diagnostics_callback:
                diagnostics_callback(family, detail)
            continue
        started = perf_counter()
        entropies, matrix = gradient_similarity([gradients[name] for name in targets])
        similarity_seconds = perf_counter() - started
        del gradients
        if any(not math.isfinite(x) for x in entropies) or any(not math.isfinite(x) for row in matrix for x in row):
            raise ValueError("Nonfinite entropy or RMI; do not continue with a fallback policy.")
        started = perf_counter()
        intervals, thresholds = greedy_intervals(matrix)
        grouping_seconds = perf_counter() - started
        detail = {"layers": list(targets), "entropy": entropies, "rmi": matrix,
                  "intervals": intervals, "thresholds": thresholds,
                  "nominal_rank_budget": len(targets) * base_rank,
                  "gradient_seconds": gradient_seconds, "similarity_seconds": similarity_seconds,
                  "grouping_seconds": grouping_seconds,
                  "warnings": ["negative differential entropy; literal paper ratios retained"] if min(entropies) < 0 else []}
        started = perf_counter()
        try:
            ranks = allocate_ranks(entropies, intervals, base_rank, rank_policy=rank_policy)
        except ValueError as error:
            detail.update(status="rank_allocation_failed", error=str(error))
            if diagnostics_callback:
                diagnostics_callback(family, detail)
            raise
        allocation_seconds = perf_counter() - started
        detail.update(status="planned", ranks=ranks, allocated_rank=sum(ranks),
                      allocation_seconds=allocation_seconds,
                      interval_entropy=[max(entropies[a:b]) for a, b in intervals])
        for (first, last), rank in zip(intervals, ranks):
            plan.append({"layers": list(targets[first:last]), "rank": rank, "family": family})
        diagnostics[family] = detail
        for key in ["gradient_seconds", "similarity_seconds", "grouping_seconds", "allocation_seconds"]:
            timings[key] += detail[key]
        if diagnostics_callback:
            diagnostics_callback(family, detail)
    return plan, diagnostics, timings
