"""Controlled LSS/HRA switches, shared by CPU tests and formal preparation."""
from time import perf_counter

from .adaptive import allocate_ranks
from .grouping import search_intervals
from .similarity import similarity_matrix


def family_plan(gradients, names, *, family, base_rank, lss, hra, similarity="rmi",
                convention=None, rank_policy="paper", grouping="greedy", dp_options=None):
    if type(lss) is not bool or type(hra) is not bool:
        raise ValueError("lss and hra must be booleans.")
    if type(base_rank) is not int or base_rank < 1 or not names or len(set(names)) != len(names):
        raise ValueError("Provide distinct layers and a positive base rank.")
    if not lss and (similarity != "rmi" or convention is not None or grouping != "greedy" or dp_options):
        raise ValueError("Similarity and grouping are inactive when LSS is disabled.")
    started = perf_counter()
    entropy, matrix = similarity_matrix([gradients[name] for name in names],
                                        metric=similarity, convention=convention)
    similarity_seconds = perf_counter() - started
    if lss:
        intervals, detail = search_intervals(matrix, method=grouping, **(dp_options or {}))
    else:
        intervals = [(i, i + 1) for i in range(len(names))]
        detail = {"method": "independent", "grouping_seconds": 0.0}
    started = perf_counter()
    ranks = (allocate_ranks(entropy, intervals, base_rank, rank_policy=rank_policy)
             if hra else [base_rank] * len(intervals))
    allocation_seconds = perf_counter() - started
    plan = [dict(family=family, layers=list(names[s:t]), rank=r)
            for (s, t), r in zip(intervals, ranks)]
    return plan, dict(detail, layers=list(names), entropy=entropy, similarity=similarity,
                      convention=convention, matrix=matrix, intervals=intervals, ranks=ranks,
                      lss=lss, hra=hra, rank_policy=rank_policy,
                      nominal_rank_budget=len(names) * base_rank, allocated_rank=sum(ranks),
                      similarity_seconds=similarity_seconds, allocation_seconds=allocation_seconds)


def build_ablation_plan(gradients, families, **options):
    names = [name for group in families.values() for name in group]
    if not names or len(names) != len(set(names)):
        raise ValueError("Families must contain distinct targets.")
    plan, diagnostics = [], {}
    for family, names in families.items():
        groups, detail = family_plan(gradients, names, family=family, **options)
        plan.extend(groups)
        diagnostics[family] = detail
    return plan, diagnostics
