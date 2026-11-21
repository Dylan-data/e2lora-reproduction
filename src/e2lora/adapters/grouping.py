"""Continuous grouping, including the Appendix G penalized DP objective.

Mean and singleton conventions must be supplied: the paper does not specify them.
All intervals are half-open. Timing includes validation and cost precomputation.
"""
import math
from time import perf_counter

from .adaptive import greedy_intervals


def search_intervals(matrix, *, method, penalty_lambda=None, mean_policy=None,
                     singleton_cost=None):
    started = perf_counter()
    matrix = [[float(x) for x in row] for row in matrix]
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("Expected a nonempty square matrix.")
    if any(not math.isfinite(x) for row in matrix for x in row):
        raise ValueError("Similarity matrix must be finite.")
    if any(not math.isclose(matrix[i][j], matrix[j][i], abs_tol=1e-12, rel_tol=1e-12)
           for i in range(n) for j in range(i)):
        raise ValueError("Expected a symmetric similarity matrix.")
    detail = {"method": method, "layers": n}
    if method == "greedy":
        if any(x is not None for x in (penalty_lambda, mean_policy, singleton_cost)):
            raise ValueError("DP options must not be supplied to Greedy.")
        intervals, thresholds = greedy_intervals(matrix)
        detail["thresholds"] = thresholds
    elif method == "dp":
        if (isinstance(penalty_lambda, bool) or not isinstance(penalty_lambda, (int, float))
                or not math.isfinite(penalty_lambda) or penalty_lambda < 0):
            raise ValueError("DP needs an explicit finite nonnegative penalty_lambda.")
        if mean_policy not in {"include_diagonal", "exclude_diagonal"}:
            raise ValueError("Specify include_diagonal or exclude_diagonal.")
        if mean_policy == "exclude_diagonal":
            if not isinstance(singleton_cost, (float, int)) or not math.isfinite(singleton_cost):
                raise ValueError("An off-diagonal mean needs an explicit finite singleton_cost.")
        elif singleton_cost is not None:
            raise ValueError("The diagonal already defines singleton cost.")
        prefix = [[0.0] * (n + 1) for _ in range(n + 1)]
        diagonal = [0.0]
        for i in range(n):
            diagonal.append(diagonal[-1] + matrix[i][i])
            for j in range(n):
                prefix[i + 1][j + 1] = (matrix[i][j] + prefix[i][j + 1]
                                          + prefix[i + 1][j] - prefix[i][j])

        def cost(s, t):
            size = t - s
            total = prefix[t][t] - prefix[s][t] - prefix[t][s] + prefix[s][s]
            if mean_policy == "include_diagonal":
                return -total / (size * size)
            return (singleton_cost if size == 1 else
                    -(total - diagonal[t] + diagonal[s]) / (size * (size - 1)))

        best, counts, previous = [0.0] + [math.inf] * n, [0] * (n + 1), [0] * (n + 1)
        for t in range(1, n + 1):
            # Tuple comparison: minimum objective, fewer groups, smaller predecessor.
            value, count, s = min((best[s] + cost(s, t) + penalty_lambda,
                                    counts[s] + 1, s) for s in range(t))
            best[t], counts[t], previous[t] = value, count, s
        intervals, t = [], n
        while t:
            s = previous[t]
            intervals.append((s, t))
            t = s
        intervals.reverse()
        detail.update(objective=best[n], penalty_lambda=penalty_lambda,
                      mean_policy=mean_policy, singleton_cost=singleton_cost,
                      tie_break="minimum objective, fewer groups, smaller predecessor recursively")
    else:
        raise ValueError("method must be greedy or dp.")
    detail.update(intervals=intervals, group_count=len(intervals),
                  grouping_seconds=perf_counter() - started)
    return intervals, detail
