"""Reproducible interval-search and RMI-noise primitives for image ablations."""

import math
from time import perf_counter_ns

import torch


def _validate_matrix(matrix):
    values = torch.as_tensor(matrix, dtype=torch.float64, device="cpu")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[0] != values.shape[1]:
        raise ValueError("RMI must be a nonempty square matrix.")
    if not torch.isfinite(values).all():
        raise ValueError("RMI entries must be finite.")
    if not torch.allclose(values, values.T, rtol=0, atol=1e-10):
        raise ValueError("The input RMI matrix must be symmetric.")
    return values


def greedy_intervals(matrix):
    """Paper-style row-mean thresholding followed by sequential interval growth."""
    rmi = _validate_matrix(matrix)
    started = perf_counter_ns()
    count = rmi.shape[0]
    thresholds = [
        float(torch.cat((rmi[row, :row], rmi[row, row + 1 :])).mean())
        if count > 1
        else 0.0
        for row in range(count)
    ]
    intervals = []
    start = 0
    while start < count:
        end = start + 1
        while end < count and float(rmi[end, start]) >= thresholds[start]:
            end += 1
        intervals.append((start, end))
        start = end
    return {
        "intervals": intervals,
        "thresholds": thresholds,
        "search_ns": perf_counter_ns() - started,
        "method": "greedy",
    }


def dynamic_programming_intervals(matrix, *, penalty_lambda):
    """Minimize sum of negative within-interval mean RMI plus lambda * interval count.

    Singleton interval similarity cost is defined as zero because there is no
    pairwise similarity inside a one-layer interval. The timer includes segment
    cost construction and the dynamic-programming/backtracking phase.
    """
    rmi = _validate_matrix(matrix)
    if not isinstance(penalty_lambda, (int, float)) or not math.isfinite(penalty_lambda):
        raise ValueError("penalty_lambda must be finite.")
    if penalty_lambda < 0:
        raise ValueError("penalty_lambda must be nonnegative.")

    started = perf_counter_ns()
    count = rmi.shape[0]
    costs = [[0.0] * count for _ in range(count)]
    for start in range(count):
        for end in range(start + 1, count):
            block = rmi[start : end + 1, start : end + 1]
            row, col = torch.triu_indices(end - start + 1, end - start + 1, offset=1)
            mean_similarity = float(block[row, col].mean())
            costs[start][end] = -mean_similarity

    best = [math.inf] * (count + 1)
    previous = [-1] * (count + 1)
    best[0] = 0.0
    for end in range(1, count + 1):
        for start in range(end):
            candidate = best[start] + costs[start][end - 1] + penalty_lambda
            if candidate < best[end]:
                best[end] = candidate
                previous[end] = start
    if previous[count] < 0:
        raise RuntimeError("Dynamic-programming search did not produce a partition.")

    intervals = []
    end = count
    while end:
        start = previous[end]
        intervals.append((start, end))
        end = start
    intervals.reverse()
    elapsed = perf_counter_ns() - started
    return {
        "intervals": intervals,
        "penalty_lambda": float(penalty_lambda),
        "objective": best[count],
        "segment_costs": costs,
        "search_ns": elapsed,
        "method": "dynamic_programming",
        "cost_definition": "negative mean of off-diagonal RMI within each interval; singleton cost=0",
        "tie_break": "first minimum in ascending start-index order",
    }


def perturb_rmi(matrix, *, sigma, seed):
    """Add seeded, symmetric zero-mean Gaussian noise to off-diagonal RMI pairs."""
    rmi = _validate_matrix(matrix)
    if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    upper = torch.randn(rmi.shape, generator=generator, dtype=torch.float64)
    noise = torch.triu(upper, diagonal=1)
    noise = (noise + noise.T) * float(sigma)
    return {
        "seed": seed,
        "sigma": float(sigma),
        "noise": noise,
        "noisy_rmi": rmi + noise,
        "noise_policy": "independent Gaussian draws for upper-triangle pairs, mirrored to preserve RMI symmetry; diagonal unchanged",
    }
