"""Explicit alternative-metric conventions; these are reproducibility choices.

Table 4 / D.3 name the metrics but do not supply these transformations.
No alternative can be selected without an explicit convention identifier.
"""
import math

import torch

from .adaptive import gradient_similarity

CONVENTIONS = {
    "cosine": "flat_cosine_zero_both_one_zero_single_zero",
    "l1": "negative_mean_absolute_distance",
    "l2": "negative_root_mean_square_distance",
    "kl": "negative_symmetric_gaussian_kl",
}


def similarity_matrix(gradients, *, metric, convention=None, epsilon=1e-12):
    entropies, rmi = gradient_similarity(gradients, epsilon=epsilon)
    if metric == "rmi":
        if convention is not None:
            raise ValueError("RMI uses the existing equations 1-3; no alternative convention.")
        return entropies, rmi
    if metric not in CONVENTIONS or convention != CONVENTIONS[metric]:
        raise ValueError("Alternative metrics require their explicit convention identifier.")
    flat = [g.detach().flatten().to(device="cpu", dtype=torch.float64) for g in gradients]
    if len({x.numel() for x in flat}) != 1:
        raise ValueError("Alternative metrics need compatible gradient shapes.")
    moments = [torch.var_mean(x, correction=0) for x in flat]
    matrix = [[0.0] * len(flat) for _ in flat]
    for i, x in enumerate(flat):
        for j in range(i + 1):
            y = flat[j]
            if metric == "cosine":
                nx, ny = torch.linalg.vector_norm(x).item(), torch.linalg.vector_norm(y).item()
                value = ((1.0 if nx == ny == 0 else 0.0) if nx == 0 or ny == 0
                         else torch.dot(x / nx, y / ny).clamp(-1, 1).item())
            elif metric == "l1":
                value = -(x - y).abs().mean().item()
            elif metric == "l2":
                value = -(x - y).square().mean().sqrt().item()
            else:
                vx, mx = moments[i]
                vy, my = moments[j]
                vx, vy = max(vx.item(), epsilon ** 2), max(vy.item(), epsilon ** 2)
                delta = (mx.item() - my.item()) ** 2
                value = -0.25 * (vx / vy + vy / vx + delta * (1 / vx + 1 / vy) - 2)
            if not math.isfinite(value):
                raise ValueError("Nonfinite alternative similarity; no fallback.")
            matrix[i][j] = matrix[j][i] = value
    return entropies, matrix
