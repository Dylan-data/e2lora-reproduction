"""Validate the full E²LoRA planning and sharing flow with a small CPU model."""
import argparse
import math
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from e2lora.adapters.lora_layers import apply_plan, trainable_count
from e2lora.adapters.adaptive import build_plan, collect_gradients, gradient_similarity
from e2lora.runner.classification_runner import run_classification, save_result


def save_heatmap(path, labels, matrix):
    low = min(map(min, matrix))
    high = max(map(max, matrix))
    size = 52
    left = 115
    top = 52
    width = left + size * len(labels) + 12
    height = top + size * len(labels) + 34
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="11" fill="#122638">',
        '<text x="12" y="21" font-size="15">Toy model gradient similarity (RMI)</text>',
    ]
    for i, label in enumerate(labels):
        parts.append(f'<text x="10" y="{top + i * size + 29}">{label}</text>')
        parts.append(f'<text x="{left + i * size + 5}" y="43">{label}</text>')
        for j, value in enumerate(matrix[i]):
            x, y = left + j * size, top + i * size
            level = (value - low) / max(high - low, 1e-12)
            shade = round(245 - 140 * level)
            parts.extend([
                f'<rect x="{x}" y="{y}" width="50" height="50" fill="rgb({shade},{shade + 8},245)"/>',
                f'<text x="{x + 6}" y="{y + 29}">{value:.3f}</text>',
            ])
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = nn.Sequential(
        nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, 6), nn.Tanh(),
        nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, 6), nn.Linear(6, 3),
    )
    names = ["0", "2", "4", "6"]
    inputs = torch.randn(48, 6)
    labels = inputs[:, :3].argmax(-1)
    batches = [
        {"inputs": x, "labels": y}
        for x, y in zip(inputs.split(8), labels.split(8))
    ]
    train, validation = batches[:4], batches[4:]

    def loss_for_batch(model, batch):
        logits = model(batch["inputs"])
        return nn.functional.cross_entropy(logits, batch["labels"]), len(logits)

    original = model(inputs).detach()
    start = perf_counter()
    gradients = collect_gradients(model, names, train, loss_for_batch)
    gradient_seconds = perf_counter() - start
    reference = torch.autograd.grad(
        nn.functional.cross_entropy(model(inputs[:32]), labels[:32]),
        [model.get_submodule(name).weight for name in names],
    )
    for name, expected in zip(names, reference):
        torch.testing.assert_close(gradients[name], expected)
    assert all(parameter.grad is None for parameter in model.parameters())

    plan, diagnostics = build_plan(gradients, {"linear": names}, base_rank=2)
    assert [name for group in plan for name in group["layers"]] == names
    left, right = torch.tensor([1.0, 2.0]), torch.tensor([-1.0, 2.0, 5.0])
    entropy, matrix = gradient_similarity([left, right])
    joined = torch.cat([left, right]).double()
    joint_entropy = math.log(joined.std(correction=0).item()) + math.log(2 * math.pi * math.e) / 2
    assert math.isclose(matrix[0][1], (sum(entropy) - joint_entropy) / min(entropy))

    results = {}
    for mode, global_down in (("e2lora", False), ("e2sharelora", True)):
        adapted = deepcopy(model)
        apply_plan(adapted, plan, alpha=4, global_down=global_down)
        torch.testing.assert_close(adapted(inputs), original)
        for group in plan:
            first, *rest = [adapted.get_submodule(name) for name in group["layers"]]
            assert all(layer.down is first.down and layer.up is first.up for layer in rest)
        ranks = [group["rank"] for group in plan]
        expected = 6 * (sum(ranks) + (max(ranks) if global_down else sum(ranks)))
        assert trainable_count(adapted) == expected
        if global_down:
            assert adapted[0].down is adapted[6].down
        results[mode] = run_classification(
            adapted, train, validation, lambda model, batch: model(batch["inputs"]),
            lr=0.01, device="cpu",
        )

    path = save_result("algorithm-checks", "adaptive-preflight", {
        "paper_reference": "§3; Appendix M; synthetic validation",
        "seed": args.seed, "rank_policy": "shift_nonpositive",
        "gradient_seconds": gradient_seconds, "gradient_examples": 32,
        "plan": plan, "diagnostics": diagnostics, "results": results,
    }, output_root=args.output_root)
    save_heatmap(path.with_name("rmi.svg"), names, diagnostics["linear"]["rmi"])
    print(path)


if __name__ == "__main__":
    main()
