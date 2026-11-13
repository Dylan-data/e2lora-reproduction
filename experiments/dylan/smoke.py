"""Offline test of training, actual shared weights, and saving results."""
import argparse

import torch
from torch import nn

from e2lora.adapters import apply_plan
from e2lora.runner import run_classification, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lora", "manual-sharing"), default="manual-sharing")
    args = parser.parse_args()
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    groups = [["0", "2"]] if args.method == "manual-sharing" else [["0"], ["2"]]
    plan = [{"layers": layers, "rank": 2} for layers in groups]
    inputs = torch.randn(16, 4)
    with torch.no_grad():
        baseline = model(inputs)
    apply_plan(model, plan, alpha=2)
    torch.testing.assert_close(model(inputs), baseline)
    if args.method == "manual-sharing":
        assert model[0].down is model[2].down and model[0].up is model[2].up

    def batch():
        inputs = torch.randn(16, 4)
        return {"inputs": inputs, "labels": inputs.argmax(dim=-1)}

    result = run_classification(
        model, [batch() for _ in range(4)], [batch()],
        lambda model, batch: model(batch["inputs"]), lr=0.01, device="cpu",
    )
    assert result["trainable_parameters"] == 16 * len(groups)
    print(save_result("dylan", "smoke", {
        **result, "method": args.method, "seed": 42, "plan": plan,
        "paper_reference": None,
    }))


if __name__ == "__main__":
    main()
