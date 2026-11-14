"""Run either full SST-2 candidate using a shared, saved allocation."""
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

from e2lora.adapters import adaptive

spec = importlib.util.spec_from_file_location(
    "train_glue", Path(__file__).with_name("train_glue.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
method = sys.argv[1]
if method not in {"e2lora", "e2sharelora"}:
    raise SystemExit("Expected e2lora or e2sharelora")

output = Path(os.environ["RANK52_OUTPUT"])
output.mkdir(parents=True, exist_ok=True)
allocation_file = output / "shared-allocation.json"
apply = runner.apply_plan
collect = runner.collect_gradients
allocate_plan = runner.build_plan
allocation = None


def allocate_52(entropies, intervals, base_rank, *, rank_policy):
    if rank_policy != "paper":
        raise ValueError("Expected the paper entropy formula")
    scores = [max(entropies[first:last]) for first, last in intervals]
    total = sum(scores)
    if abs(total) < 1e-12:
        raise ValueError("Interval entropies sum to zero")
    quotas = [52 * score / total for score in scores]
    ranks = [math.floor(quota) for quota in quotas]
    remaining = 52 - sum(ranks)
    order = sorted(range(len(ranks)), key=lambda i: quotas[i] - ranks[i], reverse=True)
    for index in order[:remaining]:
        ranks[index] += 1
    if min(ranks) < 1 or sum(ranks) != 52:
        raise ValueError(f"Invalid ranks for budget 52: {ranks}")
    return ranks


def prepare(model, names, batches, loss_callback):
    global allocation
    with (output / "allocation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if allocation_file.exists():
            allocation = json.loads(allocation_file.read_text())
            return None
        gradients = collect(model, names, batches, loss_callback)
        families = {
            projection: [name for name in names if name.endswith("." + projection)]
            for projection in runner.SETUP["model"]["target_layers"]
        }
        adaptive.allocate_ranks = allocate_52
        plan, diagnostics = allocate_plan(gradients, families, base_rank=8, rank_policy="paper")
        for info in diagnostics.values():
            info["nominal_rank_budget"] = 52
        allocation = (plan, diagnostics)
        temporary = allocation_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(allocation, indent=2))
        temporary.replace(allocation_file)
    return None


def attach(model, plan, **kwargs):
    apply(model, plan, **kwargs)
    if method == "e2sharelora":
        for group in plan:
            model.get_submodule(group["layers"][0]).down.requires_grad_(False)
    expected = 159744 if method == "e2lora" else 79872
    actual = runner.trainable_count(model)
    if actual != expected:
        raise ValueError(f"Expected {expected} trainable adapter parameters, got {actual}")
    print(f"{method} adapter parameters: {actual}", flush=True)


runner.collect_gradients = prepare
runner.build_plan = lambda *args, **kwargs: allocation
runner.apply_plan = attach
sys.argv = [
    sys.argv[0], "--task", "sst2", "--method", method, "--seed", "42",
    "--device", "cuda:0", "--output-root", str(output / "results"),
    "--checkpoint", str(output / f"{method}.pt"),
]
runner.main()
