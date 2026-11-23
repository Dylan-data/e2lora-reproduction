"""Evaluate the saved MNLI DoRA rank-4 weights without further training."""
import importlib.util
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "train_glue", root / "experiments/text-understanding/train_glue.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
checkpoint = root / "outputs/dylan/glue-full-seed42/checkpoints/mnli-dora-r4.pt"
state = runner.torch.load(checkpoint, map_location="cpu", weights_only=True)
runner.SETUP["tasks"]["mnli"]["epochs"] = state["epoch"]
sys.argv = [
    sys.argv[0], "--task", "mnli", "--method", "dora-r4", "--seed", "42",
    "--device", "cpu", "--checkpoint", str(checkpoint),
    "--output-root", str(root / "outputs/dylan/mnli-dora-r4-intermediate-evaluation"),
]
runner.main()
