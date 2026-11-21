#!/usr/bin/env python3
"""Derive the requested single-unit configs from the accepted DTD baseline."""

import copy
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
base = json.loads((TASK / "configs/dtd_lora_seed2026.json").read_text(encoding="utf-8"))
for filename, experiment, name, rank, phase in (
    ("dtd_sharelora_pilot_seed2026.json", "pilot/dtd/sharelora-r8/seed-2026", "ShareLoRA", 8, "one_batch_pilot"),
    ("dtd_sharelora_seed2026.json", "table3/dtd/sharelora-r8/seed-2026", "ShareLoRA", 8, "formal"),
    ("dtd_lora_r4_seed2026.json", "table13/dtd/lora-r4/seed-2026", "LoRA", 4, "formal"),
):
    config = copy.deepcopy(base)
    config["experiment_id"] = experiment
    config["status"] = phase
    config["method"]["name"] = name
    config["method"]["rank"] = rank
    config["method"]["sharing"] = (
        "A/down shared across layers within each projection family, B/up layer-specific"
        if name == "ShareLoRA" else "independent A/down and B/up per layer"
    )
    config["execution"]["run_root"] = f"experiments/acheng/outputs/DTD/{name}-r{rank}/2026/"
    path = TASK / "configs" / filename
    serialized = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Existing config differs: {path}")
    if not path.exists():
        path.write_text(serialized, encoding="utf-8")
    print(path)
