#!/usr/bin/env python3
"""Derive one formal ShareLoRA configuration from an accepted LoRA config."""

import argparse
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    source.relative_to((TASK / "configs").resolve())
    config = json.loads(source.read_text(encoding="utf-8"))
    dataset = config["dataset"]["name"]
    slug = dataset.lower().replace("-", "")
    if config["method"]["name"] != "LoRA" or config["method"]["rank"] != 8:
        raise ValueError("Source must be the fixed rank-8 LoRA configuration.")
    config["experiment_id"] = f"table3/{slug}/sharelora-r8/seed-2026"
    config["status"] = "formal"
    config["method"]["name"] = "ShareLoRA"
    config["method"]["sharing"] = (
        "One shared A object per target projection family (q_proj, k_proj, v_proj, "
        "out_proj, fc1, fc2); independent B per target layer. This is the validated "
        "team interpretation of image ShareLoRA sharing."
    )
    config["execution"]["run_root"] = f"experiments/acheng/outputs/{dataset}/ShareLoRA-r8/2026/"
    config.setdefault("provenance", {}).setdefault("team_choices", []).append(
        "ShareLoRA: projection-family shared A objects and per-layer B objects; strict E2LoRA claim not made"
    )
    destination = TASK / "configs" / f"{slug}_sharelora_seed2026.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != config:
            raise FileExistsError(f"Refusing to overwrite differing config: {destination}")
        print(destination)
        return
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
