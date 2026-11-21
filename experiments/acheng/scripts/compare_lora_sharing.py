#!/usr/bin/env python3
"""Compare saved LoRA and ShareLoRA object-audit evidence without model execution."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments" / "acheng"
LORA = TASK / "outputs/DTD/LoRA-r8/2026/20261119T173117+0800/result.json"
SHARE = TASK / "outputs/DTD/ShareLoRA-r8/2026/20261120T165158+0800/result.json"
OUTPUT = TASK / "reports/lora-sharelora-structural-comparison.json"


def main():
    baseline = json.loads(LORA.read_text(encoding="utf-8"))
    pilot = json.loads(SHARE.read_text(encoding="utf-8"))
    if baseline["model_revision"] != pilot["model_revision"] or baseline["rank"] != 8:
        raise ValueError("The two recorded experiments do not share the same model revision/rank.")
    if baseline["adapter_trainable_parameters"] != 1327104 or pilot["adapter_trainable_parameters"] != 718848:
        raise ValueError("Unexpected recorded parameter count.")
    if not baseline["frozen_parameters_unchanged"] or not pilot["frozen_parameters_unchanged"]:
        raise ValueError("The frozen parameter audit did not pass.")
    names = {name for family in pilot["projection_families"].values() for name in family["layers"]}
    if names != set(baseline["target_layers"]) or len(names) != 72:
        raise ValueError("The target layer sets do not match.")
    updated = set(baseline["updated_parameter_names"])
    if updated != {f"{name}.{factor}.weight" for name in names for factor in ("down", "up")}:
        raise ValueError("LoRA baseline did not record all 144 unique adapter updates.")
    if (pilot["unique_shared_A_objects"], pilot["unique_B_objects"], pilot["updated_up_count"]) != (6, 72, 72):
        raise ValueError("ShareLoRA pilot sharing or updates did not pass.")
    inventory = {x["name"]: x for x in baseline["target_layer_inventory"]}
    targets = []
    for family, info in pilot["projection_families"].items():
        a = info["shared_A_shape"]
        b = info["independent_B_shape"]
        if len(info["layers"]) != 12:
            raise ValueError(f"Unexpected family length: {family}")
        for layer in info["layers"]:
            if inventory[layer]["shape"] != [b[0], a[1]] or a[0] != 8 or b[1] != 8:
                raise ValueError(f"Layer dimensions disagree: {layer}")
            targets.append({
                "layer": layer, "family": family, "base_shape": inventory[layer]["shape"],
                "A_shape": a, "B_shape": b,
                "lora_A_object": f"lora:{layer}:A", "lora_B_object": f"lora:{layer}:B",
                "share_A_object": f"share:{family}:A", "share_B_object": f"share:{layer}:B",
                "base_weight_frozen": True, "adapter_A_trainable": True, "adapter_B_trainable": True,
            })
    lora_formula = sum(t["A_shape"][0] * t["A_shape"][1] + t["B_shape"][0] * t["B_shape"][1] for t in targets)
    shared_a = sum(info["shared_A_shape"][0] * info["shared_A_shape"][1]
                   for info in pilot["projection_families"].values())
    shared_b = sum(t["B_shape"][0] * t["B_shape"][1] for t in targets)
    if (lora_formula, shared_a, shared_b) != (1327104, 55296, 663552):
        raise ValueError("The parameter formula does not agree with the recorded structures.")
    result = {
        "status": "verified_from_recorded_baseline_and_single_batch_pilot",
        "model": baseline["model"], "model_revision": baseline["model_revision"],
        "dataset": "DTD", "seed": 2026, "rank": 8, "alpha": baseline["alpha"],
        "object_identity_note": "Stable labels describe equality groups proved by in-memory identity checks, not persisted Python addresses. LoRA uniqueness is evidenced by named_parameters deduplication and all 144 recorded updated tensor names.",
        "lora": {"A_objects": 72, "B_objects": 72, "parameters": baseline["adapter_trainable_parameters"],
                 "formula": "12*(55296 A + 55296 B) = 1327104"},
        "sharelora": {"A_objects": 6, "B_objects": 72, "A_parameters": shared_a, "B_parameters": shared_b,
                      "parameters": pilot["adapter_trainable_parameters"],
                      "formula": "55296 shared A + 663552 independent B = 718848",
                      "first_batch_loss": pilot["loss"], "first_batch_B_updates": pilot["updated_up_count"],
                      "first_batch_A_updates": pilot["updated_down_count"],
                      "first_batch_A_zero_gradient_reason": "B was initialized to zero"},
        "frozen_and_trainable": "Original model weights and text branch frozen; both A and B trainable; both audits confirmed frozen weights unchanged.",
        "scope": "Paper says A globally shared and B layer-specific; exact partition of differently shaped image projections is not specified. Sharing one A per projection family across 12 layers is the recorded team interpretation, not a proven original implementation detail.",
        "rounding": "718848 rounds to 0.72M as in the paper. 1327104 rounds to 1.33M rather than the paper's displayed 1.31M; rounding/target-scope difference alone is not proof of error.",
        "targets": targets,
        "sources": [str(LORA.relative_to(ROOT)), str(SHARE.relative_to(ROOT))],
    }
    data = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != data:
        raise FileExistsError(f"Refusing to overwrite differing evidence: {OUTPUT}")
    if not OUTPUT.exists():
        OUTPUT.write_text(data, encoding="utf-8")
    print("Verified", len(targets), "targets;", OUTPUT)
    print("LoRA A/B 72/72", lora_formula, "ShareLoRA A/B 6/72", shared_a + shared_b)


if __name__ == "__main__":
    main()
