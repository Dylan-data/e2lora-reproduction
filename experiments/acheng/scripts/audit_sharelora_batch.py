#!/usr/bin/env python3
"""Audit one real DTD training batch with the same ShareLoRA adapter installer."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from e2lora.model.clip_classification import logits, text_features
from train_image_adapter import (
    _audit_split,
    _cached_snapshot,
    _config_hash,
    _make_batches,
    _parameter_snapshot,
    _seed,
    _source_hashes,
    _stratified_split,
    _tensor_hash,
    _write_json,
    install_image_adapters,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "one_batch_pilot" or config["method"]["name"] != "ShareLoRA":
        raise ValueError("A ShareLoRA one-batch pilot config is required.")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir() or (run_dir / "result.json").exists():
        raise ValueError("Run directory must exist and must not contain a result.")
    if json.loads((run_dir / "config.json").read_text(encoding="utf-8")) != config:
        raise ValueError("Run-directory configuration mismatch.")
    start = time.perf_counter()
    _seed(config["training"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    ds = config["dataset"]
    train = load_dataset(ds["repo"], revision=ds["revision"], split=ds["train_split"], cache_dir=ds["cache_dir"])
    classes = list(train.features["label"].names)
    if classes != ds["expected_class_names"]:
        raise ValueError("Dataset classes do not match the fixed configuration.")
    _audit_split(train, ds["train_split"], classes, ds["expected_train_count"])
    train_indices, validation_indices = _stratified_split(
        train["label"], classes, ds["validation_fraction"], ds["validation_seed"]
    )
    baseline = json.loads(Path(
        "experiments/acheng/outputs/DTD/LoRA-r8/2026/20261119T173117+0800/splits.json"
    ).read_text(encoding="utf-8"))
    if train_indices != baseline["training_indices"] or validation_indices != baseline["validation_indices"]:
        raise ValueError("Pilot train/validation indices differ from the accepted DTD baseline.")
    mc = config["model"]
    processor = CLIPProcessor.from_pretrained(
        str(_cached_snapshot(mc["cache_dir"], mc["repo"], mc["revision"])), use_fast=False
    )
    model = CLIPModel.from_pretrained(
        str(_cached_snapshot(mc["cache_dir"], mc["repo"], mc["converted_safetensors_revision"])),
        use_safetensors=True, attn_implementation=mc["attn_implementation"],
    ).float().cuda().eval()
    model.requires_grad_(False)
    text, prompts = text_features(model, processor, classes)
    plan, families, count = install_image_adapters(model, config["method"])
    if count != 718848 or len(plan) != 72 or len(families) != 6:
        raise AssertionError(f"Unexpected ShareLoRA structure: {count}, {len(plan)}, {len(families)}")
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    before = _parameter_snapshot(parameters)
    frozen_before = {name: _tensor_hash(p) for name, p in model.named_parameters() if not p.requires_grad}
    ordered = random.Random(config["training"]["seed"]).sample(train_indices, len(train_indices))
    ids, pixels, labels = next(_make_batches(train, ordered, processor, torch.device("cuda"), 64))
    optimizer = torch.optim.Adam(parameters.values(), lr=config["training"]["learning_rate"],
                                 betas=tuple(config["training"]["betas"]), weight_decay=0)
    total_steps = math.ceil(len(train_indices) / config["training"]["batch_size"]) * config["training"]["max_epochs"]
    warmup_steps = max(1, round(total_steps * config["training"]["warmup_ratio"]))
    torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, max(0.0, (step + 1) / warmup_steps))
    )
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(logits(model, pixels, text), labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite pilot loss.")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
        raise FloatingPointError("Missing or non-finite adapter gradients.")
    down_grad_norms = {family: float(model.get_submodule(names[0]).down.weight.grad.norm().item())
                       for family, names in families.items()}
    optimizer.step()
    changed = [name for name, p in parameters.items() if not torch.equal(p.detach().cpu(), before[name])]
    changed_up = [name for name in changed if name.endswith(".up.weight")]
    changed_down = [name for name in changed if name.endswith(".down.weight")]
    if not changed_up:
        raise AssertionError("No B/up adapter was updated.")
    frozen_after = {name: _tensor_hash(p) for name, p in model.named_parameters() if not p.requires_grad}
    if frozen_before != frozen_after:
        raise AssertionError("Frozen weights changed.")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "status": "completed_pilot", "experiment_id": config["experiment_id"],
        "config_sha256": _config_hash(config), "source_sha256": _source_hashes(),
        "dataset_revision": ds["revision"], "model_revision": mc["revision"], "seed": config["training"]["seed"],
        "sample_ids": ids, "sample_source": "pinned DTD train split; first training batch",
        "classes": classes, "prompts": prompts, "training_count": len(train_indices),
        "validation_count": len(validation_indices), "sharing": config["method"]["sharing"],
        "projection_families": {family: {"layers": names, "shared_A_shape": list(model.get_submodule(names[0]).down.weight.shape),
                                       "independent_B_shape": list(model.get_submodule(names[0]).up.weight.shape)}
                                for family, names in families.items()},
        "unique_shared_A_objects": 6, "unique_B_objects": 72,
        "adapter_trainable_parameters": count, "total_trainable_parameters": count,
        "loss": float(loss.item()), "updated_up_count": len(changed_up), "updated_down_count": len(changed_down),
        "updated_parameter_names": changed, "shared_A_gradient_norms": down_grad_norms,
        "initialization_note": "B starts at zero; first-step A gradients and A updates are zero by construction.",
        "frozen_parameters_unchanged": True, "gpu": torch.cuda.get_device_name(),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600, "test_scored": False,
    }
    if not math.isfinite(result["loss"]):
        raise FloatingPointError("Non-finite pilot loss.")
    _write_json(run_dir / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "adapter_trainable_parameters", "loss",
                                                    "updated_up_count", "updated_down_count", "gpu_hours")}), flush=True)


if __name__ == "__main__":
    main()
