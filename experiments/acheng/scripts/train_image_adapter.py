#!/usr/bin/env python
"""Train one LoRA or ShareLoRA image unit and preserve resumable state."""

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from e2lora.adapters.lora_layers import apply_plan, trainable_count
from e2lora.model.clip_classification import (
    CONVERTED,
    MODEL,
    ORIGINAL,
    logits,
    text_features,
    visual_families,
)
from transformers import CLIPModel, CLIPProcessor


_checkpoint_requested = False


def _request_checkpoint(signum, _frame):
    global _checkpoint_requested
    _checkpoint_requested = True
    print(f"Received signal {signum}; will checkpoint after the current batch.", flush=True)


def _write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _write_or_verify_json(path, value):
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(value, ensure_ascii=False)):
            raise ValueError(f"Existing run metadata differs; refusing to overwrite {path}.")
        return
    _write_json(path, value)


def _config_hash(config):
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_hashes():
    paths = {
        Path(__file__).resolve(),
        Path(apply_plan.__code__.co_filename).resolve(),
        Path(visual_families.__code__.co_filename).resolve(),
        Path(logits.__code__.co_filename).resolve(),
    }
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_snapshot(cache_dir, repo, revision, *, hub_subdirectory=True):
    cache_root = Path(cache_dir).resolve()
    hub_root = cache_root / "hub" if hub_subdirectory else cache_root
    snapshot = (
        hub_root
        / f"models--{repo.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned model snapshot is not cached: {snapshot}")
    resolved = snapshot.resolve()
    try:
        resolved.relative_to(cache_root)
    except ValueError as error:
        raise ValueError(f"Cached model snapshot escapes the configured cache: {resolved}") from error
    return resolved


def _record_weight_provenance(model_config, run_dir):
    from safetensors import safe_open

    original_cache = Path(__file__).resolve().parents[1] / "outputs" / "cache" / "model-source"
    original_snapshot = _cached_snapshot(
        original_cache,
        model_config["repo"],
        model_config["revision"],
        hub_subdirectory=False,
    )
    original_path = original_snapshot / "pytorch_model.bin"
    if not original_path.is_file():
        raise FileNotFoundError(f"Pinned original PyTorch weights are not cached: {original_path}")
    converted_snapshot = _cached_snapshot(
        model_config["cache_dir"],
        model_config["repo"],
        model_config["converted_safetensors_revision"],
    )
    converted_path = converted_snapshot / "model.safetensors"
    if not converted_path.is_file():
        raise FileNotFoundError(f"Pinned safetensors file is not cached: {converted_path}")
    original_state = torch.load(original_path, map_location="cpu", weights_only=True)
    if not isinstance(original_state, dict):
        raise TypeError("Pinned original PyTorch weights are not a state dictionary.")
    with safe_open(str(converted_path), framework="pt", device="cpu") as checkpoint:
        tensor_names = sorted(checkpoint.keys())
    if not tensor_names:
        raise ValueError("Pinned safetensors checkpoint contains no tensors.")
    if set(original_state) != set(tensor_names):
        missing = sorted(set(original_state) - set(tensor_names))
        extra = sorted(set(tensor_names) - set(original_state))
        raise ValueError(f"Original/safetensors key mismatch; missing={missing}, extra={extra}.")
    with safe_open(str(converted_path), framework="pt", device="cpu") as checkpoint:
        for name in tensor_names:
            original_tensor = original_state[name]
            converted_tensor = checkpoint.get_tensor(name)
            if not isinstance(original_tensor, torch.Tensor):
                raise TypeError(f"Original weight {name!r} is not a tensor.")
            if original_tensor.shape != converted_tensor.shape or original_tensor.dtype != converted_tensor.dtype:
                raise ValueError(f"Original/safetensors shape or dtype mismatch for {name}.")
            if not torch.equal(original_tensor, converted_tensor):
                raise ValueError(f"Original/safetensors tensor values differ for {name}.")
    manifest = {
        "repo": model_config["repo"],
        "original_pytorch_revision": model_config["revision"],
        "original_snapshot": str(original_snapshot),
        "original_pytorch_file": "pytorch_model.bin",
        "original_pytorch_file_sha256": _file_hash(original_path),
        "converted_revision": model_config["converted_safetensors_revision"],
        "converted_snapshot": str(converted_snapshot),
        "converted_file": str(converted_path),
        "converted_sha256": _file_hash(converted_path),
        "tensor_count": len(tensor_names),
        "tensor_names_sha256": hashlib.sha256(
            "\n".join(tensor_names).encode("utf-8")
        ).hexdigest(),
        "format": "official safetensors checkpoint compared with pinned original weights",
        "equivalence_to_original_pytorch_weights": True,
        "loading_safety": "original weights loaded with weights_only=True; safetensors inspected with safetensors.safe_open",
    }
    del original_state
    _write_or_verify_json(run_dir / "model_weight_provenance.json", manifest)
    print("Pinned original and official safetensors tensors match exactly.", flush=True)


def _parameter_snapshot(parameters):
    return {name: value.detach().cpu().clone() for name, value in parameters.items()}


def _copy_parameters(parameters, values):
    if set(parameters) != set(values):
        raise ValueError("Checkpoint adapter parameter names do not match this run.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            saved = values[name]
            if parameter.shape != saved.shape:
                raise ValueError(f"Checkpoint shape mismatch for adapter parameter {name}.")
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)


def _stratified_split(labels, classes, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    pools = [[] for _ in classes]
    for index, label in enumerate(labels):
        if type(label) is not int or not 0 <= label < len(classes):
            raise ValueError(f"Invalid training label at index {index}: {label!r}")
        pools[label].append(index)
    rng = random.Random(seed)
    train_indices, validation_indices = [], []
    for label, pool in enumerate(pools):
        if len(pool) < 2:
            raise ValueError(f"Class {classes[label]!r} has too few training examples.")
        rng.shuffle(pool)
        validation_count = max(1, round(len(pool) * fraction))
        validation_indices.extend(pool[:validation_count])
        train_indices.extend(pool[validation_count:])
    if set(train_indices) & set(validation_indices):
        raise AssertionError("Training and validation indices overlap.")
    return train_indices, validation_indices


def _audit_split(dataset, split_name, classes, expected_count):
    if len(dataset) != expected_count:
        raise ValueError(
            f"{split_name} split has {len(dataset)} examples; expected {expected_count}."
        )
    feature_names = list(dataset.features["label"].names)
    if feature_names != classes:
        raise ValueError(f"{split_name} ClassLabel names/order differ from the train split.")
    counts = [0] * len(classes)
    for index, label in enumerate(dataset["label"]):
        if type(label) is not int or not 0 <= label < len(classes):
            raise ValueError(f"Invalid {split_name} label at index {index}: {label!r}")
        counts[label] += 1
    missing = [classes[index] for index, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(f"{split_name} has no examples for classes: {missing}.")
    return dict(zip(classes, counts))


def _expected_split_count(dataset_config, split_name):
    if f"expected_{split_name}_count" in dataset_config:
        return dataset_config[f"expected_{split_name}_count"]
    manifest_path = Path(__file__).resolve().parents[1] / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        item
        for item in manifest["datasets"]
        if item.get("source") == dataset_config["repo"]
        and item.get("revision") == dataset_config["revision"]
    ]
    if len(matches) != 1 or split_name not in matches[0]["splits"]:
        raise ValueError(
            f"No unique pinned expected count for {dataset_config['repo']} "
            f"revision {dataset_config['revision']} split {split_name}."
        )
    count = matches[0]["splits"][split_name]
    if type(count) is not int:
        raise ValueError(f"Dataset manifest count for {split_name} is not an integer: {count!r}.")
    return count


def _make_batches(dataset, indices, processor, device, batch_size):
    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        rows = [dataset[index] for index in batch_indices]
        images = [row["image"].convert("RGB") for row in rows]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long, device=device)
        yield batch_indices, pixels, labels


@torch.no_grad()
def _evaluate(model, dataset, indices, processor, text, device, batch_size):
    labels_out, predictions_out, indices_out = [], [], []
    for batch_indices, pixels, labels in _make_batches(
        dataset, indices, processor, device, batch_size
    ):
        predictions = logits(model, pixels, text).argmax(-1)
        labels_out.extend(labels.cpu().tolist())
        predictions_out.extend(predictions.cpu().tolist())
        indices_out.extend(batch_indices)
    correct = sum(label == prediction for label, prediction in zip(labels_out, predictions_out))
    return {
        "count": len(labels_out),
        "correct": correct,
        "accuracy": 100.0 * correct / len(labels_out),
        "indices": indices_out,
        "labels": labels_out,
        "predictions": predictions_out,
    }


def _save_checkpoint(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def install_image_adapters(model, method):
    name = method["name"]
    if name not in ("LoRA", "ShareLoRA"):
        raise ValueError(f"Unsupported image adapter method: {name}")
    families = visual_families(model)
    plan = [
        {"layers": [layer], "rank": method["rank"], "family": family}
        for family, names in families.items()
        for layer in names
    ]
    apply_plan(model, plan, alpha=method["alpha"], global_down=name == "ShareLoRA")
    expected_count = 0
    for family, names in families.items():
        layers = [model.get_submodule(layer) for layer in names]
        downs = [layer.down.weight for layer in layers]
        ups = [layer.up.weight for layer in layers]
        if len({id(p) for p in ups}) != len(ups):
            raise AssertionError(f"Up factors are not independent in {family}.")
        expected_downs = 1 if name == "ShareLoRA" else len(layers)
        if len({id(p) for p in downs}) != expected_downs:
            raise AssertionError(f"Incorrect down-factor sharing in {family}.")
        expected_count += expected_downs * downs[0].numel() + sum(p.numel() for p in ups)
    all_downs = [model.get_submodule(names[0]).down.weight for names in families.values()]
    if len({id(p) for p in all_downs}) != len(all_downs):
        raise AssertionError("Different projection families unexpectedly share a down factor.")
    count = trainable_count(model)
    if count != expected_count:
        raise AssertionError(f"Adapter count mismatch: observed={count}, expected={expected_count}.")
    return plan, families, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory must be created before submission: {run_dir}")
    result_path = run_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: {result_path}")
    config_hash = _config_hash(config)
    source_hashes = _source_hashes()
    saved_config = run_dir / "config.json"
    if not saved_config.exists():
        raise FileNotFoundError(f"Run config is missing from {run_dir}.")
    if _config_hash(json.loads(saved_config.read_text(encoding="utf-8"))) != config_hash:
        raise ValueError("Submitted config differs from the immutable run-directory config.")

    signal.signal(signal.SIGUSR1, _request_checkpoint)
    _seed(config["training"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    device = torch.device("cuda")

    dataset_config = config["dataset"]
    train_dataset = load_dataset(
        dataset_config["repo"],
        revision=dataset_config["revision"],
        split=dataset_config["train_split"],
        cache_dir=dataset_config["cache_dir"],
    )
    test_dataset = load_dataset(
        dataset_config["repo"],
        revision=dataset_config["revision"],
        split=dataset_config["test_split"],
        cache_dir=dataset_config["cache_dir"],
    )
    classes = list(train_dataset.features["label"].names)
    if len(classes) != dataset_config["expected_class_count"]:
        raise ValueError(f"Expected {dataset_config['expected_class_count']} classes, got {len(classes)}.")
    if classes != dataset_config["expected_class_names"]:
        raise ValueError("Dataset ClassLabel names/order differ from the pinned run configuration.")
    train_label_counts = _audit_split(
        train_dataset,
        dataset_config["train_split"],
        classes,
        _expected_split_count(dataset_config, "train"),
    )
    test_label_counts = _audit_split(
        test_dataset,
        dataset_config["test_split"],
        classes,
        _expected_split_count(dataset_config, "test"),
    )
    dataset_audit = {
        "repo": dataset_config["repo"],
        "revision": dataset_config["revision"],
        "train_split": dataset_config["train_split"],
        "test_split": dataset_config["test_split"],
        "train_count": len(train_dataset),
        "test_count": len(test_dataset),
        "expected_class_count": dataset_config["expected_class_count"],
        "classes_in_label_order": classes,
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "sample_id_scheme": "dataset revision + source split + zero-based row index",
        "paper_split_equivalence": "not established; pinned mirror split",
    }
    _write_or_verify_json(run_dir / "dataset_audit.json", dataset_audit)

    model_config = config["model"]
    if model_config["repo"] != MODEL or model_config["revision"] != ORIGINAL:
        raise ValueError("This runner only accepts its pinned CLIP model configuration.")
    if model_config["converted_safetensors_revision"] != CONVERTED:
        raise ValueError("The converted weight revision differs from the shared CLIP loader pin.")
    _record_weight_provenance(model_config, run_dir)
    processor_snapshot = _cached_snapshot(
        model_config["cache_dir"], model_config["repo"], model_config["revision"]
    )
    processor = CLIPProcessor.from_pretrained(str(processor_snapshot), use_fast=False)
    model_snapshot = _cached_snapshot(
        model_config["cache_dir"],
        model_config["repo"],
        model_config["converted_safetensors_revision"],
    )
    model = CLIPModel.from_pretrained(
        str(model_snapshot),
        use_safetensors=True,
        attn_implementation=model_config["attn_implementation"],
    ).float().to(device).eval()
    model.requires_grad_(False)
    text, prompts = text_features(model, processor, classes)

    validation_seed = dataset_config.get("validation_seed", config["training"]["seed"])
    train_indices, validation_indices = _stratified_split(
        train_dataset["label"],
        classes,
        dataset_config["validation_fraction"],
        validation_seed,
    )
    test_indices = list(range(len(test_dataset)))
    split = {
        "dataset": dataset_config["repo"],
        "revision": dataset_config["revision"],
        "train_source_split": dataset_config["train_split"],
        "test_source_split": dataset_config["test_split"],
        "train_source_count": len(train_dataset),
        "test_count": len(test_dataset),
        "training_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "sample_id_scheme": "dataset revision + source split + zero-based row index",
        "classes": classes,
        "validation_source": dataset_config["validation_source"],
        "validation_fraction": dataset_config["validation_fraction"],
        "seed": config["training"]["seed"],
    }
    if "validation_seed" in dataset_config:
        split["validation_seed"] = validation_seed
    _write_or_verify_json(run_dir / "splits.json", split)
    _write_or_verify_json(run_dir / "classes.json", {"classes": classes, "prompts": prompts})
    _write_or_verify_json(
        run_dir / "preprocessing.json", processor.image_processor.to_dict()
    )

    families = visual_families(model)
    plan = [
        {"layers": [name], "rank": config["method"]["rank"], "family": family}
        for family, names in families.items()
        for name in names
    ]
    target_names = [group["layers"][0] for group in plan]
    all_linear = [
        {
            "name": name,
            "shape": list(module.weight.shape),
            "targeted": name in target_names,
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    _write_or_verify_json(
        run_dir / "layer_inventory.json",
        {"target_layers": target_names, "all_linear_layers": all_linear},
    )
    installed_plan, _, count = install_image_adapters(model, config["method"])
    if installed_plan != plan:
        raise AssertionError("Adapter targets changed during installation.")
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    initial_adapter = _parameter_snapshot(parameters)
    frozen_before = {
        name: _tensor_hash(parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }

    training = config["training"]
    batch_size = training["batch_size"]
    batches_per_epoch = math.ceil(len(train_indices) / batch_size)
    total_steps = batches_per_epoch * training["max_epochs"]
    warmup_steps = max(1, round(total_steps * training["warmup_ratio"]))
    optimizer = torch.optim.Adam(
        parameters.values(),
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, max(0.0, (step + 1) / warmup_steps)),
    )
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    history = []
    best_accuracy = -math.inf
    best_epoch = None
    best_adapter = _parameter_snapshot(parameters)
    best_validation = None
    epoch_start = 0
    batch_start = 0
    epoch_order = []
    running_loss_sum = 0.0
    running_examples = 0
    global_step = 0
    checkpoint_epoch = 0
    checkpoint_next_batch = 0
    prior_elapsed_seconds = 0.0
    prior_training_seconds = 0.0
    prior_prepare_seconds = 0.0
    prior_evaluation_seconds = 0.0
    prior_peak_cuda_bytes = 0

    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("Checkpoint config hash does not match this run.")
        if checkpoint.get("source_sha256") != source_hashes:
            raise ValueError("Checkpoint code hashes differ from the current run implementation.")
        if checkpoint["split"] != split:
            raise ValueError("Checkpoint dataset split differs from the current deterministic split.")
        _copy_parameters(parameters, checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        random.setstate(tuple(checkpoint["python_rng"]))
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        history = checkpoint["history"]
        best_accuracy = checkpoint["best_accuracy"]
        best_epoch = checkpoint["best_epoch"]
        best_adapter = checkpoint["best_adapter"]
        best_validation = checkpoint["best_validation"]
        epoch_start = checkpoint["epoch"]
        batch_start = checkpoint["next_batch"]
        epoch_order = checkpoint["epoch_order"]
        running_loss_sum = checkpoint["running_loss_sum"]
        running_examples = checkpoint["running_examples"]
        global_step = checkpoint["global_step"]
        prior_elapsed_seconds = checkpoint.get("elapsed_seconds", 0.0)
        prior_training_seconds = checkpoint.get("training_seconds", 0.0)
        prior_prepare_seconds = checkpoint.get("prepare_seconds", 0.0)
        prior_evaluation_seconds = checkpoint.get("evaluation_seconds", 0.0)
        prior_peak_cuda_bytes = checkpoint.get("peak_cuda_bytes", 0)
        print(
            f"Resuming {config['experiment_id']} at epoch {epoch_start + 1}, "
            f"batch {batch_start}, global step {global_step}.",
            flush=True,
        )
    elif checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint exists; pass --resume instead of overwriting {checkpoint_path}.")

    model.eval()
    segment_prepare_seconds = time.perf_counter() - start_time
    segment_training_seconds = 0.0
    segment_evaluation_seconds = 0.0
    stop_requested = False
    for epoch in range(epoch_start, training["max_epochs"]):
        if epoch != epoch_start or not epoch_order:
            epoch_order = random.Random(training["seed"] + epoch).sample(
                train_indices, len(train_indices)
            )
            batch_start = 0
            running_loss_sum = 0.0
            running_examples = 0
        epoch_batches = math.ceil(len(epoch_order) / batch_size)
        for batch_index in range(batch_start, epoch_batches):
            batch_started = time.perf_counter()
            indices = epoch_order[batch_index * batch_size : (batch_index + 1) * batch_size]
            batch = next(_make_batches(train_dataset, indices, processor, device, batch_size))
            _, pixels, labels = batch
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(logits(model, pixels, text), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at global step {global_step}.")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for parameter in parameters.values()):
                raise FloatingPointError(f"Missing/non-finite adapter gradient at step {global_step}.")
            optimizer.step()
            scheduler.step()
            segment_training_seconds += time.perf_counter() - batch_started
            loss_value = float(loss.detach().item())
            running_loss_sum += loss_value * len(indices)
            running_examples += len(indices)
            global_step += 1
            next_batch = batch_index + 1
            if global_step % 10 == 0:
                print(
                    f"epoch={epoch + 1}/{training['max_epochs']} "
                    f"batch={next_batch}/{epoch_batches} loss={loss_value:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.9g}",
                    flush=True,
                )

            if global_step % training["checkpoint_interval_steps"] == 0 or _checkpoint_requested:
                payload = {
                    "config_sha256": config_hash,
                    "source_sha256": source_hashes,
                    "split": split,
                    "adapter": _parameter_snapshot(parameters),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "epoch": epoch,
                    "next_batch": next_batch,
                    "epoch_order": epoch_order,
                    "running_loss_sum": running_loss_sum,
                    "running_examples": running_examples,
                    "global_step": global_step,
                    "history": history,
                    "best_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "best_adapter": best_adapter,
                    "best_validation": best_validation,
                    "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - start_time,
                    "training_seconds": prior_training_seconds + segment_training_seconds,
                    "prepare_seconds": prior_prepare_seconds + segment_prepare_seconds,
                    "evaluation_seconds": prior_evaluation_seconds + segment_evaluation_seconds,
                    "peak_cuda_bytes": max(
                        prior_peak_cuda_bytes, torch.cuda.max_memory_allocated(device)
                    ),
                }
                _save_checkpoint(checkpoint_path, payload)
                checkpoint_epoch = epoch + 1
                checkpoint_next_batch = next_batch
            if _checkpoint_requested:
                stop_requested = True
                batch_start = next_batch
                print(
                    f"Checkpointed at epoch={epoch + 1}, batch={next_batch}, "
                    f"global_step={global_step}; submit a continuation with --resume.",
                    flush=True,
                )
                break

        if stop_requested:
            break
        batch_start = 0
        evaluation_started = time.perf_counter()
        validation = _evaluate(
            model,
            train_dataset,
            validation_indices,
            processor,
            text,
            device,
            batch_size,
        )
        segment_evaluation_seconds += time.perf_counter() - evaluation_started
        epoch_loss = running_loss_sum / running_examples
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "validation_accuracy": validation["accuracy"],
            "validation_correct": validation["correct"],
            "validation_count": validation["count"],
            "global_step": global_step,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch + 1
            best_adapter = _parameter_snapshot(parameters)
            best_validation = validation
        if not math.isfinite(best_accuracy):
            raise FloatingPointError("Validation selection produced a non-finite score.")

        payload = {
            "config_sha256": config_hash,
            "source_sha256": source_hashes,
            "split": split,
            "adapter": _parameter_snapshot(parameters),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "epoch": epoch + 1,
            "next_batch": 0,
            "epoch_order": [],
            "running_loss_sum": 0.0,
            "running_examples": 0,
            "global_step": global_step,
            "history": history,
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "best_adapter": best_adapter,
            "best_validation": best_validation,
            "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - start_time,
            "training_seconds": prior_training_seconds + segment_training_seconds,
            "prepare_seconds": prior_prepare_seconds + segment_prepare_seconds,
            "evaluation_seconds": prior_evaluation_seconds + segment_evaluation_seconds,
            "peak_cuda_bytes": max(
                prior_peak_cuda_bytes, torch.cuda.max_memory_allocated(device)
            ),
        }
        _save_checkpoint(checkpoint_path, payload)

    if stop_requested:
        at_epoch_boundary = checkpoint_next_batch >= epoch_batches
        _write_json(
            run_dir / "checkpoint_status.json",
            {
                "status": "checkpointed_for_time_limit",
                "config_sha256": config_hash,
                "checkpoint": str(checkpoint_path),
                "next_epoch": checkpoint_epoch + int(at_epoch_boundary),
                "global_step": global_step,
                "next_batch": 0 if at_epoch_boundary else checkpoint_next_batch,
                "resume_command": f"experiments/acheng/scripts/submit_lora.sh --resume {run_dir}",
            },
        )
        return 0

    if best_epoch is None or best_validation is None:
        raise RuntimeError("Training completed without a best validation checkpoint.")
    _copy_parameters(parameters, best_adapter)
    evaluation_started = time.perf_counter()
    test_result = _evaluate(
        model,
        test_dataset,
        test_indices,
        processor,
        text,
        device,
        batch_size,
    )
    segment_evaluation_seconds += time.perf_counter() - evaluation_started
    frozen_after = {
        name: _tensor_hash(parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    if frozen_before != frozen_after:
        raise AssertionError("A frozen parameter changed during adapter training.")
    changed = [
        name
        for name, parameter in parameters.items()
        if not torch.equal(parameter.detach().cpu(), initial_adapter[name])
    ]
    if not changed:
        raise AssertionError("No trainable adapter parameter changed.")
    torch.cuda.synchronize()
    elapsed = prior_elapsed_seconds + time.perf_counter() - start_time
    result = {
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "source_sha256": source_hashes,
        "model": MODEL,
        "model_revision": ORIGINAL,
        "converted_safetensors_revision": CONVERTED,
        "dataset": dataset_config["repo"],
        "dataset_revision": dataset_config["revision"],
        "method": config["method"]["name"],
        "sharing": (
            "A/down shared across layers within each projection family; B/up independent per layer"
            if config["method"]["name"] == "ShareLoRA"
            else "independent A/down and B/up per layer"
        ),
        "rank": config["method"]["rank"],
        "alpha": config["method"]["alpha"],
        "seed": training["seed"],
        "classes": classes,
        "prompts": prompts,
        "target_layers": target_names,
        "target_layer_inventory": all_linear,
        "physical_rank_per_target_layer": config["method"]["rank"],
        "trainable_rank_components_per_target_layer": config["method"]["rank"],
        "adapter_trainable_parameters": count,
        "total_trainable_parameters": trainable_count(model),
        "classification_head_parameters": 0,
        "updated_parameter_names": changed,
        "frozen_parameters_unchanged": True,
        "best_epoch": best_epoch,
        "max_epochs": training["max_epochs"],
        "history": history,
        "validation": best_validation,
        "test": test_result,
        "optimizer": {
            "name": "Adam",
            "learning_rate": training["learning_rate"],
            "betas": training["betas"],
            "weight_decay": training["weight_decay"],
            "batch_size": batch_size,
            "warmup_ratio": training["warmup_ratio"],
            "scheduler": config["training"]["scheduler"],
            "precision": "fp32",
        },
        "preprocessing": processor.image_processor.to_dict(),
        "training_augmentation": dataset_config["augmentation"],
        "prepare_seconds": prior_prepare_seconds + segment_prepare_seconds,
        "training_seconds": prior_training_seconds + segment_training_seconds,
        "evaluation_seconds": prior_evaluation_seconds + segment_evaluation_seconds,
        "gradient_sampling_seconds": 0.0,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600,
        "gpu": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": max(prior_peak_cuda_bytes, torch.cuda.max_memory_allocated(device)),
        "torch_version": torch.__version__,
        "exact_resume_state_saved": checkpoint_path.is_file(),
    }
    _write_json(run_dir / "predictions.json", {"validation": best_validation, "test": test_result})
    _write_json(result_path, result)
    print(
        f"COMPLETED best_epoch={best_epoch} "
        f"test_accuracy={test_result['accuracy']:.6f}% "
        f"adapter_parameters={count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
