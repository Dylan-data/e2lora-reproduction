#!/usr/bin/env python
"""Score one pinned zero-shot image-classification unit."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

from e2lora.model.clip_classification import CONVERTED, MODEL, ORIGINAL, logits, text_features
from train_lora import (
    _audit_split,
    _cached_snapshot,
    _config_hash,
    _expected_split_count,
    _make_batches,
    _record_weight_provenance,
    _seed,
    _write_or_verify_json,
)


def _write_result(path, result):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: {path}")
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    result_path = run_dir / "result.json"
    saved_config = run_dir / "config.json"
    if not run_dir.is_dir() or not saved_config.is_file():
        raise FileNotFoundError("Run directory and immutable config must exist before scoring.")
    if _config_hash(json.loads(saved_config.read_text(encoding="utf-8"))) != _config_hash(config):
        raise ValueError("Submitted config differs from the immutable run-directory config.")
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: {result_path}")

    model_config = config["model"]
    dataset_config = config["dataset"]
    evaluation_config = config["evaluation"]
    if model_config["repo"] != MODEL or model_config["revision"] != ORIGINAL:
        raise ValueError("This runner only accepts its pinned CLIP ViT-B/16 model.")
    if model_config["converted_safetensors_revision"] != CONVERTED:
        raise ValueError("Converted CLIP weight revision differs from the verified pin.")
    if model_config["precision"] != "fp32" or evaluation_config["precision"] != "fp32":
        raise ValueError("This reproduction unit requires FP32 inference.")
    if config["method"]["prompt_template"] != "a photo of a {class}":
        raise ValueError("Unexpected zero-shot prompt template.")
    _seed(evaluation_config["seed"])
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Zero-shot scoring must run on a GPU compute node.")
    device = torch.device("cuda")
    started = time.perf_counter()

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
        raise ValueError(
            f"Expected {dataset_config['expected_class_count']} ordered classes, got {len(classes)}."
        )
    if classes != dataset_config["expected_class_names"]:
        raise ValueError("Pinned dataset ClassLabel names/order differ from the config.")
    train_counts = _audit_split(
        train_dataset,
        dataset_config["train_split"],
        classes,
        _expected_split_count(dataset_config, "train"),
    )
    test_counts = _audit_split(
        test_dataset,
        dataset_config["test_split"],
        classes,
        _expected_split_count(dataset_config, "test"),
    )
    audit = {
        "repo": dataset_config["repo"],
        "revision": dataset_config["revision"],
        "train_split": dataset_config["train_split"],
        "test_split": dataset_config["test_split"],
        "train_count": len(train_dataset),
        "test_count": len(test_dataset),
        "class_count": len(classes),
        "classes_in_label_order": classes,
        "train_label_counts": train_counts,
        "test_label_counts": test_counts,
        "sample_id_scheme": "dataset revision + source split + zero-based row index",
        "paper_split_equivalence": "not established; pinned Hugging Face mirror split",
    }
    _write_or_verify_json(run_dir / "dataset_audit.json", audit)
    _write_or_verify_json(
        run_dir / "splits.json",
        {
            "dataset": dataset_config["repo"],
            "revision": dataset_config["revision"],
            "train_source_split": dataset_config["train_split"],
            "test_source_split": dataset_config["test_split"],
            "training_indices": list(range(len(train_dataset))),
            "test_indices": list(range(len(test_dataset))),
            "sample_id_scheme": audit["sample_id_scheme"],
            "validation": "not used for zero-shot scoring",
        },
    )

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
    _write_or_verify_json(
        run_dir / "classes.json",
        {"classes": classes, "labels": list(range(len(classes))), "prompts": prompts},
    )
    _write_or_verify_json(
        run_dir / "preprocessing.json", processor.image_processor.to_dict()
    )

    indices = list(range(len(test_dataset)))
    labels_out, predictions_out, indices_out = [], [], []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_indices, pixels, labels in _make_batches(
            test_dataset,
            indices,
            processor,
            device,
            evaluation_config["batch_size"],
        ):
            predictions = logits(model, pixels, text).argmax(-1)
            labels_out.extend(labels.cpu().tolist())
            predictions_out.extend(predictions.cpu().tolist())
            indices_out.extend(batch_indices)
            if len(indices_out) % 512 == 0 or len(indices_out) == len(test_dataset):
                print(f"Scored {len(indices_out)}/{len(test_dataset)} test images.", flush=True)

    if indices_out != indices:
        raise AssertionError("Scored test indices are not the complete ordered test split.")
    correct = sum(label == prediction for label, prediction in zip(labels_out, predictions_out))
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "model": {
            "repo": model_config["repo"],
            "revision": model_config["revision"],
            "converted_safetensors_revision": model_config["converted_safetensors_revision"],
            "precision": "fp32",
        },
        "dataset": audit,
        "method": config["method"],
        "seed": evaluation_config["seed"],
        "config_sha256": _config_hash(config),
        "metric": "top-1 accuracy",
        "test": {
            "count": len(labels_out),
            "correct": correct,
            "accuracy_percent": 100.0 * correct / len(labels_out),
            "indices": indices_out,
            "labels": labels_out,
            "predictions": predictions_out,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "evaluation_seconds": time.perf_counter() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _write_result(result_path, result)
    print(
        f"RESULT {correct}/{len(labels_out)} "
        f"({result['test']['accuracy_percent']:.4f}%) -> {result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
