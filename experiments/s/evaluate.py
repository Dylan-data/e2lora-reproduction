#!/usr/bin/env python3
"""Evaluate one saved math run on GSM8K test and write result.json.

This is the final benchmark entrypoint. It is deliberately separate from the
held-out MetaMathQA validation used during model selection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e2lora.adapters.generation import (
    apply_e2lora_plan,
    apply_lora,
    discover_attention_targets,
    load_adapter_state,
    adapter_parameter_counts,
)
from e2lora.dataset.text_generation import file_checksums
from e2lora.model.pretrained import load_pretrained
from e2lora.tasks.generation_metrics import score_gsm8k_generations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="training output directory")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="diagnostic subset only; not a final Table 2 score")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    output_dir = args.output_dir or run_dir / "gsm8k-test"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    plan_data = json.loads((run_dir / "adapter_plan.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / f"checkpoint_{args.checkpoint}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    max_seq_length = config["training"].get("max_seq_length")
    max_new_tokens = args.max_new_tokens or config["evaluation"].get("max_new_tokens")
    if not max_seq_length or not max_new_tokens:
        raise ValueError("Run config must contain the measured max_seq_length and verified max_new_tokens")
    training_result_path = run_dir / "result.json"
    training_result = json.loads(training_result_path.read_text(encoding="utf-8")) if training_result_path.is_file() else None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_id = config.get("run_id")
    if not run_id or checkpoint.get("config", {}).get("run_id") != run_id:
        raise ValueError("Checkpoint does not belong to this run_config.json (run_id mismatch)")
    if plan_data.get("run_id") != run_id:
        raise ValueError("adapter_plan.json does not belong to this run_config.json (run_id mismatch)")
    expected_train_rows = int(config.get("dataset", {}).get("train_count", 0))
    actual_train_rows = int(config.get("actual_train_rows", 0))
    completed_full_training = bool(
        config.get("run_status") != "SMOKE TEST"
        and expected_train_rows > 0
        and actual_train_rows == expected_train_rows
        and training_result is not None
        and training_result.get("run_id") == run_id
        and int(training_result.get("planned_steps", -1)) > 0
        and int(training_result.get("completed_steps", -1)) >= int(training_result.get("planned_steps", -1))
    )
    if args.limit is None and not completed_full_training:
        raise ValueError(
            "This run is not a verified completed full-data training run. "
            "Use --limit N for a diagnostic subset; do not report it as the Table 2 test score."
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")
    if not torch.cuda.is_available():
        raise RuntimeError("Final Llama GSM8K evaluation requires CUDA; this runtime is CPU-only")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("NLG setup requires BF16-capable CUDA hardware")

    started = time.perf_counter()
    model, tokenizer = load_pretrained(
        config["model"]["repo"], kind="generation", revision=config["model"]["revision"],
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    device = torch.device("cuda:0")
    model.to(device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = True
    model.requires_grad_(False)

    if config["method"] == "lora":
        targets = discover_attention_targets(model, config["adapter"]["target_projections"])
        apply_lora(model, targets, rank=int(config["adapter"]["rank"]), alpha=float(config["adapter"]["alpha"]))
    elif config["method"] == "e2lora":
        apply_e2lora_plan(model, plan_data["plan"], alpha=float(config["adapter"]["alpha"]))
    else:
        raise ValueError(f"Unsupported method in run config: {config['method']}")

    load_adapter_state(model, checkpoint["adapter_state"])
    counts = adapter_parameter_counts(model)

    eval_cfg = config["evaluation"]
    # Follow the repository dataset list: official grade-school-math raw test.jsonl.
    # The source URL currently tracks master, so hash the downloaded cache file and
    # pin a Git commit before treating the result as a final reproducibility artifact.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install the project [hf] extra (datasets)") from exc
    dataset = load_dataset(
        "json", data_files={eval_cfg["split"]: eval_cfg["source_url"]},
        split=eval_cfg["split"],
    )
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    if count <= 0:
        raise ValueError("GSM8K test set is empty")
    records = []
    model.eval()
    for index in range(count):
        row = dataset[index]
        prompt = eval_cfg["prompt_template"].format(question=row["question"])
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=False)
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        if ids.shape[1] > int(max_seq_length):
            ids = ids[:, -int(max_seq_length):]
            mask = mask[:, -int(max_seq_length):]
        ids, mask = ids.to(device), mask.to(device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(eval_cfg.get("do_sample", False)),
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        continuation = generated[0, ids.shape[1]:]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        records.append({"question": row["question"], "answer": row["answer"], "generation": text})

    scored = score_gsm8k_generations(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for item in scored["examples"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    result = {
        "status": "DIAGNOSTIC SUBSET ONLY" if args.limit is not None else "FINAL GSM8K TEST EVALUATION",
        "task": "math",
        "run_id": run_id,
        "method": config["method"],
        "seed": config.get("run_seed"),
        "model_repo": config["model"]["repo"],
        "model_revision": config["model"]["revision"],
        "dataset": eval_cfg["dataset"],
        "dataset_source_url": eval_cfg["source_url"],
        "dataset_revision_status": eval_cfg.get("revision_status"),
        "dataset_split": eval_cfg["split"],
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_cache_file_checksums": file_checksums(dataset),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("global_step"),
        "max_seq_length": max_seq_length,
        "max_new_tokens": max_new_tokens,
        "decode": {"do_sample": eval_cfg.get("do_sample", False), "num_return_sequences": 1},
        "parameter_counts": counts,
        "metric": scored["metric"],
        "correct": scored["correct"],
        "total": scored["total"],
        "accuracy": scored["accuracy"],
        "parse_failures": scored["parse_failures"],
        "wall_time_seconds": time.perf_counter() - started,
        "generations_file": str(output_dir / "generations.jsonl"),
        "note": "A limited subset is diagnostic only and must not be reported as the Table 2 test score." if args.limit is not None else "Final GSM8K test score.",
    }
    temporary = output_dir / "result.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(output_dir / "result.json")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
