#!/usr/bin/env python3
"""Unified NLG/commonsense/large-model fine-tuning entry point.

Supports the Table 2 method matrix, Appendix K task models and Appendix L model
matrix. Paper-unspecified hyperparameters are mandatory CLI inputs or explicitly
identified team defaults in the run metadata. This is single-process training;
see experiments/s/README_full_suites.md for the current multi-GPU limitation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e2lora.adapters.adaptive import build_plan, collect_gradients
from e2lora.adapters.generation import capture_adapter_state, load_adapter_state
from e2lora.adapters.generation_methods import (
    METHODS, actual_parameter_report, build_optimizer,
    discover_attention_targets_general, install_method,
)
from e2lora.dataset.commonsense_reasoning import (
    DATASETS as COMMONSENSE_DATASETS, load_commonsense_split,
    normalize_commonsense_row,
)
from e2lora.dataset.nlg_suites import (
    load_metamath_records, load_codefeedback_records, load_humaneval_test,
    normalize_humaneval_row,
)
from e2lora.dataset.text_generation import canonical_row_hash
from e2lora.tasks.commonsense_metrics import score_commonsense
from e2lora.tasks.code_metrics import write_humaneval_samples, run_official_humaneval
from e2lora.tasks.generation_metrics import score_gsm8k_generations

MODELS = {
    "llama31_8b": {"repo": "meta-llama/Llama-3.1-8B", "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"},
    "llama2_13b": {"repo": "meta-llama/Llama-2-13b-hf", "revision": "5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"},
    "qwen3_8b": {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
    "gpt_oss_20b": {"repo": "openai/gpt-oss-20b", "revision": "6cee5e81ee83917806bbde320786a8fb61efebee"},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("table2", "table14", "table15"), required=True)
    p.add_argument("--task", choices=("math", "code", *COMMONSENSE_DATASETS.keys()), required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model-key", choices=tuple(MODELS), default="llama31_8b")
    p.add_argument("--model", default=None)
    p.add_argument("--model-revision", default=None)
    p.add_argument("--seed", type=int, default=42, help="model initialization, shuffle and gradient-sample seed")
    p.add_argument("--data-seed", type=int, default=42, help="fixed dataset sampling seed shared across methods and independent runs")
    p.add_argument("--validation-seed", type=int, default=43, help="fixed held-out split seed shared across methods and independent runs")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, required=True, help="Paper does not specify NLG epochs; record the selected value")
    p.add_argument("--max-seq-length", type=int, required=True, help="Choose after tokenizer length/truncation audit")
    p.add_argument("--max-new-tokens", type=int, required=True, help="Task-specific decode cap; validate no answer truncation")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=64)
    p.add_argument("--gradient-examples", type=int, default=None)
    p.add_argument("--rank", type=int, default=8, help="Baseline LoRA/DoRA/etc rank")
    p.add_argument("--e2-base-rank", type=int, default=8,
                   help="E2 HRA base rank; default follows Appendix J.1 rank-matched evidence (LoRA r4 vs E2 base r8), but must be checked against each table's measured parameter budget")
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lora-plus-ratio", type=float, default=16.0)
    p.add_argument("--vera-rank", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-decay", type=float, default=0.1, help="Logged only; interpretation unresolved in Table 9")
    p.add_argument("--train-jsonl", type=Path, default=None, help="Required local Code-Feedback file containing >=100K code-only rows")
    p.add_argument("--code-only-source-confirmed", action="store_true",
                   help="Confirm that --train-jsonl has already been filtered to code-only samples")
    p.add_argument("--allow-unpinned-datasets", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--validation-size", type=int, default=None, help="Only for math/code; default 2000 team held-out rows")
    p.add_argument("--selection-validation-ratio", type=float, default=0.1,
                   help="For Table 14, hold out this share of train split for checkpoint selection; public validation remains final evaluation")
    p.add_argument("--limit-train", type=int, default=None, help="Diagnostic-only subset; never report as paper reproduction")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--do-final-eval", action="store_true", help="Run benchmark-specific final evaluation after training")
    p.add_argument("--eval-limit", type=int, default=None, help="Diagnostic-only evaluation subset")
    p.add_argument("--human-eval-timeout", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device-map", choices=("auto",), default=None,
                   help="Use Accelerate dispatch for very large models; do not combine with DDP")
    p.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default=None)
    p.add_argument("--run-human-eval-harness", action="store_true",
                   help="Execute generated code with official HumanEval; only inside a hardened sandbox")
    p.add_argument("--confirm-isolated-sandbox", action="store_true",
                   help="Acknowledge generated code is untrusted and this process is isolated from host/network/data")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_hf():
    try:
        from e2lora.model.pretrained import load_pretrained
        return load_pretrained
    except ImportError as exc:
        raise RuntimeError("Install the project's HF extra: pip install -e '.[hf]'") from exc


def load_training_data(args):
    if args.task == "math":
        validation_size = args.validation_size or 2000
        train, valid, source = load_metamath_records(
            train_count=100000 if args.limit_train is None else args.limit_train,
            validation_count=validation_size, seed=args.data_seed,
            validation_seed=args.validation_seed, cache_dir=args.cache_dir,
        )
    elif args.task == "code":
        if args.train_jsonl is None:
            raise ValueError("Table 2 code run needs --train-jsonl pointing to the verified 100K code-only Code-Feedback JSONL")
        validation_size = args.validation_size or 2000
        train, valid, source = load_codefeedback_records(
            args.train_jsonl, train_count=100000 if args.limit_train is None else args.limit_train,
            validation_count=validation_size, seed=args.data_seed, validation_seed=args.validation_seed,
            code_only_confirmed=args.code_only_source_confirmed,
        )
    else:
        if not 0.0 < args.selection_validation_ratio < 1.0:
            raise ValueError("--selection-validation-ratio must be between 0 and 1")
        raw_train = load_commonsense_split(args.task, "train", cache_dir=args.cache_dir,
                                           allow_unpinned=args.allow_unpinned_datasets)
        normalized_train = [normalize_commonsense_row(args.task, dict(row)) for row in raw_train]
        # The public validation split is reserved for the final benchmark score.
        # Epoch/checkpoint selection uses a deterministic hold-out from training only.
        n_select = max(1, round(len(normalized_train) * args.selection_validation_ratio))
        selection_ids = set(random.Random(args.validation_seed).sample(range(len(normalized_train)), n_select))
        train, valid = [], []
        for index, row in enumerate(normalized_train):
            row = dict(row)
            row["sample_id"] = f"train:{index}"
            (valid if index in selection_ids else train).append(row)
        source = {**COMMONSENSE_DATASETS[args.task], "train_count": len(train),
                  "selection_validation_count": len(valid), "selection_validation_ratio": args.selection_validation_ratio,
                  "selection_validation_seed": args.validation_seed, "selection_validation_source_split": "train",
                  "final_evaluation_split": "validation",
                  "evaluation_split_status": "team protocol; public validation held out for final evaluation; paper does not specify split"}
    if args.limit_train is not None:
        train = train[:args.limit_train]
        source["diagnostic_subset"] = True
        source["limit_train"] = args.limit_train
    else:
        source["diagnostic_subset"] = False
    return train, valid, source


class TextRows(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]


def make_collator(tokenizer, max_length: int):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    def collate(rows):
        encodings = []
        prompt_lens, answer_lens, left_truncated = [], [], []
        for row in rows:
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
            answer_ids = tokenizer(row["target"], add_special_tokens=False, truncation=False)["input_ids"]
            if tokenizer.eos_token_id is not None and row.get("target"):
                answer_ids = list(answer_ids) + [int(tokenizer.eos_token_id)]
            if len(answer_ids) >= max_length:
                raise ValueError(f"Answer length {len(answer_ids)} leaves no prompt capacity at max_seq_length={max_length}")
            trim = max(0, len(prompt_ids) + len(answer_ids) - max_length)
            prompt_ids = prompt_ids[trim:]
            input_ids = list(prompt_ids) + list(answer_ids)
            labels = [-100] * len(prompt_ids) + list(answer_ids)
            encodings.append((input_ids, labels))
            prompt_lens.append(len(prompt_ids)); answer_lens.append(len(answer_ids)); left_truncated.append(trim)
        width = max(len(x[0]) for x in encodings)
        ids, masks, labels = [], [], []
        for input_ids, label_ids in encodings:
            pad = width - len(input_ids)
            ids.append(input_ids + [tokenizer.pad_token_id] * pad)
            masks.append([1] * len(input_ids) + [0] * pad)
            labels.append(label_ids + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_lengths": prompt_lens,
            "answer_lengths": answer_lens,
            "prompt_left_truncated": left_truncated,
        }
    return collate


def make_e2_plan(model, tokenizer, train_rows, targets, args):
    count = args.gradient_examples or min(64, len(train_rows))
    if count < 1 or count > len(train_rows):
        raise ValueError(f"gradient_examples must be in [1,{len(train_rows)}]")
    ids = random.Random(args.seed + 1701).sample(range(len(train_rows)), count)
    batches = []
    collator = make_collator(tokenizer, args.max_seq_length)
    for start in range(0, len(ids), args.per_device_batch_size):
        rows = [train_rows[i] for i in ids[start:start + args.per_device_batch_size]]
        batch = collator(rows)
        batches.append({k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)})
    flat_targets = [name for names in targets.values() for name in names]
    def loss_for_batch(m, batch):
        output = m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        tokens = int((batch["labels"] != -100).sum().item())
        return output.loss, tokens
    grads = collect_gradients(model, flat_targets, batches, loss_for_batch)
    plan, diagnostics = build_plan(grads, targets, base_rank=args.e2_base_rank, rank_policy="paper")
    for group in plan:
        group["trainable_rank"] = group["rank"]
    serializable = {"plan": plan, "diagnostics": diagnostics, "sample_indices": ids,
                    "sample_ids": [train_rows[i].get("sample_id", str(i)) for i in ids],
                    "gradient_example_count": count, "rank_policy": "paper"}
    return plan, serializable


def _batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def _mean_loss(model, loader, device):
    model.eval(); total_loss = 0.0; total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            values = _batch_to_device(batch, device)
            out = model(input_ids=values["input_ids"], attention_mask=values["attention_mask"], labels=values["labels"])
            tokens = int((values["labels"] != -100).sum().item())
            total_loss += float(out.loss.detach()) * max(tokens, 1)
            total_tokens += tokens
    return total_loss / max(total_tokens, 1)


def _save_trainable_checkpoint(model, method_metadata, optimizer, scheduler, output_dir, epoch, global_step,
                               run_config, best_metric, *, filename="checkpoint_last.pt"):
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "_e2lora_adapter_bank"):
        model_state = capture_adapter_state(model)
        save_kind = "custom_adapter_bank"
    else:
        model_state = {name: parameter.detach().cpu().clone()
                       for name, parameter in model.named_parameters() if parameter.requires_grad}
        save_kind = "trainable_parameters"
    checkpoint = {"save_kind": save_kind, "model_state": model_state,
                  "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                  "epoch": epoch, "global_step": global_step, "best_metric": best_metric,
                  "run_config": run_config}
    tmp = output_dir / (filename + ".tmp")
    torch.save(checkpoint, tmp); tmp.replace(output_dir / filename)


def _restore_checkpoint_model(model, checkpoint):
    if checkpoint["save_kind"] == "custom_adapter_bank":
        load_adapter_state(model, checkpoint["model_state"])
        return
    current = dict(model.named_parameters())
    missing = sorted(set(checkpoint["model_state"]) - set(current))
    if missing:
        raise ValueError(f"Resume checkpoint has missing model parameters: {missing[:5]}")
    with torch.no_grad():
        for name, value in checkpoint["model_state"].items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))


def generate_records(model, tokenizer, rows, args, *, task: str, limit: int | None = None):
    rows = rows[:limit] if limit is not None else rows
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = []
    model.eval()
    batch_size = max(1, args.per_device_batch_size)
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        prompts = [r["prompt"] for r in group]
        tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=args.max_seq_length).to(args.device)
        with torch.no_grad():
            output = model.generate(**tokens, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, num_return_sequences=1,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
        new_tokens = output[:, tokens["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, generation in zip(group, decoded):
            record = {"generation": generation, "prompt": row["prompt"], **row}
            records.append(record)
    return records


def load_final_eval(task: str, args):
    if task == "math":
        import urllib.request
        cache_root = Path(args.cache_dir or ".cache/e2lora") / "gsm8k"
        cache_root.mkdir(parents=True, exist_ok=True)
        source = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
        path = cache_root / "gsm8k-test-3101c7d.jsonl"
        if not path.exists():
            request = urllib.request.Request(source, headers={"User-Agent": "e2lora-reproduction/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
                handle.write(response.read())
        import json as _json
        import hashlib as _hashlib
        rows = [_json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        normalized = [{"question": r["question"], "answer": r["answer"],
                       "prompt": f"Question: {r['question']}\nAnswer:", "target": r["answer"]}
                      for r in rows]
        return normalized, {"repo": "openai/grade-school-math", "commit": "3101c7d5072418e28b9008a6636bde82a006892c",
                            "split": "test", "source_url": source, "file_sha256": _hashlib.sha256(path.read_bytes()).hexdigest(),
                            "row_count": len(normalized)}
    if task == "code":
        rows = load_humaneval_test(cache_dir=args.cache_dir, allow_unpinned=args.allow_unpinned_datasets)
        normalized = [normalize_humaneval_row(dict(row)) for row in rows]
        return normalized, {"repo": "openai/openai_humaneval", "revision": None,
                            "split": "test", "row_count": len(normalized),
                            "rows_sha256": hashlib.sha256("".join(canonical_row_hash(x) for x in normalized).encode()).hexdigest(),
                            "revision_status": "unpinned; record dataset cache/source before final comparison"}
    rows = load_commonsense_split(task, "validation", cache_dir=args.cache_dir,
                                  allow_unpinned=args.allow_unpinned_datasets)
    normalized = [normalize_commonsense_row(task, dict(row)) for row in rows]
    cfg = COMMONSENSE_DATASETS[task]
    return normalized, {"repo": cfg["repo"], "subset": cfg["subset"], "revision": cfg["revision"],
                        "split": "validation", "row_count": len(normalized)}


def final_eval(model, tokenizer, task, args, output_dir):
    rows, data_manifest = load_final_eval(task, args)
    records = generate_records(model, tokenizer, rows, args, task=task, limit=args.eval_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest["evaluated_rows"] = len(records)
    data_manifest["diagnostic_subset"] = args.eval_limit is not None
    data_manifest["samples"] = [
        {"sample_id": row.get("sample_id") or row.get("task_id") or row.get("source_index") or f"eval:{i}",
         "row_sha256": canonical_row_hash(row)}
        for i, row in enumerate(rows[:len(records)])
    ]
    (output_dir / "dataset_manifest.json").write_text(json.dumps(data_manifest, indent=2, ensure_ascii=False))
    if task == "math":
        scored = score_gsm8k_generations(records)
        metric = {k: v for k, v in scored.items() if k != "examples"}
        metric["evaluation_dataset"] = "GSM8K test"
        metric["protocol_note"] = "Exact numeric extraction; prompt is a documented team choice."
        details = scored["examples"]
    elif task == "code":
        samples = write_humaneval_samples(records, output_dir / "samples.jsonl")
        metric = {**samples, "diagnostic_subset": args.eval_limit is not None}
        details = records
        (output_dir / "human_eval_samples_manifest.json").write_text(json.dumps(samples, indent=2))
        if args.run_human_eval_harness:
            metric["official_harness"] = run_official_humaneval(
                output_dir / "samples.jsonl", timeout=args.human_eval_timeout)
        else:
            metric["pass_at_1_status"] = "not computed; score samples.jsonl with the official harness in an isolated sandbox"
    else:
        scored = score_commonsense(records)
        metric = {k: v for k, v in scored.items() if k != "examples"}
        metric["evaluation_dataset"] = f"{task}/validation"
        metric["split_protocol"] = "public validation held out for final evaluation; checkpoint selection used a disjoint deterministic hold-out from the train split (team protocol; paper does not state exact split)"
        details = scored["examples"]
    (output_dir / "generations.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n")
    (output_dir / "result.json").write_text(json.dumps(metric, ensure_ascii=False, indent=2))
    return metric


def main():
    args = parse_args()
    if args.run_human_eval_harness and not args.confirm_isolated_sandbox:
        raise ValueError("HumanEval executes untrusted generated code; add --confirm-isolated-sandbox only inside a hardened, network-isolated sandbox")
    if args.epochs < 1 or args.max_seq_length < 2 or args.max_new_tokens < 1:
        raise ValueError("epochs, max-seq-length and max-new-tokens must be positive")
    if args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.task in COMMONSENSE_DATASETS and args.suite != "table14":
        raise ValueError("Commonsense tasks are only part of --suite table14")
    if args.suite == "table14" and args.task not in COMMONSENSE_DATASETS:
        raise ValueError("Appendix K / Table 14 requires one of the six commonsense tasks")
    if args.suite == "table15" and args.task != "math":
        raise ValueError("Appendix L / Table 15 is math-benchmark fine-tuning")
    if args.suite == "table2" and args.task not in ("math", "code"):
        raise ValueError("Table 2 supports math or code")
    if args.model is not None and args.model_revision is None:
        raise ValueError("Overriding --model requires an explicit --model-revision")
    model_info = MODELS[args.model_key]
    model_repo = args.model or model_info["repo"]
    model_revision = args.model_revision or model_info["revision"]
    if args.suite == "table2" and args.model_key != "llama31_8b":
        raise ValueError("Table 2 model is Llama 3.1 8B Base")
    if args.suite == "table14" and args.model_key != "llama31_8b":
        raise ValueError("Table 14 model is Llama 3.1 8B")
    if args.suite == "table15" and args.model_key == "llama31_8b":
        raise ValueError("Table 15 model-key must be llama2_13b, qwen3_8b or gpt_oss_20b")
    seed_everything(args.seed)
    train_rows, valid_rows, source = load_training_data(args)
    if not train_rows or not valid_rows:
        raise ValueError("Training and validation data must both be non-empty")
    source_fingerprint = hashlib.sha256("".join(canonical_row_hash(row) for row in train_rows).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest = {
        "train": [{"sample_id": row.get("sample_id") or row.get("source_index") or str(i), "row_sha256": canonical_row_hash(row)}
                  for i, row in enumerate(train_rows)],
        "validation": [{"sample_id": row.get("sample_id") or row.get("source_index") or str(i), "row_sha256": canonical_row_hash(row)}
                       for i, row in enumerate(valid_rows)],
        "train_count": len(train_rows), "validation_count": len(valid_rows),
        "train_manifest_sha256": source_fingerprint, "source": source,
    }
    (args.output_dir / "sample_manifest.json").write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    run_config = {
        "suite": args.suite, "task": args.task, "method": args.method,
        "model": {"repo": model_repo, "revision": model_revision},
        "dataset": source, "train_manifest_sha256": source_fingerprint,
        "seed": args.seed, "data_seed": args.data_seed, "validation_seed": args.validation_seed, "epochs": args.epochs,
        "sequence_length": args.max_seq_length, "max_new_tokens": args.max_new_tokens,
        "batching": {"per_device_batch": args.per_device_batch_size,
                     "gradient_accumulation": args.gradient_accumulation_steps,
                     "effective_batch": args.per_device_batch_size * args.gradient_accumulation_steps},
        "hyperparameters": {"learning_rate_cli": args.learning_rate, "weight_decay": args.weight_decay,
                            "warmup_ratio": args.warmup_ratio, "table9_lr_decay_unresolved": args.lr_decay,
                            "precision": "bf16 if CUDA else fp32"},
        "team_choices": {"data_sampling_seed": args.data_seed, "selection_validation_seed": args.validation_seed,
                         "baseline_rank": args.rank, "e2_base_rank": args.e2_base_rank,
                         "e2_rank_budget_status": "default base rank 8 guided by Appendix J.1 (LoRA r4 vs E2 base r8); target-suite rank is not separately stated and actual counts must be checked",
                         "alpha": args.alpha, "lora_plus_ratio": args.lora_plus_ratio,
                         "vera_rank": args.vera_rank, "selection_validation": "task-specific; see dataset metadata",
                         "attention_implementation": args.attn_implementation},
        "paper_settings": {"optimizer": "AdamW", "betas": [0.9, 0.999], "global_batch_size": 64,
                           "weight_decay": 0.0005, "C4_warmup": 0.03,
                           "Table9_warmup": 0.3, "Table9_LR_decay": 0.1},
        "diagnostic_subset": args.limit_train is not None,
    }
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing run directory: {args.output_dir}")
    if args.resume and not (args.output_dir / "checkpoint_last.pt").is_file():
        raise FileNotFoundError("--resume was set but checkpoint_last.pt does not exist")
    if args.resume and config_path.is_file():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("suite", "task", "method", "model", "dataset", "train_manifest_sha256", "seed", "data_seed", "validation_seed", "epochs", "sequence_length", "batching"):
            if old_config.get(key) != run_config.get(key):
                raise ValueError(f"Resume configuration mismatch for {key}: {old_config.get(key)!r} != {run_config.get(key)!r}")
    else:
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch runtime has no CUDA support")
    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    load_pretrained = _load_hf()
    model_options = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.device_map == "auto":
        model_options["device_map"] = "auto"
    if args.attn_implementation:
        model_options["attn_implementation"] = args.attn_implementation
    model, tokenizer = load_pretrained(model_repo, kind="generation", revision=model_revision,
                                       cache_dir=str(args.cache_dir) if args.cache_dir else None,
                                       **model_options)
    if args.device_map == "auto":
        args.device = str(model.get_input_embeddings().weight.device)
    else:
        model.to(args.device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    targets = discover_attention_targets_general(model)
    total_steps = math.ceil(len(train_rows) / (args.per_device_batch_size * args.gradient_accumulation_steps)) * args.epochs
    plan, plan_metadata = (None, None)
    if args.method in ("e2lora", "e2sharelora"):
        saved_plan_path = args.output_dir / "adapter_plan.json"
        if args.resume and saved_plan_path.is_file():
            plan_metadata = json.loads(saved_plan_path.read_text(encoding="utf-8"))
            plan = plan_metadata["plan"]
        else:
            plan, plan_metadata = make_e2_plan(model, tokenizer, train_rows, targets, args)
            saved_plan_path.write_text(json.dumps(plan_metadata, ensure_ascii=False, indent=2))
    installed = install_method(model, args.method, targets, rank=args.rank, alpha=args.alpha,
                               plan=plan, total_steps=total_steps,
                               lora_plus_ratio=args.lora_plus_ratio, vera_rank=args.vera_rank, seed=args.seed)
    model = installed["model"]
    method_metadata = installed["metadata"]
    lr_defaults = {"fft_attn_only": 5e-5, "dora_r4": 5e-5, "dora_r8": 5e-5,
                   "adalora_r4": 5e-4, "adalora_r8": 5e-4, "lora_plus": 5e-5,
                   "lora_fa": 5e-4, "vera": 5e-3, "lora": 5e-5,
                   "e2lora": 5e-5, "sharelora": 5e-5, "e2sharelora": 5e-5}
    lr = args.learning_rate if args.learning_rate is not None else lr_defaults[args.method]
    run_config["hyperparameters"]["learning_rate_used"] = lr
    run_config["method_implementation"] = method_metadata.copy()
    run_config["method_implementation"].pop("optimizer_groups", None)
    run_config["parameter_counts"] = actual_parameter_report(model)
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2))
    (args.output_dir / "parameter_counts.json").write_text(json.dumps(run_config["parameter_counts"], indent=2))

    optimizer = build_optimizer(model, method_metadata, lr=lr, weight_decay=args.weight_decay,
                                betas=(0.9, 0.999), lora_plus_ratio=args.lora_plus_ratio)
    train_loader = DataLoader(TextRows(train_rows), batch_size=args.per_device_batch_size,
                              shuffle=True, collate_fn=make_collator(tokenizer, args.max_seq_length),
                              num_workers=args.num_workers)
    valid_loader = DataLoader(TextRows(valid_rows), batch_size=args.per_device_batch_size,
                              shuffle=False, collate_fn=make_collator(tokenizer, args.max_seq_length),
                              num_workers=args.num_workers)
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history = []
    run_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    global_step = 0
    best_val = float("inf")
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.output_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        _restore_checkpoint_model(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_metric", float("inf")))
        history_path = args.output_dir / "training_history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        group_batches = []
        losses = []
        for batch in train_loader:
            group_batches.append(batch)
            if len(group_batches) < args.gradient_accumulation_steps:
                continue
            global_step += 1
            _optimizer_group_step(model, optimizer, scheduler, group_batches, args.device, losses, global_step)
            group_batches = []
        if group_batches:
            global_step += 1
            _optimizer_group_step(model, optimizer, scheduler, group_batches, args.device, losses, global_step)
        val_loss = _mean_loss(model, valid_loader, args.device)
        row = {"epoch": epoch, "global_step": global_step,
               "train_loss": sum(losses) / max(1, len(losses)), "validation_loss": val_loss,
               "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            (args.output_dir / "best_epoch.json").write_text(json.dumps(row, indent=2))
        _save_trainable_checkpoint(model, method_metadata, optimizer, scheduler, args.output_dir,
                                   epoch, global_step, run_config, best_val, filename="checkpoint_last.pt")
        if is_best:
            _save_trainable_checkpoint(model, method_metadata, optimizer, scheduler, args.output_dir,
                                       epoch, global_step, run_config, best_val, filename="checkpoint_best.pt")
        print(json.dumps(row), flush=True)
        model.train()
    hardware = {"torch_version": torch.__version__, "device": args.device,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    if torch.cuda.is_available():
        hardware["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        hardware["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    result = {"suite": args.suite, "task": args.task, "method": args.method,
              "model": model_repo, "revision": model_revision, "seed": args.seed,
              "epochs_completed": args.epochs, "global_steps": global_step,
              "best_validation_loss": best_val, "training_history": history,
              "parameter_counts": run_config["parameter_counts"],
              "hardware": hardware,
              "wall_clock_seconds": time.perf_counter() - run_start_time,
              "diagnostic_subset": args.limit_train is not None,
              "note": "This result is a local reproduction run; compare with paper Table only after full data and protocol match."}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    if args.do_final_eval:
        best_ckpt = args.output_dir / "checkpoint_best.pt"
        if best_ckpt.is_file():
            saved = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            _restore_checkpoint_model(model, saved)
        metric = final_eval(model, tokenizer, args.task, args, args.output_dir / "final-evaluation")
        result["final_evaluation"] = metric
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))


def _optimizer_group_step(model, optimizer, scheduler, batches, device, losses, global_step):
    optimizer.zero_grad(set_to_none=True)
    sizes = [max(1, int((batch["labels"] != -100).sum().item())) for batch in batches]
    total_size = sum(sizes)
    model.train()
    weighted_loss = 0.0
    for batch, size in zip(batches, sizes):
        values = _batch_to_device(batch, device)
        out = model(input_ids=values["input_ids"], attention_mask=values["attention_mask"], labels=values["labels"])
        loss = out.loss
        (loss * (size / total_size)).backward()
        weighted_loss += float(loss.detach()) * size / total_size
    optimizer.step()
    # PEFT AdaLoRA requires an explicit rank-allocation update after each step.
    tuner = getattr(model, "base_model", None)
    if tuner is not None and hasattr(tuner, "update_and_allocate"):
        tuner.update_and_allocate(global_step)
    scheduler.step()
    losses.append(weighted_loss)


if __name__ == "__main__":
    main()
