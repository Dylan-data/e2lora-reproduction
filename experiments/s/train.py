#!/usr/bin/env python3
"""Train LoRA/E2LoRA on MetaMathQA for GSM8K (Table 2 mathematics).

The default configuration is deliberately incomplete for training: max sequence
length, epochs/steps and per-device batch geometry must be selected from measured
length/GPU evidence rather than guessed. Use --prepare-only first.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e2lora.adapters.adaptive import collect_gradients, gradient_similarity, greedy_intervals
from e2lora.adapters.generation import (
    adapter_parameter_counts,
    apply_e2lora_plan,
    apply_lora,
    capture_adapter_state,
    discover_attention_targets,
    load_adapter_state,
    trainable_adapter_changed,
    trainable_adapter_snapshot,
)
from e2lora.dataset.loading import load_split
from e2lora.dataset.text_generation import (
    analyze_math_lengths,
    collate_math,
    create_or_load_manifest,
)
from e2lora.tasks.generation_metrics import extract_final_number, extract_gold_number, exact_number_match
from e2lora.model.pretrained import load_pretrained


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF model repo; defaults to pinned config")
    parser.add_argument("--task", choices=["math"], default="math")
    parser.add_argument("--method", choices=["lora", "e2lora"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true", help="create/check manifest and token length report; no model load or training")
    parser.add_argument("--smoke-samples", type=int, default=None, help="use first N rows of the shared manifest; always mark run SMOKE TEST")
    parser.add_argument("--gradient-examples", type=int, default=None, help="E2LoRA gradient sample count; small values are smoke-only")
    parser.add_argument("--max-seq-length", type=int, default=None, help="must be selected after length analysis")
    parser.add_argument("--length-candidates", type=str, default=None,
                        help="comma-separated max sequence lengths to measure answer/prompt truncation risk during --prepare-only")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="must be selected after validation generation checks")
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="bounded smoke run; mutually exclusive with --epochs")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path, method: str) -> dict:
    if path is None:
        path = ROOT / "experiments/s/configs" / f"llama31_math_{method}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["method"] != method:
        raise ValueError(f"Config method {config['method']} does not match --method {method}")
    config["_config_path"] = str(path.resolve())
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_resume_preview(path: Path, config: dict, seed: int) -> dict:
    """Read a resume checkpoint early so E²LoRA can reuse its exact saved plan."""
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config", {})
    checks = [
        ("method", saved.get("method"), config.get("method")),
        ("task", saved.get("task"), config.get("task")),
        ("run_seed", saved.get("run_seed"), seed),
        ("model.repo", saved.get("model", {}).get("repo"), config["model"]["repo"]),
        ("model.revision", saved.get("model", {}).get("revision"), config["model"]["revision"]),
        ("dataset.repo", saved.get("dataset", {}).get("repo"), config["dataset"]["repo"]),
        ("dataset.revision", saved.get("dataset", {}).get("revision"), config["dataset"]["revision"]),
        ("dataset.split", saved.get("dataset", {}).get("split"), config["dataset"]["split"]),
    ]
    for name, old, new in checks:
        if old != new:
            raise ValueError(f"Cannot resume with changed {name}: checkpoint={old!r}, current={new!r}")

    # These fields determine optimizer/scheduler and the data/loss geometry.
    immutable_training = (
        "optimizer", "learning_rate", "betas", "weight_decay", "global_batch_size",
        "per_device_batch_size", "gradient_accumulation_steps", "precision", "scheduler",
        "warmup_ratio", "min_lr_ratio", "max_seq_length", "max_new_tokens", "epochs",
        "max_steps", "drop_last", "save_steps", "eval_steps", "loss_on",
        "prompt_template", "answer_template", "gradient_clip_norm",
    )
    for name in immutable_training:
        if saved.get("training", {}).get(name) != config.get("training", {}).get(name):
            raise ValueError(
                f"Cannot resume with changed training.{name}: "
                f"checkpoint={saved.get('training', {}).get(name)!r}, "
                f"current={config.get('training', {}).get(name)!r}"
            )
    if saved.get("adapter") != config.get("adapter"):
        raise ValueError("Cannot resume with a changed adapter configuration")
    if saved.get("smoke_samples") != config.get("smoke_samples"):
        raise ValueError("Cannot resume with a changed --smoke-samples selection")
    if saved.get("evaluation") != config.get("evaluation"):
        raise ValueError("Cannot resume with changed evaluation prompt/decode settings")
    if not isinstance(checkpoint.get("adapter_state"), dict):
        raise ValueError("Resume checkpoint has no adapter state")
    if not isinstance(checkpoint.get("plan"), list):
        raise ValueError("Resume checkpoint has no adapter plan")
    if not saved.get("run_id"):
        raise ValueError("Resume checkpoint has no run_id; use a checkpoint created by this training entrypoint")
    return checkpoint


def distributed_context():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun distributed training requires CUDA in this experiment")
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def json_write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class IndexedRows(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


def get_manifest_rows(manifest, *, smoke_samples=None):
    train = list(manifest["train"])
    validation = list(manifest["validation"])
    smoke = smoke_samples is not None
    if smoke:
        if smoke_samples < 1:
            raise ValueError("--smoke-samples must be positive")
        train = train[:smoke_samples]
        validation = validation[: max(1, min(len(validation), min(16, smoke_samples // 4 or 1)))]
        if not validation:
            raise ValueError("Manifest has no validation rows; cannot run a smoke test")
    return [item["row_index"] for item in train], [item["row_index"] for item in validation], smoke


def tokenizer_from_config(config):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install the project [hf] extra (transformers, datasets)") from exc
    repo = config["model"]["repo"]
    revision = config["model"]["revision"]
    local = ROOT / "src/e2lora/model/weights" / repo / revision
    tokenizer_files_present = any((local / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"))
    source = str(local) if (local / "config.json").is_file() and tokenizer_files_present else repo
    options = {"revision": revision} if source == repo else {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_dataset_and_manifest(config, *, rank=0, world=1):
    """Load the pinned split; rank 0 alone creates the shared manifest."""
    dataset_cfg = config["dataset"]
    dataset = load_split(
        dataset_cfg["repo"], split=dataset_cfg["split"], revision=dataset_cfg["revision"]
    )
    manifest_path = ROOT / dataset_cfg["shared_manifest"]
    if rank == 0:
        manifest = create_or_load_manifest(
            dataset,
            manifest_path,
            repo=dataset_cfg["repo"],
            revision=dataset_cfg["revision"],
            split=dataset_cfg["split"],
            train_count=int(dataset_cfg["train_count"]),
            validation_count=int(dataset_cfg["validation_count"]),
            seed=int(dataset_cfg["sampling_seed"]),
            validation_seed=int(dataset_cfg["validation_seed"]),
        )
    if world > 1:
        torch.distributed.barrier()
    if rank != 0:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Rank 0 did not create the shared manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dataset, manifest, manifest_path


def calculate_length_report(tokenizer, dataset, train_indices, validation_indices, out_dir, seed,
                            candidate_max_lengths=None):
    train_report = analyze_math_lengths(
        tokenizer, (dataset[i] for i in train_indices), candidate_max_lengths=candidate_max_lengths
    )
    validation_report = analyze_math_lengths(
        tokenizer, (dataset[i] for i in validation_indices), candidate_max_lengths=candidate_max_lengths
    )
    payload = {
        "status": "LENGTH ANALYSIS ONLY; NO TRAINING",
        "seed": seed,
        "train": train_report,
        "validation": validation_report,
        "max_seq_length_selected": None,
        "selection_rule": "Review combined-length percentiles and verify answer truncation on validation before setting max_seq_length.",
    }
    json_write(out_dir / "length_stats.json", payload)
    return payload


def exact_budget(scores: list[float], budget: int, *, rank_policy: str) -> list[int]:
    if not scores or budget < len(scores):
        raise ValueError(f"Cannot allocate budget {budget} over {len(scores)} groups with at least one rank each")
    if rank_policy != "paper":
        raise ValueError(f"Unsupported rank policy {rank_policy}; do not silently transform proxy entropy")
    total = sum(scores)
    if not math.isfinite(total) or abs(total) < 1e-12:
        raise ValueError(f"Proxy-entropy scores have an invalid sum: {total}")
    quotas = [budget * score / total for score in scores]
    if any(not math.isfinite(value) or value < 0 for value in quotas):
        raise ValueError(f"Literal paper allocation produced negative/non-finite rank quotas: {quotas}")
    ranks = [int(math.floor(value)) for value in quotas]
    remaining = budget - sum(ranks)
    order = sorted(range(len(scores)), key=lambda idx: quotas[idx] - ranks[idx], reverse=True)
    for idx in order[:remaining]:
        ranks[idx] += 1
    if sum(ranks) != budget or min(ranks) < 1:
        raise ValueError(f"Invalid exact rank allocation: {ranks}; scores={scores}")
    return ranks


def validate_e2_plan(plan, targets, adapter_cfg):
    """Check Q/K/V/O coverage, model order and exact total/active rank budgets."""
    expected_families = list(targets)
    observed_families = {group.get("family") for group in plan}
    if observed_families != set(expected_families):
        raise ValueError(f"E2LoRA plan projection families differ: {observed_families} != {set(expected_families)}")
    for family in expected_families:
        groups = [group for group in plan if group.get("family") == family]
        covered = [name for group in groups for name in group.get("layers", [])]
        if covered != targets[family]:
            raise ValueError(f"E2LoRA groups for {family} must cover target layers exactly once in model order")
        ids = [group.get("adapter_id") for group in groups]
        if len(set(ids)) != len(ids):
            raise ValueError(f"E2LoRA adapter IDs are not unique for {family}")
        for group in groups:
            rank = int(group["rank"])
            active = int(group["trainable_rank"])
            if not 1 <= active <= rank:
                raise ValueError(f"Invalid group rank for {family}: rank={rank}, trainable_rank={active}")
        total_rank = sum(int(group["rank"]) for group in groups)
        total_active = sum(int(group["trainable_rank"]) for group in groups)
        if total_rank != int(adapter_cfg["total_rank_per_projection"]):
            raise ValueError(f"{family} total rank budget {total_rank} != {adapter_cfg['total_rank_per_projection']}")
        if total_active != int(adapter_cfg["trainable_rank_per_projection"]):
            raise ValueError(f"{family} trainable rank budget {total_active} != {adapter_cfg['trainable_rank_per_projection']}")


def build_e2_plan(model, tokenizer, gradient_rows, config, device):
    adapter_cfg = config["adapter"]
    grad_cfg = config["gradient_diagnostics"]
    projections = adapter_cfg["target_projections"]
    targets = discover_attention_targets(model, projections)
    names = [name for family in projections for name in targets[family]]
    grad_batch_size = 1
    batches = []
    for start in range(0, len(gradient_rows), grad_batch_size):
        rows = gradient_rows[start:start + grad_batch_size]
        batch = collate_math(tokenizer, rows, max_seq_length=int(config["training"]["max_seq_length"]))
        batches.append({
            key: value.to(device) for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        })

    plan, diagnostics = [], {}
    for family in projections:
        family_names = targets[family]
        gradients = collect_gradients(
            model,
            family_names,
            batches,
            lambda current, batch: (
                current(**batch).loss,
                int((batch["labels"] != -100).sum().item()),
            ),
        )
        entropy, rmi = gradient_similarity([gradients[name] for name in family_names])
        del gradients
        intervals, thresholds = greedy_intervals(rmi)
        group_scores = [max(entropy[first:last]) for first, last in intervals]
        family_diagnostics = {
            "layers": family_names,
            "proxy_entropy": entropy,
            "rmi": rmi,
            "thresholds": thresholds,
            "intervals_half_open": intervals,
            "group_proxy_entropy": group_scores,
        }
        diagnostics[family] = family_diagnostics
        try:
            group_ranks = exact_budget(
                group_scores,
                int(adapter_cfg["total_rank_per_projection"]),
                rank_policy=grad_cfg["rank_policy"],
            )
            active_ranks = exact_budget(
                [float(value) for value in group_ranks],
                int(adapter_cfg["trainable_rank_per_projection"]),
                rank_policy="paper",
            )
        except Exception as exc:
            family_diagnostics["rank_allocation_error"] = f"{type(exc).__name__}: {exc}"
            failure = RuntimeError(
                f"Rank allocation failed for {family}; group_proxy_entropy={group_scores}; "
                f"sample_count={len(gradient_rows)}; original_error={type(exc).__name__}: {exc}"
            )
            failure.e2_diagnostics = diagnostics
            raise failure from exc
        for group_no, ((first, last), rank, active_rank) in enumerate(
            zip(intervals, group_ranks, active_ranks)
        ):
            plan.append({
                "family": family,
                "layers": family_names[first:last],
                "rank": rank,
                "trainable_rank": active_rank,
                "adapter_id": f"{family}_{group_no:03d}",
            })
        family_diagnostics.update({
            "allocated_ranks": group_ranks,
            "trainable_ranks": active_ranks,
            "total_rank": sum(group_ranks),
            "total_trainable_rank": sum(active_ranks),
        })
    return targets, plan, diagnostics


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()
            if isinstance(value, torch.Tensor)}


def generate_math(model, tokenizer, rows, *, max_new_tokens, max_seq_length, device):
    model.eval()
    results = []
    for row in rows:
        prompt = str(row["query"]).rstrip() + "\n"
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=False)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if input_ids.shape[1] > max_seq_length:
            input_ids = input_ids[:, -max_seq_length:]
            attention_mask = attention_mask[:, -max_seq_length:]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        continuation = generated[0, input_ids.shape[1]:]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        pred = extract_final_number(text)
        gold = extract_gold_number(str(row["response"]))
        results.append({
            "question": row["query"],
            "answer": row["response"],
            "generation": text,
            "prediction": pred,
            "gold": gold,
            "correct": exact_number_match(pred, gold),
        })
    return results


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(torch.cuda.current_device()) if torch.cuda.is_available() else None,
    }


def rng_restore(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"], device=torch.cuda.current_device())


def save_checkpoint(path, raw_model, optimizer, scheduler, *, epoch, next_batch_index,
                    global_step, best_validation_accuracy, plan, config, parameter_counts,
                    validation_events=None, prompt_tokens_truncated=0, answer_tokens_truncated=0,
                    rank=0, world=1):
    local_rng = rng_snapshot()
    if world > 1:
        rng_states = [None for _ in range(world)]
        torch.distributed.all_gather_object(rng_states, local_rng)
    else:
        rng_states = [local_rng]
    if rank != 0:
        return
    checkpoint = {
        "adapter_state": capture_adapter_state(raw_model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "global_step": global_step,
        "best_validation_accuracy": best_validation_accuracy,
        "plan": plan,
        "config": config,
        "parameter_counts": parameter_counts,
        "validation_events": list(validation_events or []),
        "prompt_tokens_truncated": int(prompt_tokens_truncated),
        "answer_tokens_truncated": int(answer_tokens_truncated),
        "rng_states": rng_states,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_checkpoint(path, raw_model, optimizer, scheduler, device, *, rank=0):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    load_adapter_state(raw_model, checkpoint["adapter_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    rng_states = checkpoint.get("rng_states")
    if rng_states is not None:
        if rank >= len(rng_states):
            raise ValueError(f"Checkpoint has RNG states for {len(rng_states)} ranks, requested rank {rank}")
        rng_restore(rng_states[rank])
    elif "rng_state" in checkpoint:
        rng_restore(checkpoint["rng_state"])
    return checkpoint


def make_scheduler(optimizer, total_steps, warmup_ratio, min_lr_ratio=0.0):
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    warmup_steps = int(math.ceil(total_steps * warmup_ratio))
    # The floor is an explicit conventional default when min_lr_ratio is unset.
    # It is not an interpretation of Table 9's separately recorded LR Decay=0.1.
    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return min(1.0, float(step + 1) / warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup_steps


def validation_accuracy(model, tokenizer, dataset, validation_indices, *, max_new_tokens,
                        max_seq_length, device, limit=None):
    indices = validation_indices[:limit] if limit else validation_indices
    rows = [dataset[int(idx)] for idx in indices]
    records = generate_math(
        model, tokenizer, rows,
        max_new_tokens=max_new_tokens,
        max_seq_length=max_seq_length,
        device=device,
    )
    score = sum(int(item["correct"]) for item in records) / len(records) if records else None
    return score, records


def main():
    args = parse_args()
    config = load_config(args.config, args.method)
    if args.model:
        config["model"]["repo"] = args.model
    # Dataset sampling seed is fixed separately in the shared manifest; this is
    # the independent model-initialization/training seed for this run.
    config["reproducibility"]["run_seed"] = args.seed
    config["method"] = args.method
    config["task"] = args.task
    config["run_seed"] = args.seed
    if args.max_seq_length is not None:
        config["training"]["max_seq_length"] = args.max_seq_length
    if args.max_new_tokens is not None:
        config["evaluation"]["max_new_tokens"] = args.max_new_tokens
    if args.per_device_batch_size is not None:
        config["training"]["per_device_batch_size"] = args.per_device_batch_size
    if args.gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.save_steps is not None:
        config["training"]["save_steps"] = args.save_steps
    if args.eval_steps is not None:
        config["training"]["eval_steps"] = args.eval_steps
    config["length_candidates"] = (
        [int(value.strip()) for value in args.length_candidates.split(",") if value.strip()]
        if args.length_candidates else None
    )
    config["smoke_samples"] = args.smoke_samples
    config["gradient_examples_override"] = args.gradient_examples
    if args.method == "e2lora":
        smoke_gradient_examples = args.gradient_examples
        if smoke_gradient_examples is None:
            smoke_gradient_examples = config["gradient_diagnostics"].get("sample_count")
        if smoke_gradient_examples is None and args.smoke_samples is not None:
            smoke_gradient_examples = config["gradient_diagnostics"].get("smoke_sample_count")
        if smoke_gradient_examples is not None and smoke_gradient_examples <= 2:
            config["gradient_diagnostics"]["note"] = (
                "2-example grouping is a parameter/smoke check only; "
                "not the paper's unspecified full gradient sample protocol."
            )
    resume_preview = load_resume_preview(args.resume_from, config, args.seed) if args.resume_from else None

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rank, local_rank, world = distributed_context()
    # A single run_id must be shared by all ranks; rank-local UUIDs would make
    # otherwise valid distributed checkpoints fail provenance checks.
    run_id_payload = [
        resume_preview["config"]["run_id"] if resume_preview else (uuid.uuid4().hex if rank == 0 else None)
    ]
    if world > 1:
        torch.distributed.broadcast_object_list(run_id_payload, src=0)
    config["run_id"] = run_id_payload[0]
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    if args.epochs is not None and args.max_steps is not None:
        raise ValueError("Use only one of --epochs and --max-steps")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if not args.prepare_only and not args.max_steps and not args.epochs:
        raise ValueError("Training requires --max-steps for a bounded smoke run or --epochs for a planned run")

    dataset, manifest, manifest_path = load_dataset_and_manifest(config, rank=rank, world=world)
    train_indices, validation_indices, is_smoke = get_manifest_rows(
        manifest, smoke_samples=args.smoke_samples
    )
    config["run_status"] = "SMOKE TEST" if is_smoke or args.max_steps is not None else "TRAINING"
    config["run_seed"] = args.seed
    config["manifest_path"] = str(manifest_path)
    config["manifest_sha256"] = __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    if resume_preview is not None:
        saved_manifest_sha = resume_preview.get("config", {}).get("manifest_sha256")
        if saved_manifest_sha and saved_manifest_sha != config["manifest_sha256"]:
            raise ValueError("Cannot resume: MetaMathQA sample manifest changed")
    config["actual_train_rows"] = len(train_indices)
    config["actual_validation_rows"] = len(validation_indices)
    if rank == 0:
        json_write(output_dir / "run_config.json", config)
        json_write(output_dir / "data_provenance.json", {
            "manifest_path": str(manifest_path),
            "manifest_sha256": config["manifest_sha256"],
            "repo": manifest["repo"], "revision": manifest["revision"],
            "split": manifest["split"], "dataset_rows": manifest["dataset_rows"],
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "cache_file_checksums": manifest["cache_file_checksums"],
            "train_rows": len(train_indices), "validation_rows": len(validation_indices),
            "train_validation_overlap": False,
        })

    tokenizer = tokenizer_from_config(config)
    if args.prepare_only:
        report = calculate_length_report(
            tokenizer, dataset, train_indices, validation_indices, output_dir, args.seed,
            candidate_max_lengths=config["length_candidates"],
        )
        if rank == 0:
            print(json.dumps({"status": "LENGTH ANALYSIS ONLY", "output": str(output_dir / "length_stats.json"),
                              "train": report["train"], "validation": report["validation"]}, indent=2))
        return

    training_cfg = config["training"]
    max_seq_length = args.max_seq_length or training_cfg.get("max_seq_length")
    max_new_tokens = args.max_new_tokens or config["evaluation"].get("max_new_tokens")
    if not max_seq_length or not max_new_tokens:
        raise ValueError("Run --prepare-only first, then set --max-seq-length and --max-new-tokens after reviewing validation evidence")
    per_device = args.per_device_batch_size or training_cfg.get("per_device_batch_size")
    grad_accum = args.gradient_accumulation_steps or training_cfg.get("gradient_accumulation_steps")
    if not per_device or not grad_accum:
        raise ValueError("Set per-device batch size and gradient accumulation from a GPU memory probe")
    if world * per_device * grad_accum != int(training_cfg["global_batch_size"]):
        raise ValueError(
            f"Global batch mismatch: {world} GPUs x {per_device} per-device x {grad_accum} accumulation "
            f"!= {training_cfg['global_batch_size']}"
        )
    if len(train_indices) % world:
        raise ValueError(
            f"Training rows ({len(train_indices)}) must be divisible by WORLD_SIZE ({world}); "
            "the implementation does not duplicate samples to pad distributed ranks."
        )
    per_rank_rows = len(train_indices) // world
    batches_per_epoch = math.ceil(per_rank_rows / per_device)
    updates_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    last_micro_batch_per_rank = per_rank_rows - (batches_per_epoch - 1) * per_device
    last_update_micro_batches = batches_per_epoch % grad_accum or grad_accum
    last_update_rows_per_rank = (last_update_micro_batches - 1) * per_device + last_micro_batch_per_rank
    batching_geometry = {
        "train_rows_per_epoch": len(train_indices),
        "world_size": world,
        "train_rows_per_rank": per_rank_rows,
        "per_device_batch_size": per_device,
        "batches_per_rank_per_epoch": batches_per_epoch,
        "gradient_accumulation_steps": grad_accum,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "last_micro_batch_rows_per_rank": last_micro_batch_per_rank,
        "last_optimizer_update_micro_batches": last_update_micro_batches,
        "last_optimizer_update_rows_global": last_update_rows_per_rank * world,
        "final_incomplete_micro_batch_kept": True,
        "distributed_padding_or_duplicate_rows": 0,
    }
    config["training"]["batching_geometry"] = batching_geometry
    if resume_preview is not None:
        saved_geometry = resume_preview.get("config", {}).get("training", {}).get("batching_geometry")
        if saved_geometry is not None and saved_geometry != batching_geometry:
            raise ValueError("Cannot resume with changed world size or batch geometry")
    if not torch.cuda.is_available():
        raise RuntimeError("Real Llama generation training requires CUDA; this runtime is CPU-only")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected device does not support BF16, required by the recorded NLG setup")

    # Model loading is intentionally after data/length preparation and CUDA checks.
    model, loaded_tokenizer = load_pretrained(
        config["model"]["repo"], kind="generation", revision=config["model"]["revision"],
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    tokenizer = loaded_tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.config.use_cache = False
    model.requires_grad_(False)

    targets = discover_attention_targets(model, training_cfg.get("target_projections", config["adapter"]["target_projections"]))
    gradient_diagnostics = None
    if args.method == "lora":
        plan = apply_lora(
            model, targets,
            rank=int(config["adapter"]["rank"]),
            alpha=float(config["adapter"]["alpha"]),
        )
    else:
        if resume_preview is not None:
            plan = copy.deepcopy(resume_preview["plan"])
            validate_e2_plan(plan, targets, config["adapter"])
            apply_e2lora_plan(model, plan, alpha=float(config["adapter"]["alpha"]))
            gradient_diagnostics = {"status": "REUSED SAVED CHECKPOINT PLAN; no gradients recomputed"}
            if rank == 0 and not (output_dir / "gradient_diagnostics.json").exists():
                json_write(output_dir / "gradient_diagnostics.json", {
                    "status": "RESUME; plan loaded from checkpoint, original diagnostic not present in output directory",
                    "plan": plan,
                })
        else:
            gradient_examples = args.gradient_examples
            if gradient_examples is None:
                gradient_examples = config["gradient_diagnostics"].get("sample_count")
            if gradient_examples is None:
                if args.smoke_samples is not None:
                    gradient_examples = int(config["gradient_diagnostics"]["smoke_sample_count"])
                else:
                    raise ValueError("E2LoRA requires --gradient-examples; paper does not specify a value here")
            if gradient_examples < 1 or gradient_examples > len(train_indices):
                raise ValueError("gradient_examples must be in [1, number of training rows]")
            gradient_rows = [dataset[int(i)] for i in train_indices[:gradient_examples]]
            plan_payload = [None]
            if rank == 0:
                try:
                    target_map, plan, gradient_diagnostics = build_e2_plan(
                        model, tokenizer, gradient_rows, config, device
                    )
                    validate_e2_plan(plan, targets, config["adapter"])
                    plan_payload[0] = {
                        "ok": True, "targets": target_map,
                        "plan": plan, "diagnostics": gradient_diagnostics,
                    }
                except Exception as exc:
                    plan_payload[0] = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "diagnostics": getattr(exc, "e2_diagnostics", {}),
                    }
            if world > 1:
                torch.distributed.broadcast_object_list(plan_payload, src=0)
            plan_result = plan_payload[0]
            if not plan_result["ok"]:
                if rank == 0:
                    manifest_rows = {int(item["row_index"]): item for item in manifest["train"]}
                    json_write(output_dir / "gradient_plan_failure.json", {
                        "status": "E2LORA GRADIENT/RANK PLAN FAILED; NO TRAINING STARTED",
                        "error": plan_result["error"],
                        "gradient_examples": gradient_examples,
                        "sample_rows": [manifest_rows.get(int(index), {"row_index": int(index)})
                                        for index in train_indices[:gradient_examples]],
                        "diagnostics": plan_result.get("diagnostics", {}),
                        "required_action": "Review sample rows and group proxy-entropy values; do not shift or alter rank allocation without a team-approved shared policy.",
                    })
                raise RuntimeError(
                    "E2LoRA gradient/rank plan failed on rank 0; see gradient_plan_failure.json. "
                    + plan_result["error"]
                )
            target_map = plan_result["targets"]
            plan = plan_result["plan"]
            gradient_diagnostics = plan_result["diagnostics"]
            apply_e2lora_plan(model, plan, alpha=float(config["adapter"]["alpha"]))
            if rank == 0:
                manifest_rows = {int(item["row_index"]): item for item in manifest["train"]}
                json_write(output_dir / "gradient_diagnostics.json", {
                    "status": "SMOKE/INITIALIZATION DIAGNOSTIC; not Figure 7" if is_smoke else "E2LORA INITIALIZATION DIAGNOSTIC; not Figure 7",
                    "gradient_examples": gradient_examples,
                    "sample_rows": [manifest_rows.get(int(index), {"row_index": int(index)})
                                    for index in train_indices[:gradient_examples]],
                    "targets": target_map,
                    "diagnostics": gradient_diagnostics,
                    "plan": plan,
                })
    parameter_counts = adapter_parameter_counts(model)
    if resume_preview is not None and resume_preview.get("parameter_counts") != parameter_counts:
        raise ValueError(
            "Resume adapter parameter counts differ from checkpoint; verify model architecture, target layers and adapter plan"
        )
    plan_json = {
        "run_id": config["run_id"], "method": args.method,
        "plan": plan, "parameter_counts": parameter_counts,
    }
    if rank == 0:
        json_write(output_dir / "adapter_plan.json", plan_json)
    if world > 1:
        torch.distributed.barrier()
    model_for_forward = model
    if world > 1:
        model_for_forward = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                    find_unused_parameters=False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training_cfg["learning_rate"]),
        betas=tuple(training_cfg["betas"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        total_steps = updates_per_epoch * int(args.epochs)
    scheduler_floor_ratio = training_cfg.get("min_lr_ratio")
    scheduler_floor_ratio_used = 0.0 if scheduler_floor_ratio is None else float(scheduler_floor_ratio)
    scheduler, warmup_steps = make_scheduler(
        optimizer, total_steps, float(training_cfg["warmup_ratio"]), scheduler_floor_ratio_used
    )
    config["training"]["scheduler_floor_ratio_used"] = scheduler_floor_ratio_used
    config["training"]["scheduler_floor_explanation"] = (
        "conventional cosine-to-zero default; distinct from Table 9 LR Decay=0.1"
        if scheduler_floor_ratio is None else "explicit min_lr_ratio from team configuration"
    )
    if rank == 0:
        json_write(output_dir / "run_config.json", config)
    save_steps = args.save_steps if args.save_steps is not None else training_cfg.get("save_steps")
    eval_steps = args.eval_steps if args.eval_steps is not None else training_cfg.get("eval_steps")
    if save_steps is not None and save_steps < 1:
        raise ValueError("save_steps must be positive when specified")
    if eval_steps is not None and eval_steps < 1:
        raise ValueError("eval_steps must be positive when specified")
    epoch_start = 0
    batch_start = 0
    global_step = 0
    best_validation_accuracy = -1.0
    restored = None
    if args.resume_from:
        restored = restore_checkpoint(args.resume_from, model, optimizer, scheduler, device, rank=rank)
        epoch_start = int(restored["epoch"])
        batch_start = int(restored["next_batch_index"])
        global_step = int(restored["global_step"])
        best_validation_accuracy = float(restored["best_validation_accuracy"])
    resume_start_step = global_step
    before = trainable_adapter_snapshot(model)

    if rank == 0 and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    events = list(restored.get("validation_events", [])) if restored else []
    # Totals stored in checkpoints are already global across ranks. New per-rank
    # increments are summed once at each optimizer boundary to avoid multiplying
    # the restored baseline by WORLD_SIZE.
    prompt_truncated_total = int(restored.get("prompt_tokens_truncated", 0)) if restored else 0
    answer_truncated_total = int(restored.get("answer_tokens_truncated", 0)) if restored else 0
    prompt_truncated_since_update = 0
    answer_truncated_since_update = 0

    requested_epochs = args.epochs if args.epochs is not None else math.ceil(args.max_steps / max(1, updates_per_epoch))
    for epoch in range(epoch_start, requested_epochs):
        epoch_order = random.Random(args.seed + epoch).sample(train_indices, len(train_indices))
        if len(epoch_order) % world:
            raise RuntimeError("Sample count changed after distributed geometry validation")
        rank_order = epoch_order[rank::world]
        train_loader = DataLoader(
            IndexedRows(dataset, rank_order),
            batch_size=per_device,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(collate_math, tokenizer, max_seq_length=int(max_seq_length)),
            num_workers=0,
            pin_memory=True,
        )
        optimizer.zero_grad(set_to_none=True)
        accumulation_target_tokens = 0
        micro_in_group = 0
        for batch_index, batch in enumerate(train_loader):
            if epoch == epoch_start and batch_index < batch_start:
                continue
            is_group_end = (micro_in_group + 1 >= grad_accum) or (batch_index + 1 == len(train_loader))
            sync_context = model_for_forward.no_sync() if world > 1 and not is_group_end else __import__("contextlib").nullcontext()
            prompt_truncated_since_update += int(batch.get("prompt_tokens_truncated", 0))
            answer_truncated_since_update += int(batch.get("answer_tokens_truncated", 0))
            batch = move_batch(batch, device)
            target_tokens = int((batch["labels"] != -100).sum().item())
            with sync_context:
                outputs = model_for_forward(**batch)
                (outputs.loss * target_tokens).backward()
            accumulation_target_tokens += target_tokens
            micro_in_group += 1
            if not is_group_end:
                continue

            token_tensor = torch.tensor([accumulation_target_tokens], device=device, dtype=torch.long)
            if world > 1:
                torch.distributed.all_reduce(token_tensor, op=torch.distributed.ReduceOp.SUM)
            truncation_tensor = torch.tensor(
                [prompt_truncated_since_update, answer_truncated_since_update],
                device=device, dtype=torch.long,
            )
            if world > 1:
                torch.distributed.all_reduce(truncation_tensor, op=torch.distributed.ReduceOp.SUM)
            prompt_truncated_total += int(truncation_tensor[0].item())
            answer_truncated_total += int(truncation_tensor[1].item())
            prompt_truncated_since_update = 0
            answer_truncated_since_update = 0
            scale = float(world) / max(1, int(token_tensor.item()))
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            clip = training_cfg.get("gradient_clip_norm")
            if clip:
                torch.nn.utils.clip_grad_norm_(trainable, float(clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accumulation_target_tokens = 0
            micro_in_group = 0

            next_epoch, next_batch = epoch, batch_index + 1
            if next_batch >= len(train_loader):
                next_epoch, next_batch = epoch + 1, 0
            epoch_finished = (batch_index + 1 == len(train_loader))
            do_eval = (
                (eval_steps is not None and global_step % eval_steps == 0)
                or epoch_finished
                or global_step == total_steps
            )
            score_payload = [None]
            improved = False
            if do_eval and rank == 0:
                score, val_records = validation_accuracy(
                    model, tokenizer, dataset, validation_indices,
                    max_new_tokens=int(max_new_tokens), max_seq_length=int(max_seq_length),
                    device=device,
                    limit=(min(len(validation_indices), 16) if is_smoke else None),
                )
                validation_dir = output_dir / "validation"
                validation_dir.mkdir(parents=True, exist_ok=True)
                validation_file = validation_dir / f"step_{global_step:08d}.jsonl"
                with validation_file.open("w", encoding="utf-8") as validation_handle:
                    for item in val_records:
                        validation_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                parse_failures = sum(1 for item in val_records if item.get("prediction") is None or item.get("gold") is None)
                event = {
                    "step": global_step,
                    "epoch": epoch,
                    "validation_exact_numeric_accuracy": score,
                    "validation_parse_failures": parse_failures,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "validation_examples": len(val_records),
                    "validation_generations_file": str(validation_file),
                    "prompt_tokens_truncated": prompt_truncated_total,
                    "answer_tokens_truncated": answer_truncated_total,
                    "timestamp_seconds": time.perf_counter() - started,
                }
                events.append(event)
                score_payload[0] = score
            if do_eval and world > 1:
                torch.distributed.broadcast_object_list(score_payload, src=0)
            if do_eval:
                score = score_payload[0]
                improved = score is not None and score > best_validation_accuracy
                if improved:
                    best_validation_accuracy = float(score)
                model.train()
                if improved:
                    save_checkpoint(
                        output_dir / "checkpoint_best.pt", model, optimizer, scheduler,
                        epoch=next_epoch, next_batch_index=next_batch,
                        global_step=global_step, best_validation_accuracy=best_validation_accuracy,
                        plan=plan, config=config, parameter_counts=parameter_counts,
                        validation_events=events,
                        prompt_tokens_truncated=prompt_truncated_total,
                        answer_tokens_truncated=answer_truncated_total,
                        rank=rank, world=world,
                    )
                if rank == 0:
                    json_write(output_dir / "training_log.json", {
                        "status": config["run_status"], "events": events,
                        "updates_per_epoch": updates_per_epoch,
                        "last_batch_policy": "keep final incomplete batch; loss weighted by response-token count",
                    })
            save_due = (
                (save_steps is not None and global_step % save_steps == 0)
                or epoch_finished
                or global_step == total_steps
            )
            if save_due:
                save_checkpoint(
                    output_dir / "checkpoint_last.pt", model, optimizer, scheduler,
                    epoch=next_epoch, next_batch_index=next_batch,
                    global_step=global_step, best_validation_accuracy=best_validation_accuracy,
                    plan=plan, config=config, parameter_counts=parameter_counts,
                    validation_events=events,
                    prompt_tokens_truncated=prompt_truncated_total,
                    answer_tokens_truncated=answer_truncated_total,
                    rank=rank, world=world,
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if rank == 0:
            print(f"epoch={epoch + 1} step={global_step}", flush=True)
        if args.max_steps is not None and global_step >= args.max_steps:
            break
        batch_start = 0

    if rank == 0:
        changed = trainable_adapter_changed(before, model)
        optimizer_steps_this_invocation = global_step - resume_start_step
        peak_memory = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else None
        summary = {
            "status": config["run_status"],
            "run_id": config["run_id"],
            "method": args.method,
            "task": args.task,
            "seed": args.seed,
            "actual_train_rows": len(train_indices),
            "actual_validation_rows": len(validation_indices),
            "updates_per_epoch": updates_per_epoch,
            "batches_per_rank_per_epoch": batches_per_epoch,
            "batching_geometry": batching_geometry,
            "global_batch_size": world * per_device * grad_accum,
            "per_device_batch_size": per_device,
            "gradient_accumulation_steps": grad_accum,
            "last_batch_policy": "keep incomplete batch; loss weighted by response-token count",
            "planned_steps": total_steps,
            "completed_steps": global_step,
            "warmup_steps": warmup_steps,
            "minimum_lr_ratio_configured": training_cfg.get("min_lr_ratio"),
            "scheduler_floor_ratio_used": scheduler_floor_ratio_used,
            "scheduler_floor_explanation": training_cfg["scheduler_floor_explanation"],
            "table_9_lr_decay": training_cfg.get("table_9_lr_decay"),
            "table_9_lr_decay_interpretation": training_cfg.get("table_9_lr_decay_interpretation"),
            "parameter_counts": parameter_counts,
            "pilot_parameter_reference": config.get("reproducibility", {}).get("pilot_trainable_adapter_parameters"),
            "delta_from_pilot_parameter_reference": (
                parameter_counts["trainable_adapter_params"] - int(config["reproducibility"]["pilot_trainable_adapter_parameters"])
                if config.get("reproducibility", {}).get("pilot_trainable_adapter_parameters") is not None else None
            ),
            "adapter_weights_changed_this_invocation": changed,
            "optimizer_steps_this_invocation": optimizer_steps_this_invocation,
            "resumed_from_checkpoint": str(args.resume_from) if args.resume_from else None,
            "wall_time_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": peak_memory,
            "prompt_tokens_truncated": prompt_truncated_total,
            "answer_tokens_truncated": answer_truncated_total,
            "validation_events": events,
            "best_validation_accuracy": best_validation_accuracy,
            "checkpoint_last": str(output_dir / "checkpoint_last.pt"),
            "checkpoint_best": str(output_dir / "checkpoint_best.pt"),
            "note": "This run is a smoke test when run_status=SMOKE TEST; do not report as full reproduction.",
        }
        json_write(output_dir / "result.json", summary)
        if not changed and optimizer_steps_this_invocation > 0:
            raise RuntimeError("Optimizer stepped but no trainable adapter weight changed")
        if not changed and args.resume_from is None:
            raise RuntimeError("No trainable adapter weight changed during training")
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
