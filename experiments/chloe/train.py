"""Train the Math standard LoRA baseline inside an approved single-GPU SLURM allocation."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

from common import (attach_baseline, collate, digest, file_sha256, load_local_model,
                    load_math_examples, output_path, parameter_report, provenance,
                    read_json, resolve, seed_everything, validate_config, write_json)
from common import source_fingerprint


def save_checkpoint(path, model, plan, optimizer, scheduler, progress, fingerprint):
    import torch
    from e2lora.adapters.partial_rank import adapter_state
    state = {"schema_version": 1, "fingerprint": fingerprint, "plan": plan,
             "adapter_state": {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad},
             "complete_adapter_state": {name: value.detach().cpu().clone() for name, value in adapter_state(model).items()},
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "progress": dict(progress),
             "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_adapter_state(model, state):
    import torch
    actual = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(actual) != set(state):
        raise ValueError("Checkpoint adapter names/freeze policy differ from this model.")
    with torch.no_grad():
        for name, parameter in actual.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"Checkpoint shape mismatch: {name}")
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def load_complete_adapter_state(model, state):
    """Restore frozen adapter branches too; required for partial-rank checkpoints."""
    import torch
    from e2lora.adapters.partial_rank import adapter_state
    actual = adapter_state(model)
    if set(actual) != set(state):
        raise ValueError("Complete adapter state names/freeze policy differ.")
    with torch.no_grad():
        for name, value in actual.items():
            if value.shape != state[name].shape:
                raise ValueError(f"Complete adapter shape mismatch: {name}")
            value.copy_(state[name].to(device=value.device, dtype=value.dtype))


def fit(model, examples, plan, config, *, pad_token_id, checkpoint, fingerprint,
        resume=None, stop_after_step=None, log_path=None):
    """Deterministic example cursor and token-weighted gradient accumulation."""
    import torch
    from transformers import get_cosine_schedule_with_warmup
    train = config["training"]
    device = next(model.parameters()).device
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=train["learning_rate"],
                                  betas=tuple(train["betas"]), weight_decay=train["weight_decay"])
    per_epoch = math.ceil(math.ceil(len(examples) / train["micro_batch_size"]) / train["gradient_accumulation"])
    total_steps = train.get("max_steps") or per_epoch * train["epochs"]
    if total_steps <= 0:
        raise ValueError("No training examples or optimizer steps.")
    if total_steps > per_epoch * train["epochs"]:
        raise ValueError("Configured data/epochs cannot supply max_steps; refusing to silently shorten the run.")
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train["warmup_steps"], num_training_steps=total_steps)
    progress = {"global_step": 0, "epoch": 0, "offset": 0, "training_seconds": 0.0}
    if resume:
        # These are our local checkpoints, including Python RNG/optimizer state.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint or state["plan"] != plan:
            raise ValueError("Resume config/data/tokenizer fingerprint or adapter plan changed.")
        load_adapter_state(model, state["adapter_state"])
        if "complete_adapter_state" in state:
            load_complete_adapter_state(model, state["complete_adapter_state"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        progress = dict(state["progress"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            if device.type != "cuda" or len(state["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("Resume CUDA RNG/device count changed.")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    target = stop_after_step if stop_after_step is not None else total_steps
    if target > total_steps:
        raise ValueError("stop_after_step exceeds the configured total update budget.")
    if target <= progress["global_step"]:
        raise ValueError("Requested stop/total step is not ahead of the checkpoint.")
    model.train()
    save_seconds, last_saved = 0.0, -1
    session_start_step = progress["global_step"]
    active_epoch, order = None, None
    while progress["global_step"] < target:
        if progress["offset"] >= len(examples):
            progress["epoch"] += 1
            progress["offset"] = 0
        if progress["epoch"] >= train["epochs"]:
            raise ValueError("Cursor exhausted the configured epochs before the requested step.")
        if active_epoch != progress["epoch"]:
            order = list(range(len(examples)))
            random.Random(config["seed"] + progress["epoch"]).shuffle(order)
            active_epoch = progress["epoch"]
        start = progress["offset"]
        end = min(start + train["global_batch_size"], len(order))
        selected = [examples[i] for i in order[start:end]]
        size = train["micro_batch_size"]
        batches = [collate(selected[i:i + size], pad_token_id) for i in range(0, len(selected), size)]
        counts = [int(batch["labels"][:, 1:].ne(-100).sum()) for batch in batches]
        total_tokens = sum(counts)
        if not total_tokens or any(n == 0 for n in counts):
            raise ValueError("Every microbatch must have supervised causal targets.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        for batch, count in zip(batches, counts):
            batch = {k: value.to(device) for k, value in batch.items()}
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast:
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            (loss * (count / total_tokens)).backward()
            weighted_loss += loss.detach().float().item() * count / total_tokens
        norm = torch.nn.utils.clip_grad_norm_(parameters, train["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        progress["training_seconds"] += perf_counter() - started
        progress["global_step"] += 1
        progress["offset"] = end
        record = {"global_step": progress["global_step"], "epoch": progress["epoch"], "offset": end,
                  "loss": weighted_loss, "supervised_tokens": total_tokens, "effective_examples": len(selected),
                  "lr_after_update": scheduler.get_last_lr()[0], "gradient_norm": float(norm),
                  "run_type": config["run_type"], "session_start_step": session_start_step}
        if log_path is not None:
            with Path(log_path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        if progress["global_step"] % train["log_every"] == 0:
            print(json.dumps(record), flush=True)
        if progress["global_step"] % train["checkpoint_every"] == 0 or progress["global_step"] == target:
            started = perf_counter()
            save_checkpoint(checkpoint, model, plan, optimizer, scheduler, progress, fingerprint)
            save_seconds += perf_counter() - started
            last_saved = progress["global_step"]
    assert last_saved == progress["global_step"]
    return {"progress": progress, "planned_optimizer_steps": total_steps,
            "status": "completed" if progress["global_step"] == total_steps else "checkpointed",
            "session_checkpoint_seconds": save_seconds}


def require_gpu_allocation():
    import torch
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("8B training is allowed only inside an approved SLURM allocation; use --dry-run on the login node.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This baseline trainer is single-GPU; four-GPU training is not implemented.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one visible GPU for this trainer.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config, require_data=True)
    if args.dry_run:
        report = {"status": "validated_no_execution", "task": config["task"], "run_type": config["run_type"],
                          "config_sha256": digest(config), "output_dir": str(output_path(config["output_dir"])),
                          "gpu_job_submitted": False}
        if config["method"] == "e2lora":
            report["prepared_plan_exists"] = resolve(config["adaptive"]["prepared_plan"]).is_file()
            report["training_gate"] = "Stage A must pass and its cached plan must validate before training."
        print(json.dumps(report, indent=2))
        return
    if config["method"] == "e2lora":
        from adaptive_smoke import prepared_artifact
        prepared_artifact(config)  # Fail before GPU/model loading if Stage A is absent.
    require_gpu_allocation()
    import torch
    directory = output_path(config["output_dir"])
    if not args.resume and any((directory / name).exists() for name in ["manifest.json", "checkpoint.pt", "result.json"]):
        raise ValueError("Run directory already has state; resume explicitly or choose a fresh output_dir.")
    if args.resume and resolve(args.resume) != directory / "checkpoint.pt":
        raise ValueError("Resume the checkpoint in the configured run directory.")
    if (directory / "evaluation.json").exists():
        raise ValueError("This run already has evaluation output; use a new run directory for further training.")
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        seed_everything(config["seed"])
        model, tokenizer = load_local_model(config, "cuda:0")
        examples, data_stats = load_math_examples(config, tokenizer)
        source_identity = source_fingerprint(config, data_stats)
        adaptive_metadata = {}
        if config["method"] == "e2lora":
            from adaptive_smoke import attach_prepared
            plan, artifact = attach_prepared(model, config, source_fingerprint=source_identity)
            adaptive_metadata = {"prepared_plan_sha256": file_sha256(resolve(config["adaptive"]["prepared_plan"])),
                                 "sampling_timings": artifact["timings"], "sampling": artifact["sampling"],
                                 "preparation_gpu": artifact["gpu"], "partial_rank_sharing_enabled": False}
            fingerprint = digest({"source_fingerprint": source_identity,
                                  "prepared_plan_sha256": adaptive_metadata["prepared_plan_sha256"]})
        else:
            plan = attach_baseline(model, config)
            fingerprint = source_identity
        if config["training"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        parameters = parameter_report(model, plan)
        metadata = provenance(config)
        if args.resume:
            if read_json(directory / "manifest.json")["fingerprint"] != fingerprint:
                raise ValueError("Resume config/data/tokenizer changed; existing manifest preserved.")
        else:
            write_json(directory / "manifest.json", {"config": config, "provenance": metadata,
                                                      "data": data_stats, "plan": plan, "fingerprint": fingerprint,
                                                      "adaptive": adaptive_metadata})
        preparation = perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        outcome = fit(model, examples, plan, config, pad_token_id=tokenizer.pad_token_id,
                      checkpoint=directory / "checkpoint.pt", fingerprint=fingerprint, resume=args.resume,
                      stop_after_step=args.stop_after_step, log_path=directory / "train.jsonl")
        prior = read_json(directory / "result.json").get("sessions", []) if args.resume and (directory / "result.json").exists() else []
        session = {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "preparation_seconds": preparation,
                   "process_elapsed_seconds": perf_counter() - started, "checkpoint_seconds": outcome["session_checkpoint_seconds"],
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu_name": torch.cuda.get_device_name(),
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
        session["gpu_process_hours"] = session["process_elapsed_seconds"] / 3600
        result = {"schema_version": 1, "task": "math", "run_type": config["run_type"], "method": config["method"], "seed": config["seed"],
                  "eligible_for_formal_tables": config["run_type"] == "formal",
                  **outcome, "provenance": metadata, "parameters": parameters, "data": data_stats,
                  "gradient_seconds": 0.0, "gradient_note": "Standard LoRA does not sample gradients for grouping.",
                  "accuracy": None, "evaluation_examples": 0, "gpu_count": 1,
                  "gpu_allocation_hours": None, "allocation_hours_note": "Populate from final sacct accounting after allocation ends.",
                  "checkpoint_sha256": file_sha256(directory / "checkpoint.pt"), "sessions": prior + [session]}
        if adaptive_metadata:
            result.update(adaptive=adaptive_metadata,
                          gradient_seconds=adaptive_metadata["sampling_timings"]["gradient_seconds"],
                          gradient_note="One-time sampling belongs to the separate preparation job; not rerun on resume.")
        write_json(directory / "result.json", result)
        print(directory / "result.json", flush=True)
    except Exception as error:
        write_json(directory / "failure.json", {"run_type": config["run_type"], "task": "math", "status": "failed",
                                                "error": f"{type(error).__name__}: {error}", "accuracy": None,
                                                "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        raise


if __name__ == "__main__":
    main()
