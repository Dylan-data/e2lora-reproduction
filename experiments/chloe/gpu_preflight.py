"""Bounded A40 memory probes at 2048 tokens. Default only prints a CPU plan.

Synthetic tokens test capacity, not prompts, accuracy or paper fidelity. No
checkpoint or existing Smoke output is written. Each --execute call is one case.
"""
import argparse
from copy import deepcopy
import json
from time import perf_counter

from common import (attach_baseline, load_local_model, output_path, read_json,
                    seed_everything, write_json)


def probe_plan(*, stage, checkpointing, rank, updates, accumulation):
    if stage not in {"train", "gradient"} or checkpointing not in {"on", "off"}:
        raise ValueError("Unsupported probe case.")
    if stage == "gradient" and checkpointing != "off":
        raise ValueError("Real gradient collection uses model.eval(); training checkpointing is inactive there.")
    if rank not in {4, 8, 16, 32, 64} or updates not in {1, 2} or accumulation not in {1, 2}:
        raise ValueError("Only bounded diagnostic ranks, 1-2 updates and 1-2 microbatches are supported.")
    return {"run_type": "memory_preflight", "eligible_for_formal_tables": False,
            "status": "PREPARED_NOT_EXECUTED", "task": "math", "stage": stage,
            "seed": 42, "sequence_length": 2048, "micro_batch_size": 1,
            "diagnostic_rank": rank, "checkpointing": checkpointing,
            "optimizer_updates": updates if stage == "train" else 0,
            "diagnostic_accumulation": accumulation if stage == "train" else 1,
            "formal_global_batch_target": 64, "formal_batch_validated": False,
            "gradient_targets": ["model.layers.0.self_attn.q_proj"] if stage == "gradient" else [],
            "input": "synthetic repeated valid token, full 2048 length; first half labels masked",
            "adapter_policy": "independent FP32 LoRA, diagnostic alpha=16; no inferred partial-rank policy",
            "precision": "bf16 base and autocast", "attention": "sdpa", "use_cache": False,
            "gpu_jobs_submitted": 0,
            "limitations": ["No formal Prompt, manifest or scoring validation",
                           "One gradient target cannot validate total 128-target duration/CPU memory",
                           "Independent LoRA rank does not validate actual shared E2 group ranks",
                           "Formal throughput and generation KV memory require later representative measurements"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["train", "gradient"], default="train")
    parser.add_argument("--checkpointing", choices=["on", "off"], default="on")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--updates", type=int, default=2, choices=[1, 2])
    parser.add_argument("--accumulation", type=int, default=2, choices=[1, 2])
    parser.add_argument("--output", help="Fresh JSON path under outputs/chloe/preflight/")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = probe_plan(stage=args.stage, checkpointing=args.checkpointing, rank=args.rank,
                        updates=args.updates, accumulation=args.accumulation)
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    if not args.output:
        parser.error("--execute requires a fresh --output path")
    destination = output_path(args.output)
    if not destination.is_relative_to(output_path("outputs/chloe/preflight")):
        raise ValueError("Memory probes must use the separate preflight directory.")
    if destination.exists():
        raise FileExistsError("Existing probe report is protected; choose a fresh output path.")
    from train import require_gpu_allocation
    require_gpu_allocation()
    import torch
    from e2lora.adapters.adaptive import collect_gradients
    device_name = torch.cuda.get_device_name()
    if "A40" not in device_name:
        raise RuntimeError(f"This probe is prepared for A40, got {device_name}; record a separate hardware plan.")
    cfg = deepcopy(read_json("experiments/chloe/configs/smoke-lora-r8.json"))
    cfg["adapter"]["rank"] = args.rank
    report.update(status="RUNNING", gpu=device_name, cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    started = perf_counter()
    try:
        seed_everything(42)
        model, tokenizer = load_local_model(cfg, "cuda:0")
        model.requires_grad_(False)
        report["loading_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        # Use a valid ordinary vocabulary token without selecting an unconfirmed Prompt.
        token = tokenizer.encode("1", add_special_tokens=False)[0]
        ids = torch.full((1, 2048), token, dtype=torch.long, device="cuda:0")
        labels = ids.clone()
        labels[:, :1024] = -100
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        if args.stage == "train":
            plan = attach_baseline(model, cfg)
            if args.checkpointing == "on":
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=5e-5, betas=(.9, .999), weight_decay=5e-4)
            report["trainable_parameters"] = sum(p.numel() for p in parameters)
            report["installed_groups"] = len(plan)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            timed = perf_counter()
            step_seconds = []
            for _ in range(args.updates):
                one = perf_counter()
                optimizer.zero_grad(set_to_none=True)
                for _ in range(args.accumulation):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite diagnostic loss.")
                    (loss / args.accumulation).backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise ValueError("Disconnected or nonfinite adapter gradients.")
                optimizer.step()
                torch.cuda.synchronize()
                step_seconds.append(perf_counter() - one)
            report.update(measured_optimizer_updates=args.updates, step_seconds=step_seconds,
                          measured_phase_seconds=perf_counter() - timed)
        else:
            # Mirror actual collect_gradients: eval mode, exactly one target, no optimizer.
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            timed = perf_counter()

            def loss_for_batch(current, values):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = current(**values).loss
                return loss, int(values["labels"][:, 1:].ne(-100).sum())

            gradients = collect_gradients(model, report["gradient_targets"], [batch], loss_for_batch)
            torch.cuda.synchronize()
            if any(not torch.isfinite(g).all() for g in gradients.values()):
                raise ValueError("Nonfinite sampled gradient.")
            if any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError("Sampling failed to restore the frozen base.")
            report.update(measured_optimizer_updates=0, measured_phase_seconds=perf_counter() - timed,
                          cpu_gradient_bytes=sum(g.numel() * g.element_size() for g in gradients.values()))
        report.update(status="MEMORY_PROBE_PASSED_NOT_FORMAL_READY",
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      process_seconds=perf_counter() - started)
        report["reserved_headroom_bytes"] = report["cuda_total_bytes"] - report["peak_reserved_bytes"]
        write_json(destination, report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), process_seconds=perf_counter() - started)
        write_json(destination, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
