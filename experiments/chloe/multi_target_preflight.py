"""Bounded sequential-gradient probe. Default is a CPU plan, never GPU execution.

Only 4/12 targets, 1/2 synthetic samples at 2048, zero optimizer updates. It uses
the same family collector as formal preparation, without HRA or adapter changes.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import weakref

from common import (file_sha256, load_local_model, output_path, read_json,
                    seed_everything, write_json)
from sampling_audit import backward_geometry


def probe_plan(case, steps):
    if case not in {"first-layer4", "spread12"} or type(steps) is not int or steps not in {1, 2}:
        raise ValueError("Only first-layer4/spread12 and 1-2 logical sampling steps are permitted.")
    layers = [0] if case == "first-layer4" else [0, 15, 31]
    families = {p: [f"model.layers.{i}.self_attn.{p}" for i in layers]
                for p in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    count = sum(len(v) for v in families.values())
    return {"run_type": "multi_target_memory_preflight", "eligible_for_formal_tables": False,
            "status": "PREPARED_NOT_EXECUTED", "case": case, "sequence_length": 2048,
            "seed": 42, "families": families,
            "geometry": backward_geometry(targets=count, steps=steps, batch_size=1, micro_batch_size=1),
            "optimizer_updates": 0, "model_mode": "eval", "checkpointing": "off",
            "input": "1-2 distinct synthetic repeated-token sequences; no MetaMathQA/Prompt claim",
            "cache_policy": "FP32 CPU means retained for one family; release before next family",
            "raw_gradients_saved": False, "adapter_or_rank_policy_changed": False,
            "full_128_target_sampling_authorized": False, "gpu_jobs_submitted": 0}


def cpu_memory():
    result = {"rss_bytes": None, "peak_rss_bytes": None}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM"}:
                result["rss_bytes" if key == "VmRSS" else "peak_rss_bytes"] = int(value.split()[0]) * 1024
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["first-layer4", "spread12"], default="first-layer4")
    parser.add_argument("--sampling-steps", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-phase-seconds", type=int, default=600)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_phase_seconds <= 600:
        parser.error("Phase budget must be 1-600 seconds, checked between targets; SLURM supplies the hard limit.")
    report = probe_plan(args.case, args.sampling_steps)
    report["max_phase_seconds_checked_between_targets"] = args.max_phase_seconds
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    if not args.output:
        parser.error("--execute requires a fresh output path")
    destination = output_path(args.output)
    if not destination.is_relative_to(output_path("outputs/chloe/preflight/multi-target2048")):
        raise ValueError("Use the separate multi-target2048 output directory.")
    progress = destination.with_suffix(".targets.jsonl")
    if destination.exists() or progress.exists():
        raise FileExistsError("Existing report/progress is protected; choose a fresh name.")
    from train import require_gpu_allocation
    require_gpu_allocation()
    import torch
    from e2lora.adapters.preparation import collect_family_gradients
    if "A40" not in torch.cuda.get_device_name():
        raise RuntimeError("This capacity probe requires an A40 allocation.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.update(slurm_job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                  cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  source_sha256={str(p): file_sha256(p) for p in [Path(__file__),
                      Path("src/e2lora/adapters/adaptive.py"), Path("src/e2lora/adapters/preparation.py")]},
                  per_target=[], per_family=[])
    started = perf_counter()
    with progress.open("x", encoding="utf-8") as stream:
        try:
            seed_everything(42)
            config = read_json("experiments/chloe/configs/smoke-lora-r8.json")
            model, tokenizer = load_local_model(config, "cuda:0")
            model.requires_grad_(False)
            model.eval()
            token_ids = [tokenizer.encode(str(i + 1), add_special_tokens=False)[0] for i in range(args.sampling_steps)]
            if len(set(token_ids)) != len(token_ids):
                raise ValueError("Synthetic sample tokens must be distinct.")
            batches = []
            for token in token_ids:
                ids = torch.full((1, 2048), token, dtype=torch.long)
                labels = ids.clone()
                labels[:, :1024] = -100
                batches.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels))
            report["synthetic_sample_token_ids"] = token_ids
            report["loading_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            phase_start = perf_counter()
            forward_calls = 0

            def loss_for_batch(current, cpu_batch):
                nonlocal forward_calls
                values = {k: v.to("cuda:0") for k, v in cpu_batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = current(**values).loss
                forward_calls += 1
                return loss, int(cpu_batch["labels"][:, 1:].ne(-100).sum())

            for family, targets in report["families"].items():
                if perf_counter() - phase_start > args.max_phase_seconds:
                    raise TimeoutError("Sampling phase budget exceeded; no additional family started.")

                def observed(detail):
                    prefix = detail["target_index_in_family"] + 1
                    expected_bytes = sum(model.get_submodule(name).weight.numel() * 4 for name in targets[:prefix])
                    if (detail["target"] != targets[prefix - 1]
                            or detail["cpu_gradient_cache_bytes"] != expected_bytes):
                        raise ValueError("Observed CPU family cache differs from the actual target prefix.")
                    row = dict(detail, family=family, expected_cpu_gradient_cache_bytes=expected_bytes, **cpu_memory())
                    report["per_target"].append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(json.dumps(row), flush=True)
                    if perf_counter() - phase_start > args.max_phase_seconds:
                        raise TimeoutError("Phase budget exceeded at target boundary; stopping without expansion.")

                gradients = collect_family_gradients(model, targets, batches, loss_for_batch, on_target_complete=observed)
                if not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
                    raise ValueError("Nonfinite CPU gradient.")
                family_bytes = sum(g.numel() * g.element_size() for g in gradients.values())
                gradient_refs = [weakref.ref(g) for g in gradients.values()]
                del gradients
                torch.cuda.synchronize()
                live_tensors = sum(ref() is not None for ref in gradient_refs)
                report["per_family"].append(dict(family=family, released_gradient_bytes=family_bytes,
                    cpu_gradient_tensors_released=live_tensors == 0, cpu_gradient_live_tensor_count=live_tensors,
                    cuda_allocated_after_release_bytes=torch.cuda.memory_allocated(),
                    cuda_reserved_after_release_bytes=torch.cuda.memory_reserved(), **cpu_memory()))
                if live_tensors:
                    raise ValueError("CPU gradient mean tensors are still referenced after family release.")
                if any(p.requires_grad or p.grad is not None for p in model.parameters()):
                    raise ValueError("Frozen base flags/grad buffers were not restored.")
            report.update(status="MULTI_TARGET_PROBE_COMPLETED_REQUIRES_REVIEW",
                          measured_phase_seconds=perf_counter() - phase_start,
                          measured_forward_calls=forward_calls,
                          measured_autograd_grad_calls=sum(r["backward_passes"] for r in report["per_target"]),
                          measured_optimizer_updates=0, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(), process_seconds=perf_counter() - started)
            if (report["measured_autograd_grad_calls"] != report["geometry"]["autograd_grad_calls"]
                    or forward_calls != report["geometry"]["forward_calls"]):
                raise ValueError("Actual backward count differs from the bounded plan.")
            report["counter_method"] = "forward callback invocations counted; each successful callback is paired with one autograd.grad in the collector"
            allocated = [r["cuda_allocated_after_bytes"] for r in report["per_target"]]
            report["post_target_allocated_growth_last_minus_first_bytes"] = allocated[-1] - allocated[0]
            report["timing_scope"] = "full bounded sampling phase includes target synchronization, observation, finite checks and family release; per-target time covers gradient collection and CPU transfer"
            report["full_sampling_time_prediction_seconds"] = None
            write_json(destination, report)
            print(json.dumps({"status": report["status"], "output": str(destination)}))
            return 0
        except Exception as error:
            report.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(), process_seconds=perf_counter() - started)
            write_json(destination, report)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
