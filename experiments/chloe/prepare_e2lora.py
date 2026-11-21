"""Stage A: sample gradients, build LSS/HRA groups, count and probe shared adapters."""
import argparse
import os
from time import perf_counter

from adaptive_smoke import sampling_batches
from common import (attach_trainable_plan, baseline_plan, digest, load_local_model, load_math_examples,
                    output_path, parameter_report, provenance, read_json, resolve, seed_everything,
                    source_fingerprint, validate_config, write_json)
from train import require_gpu_allocation


def probe_gradients(model, plan, batch, loss_for_batch, base_loss):
    """No optimizer update: check zero-B initialization and shared backward connectivity."""
    import torch
    loss, _ = loss_for_batch(model, batch)
    torch.testing.assert_close(loss.detach(), base_loss, rtol=1e-6, atol=1e-6)
    loss.backward()
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
        raise ValueError("An adapter is disconnected or has nonfinite gradients.")
    up = [p for name, p in parameters.items() if ".up." in name]
    if not any(p.grad.abs().max().item() > 0 for p in up):
        raise ValueError("No B component received a nonzero gradient.")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError("Frozen base parameters received accumulated gradients.")
    report = {"optimizer_updates": 0, "initial_loss_before_adapter": base_loss.item(),
              "initial_loss_after_adapter": loss.detach().item(), "all_trainable_tensors_connected": True,
              "nonzero_gradient_tensors": sum(p.grad.abs().max().item() > 0 for p in parameters.values()),
              "unique_trainable_tensors": len(parameters),
              "note": "With B initialized to zero, A gradients are expected to be zero on this first backward."}
    model.zero_grad(set_to_none=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config, require_data=True)
    if config["method"] != "e2lora":
        raise ValueError("Use the independent E2LoRA Smoke configuration.")
    if args.dry_run:
        print({"status": "validated_no_execution", "gpu_job_submitted": False,
               "gradient_sampling": config["adaptive"]["sampling"],
               "partial_rank_sharing": config["adaptive"]["partial_rank_sharing"],
               "prepared_plan_exists": resolve(config["adaptive"]["prepared_plan"]).exists()})
        return
    directory = output_path(config["output_dir"])
    if any((directory / name).exists() for name in ["prepared-plan.json", "manifest.json", "checkpoint.pt", "result.json"]):
        raise ValueError("Existing prepared/training state is protected; choose a new E2LoRA run directory.")
    require_gpu_allocation()
    import torch
    from e2lora.adapters.preparation import collect_plan
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        seed_everything(config["seed"])
        model, tokenizer = load_local_model(config, "cuda:0")
        model.requires_grad_(False)
        model.eval()
        examples, data_stats = load_math_examples(config, tokenizer)
        batches, positions = sampling_batches(examples, config, tokenizer.pad_token_id)
        batches = [{k: v.to("cuda:0") for k, v in batch.items()} for batch in batches]
        independent = baseline_plan(model, config)
        families = {projection: [g["layers"][0] for g in independent if g["family"] == projection]
                    for projection in config["adapter"]["target_modules"]}

        def loss_for_batch(base, batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = base(**batch).loss
            return loss, int(batch["labels"][:, 1:].ne(-100).sum())

        torch.cuda.reset_peak_memory_stats()
        plan, diagnostics, timings = collect_plan(
            model, families, batches, loss_for_batch, base_rank=config["adapter"]["rank"],
            rank_policy=config["adaptive"]["rank_policy"],
            diagnostics_callback=lambda family, detail: write_json(directory / f"diagnostics-{family}.json", detail))
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Sampling did not restore the frozen, gradient-free base.")
        with torch.no_grad():
            base_loss = loss_for_batch(model, batches[0])[0].detach()
        attach_trainable_plan(model, plan, alpha=config["adapter"]["alpha"])
        parameters = parameter_report(model, plan)
        probe = probe_gradients(model, plan, batches[0], loss_for_batch, base_loss)
        manifest = read_json(config["data"]["manifest"])
        source_ids = manifest["indices"][:config["data"]["train_limit"]]
        sampling = {**config["adaptive"]["sampling"], "positions_in_training_pool": positions,
                    "source_row_indices": [source_ids[i] for i in positions],
                    "supervised_tokens_per_batch": [int(b["labels"][:, 1:].ne(-100).sum()) for b in batches],
                    "cache_contents": "plan, entropy and RMI diagnostics; raw dense gradients are not persisted",
                    "gradient_mean": "CPU FP32 sum weighted by supervised response tokens, divided before computing entropy"}
        torch.cuda.synchronize()
        artifact = {"schema_version": 1, "status": "prepared", "run_type": "smoke", "method": "e2lora",
                    "eligible_for_formal_tables": False, "optimizer_updates": 0, "partial_rank_sharing_enabled": False,
                    "config_sha256": digest(config), "source_fingerprint": source_fingerprint(config, data_stats),
                    "data": data_stats, "sampling": sampling, "plan": plan, "diagnostics": diagnostics,
                    "layer_ranks": {name: {"rank": g["rank"], "group_index": i, "family": g["family"],
                                           "trainable_rank": g["rank"], "frozen_rank": 0}
                                    for i, g in enumerate(plan) for name in g["layers"]},
                    "parameters": parameters, "gradient_probe": probe, "timings": timings,
                    "total_preparation_process_seconds": perf_counter() - started,
                    "gpu": {"name": torch.cuda.get_device_name(), "count": 1,
                            "slurm_job_id": os.environ["SLURM_JOB_ID"],
                            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()},
                    "provenance": provenance(config), "accuracy": None}
        artifact["gpu"]["gpu_process_hours"] = artifact["total_preparation_process_seconds"] / 3600
        write_json(resolve(config["adaptive"]["prepared_plan"]), artifact)
        print({"status": "prepared", "groups": len(plan), "trainable_parameters": parameters["total_trainable_parameters"],
               "optimizer_updates": 0, "timings": timings}, flush=True)
    except Exception as error:
        write_json(directory / "preparation-failure.json", {"run_type": "smoke", "status": "failed",
                                                           "optimizer_updates": 0, "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
