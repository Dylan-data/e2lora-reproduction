"""Protocol-gated single-GPU ablations; current approved scope is Math only.

Default is CPU preflight only. GPU execution is an explicit later action.
The member-4 evaluator owns prompts, decoding, GSM8K/HumanEval scoring.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

from ablations import materialize
from alignment import protocol_issues
from common import (collate, digest, file_sha256, load_local_model, load_math_examples,
                    output_path, provenance, read_json, resolve, seed_everything,
                    source_fingerprint, write_json)
from e2lora.adapters.lora_layers import apply_plan
from e2lora.adapters.partial_rank import active_rank_plan, apply_partial_plan, count_adapters
from e2lora.adapters.preparation import collect_plan
from train import fit, require_gpu_allocation
from experiment_scope import require_active_task


def runtime_config(spec):
    require_active_task(spec["task"])
    protocol = spec["resolved_protocol"]
    task = protocol["tasks"][spec["task"]]
    train = {**protocol["training"], **task["training"]}
    train.pop("seeds")
    data = {"source": task["data"]["path"], "revision": task["data"]["revision"],
            "manifest": task["manifest"]["path"], "manifest_count": task["manifest"]["count"],
            "train_limit": None, "prompt_template": task["prompt"]["train_template"]}
    return {"task": spec["task"], "method": spec["method"], "run_type": "formal", "seed": spec["seed"],
            "model": protocol["model"], "training": train, "data": data, "output_dir": spec["output_dir"]}


def fixed_sampling_batches(examples, *, steps, batch_size, micro_batch_size, seed, pad_token_id):
    """Nested subsets: logical S means exactly S*B distinct examples, no cycling.

    Microbatch counts never redefine the sweep's logical step count.
    collect_gradients weights each mean loss by supervised causal token count.
    """
    if any(type(x) is not int or x < 1 for x in [steps, batch_size, micro_batch_size]):
        raise ValueError("Sampling steps and batch sizes must be positive integers.")
    count = steps * batch_size
    if count > len(examples):
        raise ValueError("Sampling subset exceeds the manifest; no implicit repetition.")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    selected = order[:count]
    batches = []
    for first in range(0, count, batch_size):
        logical = selected[first:first + batch_size]
        for start in range(0, batch_size, micro_batch_size):
            batches.append(collate([examples[i] for i in logical[start:start + micro_batch_size]], pad_token_id))
    return batches, {"logical_steps": steps, "batch_size": batch_size,
                     "micro_batch_size": micro_batch_size, "examples": count,
                     "selected_positions": selected, "selected_positions_sha256": digest(selected),
                     "micro_batches": len(batches), "seed": seed}


def install(model, plan, spec):
    protocol = spec["resolved_protocol"]
    policy = protocol["mechanisms"]["adapter_policies"][spec["method"]]
    alpha = protocol["adapter"]["alpha"]
    if policy["mode"] == "partial_group_ab":
        budget = policy["active_rank_per_projection_by_base_rank"].get(str(spec["base_rank"]))
        if type(budget) is not int:
            raise ValueError("No confirmed active budget for this sweep rank.")
        plan = active_rank_plan(plan, {g["family"]: budget for g in plan})
        apply_partial_plan(model, plan, alpha=alpha, frozen_b_init_scale=policy["frozen_b_init_scale"])
    else:
        apply_plan(model, plan, alpha=alpha)
    for group in plan:
        layer = model.get_submodule(group["layers"][0])
        layer.down.float()
        layer.up.float()
    return plan, count_adapters(model, plan)


def validate_resolved(spec):
    if "resolved_protocol" not in spec:
        return ["unresolved experiment; use ablations.py compile with a confirmed protocol first"]
    protocol = spec["resolved_protocol"]
    issues = protocol_issues(protocol, spec["task"])
    if spec.get("protocol_sha256") != digest(protocol):
        issues.append("resolved protocol digest changed")
    # Recompile from the original specification: changing unrelated fields is rejected.
    from ablations import specifications
    source = next((s for s in specifications() if s["run_id"] == spec["run_id"]), None)
    if source is None:
        issues.append("unknown experiment ID")
    elif not issues and materialize(source, protocol, spec["repeat"]) != spec:
        issues.append("compiled experiment differs from canonical specification")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved output of ablations.py compile")
    parser.add_argument("--execute", action="store_true", help="Only after explicit GPU authorization")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    process_started = perf_counter()
    spec = read_json(args.config)
    issues = validate_resolved(spec)
    if issues:
        raise ValueError("Formal run blocked: " + "; ".join(issues))
    if not args.execute:
        print(json.dumps({"status": "validated_no_execution", "run_id": spec["run_id"],
                          "gpu_jobs_submitted": 0, "task": spec["task"]}, indent=2))
        return
    require_gpu_allocation()
    import torch
    directory = output_path(spec["output_dir"])
    checkpoint = directory / "checkpoint.pt"
    if args.resume and resolve(args.resume) != checkpoint:
        raise ValueError("Resume only this run's own checkpoint.")
    if not args.resume and any((directory / name).exists() for name in ["manifest.json", "checkpoint.pt", "result.json"]):
        raise ValueError("Output already has state; resume explicitly.")
    if (directory / "evaluation.json").exists():
        raise ValueError("Evaluation already exists; use a new run directory.")
    directory.mkdir(parents=True, exist_ok=True)
    config = runtime_config(spec)
    seed_everything(spec["seed"])
    model, tokenizer = load_local_model(config, "cuda:0")
    model.requires_grad_(False)
    examples, data_stats = load_math_examples(config, tokenizer)
    train = config["training"]
    per_epoch = math.ceil(len(examples) / train["global_batch_size"])
    train["warmup_steps"] = math.ceil((train.get("max_steps") or per_epoch * train["epochs"]) * train["warmup_ratio"])
    identity = digest({"spec": spec, "source": source_fingerprint(config, data_stats)})
    families = {p: [f"model.layers.{i}.self_attn.{p}" for i in range(model.config.num_hidden_layers)]
                for p in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    if args.resume:
        manifest = read_json(directory / "manifest.json")
        if manifest["fingerprint"] != identity:
            raise ValueError("Resume protocol/data/config identity changed.")
        plan, diagnostics, timings, sampling = [manifest[k] for k in ["plan", "diagnostics", "timings", "sampling"]]
    else:
        sampling, diagnostics, timings = {}, {}, {"gradient_seconds": 0.0}
        if spec["method"] == "lora":
            plan = [dict(family=p, layers=[name], rank=spec["base_rank"])
                    for p, names in families.items() for name in names]
        else:
            opts = spec["resolved_protocol"]["sampling"]
            sample_seed = spec["seed"] if opts["seed_policy"] == "training_seed" else opts["fixed_seed"]
            batches, sampling = fixed_sampling_batches(examples, steps=spec["sampling_steps"],
                batch_size=opts["batch_size"], micro_batch_size=opts["micro_batch_size"],
                seed=sample_seed, pad_token_id=tokenizer.pad_token_id)

            def loss_for_batch(m, batch):
                batch = {k: v.to("cuda:0") for k, v in batch.items()}
                count = int(batch["labels"][:, 1:].ne(-100).sum())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = m(**batch).loss
                return loss, count

            metric = spec["similarity"]
            convention = (None if metric == "rmi" else spec["resolved_protocol"]["mechanisms"]["metric_conventions"][metric])
            plan, diagnostics, timings = collect_plan(model, families, batches, loss_for_batch,
                base_rank=spec["base_rank"], rank_policy="paper",
                mechanism_options=dict(lss=spec["lss"], hra=spec["hra"], similarity=metric, convention=convention),
                diagnostics_callback=lambda family, detail: write_json(directory / f"diagnostics-{family}.json", detail))
            del batches
    # Sampling and installation use separate RNG streams to prevent a step sweep
    # from changing initialization merely by advancing the sampling RNG.
    seed_everything(spec["seed"])
    plan, parameters = install(model, plan, spec)
    timings["preparation_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if train["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not args.resume:
        write_json(directory / "manifest.json", dict(spec=spec, config=config, plan=plan, parameters=parameters,
            diagnostics=diagnostics, timings=timings, sampling=sampling, fingerprint=identity, provenance=provenance(config)))
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    outcome = fit(model, examples, plan, config, pad_token_id=tokenizer.pad_token_id,
                  checkpoint=checkpoint, fingerprint=identity, resume=args.resume, log_path=directory / "train.jsonl")
    write_json(directory / "result.json", dict(run_type="formal", eligible_for_formal_tables=False,
        task=spec["task"], run_id=spec["run_id"], seed=spec["seed"], protocol_sha256=spec["protocol_sha256"],
        parameters=parameters, timings=timings, training=outcome,
        training_session_seconds=perf_counter() - started, gpu_name=torch.cuda.get_device_name(),
        process_seconds_before_evaluation=perf_counter() - process_started,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu_count=1,
        finished_training_at_utc=datetime.now(timezone.utc).isoformat(),
        gpu_allocation_hours=None,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), checkpoint_sha256=file_sha256(checkpoint),
        status="TRAINED_PENDING_MEMBER4_EVALUATION_AND_BUDGET_AUDIT"))
    # Free the GPU before invoking the independently pinned evaluation driver.
    del model
    torch.cuda.empty_cache()
    evaluator = spec["resolved_protocol"]["tasks"][spec["task"]]["evaluator"]
    substitutions = {"python": sys.executable, "script": str(resolve(evaluator["path"])),
                     "config": str(resolve(args.config)), "checkpoint": str(checkpoint),
                     "output": str(directory / "evaluation.json")}
    command = [item.format_map(substitutions) for item in evaluator["argv"]]
    if substitutions["script"] not in command:
        raise ValueError("Evaluator command must include the pinned evaluator script.")
    subprocess.run(command, check=True, cwd=resolve("."))
    evaluation = read_json(directory / "evaluation.json")
    expected = spec["resolved_protocol"]["tasks"][spec["task"]]["evaluation"]
    if (evaluation.get("run_type") != "formal" or evaluation.get("task") != spec["task"]
            or evaluation.get("evaluation_examples") != expected["count"]
            or evaluation.get("protocol_sha256") != spec["protocol_sha256"]
            or evaluation.get("checkpoint_sha256") != file_sha256(checkpoint)
            or evaluation.get("metric") != expected["metric"]
            or evaluation.get("task_ids_sha256") != expected["task_ids_sha256"]
            or not isinstance(evaluation.get("score"), (int, float))
            or not math.isfinite(evaluation["score"]) or not 0 <= evaluation["score"] <= 1):
        raise ValueError("Member-4 evaluator output violates config_checkpoint_output_v1.")
    print(directory / "result.json")


if __name__ == "__main__":
    main()
