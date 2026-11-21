"""Generate and score GSM8K using the same frozen base and saved LoRA plan."""
import argparse
from contextlib import nullcontext
from fractions import Fraction
import json
import os
import re
from time import perf_counter

from common import (attach_baseline, digest, file_sha256, load_local_model, output_path,
                    read_json, resolve, seed_everything, truncate_prompt, validate_config, write_json)
from train import load_adapter_state, require_gpu_allocation

NUMBER = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+))?")


def numeric_answer(text, *, gold=False):
    """Smoke proposal: last #### suffix, else final numeric token; exact rational comparison."""
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    elif gold:
        raise ValueError("GSM8K gold answer has no #### delimiter.")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    matches = NUMBER.findall(text)
    if not matches:
        return None
    candidate = matches[-1].replace(",", "").replace(" ", "").replace("\n", "")
    try:
        if "/" in candidate:
            numerator, denominator = candidate.split("/")
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(candidate)
    except (ValueError, ZeroDivisionError):
        return None


def score_records(records):
    if not records:
        raise ValueError("No prediction records.")
    ids = [row["source_index"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation source indices.")
    scored = []
    for row in records:
        expected = numeric_answer(row["gold"], gold=True)
        if expected is None:
            raise ValueError("Invalid numeric gold answer.")
        actual = numeric_answer(row["prediction"])
        scored.append({**row, "gold_numeric": str(expected), "prediction_numeric": str(actual) if actual is not None else None,
                       "correct": actual is not None and actual == expected})
    correct = sum(row["correct"] for row in scored)
    return {"accuracy": correct / len(scored), "accuracy_percent": 100 * correct / len(scored),
            "correct": correct, "evaluation_examples": len(scored),
            "unparseable_predictions": sum(row["prediction_numeric"] is None for row in scored)}, scored


def generate_completion(model, tokenizer, prompt, *, max_prompt_length, generation_config):
    import torch
    device = next(model.parameters()).device
    tokens, truncated = truncate_prompt(tokenizer.encode(prompt, add_special_tokens=True), max_prompt_length)
    input_ids = torch.tensor([tokens], device=device, dtype=torch.long)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast:
        generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=generation_config)
    return {"prediction": tokenizer.decode(generated[0, len(tokens):], skip_special_tokens=True),
            "prompt_input_ids": tokens, "prompt_truncated": truncated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    evaluation = config["evaluation"]
    path = resolve(evaluation["local_parquet"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing existing GSM8K parquet: {path}; no download attempted.")
    if config["run_type"] == "formal" and (args.limit is not None or evaluation.get("limit") is not None):
        raise ValueError("Formal evaluation must cover all 1319 GSM8K test cases.")
    limit = args.limit if args.limit is not None else evaluation.get("limit")
    if config["run_type"] == "smoke" and (limit is None or not 1 <= limit <= 32):
        raise ValueError("Smoke evaluation must explicitly use 1-32 cases.")
    if evaluation["do_sample"] or evaluation["num_beams"] != 1:
        raise ValueError("Current Math protocol implements greedy generation only.")
    if args.dry_run:
        print(json.dumps({"status": "validated_no_execution", "task": "math", "metric": "accuracy",
                          "evaluation_examples": limit or 1319, "protocol_status": config["protocol_status"]}, indent=2))
        return
    require_gpu_allocation()
    import pyarrow.parquet as pq
    import torch
    from transformers import GenerationConfig
    directory = output_path(config["output_dir"])
    if (directory / "evaluation.json").exists():
        raise ValueError("Evaluation already exists; preserve the prior result.")
    manifest, result = read_json(directory / "manifest.json"), read_json(directory / "result.json")
    if result["status"] != "completed" or digest(config) != result["provenance"]["config_sha256"]:
        raise ValueError("Evaluate only a completed run with its unchanged training config.")
    checkpoint = directory / "checkpoint.pt"
    if file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Checkpoint digest changed since training completed.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint and run manifest differ.")
    started = perf_counter()
    seed_everything(config["seed"])
    model, tokenizer = load_local_model(config, "cuda:0")
    if config["method"] == "e2lora":
        from adaptive_smoke import attach_prepared
        plan, _ = attach_prepared(model, config, expected_sha256=manifest["adaptive"]["prepared_plan_sha256"])
    else:
        plan = attach_baseline(model, config)
    if plan != state["plan"]:
        raise ValueError("Checkpoint plan changed.")
    load_adapter_state(model, state["adapter_state"])
    model.eval()
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1319:
        raise ValueError("Expected exactly 1319 GSM8K test cases.")
    rows = rows[:limit] if limit is not None else rows
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=evaluation["max_new_tokens"],
                                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    preparation_seconds = perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = perf_counter()
    predictions = []
    temporary = directory / "predictions.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            prompt = config["data"]["prompt_template"].format(query=row["question"])
            completion = generate_completion(model, tokenizer, prompt, max_prompt_length=config["training"]["max_prompt_length"], generation_config=generation)
            record = {"source_index": index, "question": row["question"], "gold": row["answer"],
                      **completion, "run_type": config["run_type"]}
            predictions.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    torch.cuda.synchronize()
    generation_seconds = perf_counter() - started
    scores, scored = score_records(predictions)
    # Include extraction/correctness in final per-example output; failed runs keep .tmp only.
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8")
    temporary.replace(directory / "predictions.jsonl")
    summary = {"task": "math", "method": config["method"], "metric": "accuracy", "run_type": config["run_type"], **scores,
               "eligible_for_formal_tables": config["run_type"] == "formal",
               "protocol_status": config["protocol_status"], "dataset": evaluation,
               "evaluation_source_sha256": file_sha256(path), "generation_config": generation.to_dict(),
               "scoring_rule": "last #### suffix else last numeric token; comma removal and exact Fraction equality; parse failure incorrect",
               "preparation_seconds": preparation_seconds, "generation_seconds": generation_seconds,
               "gpu_count": 1, "gpu_name": torch.cuda.get_device_name(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "gpu_process_hours": (preparation_seconds + generation_seconds) / 3600,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
               "checkpoint_sha256": result["checkpoint_sha256"]}
    write_json(directory / "evaluation.json", summary)
    write_json(directory / "result.json", {**result, **scores, "evaluation": summary})
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
