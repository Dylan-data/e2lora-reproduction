"""Local, single-GPU Math baseline support. No downloads or job submission."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/chloe"


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def output_path(path):
    path = resolve(path)
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Run artifacts must be inside outputs/chloe/.")
    return path


def read_json(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_everything(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config, *, require_data=False):
    """Fail before model loading for pending protocols or unsupported methods."""
    if config["task"] == "code":
        protocol = read_json(config["protocol_file"])
        if protocol.get("status") != "confirmed":
            raise ValueError("Code protocol is pending: confirm source/count/prompt/generation/HumanEval with the generation member.")
        raise NotImplementedError("Code training/evaluation is a planned route; this implementation is Math baseline only.")
    if config.get("method") == "e2lora":
        from adaptive_smoke import validate_adaptive_config
        return validate_adaptive_config(config, require_data=require_data)
    if config["task"] != "math" or config["method"] != "lora":
        raise ValueError("Only the Math standard LoRA baseline is implemented.")
    if config["run_type"] not in {"smoke", "formal"}:
        raise ValueError("run_type must be smoke or formal.")
    train, data, adapter = config["training"], config["data"], config["adapter"]
    required = ["max_length", "max_prompt_length", "micro_batch_size", "gradient_accumulation", "global_batch_size", "epochs", "learning_rate", "warmup_steps", "checkpoint_every", "log_every", "max_grad_norm"]
    missing = [key for key in required if train.get(key) is None]
    if missing or not data.get("prompt_template"):
        raise ValueError(f"Unconfirmed training settings: {missing or ['prompt_template']}")
    if config["run_type"] == "formal":
        if config.get("protocol_status") != "confirmed":
            raise ValueError("Formal Math protocol has not been confirmed.")
        if train["global_batch_size"] != 64 or data.get("manifest_count") != 100000:
            raise ValueError("Formal Math requires the shared 100K manifest and global batch 64.")
    elif not isinstance(train.get("max_steps"), int) or not 1 <= train["max_steps"] <= 20:
        raise ValueError("Smoke must use at most 20 total optimizer steps, including resumed steps.")
    for key in ["micro_batch_size", "gradient_accumulation", "global_batch_size", "epochs", "checkpoint_every", "log_every"]:
        if not isinstance(train[key], int) or train[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if train["global_batch_size"] != train["micro_batch_size"] * train["gradient_accumulation"]:
        raise ValueError("This trainer is single-GPU; global_batch must equal micro_batch * accumulation.")
    if not 2 <= train["max_prompt_length"] < train["max_length"]:
        raise ValueError("max_prompt_length must leave at least one response token.")
    if train["precision"] != "bf16" or train["optimizer"] != "AdamW" or train["scheduler"] != "cosine":
        raise ValueError("Baseline requires BF16 / AdamW / cosine; quantized loading is unsupported.")
    if train.get("max_steps") is not None and (not isinstance(train["max_steps"], int) or train["max_steps"] <= 0):
        raise ValueError("max_steps must be null or a positive integer.")
    if not 0 <= train["warmup_steps"] or train["learning_rate"] <= 0 or train["max_grad_norm"] <= 0:
        raise ValueError("Invalid warmup, learning rate or gradient clipping.")
    if any(not math.isfinite(train[key]) for key in ["learning_rate", "weight_decay", "max_grad_norm"]) or train["weight_decay"] < 0:
        raise ValueError("Optimizer values must be finite, with nonnegative weight decay.")
    if len(train["betas"]) != 2 or any(not 0 <= value < 1 for value in train["betas"]):
        raise ValueError("AdamW betas must be in [0,1).")
    if not isinstance(adapter["rank"], int) or adapter["rank"] < 1 or not isinstance(adapter["alpha"], (float, int)) or not math.isfinite(adapter["alpha"]) or adapter["alpha"] <= 0:
        raise ValueError("Adapter rank and alpha must be positive.")
    if adapter["dropout"] != 0 or adapter["target_modules"] != ["q_proj", "k_proj", "v_proj", "o_proj"]:
        raise ValueError("This baseline implements dropout=0 and all four attention projections.")
    if adapter["sharing"] != "none" or adapter["rank_allocation"] != "uniform" or adapter["adapter_dtype"] != "float32":
        raise ValueError("Baseline requires independent uniform-rank FP32 adapters.")
    if data["loss"] != "response_only_token_weighted" or data["packing"] is not False:
        raise ValueError("Baseline implements response-only token-weighted loss without packing.")
    if data["truncation"] != "keep_BOS_and_prompt_tail_then_truncate_response_right":
        raise ValueError("Unsupported truncation policy.")
    evaluation = config["evaluation"]
    if evaluation["metric"] != "accuracy" or evaluation["repo"] != "openai/gsm8k" or evaluation["split"] != "test" or evaluation["configuration"] != "main":
        raise ValueError("Math must evaluate GSM8K main/test Accuracy.")
    if evaluation["few_shot_examples"] != 0 or evaluation["do_sample"] or evaluation["num_beams"] != 1:
        raise ValueError("This Math implementation uses zero-shot greedy generation.")
    if not isinstance(evaluation["max_new_tokens"], int) or evaluation["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be confirmed and positive.")
    output_path(config["output_dir"])
    if require_data:
        for path in [config["model"]["path"], data["source"], data["manifest"]]:
            if not resolve(path).exists():
                raise FileNotFoundError(f"Missing local input: {path}. Run prepare_data.py if the manifest is absent.")


def baseline_plan(model, config):
    from e2lora.adapters.lora_layers import manual_plan
    targets = [f"model.layers.{i}.self_attn.{projection}"
               for i in range(model.config.num_hidden_layers)
               for projection in config["adapter"]["target_modules"]]
    plan = manual_plan(targets, rank=config["adapter"]["rank"], share=False)
    for group in plan:
        group["family"] = group["layers"][0].rsplit(".", 1)[-1]
    return plan


def attach_baseline(model, config):
    plan = baseline_plan(model, config)
    attach_trainable_plan(model, plan, alpha=config["adapter"]["alpha"])
    return plan


def attach_trainable_plan(model, plan, *, alpha):
    from e2lora.adapters.lora_layers import apply_plan
    apply_plan(model, plan, alpha=alpha)
    # Frozen base weights remain BF16. FP32 adapters/AdamW states run under BF16 autocast.
    for group in plan:
        layer = model.get_submodule(group["layers"][0])
        layer.down.float()
        layer.up.float()


def parameter_report(model, plan):
    adapters = {}
    groups = []
    for group in plan:
        layer = model.get_submodule(group["layers"][0])
        for name in group["layers"]:
            other = model.get_submodule(name)
            if other.down is not layer.down or other.up is not layer.up:
                raise ValueError("Declared shared group does not share adapter objects.")
        for module in [layer.down, layer.up]:
            for parameter in module.parameters():
                adapters[id(parameter)] = parameter
        groups.append({**group, "trainable_rank": group["rank"], "frozen_rank": 0,
                       "in_features": layer.base.in_features, "out_features": layer.base.out_features})
    trainable = {id(p): p for p in model.parameters() if p.requires_grad}
    if set(trainable) != set(adapters):
        raise ValueError("Train exactly the unique A/B parameters; base/head must be frozen.")
    return {"adapter_parameters": sum(p.numel() for p in adapters.values()),
            "trainable_adapter_parameters": sum(p.numel() for p in adapters.values() if p.requires_grad),
            "total_trainable_parameters": sum(p.numel() for p in trainable.values()),
            "total_model_parameters_including_adapters": sum(p.numel() for p in model.parameters()),
            "unique_trainable_tensors": len(trainable), "groups": groups,
            "freeze_policy": ("base, embeddings and LM head frozen; all group-shared A/B trainable; no partial freezing"
                              if any(len(g["layers"]) > 1 for g in plan)
                              else "base, embeddings and LM head frozen; all independent A/B trainable"),
            "adapter_dtype": sorted({str(p.dtype) for p in adapters.values()}),
            "base_precision": sorted({str(model.get_submodule(g["layers"][0]).base.weight.dtype) for g in plan})}


def load_local_model(config, device):
    import torch
    from e2lora.model.pretrained import load_pretrained
    path = resolve(config["model"]["path"])
    if not path.is_dir():
        raise FileNotFoundError(path)
    offline()
    model, tokenizer = load_pretrained(str(path), kind="generation", revision=config["model"]["revision"],
                                      local_files_only=True, dtype=torch.bfloat16,
                                      attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode_example(row, tokenizer, *, template, max_length, max_prompt_length):
    if not isinstance(row.get("query"), str) or not isinstance(row.get("response"), str) or not row["response"].strip():
        raise ValueError("Math rows need string query and nonempty response.")
    prompt = tokenizer.encode(template.format(query=row["query"]), add_special_tokens=True)
    prompt, prompt_truncated = truncate_prompt(prompt, max_prompt_length)
    response = tokenizer.encode(row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
    room = max_length - len(prompt)
    response_truncated = len(response) > room
    response = response[:room]
    if not prompt or not response:
        raise ValueError("Encoding must preserve prompt context and response targets.")
    return {"input_ids": prompt + response, "labels": [-100] * len(prompt) + response,
            "prompt_truncated": prompt_truncated, "response_truncated": response_truncated}


def truncate_prompt(tokens, maximum):
    # Keep BOS and the prompt's tail, including the Answer delimiter.
    return (tokens, False) if len(tokens) <= maximum else ([tokens[0]] + tokens[-(maximum - 1):], True)


def collate(examples, pad_token_id):
    import torch
    maximum = max(len(row["input_ids"]) for row in examples)
    inputs, labels, masks = [], [], []
    for row in examples:
        length = len(row["input_ids"])
        inputs.append(row["input_ids"] + [pad_token_id] * (maximum - length))
        labels.append(row["labels"] + [-100] * (maximum - length))
        masks.append([1] * length + [0] * (maximum - length))
    return {"input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)}


def load_math_examples(config, tokenizer):
    data = config["data"]
    manifest = read_json(data["manifest"])
    source = resolve(data["source"])
    if manifest["source_revision"] != data["revision"]:
        raise ValueError("Data revision differs from the shared manifest.")
    if manifest["source_sha256"] != file_sha256(source):
        raise ValueError("MetaMath source digest differs from the shared manifest.")
    ids = manifest["indices"]
    if len(ids) != data["manifest_count"] or len(ids) != len(set(ids)):
        raise ValueError("Wrong manifest count or duplicate indices.")
    if manifest.get("indices_sha256") != digest(ids):
        raise ValueError("Manifest indices digest mismatch.")
    rows = read_json(source)
    if len(rows) != manifest["source_rows"] or any(type(i) is not int or not 0 <= i < len(rows) for i in ids):
        raise ValueError("Manifest source size or indices are invalid.")
    limit = data.get("train_limit")
    if config["run_type"] == "formal" and limit is not None:
        raise ValueError("Formal training cannot silently limit the 100K manifest.")
    if limit is not None and not 0 < limit <= len(ids):
        raise ValueError("Invalid train_limit.")
    ids = ids[:limit] if limit is not None else ids
    examples = [encode_example(rows[i], tokenizer, template=data["prompt_template"],
                               max_length=config["training"]["max_length"],
                               max_prompt_length=config["training"]["max_prompt_length"]) for i in ids]
    stats = {"examples": len(examples), "manifest_sha256": file_sha256(resolve(data["manifest"])),
             "used_indices_sha256": digest(ids), "prompt_truncated_examples": sum(x["prompt_truncated"] for x in examples),
             "response_truncated_examples": sum(x["response_truncated"] for x in examples),
             "supervised_tokens": sum(sum(v != -100 for v in x["labels"][1:]) for x in examples)}
    return examples, stats


def provenance(config):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()
    files = list((ROOT / "experiments/chloe").rglob("*.py")) + list((ROOT / "experiments/chloe/configs").glob("*.json"))
    files += list((ROOT / "experiments/chloe/slurm").glob("*.sbatch"))
    files += list((ROOT / "src/e2lora").rglob("*.py"))
    packages = {}
    for name in ["torch", "transformers", "datasets", "safetensors"]:
        packages[name] = importlib.metadata.version(name)
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "git_tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
            "source_files_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(files)},
            "packages": packages, "model": config["model"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "model_revision_provenance": "user-declared pinned revision; existing local weights; no upstream weight hash verification",
            "config_sha256": digest(config)}


def source_fingerprint(config, data_stats):
    token_files = {p.name: file_sha256(p) for p in resolve(config["model"]["path"]).glob("tokenizer*") if p.is_file()}
    return digest({"config": config, "data": data_stats, "tokenizer_files": token_files})
