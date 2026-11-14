"""Train RoBERTa on the five GLUE tasks in E²LoRA Table 1."""
import argparse
import json
import math
from pathlib import Path
import random
from time import perf_counter

import pyarrow.parquet as parquet
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from e2lora.adapters import apply_plan, trainable_count
from e2lora.adapters.adaptive import build_plan, collect_gradients
from e2lora.runner import save_result

ROOT = Path(__file__).resolve().parents[2]
SETUP = json.loads((ROOT / "references/text-understanding-setup.json").read_text())
TRAINING = SETUP["training"]
MODEL_REV = SETUP["model"]["revision"]
DATA_REV = SETUP["dataset"]["revision"]
TASKS = {
    "sst2": (("sentence",), 2),
    "mnli": (("premise", "hypothesis"), 3),
    "cola": (("sentence",), 2),
    "qnli": (("question", "sentence"), 2),
    "qqp": (("question1", "question2"), 2),
}


def metrics(predictions, labels):
    predictions, labels = torch.cat(predictions), torch.cat(labels)
    correct = predictions == labels
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "examples": labels.numel(), "accuracy": float(correct.float().mean()),
        "matthews": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


BASELINES = {"fft", "dora", "dora-r4", "adalora", "adalora-r4", "loraplus", "lora-fa", "vera"}


def attach_baseline(model, method, rank, total_steps):
    from peft import AdaLoraConfig, LoraConfig, VeraConfig, get_peft_model

    targets = SETUP["model"]["target_layers"]
    alpha = TRAINING["lora_alpha"]
    if method == "vera":
        config = VeraConfig(r=256, target_modules=targets)
    elif method.startswith("adalora"):
        # The E²LoRA paper does not specify the allocation schedule for these rows.
        config = AdaLoraConfig(
            init_r=max(12, rank), target_r=rank, total_step=total_steps,
            tinit=0, tfinal=max(1, total_steps // 4), deltaT=1,
            lora_alpha=alpha, target_modules=targets,
        )
    else:
        config = LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=targets,
            use_dora=method.startswith("dora"),
        )
    return get_peft_model(model, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="sst2")
    parser.add_argument("--method", choices=sorted(BASELINES | {"lora", "manual-sharing", "sharelora", "e2lora", "e2sharelora"}), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=TRAINING["base_rank"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--e2-variant", choices=("original", "freeze-a", "query-only"), default="original")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.e2_variant != "original" and args.method != "e2lora":
        parser.error("--e2-variant requires --method e2lora")
    if args.method.endswith("-r4"):
        args.rank = 4
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    fields, labels = TASKS[args.task]
    settings = SETUP["tasks"][args.task]
    batch_size, epochs, lr = (settings[key] for key in ("batch_size", "epochs", "learning_rate"))
    if args.pilot:
        epochs, batch_size = 1, 8
    length = 128 if args.pilot else TRAINING["max_length"]

    model_path = ROOT / "src/e2lora/model/weights/FacebookAI/roberta-base" / MODEL_REV
    data_path = ROOT / "src/e2lora/dataset/data/glue" / DATA_REV / args.task
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=labels, local_files_only=True,
    ).to(device)

    def read(split):
        files = sorted(data_path.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No local Parquet files for {split} in {data_path}")
        return parquet.read_table(files, columns=[*fields, "label"])

    train = read("train")
    train_ids = list(range(len(train)))
    if args.pilot:
        train_ids = rng.sample(train_ids, min(32, len(train_ids)))

    def batches(table, indices):
        for start in range(0, len(indices), batch_size):
            rows = table.take(indices[start : start + batch_size]).to_pydict()
            inputs = tokenizer(
                *[[value or "" for value in rows[field]] for field in fields],
                padding=True, truncation=True, max_length=length, return_tensors="pt",
            )
            yield {
                **{name: value.to(device) for name, value in inputs.items()},
                "labels": torch.tensor(rows["label"], device=device),
            }

    families = {
        projection: [
            f"roberta.encoder.layer.{i}.attention.self.{projection}"
            for i in range(model.config.num_hidden_layers)
        ]
        for projection in SETUP["model"]["target_layers"]
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    diagnostics = {}
    gradient_ids = []
    steps = epochs * math.ceil(len(train_ids) / batch_size)
    classifier = model.classifier
    if args.method == "fft":
        model.requires_grad_(True)
        plan = []
    elif args.method in BASELINES:
        plan = []
        model = attach_baseline(model, args.method, args.rank, steps)
        if args.method == "lora-fa":
            for name, param in model.named_parameters():
                if ".lora_A." in name:
                    param.requires_grad_(False)
    elif args.method in {"e2lora", "e2sharelora"}:
        gradient_ids = rng.sample(train_ids, min(len(train_ids), batch_size * TRAINING["gradient_sampling_steps"]))
        model.requires_grad_(False)
        gradients = collect_gradients(
            model, [name for names in families.values() for name in names],
            batches(train, gradient_ids),
            lambda model, batch: (model(**batch).loss, len(batch["labels"])),
        )
        plan, diagnostics = build_plan(gradients, families, base_rank=args.rank, rank_policy=SETUP["rank_policy"])
        del gradients
    else:
        plan = [
            {"layers": group, "rank": args.rank, "family": family}
            for family, names in families.items()
            for group in ([names] if args.method == "manual-sharing" else [[name] for name in names])
        ]
    if args.e2_variant == "query-only":
        plan = [group for group in plan if group["family"] == "query"]
    if plan:
        apply_plan(model, plan, alpha=TRAINING["lora_alpha"], global_down=args.method in {"sharelora", "e2sharelora"})
        for group in plan:
            first, *others = [model.get_submodule(name) for name in group["layers"]]
            assert all(layer.down is first.down and layer.up is first.up for layer in others)
            if args.e2_variant == "freeze-a":
                first.down.requires_grad_(False)
    adapter_parameters = None if args.method == "fft" else trainable_count(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation = perf_counter() - start
    classifier.requires_grad_(True)
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            rate = lr * 16 if args.method == "loraplus" and ".lora_B." in name else lr
            groups.setdefault(rate, []).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "lr": rate} for rate, parameters in groups.items()],
        lr=lr, weight_decay=TRAINING["weight_decay"], betas=tuple(TRAINING["betas"]),
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(steps * TRAINING["warmup_ratio"]), steps)
    history = []
    start = perf_counter()
    step = 0
    adaptive_rank = args.method.startswith("adalora")
    allocator_fields = ("ipt", "exp_avg_ipt", "exp_avg_unc")
    first_epoch = 0
    elapsed_before = 0.0
    if args.checkpoint and args.checkpoint.is_file():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["weights"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        rng.setstate(checkpoint["shuffle_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        if adaptive_rank:
            adaptive_state = checkpoint["adalora"]
            model.peft_config["default"].rank_pattern = adaptive_state["rank_pattern"]
            for field in allocator_fields:
                tensors = adaptive_state[field]
                setattr(model.base_model.rankallocator, field, {name: t.to(device) for name, t in tensors.items()})
        first_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        train_ids = checkpoint["train_ids"]
        elapsed_before = checkpoint["training_seconds"]
        step = first_epoch * math.ceil(len(train_ids) / batch_size)
    for epoch in range(first_epoch, epochs):
        epoch_start = perf_counter()
        model.train()
        rng.shuffle(train_ids)
        for batch in batches(train, train_ids):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is nonfinite")
            loss.backward()
            optimizer.step()
            if adaptive_rank:
                model.base_model.update_and_allocate(step + 1)
            step += 1
            scheduler.step()
        history.append(float(loss.detach()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(f"{args.task} {args.e2_variant} epoch {epoch + 1}/{epochs} loss {history[-1]:.4f} seconds {perf_counter() - epoch_start:.2f}", flush=True)
        if args.checkpoint:
            temporary = args.checkpoint.with_suffix(".tmp")
            adaptive_state = None
            if adaptive_rank:
                adaptive_state = {
                    field: {name: t.cpu() for name, t in getattr(model.base_model.rankallocator, field).items()}
                    for field in allocator_fields
                }
                adaptive_state["rank_pattern"] = model.peft_config["default"].rank_pattern
            torch.save({
                "adalora": adaptive_state,
                "weights": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "shuffle_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "epoch": epoch + 1, "history": history, "train_ids": train_ids,
                "training_seconds": elapsed_before + perf_counter() - start,
            }, temporary)
            temporary.replace(args.checkpoint)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = elapsed_before + perf_counter() - start

    scores = {}
    validation_sample_ids = {}
    model.eval()
    splits = ["validation_matched", "validation_mismatched"] if args.task == "mnli" else ["validation"]
    with torch.inference_mode():
        for split in splits:
            validation = read(split)
            indices = list(range(len(validation)))
            if args.pilot:
                indices = random.Random(args.seed).sample(indices, min(64, len(indices)))
            if args.pilot:
                validation_sample_ids[split] = indices
            predictions, targets = [], []
            for batch in batches(validation, indices):
                predictions.append(model(**batch).logits.argmax(dim=-1).cpu())
                targets.append(batch["labels"].cpu())
            scores[split] = metrics(predictions, targets)
    active_adapter_parameters = adapter_parameters
    if adaptive_rank:
        config = model.peft_config["default"]
        masks = config.rank_pattern or {}
        if not masks:
            raise RuntimeError("AdaLoRA did not produce a rank allocation.")
        active_ranks = sum(sum(mask) for mask in masks.values())
        width = model.config.hidden_size
        active_adapter_parameters = active_ranks * (2 * width + 1)
    result = {
        "task": args.task, "method": args.method, "seed": args.seed, "pilot": args.pilot,
        "model_revision": MODEL_REV, "dataset_revision": DATA_REV, "rank": args.rank,
        "rank_policy": SETUP["rank_policy"] if diagnostics else None, "alpha": TRAINING["lora_alpha"],
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "max_length": length,
        "train_examples": len(train_ids), "gradient_sample_ids": gradient_ids,
        "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "e2_variant": args.e2_variant, "train_sample_ids": sorted(train_ids),
        "validation_sample_ids": validation_sample_ids,
        "plan": plan, "diagnostics": diagnostics, "adapter_parameters": adapter_parameters,
        "active_adapter_parameters": active_adapter_parameters,
        "adalora_active_ranks": active_ranks if adaptive_rank else None,
        "loraplus_b_lr_multiplier": 16 if args.method == "loraplus" else None,
        "vera_rank": model.peft_config["default"].r if args.method == "vera" else None,
        "seed": args.seed,
        "trainable_parameters_with_classifier": trainable_count(model),
        "preparation_seconds": preparation, "training_seconds": train_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "epoch_final_losses": history, "validation": scores,
    }
    print(save_result("text-understanding", f"glue-{args.task}-{args.method}-{args.e2_variant}", result, output_root=args.output_root))


if __name__ == "__main__":
    main()
