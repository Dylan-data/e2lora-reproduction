"""Train RoBERTa on the five GLUE tasks in E²LoRA Table 1."""
import argparse
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
MODEL_REV = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
DATA_REV = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
TASKS = {
    "sst2": (("sentence",), 2, 16, 60, 5e-4),
    "mnli": (("premise", "hypothesis"), 3, 16, 30, 5e-4),
    "cola": (("sentence",), 2, 32, 80, 4e-4),
    "qnli": (("question", "sentence"), 2, 32, 25, 4e-4),
    "qqp": (("question1", "question2"), 2, 16, 25, 5e-4),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="sst2")
    parser.add_argument("--method", choices=["lora", "manual-sharing", "sharelora", "e2lora", "e2sharelora"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    fields, labels, batch_size, epochs, lr = TASKS[args.task]
    if args.pilot:
        epochs, batch_size = 1, 8
    length = 128 if args.pilot else 512

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
        for projection in ("query", "value")
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    diagnostics = {}
    gradient_ids = []
    if args.method in {"e2lora", "e2sharelora"}:
        gradient_ids = rng.sample(train_ids, min(len(train_ids), batch_size * 64))
        model.requires_grad_(False)
        gradients = collect_gradients(
            model, [name for names in families.values() for name in names],
            batches(train, gradient_ids),
            lambda model, batch: (model(**batch).loss, len(batch["labels"])),
        )
        plan, diagnostics = build_plan(gradients, families, base_rank=args.rank)
        del gradients
    else:
        plan = [
            {"layers": group, "rank": args.rank, "family": family}
            for family, names in families.items()
            for group in ([names] if args.method == "manual-sharing" else [[name] for name in names])
        ]
    apply_plan(model, plan, alpha=16, global_down=args.method in {"sharelora", "e2sharelora"})
    adapter_parameters = trainable_count(model)
    for group in plan:
        first, *others = [model.get_submodule(name) for name in group["layers"]]
        assert all(layer.down is first.down and layer.up is first.up for layer in others)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation = perf_counter() - start
    model.classifier.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.1,
    )
    steps = epochs * math.ceil(len(train_ids) / batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(steps * 0.06), steps)
    history = []
    start = perf_counter()
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_ids)
        for batch in batches(train, train_ids):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is nonfinite")
            loss.backward()
            optimizer.step()
            scheduler.step()
        history.append(float(loss.detach()))
        print(f"{args.task} {args.method} epoch {epoch + 1}/{epochs} loss {history[-1]:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = perf_counter() - start

    scores = {}
    model.eval()
    splits = ["validation_matched", "validation_mismatched"] if args.task == "mnli" else ["validation"]
    with torch.inference_mode():
        for split in splits:
            validation = read(split)
            indices = list(range(len(validation)))
            if args.pilot:
                indices = random.Random(args.seed).sample(indices, min(64, len(indices)))
            predictions, targets = [], []
            for batch in batches(validation, indices):
                predictions.append(model(**batch).logits.argmax(dim=-1).cpu())
                targets.append(batch["labels"].cpu())
            scores[split] = metrics(predictions, targets)
    result = {
        "task": args.task, "method": args.method, "seed": args.seed, "pilot": args.pilot,
        "model_revision": MODEL_REV, "dataset_revision": DATA_REV, "rank": args.rank,
        "rank_policy": "shift_nonpositive" if diagnostics else None, "alpha": 16,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "max_length": length,
        "train_examples": len(train_ids), "gradient_sample_ids": gradient_ids,
        "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "plan": plan, "diagnostics": diagnostics, "adapter_parameters": adapter_parameters,
        "trainable_parameters_with_classifier": trainable_count(model),
        "preparation_seconds": preparation, "training_seconds": train_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "epoch_final_losses": history, "validation": scores,
    }
    print(save_result("dylan", f"glue-{args.task}-{args.method}", result, output_root=args.output_root))


if __name__ == "__main__":
    main()
