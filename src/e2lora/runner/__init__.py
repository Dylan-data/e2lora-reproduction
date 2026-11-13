"""Train and score a small classification run, then store its result."""
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import torch

from e2lora.adapters import trainable_count
from e2lora.tasks import accuracy, classification_loss


def run_classification(model, train, evaluate, predict, *, lr, device):
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=lr,
    )

    def move(batch):
        return {name: value.to(device) for name, value in batch.items()}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    model.train()
    for batch in train:
        batch = move(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = classification_loss(predict(model, batch), batch["labels"])
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in evaluate:
            batch = move(batch)
            labels = batch["labels"]
            correct += (predict(model, batch).argmax(-1) == labels).sum().item()
            total += labels.numel()

    return {
        "accuracy": accuracy(correct, total),
        "evaluation_examples": total,
        "trainable_parameters": trainable_count(model),
        "training_seconds": elapsed,
        "device": str(device),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def save_result(member, name, result, *, output_root="outputs"):
    parent = Path(output_root) / member
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix=f"{name}-", dir=parent))
    path = run_dir / "result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
