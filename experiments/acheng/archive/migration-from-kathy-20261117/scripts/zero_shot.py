import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--limit", type=int, default=128, help="0 scores the full test split")
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
if (output / "result.json").exists():
    raise RuntimeError("Result already exists; use a new output directory")
assert torch.cuda.is_available(), "CUDA unavailable"
torch.set_num_threads(4)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

model_id = "openai/clip-vit-base-patch16"
original = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
converted = "5ef227a78de3f75873f373246dac80def63b0003"
data_id = "tanganke/eurosat"
data_revision = "43750fc422c7fbd4289e2df3f7473617a5937acc"

print("Loading pinned EuroSAT test split", flush=True)
data = load_dataset(data_id, revision=data_revision, split="test")
classes = data.features["label"].names
assert len(classes) == len(set(classes)) == 10
if args.limit < 0:
    raise ValueError("--limit must be nonnegative")
indices = random.Random(2026).sample(range(len(data)), min(args.limit, len(data))) if args.limit else list(range(len(data)))
prompts = ["a photo of a " + name for name in classes]
print("Class prompts:", prompts, flush=True)

print("Loading pinned CLIP safetensors conversion", flush=True)
processor = CLIPProcessor.from_pretrained(
    model_id, revision=original, use_fast=False
)
model = CLIPModel.from_pretrained(
    model_id, revision=converted, use_safetensors=True,
    attn_implementation="eager",
).float().cuda().eval()
model.requires_grad_(False)

predictions, labels = [], []
with torch.inference_mode():
    tokens = processor(
        text=prompts, padding=True, return_tensors="pt"
    ).to("cuda")
    text = model.get_text_features(**tokens)
    text = torch.nn.functional.normalize(text, dim=-1)

    for offset in range(0, len(indices), 16):
        rows = [data[i] for i in indices[offset:offset + 16]]
        images = [row["image"].convert("RGB") for row in rows]
        pixels = processor(
            images=images, return_tensors="pt"
        )["pixel_values"].cuda()
        features = model.get_image_features(pixel_values=pixels)
        features = torch.nn.functional.normalize(features, dim=-1)
        logits = model.logit_scale.exp() * features @ text.T
        predictions.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(int(row["label"]) for row in rows)
        print(f"Scored {len(labels)}/{len(indices)}", flush=True)

torch.cuda.synchronize()
correct = sum(p == y for p, y in zip(predictions, labels))
result = dict(
    status="pilot" if args.limit else "full-test", dataset=data_id, dataset_revision=data_revision,
    split="test", model=model_id, original_revision=original,
    safetensors_revision=converted, method="zero-shot", seed=2026,
    count=len(labels), correct=correct,
    accuracy=100 * correct / len(labels),
    indices=indices, classes=classes, prompts=prompts,
    labels=labels, predictions=predictions, precision="fp32",
    preprocessing=processor.image_processor.to_dict(),
    gpu=torch.cuda.get_device_name(0),
    torch_version=torch.__version__,
    peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    elapsed_seconds=time.perf_counter() - start,
)
(output / "result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2)
)
print(
    f"Evaluation accuracy: {result['accuracy']:.2f}% ({correct}/{len(labels)})",
    flush=True,
)
