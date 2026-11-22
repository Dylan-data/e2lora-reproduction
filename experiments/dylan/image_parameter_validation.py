"""Measure CLIP sharing plans on fixed EuroSAT training samples."""
from fractions import Fraction
from io import BytesIO
import json
from pathlib import Path
import random
import torch
import pyarrow.parquet as pq
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from e2lora.adapters.adaptive import collect_gradients, build_plan

root = Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
torch.manual_seed(42)
model_path = root / "src/e2lora/model/weights/openai/clip-vit-base-patch16/57c216476eefef5ab752ec549e440a49ae4ae5f3"
data_path = root / "src/e2lora/dataset/data/eurosat/43750fc422c7fbd4289e2df3f7473617a5937acc/train.parquet"
model = CLIPModel.from_pretrained(model_path, local_files_only=True).cuda().eval()
processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
classes = ["annual crop", "forest", "herbaceous vegetation", "highway", "industrial buildings", "pasture", "permanent crop", "residential buildings", "river", "sea or lake"]
table = pq.read_table(data_path)
print(table.schema, flush=True)
ids = random.Random(42).sample(range(len(table)), 512)
texts = processor(text=[f"a photo of a {name}." for name in classes], return_tensors="pt", padding=True)
with torch.no_grad():
    features = model.get_text_features(**{k:v.cuda() for k,v in texts.items()})
    features = features / features.norm(dim=-1, keepdim=True)
families = {}
for name, module in model.named_modules():
    if name.startswith("vision_model.encoder.layers.") and isinstance(module, torch.nn.Linear):
        families.setdefault(name.split(".")[-1], []).append(name)
print("targets", families, flush=True)
model.requires_grad_(False)
def batches():
    for start in range(0, len(ids), 8):
        rows = table.take(ids[start:start+8]).to_pylist()
        images = [Image.open(BytesIO(row["image"]["bytes"])).convert("RGB") for row in rows]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].cuda()
        yield pixels, torch.tensor([row["label"] for row in rows], device="cuda")
def loss_for_batch(model, batch):
    pixels, labels = batch
    vision = model.vision_model(pixel_values=pixels).pooler_output
    image_features = model.visual_projection(vision)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = model.logit_scale.exp() * image_features @ features.T
    return torch.nn.functional.cross_entropy(logits, labels), len(labels)
gradients = collect_gradients(model, [n for ns in families.values() for n in ns], batches(), loss_for_batch)
plan, diagnostics = build_plan(gradients, families, base_rank=8, rank_policy="paper")
def allocate(weights, budget):
    quotas = [Fraction(w)*budget/sum(weights) for w in weights]
    result = [int(q) for q in quotas]
    order = sorted(range(len(weights)), key=lambda i: quotas[i]-result[i], reverse=True)
    for i in order[:budget-sum(result)]:
        result[i] += 1
    return result
counts = {"lora_rank8": 0, "sharelora_rank8": 0, "literal_e2lora": 0, "fixed_budget_e2lora": 0, "partial48_e2lora": 0, "partial52_e2lora": 0, "partial48_e2share_frozen_A": 0, "partial48_e2share_trainable_A": 0}
for family,names in families.items():
    layer = model.get_submodule(names[0])
    ni,no = layer.in_features,layer.out_features
    groups = [g for g in plan if g["family"] == family]
    ranks = [g["rank"] for g in groups]
    fixed = allocate(ranks,96)
    active48, active52 = allocate(fixed,48), allocate(fixed,52)
    counts["lora_rank8"] += len(names)*8*(ni+no)
    counts["sharelora_rank8"] += ni*8+len(names)*no*8
    counts["literal_e2lora"] += sum(ranks)*(ni+no)
    counts["fixed_budget_e2lora"] += 96*(ni+no)
    counts["partial48_e2lora"] += 48*(ni+no)
    counts["partial52_e2lora"] += 52*(ni+no)
    counts["partial48_e2share_frozen_A"] += 48*no
    counts["partial48_e2share_trainable_A"] += 48*no+max(active48)*ni
    diagnostics[family].update(shape=[no,ni],fixed_ranks=fixed,active48=active48,active52=active52)
result = dict(model_revision=model_path.name,dataset_revision=data_path.parent.name,seed=42,sample_ids=ids,sample_count=len(ids),micro_batch=8,gradient_batches=64,classes=classes,counts=counts,plan=plan,diagnostics=diagnostics)
output=root/"outputs/dylan/image-parameter-validation/result.json"
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2))
print(json.dumps(counts),flush=True)
from e2lora.adapters import apply_plan, trainable_count
from sst2_partial_rank import SplitLinear
for family in families:
    groups = [g for g in plan if g["family"] == family]
    fixed = allocate([g["rank"] for g in groups], 96)
    active = allocate(fixed, 48)
    for group, rank, trained in zip(groups, fixed, active):
        group.update(rank=rank, trainable_rank=trained)
apply_plan(model, plan, alpha=16)
actual_full = trainable_count(model)
for group in plan:
    layers = [model.get_submodule(name) for name in group["layers"]]
    down = SplitLinear(layers[0].down.weight, group["trainable_rank"], axis=0)
    up = SplitLinear(layers[0].up.weight, group["trainable_rank"], axis=1)
    for layer in layers:
        layer.down, layer.up = down, up
    assert all(layer.down is down and layer.up is up for layer in layers)
actual_partial = trainable_count(model)
assert actual_full == counts["fixed_budget_e2lora"]
assert actual_partial == counts["partial48_e2lora"]
loss, _ = loss_for_batch(model, next(batches()))
loss.backward()
result["actual_parameter_checks"] = dict(full=actual_full, partial48=actual_partial, backward_finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()))
result["plan"] = plan
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result["actual_parameter_checks"]), flush=True)
