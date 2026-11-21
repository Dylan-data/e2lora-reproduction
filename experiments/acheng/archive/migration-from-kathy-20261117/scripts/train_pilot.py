"""EuroSAT integration only: four updates, score, checkpoint and exact resume check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from datasets import load_dataset
from e2lora.model.clip_classification import (load_clip, visual_families, logits, text_features,
                                             MODEL, ORIGINAL, CONVERTED)
from e2lora.adapters.adaptive import collect_gradients, build_plan, gradient_similarity, greedy_intervals
from e2lora.adapters.lora_layers import apply_plan, trainable_count

DATA = "tanganke/eurosat"
REV = "43750fc422c7fbd4289e2df3f7473617a5937acc"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["lora", "e2lora", "manual"], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise RuntimeError("Result already exists")
    seed = 2026
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train = load_dataset(DATA, revision=REV, split="train")
    test = load_dataset(DATA, revision=REV, split="test")
    classes = train.features["label"].names
    assert classes == test.features["label"].names and len(classes) == 10
    # Stratified holdout comes exclusively from train. test never selects settings.
    pools = [[] for _ in classes]
    for i, y in enumerate(train["label"]):
        pools[y].append(i)
    rng = random.Random(seed)
    training, validation = [], []
    for pool in pools:
        rng.shuffle(pool)
        n = max(1, len(pool) // 10)
        validation.extend(pool[:n])
        training.extend(pool[n:])
    training = rng.sample(training, 256)
    validation = rng.sample(validation, 128)
    testing = random.Random(seed).sample(range(len(test)), 128)
    assert not set(training).intersection(validation)
    split = dict(dataset=DATA, revision=REV, train_count=len(train), test_count=len(test),
                 training_indices=training, validation_indices=validation, test_indices=testing,
                 validation_source="stratified 10% holdout from train; pilot subset128", classes=classes)
    save_json(out / "split.json", split)
    model, processor = load_clip()
    text, prompts = text_features(model, processor, classes)
    torch.save(text.cpu(), out / "text-features.pt")
    families = visual_families(model)
    targets = [n for names in families.values() for n in names]
    inventory = [dict(name=n, shape=list(m.weight.shape), targeted=n in targets)
                 for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    save_json(out / "linear-layers.json", inventory)
    save_json(out / "preprocessing.json", processor.image_processor.to_dict())

    def batches(ds, indices):
        for offset in range(0, len(indices), 64):
            ids = indices[offset:offset + 64]
            rows = [ds[i] for i in ids]
            pixels = processor(images=[r["image"].convert("RGB") for r in rows], return_tensors="pt")["pixel_values"].cuda()
            labels = torch.tensor([r["label"] for r in rows], device="cuda")
            yield pixels, labels

    def loss_for_batch(m, b):
        x, y = b
        return nn.functional.cross_entropy(logits(m, x, text), y), len(y)

    diagnostics = {}
    if a.method == "e2lora":
        gstart = time.perf_counter()
        gradients = collect_gradients(model, targets, batches(train, training[:64]), loss_for_batch)
        gradient_seconds = time.perf_counter() - gstart
        torch.save(gradients, out / "training-gradients.pt")
        # Persist diagnostics even when literal rank allocation fails.
        raw = {}
        for family, names in families.items():
            entropy, matrix = gradient_similarity([gradients[n] for n in names])
            intervals, thresholds = greedy_intervals(matrix)
            raw[family] = dict(entropy=entropy, rmi=matrix, intervals=intervals, thresholds=thresholds,
                               representatives=[max(entropy[i:j]) for i,j in intervals])
        save_json(out / "pre-allocation.json", raw)
        plan, diagnostics = build_plan(gradients, families, base_rank=8, rank_policy="paper")
        del gradients
    else:
        gradient_seconds = 0
        width = 1 if a.method == "lora" else 2
        plan = [dict(layers=names[i:i+width], rank=8, family=f)
                for f, names in families.items() for i in range(0, len(names), width)]
    save_json(out / "plan.json", dict(plan=plan, diagnostics=diagnostics, rank_policy="paper"))
    apply_plan(model, plan, alpha=16)
    expected = sum(g["rank"] * (model.get_submodule(g["layers"][0]).base.in_features +
                                model.get_submodule(g["layers"][0]).base.out_features) for g in plan)
    count = trainable_count(model)
    assert count == expected
    for g in plan:
        first = model.get_submodule(g["layers"][0])
        assert all(model.get_submodule(n).down is first.down and model.get_submodule(n).up is first.up for n in g["layers"])
    frozen_before = {n:digest(v) for n,v in model.named_parameters() if not v.requires_grad}
    active = {n:v for n,v in model.named_parameters() if v.requires_grad}
    before = {n:v.detach().cpu().clone() for n,v in active.items()}
    optimizer = torch.optim.Adam(active.values(), lr=1e-4, betas=(.9,.999), weight_decay=0)
    train_batches = list(batches(train, training))
    prepare_seconds = time.perf_counter() - start
    torch.cuda.synchronize()
    tstart = time.perf_counter()
    losses = []
    for step, batch in enumerate(train_batches):
        lr = 1e-4 * min(1., (step+1)/(4*.3))
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_for_batch(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in active.values())
        optimizer.step()
        losses.append(loss.item())
        print(f"{a.method}: step {step+1}/4 loss={loss.item():.6f}", flush=True)
        # Persist enough state to resume at the next deterministic batch.
        temp = out / "checkpoint.tmp"
        torch.save(dict(adapter={n:v.detach().cpu() for n,v in active.items()}, optimizer=optimizer.state_dict(),
                        next_step=step+1, plan=plan, split=split, seed=seed,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temp)
        os.replace(temp, out / "checkpoint.pt")
        if step == 1:
            # Compare an uninterrupted optimizer update with a restored update below.
            midpoint = torch.load(out / "checkpoint.pt", weights_only=False)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - tstart
    final = {n:v.detach().cpu().clone() for n,v in active.items()}
    # Restore the midpoint, repeat the remaining steps, compare bitwise with uninterrupted training.
    with torch.no_grad():
        for n,v in active.items():
            v.copy_(midpoint["adapter"][n])
    optimizer.load_state_dict(midpoint["optimizer"])
    torch.set_rng_state(midpoint["torch_rng"])
    torch.cuda.set_rng_state_all(midpoint["cuda_rng"])
    for step in range(midpoint["next_step"], 4):
        for g in optimizer.param_groups:
            g["lr"] = 1e-4 * min(1., (step+1)/(4*.3))
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_for_batch(model, train_batches[step])
        loss.backward()
        optimizer.step()
    assert all(torch.equal(v.detach().cpu(), final[n]) for n,v in active.items()), "Resume differs"
    assert frozen_before == {n:digest(v) for n,v in model.named_parameters() if not v.requires_grad}
    changed = [n for n,v in active.items() if not torch.equal(v.detach().cpu(), before[n])]
    assert changed and any("up.weight" in n for n in changed) and any("down.weight" in n for n in changed)

    @torch.no_grad()
    def score(ds, ids):
        labels, predictions = [], []
        for x,y in batches(ds, ids):
            predictions.extend(logits(model,x,text).argmax(-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
        correct = sum(p==y for p,y in zip(predictions,labels))
        return dict(count=len(ids),correct=correct,accuracy=100*correct/len(ids),labels=labels,predictions=predictions,indices=ids)

    result = dict(status="pilot", method=a.method, seed=seed, model=MODEL, original_revision=ORIGINAL,
                  safetensors_revision=CONVERTED, dataset=DATA, dataset_revision=REV,
                  validation=score(train,validation),test=score(test,testing),prompts=prompts,
                  adapter_parameters=count,total_trainable_parameters=count,classification_head_parameters=0,
                  updated_parameter_tensors=changed,frozen_unchanged=True,exact_resume_passed=True,
                  losses=losses,gradient_seconds=gradient_seconds,prepare_seconds=prepare_seconds,
                  training_seconds=training_seconds,elapsed_seconds=time.perf_counter()-start,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__,steps=4,batch_size=64,precision="fp32",
                  scheduler="pilot choice: 30% linear warmup then constant",alpha=16)
    result["gpu_hours"] = result["elapsed_seconds"]/3600
    save_json(out / "result.json",result)
    print(json.dumps({k:result[k] for k in ["method","adapter_parameters","training_seconds","peak_cuda_bytes","exact_resume_passed"]}), flush=True)
    print("Test accuracy",result["test"]["accuracy"],flush=True)


if __name__ == "__main__":
    main()
