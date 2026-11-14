"""Measure trainable adapters on real Llama projections after sampling gradients."""
import json
import random
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2lora.adapters import apply_plan, trainable_count
from e2lora.adapters.adaptive import collect_gradients, gradient_similarity, greedy_intervals
from sst2_partial_rank import SplitLinear

ROOT = Path(__file__).resolve().parents[2]
SETUP = json.loads((ROOT / "references/text-generation-setup.json").read_text())


def allocate(scores, budget):
    total = sum(scores)
    quotas = [budget * score / total for score in scores]
    ranks = [int(quota // 1) for quota in quotas]
    order = sorted(range(len(scores)), key=lambda i: quotas[i] - ranks[i], reverse=True)
    for index in order[:budget - sum(ranks)]:
        ranks[index] += 1
    if sum(ranks) != budget or min(ranks) < 1:
        raise ValueError(f"Invalid allocation for budget {budget}: {ranks}")
    return ranks


def samples():
    math_path = hf_hub_download("meta-math/MetaMathQA", "MetaMathQA-395K.json",
                               repo_type="dataset", revision="aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18")
    code_source = SETUP["tasks"]["code"]["public_dataset"]
    code_path = hf_hub_download(code_source["repo"], "Code-Feedback.jsonl",
                               repo_type="dataset", revision=code_source["revision"])
    with open(math_path) as file:
        rows = json.load(file)
    ids = random.Random(42).sample(range(len(rows)), 2)
    yield "GSM8K", ids, [rows[i]["query"] + "\n" + rows[i]["response"] for i in ids]
    del rows
    with open(code_path) as file:
        rows = [json.loads(line) for line in file if line.strip()]
    ids = random.Random(42).sample(range(len(rows)), 2)
    texts = []
    for i in ids:
        row = rows[i]
        if "query" in row and "answer" in row:
            texts.append(str(row["query"]) + "\n" + str(row["answer"]))
        else:
            texts.append(json.dumps(row, ensure_ascii=False))
    yield "HumanEval", ids, texts


def check(model, plan, freeze_a):
    originals = {name: model.get_submodule(name) for group in plan for name in group["layers"]}
    try:
        apply_plan(model, plan, alpha=16, global_down=freeze_a)
        for group in plan:
            layers = [model.get_submodule(name) for name in group["layers"]]
            first = layers[0]
            rank = group["trainable_rank"]
            down = first.down
            if freeze_a:
                down.requires_grad_(False)
            else:
                down = SplitLinear(down.weight, rank, axis=0)
            up_weight = first.up.weight.detach().clone()
            up_weight[:, rank:].uniform_(-0.01, 0.01)
            up = SplitLinear(up_weight, rank, axis=1)
            for layer in layers:
                layer.down, layer.up = down, up
        return trainable_count(model)
    finally:
        for name, layer in originals.items():
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, layer)


def main():
    torch.manual_seed(42)
    model_setup = SETUP["model"]
    cfg = SETUP["partial_rank_sharing"]["final_settings"]
    snapshot = snapshot_download(model_setup["repo"], revision=model_setup["revision"],
                                 allow_patterns=["config.json", "tokenizer*", "*.safetensors", "*.safetensors.index.json"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=torch.bfloat16,
                                                low_cpu_mem_usage=True).cuda()
    model.requires_grad_(False)
    model.config.use_cache = False
    families = {
        projection: [f"model.layers.{i}.self_attn.{projection}" for i in range(model.config.num_hidden_layers)]
        for projection in cfg["target_projections"]
    }
    for benchmark, ids, texts in samples():
        print(benchmark, "sampling gradients", ids, flush=True)
        batches = [tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to("cuda") for text in texts]
        plan = []
        for projection, names in families.items():
            gradients = collect_gradients(model, names, batches, lambda model, batch:
                                          (model(**batch, labels=batch.input_ids).loss, batch.input_ids.numel()))
            entropy, similarity = gradient_similarity([gradients[name] for name in names])
            del gradients
            intervals, _ = greedy_intervals(similarity)
            ranks = allocate([max(entropy[start:end]) for start, end in intervals], cfg["total_rank_per_projection"])
            active = allocate(ranks, cfg["trainable_rank_per_projection"])
            plan.extend(dict(family=projection, layers=names[start:end], rank=rank, trainable_rank=n)
                        for (start, end), rank, n in zip(intervals, ranks, active))
        counts = {"e2lora": check(model, plan, False), "e2sharelora": check(model, plan, True)}
        result = dict(benchmark=benchmark, sample_ids=ids, plan=plan, trainable_adapter_parameters=counts)
        Path(f"{benchmark}.json").write_text(json.dumps(result, indent=2))
        print(benchmark, counts, flush=True)
        assert counts == cfg["estimated_trainable_adapter_parameters"]


if __name__ == "__main__":
    main()
