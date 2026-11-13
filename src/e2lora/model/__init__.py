"""Load public starting weights through a shared entry point."""
import json
from pathlib import Path


def load_pretrained(repo, *, kind, revision, cache_dir=None, **options):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoProcessor,
        AutoTokenizer,
        CLIPModel,
    )

    loaders = {
        "text": (AutoModelForSequenceClassification, AutoTokenizer),
        "vision": (CLIPModel, AutoProcessor),
        "generation": (AutoModelForCausalLM, AutoTokenizer),
    }
    model_loader, input_loader = loaders[kind]
    local = Path(__file__).parent / "weights" / repo / revision
    index = local / "model.safetensors.index.json"
    if index.is_file():
        weights = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        ready = bool(weights) and all((local / name).is_file() for name in weights)
    else:
        ready = any((local / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    if ready and (local / "config.json").is_file():
        repo = str(local)
        source = {"local_files_only": True}
    else:
        source = {"revision": revision, "cache_dir": cache_dir}
    return (
        model_loader.from_pretrained(repo, **source, **options),
        input_loader.from_pretrained(repo, **source),
    )


def roberta_targets(model):
    return [
        f"roberta.encoder.layer.{index}.attention.self.{name}"
        for index in range(len(model.roberta.encoder.layer))
        for name in ("query", "value")
    ]
