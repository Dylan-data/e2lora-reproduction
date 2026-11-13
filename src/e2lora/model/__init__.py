"""Load public starting weights through a shared entry point."""


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
