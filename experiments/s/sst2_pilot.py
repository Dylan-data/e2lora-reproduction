"""Train a tiny RoBERTa/SST-2 integration run using public weights and data."""
import argparse
from time import perf_counter

import torch

from e2lora.adapters import apply_plan, manual_plan
from e2lora.dataset import load_split, sample_indices
from e2lora.model import load_pretrained, roberta_targets
from e2lora.runner import run_classification, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lora", "manual-sharing"), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model_repo = "FacebookAI/roberta-base"
    dataset_repo = "nyu-mll/glue"
    model, tokenizer = load_pretrained(
        model_repo, kind="text", revision=args.model_revision, num_labels=2,
    )
    train = load_split(dataset_repo, "sst2", split="train", revision=args.dataset_revision)
    validation = load_split(
        dataset_repo, "sst2", split="validation", revision=args.dataset_revision,
    )
    train_ids = sample_indices(len(train), 32, args.seed)
    validation_ids = sample_indices(len(validation), 32, args.seed)

    def batches(dataset):
        for start in range(0, len(dataset), 8):
            rows = dataset[start : start + 8]
            batch = tokenizer(
                rows["sentence"], padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            yield {**batch, "labels": torch.tensor(rows["label"])}

    def predict(model, batch):
        return model(**{key: value for key, value in batch.items() if key != "labels"}).logits

    start = perf_counter()
    plan = manual_plan(
        roberta_targets(model), rank=8, share=args.method == "manual-sharing",
    )
    apply_plan(model, plan, alpha=16)
    model.classifier.requires_grad_(True)
    preparation_seconds = perf_counter() - start
    result = run_classification(
        model, batches(train.select(train_ids)), batches(validation.select(validation_ids)),
        predict, lr=5e-4, device=args.device,
    )
    print(save_result("s", "sst2-pilot", {
        **result, **vars(args), "model": model_repo, "dataset": dataset_repo,
        "subset": "sst2", "train_split": "train", "evaluation_split": "validation",
        "train_indices": train_ids, "evaluation_indices": validation_ids,
        "plan": plan, "alpha": 16, "learning_rate": 5e-4,
        "preparation_seconds": preparation_seconds,
        "paper_reference": None,
    }))


if __name__ == "__main__":
    main()
