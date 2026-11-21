#!/usr/bin/env python
"""Read-only validation of the existing EuroSAT zero-shot and pilot artifacts."""

import argparse
import json
import math
from pathlib import Path


HISTORICAL_ROOT = Path("outputs/image-classification")
ZERO_SHOT_RUNS = (
    "eurosat-zero-shot-pilot-54494",
    "eurosat-zero-shot-full-54495",
)
TRAINING_RUNS = (
    "eurosat-lora-pilot-54503",
    "eurosat-manual-pilot-54505",
    "eurosat-e2lora-pilot-54504",
)


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required historical artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_predictions(labels, predictions, indices, *, count, correct, accuracy, label_count):
    if not (len(labels) == len(predictions) == len(indices) == count):
        raise ValueError("Prediction, label, index, and declared example counts differ.")
    if len(set(indices)) != len(indices):
        raise ValueError("Evaluation sample indices are not unique.")
    if any(type(label) is not int or not 0 <= label < label_count for label in labels):
        raise ValueError("An evaluation label is outside the declared class range.")
    if any(type(prediction) is not int or not 0 <= prediction < label_count for prediction in predictions):
        raise ValueError("A predicted class is outside the declared class range.")
    actual_correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    actual_accuracy = 100.0 * actual_correct / count
    if actual_correct != correct or not math.isclose(actual_accuracy, accuracy, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Stored accuracy does not match the stored labels and predictions.")
    return actual_correct, actual_accuracy


def audit_zero_shot(root, run_name):
    result = load_json(root / run_name / "result.json")
    classes = result["classes"]
    prompts = result["prompts"]
    if len(classes) != 10 or len(set(classes)) != len(classes):
        raise ValueError(f"{run_name}: expected 10 unique class names.")
    expected_prompts = [f"a photo of a {name}" for name in classes]
    if prompts != expected_prompts:
        raise ValueError(f"{run_name}: class prompts are not aligned with class order.")
    correct, accuracy = validate_predictions(
        result["labels"],
        result["predictions"],
        result["indices"],
        count=result["count"],
        correct=result["correct"],
        accuracy=result["accuracy"],
        label_count=len(classes),
    )
    print(
        f"PASS {run_name}: classes/prompts={len(classes)}, "
        f"accuracy={accuracy:.12f}% ({correct}/{result['count']})"
    )
    return classes


def audit_training_pilot(root, run_name, reference_split):
    run_dir = root / run_name
    split = load_json(run_dir / "split.json")
    result = load_json(run_dir / "result.json")
    train = split["training_indices"]
    validation = split["validation_indices"]
    test = split["test_indices"]
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise ValueError(f"{run_name}: train/validation/test indices overlap.")
    if (len(train), len(validation), len(test)) != (256, 128, 128):
        raise ValueError(f"{run_name}: unexpected pilot split sizes.")
    split_signature = (train, validation, test)
    if reference_split is not None and split_signature != reference_split:
        raise ValueError(f"{run_name}: pilot method used a different fixed split.")

    classes = split["classes"]
    if len(classes) != 10 or len(set(classes)) != 10:
        raise ValueError(f"{run_name}: invalid class-name mapping.")
    if result["prompts"] != [f"a photo of a {name}" for name in classes]:
        raise ValueError(f"{run_name}: class prompts do not match split class order.")
    for metric in ("validation", "test"):
        part = result[metric]
        if part["indices"] != split[f"{metric}_indices"]:
            raise ValueError(f"{run_name}: {metric} predictions do not match split indices.")
        validate_predictions(
            part["labels"],
            part["predictions"],
            part["indices"],
            count=part["count"],
            correct=part["correct"],
            accuracy=part["accuracy"],
            label_count=len(classes),
        )
    if result["steps"] != 4 or len(result["losses"]) != 4:
        raise ValueError(f"{run_name}: unexpected pilot step count.")
    if not all(math.isfinite(loss) for loss in result["losses"]):
        raise ValueError(f"{run_name}: non-finite training loss.")
    if not result.get("exact_resume_passed") or not result.get("frozen_unchanged"):
        raise ValueError(f"{run_name}: recorded parameter/resume checks did not pass.")
    print(
        f"PASS {run_name}: split disjoint and aligned, "
        f"params={result['adapter_parameters']}, "
        f"validation={result['validation']['accuracy']:.6f}%, "
        f"test={result['test']['accuracy']:.6f}%"
    )
    return split_signature, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root containing outputs/image-classification.",
    )
    args = parser.parse_args()
    historical_root = args.repo_root / HISTORICAL_ROOT

    class_maps = [audit_zero_shot(historical_root, run) for run in ZERO_SHOT_RUNS]
    if any(classes != class_maps[0] for classes in class_maps[1:]):
        raise ValueError("Zero-shot pilot and full-test class order differ.")

    reference_split = None
    pilot_results = {}
    for run in TRAINING_RUNS:
        reference_split, result = audit_training_pilot(
            historical_root, run, reference_split
        )
        pilot_results[run] = result

    lora_params = pilot_results["eurosat-lora-pilot-54503"]["adapter_parameters"]
    e2_params = pilot_results["eurosat-e2lora-pilot-54504"]["adapter_parameters"]
    print(
        "REVIEW E² rank-budget discrepancy: "
        f"LoRA={lora_params}, E²LoRA={e2_params}, "
        f"ratio={e2_params / lora_params:.6f}; not a formal result."
    )


if __name__ == "__main__":
    main()
