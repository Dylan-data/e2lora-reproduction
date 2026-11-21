"""Recalculate Math-only counts and explicit risk scenarios, not a reservation."""
import argparse
from collections import Counter
import json
import math

from ablations import specifications
from common import read_json, write_json


def estimate():
    specs = specifications()
    by_study = Counter(s["study"] for s in specs)
    measured = read_json("experiments/chloe/report/resource-estimates.scenario.json")["measurement_sources"]
    train_seconds = measured["stageB_train_seconds_for_20_updates_global8_length256"]
    gradient_seconds = measured["stageA_gradient_seconds_for_2x1_samples_at_length256"]
    steps = [s["sampling_steps"] for s in specs if s["study"] == "figure5_steps"]
    runs = len(specs) * 3
    # Both are illustrative extrapolations of the same short Smoke, not a confidence interval.
    sample_only = train_seconds * (100000 / 160) / 3600
    cap_scaled = sample_only * (2048 / 256)
    gradient_per_example_cap_scaled = gradient_seconds / 2 * (2048 / 256) / 3600
    sampling_example = {"label": "illustrative_only_not_confirmed_protocol", "batch_size": 64,
                        "main_steps": 64, "rank_sweep_fixed_steps": 64,
                        "sampling_micro_batch_size": 1, "reuse": False}
    # Nine adaptive Table4/5 configurations + five rank settings + the seven step settings.
    logical = 3 * (9 * sampling_example["main_steps"] + 5 * sampling_example["rank_sweep_fixed_steps"] + sum(steps))
    gradient_total = logical * sampling_example["batch_size"] * gradient_per_example_cap_scaled
    return {"run_type": "math_only_resource_planning", "date": "2026-11-18", "gpu_jobs_submitted": 0,
            "formal_gpu_execution_authorized": False, "configurations": len(specs),
            "configurations_by_study": dict(by_study), "planned_repeats": 3, "planned_runs": runs,
            "runs_by_study": {k: v * 3 for k, v in by_study.items()},
            "known_seed": 42, "first_seed_configurations": len(specs), "additional_repeat_seeds": None,
            "code_gpu_runs": 0, "table10_cpu_only": True,
            "one_epoch_updates_at_100k_batch64": math.ceil(100000 / 64), "last_update_examples": 32,
            "total_optimizer_updates_if_one_epoch_each": runs * math.ceil(100000 / 64),
            "base_training_gpu": {"count": 1, "model": "A40", "micro_batch_candidate": 1,
                                  "accumulation_candidate_for_global64": 64,
                                  "checkpointing": "candidate; compare on/off, formal value unconfirmed"},
            "measurements_from_existing_smoke": measured,
            "training_extrapolations_not_bounds": {
                "sample_count_only_hours_per_run": sample_only,
                "sample_count_only_hours_first_seed": sample_only * len(specs),
                "sample_count_only_hours_all_repeats": sample_only * runs,
                "all_examples_at_cap2048_vs256_hours_per_run": cap_scaled,
                "all_examples_at_cap2048_vs256_hours_first_seed": cap_scaled * len(specs),
                "all_examples_at_cap2048_vs256_hours_all_repeats": cap_scaled * runs},
            "sampling_formula_logical_steps_all_repeats": "3 * (9 * main_steps + 5 * rank_fixed_steps + 1016)",
            "sampling_illustrative_assumptions": sampling_example,
            "sampling_cap_scaled_risk_example": {
                "logical_steps_all_repeats": logical,
                "examples_all_repeats": logical * sampling_example["batch_size"],
                "hours_by_step_setting_one_run": {str(s): s * 64 * gradient_per_example_cap_scaled for s in steps},
                "all_repeats_gpu_hours": gradient_total},
            "preflight_allocation_envelope": {"gpu_count": 1, "cpus": 8, "host_memory_gib": 96,
                "minutes_per_case_limit": 30, "required_cases": 3, "optional_high_rank_cases": 1,
                "maximum_gpu_hours_required_cases": 1.5, "maximum_gpu_hours_including_optional_case": 2.0},
            "excluded_costs": ["generation of 1319 GSM8K answers up to 512 tokens", "validation/model selection",
                               "checkpoint I/O", "model loading", "queueing", "CPU similarity", "high-rank overhead"],
            "limitations": ["Neither extrapolation is a prediction or a lower/upper bound",
                            "Dynamic padding with microbatch=1 does not pad every example to 2048",
                            "Only length256/global8 Smoke was measured; checkpointing and long attention are nonlinear",
                            "Sampling batch/fixed steps are unconfirmed; their example must not enter formal configs",
                            "Sampling currently replays 128 targets; no persistent raw-gradient cache/resume",
                            "Use representative real-data 2048 training and sampling measurements before scheduling batches"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/chloe/report/math-only-resource-plan.json")
    args = parser.parse_args()
    report = estimate()
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
