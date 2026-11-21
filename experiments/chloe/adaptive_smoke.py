"""Explicit, all-trainable group A/B policy for the first E²LoRA mechanism smoke."""
from copy import deepcopy
import random

from common import (attach_trainable_plan, baseline_plan, digest, file_sha256,
                    output_path, parameter_report, read_json, resolve, validate_config)


def validate_adaptive_config(config, *, require_data=False):
    if config["task"] != "math" or config["run_type"] != "smoke":
        raise ValueError("E2LoRA is enabled only for the pending Math mechanism Smoke, not formal training.")
    adaptive = config["adaptive"]
    if not adaptive["lss"] or not adaptive["hra"] or adaptive["similarity"] != "rmi" or adaptive["grouping"] != "greedy":
        raise ValueError("First E2LoRA Smoke requires RMI + Greedy LSS + HRA.")
    if adaptive["rank_policy"] != "paper":
        raise ValueError("Use literal paper ranks; no silent entropy shifting or budget correction.")
    if adaptive["partial_rank_sharing"]["enabled"] or adaptive["trainable_components"] != "all_group_A_and_B":
        raise ValueError("Partial-rank/freezing alignment is pending; no 139-rank assumption is allowed.")
    if config["adapter"]["sharing"] != "A_and_B_within_group" or config["adapter"]["rank_allocation"] != "entropy":
        raise ValueError("E2LoRA must share A/B by contiguous projection group and use entropy ranks.")
    sampling = adaptive["sampling"]
    if type(sampling["steps"]) is not int or not 1 <= sampling["steps"] <= 8 or sampling["batch_size"] != 1:
        raise ValueError("Low-cost first Smoke samples 1-8 fixed logical batches of one example.")
    if sampling["target_block_size"] != 1 or sampling["seed"] != config["seed"] or sampling["order"] != "baseline_epoch0_order_prefix":
        raise ValueError("Sampling must replay the fixed training-order prefix one target at a time.")
    if config["data"]["train_limit"] != 160 or sampling["steps"] > config["data"]["train_limit"]:
        raise ValueError("Reuse exactly the successful baseline's 160-example Smoke pool.")
    directory = output_path(config["output_dir"])
    protected = output_path("outputs/chloe/smoke/math-lora-r8-seed42")
    if directory == protected or directory.is_relative_to(protected) or protected.is_relative_to(directory):
        raise ValueError("E2LoRA artifacts must not overwrite or contain the passed LoRA baseline directory.")
    if output_path(adaptive["prepared_plan"]).parent != directory or output_path(adaptive["prepared_plan"]).name != "prepared-plan.json":
        raise ValueError("Keep the prepared plan in this independent E2LoRA run directory.")
    # Reuse baseline checks for training/data/evaluation settings; this is validation
    # only. The original E2LoRA mechanism configuration is retained unchanged.
    ordinary = deepcopy(config)
    ordinary["method"] = "lora"
    ordinary["adapter"].update(sharing="none", rank_allocation="uniform")
    validate_config(ordinary, require_data=require_data)


def sampling_batches(examples, config, pad_token_id):
    from common import collate
    order = list(range(len(examples)))
    random.Random(config["adaptive"]["sampling"]["seed"]).shuffle(order)
    selected = order[:config["adaptive"]["sampling"]["steps"]]
    return [collate([examples[i]], pad_token_id) for i in selected], selected


def prepared_artifact(config, *, source_fingerprint=None, expected_sha256=None):
    path = resolve(config["adaptive"]["prepared_plan"])
    if not path.is_file():
        raise FileNotFoundError("E2LoRA Stage A has not produced prepared-plan.json; complete and review preparation first.")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Prepared E2LoRA plan changed since training started.")
    artifact = read_json(path)
    if artifact["status"] != "prepared" or artifact["config_sha256"] != digest(config):
        raise ValueError("E2LoRA plan is not prepared for this unchanged Smoke config.")
    if source_fingerprint is not None and artifact["source_fingerprint"] != source_fingerprint:
        raise ValueError("E2LoRA sampling source/data/tokenizer fingerprint changed.")
    if artifact["optimizer_updates"] != 0 or artifact["partial_rank_sharing_enabled"]:
        raise ValueError("Preparation must perform zero updates and use no unconfirmed partial freezing.")
    return artifact


def attach_prepared(model, config, *, source_fingerprint=None, expected_sha256=None):
    artifact = prepared_artifact(config, source_fingerprint=source_fingerprint, expected_sha256=expected_sha256)
    plan = artifact["plan"]
    if {g['family'] for g in plan} != set(config['adapter']['target_modules']):
        raise ValueError("Prepared plan must contain exactly the four attention projection families.")
    independent = baseline_plan(model, config)
    for family in config["adapter"]["target_modules"]:
        expected = [g["layers"][0] for g in independent if g["family"] == family]
        actual = [name for g in plan if g["family"] == family for name in g["layers"]]
        if actual != expected:
            raise ValueError(f"Prepared groups must cover {family} once, in contiguous depth order.")
        ranks = [g["rank"] for g in plan if g["family"] == family]
        if ranks != artifact["diagnostics"][family]["ranks"]:
            raise ValueError("Prepared ranks and diagnostics differ.")
    attach_trainable_plan(model, plan, alpha=config["adapter"]["alpha"])
    actual = parameter_report(model, plan)
    for key in ["adapter_parameters", "trainable_adapter_parameters", "total_trainable_parameters"]:
        if actual[key] != artifact["parameters"][key]:
            raise ValueError(f"Actual E2LoRA parameter count changed: {key}")
    return plan, artifact
