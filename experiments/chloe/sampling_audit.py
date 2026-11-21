"""CPU-only sampling geometry and storage accounting from real weight headers."""
import argparse
import json
from pathlib import Path
import struct

from common import read_json, resolve, write_json


def backward_geometry(*, targets, steps, batch_size, micro_batch_size):
    if any(type(x) is not int or x < 1 for x in [targets, steps, batch_size, micro_batch_size]):
        raise ValueError("Targets, logical steps and batch sizes must be positive integers.")
    per_step = (batch_size + micro_batch_size - 1) // micro_batch_size
    return {"targets": targets, "logical_steps": steps, "logical_batch_size": batch_size,
            "micro_batch_size": micro_batch_size, "examples": steps * batch_size,
            "micro_batches_per_target": steps * per_step,
            "forward_calls": targets * steps * per_step,
            "autograd_grad_calls": targets * steps * per_step, "optimizer_updates": 0}


def weight_inventory(model_directory):
    """Read JSON/safetensors headers only; never load a model or weight tensors."""
    directory = Path(model_directory)
    config = json.loads((directory / "config.json").read_text())
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        with (directory / shard).open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            headers.update(json.loads(handle.read(size)))
    families = {}
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        rows = []
        for layer in range(config["num_hidden_layers"]):
            name = f"model.layers.{layer}.self_attn.{projection}"
            info = headers[name + ".weight"]
            if len(info["shape"]) != 2:
                raise ValueError("Expected two-dimensional attention weights.")
            count = info["shape"][0] * info["shape"][1]
            rows.append({"target": name, "shape": info["shape"], "weight_dtype": info["dtype"],
                         "elements": count, "cpu_fp32_gradient_bytes": count * 4})
        families[projection] = {"targets": rows, "cpu_fp32_family_cache_bytes": sum(r["cpu_fp32_gradient_bytes"] for r in rows)}
    return {"layer_count": config["num_hidden_layers"], "target_count": sum(len(v["targets"]) for v in families.values()),
            "families": families,
            "all_targets_cpu_fp32_bytes_if_retained": sum(v["cpu_fp32_family_cache_bytes"] for v in families.values()),
            "sequential_peak_family_cpu_fp32_bytes": max(v["cpu_fp32_family_cache_bytes"] for v in families.values()),
            "largest_single_fp32_gradient_bytes": max(r["cpu_fp32_gradient_bytes"] for f in families.values() for r in f["targets"]),
            "largest_single_fp64_moment_copy_bytes": max(r["elements"] * 8 for f in families.values() for r in f["targets"]),
            "largest_alternative_metric_family_fp64_copies_bytes": max(sum(r["elements"] * 8 for r in f["targets"]) for f in families.values())}


def make_report():
    cfg = read_json("experiments/chloe/configs/smoke-lora-r8.json")
    inventory = weight_inventory(resolve(cfg["model"]["path"]))
    settings = read_json("references/ablation-setup.json")["gradient_sampling_steps"]
    return {"run_type": "cpu_sampling_resource_audit", "eligible_for_formal_tables": False,
            "gpu_operations": False, "base_weights_loaded": False, "model": cfg["model"],
            "inventory": inventory, "formula": "targets * steps * ceil(logical_batch_size / micro_batch_size)",
            "count_examples_not_formal_batch_confirmation": {
                "B1_micro1": [backward_geometry(targets=inventory["target_count"], steps=s, batch_size=1, micro_batch_size=1) for s in settings],
                "B64_micro1": [backward_geometry(targets=inventory["target_count"], steps=s, batch_size=64, micro_batch_size=1) for s in settings]},
            "cpu_input_tensor_upper_bound_example": {
                "assumptions": "illustrative S512 B64 L2048, input_ids/mask/labels all int64 on CPU",
                "bytes": 512 * 64 * 2048 * 3 * 8},
            "total_sampling_time_seconds": None,
            "timing_status": "one q_proj observation is not a 128-target or multi-step throughput calibration",
            "memory_limits": ["Family cache is retained until similarity is computed, not freed after each target",
                              "FP32 transfer/accumulation, FP64 moments, finite-check temporaries and allocator overhead add memory",
                              "Alternative metrics retain a full family's FP64 copies in addition to original FP32 gradients",
                              "CPU batches, Python token lists, source JSON and model loading require additional RAM",
                              "Reserved memory may grow from allocator caching; observe allocated bytes and tensor liveness separately"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/chloe/report/sampling-resource-audit.json")
    args = parser.parse_args()
    result = make_report()
    write_json(args.output, result)
    print(json.dumps({"targets": result["inventory"]["target_count"],
                      "peak_family_bytes": result["inventory"]["sequential_peak_family_cpu_fp32_bytes"],
                      "all_target_bytes_if_retained": result["inventory"]["all_targets_cpu_fp32_bytes_if_retained"],
                      "time_prediction": None}, indent=2))


if __name__ == "__main__":
    main()
