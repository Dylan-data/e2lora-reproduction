"""Instantiate the actual model architecture and count unique installed adapters."""
import argparse
import json
import struct

from common import attach_baseline, offline, output_path, parameter_report, provenance, read_json, resolve, validate_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/chloe/configs/smoke-lora-r8.json")
    parser.add_argument("--output", default="outputs/chloe/budget/lora-r8.json")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    if config["method"] != "lora":
        raise ValueError("This independent-rank budget check is LoRA-only; E2LoRA must count its actual installed Stage A groups.")
    offline()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    architecture = AutoConfig.from_pretrained(resolve(config["model"]["path"]), local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(architecture, dtype=torch.bfloat16, attn_implementation="sdpa")
        plan = attach_baseline(model, config)
    weight_dir = resolve(config["model"]["path"])
    index = read_json(weight_dir / "model.safetensors.index.json")
    headers = {}
    for shard in set(index["weight_map"].values()):
        with (weight_dir / shard).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            headers.update(json.loads(stream.read(size)))
    for group in plan:
        name = group["layers"][0]
        if list(model.get_submodule(name).base.weight.shape) != headers[name + ".weight"]["shape"]:
            raise ValueError(f"Real weight header and instantiated module shape differ: {name}")
    result = {"run_type": "parameter_check", "task": "math", "metric": None,
              "measurement": "unique installed Parameters on the local model architecture, meta device; no weight tensors loaded",
              "real_weight_header_shapes_verified": len(plan),
              "provenance": provenance(config), "parameters": parameter_report(model, plan)}
    write_json(output_path(args.output), result)
    print(f"{args.output}: {result['parameters']['total_trainable_parameters']} trainable; no training performed")


if __name__ == "__main__":
    main()
