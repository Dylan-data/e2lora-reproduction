#!/usr/bin/env python3
"""Generate/execute the 126-run Table 2, Table 14 and Table 15 matrix.

The default is --plan-only. No run is launched unless --execute is supplied.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS_TABLE2 = ["fft_attn_only", "dora_r4", "dora_r8", "adalora_r4", "adalora_r8",
                  "lora_plus", "lora_fa", "vera", "lora", "e2lora", "sharelora", "e2sharelora"]
COMMONSENSE_TASKS = ["arc_challenge", "arc_easy", "boolq", "openbookqa", "piqa", "siqa"]
LARGE_MODELS = ["llama2_13b", "qwen3_8b", "gpt_oss_20b"]
SEEDS = [42, 43, 44]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path("outputs/e2lora-all-suites"))
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--max-seq-length", type=int, required=True)
    p.add_argument("--max-new-tokens-math", type=int, required=True)
    p.add_argument("--max-new-tokens-code", type=int, required=True)
    p.add_argument("--max-new-tokens-commonsense", type=int, required=True)
    p.add_argument("--train-jsonl", type=Path, default=None)
    p.add_argument("--code-only-source-confirmed", action="store_true")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=64)
    p.add_argument("--gradient-examples", type=int, default=64)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="enable model gradient checkpointing to reduce activation memory")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--validation-seed", type=int, default=43)
    p.add_argument("--e2-base-rank", type=int, default=8)
    p.add_argument("--allow-unpinned-datasets", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", choices=("auto",), default=None)
    p.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default=None)
    p.add_argument("--score-humaneval-in-sandbox", action="store_true",
                   help="run official HumanEval scorer; only use on an isolated sandbox host")
    p.add_argument("--plan-only", action="store_true", default=True)
    p.add_argument("--execute", action="store_true", help="actually launch the full sequential matrix")
    p.add_argument("--limit-train", type=int, default=None, help="diagnostic pilot only; scores cannot be reported as Table values")
    args = p.parse_args()

    commands = []
    common = [sys.executable, "experiments/s/suite_train.py",
              "--epochs", str(args.epochs), "--max-seq-length", str(args.max_seq_length),
              "--per-device-batch-size", str(args.per_device_batch_size),
              "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
              "--gradient-examples", str(args.gradient_examples), "--rank", str(args.rank),
              "--e2-base-rank", str(args.e2_base_rank), "--data-seed", str(args.data_seed),
              "--validation-seed", str(args.validation_seed), "--device", args.device]
    if args.device_map:
        common += ["--device-map", args.device_map]
    if args.gradient_checkpointing:
        common += ["--gradient-checkpointing"]
    if args.attn_implementation:
        common += ["--attn-implementation", args.attn_implementation]
    if args.allow_unpinned_datasets:
        common += ["--allow-unpinned-datasets"]
    if args.limit_train is not None:
        common += ["--limit-train", str(args.limit_train)]

    for task in ("math", "code"):
        max_new = args.max_new_tokens_math if task == "math" else args.max_new_tokens_code
        for method in METHODS_TABLE2:
            for seed in SEEDS:
                cmd = common + ["--suite", "table2", "--task", task, "--model-key", "llama31_8b",
                                "--method", method, "--seed", str(seed), "--max-new-tokens", str(max_new),
                                "--output-dir", str(args.output_root / "table2" / task / method / f"seed_{seed}")]
                cmd += ["--do-final-eval"]
                if args.score_humaneval_in_sandbox and task == "code":
                    cmd += ["--run-human-eval-harness", "--confirm-isolated-sandbox"]
                if args.train_jsonl and task == "code":
                    cmd += ["--train-jsonl", str(args.train_jsonl)]
                if args.code_only_source_confirmed and task == "code":
                    cmd += ["--code-only-source-confirmed"]
                commands.append(cmd)

    for task in COMMONSENSE_TASKS:
        for method in ("lora", "e2lora"):
            for seed in SEEDS:
                cmd = common + ["--suite", "table14", "--task", task, "--model-key", "llama31_8b",
                                "--method", method, "--seed", str(seed),
                                "--max-new-tokens", str(args.max_new_tokens_commonsense),
                                "--output-dir", str(args.output_root / "table14" / task / method / f"seed_{seed}"),
                                "--do-final-eval"]
                commands.append(cmd)

    for model_key in LARGE_MODELS:
        for method in ("lora", "e2lora"):
            for seed in SEEDS:
                cmd = common + ["--suite", "table15", "--task", "math", "--model-key", model_key,
                                "--method", method, "--seed", str(seed),
                                "--max-new-tokens", str(args.max_new_tokens_math),
                                "--output-dir", str(args.output_root / "table15" / model_key / method / f"seed_{seed}"),
                                "--do-final-eval"]
                commands.append(cmd)

    script_path = args.output_root / "run_all_commands.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "# Generated command plan; inspect before execution."]
    lines += [shlex.join(cmd) for cmd in commands]
    script_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    print(f"Prepared {len(commands)} commands")
    print(f"Command plan: {script_path}")
    print("Matrix: Table 2 = 72 runs; Table 14 = 36 runs; Table 15 = 18 runs.")
    if args.limit_train is not None:
        print("WARNING: --limit-train creates diagnostic subset runs; scores are not Table reproduction results.")
    if args.execute:
        if args.limit_train is not None:
            raise SystemExit("Refusing to execute a diagnostic subset matrix as a full paper run.")
        if args.train_jsonl is None or not args.code_only_source_confirmed:
            raise SystemExit("Full Table 2 matrix requires --train-jsonl and --code-only-source-confirmed for the 100K code-only Code-Feedback subset.")
        if not args.allow_unpinned_datasets:
            raise SystemExit("Full matrix includes HumanEval, PIQA and SIQA sources that are not pinned in the repository; pass --allow-unpinned-datasets only after reviewing this caveat.")
        for index, cmd in enumerate(commands, 1):
            print(f"[{index}/{len(commands)}] {shlex.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        print("Plan only. Add --execute after data/model/GPU preflight to launch.")


if __name__ == "__main__":
    main()
