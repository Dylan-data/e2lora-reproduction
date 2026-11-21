# Full implementation guide: NLG Table 2, Appendix K Table 14, Appendix L Table 15

This guide documents the experiment runners added under `experiments/s/`. It separates paper-stated settings from reproduction choices that the paper does not disclose. The runner reuses the repository's pinned model/data lists, pretrained loader, dataset loader, gradient/RMI/greedy grouping implementation, and the core E²LoRA adapter implementation.

## 1. Scope and implementation map

| Paper scope | Models | Tasks / evaluation | Methods | Entrypoint |
|---|---|---|---|---|
| NLG, Table 2 | Llama 3.1 8B Base | MetaMathQA → GSM8K accuracy; Code-Feedback → HumanEval pass@1 | 12 methods | `suite_train.py --suite table2` |
| Appendix K, Table 14 | Llama 3.1 8B | ARC-Challenge, ARC-Easy, BoolQ, OpenBookQA, PIQA, Social IQa | LoRA, E²LoRA | `suite_train.py --suite table14` |
| Appendix L, Table 15 | Llama-2-13B, Qwen3-8B, GPT-OSS-20B | Math-benchmark accuracy | LoRA, E²LoRA | `suite_train.py --suite table15` |

`run_matrix.py` builds a **126-run command plan** for three independent runs per setting: 12 Table 2 methods × 2 tasks × 3 seeds = 72; 6 Table 14 tasks × 2 methods × 3 seeds = 36; 3 Table 15 models × 2 methods × 3 seeds = 18. The default is plan-only; it does not start training unless `--execute` is passed.

## 2. Table 2: all twelve methods

All Table 2 adapters target the attention Q/K/V/O projection linears. Target discovery checks model block depth and projection suffixes at runtime; the script fails if a model does not expose all four projection families with compatible layer counts.

| Config ID | Paper method | Implementation | Rank / special setup | Learning rate used by default |
|---|---|---|---|---:|
| `fft_attn_only` | FFT (Attn-only) | Unfreeze only Q/K/V/O attention projection weights and biases | Full attention-projection fine-tuning | 5e-5 (team default; Table C.4 lists base LR for other named baselines) |
| `dora_r4` | DoRA (rank 4) | PEFT DoRA | r=4 | 5e-5 (team default; DoRA LR not separately specified) |
| `dora_r8` | DoRA | PEFT DoRA | r=8 | 5e-5 (team default; DoRA LR not separately specified) |
| `adalora_r4` | AdaLoRA (rank 4) | PEFT AdaLoRA | target r=4; PEFT schedule parameters are team defaults | 5e-4 (Appendix C.4) |
| `adalora_r8` | AdaLoRA | PEFT AdaLoRA | target r=8; PEFT schedule parameters are team defaults | 5e-4 (Appendix C.4) |
| `lora_plus` | LoRA+ | PEFT LoRA with separate A/B optimizer groups | B/A LR ratio 16 is a team default | 5e-5 (team default; paper does not state a separate LoRA+ ratio/LR) |
| `lora_fa` | LoRA-FA | PEFT LoRA with A frozen and B trainable | Base r=8; only B is trainable | 5e-4 (Appendix C.4) |
| `vera` | VeRA | PEFT VeRA | r=256 and projection defaults are team choices | 5e-3 (Appendix C.4) |
| `lora` | LoRA | Custom adapter: independent trainable A/B per projection layer | r=8 | 5e-5 (Appendix C.4) |
| `e2lora` | E²LoRA | Algorithm 1–3: interval-shared A/B with proxy-entropy rank allocation | base rank 8; rank-matched inference guided by Appendix J.1 | 5e-5 (Appendix C.4) |
| `sharelora` | ShareLoRA | One global A per projection family; layer-specific B | r=8 | 5e-5 (Appendix C.4) |
| `e2sharelora` | E²ShareLoRA | Global A per projection plus interval-shared B with adaptive rank slicing | base rank 8; rank-matched inference guided by Appendix J.1 | 5e-5 (Appendix C.4) |

**Parameter/rank caveat.** Algorithm 1 describes a rank budget proportional to `N × r_vanilla`; the main experiment tables show E²LoRA parameter counts around half the LoRA baseline. The paper does not state a per-method NLG base rank in Table 2. Appendix J.1 provides a stronger rank-matching clue: it compares LoRA rank 4 with E²LoRA base rank 8, and LoRA rank 8 with E²LoRA base rank 16 at matched parameter budgets. The runner therefore defaults to baseline `--rank 8` and E² `--e2-base-rank 8` as a rank-matched inference for the target tables, not as an explicitly printed Table 2/14/15 setting. It never forces the paper's target parameter number: each run records actual trainable, frozen-adapter and total parameter counts. Verify these counts against the target table before claiming a matched implementation.

### Table 2 reference values from the paper

| Method | GSM8K params | GSM8K accuracy | HumanEval params | HumanEval pass@1 |
|---|---:|---:|---:|---:|
| FFT (Attn-only) | 1.34B | 73.69 | 1.34B | 51.63 |
| DoRA (rank 4) | 3.74M | 67.40 | 3.74M | 43.90 |
| DoRA | 6.82M | 69.17 | 6.82M | 43.70 |
| AdaLoRA (rank 4) | 3.41M | 56.56 | 3.41M | 35.37 |
| AdaLoRA | 6.82M | 70.63 | 6.82M | 41.46 |
| LoRA+ | 6.82M | 71.29 | 6.82M | 44.51 |
| LoRA-FA | 3.41M | 68.66 | 3.41M | 42.07 |
| VeRA | 0.33M | 67.63 | 0.33M | 42.48 |
| LoRA | 6.82M | 70.21 | 6.82M | 42.68 |
| E²LoRA | 3.61M | 70.51 | 3.66M | 44.31 |
| ShareLoRA | 2.75M | 70.51 | 2.75M | 43.69 |
| E²ShareLoRA | 1.59M | 70.26 | 1.59M | 44.71 |

## 3. Appendix K: Table 14 commonsense reasoning

The six subtasks are loaded from the repositories already listed in `src/e2lora/dataset/LIST.md`:

| Task key | Dataset | Configured source | Metric |
|---|---|---|---|
| `arc_challenge` | ARC-Challenge | `allenai/ai2_arc`, subset `ARC-Challenge`, pinned revision | accuracy |
| `arc_easy` | ARC-Easy | `allenai/ai2_arc`, subset `ARC-Easy`, pinned revision | accuracy |
| `boolq` | BoolQ | `google/boolq`, pinned revision | accuracy |
| `openbookqa` | OpenBookQA | `allenai/openbookqa`, subset `main`, pinned revision | accuracy |
| `piqa` | PIQA | `ybisk/piqa`, current repo list has no commit revision | accuracy |
| `siqa` | Social IQa | `allenai/social_i_qa`, current repo list has no commit revision | accuracy |

Each row is normalized to an instruction-style multiple-choice prompt and an answer-letter target. The public `validation` split is reserved for final evaluation. The runner makes a deterministic 10% holdout from the original `train` split (default `--selection-validation-ratio 0.1`, seed `--validation-seed 43`) for epoch/checkpoint selection, to avoid selecting a checkpoint on the same examples later reported as the final score. This prompt and split protocol is a **team implementation choice**: Appendix K does not publish exact prompts or exact split/epoch settings. PIQA and SIQA require `--allow-unpinned-datasets`; the downloaded data cache and row manifest must be retained.

### Table 14 paper reference values

| Method | #Params | Avg. | ARC-C | ARC-E | BoolQ | OBQA | PIQA | SIQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 6.82M | 82.18 | 82.50 | 93.09 | 64.95 | 86.40 | 88.46 | 77.67 |
| E²LoRA | 3.26M | 82.27 | 81.14 | 91.45 | 70.06 | 84.20 | 86.07 | 80.72 |

## 4. Appendix L: Table 15 larger/diverse models

The runner supports the three models and revisions in `src/e2lora/model/LIST.md`:

| Table 15 model | Key | Model repository | Team-pinned revision | Architecture |
|---|---|---|---|---|
| Llama-2-13B | `llama2_13b` | `meta-llama/Llama-2-13b-hf` | `5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` | Dense |
| Qwen3-8B | `qwen3_8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | Dense |
| GPT-OSS-20B | `gpt_oss_20b` | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | MoE |

Appendix L says “math benchmarks” and accuracy but does not independently restate its training dataset, sample count, test split, prompts, epochs, or per-model hyperparameters. The runner explicitly reuses the Table 2 math protocol (100K MetaMathQA sampled with fixed data seed 42; GSM8K test for final accuracy) as a **documented extension assumption**, not as a detail independently confirmed by Appendix L. It measures the actual attention targets and adapter parameter count on each model. GPT-OSS target discovery and all three model loads must pass preflight before a full run.

### Table 15 paper reference values

| Model (architecture) | Method | #Params | Accuracy |
|---|---|---:|---:|
| Llama-2-13B (Dense) | LoRA | 13.11M | 52.94 |
| Llama-2-13B (Dense) | E²LoRA | 6.55M | 53.82 |
| Qwen3-8B (Dense) | LoRA | 7.67M | 82.49 |
| Qwen3-8B (Dense) | E²LoRA | 4.06M | 82.49 |
| GPT-OSS-20B (MoE) | LoRA | 3.98M | 82.03 |
| GPT-OSS-20B (MoE) | E²LoRA | 1.75M | 83.17 |

## 5. Paper settings versus team defaults

| Setting | Paper text | Runner default / handling |
|---|---|---|
| Llama model for Tables 2/14 | Llama 3.1 8B Base | exact team-pinned revision from repo model list |
| Target modules | attention linear layers | discover Q/K/V/O projections and log exact layer paths |
| Optimizer | AdamW, β=(0.9,0.999), weight decay 5e-4 | same |
| Global batch | 64 | default per-device batch 1 × grad accumulation 64 (single process) |
| Precision | BF16 | BF16 only if CUDA runtime supports it; otherwise FP32 for smoke/testing |
| Scheduler | cosine; C.4 says 3% warmup | `--warmup-ratio 0.03` |
| Table 9 warmup | Table 9 lists 0.3, conflicting with C.4's 3% | preserve both in run metadata; use 0.03 by default; change only with an explicit decision |
| Learning rate | 5e-5 LoRA/ShareLoRA/E²LoRA; 5e-4 AdaLoRA/LoRA-FA; 5e-3 VeRA | configured method defaults; DoRA/LoRA+ LR remains a team default because no separate rate is specified |
| LR decay | Table 9 gives 0.1 | logged but not silently interpreted as cosine min-LR ratio |
| Number of runs | results averaged over three runs | seeds 42/43/44 are selected by team; paper does not publish seed values |
| Epochs, max sequence length, max generation tokens | not specified for the target NLG suite | required CLI arguments; measure sequence length first and record choices |
| E² base rank | no per-row NLG rank directly specified | default 8 is an inference guided by Appendix J.1 (LoRA r=4 vs E² base r=8); measured count always wins over reference count |
| LoRA+ B/A LR multiplier | not specified | default 16 is an engineering choice |
| AdaLoRA schedule knobs / VeRA rank | not specified | PEFT defaults exposed in method metadata; verify actual counts |

## 6. Install and preflight

From the repository root, use a GPU environment with a compatible PyTorch/CUDA build:

```bash
pip install -e '.[nlg]'
# Optional package shortcut for the official evaluator:
pip install -e '.[humaneval]'
# The official upstream install procedure is also supported:
git clone https://github.com/openai/human-eval
pip install -e ./human-eval
```

Llama repositories are gated: log in to Hugging Face and obtain access to the model repositories through their official model pages. The team-pinned revisions in the model list must be used. Before any long run, verify CUDA/BF16, model access, model target discovery and dataset access.

### Token-length audit (must precede the run)

Math on Llama 3.1 8B:

```bash
python experiments/s/inspect_token_lengths.py \
  --task math --model-key llama31_8b \
  --output outputs/text-generation/preflight/metamath_llama31.json
```

Code uses the verified, prefiltered local code-only JSONL (length audit does not decide the code-only provenance):

```bash
python experiments/s/inspect_token_lengths.py \
  --task code --model-key llama31_8b --code-jsonl /path/to/code_only_102k.jsonl \
  --output outputs/text-generation/preflight/code_llama31.json
```

Commonsense example (PIQA/SIQA also require `--allow-unpinned-datasets`):

```bash
python experiments/s/inspect_token_lengths.py \
  --task arc_challenge --model-key llama31_8b \
  --output outputs/text-generation/preflight/arc_challenge.json
```

Choose the smallest `max_seq_length` that preserves every answer plus EOS; prompt truncation is from the left to retain the end of the instruction/question/options. The audit reports answer-too-long counts and prompt truncation counts for candidate lengths. Do not blindly use 2048 or 4096 without inspecting these reports.

## 7. Run examples

The examples use `epochs=1`, `max_seq_length=2048`, and generation caps only as **illustrative team choices**. Replace sequence length and generation caps using the token-length audit and validate generation completeness. `epochs` is not stated by the paper and must be finalized by the experiment owner.

### Table 2: math E²LoRA

```bash
python experiments/s/suite_train.py \
  --suite table2 --task math --method e2lora --model-key llama31_8b \
  --seed 42 --data-seed 42 --validation-seed 43 \
  --epochs 1 --max-seq-length 2048 --max-new-tokens 512 \
  --per-device-batch-size 1 --gradient-accumulation-steps 64 --gradient-examples 64 \
  --output-dir outputs/table2/math/e2lora/seed_42 --do-final-eval
```

### Table 2: run all twelve methods and both tasks

A full table requires a verified local Code-Feedback JSONL with at least 102,000 usable code-only examples (100K training + 2K disjoint held-out validation). The repository's pinned public snapshot is listed at approximately 66.4K rows, so it does not meet the paper's 100K requirement. Do not duplicate or pad it.

First generate and inspect the 126-command plan:

```bash
python experiments/s/run_matrix.py \
  --epochs 1 --max-seq-length 2048 \
  --max-new-tokens-math 512 --max-new-tokens-code 512 --max-new-tokens-commonsense 8 \
  --train-jsonl /path/to/code_only_102k.jsonl --code-only-source-confirmed \
  --allow-unpinned-datasets \
  --output-root outputs/e2lora-all-suites
```

The first command is plan-only and writes `outputs/e2lora-all-suites/run_all_commands.sh`. Review the generated commands. After data/model/GPU preflight, rerun the same `run_matrix.py` command with `--execute` added (do not add `--execute` to `suite_train.py`). The 126 runs are sequential and can take substantial accelerator time. To score HumanEval inside an isolated sandbox, also pass `--score-humaneval-in-sandbox`; otherwise the run only writes `samples.jsonl` and marks pass@1 as not computed.

### Table 14: one commonsense subtask

```bash
python experiments/s/suite_train.py \
  --suite table14 --task arc_challenge --method e2lora --model-key llama31_8b \
  --seed 42 --data-seed 42 --validation-seed 43 \
  --epochs 1 --max-seq-length 2048 --max-new-tokens 8 \
  --per-device-batch-size 1 --gradient-accumulation-steps 64 --gradient-examples 64 \
  --output-dir outputs/table14/arc_challenge/e2lora/seed_42 --do-final-eval
```

Run both `--method lora` and `--method e2lora` for each of `arc_challenge`, `arc_easy`, `boolq`, `openbookqa`, `piqa`, `siqa`. For the last two, add `--allow-unpinned-datasets` after reviewing that limitation.

### Table 15: one larger/diverse model

```bash
python experiments/s/suite_train.py \
  --suite table15 --task math --method e2lora --model-key qwen3_8b \
  --seed 42 --data-seed 42 --validation-seed 43 \
  --epochs 1 --max-seq-length 2048 --max-new-tokens 512 \
  --per-device-batch-size 1 --gradient-accumulation-steps 64 --gradient-examples 64 \
  --output-dir outputs/table15/qwen3_8b/e2lora/seed_42 --do-final-eval
```

Change `--model-key` to `llama2_13b` or `gpt_oss_20b` and repeat for LoRA/E²LoRA and seeds 42/43/44.

### HumanEval official pass@1 scoring

HumanEval runs generated programs. The official project explicitly warns that generated code can be destructive and strongly recommends a robust sandbox. `reliability_guard` is not itself a security sandbox. Run the scorer only in a hardened, network-isolated container/VM with no credentials, sensitive mounts or host access.

Important upstream behavior: in the official `human_eval/execution.py`, the actual `exec(check_program, exec_globals)` execution statement is deliberately commented out by default. After reading the official security disclaimer, manually enable that line **only inside the isolated sandbox** before running the official evaluator. This implementation will not modify the installed upstream package or uncomment code automatically. Merely passing `--confirm-isolated-sandbox` acknowledges the safety boundary; it does not enable the upstream execution line.

```bash
python experiments/s/evaluate_humaneval.py \
  --samples outputs/table2/code/lora/seed_42/final-evaluation/samples.jsonl \
  --confirm-isolated-sandbox
```

Alternatively add both `--run-human-eval-harness --confirm-isolated-sandbox` to the training command while already inside that sandbox. The runner does not execute generated code by default. Once the upstream execution line is manually enabled in the isolated sandbox, the official harness returns pass@1 and writes per-sample pass/fail records. Without that manual upstream change, generated completions are not executed and the score is not valid.

### Summarize all completed runs

```bash
python experiments/s/summarize_results.py \
  --output-root outputs/e2lora-all-suites \
  --output-dir outputs/e2lora-all-suites/summary
```

Summary scores are converted to percentages (0–100) to match paper tables; raw `result.json` stores accuracy/pass@1 as fractions (0–1). The summary includes the paper reference score and delta where available. Missing result rows are left empty; diagnostic subsets are not valid Table results.

## 8. Current limitations that must be resolved for a paper-faithful reproduction

1. **No local real run has been executed here.** Current authoring environment previously reported CPU-only PyTorch, no CUDA, and no installed Transformers/Datasets/PEFT. Synthetic CPU tests verify a limited part of custom adapter behavior only.
2. **The training entrypoint is single-process.** The paper's NLG Appendix C.6 uses 8×A800, DeepSpeed ZeRO2 and FlashAttention-2. This implementation supports a single process, gradient accumulation and optional `device_map=auto`, but does not implement the paper's ZeRO2/distributed training path. It is an executable reproduction scaffold, not hardware-identical to the paper setup.
3. **Code-Feedback 100K code-only source is missing.** The pinned repo snapshot (~66.4K) is insufficient. The code runner intentionally refuses to launch without a user-supplied, provenance-verified 102K+ code-only JSONL.
4. **HumanEval revision is not pinned in the repository's list.** The script requires explicit `--allow-unpinned-datasets` and records cache row hashes. The exact source snapshot should be frozen before final comparison.
5. **PIQA and Social IQa revisions are not pinned.** The same opt-in and manifest policy applies.
6. **Appendix K does not specify prompts/splits/training schedule.** The multiple-choice prompt, 10% train-only selection holdout and public validation final evaluation are transparent team choices.
7. **Appendix L does not restate its exact math data protocol.** Reusing Table 2's MetaMathQA/GSM8K is a clearly labeled extension assumption.
8. **Several baseline knobs are not paper-specified.** LoRA+ B/A ratio, AdaLoRA schedule values, VeRA rank/projection configuration, DoRA/LoRA+ learning rates, max lengths, generation limits and epochs are logged as team choices or unresolved values.
9. **The Table 9 `Warmup=0.3` conflicts with C.4's 3% warmup.** Runner default uses 0.03 and records both values. `LR Decay=0.1` is logged but not reinterpreted as a cosine minimum-LR ratio.
10. **E² base rank is inferred.** The default 8 is a rank-matched inference guided by Appendix J.1 (LoRA r=4 vs E² base r=8); the exact target-table rank is still not stated directly. Actual parameter counts must be compared with the paper values.
11. **Model access and architecture need smoke tests.** Llama2/Llama3 require HF access. GPT-OSS MoE target naming, module compatibility, adapter parameter count and actual train path must be verified using the exact pinned checkpoint.

## 9. Output contract

Each run directory contains `run_config.json`, `sample_manifest.json`, `parameter_counts.json`, `training_history.json`, `checkpoint_last.pt`, `checkpoint_best.pt`, and `result.json`; E² runs additionally save `adapter_plan.json`. Final evaluation creates a separate `final-evaluation/` directory with a data manifest, generations, per-example outputs and metric result. All result files record the paper/extension/team protocol distinctions, seeds, model revision, source revision/checksum where available, actual trainable parameter counts and hardware metadata.
