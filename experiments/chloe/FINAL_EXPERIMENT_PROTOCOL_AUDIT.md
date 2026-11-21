# Protocol audit and resolution record

I wrote the initial protocol audit before the full mathematical baselines. This page records what needed verification and how I now distinguish resolved baseline settings from unrun ablation choices. The executable source of settings is `configs/suite-math.json`, with commands in [README.md](README.md).

## What I checked

I required actual model/tokenizer identity, source and ordered training/validation manifests, exact prompts and loss masks, optimizer/batch/schedule settings, adapter initialization and freezing, checkpoint recovery, complete GSM 8 K content/order and a reproducible scorer. Knowing only seed 42, one epoch, length 2048 and generation 512 did not establish all of those details.

The completed LoRA runs verify a fixed 100 K training selection, disjoint 2 K validation selection, normalized ordered training hash, and all 1319 ordered GSM 8 K evaluation hashes. They use the recorded response-only token-weighted protocol, effective batch 64 and 1,563 updates. Their test results are 69.07% for r4 and 68.08% for r8. I retain their original source/configuration identities.

## Parameter interpretation

| Configuration | Count from the earlier architecture audit | Interpretation |
|---|---:|---|
| LoRA r4 | 3,407,872 | Independent adapters |
| LoRA r8 | 6,815,744 | Independent adapters |
| E2LoRA r8, cached all-trainable plan | 6,810,624 |24 shared groups, all A/B components trained |
| E2LoRA r16, same cached statistics | 13,631,488 | Static recalculation, not a trained result |
| Team partial E2LoRA r8 | 3,700,736 | Explicit 139-active budget per projection |

The paper displays 3.41 M/6.82 M for LoRA r4/r8 and 3.61 M/7.22 M for E2LoRA r8/r16. Ordinary rounding explains the LoRA display values but cannot explain the approximately halved shared-adapter counts. I keep the team's partial policy and exact budget correction separate from the paper's published round-based formula.

## Implemented choices and remaining experiments

I explicitly state Table 4's base rank, the main sampling count, similarity conventions, the Figure 5 fixed values and the rank-budget extension in the suite JSON. These are documented team choices where the publication does not determine an executable setting. I do not silently fill shared reference nulls or tune a freeze ratio to a paper parameter value.

The historical plan had 24 rows and 72 three-repeat slots. The current compiler combines matching settings across tables/curves into 19 unique seed 42 runs. Additional seeds are explicit configuration entries; unfinished repeats do not enter a mean/std calculation. I have not completed the remaining Table 4, E2LoRA r16 or Figure 5 full runs.

## Capacity evidence

Jobs 55643/55644/55648 checked two synthetic length 2048 LoRA updates, one original-weight gradient target, and four targets respectively. Job 55648 recorded 8 backward calls, zero updates, phase 7.339 seconds, peak allocated 28,383,159,808 bytes and peak reserved 29,183,967,232 bytes. Allocated memory after each target returned to 16,077,562,368 bytes.

These observations do not validate an arbitrary 128-target, high-rank or alternative-similarity workload. The 12-target proposal remained a separate historical plan. I use actual current account limits and representative smoke checks for newly selected configurations. I record gradient preparation, training, validation, generation and checkpoint time separately rather than extrapolating one target's timing.

I exclude Code/HumanEval under the agreed project scope. Table 10 task results remain with the GLUE/CV members; my contribution is DP correctness and matched-input search timing. Detailed historical evidence remains in [the alignment audit](MEMBER4_ALIGNMENT_REPORT.md), [parameter audit](TABLE5_PARAMETER_BUDGET_AUDIT.md), [sampling review](GRADIENT_SAMPLING_REVIEW.md) and their machine-readable records.
