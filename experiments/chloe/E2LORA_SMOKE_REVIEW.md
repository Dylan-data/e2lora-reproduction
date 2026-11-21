# Historical E2LoRA smoke checks

I split the initial E2LoRA check into gradient preparation and training/recovery so that I could inspect the installed plan before taking optimizer steps. These checks used the older all-trainable adapter policy, not the later partial-rank policy.

## Preparation: Job 54457

I used two examples from the 160-example smoke pool, length 256, and replayed them for each of the 128 q/k/v/o targets. This gives 256 original-weight backward calls and two logical sampling steps, not 256 sampling steps. The zero-update adapter probe was separate.

The saved plan contained 24 continuous groups and 6,810,624 unique trainable adapter parameters. All group A/B components were trainable. Stage A completed with exit 0:0 and 60 seconds of allocated time. I retained entropy, RMI, thresholds, intervals, actual ranks, diagnostics and the prepared-plan hash; I did not persist the large raw gradient matrices.

I reused the public gradient averaging, population-variance entropy, RMI, greedy grouping and rank allocation. Sampling temporarily enabled only the target weight and disabled dropout. CPU tests checked weighted gradients, restored flags, continuous coverage, shared Parameter identity and initialization. Negative differential entropies were kept under `rank_policy=paper`; invalid rank ratios stopped rather than falling back to shifted entropies.

## Training: Job 54466

I rebuilt the saved shared plan without resampling, trained to update 10, restored adapter/optimizer/scheduler/RNG/cursor state in a new process and continued to update 20. The run completed with exit 0:0 and scored 12/32=37.5% on the fixed GSM 8 K smoke subset. LoRA Job 54429 scored 11/32=34.375%. Neither is a formal full-test result.

The data and training setup matched the earlier LoRA smoke:160 examples, seed 42, effective batch 8, sequence limit 256, BF16 base and FP32 adapters. I kept output directories separate and bound checkpoint recovery to configuration, tokenizer/data and prepared-plan hashes.

With A/B both trained and per-family total rank approximately 32*r, sharing changes the number and sizes of matrices but does not automatically halve the total parameter count. The 5,120-parameter difference from LoRA r8 came from the V-family rounding deficit. I did not label it the paper's claimed parameter saving.

The later partial-rank check is documented in [TEAM_RANK_CHECK_README.md](TEAM_RANK_CHECK_README.md). Current suite settings and method branches are in README.md. Detailed original-plan verification is in [E2LORA_STAGE_B_CHECK.md](E2LORA_STAGE_B_CHECK.md).
