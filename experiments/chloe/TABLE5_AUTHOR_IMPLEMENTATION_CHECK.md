# Public implementation search for Table 5

I reviewed fixed public snapshots of [hhnqqq/MyTransformers](https://github.com/hhnqqq/MyTransformers) on November 18,2026. Within that scope I found a research README entry and two E2LoRA parser flags, but not a complete wired E2LoRA/LSS/HRA path, the four Table 5 run configurations or per-projection rank artifacts. This is a scoped search result, not a claim about private or unpublished code.

| Branch | Commit | Reachable commits | Text files inspected |
|---|---|---:|---:|
| main | `4037d4d6ba67ff88de87f90b943ff4e3a3649b67` |449 |148 |
| glue | `1d94fb595901ddb9cb5ec1ef1ef66b5997c1a408` |404 |130 |
| LoRAEden | `c8027d7af545920cd40316cbe367c82f8a56b246` |211 |121 |

I inspected 399 file versions covering 272 distinct text blobs, approximately 1.92MB, and checked metadata for 451 reachable commits and 222 historical paths. I did not read every historical file body, private/deleted refs or PR-only refs. I retrieved source text without installing the project, running its setup/training packages, fetching weights or merging it into this repository.

## Findings

The [README entry](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/readme.md#L35-L40) establishes a research connection but does not identify the paper's experiment commit. `common/parser.py` declares `--use-e2lora` and `--e2lora-similarity-threshold` with a default 0.85. I found no runtime consumers of those E2LoRA fields in the inspected source.

The 47-variant registry and dispatch in `common/lora_modules/lora_set_up.py` did not register E2LoRA. The shared initialization and gradient-preparation paths in `common/lora_modules/__init__.py` did not add an E2LoRA/LSS/HRA branch. Small isolated dispatch checks with stub classes selected the same implementation with the E2LoRA flag on or off. I did not treat this as an executable end-to-end training test or use the unused 0.85 flag as the paper's actual grouping threshold.

ShareLoRA provides global shared factors and forward slicing; that is not evidence of continuous LSS groups or HRA. GoRA contains Gaussian gradient entropy and module-budget allocation, but its path is a different method. The HiRA module implements Hadamard adaptation and is not the paper's Heterogeneity-based Rank Allocation. LoRA-FA's B-only parameter selection is also a separate method.

The mathematical script `scripts/gora/llama3.1_metamathqa.sh` specifies q/k/v/o equivalents, rank 8, GoRA and LoRA+, length 1024 and 8-GPU training. I did not relabel that script as a Table 5 E2LoRA configuration. Generic rank 4/rank 16 support or an r8 parser default does not establish the identity of any of the four target runs.

The general parameter utility counts unique `named_parameters()` with `requires_grad`, reporting decimal millions to two places. Forward slicing does not reduce a full trainable Parameter's physical `numel()`. I found no 139-active E2LoRA policy or experiment artifacts that explain 3.61 M/7.22 M in the inspected snapshots.

## Consequence for my reproduction

I retain the publication's formulas and clearly label team-specific partial freezing and integer correction. I do not present ShareLoRA/GoRA as the missing E2LoRA implementation or infer a freeze ratio from the displayed Table 5 counts. Reproducible evidence is stored in [table 5-author-implementation-check-v1.json](report/table5-author-implementation-check-v1.json); the independent numerical derivation is in [TABLE 5_PARAMETER_BUDGET_AUDIT.md](TABLE5_PARAMETER_BUDGET_AUDIT.md).
