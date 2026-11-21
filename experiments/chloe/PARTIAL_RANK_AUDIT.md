# Parameter counting and partial ranks

I audited physical trainable parameters separately from sharing, forward slices and nominal rank labels. This note preserves the earlier all-trainable counts and explains the later partial-rank choice. The current suite policy is in [README.md](README.md).

Llama 3.1-8B has 32 layers. Q/O have dimensions 4096->4096 and K/V4096->1024. One shared group contributes `r * (d_in + d_out)` parameters when both factors are trainable. Sharing one Parameter across layers counts it once. The four projection coefficients sum to 26,624, giving independent LoRA `32 * r * 26624` parameters.

| Configuration | Unique trainable parameters |
|---|---:|
| LoRA r4 | 3,407,872 |
| LoRA r8 | 6,815,744 |
| LoRA r16 | 13,631,488 |
| LoRA r32 | 27,262,976 |
| LoRA r64 | 54,525,952 |
| Cached all-trainable E2LoRA r4 | 3,412,992 |
| Cached all-trainable E2LoRA r8 | 6,810,624 |
| Cached all-trainable E2LoRA r16 | 13,631,488 |
| Cached all-trainable E2LoRA r32 | 27,276,288 |
| Cached all-trainable E2LoRA r64 | 54,517,760 |
| Cached LSS-only, fixed group rank 8 | 1,253,376 |
| Cached HRA-only, base rank 8 | 6,818,816 |
| Team partial E2LoRA,139 active ranks per projection | 3,700,736 |
| Team E2ShareLoRA installation | 1,423,360 |

The cached counts use the old length 256/two-example smoke statistics. I do not present them as fresh formal groupings, GPU capacity measurements or Table 4 scores. The partial installation was subsequently checked in Job 56314; see [TEAM_RANK_CHECK_README.md](TEAM_RANK_CHECK_README.md).

## Why sharing alone did not halve the count

The round-based HRA formula assigns approximately `N * base_rank` total rank per family. With constant shapes within a family and both factors trained, the parameter count is therefore approximately conserved. The old r8 smoke used total ranks 256/256/255/256 across Q/K/V/O, giving 6,810,624 parameters. Its 5,120 difference from LoRA r8 is one missing V rank after rounding, not a50% saving.

The paper reports 3.61 M and 7.22 M for E2LoRA r8/r16. I do not derive the team's139-active budget or nonzero frozen-B initialization from that round-based formula. At 3.700736 M, the team policy is also 8.594% larger than LoRA r4 and differs from the paper's3.61 M. I disclose these differences rather than tuning ranks to displayed parameter totals.

## Method controls

I cannot apply a fixed active budget 139 to every mechanism blindly: LSS-only can have less total group rank than 139, while partial freezing would also change the HRA-only count from its all-trainable value. The current suite instead states a proportional policy explicitly, with a separate all-trainable option and separate protocol hashes. It is a controlled team variant, not a claim of exact paper implementation.

Fixed B initialized to zero would give a frozen A/B branch no forward contribution; nonzero fixed B changes the initial function. I record that choice, its scale, `alpha/group_rank` scaling and complete fixed-buffer state. Forward slicing does not shrink a Parameter's physical size. Zero gradient coordinates remain trainable and may move under weight decay or optimizer momentum. I use separate fixed buffers to avoid that ambiguity.

I verify group identity, count unique trainable tensors, and report other trainable modules separately. Global frozen A belongs to the E2ShareLoRA variant and is not substituted for ordinary groupwise E2LoRA. Detailed evidence is in [table 5-budget-audit-v1.json](report/table5-budget-audit-v1.json), [cpu-partial-rank-audit.json](report/cpu-partial-rank-audit.json), and [the exact budget audit](TABLE5_PARAMETER_BUDGET_AUDIT.md).
