# Math ablation readiness

I use this page to distinguish completed experiments, implemented configurations and results that have not been run. The current executable protocol and commands are in [README.md](README.md) and `configs/suite-math.json`.

## Current status

- I completed LoRA r4 and r8 on the fixed 100 K MetaMathQA subset with seed 42, one epoch and 1,563 optimizer updates. GSM 8 K test scores are 69.07% and 68.08%.
- I implemented the eight method branches, rank and sampling sweeps, checkpoint recovery, independent evaluation and summaries. Compilation produces 19 unique seed 42 configurations after combining overlaps across studies.
- I validated partial-rank installation, sharing, fixed buffers and checkpoint recovery in the independent short run. Its 8-question score is a smoke result, not a full-test E2LoRA score.
- I have not produced the remaining Table 4, E2LoRA r16 or Figure 5 full-training results.
- I exclude the Code/HumanEval route under the project's Math-only scope.

## Protocol

I use the shared model/data revisions, a fixed 100 K training manifest, disjoint 2 K validation subset, length 2048, effective batch 64, one epoch, response-only token-weighted loss and greedy 512-token GSM 8 K evaluation. I verify the normalized training hash and the ordered 1319-question evaluation manifest.

My suite records the unspecified Figure 5 fixed values, alternative metric transformations, exact budget correction and partial-rank freezing as team reproduction choices. They are not evidence of the authors' exact implementation. I report actual parameter counts and their difference from the paper rather than adjusting ranks to match a displayed value.

I select only seed 42 by default. The earlier three-repeat plan is a target, not a reason to invent seeds or label unfinished results complete. Different protocol hashes stay separate in summaries.

## Checks before each new configuration

I inspect actual SLURM limits, validate the resolved JSON, and run the separate smoke phase before full training. I check grouping coverage, parameter identity and counts, fixed-buffer stability, finite training values, restored checkpoint state and generated outputs. An 8-question smoke need not match the paper's accuracy, but empty or unparseable outputs require inspection.

The capacity records for Jobs 55643/55644/55648 cover synthetic length 2048 training, one gradient target and four gradient targets respectively. I do not treat them as proof of every high-rank or long-sampling configuration. The older 96 G preflight request is superseded by the observed 30 G user limit.

## Historical records

The pending protocol files and the November 18 audits describe an earlier stage when data equivalence, scoring and partial-rank policy had not yet been resolved for the baseline route. I retain them as dated evidence, not current blanket blockers. The source code and hashes used by completed runs remain unchanged.

I contribute the DP algorithm and CPU timing to Table 10; GLUE/CV training results remain with the members running those tasks. Supporting notes: [parameter audit](PARTIAL_RANK_AUDIT.md), [sampling review](GRADIENT_SAMPLING_REVIEW.md), [DP plan](DP_GROUPING_PLAN.md), [validation](VALIDATION.md).
