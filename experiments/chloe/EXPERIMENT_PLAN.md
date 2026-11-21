# Experiment plan

I cover Tables 4/5, the mathematical reasoning part of Figure 5, and the DP implementation/timing work for Table 10. I exclude Code-Feedback/HumanEval under the agreed project scope.

| Study | Settings |
|---|---|
| Table 4 | LoRA reference, LSS-only, HRA-only, cosine/L1/L2/KL with rank allocation, full E2LoRA |
| Table 5 | LoRA r4 versus E2LoRA r8; LoRA r8 versus E2LoRA r16 |
| Figure 5 rank | 4,8,16,32,64 |
| Figure 5 sampling | 8,16,32,64,128,256,512 logical steps |
| Table 10 contribution | DP correctness and greedy/DP search timing on identical inputs |

The original matrix listed 24 rows before combining overlapping settings, with 72 slots for three repeats. My current compiler combines identical settings across studies and produces 19 unique runs for the explicitly selected seed 42. I add further seeds only by naming them in the configuration; I do not report a three-run mean from one run.

My order is data/protocol verification, parameter and CPU checks, representative GPU smoke validation, full baselines, mechanism comparisons, then the sweeps. The completed LoRA r4/r8 baselines are recorded in [README.md](README.md). Unrun scores remain pending. I keep smoke tests separate from full evaluations and do not choose settings using GSM 8 K test scores.

The suite JSON specifies the Figure 5 fixed values, metric conventions and rank-budget extension as reproduction choices. Historical pending configurations remain available for comparison. The current commands and protocol are in README.md rather than the older readiness snapshots.
