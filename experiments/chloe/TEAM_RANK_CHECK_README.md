# Partial-rank acceptance check

I added `team_rank_check.py` and `configs/team-partial-rank-short.json` to check the team's partial-rank interpretation independently of the old all-trainable smoke. I reused the public collection, entropy, grouping, partial adapter and state/count routines.

For each projection, I allocated total rank 256 with exact largest remainders from unshifted interval entropy ratios, then distributed active rank 139 in proportion to group ranks. This integer-budget correction and partial freezing are team reproduction choices, not a claim about released author code.

I sampled two distinct examples with Random(42).sample from the actual 128-example training pool. I averaged original-weight gradients by supervised causal token counts before entropy calculation. Sampling disabled dropout and took no optimizer steps. A used the public nn.Linear initialization; active B started at zero; fixed B used uniform(-0.01,0.01). Fixed components were forward-participating buffers outside AdamW, and checkpoints included them.

## Recorded run

Job 56314 completed with exit 0:0 in 7m25s on an A40. I used 128 training examples, two epochs, two updates per epoch, BF16 base/FP32 adapters, microbatch 1, accumulation 64 and sequence limit 2048. The smoke optimizer used AdamW 5e-5, betas(.9,.999), weight decay.0005, cosine scheduling, one warmup step and clipping 1. Its executed update learning rates were 0,5e-5,3.75e-5 and 1.25e-5; a logged post-update LR refers to the next update.

The eight-question GSM 8 K check used the older Question/Answer smoke prompt and greedy 512-token generation. I did not truncate responses or pad every example to 2048. This prompt and schedule are historical smoke settings, not the current formal baseline protocol.

| Projection | Groups | Total rank | Active rank | Trainable E2LoRA parameters |
|---|---:|---:|---:|---:|
| Q | 8 | 256 | 139 | 1,138,688 |
| K | 7 | 256 | 139 | 711,680 |
| V | 5 | 256 | 139 | 711,680 |
| O | 3 | 256 | 139 | 1,138,688 |

I measured 3,700,736 E2LoRA parameters and 1,423,360 E2ShareLoRA parameters. E2ShareLoRA was installed and audited, not trained. I verified 46 unchanged fixed tensors, updated active A/B tensors, group sharing identity and complete restoration between epochs.

I queried MGPU-TC 2/stud/normal and recorded limits of one GPU,10 CPU,30 G host memory, six hours and two submitted jobs per user. The actual A40 reported 47,709,159,424 bytes. This check requested 8 CPU,1 GPU,30 G and 30 minutes.

```bash
conda run -n e2lora python experiments/chloe/test_team_rank_check.py
# Historical submission; use a fresh output directory for a new run.
sbatch experiments/chloe/slurm/team-rank-check.sbatch
squeue -u "$USER"
```

The original artifacts remain under `outputs/chloe/smoke/team-partial-rank-20261120a/`. Full results and examples are in [report/team-rank-check-56314.md](report/team-rank-check-56314.md). Passing this check does not turn its eight-question score into a formal result or change older entry points implicitly.
