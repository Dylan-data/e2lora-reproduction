# GPU capacity checks

I used the checks below to separate training capacity, gradient sampling capacity and end-to-end experiment correctness. They are historical preflight results, not Table 4/5 or Figure 5 scores. The current training protocol is in [README.md](README.md).

## Completed checks

| Check | Job | Input and updates | Peak allocated | Peak reserved |
|---|---:|---|---:|---:|
| LoRA r8 training | 55643 | Synthetic length 2048, microbatch 1, checkpointing enabled, 2 updates | 19,882,919,936 B | 20,912,799,744 B |
| One original-weight gradient | 55644 | layer 0 q_proj, one synthetic microbatch, 0 updates | 28,374,689,280 B | 29,183,967,232 B |
| First-layer q/k/v/o gradients | 55648 | Two sequences, 4 targets, 8 backward calls, 0 updates | 28,383,159,808 B | 29,183,967,232 B |

I recorded COMPLETED/0:0 for these jobs from the supplied accounting records. The first training check took 10.978 seconds in its measured compute phase and 403.730 seconds of process time. The single-target gradient phase took 2.017151 seconds; its process time was 12.840 seconds. Job 55648 took 7.339354547 seconds in the measured phase, 22.535120953 seconds of process time, and 30 seconds in SLURM accounting. These timings cover different boundaries and are not interchangeable throughput estimates.

For Job 55648, allocated memory after every target and after every family returned to 16,077,562,368 bytes. Target-level reserved memory was 29,183,967,232 bytes. Q/O each cached 64 MiB of FP32 means and K/V each cached 16 MiB. I did not observe cumulative allocated-memory growth within this four-target check. The old record does not contain family-level reserved memory or direct weak-reference release measurements.

RSS fluctuated rather than increasing continuously: the recorded family-release values were 1414.957, 1482.395, 1449.758 and 1475.852 MiB for Q/K/V/O. Peak RSS was 5,874,597,888 bytes. RSS includes loading and allocator behavior; it is not a count of live gradient tensors. Identical per-target peak values are cumulative peaks since reset, not independent target peaks.

Evidence: [two-check summary](report/gpu-preflight-passed-55643-55644.json), [four-target review](report/gpu-preflight-reviewed-55648.json).

## Historical 12-target proposal

I prepared `spread12` to test layers 0/15/31 within each family, releasing Q before collecting K, then V and O. Two synthetic length 2048 sequences give 24 forward/autograd calls and no optimizer updates. The proposal does not install adapters or perform HRA.

| Targets cached within one family | Q/O | K/V |
|---|---:|---:|
| layer 0 | 64 MiB | 16 MiB |
| layers 0/15 | 128 MiB | 32 MiB |
| layers 0/15/31 | 192 MiB | 48 MiB |
| After family release | 0 live mean tensors | 0 live mean tensors |

The exact Q/O prefixes are 67,108,864 / 134,217,728 / 201,326,592 bytes. K/V prefixes are 16,777,216 / 33,554,432 / 50,331,648 bytes. Each target stores one averaged gradient, not one copy per sampled example. The largest persistent raw cache in this proposal is one 192 MiB family; additional transfer, statistical and allocator workspaces still consume memory.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python experiments/chloe/multi_target_preflight.py --case spread12 --sampling-steps 2 --max-phase-seconds 600
```

I did not submit the proposed spread 12 job in this record. Its old 96 G host-memory request exceeds the subsequently observed 30 G normal-QoS user limit. I do not reuse that request as a current submission command. Current resource inspection and submission are documented in README.md.

## Acceptance criteria

I check finite losses and gradients, correct target/call counts, restored mode and freeze flags, cache bytes computed from actual weight shapes, and zero surviving mean tensors after releasing a family. The 600-second phase budget is checked only between targets and cannot interrupt an individual CUDA call.

I inspect allocated memory after warmup for sustained growth and compare family-release states. A single RSS or reserved-memory increase does not establish a leak. I retain partial JSONL and error output on OOM, timeout, invalid gradients or failed release checks; an incomplete run does not pass. Existing output paths are protected from replacement.

The four-target results do not validate a whole 128-target pass, real MetaMathQA inputs, alternative-similarity workspaces or high-rank adapters. I do not estimate full sampling time by multiplying the single-target or four-target timing. Detailed lifecycle checks are in [GRADIENT_SAMPLING_REVIEW.md](GRADIENT_SAMPLING_REVIEW.md).
