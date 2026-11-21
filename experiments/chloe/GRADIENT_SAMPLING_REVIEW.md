# Gradient sampling and memory lifecycle

I reviewed the public gradient path independently of the training optimizer. `collect_family_gradients` replays fixed CPU batches for one original linear weight at a time and calls `collect_gradients`. The base model stays frozen outside that call. Sampling uses evaluation mode, restores the previous flags and modes, and makes no weight updates.

I weight each batch's mean loss by the number of supervised causal prediction positions, `labels[:,1:] != -100`. The returned gradient is a detached FP32 CPU mean. I calculate entropy from that mean, using population variance, rather than averaging entropies from individual batches.

## Object lifetime

I use `autograd.grad(..., create_graph=False, retain_graph=False)`. Each microbatch releases the temporary loss, GPU gradients and transfer references before the next forward pass. After extracting moments, the temporary FP64 view is released. Parameters do not accumulate `.grad` during sampling.

The earlier local variables could overlap one microbatch with the next until reassignment. I shortened those lifetimes; that bounded overlap was not evidence of an unbounded cross-target leak. The change does not alter the averaging weights, RMI equations, HRA, sharing or initialization.

I retain CPU means for one projection family until its similarity matrix and plan have been computed, then release them before the next family. Raw gradient matrices are not serialized into diagnostics. The older `prepare_e2lora.py` smoke path preloads two small GPU batches; I do not use that path for long sampling sweeps.

## Memory accounting

| Projection | Weight shape | One FP32 mean | 32-layer family |
|---|---|---:|---:|
| Q | 4096x4096 | 64 MiB | 2 GiB |
| K | 1024x4096 | 16 MiB | 0.5 GiB |
| V | 1024x4096 | 16 MiB | 0.5 GiB |
| O | 4096x4096 | 64 MiB | 2 GiB |

All 128 means would occupy 5,368,709,120 bytes if retained together. Family-wise collection limits the persistent raw cache to 2,147,483,648 bytes. This excludes the model, inputs, transfer buffers and statistical workspaces. A Q/O FP64 moment copy adds 128 MiB. The alternative metrics currently retain a whole FP64 family: approximately 4 GiB plus 2 GiB of original means for Q/O, before temporary operations.

The mean cache does not grow with sampling steps. CPU input batches and compute do grow. For S logical steps, B examples per step and microbatch m, the replayed backward count is `128 * S * ceil(B/m)` when each logical batch is split separately. At S512, B1/m1 gives 65,536 backward calls; B64/m1 gives 4,194,304. These are call counts, not runtime estimates. The current suite explicitly selects B1.

## Evidence

CPU tests compare sequential collection against independent weighted autograd results, inspect weak references between forwards and families, and verify flag restoration after failure. Job 55648 covered four real GPU targets and eight backward calls, with no optimizer updates. Allocated memory after targets and family release was consistently 16,077,562,368 bytes. I did not observe accumulation within that scope; it does not establish the same result for 128 targets.

The proposed spread 12 diagnostics add actual-shape cache checks, family-level reserved memory and live weak-reference counts. I do not call `empty_cache` to conceal a growing allocation baseline. The historical 96 G proposal is not a valid normal-QoS request under the later 30 G limit.

Records: [resource audit](report/sampling-resource-audit.json), [lifecycle tests](report/cpu-verification.sampling-review.json), [four-target review](report/gpu-preflight-reviewed-55648.json), [capacity notes](GPU_PREFLIGHT.md).
