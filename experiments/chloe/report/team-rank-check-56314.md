# Partial-rank smoke result: Job 56314

I ran this independent acceptance check on the school SLURM cluster. It completed with exit 0:0 in 7m25s. I used Llama 3.1-8B Base revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` and MetaMathQA revision `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`.

I selected 128 training examples from the existing 100 K manifest, SHA256 `32aa1d172728a66da282f480f7cf7d2262d8620ae606dc19d2f34f612d21d56b`. Gradient sampling used seed 42, two examples without replacement, pool positions 28/6 and original rows 367696/117026. I disabled dropout and took no sampling updates, then computed population-variance entropy from token-weighted mean gradients.

The training check used two epochs and four updates in total, effective batch 64, BF16 base/FP32 adapters and maximum length 2048. The four executed learning rates were 0,5e-5,3.75e-5 and 1.25e-5. No response or prompt truncation was recorded. Evaluation covered the first eight GSM 8 K test questions at revision `740312add88f781978c0658806c59bc2815b9866`, using the historical smoke template and greedy 512-token decoding.

| Projection | Groups | Total rank | Active rank | Unique trainable parameters |
|---|---:|---:|---:|---:|
| Q | 8 | 256 | 139 | 1,138,688 |
| K | 7 | 256 | 139 | 711,680 |
| V | 5 | 256 | 139 | 711,680 |
| O | 3 | 256 | 139 | 1,138,688 |

I counted 3,700,736 E2LoRA parameters, with no other trainable modules. Installing E2ShareLoRA on the same plan gave 1,423,360; I did not train that variant. The 46 fixed tensors remained unchanged, and the active A/B tensors in all 23 groups updated. I checked shared object identity and restored complete adapter state between epochs. Fixed components are buffers, initialized with nonzero fixed B, and remain part of the forward pass.

| Epoch | Token-weighted training loss | GSM 8 K | Empty outputs | Extraction failures | Repeated long-line cases |
|---|---:|---:|---:|---:|---|
| 1 | 0.60097 | 2/8=25% | 0 | 0 | 0,4 |
| 2 | 0.59887 | 2/8=25% | 0 | 0 | 0,6 |

Losses and gradient norms were finite. Peak CUDA allocated memory was approximately 17.14 GiB and batch MaxRSS was 3,815,856 K. Example 0 extracted 18, matching the gold 18, but repeated the question. Example 1 generated `2 + 1/2 = 2.5 bolts`, yielding 2.5 instead of 3. Example 2 generated `Josh made $25,000 in profit.`, yielding 25000 instead of 70000.

I treat these as diagnostic outputs from an eight-question smoke run. They do not establish full-test accuracy or resolve the observed repetition. This record validates wiring, parameter counting, freezing, sharing and recovery. Current training settings are in the parent README; weights, raw data and checkpoints remain on the cluster.
