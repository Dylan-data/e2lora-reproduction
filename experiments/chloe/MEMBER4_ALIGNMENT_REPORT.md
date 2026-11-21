# Alignment with the generation experiments

I originally reviewed the shared generation source, local datasets and my training path on November 18. This is a historical alignment audit. The completed baseline protocol and current suite are documented in [README.md](README.md); later verification supersedes the earlier missing-artifact status recorded here.

## Data identity

I checked the local MetaMathQA source and seed 42 selection:

| Item | Recorded value |
|---|---|
| Source rows / selected rows / unique selected rows | 395,000 /100,000 /100,000 |
| Dataset revision | `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18` |
| Source SHA256 | `fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725` |
| Manifest SHA256 | `32aa1d172728a66da282f480f7cf7d2262d8620ae606dc19d2f34f612d21d56b` |
| Index digest | `3c297cb3bb2908c36a7af7e2a9b98ef138f13ae0b04af00af5a4de68d390c7c6` |

The ordered indices matched `Random(42).sample(range(395000),100000)`. The first ten were 335243,58369,13112,388786,144194,128393,117026,73158,386123,53736. Matching the source revision alone did not prove that another run used the same subset or normalization.

I subsequently verified the normalized ordered 100 K training hash `d0d8fc23d349b3b7287a7c5a5b41eee1240be5460276fe1e3bf5c91c101df700` for the completed baselines, and compared every ordered GSM 8 K evaluation-row hash with the uploaded generation manifest.

## Differences I checked

I compared exact train/eval prompts, BOS/EOS handling, response preservation, loss masks, token weighting, batch accumulation, clipping, warmup rounding and first-update LR, shuffling, checkpoint selection and complete-state restoration. The earlier smoke encoder could right-truncate responses; the formal baseline preserves responses and trims the left prompt. The smoke prompt was not taken as evidence of the formal generation prompt.

My formal baseline uses a Python `Random(seed+epoch)` shuffle. I record it as my implementation choice and do not claim an identical DataLoader permutation. I distinguish LR used by the current update from LR logged after a scheduler step. Validation loss and test generation also use separate sample manifests and purposes.

For GSM 8 K, I verify all 1319 questions and use the recorded `Question: {question}\nAnswer:` prompt. My scorer extracts the final #### suffix or final numeric token, handles commas, fractions, scientific notation and simple LaTeX fractions, and compares exact Fraction values. Unparseable predictions count as wrong; invalid gold answers raise an error. The code is checked with fixtures rather than inferred from an informal description of exact matching.

## Repository gaps at the time of review

The generation entry points referenced modules that were absent from the reviewed snapshot: `adapters/generation.py`, `adapters/generation_methods.py`, and `tasks/generation_metrics.py`, `tasks/code_metrics.py`, `tasks/commonsense_metrics.py`. Several documented configuration and synthetic-test paths were also absent. I treated these as snapshot-specific evidence gaps, not proof that a teammate had not run an experiment elsewhere.

A default flag or README command did not establish a completed run, a scorer version or a parameter policy. I therefore kept the mathematical baseline independent of unavailable generation modules and verified the actual uploaded manifests and outputs when they became available.

## Parameter and scope checks

I verified LoRA r4/r8 physical counts of 3,407,872/6,815,744. The old all-trainable shared r8 plan counted 6,810,624, not the paper's3.61 M. The team partial policy was later checked separately and is explicitly labeled a reproduction choice. I compare actual total/active group ranks and unique trainable parameters rather than rank labels alone.

The original Code-Feedback/HumanEval plan lacked a fixed complete training source, multi-turn mapping, pinned task set and tested isolated harness. The project subsequently excluded that route. I do not let its unresolved fields block Math or substitute mathematical scores for code results.

Supporting records are in the existing `report/` JSON files. The current handoff checklist is [MEMBER 4_HANDOFF.md](MEMBER4_HANDOFF.md), and the parameter derivation is [TABLE 5_PARAMETER_BUDGET_AUDIT.md](TABLE5_PARAMETER_BUDGET_AUDIT.md).
