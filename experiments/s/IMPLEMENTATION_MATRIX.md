# Implementation inventory

| File | Purpose |
|---|---|
| `suite_train.py` | One training/evaluation entrypoint for Table 2, Table 14 and Table 15; answer-only SFT masks prompt tokens; validates model/task/method pairing; checkpoint resume; validation checkpoint selection; final benchmark evaluation. |
| `run_matrix.py` | Generates the 126-run shell plan for three seeds; plan-only unless `--execute` is supplied. |
| `summarize_results.py` | Aggregates per-method/per-task/per-model results and parameter counts into JSON/CSV. |
| `inspect_token_lengths.py` | Tokenizer-only prompt/answer length and truncation audit before max-sequence selection. |
| `evaluate_humaneval.py` | Runs the official HumanEval pass@1 scorer only after explicit isolated-sandbox acknowledgement. |
| `suite_synthetic_smoke_test.py` | CPU synthetic update/save/restore/parser test for custom LoRA, ShareLoRA, E²LoRA, E²ShareLoRA, FFT attention-only and multiple-choice scoring. |
| `REPRODUCTION_AUDIT.md` | Current verification, source/protocol caveats, and remote repository status. |
| `configs/table2_nlg.json` | All 12 Table 2 methods, learning-rate notes, paper values and unresolved implementation fields. |
| `configs/table14_commonsense.json` | Six Appendix K datasets, source revisions, Table 14 reference metrics and evaluation caveats. |
| `configs/table15_larger_models.json` | Three Appendix L models, pinned team revisions, reference metrics and data-protocol caveat. |
| `src/e2lora/adapters/generation_methods.py` | Method-specific implementations: FFT-attention-only; PEFT DoRA/AdaLoRA/VeRA; PEFT LoRA+ and LoRA-FA; native LoRA, ShareLoRA, E²LoRA and E²ShareLoRA. |
| `src/e2lora/dataset/nlg_suites.py` | MetaMathQA 100K/held-out sampling, Code-Feedback provenance checks, HumanEval normalization. |
| `src/e2lora/dataset/commonsense_reasoning.py` | Six commonsense dataset field mappings and instruction-style multiple-choice normalization. |
| `src/e2lora/tasks/commonsense_metrics.py` | First valid A–D answer parsing and accuracy. |
| `src/e2lora/tasks/code_metrics.py` | HumanEval sample serialization and official harness wrapper. Generated code is untrusted; upstream execution is deliberately disabled until manually enabled in an isolated sandbox. |

## Reused repository implementation

- `src/e2lora/model/pretrained.py` for pinned model/tokenizer loading.
- `src/e2lora/dataset/loading.py` for Hugging Face dataset split loading.
- `src/e2lora/adapters/adaptive.py` for proxy entropy, RMI, greedy consecutive intervals and paper-style rank allocation (`rank_policy="paper"`).
- `src/e2lora/adapters/generation.py` for trainable/frozen low-rank factors, adapter bank serialization, base-layer wrapping and ordinary/E²LoRA adapters.
- `references/text-generation-setup.json`, `src/e2lora/model/LIST.md`, `src/e2lora/dataset/LIST.md` for common model/data/hyperparameter pins.

## Tests run in the current authoring environment

`python -m compileall -q src/e2lora experiments/s` (syntax check) and the CPU synthetic tests are the available checks. They do not load any Llama weights, run any real benchmark dataset, validate PEFT against each model architecture, or establish Table 2/14/15 scores.
