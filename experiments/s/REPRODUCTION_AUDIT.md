# NLG / Commonsense / Larger-Model Reproduction Audit

## Scope

This implementation adds executable experiment scaffolding for ICLR 2026 E²LoRA:

- Table 2: all 12 listed methods on MetaMathQA→GSM8K and Code-Feedback→HumanEval.
- Appendix K / Table 14: LoRA and E²LoRA on ARC-Challenge, ARC-Easy, BoolQ, OpenBookQA, PIQA and Social IQa.
- Appendix L / Table 15: LoRA and E²LoRA on Llama-2-13B, Qwen3-8B and GPT-OSS-20B.

## Reused repository code

The implementation reuses the repository's `pretrained.py`, `dataset/loading.py`, `adapters/adaptive.py`, `adapters/generation.py`, model and dataset revision lists, and the existing text-generation setup. No original shared modules were rewritten as part of this add-on; new generation method, dataset normalization, metric and experiment entrypoint modules were added separately.

## Current verification

- `python -m compileall -q src/e2lora experiments/s`: syntax check.
- `PYTHONPATH=src python experiments/s/synthetic_smoke_test.py`: CPU-only synthetic checks for earlier math implementation.
- `PYTHONPATH=src python experiments/s/suite_synthetic_smoke_test.py`: CPU-only synthetic checks for custom adapters, multiple-choice parsing, validation holdout separation, GSM8K numeric parsing and HumanEval sample serialization.
- `run_matrix.py`: creates a 126-command plan; no benchmark matrix has been launched.

These checks do **not** establish real-model correctness, paper scores, PEFT compatibility on each architecture, or official HumanEval execution.

## Blocking items before full benchmark runs

1. Current authoring environment is CPU-only and lacks the CUDA/HF/PEFT/Accelerate stack and model weights.
2. The repo's pinned Code-Feedback public revision is about 66.4K rows, not the paper's 100K code-only examples. A provenance-verified 100K code-only source is required.
3. HumanEval dataset/source revision is not pinned. Official `human_eval/execution.py` deliberately leaves generated-code execution disabled by default. After reading its security notice, enable it manually only in a robust isolated sandbox; the runner will not alter upstream code.
4. PIQA and Social IQa are not commit-pinned in the repository. The runner requires explicit opt-in and logs data fingerprints.
5. Paper Appendix C.6 reports 8×A800, DeepSpeed ZeRO2 and FlashAttention-2. This add-on currently implements a single-process training path, not the paper's distributed ZeRO2 path.
6. Epoch count, max sequence length, max generation tokens, exact train prompts/splits for Table 14 and Table 15's data split are not specified by the paper. These remain explicit experiment-owner choices.
7. E² base rank 8 is an inference guided by Appendix J.1 matched-budget experiments (LoRA r=4 vs E² base r=8); the exact rank for Tables 2/14/15 is not directly stated. Actual parameter counts are recorded and must be compared with reference counts.
8. The paper gives C.4 warmup 3%, while Table 9 reports 0.3; these remain separately documented. LR Decay=0.1 is not assumed to be cosine minimum LR.

## Remote repository status

This work was prepared in a local overlay based on GitHub `main` at `d13dc42784dea9c3114e66f53606874f0ab41df8`. No files have been pushed and no branch or pull request has been created. The remote `experiments/s/README.md` was a placeholder when inspected. Merge only after reviewing and testing the local changes in a real clone.
