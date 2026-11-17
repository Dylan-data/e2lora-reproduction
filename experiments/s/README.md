# Member 4 / NLG and Appendix K/L experiment suite

This directory contains the text-generation implementation for:

- NLG Table 2: all 12 reported methods, GSM8K math accuracy and HumanEval pass@1;
- Appendix K Table 14: LoRA/E²LoRA on six commonsense reasoning subtasks;
- Appendix L Table 15: LoRA/E²LoRA on Llama-2-13B, Qwen3-8B and GPT-OSS-20B.

Start with [the full implementation and execution guide](README_full_suites.md) and [the file-by-file implementation inventory](IMPLEMENTATION_MATRIX.md). The legacy `train.py` / `evaluate.py` scripts remain available for the earlier GSM8K LoRA/E²LoRA-only reproduction path.

The default matrix builder is plan-only. Do not start full runs until model access, dataset provenance, token-length audits, GPU smoke tests and the documented paper-vs-team choices are reviewed.
