# Validation record

I validated the suite on an isolated copy of main before integrating it into my school SSH checkout. The initial integration base was `722f013e71d5e59d214cac1c4c2fcb0c069e0771`; the later merged main was `ae243032b96b175d1a9f83fe3d37e3d515449c39`.

## Recorded checks

- CPU regression:128 passed,5 skipped. The skipped historical audits require school-local model headers or previous smoke artifacts and can be enabled with `--cluster-assets`.
- The 17 new suite tests cover all eight methods, real adapter updates, sharing and fixed-buffer stability, nested sampling, unresolved-null rejection, separate protocol summaries, source-only export, and bit-identical interrupted/resumed LoRA/E2LoRA toy training.
- The local CPU environment used Python 3.12, PyTorch 2.15.0+cpu and transformers 4.57.6. It did not constitute an 8B GPU smoke run.
- The school environment passed 128 tests with 5 skips before upload in 10.35s and after merging main in 11.19s, with CUDA_VISIBLE_DEVICES empty.
- Compilation produced 19 unique seed 42 configurations and submitted no jobs. Every unresolved reference field was listed.
- The summarizer read the two completed baseline results and reproduced 69.07%/68.08%. It did not generate scores for unrun ablations.
- I retained the original formal_math.py/train.py/evaluate.py/common.py/team_rank_check.py source used by the existing runs.

I did not launch a new GPU experiment for the source delivery. New Figure 5 fixed settings and rank extensions are documented reproduction choices; Code/HumanEval remains excluded. The Windows host could not initialize MSYS for a shell syntax check, so I did not claim a local bash/SLURM execution pass.

I keep later documentation edits separate from executable changes. Translation and wording do not alter training code, configuration values, data identities or recorded experimental results.
