# Code delivery validation — 2026-11-21

- Integration base: GitHub main 722f013e71d5e59d214cac1c4c2fcb0c069e0771. Source-only snapshot, isolated from the SSH working branch.
- CPU regression: 128 passed, 5 skipped. The five explicitly skipped historical audits require school-local model headers / previous smoke artifacts; enable with --cluster-assets on that checkout. They are not claimed as passed.
- New suite tests: 17 passed, including all eight method branches, real parameter updates/sharing/frozen buffers, nested sampling, null rejection, protocol-separated summaries, credential/weight-free source export, and bit-identical interrupted/resumed LoRA/E2 toy training.
- Runtime: Windows, Python3.12, CPU PyTorch2.15.0+cpu, transformers4.57.6. This is a CPU correctness check, not a new 8B GPU smoke result.
- JSON compile: 19 unique math configurations for explicit seed42, zero GPU submissions; all reference null paths listed.
- Existing formal results: summarizer read 2 eligible baseline runs and reproduced 69.07% / 68.08%; no new ablation scores created.
- Original formal_math.py/train.py/evaluate.py/common.py/team_rank_check.py text preserved relative to the remote snapshot.
- Local preparation performed no GPU job, SLURM submission or Git operation. The user subsequently authorized SSH write-back and GitHub upload.
- Shell scripts were reviewed, but bash/SLURM execution was not validated on this Windows host (MSYS initialization denied). School runtime verification remains separate.
- New Figure5 settings/rank extensions are explicit reviewable reproduction choices. Code/HumanEval remains excluded under the existing Math-only scope.

School SSH environment validation before upload: 128 passed, 5 explicitly skipped, 10.35s; CUDA_VISIBLE_DEVICES empty. No new GPU job submitted.
