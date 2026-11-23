# Member 2: language understanding

This directory retains the GLUE training entry, learning-rate validation scripts and saved-weight evaluation. Shared gradient analysis and adapter code live in `src/e2lora/adapters/`.

## What is included

- `train_glue.py`: task runner with validation mode, per-method learning rates, prediction counts, saved training state and final metrics.
- `train_tc2.sh`: TC2 launcher. It calls `../dylan/glue_full.py`, which applies the selected fixed/partial-rank settings for E²LoRA and E²ShareLoRA.
- `validate_fft_loraplus_tc2.sh`: SST-2, QNLI and CoLA validation.
- `validate_qqp_fft_tc2.sh`, `validate_qqp_loraplus_tc2.sh`: QQP learning-rate comparisons with scores after each epoch.
- `evaluate_mnli_dora_r4.py`: evaluate the saved MNLI DoRA rank-4 model without further training.

The formal wrapper uses `experiments/dylan/train_glue.py`. It retains reusable tokenized Parquet data and the same validation/reporting features as the task runner. `glue_full_tc2.sh` and `glue_follow_tc2.sh` retain resumable runs and the submission-list workflow in `experiments/dylan/`. Gradient plot scripts and CSV evidence are also retained there.

## Settings, data and dependencies

Read `references/text-understanding-setup.json`. It specifies the pinned model/data revisions, five GLUE tasks, learning rates by method, rank budgets and training schedule. Model/data storage is listed in `src/e2lora/model/LIST.md` and `src/e2lora/dataset/LIST.md`. Weights, dataset caches and training-state files are not included in Git.

Use the cluster environment documented in `handoff/glue-cluster/README.md` and its `requirements.txt`; install the package so `e2lora` resolves, or set `PYTHONPATH=src`.

## Run from the repository root

```bash
export PYTHONPATH=src
# Complete run, including partial-rank adapter setup and resumption.
python experiments/dylan/glue_full.py --task sst2 --method e2lora --seed 42 \
  --device cuda:0 --checkpoint outputs/text-understanding/sst2-e2lora.pt \
  --output-root outputs/text-understanding/results
# Three-epoch validation on up to 2,048 examples.
python experiments/dylan/glue_full.py --task qqp --method loraplus --seed 42 \
  --validation-run --validation-learning-rate 5e-5 --device cuda:0 \
  --output-root outputs/text-understanding/validation
# TC2 launcher; directory for the Slurm log must exist before submission.
mkdir -p outputs/text-understanding
sbatch experiments/text-understanding/train_tc2.sh --task cola --method fft --seed 42
```

Methods: `fft`, `lora`, `dora`, `dora-r4`, `adalora`, `adalora-r4`, `loraplus`, `lora-fa`, `vera`, `sharelora`, `e2lora`, `e2sharelora`. Tasks: `mnli`, `sst2`, `cola`, `qnli`, `qqp`.

Each result JSON records the configuration, parameter counts, prediction distribution, validation score and training time. With `--validation-learning-rate`, it also records each epoch's validation. The formal wrapper checks the selected adaptive parameter budget. Figures and reusable diagnostic CSV files are in `experiments/dylan/results/`; report scores are in the formal report at the repository root.

No training or cluster submission occurs merely by adding these scripts to Git.
