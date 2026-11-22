# Math ablation experiments

I maintain the mathematical reasoning ablations for Tables 4 and 5, the mathematical curves in Figure 5, and the DP grouping implementation for Table 10. I use Llama 3.1 8B Base, a fixed MetaMathQA training subset, and GSM 8 K test. Code-Feedback and HumanEval are outside the agreed scope of this reproduction.

## Entry points

| Purpose | File |
|---|---|
| Select and record the fixed training subset | `prepare_data.py` |
| Reproduce my completed LoRA baselines | `formal_math.py` |
| Train the current ablation suite | `suite_train.py` |
| Define method switches, sampling and rank budgets | `suite_plan.py` |
| Compile configurations, submit jobs and inspect progress | `run_ablations.py` |
| Resume interrupted jobs in sequence | `watch_suite.py` |
| Generate and score GSM 8 K answers | `evaluate.py`, `evaluate_suite.py` |
| Produce tables, budget comparisons and curves | `summarize.py` |
| Inspect cluster resources and launch training | `slurm/inspect_cluster.sh`, `slurm/train.sbatch` |
| Compare greedy and DP grouping | `benchmark_grouping.py` |
| Export and verify source files | `export_code.py`, `review_upload.py` |

I retain the original `formal_math.py`, `train.py` and `evaluate.py` interfaces so that earlier runs remain reproducible. New experiments use `run_ablations.py` and `configs/suite-math.json`. The older pending configurations and audit notes document previous checks; they do not define the current suite.

## Environment

I ran the baselines on the school SLURM cluster with an NVIDIA A40, PyTorch 2.7.1+cu 128 and the `e2lora` environment. Python 3.10 or later is required. I install the additional packages into the existing CUDA environment:

```bash
python -m pip install -e .
python -m pip install -r experiments/chloe/requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES='' python -m pytest experiments/chloe/tests experiments/chloe/test_auto_resume.py experiments/chloe/test_formal_math.py experiments/chloe/test_team_rank_check.py -q
```

Five historical checks need the model headers and smoke artifacts stored on the school server. I run those separately with `--cluster-assets`; the default CPU run reports them as skipped.

## Training and evaluation

I pin the model to `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, MetaMathQA to `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`, and GSM 8 K to `740312add88f781978c0658806c59bc2815b9866`. Local paths are specified in the configuration.

I use 100,000 fixed training examples and 2,000 disjoint validation examples, one epoch, a maximum sequence length of 2048, and response-only loss weighted by the number of supervised positions after the causal shift. The effective batch is 64: one example per microbatch and 64 accumulation steps, with a smaller final batch. I preserve the response and EOS and trim only the left side of the prompt; a response that cannot fit raises an error.

The optimizer is AdamW with learning rate 5e-5, betas (0.9, 0.999), and weight decay 0.0005. I use BF16 base weights, FP32 adapters, no gradient clipping, and a cosine schedule with a zero minimum and `floor(total_steps * 0.03)` warmup steps. The adapter targets are q/k/v/o, alpha is 16, and dropout is zero.

For training, I append a newline to the stripped query and EOS to the stripped response. For evaluation, I use `Question: {question}\nAnswer:`, greedy decoding and at most 512 new tokens. I evaluate all 1,319 GSM 8 K test questions, verify their ordered hashes against the shared manifest, and compare extracted answers as exact rational numbers. Empty and unparseable outputs count as incorrect. I record validation loss, evaluation scores, generated examples and repeated-output diagnostics after each epoch.

## Ablation settings

I distinguish the paper's settings from the team's implementation choices. Partial rank freezing, exact integer budget correction and the transformations used for alternative similarities are reproduction choices, not a claim about released author code.

The current suite specifies rank 8 and 64 sampling steps for the main comparison. Figure 5 fixes 64 sampling steps when varying rank and fixes rank 8 when varying sampling steps. One logical sampling step contains one example. These fixed values are documented choices where the paper leaves details unspecified.

For HRA, I specify a total rank budget of `32 * base_rank` per projection across all groups. The partial policy uses an active fraction of `139/256`: I round the active total half up and distribute it across groups with the public largest-remainder routine. At base rank 8, the totals are 256 and 139. With HRA disabled, each group keeps the specified base rank, so the projection total depends on the number of groups. Ordinary LoRA trains all rank components. `all_trainable` is available as a separate configured policy; I keep its results in a separate protocol and output directory.

I currently select only seed 42. Seeds 43 and 44 are not prescribed by the paper. If I add independent repeats, I list the seeds explicitly and report the actual number of completed runs. The reference three-repeat target is not evidence that three runs have finished.

Compilation lists every null field in the reference configuration. In the suite configuration, `data.train_limit=null` and `evaluation.limit=null` mean the complete specified datasets, `training.max_steps=null` means complete epochs, and `training.max_grad_norm=null` means no clipping. Other unresolved nulls raise an error.

## Commands

I run these commands from the repository root after setting the local model and dataset paths:

```bash
python experiments/chloe/prepare_data.py --source /path/to/MetaMathQA-395K.json --output outputs/chloe/data/math-100k-seed42.json --count 100000 --expected-rows 395000 --seed 42 --revision aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18

# Compile configurations without submitting GPU jobs.
python experiments/chloe/run_ablations.py compile
python experiments/chloe/run_ablations.py status

# Check the actual account limits before submitting.
bash experiments/chloe/slurm/inspect_cluster.sh
mkdir -p outputs/chloe/slurm
python experiments/chloe/run_ablations.py submit --index 0

# Run only inside an allocated GPU job.
python experiments/chloe/run_ablations.py run --index 0 --smoke-only
python experiments/chloe/run_ablations.py resume --index 0
python experiments/chloe/evaluate_suite.py --run-dir outputs/chloe/suite-v2/formal/lora-r8-n0/seed-42 --epoch 1

# Optional serial controller in a persistent terminal session.
tmux new -s chloe-suite
python experiments/chloe/watch_suite.py --indices 0 1 2 --max-resumes 20

python experiments/chloe/summarize.py --roots outputs/chloe/suite-v2 outputs/chloe/ablations-math-v1 --output outputs/chloe/summary
python experiments/chloe/benchmark_grouping.py --input outputs/chloe/similarity-matrices.json --penalty-lambda 1 --mean-policy include_diagonal --warmup 20 --repeats 200
python experiments/chloe/export_code.py --output outputs/chloe/review/chloe-source.zip
python experiments/chloe/review_upload.py --manifest /path/to/UPLOAD-MANIFEST.json
```

I sample gradient examples without replacement from a fixed permutation of the actual training pool. Step sweeps use nested prefixes. The sample count is independent of the microbatch size. I compute gradients on the unmodified base model in evaluation mode, make no optimizer updates during sampling, and average gradients before calculating the population-variance entropy proxy. The implementation caches one projection family on CPU at a time and reuses the public gradient, grouping and adapter routines.

## Checkpoints and job limits

I save the adapter parameters, fixed buffers, optimizer, scheduler, random states and update cursor with an atomic checkpoint replacement. Resuming verifies the configuration, source hashes, data and tokenizer identity. A saved plan is rebuilt without resampling gradients. Evaluation can continue from completed prediction records.

The controller retries only a timeout or exit code 75 when saved state has advanced. Cancellation, OOM, configuration failures, repeated lack of progress or the retry limit stop it for inspection. Gradient preparation has no mid-backward checkpoint. I keep the controller in tmux; SLURM manages jobs that have already been submitted.

## Completed baselines

| Method | Seed | Trainable parameters | GSM 8 K test |
|---|---:|---:|---:|
| LoRA r4 | 42 | 3,407,872 | 911/1319 = 69.07% |
| LoRA r8 | 42 | 6,815,744 | 898/1319 = 68.08% |

Both runs completed 100 K examples, one epoch and 1,563 updates under the original `formal_math.py` protocol. I keep their protocol hashes separate from new suite runs. I have not filled in unrun Table 4, E2LoRA r16 or Figure 5 scores, and I do not report smoke scores as full-test results.

I commit source, configuration and small audit records only. Models, raw data, credentials, logs and checkpoints remain outside Git. [Validation](VALIDATION.md), [source review](UPLOAD_REVIEW.md), and the historical audit notes provide the supporting records.


## HumanEval Figure 5 extension

I have added a separate HumanEval sensitivity suite for Figure 5. This extends the earlier math-only scope described above. All rank and gradient-sampling points use the same Llama 3.1 8B Base model, train on Code-Feedback, and evaluate HumanEval pass@1. See HUMANEVAL_ABLATIONS.md for the configuration, data requirements, SLURM command, checkpoint recovery, and sandbox scoring steps.
