# Member-three execution rules

- The owner is member three, `acheng`; the assigned work is image-classification reproduction.
- All new code, configuration, logs, results, figures, checkpoints, and report copies for this task belong under `experiments/acheng/`.
- Do not create, edit, or write experiment content under `experiments/kathy/`. Historical files moved from there are archived under `experiments/acheng/archive/`; they are not active entry points.
- Before every `sbatch`, verify the repository root and working directory, task registry, `squeue`, `sacct`, current QoS/resource limits, and absence of another pending/running acheng unit.
- Before every `sbatch`, verify that the job script/configuration are under `experiments/acheng/`, and that the pre-created run directory, Slurm stdout/stderr, result, predictions, and checkpoint paths all resolve inside that run directory under `experiments/acheng/outputs/`.
- For resume, require the same immutable run configuration and matching checkpoint/code provenance; never create a new run over existing results.
- Run one model version, one dataset, one method, one seed, and one fixed configuration at a time. Do not use arrays or multi-experiment job loops. Validate the completed result before submitting the next unit.
- The previous additional-experiment cap of 2 GPU-hours total and 1 GPU-hour per unit is revoked by the user on 2026-11-20. Do not enforce or reintroduce either cap. Plan full scientific runs; when one exceeds the cluster's per-job wall-time limit, resume serially from the matching full-state checkpoint. Continue until the explicit task deadline or the executable experiment list is exhausted.
- The revoked experiment caps do not change cluster QoS, account, partition, GPU, memory, or per-job wall-time limits. Check current Slurm policy before every submission and set each segment within those limits.
- Run model computation, training, scoring, and gradient sampling only through Slurm compute jobs. Keep model weights, raw datasets, caches, and checkpoints ignored by Git; do not commit, push, create a PR, or upload artifacts.
