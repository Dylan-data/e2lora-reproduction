# Acheng — image-classification reproduction

This directory is the member-three work area for the CLIP ViT-B/16 image-classification task, as explicitly assigned by the user. The repository README currently assigns different member numbers; that conflict is not resolved by changing anyone else's files.

- [PROGRESS.md](./PROGRESS.md): current status, blockers, and immediate next action.
- [AGENTS.md](./AGENTS.md): permanent ownership, path, submission, and serialization constraints.
- [PLAN.md](./PLAN.md): experiment scope, provenance, execution order, and validation rules.
- [runs.json](./runs.json): prior run evidence and structured experiment-unit templates.
- [MIGRATION_MANIFEST.json](./MIGRATION_MANIFEST.json): Kathy-to-acheng inventory, checksums, and access limitations.
- [data_manifest.json](./data_manifest.json): pinned sources, public split counts, and pending audits.
- [configs/](./configs/): acheng-owned fixed and pending experiment configurations.
- [base image config](./configs/base_image_classification.json): paper settings and unresolved fields.
- [outputs/](./outputs/): per-run logs, results, predictions, checkpoints, and raw diagnostics.
- [figures/](./figures/): generated figures and plotting data.
- [reports/](./reports/): report copies and result summaries.
- [scripts/verify_eurosat_history.py](./scripts/verify_eurosat_history.py): read-only prior-run audit.
- [scripts/adaptive_search.py](./scripts/adaptive_search.py): greedy/DP grouping and seeded RMI perturbation primitives.
- [scripts/submit_lora.sh](./scripts/submit_lora.sh): one-unit EuroSAT LoRA submit/resume entry point.
- [archive/migration-from-kathy-20261117/](./archive/migration-from-kathy-20261117/): verbatim historical source files; not active entry points.

From the repository root, submit the current single unit with
`bash experiments/acheng/scripts/submit_lora.sh`. To resume a checkpointed
segment, pass that existing run directory with `--resume`; the wrapper rejects
active duplicate acheng jobs and refuses completed or mismatched run directories.
To monitor the already-submitted 30-epoch EuroSAT LoRA run and submit each
checkpoint continuation only after the prior segment is accounted for and
validated, use:
`python experiments/acheng/scripts/continue_lora_until_complete.py --run-dir experiments/acheng/outputs/EuroSAT/LoRA-r8/2026/<run-id> --job-id <current-job-id>`.
This controller is scoped to the registered EuroSAT LoRA seed-2026 run; it
refuses other active acheng jobs, checks QoS and source/config hashes, records
each segment, and stops on unexpected errors or invalid final results.
- `image-classification-prompt.md`: supplied workflow notes; authoritative execution constraints are in the user's task.

All formal GPU experiments must be separate, sequential Slurm jobs. New code, configurations, logs, results, predictions, checkpoints, figures, and report copies belong under this directory. Previously existing files under the repository-level `outputs/image-classification/` remain historical read-only evidence and are not moved.
