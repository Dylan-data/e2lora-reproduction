# E²LoRA image-classification plan — acheng

## Scope and evidence labels

## 72-hour phase override (2026-11-17)

This short-horizon plan supersedes the earlier full-scope execution order below; the broader seven-dataset/Table 3, 6, 10–13 and Figure 6 scope remains a deferred backlog, not a 72-hour completion claim. Deadline: approximately 2026-11-20 22:59 Asia/Shanghai. Preserve valid historical results and never rerun only to duplicate them.

1. **P0 — scientific validity**: audit immutable model/data/split/class maps, prompts/scoring, target-layer names, physical/shared/trainable parameter accounting and literal paper rank equations. Keep the image E²LoRA unit blocked while its pilot uses paper-policy ranks yet yields 1,325,568 parameters versus LoRA's 1,327,104 (ratio 0.998843), inconsistent with the paper's roughly half-budget display. Do not change equations or shared/frozen factors to force agreement.
2. **P1 — core EuroSAT evidence**: finish the already-registered LoRA-r8 seed-2026, 30-epoch run from its checkpoint; select by train-only validation and score the complete pinned test once. The existing zero-shot job 54495 already covers all 2,700 pinned-mirror test examples at 1,424/2,700 (52.7407%) and passes artifact consistency audit; reuse it, explicitly not as proof of the paper's original split. Run an E² paired experiment only if P0 resolves its scientific definition.
3. **P2 — optional second result**: after the core EuroSAT LoRA run is validated, prefer a full EuroSAT LoRA seed-2027 run if time permits, even if E² remains blocked; report it as an independent baseline replicate only, not as a paired comparison or a three-seed stability estimate. A second dataset is lower priority and only appropriate if its data audit and both paired methods are ready.
4. **P3 — Cars Figure 6 diagnostics**: optional only after confirming Cars source/split, training-only sample IDs, target layers, gradient and RMI provenance, and only if enough time remains.
5. Defer the remaining baseline methods, Table 6 combinations, Table 10 DP/greedy, Table 11 noise, Table 12 seeds and Table 13 rank budgets. Keep the final 4–6 hours exclusively for results audit, figures, difference analysis, runnable commands and report copy.

Observed full EuroSAT LoRA progress has cost several resumable segments; estimate the remaining epoch-23-through-30 training at roughly 35–50 minutes of serial continuation segments, then revise from the next completed segment. Do not sacrifice full epochs, test isolation or the final reporting buffer to fit the deadline.

This is the member-three plan for CLIP ViT-B/16, seven image datasets, paper Tables 3, 6, 10–13, and Figure 6. The user's explicit assignment to `experiments/acheng/` takes precedence over the inconsistent member numbering in the repository README. No other member's work is being moved, overwritten, or run on their behalf.

Every setting below is marked as one of:

- **Paper fact**: explicitly stated in the supplied ICLR 2026 paper/report.
- **Repository pin**: exact version or implementation already recorded in this checkout.
- **Team choice**: an explicit reproducibility choice, not attributed to the paper.
- **Open**: requires validation or a scientific decision.

## Initial inventory (2026-11-17)

- **Checkout**: `main` at `c9f76648020a4dcb5ccc41a52512686fa1015642`; `origin/main` is `d13dc42784dea9c3114e66f53606874f0ab41df8`. `git fetch origin --prune` succeeded and found no newer remote commit. Local main is one commit ahead. No commit, push, PR, or merge was made.
- **Uncommitted files preserved**: the supplied Word report, paper PDF, and original prompt in this directory. The confirmed EuroSAT image-classification files previously under `experiments/kathy/` were moved verbatim to `experiments/acheng/archive/migration-from-kathy-20261117/`; `MIGRATION_MANIFEST.json` records hashes and provenance.
- **Cluster**: the current shell is already on `CCDS-TC2.cm.cluster` as `weicheng001`; use this existing login environment instead of establishing another SSH connection. Two explicit SSH attempts to `10.96.189.12` timed out. Slurm is available locally.
- **QoS**: `normal`, partition `MGPU-TC2`; limits observed at initial inventory were 1 GPU, 10 CPUs, 30 GiB memory, 6 hours/job and at most 2 jobs. Job 55170 (`codex_gpu`) was a different task and was not modified. Migration-time reconciliation found no active task jobs; 55190 is `FAILED` and is eligible for a new run only after the wrapper and preflight checks pass.
- **Environment**: `e2image`, Python 3.11; PyTorch 2.5.1, NumPy 1.26.4, Pandas 2.2.3, SymPy 1.14.0, Transformers 4.57.6, Datasets 4.8.5, PyArrow 21.0.0, Pillow 13.0.0, huggingface-hub 0.36.2, safetensors 0.8.0, Matplotlib 3.11.2. `pip check` passes. Do not upgrade or reinstall.
- **Available capacity**: `/home/msds` has 7.9 TiB free. Only EuroSAT and the pinned CLIP checkpoint are presently cached; this is not evidence that the other six datasets are staged or audited.
- **Shared source**: `src/e2lora/model/clip_classification.py`, `src/e2lora/adapters/adaptive.py`, and `src/e2lora/adapters/lora_layers.py` contain the current CLIP, RMI/grouping, rank-allocation, and shared-adapter helpers. No shared source has been edited.
- **Output isolation**: the root `.gitignore` previously ignored every `outputs/` directory. A narrow exception now allows only `experiments/acheng/outputs/` metadata through; the acheng-local `.gitignore` and existing root weight/checkpoint rules continue to exclude large/private artifacts. Other output directories remain ignored.
- **Weight provenance check**: the pinned original `pytorch_model.bin` and later official `model.safetensors` revisions are both present in their existing caches. Job 55190 failed because the Hugging Face cache API did not resolve the safetensors snapshot offline, although the pinned snapshot file existed. Direct cache-snapshot lookup now compares all tensors using `torch.load(weights_only=True)` and `safetensors.safe_open`; exact equality passed for 400 tensors. No network restriction was bypassed.

## Paper facts, pins, choices, and open points

| Topic | Evidence and status |
|---|---|
| Model | **Repository pin**: `openai/clip-vit-base-patch16`, original revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`; official safetensors revision `5ef227a78de3f75873f373246dac80def63b0003`. Both cached files were compared safely and exactly (400 tensors). Record each revision and SHA256 in every run; load the converted checkpoint from its pinned local snapshot to avoid network access. |
| Target layers | **Paper fact**: FP32 updates to linear layers of the visual backbone. Existing shared helper targets q/k/v/out and fc1/fc2 in each of 12 transformer blocks; it excludes `visual_projection`. **Open**: compare the named module inventory and parameter budget with the supplied paper's intended visual-backbone scope before freezing the formal target list. |
| Optimizer/config | **Paper fact (C.5/Table 9)**: Adam, LR 1e-4, betas (0.9, 0.999), effective batch 64, weight decay 0, warmup ratio 0.3, FP32; Table 9 LR decay is 0. **Open**: define scheduler semantics and epochs using only a fixed validation split. Do not infer the image schedule from the text-generation cosine schedule. |
| Evaluation | **Paper fact**: accuracy; class prompts follow `a photo of a {class}`. **Team choice**: cache deterministic text features and CLIP processor outputs only when exact inputs are fixed. Verify dataset class-name/index mapping and preserve per-example predictions. |
| Datasets | **Repository pins**: Stanford Cars `tanganke/stanford_cars@9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40`; DTD official r1.0.1 with official split index to be recorded; EuroSAT `tanganke/eurosat@43750fc422c7fbd4289e2df3f7473617a5937acc`; GTSRB `tanganke/gtsrb@ea889fff70dda7b09e94a4b207dded438deca549`; RESISC45 `tanganke/resisc45@5c484982a196dcb501b8e79c6cf4680dc6f2ccdf`; SUN397 `tanganke/sun397@9c4afd6174afa3974f209a0ef099dbbe244dfd27`; SVHN cropped `ufldl-stanford/svhn@f9e1717d73324ebbc1ece84267ccd80d0aca0690`. **Open**: audit every split, class mapping, counts, licensing/access, and differences from the paper; the EuroSAT mirror is not to be called the paper's original split. |
| Greedy RMI/rank | **Paper fact**: Gaussian differential entropy, RMI, row-wise mean thresholds and sequential contiguous grouping; interval representative is max proxy entropy; normalized entropy factors allocate ranks by Eq. (5)–(7), with the paper pseudocode explicitly using `N × r_vanilla`. **Repository behavior**: `rank_policy="paper"` raises on nonpositive ranks rather than silently shifting entropy. |
| Adapter parameter mismatch | **Observed**: the existing 4-step EuroSAT LoRA pilot has 1,327,104 adapter parameters; E²LoRA base-rank 8 has 1,325,568, not approximately half. The paper's Table 3/13 report about 0.66M for E²LoRA rank 8 and 1.31M for LoRA rank 8. The supplied report describes partial-rank updates for text but explicitly says not to directly transfer that scheme to image. **Blocked scientific decision**: do not alter Eq. (7), freeze factors, change total rank, or claim Table 13 parity until the paper-versus-implementation discrepancy is resolved. |
| Checkpoint and accounting | **Team choice**: record physical rank, trainable rank/components, fixed-component initialization and contribution, unique shared-parameter count, adapter trainable parameters, and all trainable parameters separately. Save optimizer, scheduler, global step, and RNG states for cross-job resume. |
| Replicates | **Paper/report**: three independent runs for means. **Team choice**: use independent training/evaluation/gradient-sampling seeds, explicitly recorded; Table 12 uses its own required seeds 0, 18, 42. Do not report a seven-dataset mean while any dataset/method is missing. |
| Table 11 noise implementation | **Team choice, explicitly recorded**: draw independent Gaussian noise for upper-triangle RMI pairs, mirror it to preserve pairwise symmetry, leave the diagonal unchanged, then add before threshold/grouping. The actual matrix and separate noise seed must be saved. |
| LoRI-S image transfer | **External method fact**: original LoRI fixes random A and sparsifies B with masks extracted by a calibration/LoRI-D stage (`https://arxiv.org/abs/2504.07448`, `https://github.com/juzhengz/LoRI`). The upstream implementation targets LLMs; E²LoRA Table 6 does not specify a CLIP mask-generation procedure. **Blocked**: do not infer an image recipe or substitute standard LoRA. |

## Existing EuroSAT evidence (reuse; do not rerun merely to duplicate it)

The raw files live in the shared ignored `outputs/image-classification/` area and are not changed by this plan.

| Slurm job | Existing unit | Evidence | Current disposition |
|---|---|---|---|
| 54488 | Zero-shot pilot | Failed; pinned output shows the prior PyArrow/NumPy incompatibility. | Root cause is resolved by PyArrow 21.0.0; do not repeat the broken configuration. |
| 54493 | Zero-shot retry | Failed dependency check on PyArrow 19.0.1. | Superseded, not a valid score. |
| 54494 | Zero-shot pilot | 71/128 = 55.46875%, A40, FP32. | Pipeline example only. |
| 54495 | Zero-shot full test | 1,424/2,700 = 52.7407407407%, A40, FP32. | Valid measurement for the pinned mirror/test split, not assumed paper-equivalent. |
| 54503 | LoRA pilot | 4 updates, validation 71.875%, test 72.65625%, 1,327,104 adapter params, A40, exact in-process resume check passed. | Small-sample pilot, not formal result. |
| 54505 | Manual two-layer sharing check | 4 updates, validation 67.1875%, test 64.0625%, 663,552 adapter params, A40, shared-object/frozen-weight and in-process resume checks passed. | Integration check, not E²LoRA or ShareLoRA. |
| 54504 | E²LoRA pilot | 4 updates, validation 64.84375%, test 61.71875%, 1,325,568 adapter params, A40, literal rank-policy pilot. | **Not accepted**: rank budget contradicts paper's displayed efficiency; investigate before further E² training. |

The previous wall times around 19–21 seconds are pilot overheads for 256 train / 128 validation / 128 test examples and four updates only. They are not full-run throughput estimates. The full EuroSAT zero-shot run took about 21 seconds; no formal training duration estimate is established.

## Planned experiment units and dependencies

All new output templates resolve under `experiments/acheng/outputs/<dataset>/<method>/<seed>/<run-id>/`. Each run-id directory contains its config, environment/model/data provenance, sample IDs, logs, result, predictions when needed, checkpoint, and method diagnostics. Each row below expands to one model × dataset × method × seed × fixed-config unit; each such unit must be submitted and checked serially. Three-run seeds are provisional team choices, not paper-specified exact seeds. No future unit is submitted while its configuration is open or blocked.

| Priority | Table/Figure | Unit family (per-unit expansion) | Dataset(s) | Seeds | Prerequisite / state |
|---|---|---|---|---|---|
| P0 | EuroSAT gate | zero-shot full-test verification record; LoRA and manual-sharing pilot evidence import; E² rank-budget diagnosis | EuroSAT | 2026 pilot; Table 12 separately | Prior results inspected; E² parameter discrepancy blocked. No duplicate pilot submitted. |
| P0 | Preflight | split/class audit, processor and named visual-layer inventory, weight-conversion verification | All seven; Cars gradient subset | fixed audit seed | Source/version and split audit not complete; CPU inspection now, any model/GPU computations via one sbatch unit. |
| P1 | Table 3 | zero-shot, FFT (ViT-only), DoRA rank 4/full, AdaLoRA rank 4/full, LoRA+, LoRA-FA, VeRA, LoRA, E²LoRA, ShareLoRA, E²ShareLoRA; separate unit per method/dataset/seed | Cars, DTD, EuroSAT, GTSRB, RESISC45, SUN397, SVHN | 3 independent seeds | Method implementations, split audit, validation-based epochs, and E² decision required. EuroSAT LoRA-r8 seed 2026 is the current first unit; non-LoRA rows must use their actual method. |
| P1 | Table 6 image | DoRA/E²DoRA, VeRA/E²VeRA, LoRI-S/E²LoRI-S; separate unit per method/dataset/seed | Seven datasets | 3 independent seeds | Implement and test each underlying adapter and E² composition; no ordinary-LoRA substitution. |
| P1 | Table 13 image | LoRA r4, E²LoRA base r8; LoRA r8, E²LoRA base r16; separate unit per dataset/config/seed | Seven datasets | 3 independent seeds | E² rank/parameter interpretation is blocked; report integer counts, not rounded-only values. |
| P2 | Figure 6 | save training-only gradient samples, per-layer entropy, complete RMI matrices, thresholds, intervals, ranks, and plot | Cars | one declared sampling seed initially | Cars train split/sample IDs and target layers must be audited; compute on GPU by one sbatch unit. |
| P2 | Table 12 | EuroSAT key projection sampling, grouping and rank record, then one training/evaluation unit per seed | EuroSAT | 0, 18, 42 | Record actual subset IDs, projection, RMI, thresholds, groups, ranks, integer params, and scores. Never copy paper structures. |
| P2 | Table 11 | clean RMI baseline plus σ=0.001/0.005/0.01/0.05; draw and save zero-mean Gaussian noise before thresholds/grouping | Seven datasets | 3 independent seeds plus separate noise seed | Same base RMI, train/eval config, and method within each comparison; save actual noise matrix. |
| P2 | Table 10 | greedy and DP grouping with the same RMI, training config, and explicit cost/timing boundaries | Seven datasets | 3 independent seeds | λ absent from paper. Define and document λ/cost from the paper's appendix before training; run DP and greedy as distinct units. |
| P3 | Report and plots | summary tables, means/std only when complete, Cars plot/caption/source data, Word report copy | Acheng results only | n/a | Preserve original report; update a separate copy after verified results. |

## Execution and acceptance order

1. Keep the current EuroSAT full-test result and three pilot runs as labeled historical evidence. Do not rerun them unless a reproducible-code discrepancy is found.
2. Resolve and record the E² rank/physical-versus-trainable component interpretation before claiming a successful E² acceptance. Until then, ordinary LoRA, data/split audits, inventory and method implementation can proceed without GPU training.
3. Audit dataset origin, pinned revision, split, label names/order, sample counts/IDs, image processing, and model-weight conversion. Fix an immutable training/validation/test assignment before comparing methods.
4. On the fixed EuroSAT split, verify LoRA updates, frozen weights, scores, parameter counting, timing, memory, checkpoint recovery and class alignment. Reuse earlier evidence where its exact configuration matches; new acceptance runs must use acheng code and be separately identified.
5. Execute formal units one at a time: after each completion, inspect `sacct` ExitCode, logs, result schema/count, config hash, finite losses/scores, parameter deltas, split isolation, checkpoint and resource metrics. Only then submit the next unit.
6. First finish core Table 3 baseline/E² pairs. Then execute dependent Table 6/10–13 comparisons and Figure 6 in the dependency order above. Each failed configuration may be retried at most twice after diagnosis; never launch a batch or array.
7. Preserve per-seed raw results; compute means/std only for complete required seed/dataset sets. Keep pilots, failed runs, missing cells and any non-paper choices explicit.
8. Produce scripts/configs, raw results, plot data and plot, analysis, final checklist, and an updated Word report copy under acheng-owned paths. Do not commit/push or upload model/data/checkpoints/secrets.
