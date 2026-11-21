# Historical LoRA smoke run

I retain this procedure for Job 54429, which completed successfully. I do not rerun it into the existing output directory. New suite experiments follow [README.md](README.md).

I used `train.py`, `evaluate.py`, `configs/smoke-lora-r8.json` and `slurm/train-1gpu.sbatch`. The run used 160 examples from the fixed seed 42 manifest, one epoch, microbatch 1, accumulation 8 and 20 optimizer updates. I saved at update 10 and resumed in a new process through update 20, then evaluated 32 GSM 8 K questions. Length 256 and prompt limit 128 caused 7 prompt and 43 response truncations; this was a pipeline check, not the full-training protocol.

Job 54429 completed with exit 0:0 in 6m59s. The score was 11/32=34.375%, with no answer-extraction failures and 6,815,744 trainable adapter parameters. Training peak allocated memory was 16,654,421,504 bytes; evaluation peak was 16,151,098,368 bytes. I keep the two stages separate in the record.

The original submission was:

```bash
cd /home/msds/yahui003/e2lora-reproduction
mkdir -p outputs/chloe/slurm
sbatch --parsable --chdir="$PWD" experiments/chloe/slurm/train-1gpu.sbatch --config experiments/chloe/configs/smoke-lora-r8.json
```

This historical script requested 64 G host RAM. I do not copy that request into current jobs: the later verified normal-QoS user limit is 30 G. GPU memory and host memory are separate resources.

## Records and monitoring

The original directory is `outputs/chloe/smoke/math-lora-r8-seed42/`. It contains the manifest, update log, adapter/optimizer/scheduler/RNG checkpoint, result and evaluation files. SLURM logs and 10-second device samples are under `outputs/chloe/slurm/`. A sampled NVML value is not an exact peak, and SLURM MaxRSS measures host memory.

```bash
SMOKE_JOB_ID=54429
squeue -j "$SMOKE_JOB_ID" -o '%.18i %.9P %.8T %.10M %.10l %R'
sacct -j "$SMOKE_JOB_ID" --format=JobID,State,ExitCode,Elapsed,AllocTRES
tail -F "outputs/chloe/slurm/smoke-lora8-${SMOKE_JOB_ID}.out"
```

After sbatch returns a job ID, SLURM manages the batch job independently of my SSH session. I reconnect to inspect accounting and output files. Completion requires both successful accounting and the expected update/evaluation counts. I never enter a smoke score in a full-data result table.

Evidence: [lora-smoke-54429.json](report/lora-smoke-54429.json). The subsequent E2LoRA checks are in [E2LORA_SMOKE_REVIEW.md](E2LORA_SMOKE_REVIEW.md).
