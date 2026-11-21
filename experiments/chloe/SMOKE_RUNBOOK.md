> 历史记录：E²LoRA Stage B 已通过 Job54466（COMPLETED/0:0，20步，第10步恢复，12/32=37.5%）。LoRA Job54429亦通过；均不可进入论文正式表图。当前实现、阻断项和运行计划见 [FORMAL_ABLATION_READINESS.md](FORMAL_ABLATION_READINESS.md)。下文保留当时的准备状态。

# Math LoRA GPU Smoke：待确认的提交与监控说明

首次LoRA Smoke已通过（用户提交的Job 54429）。本页保留该次操作说明，不应重复提交或覆盖其成功结果；新的E2LoRA Smoke使用 [独立审查与两阶段方案](E2LORA_SMOKE_REVIEW.md)。Code-Feedback/HumanEval保持待确认。

1. **训练入口**：`experiments/chloe/train.py`；配置 `experiments/chloe/configs/smoke-lora-r8.json`；提交脚本 `experiments/chloe/slurm/train-1gpu.sbatch`。不调用其他成员的训练器。训练后调用本目录 `evaluate.py`。
2. **资源**：MGPU-TC2 / stud / normal，1 node、1 task、1 GPU、8 CPU、64G 系统内存、01:00:00 墙钟上限。GPU 预期 A40，约45 GiB显存；64G 是主机内存。GRES 只申请 `gpu:1`，未绑定 A40 类型；作业启动后检查 A40/BF16/单卡，不符合就退出，不启动训练。
3. **步数**：最多20个 optimizer updates，脚本先跑至第10步，保存并退出训练进程，恢复相同 checkpoint 后继续到总第20步。不是另跑20步。配置校验拒绝 smoke max_steps >20，首次验收脚本要求恰好20。后续32题评估不做优化器更新。
4. **数据和 Batch**：seed42 的100K候选清单取前160条，训练时在这160条内部固定打乱；1 epoch。microbatch1 × accumulation8 × GPU1 = 有效Batch8，共160个microbatches、20次更新。不是正式global batch64。max_len256、Prompt上限128、r8/alpha16；BF16基础权重、FP32 adapter/AdamW状态、BF16 autocast，无量化。7条Prompt和43条Response被截断，只做流程验收。

## 5. 确认后使用的准确提交命令

下面是第一次LoRA Smoke的提交方式，保留查阅；它不是新的E2LoRA提交命令：

```bash
cd /home/msds/yahui003/e2lora-reproduction
mkdir -p outputs/chloe/slurm
sbatch --parsable --chdir="$PWD" \
  experiments/chloe/slurm/train-1gpu.sbatch \
  --config experiments/chloe/configs/smoke-lora-r8.json
```

命令成功返回的数字是Job ID，保存它。脚本内部激活 conda e2lora，使用已有本地模型/数据并设置离线模式；提交终端无需先激活环境。全部加载、前10步、恢复后10步及评估合计受1小时时限约束；尚未测得真实A40耗时，不能保证提前完成。

当前输出目录已存在成功结果，不应再次执行首次提交命令。脚本不自动覆盖旧结果；其他未完成运行的恢复需要单独审阅resume命令。

## 6. 输出位置

相对项目根目录：

| 文件 | 内容 |
|---|---|
| `outputs/chloe/slurm/smoke-lora8-JOBID.out` / `.err` | SLURM标准输出/错误；每个更新步打印loss、LR和global_step |
| `outputs/chloe/slurm/device-JOBID.json` | 实际GPU型号、UUID、总显存、BF16及Torch/CUDA版本 |
| `outputs/chloe/slurm/gpu-JOBID.csv` | 每10秒采样一次本作业GPU的NVML显存占用和利用率；不是精确峰值 |
| `outputs/chloe/smoke/math-lora-r8-seed42/checkpoint.pt` | 最新adapter、optimizer、scheduler、RNG和数据游标；第10步保存，第20步替换同名文件；不包含基础模型权重 |
| 同目录 `train.jsonl` | 逐更新步训练日志，覆盖0→10→20的两个训练进程 |
| 同目录 `manifest.json` | 配置、Git/代码/数据来源、样本清单散列、LoRA计划 |
| 同目录 `result.json` | 训练状态、最终global_step、参数数、分阶段耗时和PyTorch峰值显存；评估完成后合并Accuracy |
| 同目录 `predictions.jsonl`、`evaluation.json` | 固定32题GSM8K的逐题生成与评分；评估未完成时Accuracy仍为null，可能只有`.tmp` |
| 同目录 `failure.json`（训练失败时） | 训练异常；GPU前置检查或评估异常以SLURM错误日志为准 |

第10步后重启会再次加载基础模型，评估进程也会加载一次；这期间没有新的step日志是可能的。输出全部标记smoke，不填入论文正式结果。

## 7. 查看状态、显存和结果

在登录节点运行下面的读文件/查询命令即可。示例12345必须替换为实际Job ID：

```bash
cd /home/msds/yahui003/e2lora-reproduction
SMOKE_JOB_ID=12345  # 替换为sbatch实际返回的ID
squeue -j "$SMOKE_JOB_ID" -o '%.18i %.9P %.8T %.10M %.10l %R'
sacct -j "$SMOKE_JOB_ID" --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

PENDING表示排队，RUNNING表示运行；任务结束后通常从squeue消失，继续用sacct查看。完整验收要求主作业为COMPLETED、ExitCode 0:0，同时确认训练global_step=20和evaluation_examples=32。TIMEOUT/OOM/FAILED不能当作通过。[Slurm accounting说明](https://slurm.schedmd.com/sacct.html)

分别在终端查看；`Ctrl+C`只结束tail，不取消batch作业。文件在作业启动后出现，`tail -F`会等待并重试：

```bash
tail -F "outputs/chloe/slurm/smoke-lora8-${SMOKE_JOB_ID}.out"
tail -F "outputs/chloe/slurm/smoke-lora8-${SMOKE_JOB_ID}.err"
tail -F "outputs/chloe/slurm/gpu-${SMOKE_JOB_ID}.csv"
```

GPU CSV中memory.used/memory.total为MiB；每10秒采样可能漏过短峰。训练和评估的PyTorch allocated/reserved峰值分别记录于result的sessions和evaluation字段，不把sacct的MaxRSS当GPU显存。[NVIDIA nvidia-smi说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

训练开始产生result.json后，可用目标环境的Python读取摘要；Accuracy在32题评估完成前为null：

```bash
/home/msds/yahui003/.conda/envs/e2lora/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/chloe/smoke/math-lora-r8-seed42/result.json')
r = json.loads(p.read_text())
print('training_status:', r['status'])
print('optimizer_steps:', r['progress']['global_step'])
print('accuracy:', r['accuracy'], 'evaluation_examples:', r['evaluation_examples'])
for i, session in enumerate(r['sessions'], 1):
    print('training session', i, 'peak_allocated_GiB:', session['peak_cuda_allocated_bytes'] / 1024**3,
          'peak_reserved_GiB:', session['peak_cuda_reserved_bytes'] / 1024**3)
PY
```

## 8. 提交成功后关闭VS Code

可以。等sbatch成功返回Job ID后，SLURM已接管排队和执行；关闭VS Code或断开SSH不会停止该batch作业。保存Job ID，之后重新连接服务器读取上述文件或运行squeue/sacct。无需让Codex对话持续运行。[Slurm sbatch说明](https://slurm.schedmd.com/sbatch.html)

## 本次脚本检查

原CPU检查记录在 `outputs/chloe/validation/smoke-script-check.json`。后续Job54429已通过真实GPU验收，详见 [Passed记录](report/lora-smoke-54429.json)。新的E2LoRA采样与训练作业尚未提交。
