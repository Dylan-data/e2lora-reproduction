> 历史记录：E²LoRA Stage B 已通过 Job54466（COMPLETED/0:0，20步，第10步恢复，12/32=37.5%）。LoRA Job54429亦通过；均不可进入论文正式表图。当前实现、阻断项和运行计划见 [FORMAL_ABLATION_READINESS.md](FORMAL_ABLATION_READINESS.md)。下文保留当时的准备状态。

# E²LoRA Stage B 实际计划核对

结论：当前字面机制Smoke没有发现阻断性问题；Stage B尚未提交。Stage A为用户提交的Job54457，COMPLETED/0:0，allocation时长60秒。检查未修改prepared-plan、Stage A诊断或已通过的LoRA原始结果；未加载8B权重或使用GPU。

详细机器记录：`outputs/chloe/validation/e2lora-stage-b-readiness.json`。

## 计划、配置与实际安装

配置语义SHA256为`24f0919860e6df77fb05477180f6400167099be86311504c22ee006db0422b60`，与Stage A一致。prepared-plan文件SHA256为`2e3c321a148ad3fb34c0a7d1c219df7c746a1d0aa52cf3c77cce53270c0f9064`。重新读取本地源文件、100K清单和tokenizer、编码160条训练池后，data/source fingerprint一致。相关运行代码也与Stage A的provenance散列一致。

从保存的RMI重新计算Greedy阈值/区间，从熵重新按paper策略计算HRA，四类独立diagnostics与prepared-plan逐项一致。24个连续组覆盖128个投影层，每个投影的0–31层各出现一次，无重叠、遗漏或跨投影混组；所有rank为正整数。

以下区间使用0-based层号，左右端点均包含；每行区间与rank按顺序一一对应：

| 投影 | 共享层区间 | 每组rank | rank总和 |
|---|---|---|---|
| Q | 0–1；2–14；15；16；17–28；29；30；31 | 33,30,33,31,33,29,35,32 | 256 |
| K | 0；1；2–14；15–17；18–26；27；28–30；31 | 31,32,31,32,33,32,33,32 | 256 |
| V | 0–4；5–15；16–21；22–28；29–31 | 39,47,52,58,59 | 255 |
| O | 0–15；16–18；19–31 | 78,87,91 | 256 |

还逐项核对了prepared-plan的128条layer_ranks、group_index、trainable_rank和frozen_rank。用本地真实Llama配置实例化meta模型、直接调用Stage B的attach_prepared后，实数计数为48个唯一可训练tensor、6,810,624个参数；同组A/B身份共享，base/embedding/head冻结。该检查使用模型架构实例，未读取8B权重到内存。

四个投影均保留负差分熵警告，仍按已批准用于机制Smoke的字面paper规则计算，未静默平移。V分组分别round后总和255，较标称256少1；相对LoRA rank8的5,120参数差额恰为一个V的rank系数4096+1024。这是取整差异，不是论文约50%的参数节省。

## Stage B执行路径及限制

- train.py在method=e2lora分支调用attach_prepared；实际安装的是artifact中的24组及其rank。baseline_plan在此仅枚举目标层用于覆盖检查，不会安装普通独立LoRA。安装后再次核对实际参数量与Stage A一致。
- 首次训练、恢复和评估均从同一prepared计划重建共享对象。训练fingerprint绑定配置、数据/tokenizer和prepared-plan散列；恢复校验Checkpoint计划和fingerprint，评估也校验计划散列，变化会拒绝执行。
- `--stage smoke`不运行采梯度。脚本先训练到第10个optimizer update并保存，再用新进程恢复到总第20步；保存adapter、optimizer、scheduler、RNG和数据游标。160条、microbatch1×accum8×GPU1=有效Batch8，正好20个updates。之后评估固定32题，不更新参数。
- 当前E2目录只有prepared-plan与四类diagnostics，没有既有Stage B的manifest/checkpoint/result/evaluation，适合首次Stage B提交；已有训练状态时不会自动覆盖。
- GPU UUID修复`GPU-${gpu_uuid#GPU-}`在初始nvidia-smi和10秒采样两处均保留；适用于Stage A日志中不带GPU-前缀的UUID。启动仍检查单卡A40/BF16。

## 采样与冻结口径

Stage A是**2个逻辑gradient sampling steps，每步1条**，使用训练池位置61和89（原始行号329588、37435）。128个目标逐个重放两批，因此dense gradient backward计数为128×2=256；另外的adapter梯度探针不执行optimizer.step。不得将其登记为Figure 5的256 sampling steps。

partial-rank-sharing仍未启用，所有24组的A/B全部可训练，frozen_rank=0；实际参数数为6,810,624。论文参数节省尚未复现，正式冻结策略和Code-Feedback/HumanEval仍pending alignment。所有结果保留run_type=smoke，不进入正式表图。

## 完整待确认命令

下面命令尚未执行；它只提交Stage B，不重新运行Stage A：

```bash
cd /home/msds/yahui003/e2lora-reproduction
mkdir -p outputs/chloe/slurm
sbatch --parsable --chdir="$PWD" \
  --job-name=chloe-e2-smoke \
  --partition=MGPU-TC2 --account=stud --qos=normal \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time=00:20:00 \
  experiments/chloe/slurm/e2lora-1gpu.sbatch \
  --stage smoke --config experiments/chloe/configs/smoke-e2lora-r8.json
```

独立输出目录：`/home/msds/yahui003/e2lora-reproduction/outputs/chloe/smoke/math-e2lora-r8-seed42/`。

- prepared-plan与diagnostics继续保留原样。
- 新增`manifest.json`、`train.jsonl`、`checkpoint.pt`、`result.json`、`predictions.jsonl`、`evaluation.json`；checkpoint在10步保存，20步替换同名文件。
- SLURM输出：`outputs/chloe/slurm/smoke-e2lora8-JOBID.out/.err`。
- 设备/显存：`outputs/chloe/slurm/device-JOBID.json`和`gpu-JOBID.csv`。
- 已通过LoRA目录`outputs/chloe/smoke/math-lora-r8-seed42/`不写入，六个原始文件散列仍与Passed记录相同。

资源为1 GPU（运行时检查A40）、8 CPU、64G系统RAM，20分钟覆盖两段训练/恢复和32题评估。配置和prepared-plan在提交前应保持不变；本轮没有提交GPU作业、启动正式训练或Push。
