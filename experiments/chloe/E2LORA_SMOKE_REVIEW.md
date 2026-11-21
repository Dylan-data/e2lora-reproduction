> 历史记录：E²LoRA Stage B 已通过 Job54466（COMPLETED/0:0，20步，第10步恢复，12/32=37.5%）。LoRA Job54429亦通过；均不可进入论文正式表图。当前实现、阻断项和运行计划见 [FORMAL_ABLATION_READINESS.md](FORMAL_ABLATION_READINESS.md)。下文保留当时的准备状态。

# E²LoRA 首次 GPU Smoke：审查与待批准方案

本轮仅代码审查、CPU 测试与配置准备，没有提交 GPU 作业。第一次 LoRA GPU Smoke 的原始目录和脚本保持不动。

更新：用户提交的Stage A Job54457已成功完成，24组、6,810,624可训练参数、零optimizer update，allocation时长60秒。Stage B已基于真实prepared-plan核对，详见 [完整检查和待提交命令](E2LORA_STAGE_B_CHECK.md)。以下审查及资源初估保留为Stage A前的方案记录。

## LoRA Baseline Smoke Test Passed

Job 54429，用户提供的 accounting 为 COMPLETED / 0:0 / 00:06:59。本地 result、逐步日志和重新计分确认：20 optimizer updates、160 examples，第10步保存后从11到20步恢复，GSM8K 11/32 = 34.375%，无解析失败，可训练adapter为6,815,744。

该准确率仅属于32题Smoke，不填入 Table 4/5 或 Figure 5 正式结果。记录在 [report/lora-smoke-54429.json](report/lora-smoke-54429.json) 和 `outputs/chloe/validation/lora-smoke-54429-passed.json`；不修改原始result。

显存需区分阶段：用户报告的16,151,098,368 bytes对应evaluation；两段training的allocated峰值都是16,654,421,504 bytes。新记录同时保留两者，不能用较小的evaluation峰值替代整次作业已观测峰值。

## 实现审查

已复核 [官方论文 §3、Appendix F/G/M](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf)。下面明确区分已有算法、当前代码行为与本次Smoke选择。

| 范围 | 已有行为及问题 | 本次处理 / 仍未完成 |
|---|---|---|
| 梯度估计 | `collect_gradients`按回调权重累积CPU FP32均值，临时解冻目标、eval模式并恢复flags/mode；一次传多个targets会同时生成这些目标的dense梯度 | 新增公共 `adapters/preparation.py`，每次只传一个target，固定批次重放，按投影处理并释放raw梯度。和全目标采集的CPU结果对照通过。该实现通过重算控制显存，不能宣称达到作者采样耗时 |
| RMI | 使用总体方差和展平后拼接分布的统计公式；对零方差/近零分母用epsilon；对角线固定1 | 保留原始值、不裁剪。负差分熵下RMI可超出[0,1]，CPU反例已验证；epsilon、对角线和模型eval模式均作为实现约定记录 |
| LSS | Greedy从首层开始，与组首相似度达到组首全局阈值时扩展连续区间 | 按Q/K/V/O分别分组，禁止跨投影混组；恢复前检查覆盖、层序、family与参数身份。LSS-only等批量消融开关仍未接入 |
| HRA | 区间最大熵，按`round(score/sum × N × base_rank)`分秩 | 显式 `rank_policy=paper`。现有默认值是shift_nonpositive，不能用于此Smoke；失败保存诊断并停止，不自动平移或改最大余数法 |
| HRA边界 | 负熵可得到反向的“重要性”比例；混合符号可能产生非正rank；分别round可让实际总rank偏离nominal；Python round的半整数采用ties-to-even | CPU覆盖上述情况；保存实际总和与每层rank。正式协议需确认非正情况、舍入和rank上界，不能自行修补后声称严格复现 |
| LSS共享 | 公共apply_plan在组内让A/B是同一对象；默认全部可训练。global_down只是额外全局共享A，并不自动冻结A | 本次只用组内A/B共享，global_down=False；基础权重、embedding/head冻结。CPU验证共享梯度是各层贡献之和、共享参数只计一次 |
| 初始化 / 梯度 | B初始化为零，A第一步梯度为零是预期现象 | 准备阶段比较安装前后loss，并做一次backward、零optimizer update；检查所有适配参数连接且梯度有限，至少一个B梯度非零。CPU同时验证B非零后A能收到梯度 |
| partial-rank-sharing | 生成JSON选用每投影256 total / 139 trainable、最大余数法及非零冻结分量；消融JSON仍pending。SplitLinear只在Dylan参考脚本，公共adapter没有该机制 | 不复制、导入或修改其他成员实现。当前Smoke显式禁用partial-rank；其正式策略仍为pending alignment。未实现/未确认的partial配置会被拒绝 |
| 参数预算 | 当各投影的组rank之和约为32×r，组内A/B全训练的唯一参数量约与LoRA相同，不会自动减半 | 用真实Llama配置的meta模型安装人工两组/rank128方案，实数计数6,815,744；这证明预算问题，不是实际采样分组。真实Smoke参数量必须在Stage A实际分组、安装后输出，不能预填3.61M或3,700,736 |
| Checkpoint | 原Math训练器只安装独立LoRA | 增加E2分支复用同一fit/evaluate；先重建共享对象再加载唯一adapter状态，绑定prepared-plan散列。CPU验证共享Llama重启后参数与不间断训练完全一致，含non-reentrant checkpoint、dropout、optimizer/RNG/游标 |
| 替代相似度 / 分布式 | 没有生产级Cosine/L1/L2/KL切换、all-reduce或多卡生成训练 | 仍列为后续任务；本次只在单卡RMI+Greedy+HRA下验收，不声称覆盖所有Table 4方法 |
| DP | src中没有独立DP搜索实现 | [独立实现与穷举对照计划](DP_GROUPING_PLAN.md)；本轮没有将Greedy包装成DP，也未生成Table 10最终数字 |

## partial-rank 与参数数的决定

本次选择是**字面机制 Smoke**：LSS+HRA，组内共享A/B且所有分量更新，partial_rank_sharing.enabled=false。它是供审阅的临时配置，并不证明论文中的冻结策略或参数减半。

冻结仅针对base/embedding/head；rank来自数据而非硬编码139。实际计数包括unique adapter总参数、unique trainable adapter和模型总可训练参数。全部group A/B可训练时三者应相等。每层记录group_index、family、rank、trainable_rank和frozen_rank=0。

团队139方案的`139 × [(4096+4096)+(4096+1024)+(4096+1024)+(4096+4096)] = 3,700,736`仅是核算其已登记规则；该数不来自本次采样，不自动作为E2LoRA通过条件。正式采用之前还要统一冻结分量初始化、forward贡献、HRA-only/LSS-only如何控制同一冻结变量，以及不同base rank的active budget缩放。

## 两阶段 GPU 验收

配置：[configs/smoke-e2lora-r8.json](configs/smoke-e2lora-r8.json)。训练器仍为本目录train.py/evaluate.py，数据、100K候选清单前160条、seed42、Prompt、max_len256、microbatch1/accum8/global8、BF16、32题GSM8K及其生成/评分均复用已通过的LoRA Smoke。

**Stage A：采样、分组和零更新梯度探针。** 取baseline epoch0打乱顺序前2条，2个逻辑采样步骤，每步1条；这只是低成本Smoke取值，不是Figure 5或论文主配置。32层×4投影×2批次 = 256次逐目标梯度backward重放，另做1次已安装adapter的梯度探针，不执行optimizer.step。每次GPU只计算一个目标的dense梯度；CPU至多保留一个投影族，Q/O FP32均值约2 GiB。保存每个投影的熵、RMI、阈值、区间、rank、失败原因及阶段时间。

Stage A通过条件：采样有限、基础参数flags恢复且无累积grad、LSS完整连续覆盖、paper分秩合法、共享身份和unique参数计数一致、零B初始loss保持一致、梯度探针通过。任何失败不自动回退策略，也不进入训练。成功后写prepared-plan.json；不持久化大型原始梯度，后续只复用已保存分组/统计量。

**Stage B：在Stage A结果审阅通过后训练。** 不重新采样，验证配置/数据/tokenizer/plan散列；先训练至第10步保存，再启动新进程恢复至总第20步，最后评估同一32题。恢复也必须恢复共享身份，不只恢复数值。该阶段不能在prepared-plan缺失或变更时启动。

| 阶段 | 资源 | 初步耗时与上限 |
|---|---|---|
| A | MGPU-TC2 / stud / normal；1 A40、8 CPU、64G RAM | 初步2–15分钟，逐目标重算尚未GPU实测；硬时限30分钟 |
| B | 同上 | LoRA已测6分59秒供参考；E2初步7–15分钟，硬时限20分钟 |

45 GiB是GPU总显存，64G是系统RAM。GPU峰值需分别记录sampling/probe、两段training和evaluation；不能预设与LoRA相同。作业启动核对A40/BF16并按GPU UUID记录10秒显存采样，保留成功LoRA脚本中已修正的GPU UUID前缀处理。

## 待批准命令

下面两条都未执行。先批准并运行A；检查其实际分组/参数/显存后，再确认B。

```bash
cd /home/msds/yahui003/e2lora-reproduction
mkdir -p outputs/chloe/slurm
sbatch --parsable --chdir="$PWD" --job-name=chloe-e2-prepare --time=00:30:00 \
  experiments/chloe/slurm/e2lora-1gpu.sbatch \
  --stage prepare --config experiments/chloe/configs/smoke-e2lora-r8.json
```

```bash
sbatch --parsable --chdir="$PWD" --job-name=chloe-e2-smoke --time=00:20:00 \
  experiments/chloe/slurm/e2lora-1gpu.sbatch \
  --stage smoke --config experiments/chloe/configs/smoke-e2lora-r8.json
```

输出目录为 `outputs/chloe/smoke/math-e2lora-r8-seed42/`，与成功baseline分离。A写prepared-plan.json、diagnostics-q/k/v/o对应的完整projection名JSON；B写manifest、train.jsonl、checkpoint、result、predictions、evaluation。SLURM日志为`outputs/chloe/slurm/smoke-e2lora8-JOBID.out/.err`，显存/设备日志仍以新JobID区分。失败不得用baseline checkpoint补齐结果。

## CPU 检查及剩余范围

CPU测试覆盖HRA最大熵/舍入/负熵/失败、LSS连续区间、共享身份与梯度求和、冻结和唯一计数、实际Llama架构预算、采样等价/异常恢复、零更新探针、prepared缓存及共享checkpoint恢复，并回归原LoRA测试。当前GPU分组和rank已由Stage A产生，Stage B的E2训练性能尚未产生。

Figure 5继续保留Math Accuracy与Code Pass@1两条路线，以及完整rank和sampling steps列表。Code-Feedback/HumanEval来源、数量、Prompt、生成及评分协议继续 **pending alignment**；不下载、不启动其正式训练，不用Math checkpoint或32题Smoke分数填Code曲线。
