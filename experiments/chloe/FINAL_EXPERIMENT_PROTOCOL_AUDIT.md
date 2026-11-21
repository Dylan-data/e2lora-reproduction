# 正式消融实验前最终配置核对

审计日期：2026-11-19（Asia/Singapore）
结论：**协议仍未通过正式实验放行。** 本报告区分组员4经转达确认的 Table 2 条目、仓库声明的候选值，以及没有正式运行证据的字段。**MISSING** 表示没有可核验的组员4实际配置/产物；不以 Smoke、README 示例或论文默认值补齐。

本轮只检查并生成本报告与配套 JSON；没有启动 GPU、训练、正式评估、切换分支或 Push。已有工作区改动及结果均保留。

## 1. 仓库版本与数据来源

| 项目 | 核验结果 |
|---|---|
| 当前工作分支 / HEAD | chloe-ablation / d13dc42784dea9c3114e66f53606874f0ab41df8 |
| 当前 GitHub origin/main | 1c4f3bbc0f20659dac57d484d231eb84bbb3c27e，本轮通过只读 git ls-remote --heads origin main 查询确认 |
| Git remote | https://github.com/Dylan-data/e2lora-reproduction.git |
| 工作区 | 检查前已有48条 modified/untracked 状态；未覆盖这些文件或任何结果 |
| 数据清单中的数据集版本 | Hugging Face meta-math/MetaMathQA，revision aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18，文件 MetaMathQA-395K.json。仓库数据清单指向这个固定 HF snapshot，不是将数据文件存进 GitHub 项目。[数据清单](../../src/e2lora/dataset/LIST.md) |
| 本地源文件 | /home/msds/yahui003/e2lora-storage/datasets/metamathqa/MetaMathQA-395K.json，395,000行，395,626,321 bytes，SHA-256 fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725 |
| Chloe 100K清单 | outputs/chloe/data/math-100k-seed42.json，manifest SHA-256 32aa1d172728a66da282f480f7cf7d2262d8620ae606dc19d2f34f612d21d56b；源文件哈希一致；100,000个索引的 SHA-256 3c297cb3bb2908c36a7af7e2a9b98ef138f13ae0b04af00af5a4de68d390c7c6 |
| 清单能否代表组员4 Table 2 实际输入 | **MISSING**。该清单状态仍为 candidate_pending_team_manifest_alignment；没有组员4实际有序 manifest/源文件哈希来逐项比对 |

项目 GitHub main 和固定 HF 数据集 revision 可在线查看：[项目仓库](https://github.com/Dylan-data/e2lora-reproduction)、[MetaMathQA固定 snapshot](https://huggingface.co/datasets/meta-math/MetaMathQA/tree/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18)。

## 2. 100K选择、过滤、排序与 shuffle

本地候选清单由 [prepare_data.py](prepare_data.py#L10-L35) 生成：读取源 JSON 行序后，以 sample_indices 生成100K索引；实现是 Python random.Random(seed).sample(range(size), count)，见 [loading.py](../../src/e2lora/dataset/loading.py#L11-L14)。候选 manifest 的 seed=42、source_rows=395000、index_base=0。random.sample 无放回，返回顺序保留在清单中，**没有按源行号排序**。

候选预处理没有内容级筛选、去重或排序。它只验证被选中行的 query 和 response 是字符串；当前 Chloe 编码器另外拒绝空白 response，但这是报错，不是筛除后补足100K。相同行内容若出现在不同源行中，也没有内容去重。见 [prepare_data.py](prepare_data.py#L22-L35) 与 [common.py](common.py#L205-L217)。

训练阶段顺序规则依实现而不同：

- 成员4仓库 main 的新 suite 对训练 DataLoader 使用 shuffle=True；未发现独立 sampler 的正式运行记录，故实际 epoch 顺序和 RNG 状态为 **MISSING**。成员4数据加载代码另会从未选入训练集的行中抽2K validation（seed 43），但没有证据证明 Table 2 实际运行采用该 validation。[suite_train.py](../../experiments/s/suite_train.py#L522-L527)、[nlg_suites.py](../../src/e2lora/dataset/nlg_suites.py#L60-L83)
- 当前 Chloe 训练器则对 manifest 的位置列表执行 random.Random(seed + epoch).shuffle(order)；这不是成员4训练顺序已对齐的证据。[train.py](train.py#L110-L118)
- 当前 Chloe manifest 只包含100K训练索引；与成员4另抽2K validation 的样本 ID、排序和选择用途对齐情况均为 **MISSING**。

**判定：** 已复核本地候选清单的算法和哈希；无法确认它就是组员4正式 Table 2 使用的那100K样本，也无法确认实际训练看到的样本顺序。不能用“seed 都是42”替代有序 ID/内容哈希比对。

## 3. Table 2设置与完整训练超参数

### 组员4经用户转达确认的 Table 2 项

| 设置 | 状态 | 值 |
|---|---|---|
| Model | 已确认 | Llama 3.1 8B Base |
| Dataset | 已确认到数据集/数量描述 | 固定版本 MetaMathQA，100K；确切 manifest 与行顺序 **MISSING** |
| Training seed | 已确认 | 42 |
| Epochs | 已确认 | 1 |
| max_seq_length | 已确认 | 2048 |
| max_new_tokens | 已确认 | 512 |
| Evaluation | 已确认到 benchmark 名称 | GSM8K；固定题集、版本、完整题序与实际评分器 **MISSING** |

现有 [member4-table2-confirmed.json](configs/member4-table2-confirmed.json#L1-L14) 只记录 seed、epoch、长度和生成上限四项，来源为用户转达；文件明确标注完整协议未确认。

### 正式运行的完整配置

没有发现可归属于组员4 Table 2 的正式 run_config、训练 manifest、checkpoint/result 或逐题 evaluation 产物。当前 [formal-protocol.pending.json](configs/formal-protocol.pending.json#L1-L43) 仍是 pending；其中空值没有由本报告填补。以下字段作为“实际执行值”全部为 **MISSING**：

| 类别 | MISSING 字段 |
|---|---|
| 运行身份 | 实际仓库 commit、实际入口、依赖环境锁、实际加载的模型 revision、正式运行日志/结果 |
| 数据 | 组员4实际源文件 SHA、100K有序 ID/行哈希、预处理映射、validation 清单、训练 shuffle/RNG 与恢复规则 |
| 优化器 | Table 2 实际 learning rate、AdamW 参数组、betas/weight decay 是否确实按共用设置执行、optimizer state 精度 |
| Batch与训练长度 | per-device/micro batch、gradient accumulation、effective/global batch、尾批处理、max_steps、梯度裁剪、更新总步数 |
| 调度与精度 | 实际 scheduler、逐步 LR、warmup 舍入/步序、BF16/FP32适用范围、gradient checkpointing、attention backend |
| Adapter | 各方法的 rank、alpha、dropout、初始化、缩放、目标模块、参数冻结、optimizer 纳入规则 |
| 样本目标 | train prompt、BOS/EOS、答案截断、loss mask/归一化、packing、checkpoint/validation 选择和保存恢复策略 |
| GSM8K | 数据固定 revision/内容题序 hash、评估 prompt、prompt truncation、生成解码选项、EOS/stop、失败解析计分规则、评分脚本 SHA |

当前 GitHub main 的共享参考文件声明了 AdamW、betas (0.9,0.999)、weight decay 0.0005、batch size 64、BF16、cosine scheduler、warmup ratio 0.03 和多数方法 learning rate 5e-5；suite 参数默认 per-device batch 1、gradient accumulation 64。它们是仓库参考/代码默认，不能证明成员4 Table 2 的实际 run 使用了这些值。旧 README 也注明 epochs/长度/生成长度需运行者选择。实际完整超参数仍为 **MISSING**。[成员4配置审查](MEMBER4_ALIGNMENT_REPORT.md#L90-L105)、[suite 参数定义](../../experiments/s/suite_train.py#L54-L100)

Smoke 配置的 alpha=16、dropout=0、max_length=256、global batch=8、max_steps=20、warmup_steps=1、max_new_tokens=256 等是 Smoke 专用值；不作为正式 Table 2 值。[LoRA Smoke 配置](configs/smoke-lora-r8.json)、[E²LoRA Smoke 配置](configs/smoke-e2lora-r8.json)

## 4. Prompt、GSM8K题集与评分

### Prompt

- 组员4新 suite 的 MetaMath 训练文本候选是 query.strip() + newline，没有 Question/Answer wrapper；旧 legacy 路径是 query.rstrip() + newline。是否为 Table 2 实际入口/实际训练 prompt：**MISSING**。[nlg_suites.py](../../src/e2lora/dataset/nlg_suites.py#L71-L83)、[对齐审查](MEMBER4_ALIGNMENT_REPORT.md#L108-L116)
- 当前 Chloe Smoke prompt 是 Question: {query} 加换行、Answer: 加换行。正式协议的 train_template 为空；不能将 Smoke 模板视为确认值。[formal protocol](configs/formal-protocol.pending.json#L94-L107)、[common.py](common.py#L205-L216)
- 成员4 suite GSM8K 候选评估 prompt 是 Question: {question} 加换行、Answer:，末尾没有换行。当前 Chloe evaluator 则复用配置 prompt_template；Smoke 多一个末尾换行。组员4实际 Table 2 evaluation prompt：**MISSING**。[suite_train.py](../../experiments/s/suite_train.py#L317-L335)、[evaluate.py](evaluate.py#L130-L136)

### GSM8K题集与评分

- 成员4 suite 代码把 GSM8K test 固定到 OpenAI grade-school-math commit 3101c7d5072418e28b9008a6636bde82a006892c 的 test.jsonl。成员4实际运行的题集文件 hash/有序题目 ID仍为 **MISSING**。[suite_train.py](../../experiments/s/suite_train.py#L317-L336)
- 当前 Chloe 本地候选是 HF openai/gsm8k，configuration main，split test，revision 740312add88f781978c0658806c59bc2815b9866，1,319题；本地 parquet SHA-256 为 ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59。它与 OpenAI JSONL 的 canonical question/answer 与题序是否完全相同：**MISSING**。原始文件格式不同，不能仅比较文件字节哈希。[formal protocol](configs/formal-protocol.pending.json#L120-L139)、[既有对齐审查](MEMBER4_ALIGNMENT_REPORT.md#L207-L211)
- 当前本地 experiments/chloe/evaluate.py 是 Smoke scorer 候选：取最后一个 #### 后缀；若预测没有 #### 则退回最后一个数值 token；支持逗号、科学记数、分数及简单 LaTeX frac，用 Fraction 精确相等；无法解析预测计错。Gold 必须含 ####。代码不是已确认的成员4 scorer。
- 成员4 suite 调用 score_gsm8k_generations，但其 src/e2lora/tasks/generation_metrics.py 在当前 GitHub main 中缺失；因此 scorer 的版本、抽取规则、边界行为和脚本 SHA 均为 **MISSING**。[Chloe scorer](evaluate.py#L14-L54)、[缺失项记录](MEMBER4_ALIGNMENT_REPORT.md#L120-L146)
- max_new_tokens=512 已转达确认；do_sample、temperature、top_p/top_k、num_beams、stop sequences、实际触顶比例、解析失败统计与生成完整性均为 **MISSING**。本地 Smoke 的 greedy/32题/256 token 仅用于流程验证。

## 5. LoRA / E²LoRA可训练参数

以下是现有 CPU/meta 架构审计中按实际安装的 requires_grad Parameter 去重计数，覆盖 Llama 3.1 8B 的32层 Q/K/V/O。它不是成员4 Table 2 checkpoint 计数，也不是2048正式采样后的新参数计划。LoRA 数量由层形状和rank确定；E²数量依赖分组/Rank。本地 E² r8使用2条样本、256长度 Smoke 的缓存分组，r16沿同一缓存统计重算。全部数字均不得标作正式 Table 5结果。[参数审计](TABLE5_PARAMETER_BUDGET_AUDIT.md)、[机器记录](report/table5-budget-audit-v1.json)、[Figure 5 rank记录](report/math-only-partial-rank-audit-20261118.json)

| 配置 | 唯一物理可训练参数 | 依据 / 状态 |
|---|---:|---|
| LoRA r4 | 3,407,872 | CPU/meta安装计数；与论文3.41M显示舍入一致 |
| E²LoRA base r8 | 6,810,624 | 24个Smoke缓存共享组，48个唯一A/B张量；全A/B可训练 |
| LoRA r8 | 6,815,744 | CPU/meta安装计数；与论文6.82M显示舍入一致 |
| E²LoRA base r16 | 13,631,488 | 同一Smoke缓存统计重算；全A/B可训练 |
| LoRA r16 / r32 / r64 | 13,631,488 / 27,262,976 / 54,525,952 | Figure 5候选rank的LoRA静态安装计数 |
| E²LoRA base r4 | 3,412,992 | 缓存Smoke groups；非正式采样 |
| E²LoRA base r32 / r64 | **MISSING** | 没有匹配这两个rank的安装计数/正式计划 |
| Table 4各方法 | **MISSING** | base rank、正式采样、替代相似度计划未确认；旧Smoke的sharing-only/HRA-only数字不是Table 4结果 |

Table 5目标显示值为 LoRA r4 3.41M、LoRA r8 6.82M、E² r8 3.61M、E² r16 7.22M。对照现有静态计数：

| 配置 | 论文显示中心 | 当前静态计数 | 当前计数 − 显示中心 |
|---|---:|---:|---:|
| LoRA r4 | 3,410,000 | 3,407,872 | −2,128 |
| LoRA r8 | 6,820,000 | 6,815,744 | −4,256 |
| E² r8 | 3,610,000 | 6,810,624 | +3,200,624 |
| E² r16 | 7,220,000 | 13,631,488 | +6,411,488 |

当前 E² 配置没有 partial-rank 冻结：组内 A/B 全部可训练，物理参数数与前向激活的 trainable Rank 坐标数相等。共享区间内层共享同一 Parameter，计数已按唯一 Parameter 去重。不能把共享层数乘回去，也不能把初始化零梯度当成冻结参数。Table 5预算歧义未解决；不得靠改 rank、冻结比例或参数统计口径来匹配目标。作者预算政策尚未确定前，Table 5正式参数对照为 **BLOCKED**。

## 6. Table 2与Table 4/5/Figure 5协议能否共享

| 研究 | 当前核验 | 是否可声称与 Table 2 共用实际训练/评估协议 |
|---|---|---|
| Table 2 | 仅用户转达的模型/数据集名称、seed42、epoch1、长度2048、生成512、GSM8K已确认；完整 run config、输入清单和评估包缺失 | 否；完整协议 **MISSING** |
| Table 4 | 8项机制配置已列出，但 formal JSON 的 base rank、sampling steps、指标定义、方法冻参政策未定，全部 BLOCKED_PROTOCOL_ALIGNMENT | 否；即使公共训练参数未来相同，也不能确认实际协议相同 |
| Table 5 | LoRA4↔E²8、LoRA8↔E²16 已配置为候选；E²静态参数约为对应 LoRA的2倍，与论文表预算不同；参数规则/容差未解决 | 否；参数预算与方法实现阻断 |
| Figure 5 | rank 4/8/16/32/64和采样步数8/16/32/64/128/256/512列为计划；固定采样步数、固定 rank、batch/microbatch、seed/subset/嵌套顺序均缺失 | 否；“steps”所代表的梯度样本数/批次也未确认 |

正式矩阵共24个配置、计划72次三重复；目前只知道seed42，剩余训练seed为 **MISSING**。现有配置均标记为不可进入正式表格；见 [formal配置](configs/formal/) 与 [Formal readiness](FORMAL_ABLATION_READINESS.md#L13-L25)。

## 7. 2048长度完整 E²LoRA 梯度采样资源验收

**现有证据不充分，不能证明完整128目标安全，也不能据此批准正式消融。**

| 已有预检 | 实际覆盖 | 可支持的结论 |
|---|---|---|
| Job 55643 | 合成2048输入上的独立 LoRA r8训练，2次更新；micro1/accum1 | 训练容量局部证据；不是正式global batch64训练 |
| Job 55644 | layer0 q_proj单目标、合成2048微批、一次目标梯度 | 单目标显存/时间证据 |
| Job 55648 | 4个目标，Q/K/V/O各1层；8次forward/autograd.grad；0 optimizer update；30秒SLURM；实测phase 7.339秒；peak allocated 28,383,159,808 B、peak reserved 29,183,967,232 B；目标结束后allocated均为16,077,562,368 B | 这4目标范围内没有 allocated 阶梯增长/泄漏信号；每族只有1层 |

Job 55648审阅还记录了目标后RSS约1.53–1.59 GB、进程 peak RSS约5.87 GB，但RSS含模型加载与 allocator 缓存；原记录 per-family reserved 缺失，也没有直接 CPU Tensor 弱引用释放证据。详见 [55648审计JSON](report/gpu-preflight-reviewed-55648.json#L14-L61)。

尚未验证的部分：完整128个目标；同一投影族多层累积后释放；实际2048 MetaMathQA tokenized batch；正式采样 steps × batch size × microbatch 的总梯度调用数/用时；完整 E²正式 Rank/冻结计划；RMI/其他相似度的CPU工作区峰值；完整128梯度缓存的实际RSS与生命周期；端到端保存/恢复与 GSM8K评估。当前审计估算每个Q/O目标梯度为64 MiB FP32、K/V为16 MiB，完整单族缓存可能达到GB级；估算不能替代实测。12目标 spread12 方案虽已准备，但用户已要求暂停新GPU预检，本轮没有提交或运行。

## 8. 放行前仍需补齐

1. 组员4实际 run_config、执行入口/commit、模型 revision 和依赖锁；确认完整优化器、batch、scheduler、clip、adapter、checkpoint 规则。
2. 组员4实际 MetaMathQA 文件 SHA、100K有序源行ID/行哈希、过滤处理链、训练 shuffle 与validation清单；和本地候选做逐项对齐。
3. 确认 Table 2 实际训练 prompt、GSM8K eval prompt、截断/BOS/EOS与生成参数。
4. 交付可核验的固定 GSM8K题集、评分脚本/版本/SHA、评分正反例及触顶/不可解析统计；比较 GitHub JSONL 与本地 HF题目内容和顺序。
5. 明确 Table 4机制控制，Table 5预算/冻结解释，Figure 5固定值与梯度采样定义；重算正式配置的唯一可训练参数量。
6. 在获得明确授权后，继续分级资源验证到同族多层和完整128目标，并用真实2048输入检查内存、RSS、工作区与完整调用量。当前结果不满足该条件。

**最终判断：**已确认的 Table 2 项目描述不足以重建实际训练和评估协议；其余字段必须保持 **MISSING**。Table 4、Table 5和 Figure 5目前都不能被认定与 Table 2共享同一实际协议。
