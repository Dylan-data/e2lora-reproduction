# Math-only 正式消融准备状态

更新：2026-11-18（Asia/Singapore）。**当前尚不具备第一组Math正式消融条件。** 用户已完成训练、单目标及四目标2048非正式GPU预检；本次续审仅做代码/CPU测试和记录，不另提交GPU作业、不Push，既有Smoke与预检原文件保持原样。

## 已批准范围与新确认设置

**HumanEval 因隔离执行环境限制，经组内批准不纳入本次复现范围。** Code-Feedback同步排除。Code框架及12份配置保留并标记禁用，活动矩阵、编译和formal入口拒绝Code。只读 `squeue --me` 返回空，未发现待取消Code作业；没有执行scancel或提交新任务。

成员4 Table2 Math已确认：**seed42、epochs1、max_seq_length2048、max_new_tokens512**，1024会报错。来源是用户本轮转达，记录在 [member4-table2-confirmed.json](configs/member4-table2-confirmed.json)，相应Math pending字段已回填。成员4其他两表仍待最终确认；不将本次四个值当作全部训练/评分协议确认，也不猜另两颗seed。

[正式协议](configs/formal-protocol.pending.json) 保持pending。seed列表现在只有[42]；支持协议齐备后先编译这一颗seed，完整三重复统计仍须另外两颗明确种子。新加formal_preflight门禁，旧256长度Smoke通过不能放行2048正式配置。GSM8K不执行生成程序，不再要求填HumanEval类型的执行timeout/isolation。

## 活动矩阵

| 范围 | 配置数 | 三重复计划 | 当前已知seed42槽位 |
|---|---:|---:|---:|
| Table4：LoRA参考、LSS only、HRA only、Cosine/L1/L2/KL、E²LoRA | 8 | 24 | 8 |
| Table5：LoRA r4/E² r8、LoRA r8/E² r16 | 4 | 12 | 4 |
| Figure5 Math rank：4/8/16/32/64 | 5 | 15 | 5 |
| Figure5 Math sampling steps：8/16/32/64/128/256/512 | 7 | 21 | 7 |
| 合计 | **24** | **72** | **24，全部仍被协议阻断** |
| Code活动GPU作业 | 0 | 0 | 0 |
| Table10 DP正确性/搜索耗时 | CPU方法交付 | 不计入72次训练 | 不需要GPU |

[formal-matrix.csv](formal-matrix.csv) 仅含72个Math槽位，48个额外重复seed为PENDING；[experiment-matrix.csv](experiment-matrix.csv)为24个数学配置加Table10概览。目录仍有24份活动+12份禁用Code配置，无删除框架。seed42下不提前复用跨表checkpoint；将来只有协议、方法、rank、采样与数据完全同一并登记来源时才考虑去重。

## 真实参数及论文一致性

[本轮meta审计](report/math-only-partial-rank-audit-20261118.json) 在实际Llama架构安装adapter，未加载8B权重、未用GPU；使用旧2-example/length256 Smoke的缓存分组/entropy，所以不是正式采样结果。

| 配置 | 唯一可训练参数 | 意义 |
|---|---:|---|
| LoRA r4 / r8 | 3,407,872 / 6,815,744 | 两个基线预算 |
| 全A/B E² r8 / r16，缓存Smoke统计 | 6,810,624 / 13,631,488 | 对应LoRA r4/r8分别高约99.85%/100%，不能宣称Table5预算匹配 |
| active139，组内A/B候选 | 3,700,736 | 相对LoRA r4高8.594%，并非已批准策略 |
| LSS only固定组rank8 / HRA only，缓存统计 | 1,253,376 / 6,818,816 | 仅核验开关计数，分组来自短Smoke，不能填Table4 |

论文Table4含7行机制/相似度，LoRA为本组补充参考，共8配置；Table5比较LoRA4/E²8和LoRA8/E²16。参数公式/表值的冲突、group-rank缩放、冻结分支初始化和各rank active budget均未解决。不得按3.61M/7.22M反推冻结比例。替代指标定义和Table4统一控制、Table5预算差容差必须先登记，详见 [PARTIAL_RANK_AUDIT.md](PARTIAL_RANK_AUDIT.md)。

## A40：两项2048容量预检已通过

| 验收 | Job / 状态 | 实测结果 | 适用范围 |
|---|---|---|---|
| 独立LoRA r8训练，checkpoint on，micro1/accum1 | 55643，COMPLETED 0:0 | 2更新；allocated 19,882,919,936 B；reserved 20,912,799,744 B | 两次合成2048序列训练，非正式结果 |
| 单目标eval梯度，layer0 q_proj | 55644，COMPLETED 0:0 | 0更新；2.017151 s；CPU梯度64MiB；allocated 28,374,689,280 B；reserved 29,183,967,232 B | 一个目标/一个合成微批，非完整采样 |

本地JSON/stdout已交叉核对，SLURM状态依据用户提供的账单，没有冒充本次查询。记录见 [两项验收摘要](report/gpu-preflight-passed-55643-55644.json)。训练阶段实测10.978 s（两步约9.769/1.209 s），JSON进程总计403.730 s；采样进程总计12.840 s。加载/首步/文件缓存影响明显，均不等于稳定正式吞吐或完整allocation账单。

这两项无需重复提交。checkpoint-off训练和r64不是本次下一项优先级；两项通过也不等于真实数据、完整128目标、partial-rank或评分链路已通过。正式global64若使用micro1仍需accum64，尚未据此验收正式batch。协议formal_preflight继续pending。


四目标Job55648也已完成并通过本次受限审阅：用户提供COMPLETED/0:0、30秒；本地JSON/JSONL核对4目标、8次forward/autograd.grad、0更新，phase7.339354547秒。所有目标与族结束后的allocated均为16,077,562,368 B，目标后reserved均为29,183,967,232 B；CPU缓存64/16/16/64MiB符合形状。RSS波动而非持续增长，未见本次范围的泄漏信号。旧per_family缺reserved且没有直接弱引用证明，不能扩展为128目标已无泄漏。见 [55648审阅记录](report/gpu-preflight-reviewed-55648.json)。原始结果状态不改写。

## 完整梯度采样：本次CPU审查

详见 [GRADIENT_SAMPLING_REVIEW.md](GRADIENT_SAMPLING_REVIEW.md) 和 [资源/次数核算](report/sampling-resource-audit.json)。完整路径逐个临时解冻目标，以autograd.grad计算，不创建optimizer、不累积Parameter.grad；CPU缓存按投影保留，处理完投影再释放。

本轮仅收紧临时量生命周期：每微批显式释放loss/梯度/搬运副本，显式关闭retain/create graph，FP64矩副本及时释放；数学规则、loss加权、HRA、共享/冻结政策不变。90项CPU测试通过，包括独立梯度数值对照、弱引用生命周期和跨投影缓存释放。四目标GPU已验证该范围allocated不累积；同族多层/完整128目标尚未验收。

128个目标FP32总量若全留为5GiB，当前单投影最大缓存为2GiB；另有传输/统计副本、Python数据与batch。RMI临时FP64副本最大128MiB；替代指标会保留整个投影的FP64副本，Q/O约4GiB，加原始FP32约6GiB及工作区。96G主机RAM仍须用实际RSS核验，不能把2GiB当完整需求。

真实反向次数为 `128 * S * ceil(B/m)`，S为逻辑steps、B为每步样本、m为微批。S=512时，B1/m1为65,536次，若采用尚未确认的B64/m1则为4,194,304次。梯度均值缓存不随S增长，但CPU输入缓存和计算次数增长。不得将q0单目标2.017 s直接乘这些次数预测总耗时。

## 下一项GPU方案：12目标，准备未提交

[GPU_PREFLIGHT.md](GPU_PREFLIGHT.md)已登记四目标通过及12目标命令。spread12每族包含layers0/15/31，Q三层→释放→K三层→释放→V三层→释放→O三层→释放。固定2条合成2048序列，24次反向、0更新；1×A40、8CPU、96G、30分钟硬上限，另有600秒目标边界预算。

预期Q/O缓存64→128→192MiB，K/V为16→32→48MiB；族释放后mean Tensor弱引用全部死亡。仅增加诊断：实际权重形状的预期缓存前缀校验、释放后reserved、弱引用存活数；不改采样内核、HRA或冻结规则。OOM/非finite/缓存大小不符/释放失败/计数或模式错误/预算超时自动停止；目标后allocated持续阶梯上涨需人工停止或拒绝扩展。RSS/reserved单点上涨不直接判泄漏。

新输出使用spread12-v1，first-layer4-v1及旧结果不覆盖。12目标完成后先审阅，再决定是否另行准备单族32层和全部128；当前CLI仍拒绝full128和大于2步，不会自动扩大。7.339秒是4目标实测，不乘3当12目标预测。

## 资源估算口径

活动矩阵仍是24配置/72次三重复槽位，只有seed42已知。100K、1epoch、global64时每次1,563更新，尾批32，72次共112,536更新；这是精确条件计数，不是耗时预测。

[旧资源情景](report/math-only-resource-plan.json)基于长度256短Smoke的线性假设，仅保留历史风险说明，不能用新单目标结果重新背书。当前完整128目标/不同S的总耗时仍为null，需4→12→单投影32→全128的受限分级测量及确认真实输入；加载、梯度、CPU统计、训练、验证、生成、保存分别记录。2.017秒不能直接代表所有层/投影/样本，也不能用两个训练更新替代真实global64吞吐。

## 剩余Math交接与实现阻断

1. **数据**：成员4实际100K有序ID/原始或规范化行哈希、源SHA/行顺序、validation选择集。现存Chloe清单核验通过，但仍是候选，未与成员4实物比较。
2. **完整训练配置**：除四项新确认值外，还需alpha/dropout/dtype/scaling/初始化、clip、micro/accum、scheduler逐步LR/取整、shuffle/RNG/恢复、验证频率与best/final选择。当前实现与成员4suite/legacy存在差异，不能只填status。
3. **Prompt/预处理**：train与eval模板分别确认，包括末尾换行、BOS/EOS、max_prompt_length、保留答案还是右截响应、packing、监督mask及规范化。成员4保留全答案，Chloe现有支持响应右截，需按共同决定修改后验收。
4. **GSM8K**：固定1319题内容/ID/题序，证明成员4OpenAI JSONL与本地HF版本等价；实际numeric评分模块、SHA256及正反例，完整生成/停止配置，加载Chloe完整冻结状态的独立driver与环境锁。目前旧main数学评分/adapter模块缺失；不要求补做Code，只要求数学入口可独立运行。
5. **消融口径**：Table4 base rank、主采样步数、指标定义/统一冻参控制；Table5真实预算政策与容差；Figure5采样batch/seed/顺序/逻辑steps与两个固定值。
6. **正式预检**：2048独立训练与单目标容量已通过，但多目标采样、真实group/partial策略、保存恢复/完整评分验收和资源标定仍未通过。Job54429/54466与55643/55644都仅为流程/容量凭据。

91条机器检查提示对应上述未确认字段与证据，不是91个独立程序错误；详见[本次formal-preflight](report/formal-preflight.sampling-review.json)。HumanEval不再参与Math门禁。详细交接清单见 [MEMBER4_HANDOFF.md](MEMBER4_HANDOFF.md)，对齐审查最新结论见 [MEMBER4_ALIGNMENT_REPORT.md](MEMBER4_ALIGNMENT_REPORT.md)。

## 从现在到首组正式Math的最短步骤

1. 接收成员4的Math run_config、100K+validation清单、精确Prompt和可执行GSM8K评分包；只围绕Math确认，其他两表试跑和HumanEval不进入关键路径。
2. 确定共同的shuffle、scheduler、clip、截断、checkpoint选择和adapter策略；解决Table4/5预算问题，记录Figure5固定值。按确认规则修补接口并跑CPU测试、数据/题集哈希与评分正反例。
3. 审阅并执行受限多目标采样，再用确认真实数据/adapter做小规模端到端训练、恢复及GSM8K生成/评分验收；测真实吞吐与采样耗时。独立训练/单目标容量已通过，本轮不重复或自动扩大。
4. 将验收报告/SHA回填正式协议，运行alignment/preflight使Math阻断清空；编译seed42批次，选择一组已确认rank/采样的LoRA/E²对照逐项审阅。先默认CPU验证，不提交72项array。
5. 首组条件齐备并获正式GPU启动指示后，再启动该组；完成并审计后扩展Table4/5和Figure5。额外两个seed确认后再补齐三重复，完整前不报三次mean/std。

首组仍被**真实清单/Prompt/评分脚本、更新规则差异、预算策略和2048端到端预检**阻断；不是被已取消的HumanEval阻断。

## Table10与验证

DP接口、O(N²)前缀和/回溯、N1…8全连续分区oracle、tie/非法输入测试保留。本次续审CPU全套90项通过，另用旧Smoke真实32层Q/K/V/O缓存RMI重测：每投影Greedy约0.203ms、DP约0.462ms；四投影逐轮求和中位数约0.820ms/1.879ms，1000次、预热20次。λ1/含对角线仅为开发计时约定，未获正式确认。见 [DP_GROUPING_PLAN.md](DP_GROUPING_PLAN.md) 和 [本轮计时](report/math-only-dp-benchmark-20261118.json)。这些不是论文Table10的GLUE/CV跨数据集成绩，最终任务效果仍待对应成员交付。
