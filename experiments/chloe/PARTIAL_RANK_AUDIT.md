# partial-rank 与论文参数量核查

本轮优先审计见 [TABLE5_PARAMETER_BUDGET_AUDIT.md](TABLE5_PARAMETER_BUDGET_AUDIT.md) 与 [逐组CPU meta计数](report/table5-budget-audit-v1.json)：包含四个投影族全部24个区间、r8/r16的A/B形状、实际唯一可训练计数和舍入差额，并区分物理参数、前向切片及有效坐标。按当前逐族预算解释，公式（7）与Algorithm 3不足以得到3.61M/7.22M；Table 5继续阻断。新的GPU预检已按用户要求暂停，未修改HRA、Rank或冻结规则。

本次采样续审重新运行meta计数，E² r8/r16仍为6,810,624/13,631,488，active139候选仍为3,700,736，见 [续审计数](report/partial-rank-sampling-review.json)。2048两项容量预检没有证明partial-rank策略与论文一致；`paper_parameter_resolution_confirmed=false`继续保留。此次释放张量的修改不改变HRA/冻结/共享政策，不按表值调参。

## 2026-11-18：Math-only重算与论文核对

Code/HumanEval已按组内批准排除；此处只核对Math。新增记录 [math-only-partial-rank-audit-20261118.json](report/math-only-partial-rank-audit-20261118.json) 使用实际Llama meta架构和旧Smoke缓存entropy/RMI，未用GPU或重新采样。旧Smoke统计不能代替正式采样。

| 对照 | 本次唯一可训练参数 | 结论 |
|---|---:|---|
| LoRA r4 / r8 | 3,407,872 / 6,815,744 | 实际安装计数 |
| E²全A/B，缓存RMI，r4/r8/r16 | 3,412,992 / 6,810,624 / 13,631,488 | r8对LoRA4多99.8498%；r16对LoRA8多100% |
| E²全A/B，r32/r64 | 27,276,288 / 54,517,760 | group rounding影响总和；不是GPU显存验收 |
| LSS only，固定每组r8 | 1,253,376 | 与当前缓存组数有关，不能拿来填论文Table4 |
| HRA only，逐层entropy分秩r8 | 6,818,816 | 与独立LoRA8接近，符合总rank基本守恒的计数关系 |
| active139组内A/B候选 | 3,700,736 | 仍需冻结/初始化协议，不自动采用 |

重新核对[官方论文Table4/5及式(5)–(7)](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf#page=9)：Table4的LSS-only/HRA-only/完整E²分别报告3.51M/6.82M/3.61M；替代Cosine/L1/L2为3.75M、KL为3.43M。Table4本身7行，本组另加LoRA参考共8配置。Table5报告LoRA4/8为3.41M/6.82M，E²8/16为3.61M/7.22M。按式(7)逐组round分配N×base_rank且组内A/B都训练时，数学计数几乎守恒；当前实现与约半数参数的表值仍未对上。active139不能由该公式直接推出。

**Table4还有一个具体可实现性问题：** 现有alignment要求非LoRA方法使用完全相同的冻参policy。如果把active139硬套全部方法，HRA-only也会被减到约3.70M，与论文6.82M不同；LSS-only的每投影总rank=组数×base_rank，在当前缓存组下甚至不足139，`active_rank_plan`会拒绝。这说明“同一个active预算字典”不是已证明的论文公平控制。须取得明确作者实现/组内复现规则后，才决定如何修订此门禁与控制；本轮不按表值反推各方法冻结比例、不静默放宽校验。

冻结分支B非零初始化会改变初始函数；设为零又使该冻结分支无前向贡献。各rank active预算、同组A/B共享身份、scaling=alpha/group_rank、完整冻结buffer保存/恢复都必须同时登记。替代相似度参数计数不能从RMI缓存推导，必须拿到同一批真实梯度和确认的指标定义。

以下为既有审计说明，仍保留供对照。

依据：[ICLR 2026官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf)，重点§3.2–3.3、Table4/5、Appendix M；团队 `references/text-generation-setup.json` 和既有参数检查代码。以下是本地实现/计数核查，不是正式复现成绩。

Llama-3.1-8B有32层。Q/O为4096→4096，K/V为4096→1024。同组共享A/B只计一份参数，每组贡献 `r×(d_in+d_out)`；四投影每rank系数合计26624。独立LoRA r8为 `32×8×26624=6,815,744`。

| 配置 | 唯一可训练参数 | 证据与限制 |
|---|---:|---|
| LoRA r4 | 3,407,872 | 实际Llama meta架构安装adapter |
| LoRA r8 | 6,815,744 | 同上，与Smoke相符 |
| LoRA r16 | 13,631,488 | 同上 |
| LoRA r32 | 27,262,976 | 同上 |
| LoRA r64 | 54,525,952 | 同上，未验证GPU显存 |
| Stage A真实24组，全A/B训练 | 6,810,624 | Q/K/O总rank256、V255，与Stage B实测一致 |
| 真实组 + 每投影139活跃rank | 3,700,736 | 团队候选，未批准为Chloe正式策略 |
| 真实组 + 全局冻结A + 每投影139活跃B rank | 1,423,360 | E²ShareLoRA候选，不是Table4/5默认 |

重跑入口为 `audit_partial_rank.py`，完整结果见 [cpu-partial-rank-audit.json](report/cpu-partial-rank-audit.json)。只实例化meta模型，没有加载8B权重或运行GPU。

式(7)分配的共享组rank总和约为N×base_rank；A/B均训练、各组同形状时，对应参数量也几乎不变。Smoke比LoRA少5,120，仅来自V逐组round差一rank，不是约50%节省。Python round在半整数采用ties-to-even，论文未注明破同分。paper策略保留负差分熵比值，可能反向分配；和为零或产生非正秩须中止，不静默平移或补足预算。

团队候选为每投影total256/active139、组内共享A/B、冻结维度有非零前向贡献。它解释了3,700,736计数，但论文正文/Algorithm3没有给139、冻结比例或此初始化规则。3.700736M也不同于论文Table4/5的3.61M。较LoRA r4多8.594%；预算差距容差须事先登记。

正式阻断：

- 为什么名义总rank不变的公式对应约半数可训练参数：须以作者实现或明确的团队复现差异说明解决，不能为凑表中数字反推冻参比例。
- r4/r16/r32/r64活跃预算未定，不能自动推广139/256，也不能固定139。
- LSS only固定组rank、HRA only逐层熵分配时，冻结/初始化须一致，避免同时改变多个因素。
- 既有团队候选使用精确最大余数，论文HRA使用逐组round；新代码仅对活跃rank子预算用最大余数，HRA仍保留paper rounding。
- 冻结B若为零，冻结A/B对forward无贡献；显式非零B初始化改变初始模型输出，与LoRA零更新初始化构成额外因素，其尺度与公平性未确认。
- 全局冻结A的最大rank切片属于E²ShareLoRA，不能混成E²LoRA组内A/B共享。

工程上已验证共享身份、参数去重、预算容量、冻结梯度、非零前向贡献和冻结buffer保存/恢复。GPU Smoke仅验证全A/B共享训练，**新partial-rank候选只有CPU测试，尚未经过GPU验收**。
