# 成员5实验计划：Math-only

更新：2026-11-18。以 [FORMAL_ABLATION_READINESS.md](FORMAL_ABLATION_READINESS.md) 为当前执行准备状态。

**HumanEval 因隔离执行环境限制，经组内批准不纳入本次复现范围。** Code-Feedback同步排除；保留已有Code框架、pending资料和12份标记禁用的配置，不提交Code GPU作业。

成员4 Table2 Math最新确认seed42、epochs1、max_seq_length2048、max_new_tokens512。其他两表配置仍待最终确认。不能将这些四项等同于完整Math交接；额外重复seed、训练/评分及消融机制未确认项保持pending。

| 交付 | 内容 | 活动配置/计划重复 |
|---|---|---:|
| Table4 | Math：LoRA参考、LSS/HRA、Cosine/L1/L2/KL、完整E² | 8 / 24 |
| Table5 | Math：LoRA4↔E²8、LoRA8↔E²16，实际唯一参数预算检查 | 4 / 12 |
| Figure5 rank | MetaMathQA→GSM8K，r4/8/16/32/64 | 5 / 15 |
| Figure5 sampling | 同一路线，steps8/16/32/64/128/256/512 | 7 / 21 |
| Table10 | DP、穷举正确性、同输入Greedy计时与任务成员交接 | CPU，不计入72次 |

合计24个Math配置，按既有三重复要求计划72次；seed42已知，另两颗不猜测。支持先编译已确认seed，未确认协议不能运行。当前无正式结果，不提前去重跨表设置。

执行顺序：Math交接/规则统一 → CPU核验与预算解决 → A40 2048容量预检 → 同协议真实数据/恢复/评分小验收与资源标定 → seed42首组正式对照 → 其余机制与扫描 → 补齐确认种子/均值与样本标准差。

资源方案、完整剩余字段与可审阅命令见 [GPU_PREFLIGHT.md](GPU_PREFLIGHT.md)、[MEMBER4_HANDOFF.md](MEMBER4_HANDOFF.md)、[资源报告](report/math-only-resource-plan.json)。历史1024长度、双任务108次及43/44种子提案已被当前方案取代，不能作为开跑配置。只有seed42/epoch1/2048/512的新确认值已记录，其他空值不借Smoke填补。

真实参数核验与论文Table4/5冲突见 [PARTIAL_RANK_AUDIT.md](PARTIAL_RANK_AUDIT.md)。Table10开发计时不填论文GLUE/CV成绩。旧Smoke结果及原始输出保持不动；本轮无GPU训练、无Push。
