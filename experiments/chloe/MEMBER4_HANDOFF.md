# 与组员4的正式协议交接

## 当前只交接Math（2026-11-18）

HumanEval因隔离执行环境限制，经组内批准不纳入本次复现范围，Code-Feedback同步排除。下方旧表中的Code/HumanEval条目仅保留框架文档，不再要求补齐，也不阻断Math。

已知Table2 Math设置为seed42、epochs1、2048、生成512，见 [确认记录](configs/member4-table2-confirmed.json)。尚需以下Math实物，不能只给README示例：

1. 实际run_config、入口及commit：模型/tokenizer revision、micro/accum/global64、optimizer/betas/eps/decay、alpha/dropout/初始化/scaling、base/adapter精度、gradient checkpoint/attention backend、clip、scheduler首更新LR及warmup取整、shuffle/恢复、best/final选择/频率；另两颗训练seed可后续确认。
2. 实际100K有序样本清单、源文件SHA/行顺序、原始ID/规范化行映射和行hash；2K或其他选择验证集的正式决定及不重叠清单。本地候选manifest不自动等于成员4实物。
3. train/eval模板分别逐字交付：换行/空白、BOS/EOS、few-shot、max_prompt_length、mask、答案保留/截断、packing、规范化；不能用Smoke模板代替。
4. GSM8K固定1319题ID/内容校验及与本地HF版本的等价性；完整generation config及EOS/stop、512上限触顶统计；实际numeric scorer/版本/SHA、无法解析与分数/科学记数等正反例、逐题记录。
5. 接受Chloe compiled config、checkpoint和output的独立driver，使用完整plan及complete_adapter_state恢复，返回config_checkpoint_output_v1字段；环境锁、正确/错误样例验收及脚本SHA。
6. Table4 base rank/主采样steps/替代指标定义/统一控制；Table5各rank真实参数政策、冻结初始化及容差；Figure5两个固定值、采样batch/微批/seed/顺序/nested subset/是否复用统计。

只确认四项训练值尚不足以确认上述内容。P的status和双方完整confirmation继续pending；新formal_preflight还需在正式2048/实际adapter/评分接口下验收。当前代码与成员4存在的shuffle、warmup、截断、clip和checkpoint差异见 [审查报告](MEMBER4_ALIGNMENT_REPORT.md)。

## 既有通用框架与历史双任务schema说明

机器入口：[formal-protocol.pending.json](configs/formal-protocol.pending.json)。Math/Code都需交接，不以Smoke设置填补空缺；当前不自动联络组员、下载数据或安装评分器。

| 接收物 | 必须包含 |
|---|---|
| 训练配置 | 模型路径/revision，epochs/max_steps，3个训练种子，序列/Prompt长度，batch/accumulation，optimizer/LR/scheduler/warmup/clipping，alpha/dropout/scaling/初始化，精度/checkpoint频率 |
| 数据与清单 | 来源/revision、SHA256、有序原始样本/多轮ID、数量、清单SHA256、过滤/去重/多轮展开/重复采样说明；Math100K与Code实际数量分别确认 |
| 预处理/Prompt | 规范化query/response，Base模板、角色/代码围栏、EOS、监督mask、截断、packing、few-shot内容顺序、评估Prompt |
| 梯度采样 | logical steps、每步batch、微批、顺序与seed策略、loss权重、Figure5两个固定值、是否复用统计 |
| 共享/冻参/相似度 | Table4统一控制、各rank活跃预算、冻结分支初始化、paper参数差异证据、所有替代指标的定义/归一化 |
| 生成参数 | do_sample、temperature、top_p/top_k、beams、max_new_tokens、stop/EOS、生成seed、每题样本数 |
| GSM8K评分包 | revision、完整1319题ID、数值抽取/归一化/无法解析规则、评分脚本SHA256和版本、逐题输出、正确错误样例验收 |
| HumanEval评分包 | commit、task_id、prompt/completion抽取、每题样本数、pass@1估计/分母、Python/依赖锁、执行器版本、超时/隔离、正确错误样例及日志 |
| 确认记录 | 双方标识/时间/文档版本、Code独立Smoke报告及SHA256、Table5允许的预算差距 |

当前代码支持：单GPU，BF16 base/FP32 adapter，AdamW/cosine，alpha/group_rank，A Kaiming/可训练B为零，response-only token-weighted loss，不packing，保留BOS及Prompt尾部、响应右截断，先加EOS再截断。这是**支持范围，不是组员已同意的设置**。协议不同则先实现并测试，不能要求组员为适配代码改变实验。

`normalized_query_response_json` 输入是已确认预处理器输出的JSON数组，每行有query/response字符串。Manifest字段为 `source_revision/source_sha256/source_rows/indices/indices_sha256/count`，行号针对该规范化文件；另交付原始行/轮次到规范化行号的追溯记录。Code原始多轮转换规则尚未猜测。

空值的明确含义：max_steps=null表示按确认epochs训练；seed_policy=training_seed时fixed_seed=null；greedy时temperature/top_p/top_k可为null；stop_sequences=[]表示无额外stop。其他null阻断。旧code-protocol.pending.json仅为早期提纲，正式入口使用新schema。

评分接口 `config_checkpoint_output_v1` 的argv由组员提供，以参数列表执行、不经过shell。可用占位符为 `{python}/{script}/{config}/{checkpoint}/{output}`。评分器从compiled config的resolved_protocol读取生成设置；通过 `formal_runner.install` 和checkpoint中的plan重建adapter，再以 `train.load_complete_adapter_state` 恢复complete_adapter_state，包含冻结分支。旧evaluate.py仍是Smoke临时规则，不自动升级为正式评分器。

输出至少包含：run_type=formal、task、protocol_sha256、checkpoint_sha256、task_ids_sha256、evaluation_examples、metric和0…1的score，另保存逐题记录。Math metric=accuracy，Code metric=pass@1；task_ids_sha256为协议中题ID文件的SHA256。完整任务覆盖及HumanEval隔离执行由确认评分器负责。

```bash
PYTHONPATH=src python experiments/chloe/alignment.py --task math --protocol <chloe-protocol.json> --compare <member4-protocol.json>
PYTHONPATH=src python experiments/chloe/alignment.py --task code --protocol <chloe-protocol.json> --compare <member4-protocol.json>
```

双方文件须同schema；逐字段比较含新增/缺失字段，默认不忽略差异，并验证实际文件SHA256。协议改变须版本化并使用新输出目录；编译配置绑定协议摘要，恢复拒绝配置漂移。
