> 2026-11-18范围决定：HumanEval因隔离执行环境限制，经组内批准不纳入本次复现范围；Code-Feedback同步排除。以下Code计划只保留历史框架，当前不安排下载、交接、Smoke或正式GPU作业，也不阻断Math。活动范围见 [FORMAL_ABLATION_READINESS.md](FORMAL_ABLATION_READINESS.md)。

> 历史记录：E²LoRA Stage B 已通过 Job54466（COMPLETED/0:0，20步，第10步恢复，12/32=37.5%）。LoRA Job54429亦通过；均不可进入论文正式表图。下文保留当时的准备状态。

# Code 路线与文本生成成员的协议交接

状态：待共同确认。Code-Feedback 尚未下载；本轮未下载、未安装 HumanEval、未启动代码训练。仓库现有资料不足以认定训练/评分协议已经统一。请文本生成成员在同一份协议中确认下表，并登记确认人、版本及时间；可将最终协议放入团队公共目录，Chloe 随后引用该文件。

## 两条独立训练路线

| 路线 | 训练 | 评估 | 指标 | 实验范围 |
|---|---|---|---|---|
| Math | 固定 MetaMathQA 100K 清单 | GSM8K main/test，1319 题 | Accuracy | Table 4、Table 5、Figure 5 两项 Sweep |
| Code | Code-Feedback，版本/实际训练数量按统一协议 | HumanEval，任务集及执行器版本待统一 | Pass@1 | Figure 5 两项 Sweep；独立代码训练 checkpoint |

Figure 5 Rank Sweep：两条路线分别覆盖 `[4,8,16,32,64]`。Sampling Steps Sweep：分别覆盖 `[8,16,32,64,128,256,512]`。每条路线内部同一种子使用同一采样序列前 N 个逻辑步骤。Math checkpoint 不能替代 Code checkpoint，GSM8K 指标不能填入 HumanEval 栏。

## 公共 JSON 已核对的内容

`references/text-generation-setup.json` 与 `references/ablation-setup.json` 的 `tasks.code` 内容完全一致：

- 公开源：`m-a-p/Code-Feedback`，revision `f411b16a97c910ac9acf8b0d0948e340aa77cc34`，split=train。
- 公开数据约 66400 行；论文样本数是 100000；两份 `training_source` 都是 null。
- 评估数据 HumanEval，指标 pass@1。

“固定公开版本”不等于已经决定正式训练来源、样本数量或每行的处理方式。Dylan 的两样本参数检查不是统一 SFT/评分协议，其 JSON fallback 也不能直接当作正式代码训练 Prompt。

## 必须回填的协议

| 协议项 | 现状 | 两条生成实验路线共同确认的内容 |
|---|---|---|
| 训练来源 | null | 仓库/revision/具体文件及 SHA256；公开版是否为最终来源 |
| 样本数量 | 66400 行 vs 论文 100K | 实际行数、code-only 过滤、有效训练实例数；多轮是否拆分；去重/重复采样规则及与论文差异 |
| 样本清单 | 未提供 | 固定原始行号/轮次号、顺序、数据种子、清单 SHA256；训练种子另列 |
| Prompt 与 loss | 未提供 | Base 模型模板、角色分隔、代码围栏、多轮上下文、EOS、仅响应监督或全文监督 |
| 训练设置 | 公共优化器设置已有 | epochs/steps、长度、截断/packing、每卡 batch/accumulation、alpha、共享/冻结策略 |
| 生成设置 | 未提供 | do_sample、temperature、top_p/top_k、每题样本数、max_new_tokens、stop/EOS、随机种子 |
| HumanEval 任务 | 仅名称 | 数据仓库/commit、文件 SHA256、题目数与 task_id 清单；是否使用原始函数签名 Prompt |
| Completion 后处理 | 未提供 | 是否去代码围栏、是否剥离重复 Prompt、缩进、stop 截断与解析失败规则 |
| Pass@1 | 仅指标名 | 每题样本数量、估计公式、汇总分母；不能混用单样本正确率和未注明的多样本估计 |
| 执行器 | 未安装 human-eval；无团队实现 | 实现/commit、Python 与依赖版本、隔离环境、超时、失败分类及逐题日志 |
| Figure 5 固定设置 | 仍为 null | Rank Sweep 固定采样步数；Step Sweep 固定 base rank；随 r 改变的总秩和 active rank 规则 |

机器可读占位：[code-protocol.pending.json](configs/code-protocol.pending.json)。不能只把 status 改成 confirmed：其 pending 字段、确认文档和执行器都需要实际交付；当前代码实现即使看到 confirmed 仍明确拒绝未实现的 Code 训练路径。

## 环境、数据、训练与评估准备顺序

1. 对齐上述协议并冻结版本。在确认来源后，登记实际下载路径；只下载缺失的 Code-Feedback/HumanEval，不重复下载模型或 Math 数据。
2. 下载后验证 revision、文件 SHA256、原始行数、字段、code-only 过滤及多轮处理统计，产出可复用清单。若最终训练量不足 100K，如实记录，不默认重复补齐。
3. 在 e2lora 环境中接入团队 HumanEval 执行器，锁定依赖和任务集版本；先用已知正确/错误 completion 验收执行与计分。
4. 开发 Code 数据适配、响应 mask、独立训练输出与 generation/evaluation 接口，先做 CPU/小模型测试。
5. 申请独立 Code Smoke 提交批准：最多 20-step，验证数据处理、checkpoint 恢复、少量任务生成/执行、峰值显存和退出码。所有结果标 smoke。
6. 共享协议与 Smoke 都通过且用户批准后，再分别运行两项 Sweep 的 Code 正式训练，三次独立种子；输出 `outputs/chloe/code/<run_id>/`。
7. Figure 5 汇总同时要求 Math Accuracy 和 Code Pass@1。Code 尚未完成时显示 pending/null，不隐藏 HumanEval 曲线、不用 0 或数学分数填充。

Math Baseline 开发和本地验证不依赖这些 Code 待确认项，因此继续推进。下载完成、协议统一、代码实现和正式训练批准是不同的状态，不能互相替代。
