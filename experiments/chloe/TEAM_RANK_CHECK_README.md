# 团队部分秩验收（独立入口）

本次从实际 VS Code SSH 工作区读取代码，而不是用较旧的本地包覆盖远端。
远端 HEAD 为 d13dc42784dea9c3114e66f53606874f0ab41df8，另有大量未提交改动。
本次只新增四个验收文件，保留旧 smoke、正式入口与其他成员目录。

## 修正点

- 普通 LoRA r8 的 6,815,744 是正确的全训练 baseline 参数量。
  旧 E2 cached smoke 的 6,810,624 来自全秩训练及整数 round 后预算偏差；不是应报告的团队部分秩结果。
- 直接使用当前公共 collect_family_gradients（内部 collect_gradients）、gradient_similarity、
  greedy_intervals、active_rank_plan、apply_partial_plan、adapter_state/count_adapters。
- 总秩按未平移代理熵比例和最大余数法分配，每种投影跨组严格256；
  活跃秩按组秩比例和最大余数法分配，每种投影跨组严格139。
  exact_ranks 是团队预算整数化接线，不重写公共 E2 算法，也不声称是作者公开实现。
- 采样总数显式为2（仅短跑选择），来自实际128条训练池，Random(42).sample，不放回。
  不随 micro batch 改变。实际 source IDs 保存到 sampling.json。
- 先求每个原始权重的 token 加权平均梯度，再计算总体方差高斯代理熵；
  有效 token 为 labels[:,1:] != -100。采样无优化器更新，dropout关闭。
- A 沿用公共 nn.Linear 初始化；B 活跃部分为0，固定部分 uniform(-0.01,0.01)。
  固定部分是 buffer，参与前向；不是梯度掩码，不受 AdamW 权重衰减影响。
  初始适配器完整状态保存在远端 initial-adapter.pt，断点也保存固定分量。
- 验证真实模型 E2ShareLoRA 参数和 A 跨层共享后恢复原底模，再训练 E2LoRA。
  不把 E2ShareLoRA 的参数安装检查称为它已完成训练。
- 复用 Chloe fit 的 response-only token 加权训练、优化器/调度器及断点恢复；
  复用现有 generate_completion/score_records。两轮间从 checkpoint 恢复，逐轮验证。

## 短跑配置

配置：configs/team-partial-rank-short.json。
128条固定训练样本、seed42、2轮、每轮2步；BF16底模、FP32适配器、SDPA，序列上限2048；
每卡1条，累积64，全局64；AdamW lr5e-5、betas(0.9,0.999)、wd0.0005；
cosine、warmup1步、clip1。首次更新lr0（warmup），之后5e-5、3.75e-5、1.25e-5；
日志 lr_after_update 是下一步学习率，不要将其误称为刚执行一步的lr。
沿用现有 Chloe smoke 的 Question/Answer 提示格式；8条固定 GSM8K、greedy512。
该模板与 GitHub 成员4记录仍需统一，不能声称已通过正式协议对齐。
长度2048是上限，不表示本次每条都恰好2048 tokens；真实原文样本不做合成padding扩长。
短跑不允许截断答案；若现有编码器报告 response_truncated，则立即停止。

## 实际资源与运行

实时查询确认 MGPU-TC2 / stud / normal；normal 上限每用户10CPU、1GPU、30G、6h，
最多2个运行/提交作业。实际 GPU 探测为 NVIDIA A40，47709159424 bytes。
旧脚本申请64G/96G不可照搬。新脚本申请8CPU/1GPU/30G/30分钟。

在远端仓库根目录：

```bash
conda run -n e2lora python experiments/chloe/test_team_rank_check.py
sbatch experiments/chloe/slurm/team-rank-check.sbatch
squeue -u "$USER"
```

当前独立验收作业56314；输出 outputs/chloe/smoke/team-partial-rank-20261120a/。
该输出受到覆盖保护；重跑时复制配置，改为一个新的 output_dir。
原始矩阵、每组层编号/秩、精确参数量、初始权重、逐轮评分与续跑 checkpoint 都保留在远端。
数据、权重、checkpoint 不纳入交付包或Git。尚未提交或推送。

## 正式实验门槛

短跑通过只证明当前独立入口的接线；旧 adaptive_smoke/formal_runner 不会因此自动换冻结策略。
正式入口仍须显式选择各消融方法的预算/冻结规则，确定正式梯度采样总数、Table4基准秩、
Figure5固定值及rank变化时的活跃预算、替代相似度定义，并统一提示/评分协议。
现有 formal-protocol.pending.json 中的 null 保留，没有静默补值，也没有绕过正式门槛。
不根据8条 smoke 分数选超参数或启动全量训练，不将它写入正式报告。
满足实际已确认的完整配置及验证条件后，用户已授权继续，无需再询问一般性许可。
