# Chloe / 成员5：数学消融代码

本次交付为代码补齐，不启动新GPU作业。源码准备阶段未commit/push；后续按成员5明确授权写回其分支并上传GitHub。当前已批准范围：Table4/5数学、Figure5数学秩/采样步数、Table10 DP分组。Code-Feedback/HumanEval维持组内排除，不声称完成代码任务曲线。旧pending配置、审查文档和formal_runner.py是历史方案，新批次统一使用suite入口。

## 文件与入口

|用途|文件|
|---|---|
|固定100K训练编号|prepare_data.py|
|已完成LoRA的原始入口，保留不变|formal_math.py|
|八种方法、已配置任意秩、逐轮验证、续跑|suite_train.py|
|方法开关、采样、整数预算|suite_plan.py，复用src/e2lora/adapters|
|JSON编译、短跑、提交、续跑、状态|run_ablations.py|
|可选单GPU串行自动续跑|watch_suite.py|
|GSM8K生成与精确数值评分|evaluate.py公共函数；evaluate_suite.py独立加载suite checkpoint|
|表4/5、Figure5、逐轮曲线与预算差异|summarize.py|
|集群查询、GPU训练|slurm/inspect_cluster.sh、slurm/train.sbatch|
|DP/greedy正确性及耗时|benchmark_grouping.py、公共grouping.py|
|源码导出与大文件/凭证检测|export_code.py|

旧train.py/evaluate.py CLI保留历史smoke格式，不能直接接收suite JSON。统一入口是run_ablations.py。

## 环境与统一协议

已运行环境：Linux / SLURM / A40 48GB，PyTorch2.7.1+cu128。Python>=3.10；CPU测试版本见交付验证记录。先激活现有CUDA环境，不要安装CPU版torch替换它。

```bash
python -m pip install -e .
python -m pip install -r experiments/chloe/requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES='' python -m pytest experiments/chloe/tests experiments/chloe/test_auto_resume.py experiments/chloe/test_formal_math.py experiments/chloe/test_team_rank_check.py -q
```

模型Llama3.1-8B Base revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b；MetaMathQA revision aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18；GSM8K main/test revision740312add88f781978c0658806c59bc2815b9866。完整配置在configs/suite-math.json。换机器只改本地模型/数据/parquet路径。

统一训练：固定100K训练、额外2K不相交验证、1epoch、长度2048、response-only有效shift后token加权、microbatch1×累积64、BF16基座/FP32适配器、AdamW LR5e-5/betas.9/.999/weight_decay.0005、无裁剪、cosine最低0、floor(total*.03) warmup、alpha16/dropout0、q/k/v/o。训练prompt=query.strip()+换行；只左截提示，答案/EOS保留，答案过长报错。

评估：完整1319 GSM8K test、Question: {question}换行Answer:、greedy512新token、精确Fraction评分；逐条有序hash对齐成员4实际manifest。空输出/提取失败不从分母移除，重复行单列记录。每轮保存held-out token loss、GSM8K分数和生成例子。不按test分数选epoch。

## 空字段与团队选择

compile逐项列出references/ablation-setup.json的null，不静默兜底。新配置允许四个有明确含义的null：data.train_limit/evaluation.limit=完整集合；training.max_steps=完整epoch；training.max_grad_norm=不裁剪。其他null直接报错。

configs/suite-math.json中的新增值是**显式待审阅团队方案**，不是作者公开实现：Figure5固定64采样步/rank8、每逻辑步1样本；r8每投影跨组总预算256、可训练预算139；其他rank采用显式32*r总预算和139/256活跃比例。活跃总预算half-up舍入，组内比例使用公共最大余数分配。HRA关闭时每组固定rank，投影总秩随组数变化。所有自适应方法用相同部分冻结比例，普通LoRA全部训练。可显式切换suite.adapter_policy=all_trainable，但须使用新批次/输出目录，不混合协议。

替代相似度约定、精确预算修正、部分冻结均为团队复现选择，不为凑论文参数量改变秩。打印每投影分组层号、总秩/活跃秩、去重参数、其他可训练参数。共享身份由公共count_adapters核对，固定分量为buffer而非梯度掩码Parameter，AdamW不修改它们，首次更新及各checkpoint边界检查。

五项历史审计依赖学校本地模型header/旧smoke文件，普通CPU测试显式skip；在学校仓库加 --cluster-assets 运行，不把skip声称通过。

仅选择seed42；43/44不是论文指定，不会自动追加。可显式增加seeds列表，每次独立模型/优化器/目录。reference的三重复目标不等于已经完成，单种子结果标n=1。

## 命令

从仓库根目录运行，先修改本地路径：

```bash
python experiments/chloe/prepare_data.py --source /path/to/MetaMathQA-395K.json --output outputs/chloe/data/math-100k-seed42.json --count 100000 --expected-rows 395000 --seed 42 --revision aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18
# 只编译，不提交。重叠表/曲线点只训练一次。
python experiments/chloe/run_ablations.py compile
python experiments/chloe/run_ablations.py status
# 查询后核对configs/slurm-resources.json。
bash experiments/chloe/slurm/inspect_cluster.sh
mkdir -p outputs/chloe/slurm
# 激活团队CUDA环境后提交单个配置，短跑通过后完整训练。
python experiments/chloe/run_ablations.py submit --index 0
# 仅在GPU allocation内运行，不在login节点执行。
python experiments/chloe/run_ablations.py run --index 0 --smoke-only
python experiments/chloe/run_ablations.py resume --index 0
# 仅评估某轮实际保存的checkpoint，不更新模型；已有评分直接返回。
python experiments/chloe/evaluate_suite.py --run-dir outputs/chloe/suite-v2/formal/lora-r8-n0/seed-42 --epoch 1
# 可选自动续跑控制器，只有执行此命令才会提交后续作业。
tmux new -s chloe-suite
python experiments/chloe/watch_suite.py --indices 0 1 2 --max-resumes 20
python experiments/chloe/summarize.py --roots outputs/chloe/suite-v2 outputs/chloe/ablations-math-v1 --output outputs/chloe/summary
# 输入为family->matrix，synthetic耗时不能填论文正式结果。
python experiments/chloe/benchmark_grouping.py --input outputs/chloe/similarity-matrices.json --penalty-lambda 1 --mean-policy include_diagonal --warmup 20 --repeats 200
python experiments/chloe/export_code.py --output outputs/chloe/review/chloe-source.zip
```

每份编译JSON记录方法/秩/采样步/种子/归属表图及hash。采样从正式训练池同一随机排列取前缀，无放回且嵌套；数量=逻辑步×显式每步样本数，不随显存microbatch变化。原始基座关闭dropout、不更新权重，collect_gradients按有效shift后token加权，先平均梯度再求总体方差代理熵。每次只缓存一投影的CPU均值梯度。初始化前重置独立随机流。

续跑保存全部适配器/固定buffers、optimizer/scheduler、RNG和step，checkpoint原子替换。配置及相关源码变化拒绝续跑；有checkpoint时不重采梯度。每轮验证与逐条预测可接续。USR1安全边界退出75。梯度准备阶段尚无checkpoint，超时需检查后重新准备，不声称任意backward中间可恢复。控制器只自动重试有进展的TIMEOUT/exit75；OOM、配置错误、CANCELLED、无进展保留attention状态停止。控制器需保持tmux进程，已提交作业由SLURM管理。

## 已有正式结果

|方法|seed|可训练参数|GSM8K test|
|---|---:|---:|---:|
|LoRA r4|42|3,407,872|911/1319=69.07%|
|LoRA r8|42|6,815,744|898/1319=68.08%|

均完整100K/1epoch/1563更新，属保留的formal_math.py协议。没有自动标成新suite协议，也没有补写Table4、E2r16或Figure5未跑分数。summarize保留不同protocol hash，输出真实预算差，未完成保持pending。短跑与正式分开，首次正式填报告仍沿用团队核对流程。

## 上传与分支

不要pull覆盖dirty分支。先备份当前未提交源码，比较最新main，再在自己的分支合并。交付UPLOAD_REVIEW.md/manifest列出共享改动，只stage源码清单，不要git add .。不提交outputs、模型、数据、Token、pt/pth/ckpt/safetensors、缓存或日志。本包不含其他成员实验目录。合并后重跑CPU测试、检查diff，再由成员5决定commit/push。历史审查报告保留作追溯，过时pending结论以本README和新配置为准。
