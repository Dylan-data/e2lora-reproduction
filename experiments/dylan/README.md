# dylan · 共用算法与诊断

`smoke.py` 可离线检查普通 LoRA、手写共享和参数计数。后续在此放梯度热图、代理熵变化和开销测量的脚本。供团队调用的算法写在 `src/e2lora/adapters/`，结果放 `outputs/dylan/`。

## 成员 1 的离线验证

在仓库根目录安装项目后运行 `python experiments/dylan/preflight.py`。脚本用人工样本训练四层小模型，生成 `outputs/dylan/adaptive-preflight-*/result.json` 和 `rmi.svg`。JSON 包含梯度采集耗时、代理熵、层间相似度、阈值、连续分组、秩和可训练参数量。SVG 是同一轮的相似度热图。结果用于验证代码和产物格式。

### 正式实验怎么接入

1. 从正式训练划分抽样，准备能返回平均训练损失和有效样本数或 token 数的函数。调用 `collect_gradients` 获取指定层的平均梯度。
2. 将同一类型的投影按深度从浅到深传给 `build_plan`。例如一组只放 query 投影，另一组只放 value 投影。保存返回的分组方案和诊断数据。
3. 调用 `apply_plan` 把分组方案应用到模型。组内共享两个小矩阵。设置 `global_down=True` 时，同类投影跨组共享 A，各组取对应秩的切片。训练和评分由任务侧调用。
4. 正式运行记录权重版本、数据划分与样本编号、目标层、种子、参数统计和准备耗时。图 1、3、6–8 的真实梯度诊断，以及图 4、9–13 的训练过程曲线，等对应任务输入备齐后再运行。

### 实现口径

代理熵使用总体标准差。计算两层拼接后的方差时使用统计量，避免为每一对层复制大张量。RMI 按论文公式计算，保留原始数值。零标准差和接近零的分母用 epsilon 处理。梯度采集使用评估模式关闭随机失活，以 CPU FP32 累加各批梯度。当前仍会为一次反向计算同时生成所有目标层梯度，正式大模型需实测峰值显存。

差分熵可以为负。离线验证将组分数平移到正数再分秩，最小秩为 1。可选 `rank_policy="paper"` 直接使用论文公式；遇到零分母或非正秩会报错，避免静默更换口径。整数舍入后记录实际秩总和。按式 (7) 分配时，参数量下降幅度取决于共享方式和实际秩，离线结果不作为论文参数降幅的依据。

## 交接参考

统一设置见[共用配置](../../references/README.md)。本目录的 [result.json](result.json) 和 [rmi.svg](rmi.svg) 是四层小模型的检查结果及热图。[glue-pilot.csv](glue-pilot.csv) 记录真实 GLUE 小样本联调的分数和参数量。SST-2 的普通 LoRA 有 294,912 个适配器参数，当前 E²LoRA 有 296,448 个，正式对比需核对秩预算。

## tc2 实验脚本

| 脚本 | 用途 |
|---|---|
| `train_glue.py`、`glue_pilots_tc2.sh` | 用 `--pilot` 做 GLUE 小样本训练和参数检查 |
| `sst2_rank52.py`、`sst2_partial_rank.py` | 分别验证直接降秩、保留原秩但只更新部分分量 |
| `generation_parameter_check.py` | 用少量数学和代码训练样本验证两个 E² 方法的参数量 |
| `glue_full.py`、`glue_full_tc2.sh` | 跑完整 GLUE 训练。每轮保存进度，同一任务重新提交时从已保存进度接着跑 |
| `glue_full_runs.txt` | 当前优先运行的组合。已完成的 SST-2 LoRA 和 E²LoRA 使用作业 53975 的结果 |

从仓库根目录运行。tc2 脚本使用 `.cache/glue-env/bin/python`，模型和数据路径由 `references/text-understanding-setup.json` 指定。完整训练的日志、进度和评分分别写入 `outputs/dylan/glue-full-seed42/`。旧的 SST-2 候选脚本分别通过 `RANK52_OUTPUT` 和 `PARTIAL_RANK_OUTPUT` 指定结果目录，后者还需要小样本结果中的 `plan` 和 `diagnostics`，保存为该目录的 `pilot-plan.json`。
