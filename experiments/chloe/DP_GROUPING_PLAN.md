# Table 10：DP实现、穷举验证与耗时

## 2026-11-18 更新

Math-only范围调整不取消Table10的算法、正确性和计时交付。本轮CPU全套77项通过，包含N1…8独立穷举oracle及边界/tie/非法输入。用旧Math Smoke的真实32层Q/K/V/O缓存RMI，在相同输入上各预热20次、重复1000次：

| 输入 | Greedy中位数ms | DP中位数ms |
|---|---:|---:|
| Q | 0.20358 | 0.46329 |
| K | 0.20363 | 0.46329 |
| V | 0.20334 | 0.46198 |
| O | 0.20316 | 0.46207 |
| 每轮四投影kernel时间相加后取中位数 | 0.82008 | 1.87910 |

记录见 [math-only-dp-benchmark-20261118.json](report/math-only-dp-benchmark-20261118.json)。aggregate是每轮四次搜索耗时之和的分布，并非将四个median直接相加。λ1、include_diagonal只是开发约定，不是新增的正式选择。新计时不覆盖旧记录。

官方[Table10](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf#page=23)在CLIP-ViT/CV和RoBERTa/GLUE上汇总跨数据集搜索时间、参数和任务成绩；这里的Llama缓存RMI是额外CPU方法验收，不能替代那两条任务结果。还需对应成员确认λ/均值/singleton和跨任务聚合方式，按同一训练设置给出真实任务性能；本成员不新增该部分GPU作业。

以下为原有实现与开发基准说明。

公共入口为 `e2lora.adapters.grouping.search_intervals`，Greedy沿用原anchor/阈值策略，DP最小化 `sum(-mean(segment))+lambda*K`，二维前缀和、O(N²)搜索与前驱回溯。区间为连续半开区间。

依据：[论文Appendix G](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf#page=22)。λ和均值细节未给定，必须显式传入有限非负penalty_lambda与mean_policy（include_diagonal/exclude_diagonal）。排除对角线时singleton_cost也必须给定；包含时单层代价由矩阵对角值决定。

目标完全相同时先较少组，再较小前驱，递归保持同一规则；不以近似相等容差接受更差目标。输入须非空、方形、有限、对称。

CPU测试独立枚举N=1…8全部2^(N−1)个连续分区，直接逐元素求均值，不复用DP前缀和。覆盖4个λ、含/不含对角线、两种singleton代价、负相似度、不均匀矩阵、全等/tie、单层和非法输入；核验最优目标、实际分区目标及完整无重叠覆盖。

开发benchmark采用相同seed101 synthetic block矩阵，λ=1/含对角线，预热20次、重复200次，Python单线程。中位数：

| 层数 | Greedy ms | DP ms |
|---|---:|---:|
| 12 | 0.03855 | 0.07894 |
| 32 | 0.20450 | 0.46377 |

CPU型号、Python/平台、矩阵摘要及min/median/mean/p95/max见 [cpu-grouping-benchmark.json](report/cpu-grouping-benchmark.json)。计时含输入校验、阈值或DP前缀和、搜索和回溯，排除梯度/RMI/分秩/安装/加载/训练。**这是开发synthetic耗时，不是Table10复现结果。**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python experiments/chloe/benchmark_grouping.py --penalty-lambda 1 --mean-policy include_diagonal
```

通过 `--input <json>` 可读取实际缓存RMI，JSON为 `{family_name: square_matrix}`。成员2/3须先确认λ、mean/singleton、跨投影及数据集时间聚合方式；λ只用预先登记的训练诊断选择，不能看最终测试成绩调参。最终GLUE/CV真实参数、训练与评分归对应成员负责，本次未运行。
