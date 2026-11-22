# 四个任务的配置

| 成员 | 任务 | 可读取的配置 |
|---|---|---|
| 2 | 文本理解 | [text-understanding-setup.json](text-understanding-setup.json) |
| 3 | 图像分类 | [image-classification-setup.json](image-classification-setup.json) |
| 4 | 文本生成 | [text-generation-setup.json](text-generation-setup.json) |
| 5 | 消融对照 | [ablation-setup.json](ablation-setup.json) |

训练脚本可直接用 Python 标准库读取。从仓库根目录运行时，例如：

```python
import json
from pathlib import Path

setup = json.loads(Path("references/text-understanding-setup.json").read_text())
learning_rate = setup["tasks"]["sst2"]["learning_rate"]
```

`null` 表示论文未写明或当前还无法确认的值，由执行成员按实际实验记录。模型提交号来自仓库清单，是团队可用版本。任务内比较不同方法时使用相同的训练样本与评分数据。论文原文与配置表有差异的地方在 JSON 中分别保留。

## 成员之间统一的规则

- 四份配置用 `rank_policy: "paper"`。调用 `build_plan(gradients, families, base_rank=base_rank, rank_policy=setup["rank_policy"])` 时显式传入，按论文式 (5)–(7) 分秩。若组代表值之和接近零或算出非正整数秩，代码会报错。记录样本、组代表值和错误，待团队确定统一处理后再运行这组对照。小模型联调记录中的平移方式属于此前的联调设置。
- 参数量分别记录适配器参数和总可训练参数。共享矩阵只计一次，参与训练的分类头计入总量。
- 分开记录训练前采梯度、分组和分秩的时间，以及正式训练时间和峰值显存。GPU 小时按卡数乘运行小时。
- 成员 4 和 5 若直接比较数学或代码成绩，使用同一评估集版本和评分规则。GSM8K 报正确率，HumanEval 报 pass@1。

成员 4、5 的代码训练设置同时记录了清单中的公开 Code-Feedback 下载版本和论文所述的 100,000 条。公开版本约 66,400 条，实际训练来源的字段留待实验记录。图 7 的梯度使用 GSM8K 训练集，数学成绩使用 GSM8K 测试集。表 10 的惩罚系数和图 5 的两个固定值在论文中未明确，配置使用 null。

文本生成配置是消融共同设置的来源。消融 JSON 已保存同一套完整方法起始设置。文本生成负责人试跑后填写样本编号文件、训练和评分字段，并把配置提交号与评分脚本路径交给成员 5。成员 5 同步这些共用字段，独立确定表 5 和图 5 的秩变化规则与固定值。默认比较种子为 42、43、44。每卡批量与梯度累积可按集群调整，总批量为 64。
