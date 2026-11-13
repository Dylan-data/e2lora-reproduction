# E²LoRA 论文复现

五个人使用同一套模型加载、数据读取和训练代码。各自在自己的目录提交实验脚本和配置。

## 每个人的代码和结果放哪里

| 成员 | 实验规划（仅 dylan 已分配） | 脚本和配置 | 运行结果 |
|---|---|---|---|
| dylan | 共用算法，梯度热图、训练中代理熵、准备开销。图 1、3、4、6–13，§5 和附录 F | `experiments/dylan/` | `outputs/dylan/` |
| s | GLUE 文本理解。表 1 和表 6、10、13 的文本部分 | `experiments/s/` | `outputs/s/` |
| kathy | 图像分类、噪声和种子实验。表 3、11、12 和表 6、10、13 的图像部分 | `experiments/kathy/` | `outputs/kathy/` |
| chloe | 数学、代码、常识推理和扩展模型。表 2、14、15 | `experiments/chloe/` | `outputs/chloe/` |
| acheng | 只分组、只分秩、换相似度、参数预算和采样对照。表 4、5 和图 5。提供表 10 的动态规划分组代码 | `experiments/acheng/` | `outputs/acheng/` |

表 10 的文本和图像部分按实验规划分别填报。

## 哪些代码供大家共用

- [model](src/e2lora/model/) 加载预训练模型和对应的输入处理工具。参见[模型和版本清单](src/e2lora/model/LIST.md)。
- [dataset](src/e2lora/dataset/) 读取数据划分，按种子抽样。参见[数据集和版本清单](src/e2lora/dataset/LIST.md)。
- [tasks](src/e2lora/tasks/) 计算训练误差和任务分数。
- [adapters](src/e2lora/adapters/) 将 LoRA 小矩阵接到模型上。分组方案列出每组的层和秩。普通 LoRA 每层一组，联调可以手写共享组。
- [runner](src/e2lora/runner/) 执行分类训练、评分，并把结果写入独立的运行目录。

新模型加载逻辑、数据读取和评分方法放到共用目录。论文表图对应的实验配置留在各成员目录。当前已接通分类流程和手写共享。正式 E²LoRA 的梯度、代理熵、自动分组、秩分配，以及完整的图像与生成实验由对应成员接入。

## 怎么跑

在仓库根目录执行。建议用 Python 3.10 或更新版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python experiments/dylan/smoke.py --method lora
python experiments/dylan/smoke.py --method manual-sharing
```

这两条运行命令用 CPU 和小型人工数据，检查训练、组内共享以及参数计数。终端会显示写出的 `result.json` 路径。

模型权重放在 `src/e2lora/model/weights/`，数据缓存和运行结果保存在本地。成员目录用于后续分配，当前仅 dylan 的任务已分配。

## 怎么找结果

运行结果按 `outputs/名字/本次运行/result.json` 保存。正式脚本要记录对应的论文表图、模型和数据版本、划分、种子、方法、可训练参数量、分数和耗时。例如 acheng 的表 4 实验写入 `outputs/acheng/`，用结果里的 GSM8K 正确率和参数量填写 Word 报告。
