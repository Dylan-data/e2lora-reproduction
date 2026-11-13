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
