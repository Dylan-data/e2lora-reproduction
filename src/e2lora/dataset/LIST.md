# 数据集清单

| 数据集 | 版本、子集与来源 | 用途 |
|---|---|---|
| GLUE | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue)，`mnli`、`sst2`、`cola`、`qnli`、`qqp` | s 的表 1 等；dylan 的 SST-2 热图。小样例使用 SST-2 的 train 和 validation |
| Stanford Cars | Stanford Cars 196 类，[公开镜像](https://huggingface.co/datasets/tanganke/stanford_cars) | 图 6、表 3 等 |
| DTD | [官方 DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/)，47 类，记录使用的官方划分编号 | 表 3 等 |
| EuroSAT | [TFDS EuroSAT](https://www.tensorflow.org/datasets/catalog/eurosat)，RGB 版 | 表 3、12 等 |
| GTSRB | [Visual Domain Decathlon 的 GTSRB](https://www.tensorflow.org/datasets/catalog/visual_domain_decathlon#visual_domain_decathlongtsrb)，43 类 | 表 3 等 |
| RESISC45 | [TFDS RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45)，45 类 | 表 3 等 |
| SUN397 | [TFDS SUN397](https://www.tensorflow.org/datasets/catalog/sun397)，397 类，实际使用的划分待登记 | 表 3 等 |
| SVHN | [TFDS SVHN Cropped](https://www.tensorflow.org/datasets/catalog/svhn_cropped)，单数字裁剪版 | 表 3 等 |
| MetaMathQA | [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)，取 100,000 条训练样本，原论文抽样编号待确认 | 数学微调和诊断 |
| Code-Feedback | [m-a-p/Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)，论文使用代码部分的 100,000 条，子集和样本编号待确认 | 代码微调和诊断 |
| GSM8K | [OpenAI GSM8K](https://github.com/openai/grade-school-math)，原版 test | 数学正确率；图 7 单独记录实际诊断划分 |
| HumanEval | [OpenAI HumanEval](https://github.com/openai/human-eval)，原版测试题 | 代码 pass@1 |
| ARC-Challenge、ARC-Easy | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)，`ARC-Challenge` 和 `ARC-Easy` | 表 14 |
| BoolQ | [google/boolq](https://huggingface.co/datasets/google/boolq)，原版 | 表 14 |
| OpenBookQA | [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa)，`main` 子集为候选 | 表 14 |
| PIQA | [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa)，原版 | 表 14 |
| Social IQa（SIQA） | [allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa)，原版 | 表 14 |

表 14 的训练来源和评估划分待确认。正式运行时记录所用数据提交号或发布版本、划分、随机种子和采样编号。使用压缩包时记录文件校验值。图像目录只登记了来源，实际使用的发布版本与训练、评估划分由 kathy 在实验配置中补齐。离线小样例使用脚本生成的数据。
