# 数据集清单

| 数据集 | 版本、子集与来源 | 用途 |
|---|---|---|
| GLUE | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) · [固定版本文件](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c)，`mnli`、`sst2`、`cola`、`qnli`、`qqp` | 文本理解路线的表 1 等；dylan 的 SST-2 热图。小样例使用 SST-2 的 train 和 validation |
| Stanford Cars | Stanford Cars 196 类，[公开镜像](https://huggingface.co/datasets/tanganke/stanford_cars) · [固定版本文件](https://huggingface.co/datasets/tanganke/stanford_cars/tree/9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40) | 图 6、表 3 等 |
| DTD | [官方 DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) · [下载 DTD r1.0.1](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)，47 类，记录使用的官方划分编号 | 表 3 等 |
| EuroSAT | [TFDS EuroSAT](https://www.tensorflow.org/datasets/catalog/eurosat) · [公开下载](https://huggingface.co/datasets/tanganke/eurosat/tree/43750fc422c7fbd4289e2df3f7473617a5937acc)，RGB 版 | 表 3、12 等 |
| GTSRB | [Visual Domain Decathlon 的 GTSRB](https://www.tensorflow.org/datasets/catalog/visual_domain_decathlon#visual_domain_decathlongtsrb) · [公开下载](https://huggingface.co/datasets/tanganke/gtsrb/tree/ea889fff70dda7b09e94a4b207dded438deca549)，43 类 | 表 3 等 |
| RESISC45 | [TFDS RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45) · [公开下载](https://huggingface.co/datasets/tanganke/resisc45/tree/5c484982a196dcb501b8e79c6cf4680dc6f2ccdf)，45 类 | 表 3 等 |
| SUN397 | [TFDS SUN397](https://www.tensorflow.org/datasets/catalog/sun397) · [公开下载](https://huggingface.co/datasets/tanganke/sun397/tree/9c4afd6174afa3974f209a0ef099dbbe244dfd27)，397 类，实际使用的划分待登记 | 表 3 等 |
| SVHN | [TFDS SVHN Cropped](https://www.tensorflow.org/datasets/catalog/svhn_cropped) · [裁剪数字下载](https://huggingface.co/datasets/ufldl-stanford/svhn/tree/f9e1717d73324ebbc1ece84267ccd80d0aca0690/cropped_digits)，单数字裁剪版 | 表 3 等 |
| MetaMathQA | [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) · [固定版本文件](https://huggingface.co/datasets/meta-math/MetaMathQA/tree/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18)，取 100,000 条训练样本，原论文抽样编号待确认 | 数学微调和诊断 |
| Code-Feedback | [m-a-p/Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) · [固定版本文件](https://huggingface.co/datasets/m-a-p/Code-Feedback/tree/f411b16a97c910ac9acf8b0d0948e340aa77cc34)，论文描述为 100,000 条代码样本。此公开版本约 66,400 条，正式训练来源和样本编号待核对 | 代码微调和诊断 |
| GSM8K | [OpenAI GSM8K](https://github.com/openai/grade-school-math) · [下载测试集](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl)，原版 test | 数学正确率；图 7 单独记录实际诊断划分 |
| HumanEval | [OpenAI HumanEval](https://github.com/openai/human-eval) · [下载测试题](https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz)，原版测试题 | 代码 pass@1 |
| ARC-Challenge、ARC-Easy | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) · [固定版本文件](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453)，`ARC-Challenge` 和 `ARC-Easy` | 表 14 |
| BoolQ | [google/boolq](https://huggingface.co/datasets/google/boolq) · [固定版本文件](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5)，原版 | 表 14 |
| OpenBookQA | [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) · [固定版本文件](https://huggingface.co/datasets/allenai/openbookqa/tree/388097ea7776314e93a529163e0fea805b8a6454)，`main` 子集为候选 | 表 14 |
| PIQA | [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) · [训练与验证](https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip) · [测试](https://yonatanbisk.com/piqa/data/tests.jsonl) | 表 14 |
| Social IQa（SIQA） | [allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa) · [训练与验证](https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip) | 表 14 |

表 14 的训练来源和评估划分待确认。正式运行时记录所用数据提交号或发布版本、划分、随机种子和采样编号。使用压缩包时记录文件校验值。图像目录只登记了来源，实际使用的发布版本与训练、评估划分在交接配置中补齐。离线小样例使用脚本生成的数据。

本地原始数据位于 `src/e2lora/dataset/data/`，该目录由 Git 忽略。Hugging Face 下载按数据集名称与提交号保存。EuroSAT、GTSRB、RESISC45、SUN397 使用 tanganke 的公开分发版本，SVHN 使用 ufldl-stanford 的裁剪数字子集。正式实验前按论文口径确认图像划分。
