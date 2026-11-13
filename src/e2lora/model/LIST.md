# 模型清单

| 模型和版本 | 权重来源 | 用途 |
|---|---|---|
| RoBERTa-base，基础版 | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)，团队小样例使用 `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | GLUE 实验与 SST-2 梯度图 |
| CLIP ViT-B/16，OpenAI 224×224 版 | [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)，`57c216476eefef5ab752ec549e440a49ae4ae5f3` | 图像分类实验与 Cars 梯度图 |
| Llama 3.1 8B Base | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)，可用版本 `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | 数学、代码、常识推理、消融与梯度诊断 |
| Llama 2 13B Base | [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)，可用版本 `5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` | 表 15 |
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)，可用版本 `b968826d9c46dd6066d109eabc6255188de91218` | 图 7、表 15 |
| GPT-OSS-20B，MoE | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)，可用版本 `6cee5e81ee83917806bbde320786a8fb61efebee` | 图 7、表 15 |
| Phi-4-mini，图 7 的候选为 Instruct 版 | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)，`cfbefacb99257ffa30c83adab238a50856ac3083` | 图 7，论文具体权重版本待确认 |
| Gemma 9B IT，图 7 的候选为 Gemma 2 9B IT | [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | 图 7，论文具体权重版本待确认 |

Llama 和 Gemma 的访问以各发布页面的账号及许可要求为准。上表的提交号是团队可以采用的发布版本，论文实际下载的提交号待确认。正式实验配置记录权重仓库和完整提交号。本地离线小样例使用脚本生成的小模型。

本地权重目录是 `src/e2lora/model/weights/组织名/模型名/提交号/`。可以用 `python scripts/download_models.py 组织名/模型名 --revision 提交号` 下载公开权重。加载时传入同一提交号即可优先使用下载完整的本地权重。
