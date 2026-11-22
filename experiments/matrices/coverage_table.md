| Suite | Model | Tasks | Methods / variants | Configuration |
|---|---|---|---|---|
| glue | RoBERTa-base | MNLI, SST-2, CoLA, QNLI, QQP | FFT, LoRA, DoRA, AdaLoRA, LoRA+, LoRA-FA, VeRA, ShareLoRA, E2LoRA, E2ShareLoRA; rank 4: DoRA, AdaLoRA | references/text-understanding-setup.json |
| vision | CLIP ViT-B/16 | Cars, DTD, EuroSAT, GTSRB, RESISC45, SUN397, SVHN | Zero-shot, FFT, LoRA, DoRA, AdaLoRA, LoRA+, LoRA-FA, VeRA, ShareLoRA, E2LoRA, E2ShareLoRA; rank 4: DoRA, AdaLoRA | references/image-classification-setup.json |
| generation | Llama 3.1 8B Base | GSM8K, HumanEval | FFT, LoRA, DoRA, AdaLoRA, LoRA+, LoRA-FA, VeRA, ShareLoRA, E2LoRA, E2ShareLoRA; rank 4: DoRA, AdaLoRA | references/text-generation-setup.json |
| commonsense | Llama | ARC-C, ARC-E, BoolQ, OBQA, PIQA, SIQA | LoRA, E2LoRA; rank 4:  | experiments/s/configs/table14_commonsense.json |

Seed 42 is the reported profile. Seeds 42/43/44 define a planned extension, not completed repetitions. Rows declare coverage, not observed run status.
