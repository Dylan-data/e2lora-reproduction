# QNLI 和 SST-2 基线迁移包

六组方法见 methods.txt。已有 SST-2 结果位于 outputs/dylan，可用来核对。十二组迁移任务从第 0 轮开始，训练脚本在每轮末保存可续跑状态。

```bash
sha256sum -c checksums.sha256
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
export PYTHON_BIN="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/src"
"$PYTHON_BIN" experiments/dylan/glue_full.py --task sst2 --method dora --pilot --device cuda:0 --output-root outputs/sst2-smoke
sbatch slurm.sh qnli
# QNLI 完成后再提交 SST-2
sbatch slurm.sh sst2
```

在已分配的 GPU 上执行试跑。PyTorch 构建、分区和账号按目标集群调整，TC2 环境用 Python 3.11 和 cu118。正式结果与续跑文件分别放在 outputs/<任务>-transfer-seed42 的 results 和 checkpoints 下。也可以在 GPU 分配内用 bash run.sh TASK METHOD 单独运行。
