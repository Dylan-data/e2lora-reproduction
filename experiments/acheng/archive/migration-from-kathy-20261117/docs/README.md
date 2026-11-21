# kathy · 图像分类

实验计划与未确认设置见 [PLAN.md](PLAN.md)。仅用计算节点运行，仓库根目录提交：

```bash
sbatch experiments/kathy/train_pilot.sh lora
# 等上一作业 COMPLETED 0:0 并检查 result.json 后，再选下一种方法
sbatch experiments/kathy/train_pilot.sh e2lora
sbatch experiments/kathy/train_pilot.sh manual
```

每个作业仅一种方法、一个数据集。日志在 `outputs/image-classification/logs/`，结果在 `eurosat-{method}-pilot-{jobid}/`。依赖环境为 e2image，PyArrow固定21.0.0，NumPy1.26.4；提交前检查依赖，避免54488的ABI错误。原始失败作业和修复记录保留。

`train_pilot.py` 使用相同训练256张、训练集独立留出验证128张、test128张。只有训练样本参与梯度采集，E²采用paper分秩，不作熵平移。4步训练只用于验收；30%线性预热后恒定学习率、alpha16和六类投影属于试跑选择。结果中保存模型/数据版本、样本编号、文本特征缓存、预处理、目标层、计划、整数参数、分项耗时、显存和GPU小时。

`checkpoint.pt` 在每一步保存，包含适配器、优化器、下个步号、分组、划分及随机状态。验收程序恢复第2步后续训并与不中断的第4步逐位比较。当前外部进程重提不会自动续跑；此检查仅验证状态完整性及同进程恢复，正式长训前仍需完成跨进程恢复入口。重复输出目录有result.json时拒绝覆盖。

原始zero-shot脚本及独立全测试集脚本保留。试跑与全量零样本结果分开记录；不得用4步试跑分数或论文参考数值填正式训练表格。完整训练须先获得用户对计划和设置的确认。
