# 2048预检：已通过项目与受限多目标方案

2026-11-18 用户要求暂停新的GPU预检，优先进行 [Table 5参数预算静态审计](TABLE5_PARAMETER_BUDGET_AUDIT.md)。下文12目标方案仅保留为准备记录，当前不提交、不执行，也不据四目标通过启动128目标或正式实验。

训练容量Job55643、单目标梯度Job55644和四目标Job55648均通过各自受限范围的验收，**均为非正式实验，不能进入Table4/5或Figure5**。本轮审阅四目标结果并准备12目标方案，没有提交新GPU作业。

## 已核对的两项结果

本地JSON与对应stdout内容一致；COMPLETED/0:0状态来自用户提供的SLURM账单，本轮未查询sacct。摘要与原文件SHA见 [验收记录](report/gpu-preflight-passed-55643-55644.json)。原始结果和旧Smoke均未改写。

| 项目 | 训练容量 | 单目标梯度 |
|---|---|---|
| Job / 状态 | 55643 / COMPLETED 0:0 | 55644 / COMPLETED 0:0 |
| 输入 | 合成2048序列，micro1、accum1 | 合成2048序列、1微批 |
| 模式 | LoRA r8、checkpoint on | eval、checkpoint off、layer0 q_proj |
| Optimizer updates | 2 | 0 |
| Peak allocated | 19,882,919,936 B（18.517 GiB） | 28,374,689,280 B（26.426 GiB） |
| Peak reserved | 20,912,799,744 B（19.477 GiB） | 29,183,967,232 B（27.180 GiB） |
| CPU梯度 | 不适用 | 67,108,864 B（64 MiB） |
| 实测计算阶段 | 10.978 s，两步约9.769/1.209 s | 2.017151 s |
| JSON process_seconds | 403.730 s | 12.840 s |

阶段时间与process_seconds来自JSON，不是完整allocation账单。首步、加载、文件缓存、目标位置/形状均影响时间；不能将1.209 s当正式训练吞吐，也不能用2.017 s乘128或采样步数当完整采样预测。当前完整采样总耗时仍为null。

这证明在该A40和具体配置下，2048独立LoRA训练及一个q_proj的梯度可以运行；checkpoint-off训练、完整128目标、实际MetaMathQA输入、E²高group rank/partial冻结、512-token生成与正式GSM8K评分均未由此验收。正式协议的formal_preflight继续pending。

## Job55648四目标审阅：通过，结论仅限本次范围

核对 [原始JSON](../../outputs/chloe/preflight/multi-target2048/first-layer4-v1.json) 及逐目标JSONL，二者一致；采样核心源码SHA与运行时记录一致。用户提供SLURM为COMPLETED/0:0、30秒。本地JSON显示4目标、8次forward/autograd.grad、0更新，phase=7.339354547 s，process=22.535120953 s。原始JSON的COMPLETED_REQUIRES_REVIEW状态不改写，独立审阅结论见 [gpu-preflight-reviewed-55648.json](report/gpu-preflight-reviewed-55648.json)。

| family | CPU均值缓存 | 目标后RSS MiB | 族释放后RSS MiB | 族释放后allocated |
|---|---:|---:|---:|---:|
| Q | 64 MiB | 1480.289 | 1414.957 | 16,077,562,368 B |
| K | 16 MiB | 1462.895 | 1482.395 | 16,077,562,368 B |
| V | 16 MiB | 1482.395 | 1449.758 | 16,077,562,368 B |
| O | 64 MiB | 1511.793 | 1475.852 | 16,077,562,368 B |

四个目标结束后的allocated也全部为16,077,562,368 B，last-minus-first=0；per_target的reserved全部为29,183,967,232 B。峰值allocated=28,383,159,808 B，reserved=29,183,967,232 B。这说明本次范围内未观察到跨目标GPU分配累积；reserved保留瞬时计算空间属于缓存分配器行为，本例不能据此判为图泄漏。

`per_target.cuda_peak_*` 是本次采样阶段自reset以来的累计峰值，四行相同不表示每个目标各自的峰值相同。`per_family` 的allocated四行均回到同一基线；原结果没有族释放后的reserved，不能补写或由目标后reserved推定为已测量值。

CPU缓存64/16/16/64MiB与实际权重形状一致，两个微批被合成为一个mean，没有按样本翻倍缓存。RSS没有持续递增；K释放后反而上升19.5MiB，说明RSS不是存活Tensor字节数。peak RSS四行均为5,874,597,888 B（5.471 GiB），是进程累计高水位，可能包含加载阶段，不是四目标梯度缓存相加。

证据边界：旧per_family没有reserved字段；旧released_gradient_bytes只是del之前的字节计数，不是独立的释放证明。结合代码、此前CPU弱引用测试及无持续RSS增长，本次未发现CPU缓存泄漏信号，但不能据四个单层族证明全部缓存均无问题，也没有覆盖同族多层累积、32层/128目标、真实数据或相似度工作区。

采样数学规则、HRA、共享和参数冻结代码均未修改。为12目标只增强诊断记录：目标前缀的预期CPU缓存字节与实际值比较；per_family补充reserved；del后记录弱引用存活Tensor数并要求为0。此检查证明Python Tensor对象已释放，不要求CPU allocator立即把RSS还给操作系统。

## 下一项：12目标、同族3层，准备未提交

使用 [multi_target_preflight.py](multi_target_preflight.py) 的spread12：每族依次layers0/15/31，顺序为Q三层→释放→K三层→释放→V三层→释放→O三层→释放。固定2条合成2048序列，B1/m1，共**24次forward/autograd.grad、0更新**。不安装adapter、不做HRA/分组、不改变冻结规则，不依赖未交接的Prompt。

| 同族目标进度 | Q/O预期CPU缓存 | K/V预期CPU缓存 |
|---|---:|---:|
| layer0后 | 64 MiB | 16 MiB |
| layer15后 | 128 MiB | 32 MiB |
| layer31后 | 192 MiB | 48 MiB |
| 族释放后 | live Tensor数0 | live Tensor数0 |

对应精确字节数：Q/O为67,108,864 → 134,217,728 → 201,326,592 B，K/V为16,777,216 → 33,554,432 → 50,331,648 B。预期按目标数量累积一次FP32均值；不能按两个微批翻倍，也不能把前一族缓存带到下一族。逐目标检查以加载模型的实际weight形状计算，表中数值用于核对本次Llama-3.1-8B。

最大原始均值缓存为单族192MiB，不是四族合计480MiB；仍有临时量、模型加载和allocator额外占用。RSS无需严格按这些数值回落。GPU目标后allocated以本次约16.078GB为参考，期待稳定而非随族内目标数持续阶梯上涨；不同深度可能建立小型缓存，不把单次微小变化直接当泄漏。reserved增长本身不作为失败判据。

执行限定1×A40、8CPU、96G主机RAM、30分钟硬上限，最多0.5 GPU-h allocation；600秒采样预算仅在目标边界检查，无法中断单个CUDA调用。**不是把7.339秒乘3预测耗时。** 模板使用MGPU-TC2/stud/normal和`--gres=gpu:1`申请一张GPU；若调度到非A40，脚本在加载模型前拒绝运行。

CPU预览（不使用GPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python experiments/chloe/multi_target_preflight.py --case spread12 --sampling-steps 2 --max-phase-seconds 600
```

可审阅的SLURM命令，**本轮未执行**：

```bash
cd /home/msds/yahui003/e2lora-reproduction
mkdir -p outputs/chloe/slurm
sbatch --parsable --time=00:30:00 \
  experiments/chloe/slurm/multi-target2048-preflight.sbatch \
  --case spread12 --sampling-steps 2 --max-phase-seconds 600 \
  --output outputs/chloe/preflight/multi-target2048/spread12-v1.json
```

输出为：

- JSON：`outputs/chloe/preflight/multi-target2048/spread12-v1.json`
- 逐目标：`outputs/chloe/preflight/multi-target2048/spread12-v1.targets.jsonl`
- 日志：`outputs/chloe/slurm/multi-target2048-<JOB_ID>.out/.err`

已有JSON或进度文件会使脚本拒绝覆盖；重试用新文件名。first-layer4-v1及所有Smoke/既有预检保持不动。

## 12目标验收、停止与后续边界

验收需SLURM COMPLETED/0:0、case=spread12、12条per_target/4条per_family、24次forward/autograd.grad、0更新、每目标2次反向；CPU缓存前缀准确，per_family的cpu_gradient_tensors_released=true且live count=0，冻结状态恢复、梯度有限，并人工审阅GPU allocated/RSS趋势。

脚本自动失败停止：OOM、非有限loss/梯度、CPU缓存字节与实际形状不符、族释放后仍有mean Tensor引用、冻结恢复失败、调用次数错误、在目标边界超过600秒；SLURM运行超30分钟会终止作业。时间限制/系统OOM杀进程可能没有最终JSON，保留JSONL与stderr排查，不当作通过。

人工停止/不再扩展：预热后allocated随目标数持续累积、族结束仍明显不回落，或RSS持续按已释放族的梯度大小累加。可用tail查看stdout/JSONL，必要时对该预检Job执行scancel；本轮未执行任何scancel。RSS/reserved单点增长仅作为待查信号，不直接证明泄漏。完成后先审阅新结果，不自动提交32层或128目标。

## 扩展顺序与缓存

1. 四目标已通过受限验收，无需重复提交；下一项仅为上述spread12。
2. 审阅12目标的族内累积/释放与显存趋势。CLI最多允许这12目标和2步，不能传full128或512步。
3. 再决定是否准备单投影32目标（检验2GiB CPU缓存及相似度工作区），之后才考虑全部128目标；这两级尚未提供可直接执行的CLI，不会自动扩大。

复用现有模型/tokenizer即可。旧长度256的prepared-plan/RMI不能代替2048数据梯度；现有文件没有持久原始梯度缓存。新容量预检也不产出正式StageA plan，不能将其2条合成样本用于正式分组。复用正式统计须另行确认模型、源数据/有序ID、Prompt/tokenizer、长度、精度、采样seed和loss等完整身份。

若A40多目标OOM，保留失败JSON/逐目标进度，区分目标深度、输入、GPU临时量和CPU相似度工作区；不默认缩到1024、不改partial秩或量化来宣称原配置通过。既有容量/四目标通过及CPU测试都不解除Table5预算及成员4Math交接阻断。

本轮CPU验证共96项通过，其中新增6项覆盖12目标编排、同族均值缓存累积及释放、故意保留CPU均值引用时停止、缓存大小错误时停止、600秒边界停止和既有JSON/JSONL保护。新增测试使用小型CPU模型与模拟CUDA传输/指标，实际执行24次小模型autograd以验证编排，不能作为12目标GPU实测。默认CPU计划及SLURM脚本语法也已检查，未创建`spread12-v1.json`或其进度文件。

本轮验证与源码SHA：[cpu-verification.spread12-preparation-v1.json](report/cpu-verification.spread12-preparation-v1.json)。此前CPU续审记录：[cpu-verification.sampling-review.json](report/cpu-verification.sampling-review.json)。

本次12目标诊断准备的CPU验证：[cpu-verification.55648-review.json](report/cpu-verification.55648-review.json)。
