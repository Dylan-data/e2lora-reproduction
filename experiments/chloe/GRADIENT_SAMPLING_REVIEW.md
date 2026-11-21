# 2048完整梯度采样：代码与CPU续审

## Job55648续审：四目标通过，下一项12目标

真实JSON/JSONL已核对，4目标/8次反向/0更新；GPU目标后和族释放后allocated均16,077,562,368 B，无增长；目标后reserved稳定，CPU缓存大小与形状一致，RSS没有持续增长。本次范围未见GPU/CPU泄漏信号，结论不覆盖同族多层或128目标。COMPLETED/0:0、30秒依据用户提供的账单，原始文件不改写，见 [审阅记录](report/gpu-preflight-reviewed-55648.json)。

旧per_family只有allocated/RSS及声明释放字节，没有reserved或直接Tensor存活检查。为下一项spread12只增强诊断，补per_family reserved、mean Tensor弱引用存活数、前缀缓存字节校验；不修改采样算法/HRA/冻结规则。下一项每族3层、2条合成序列、24次反向，命令和停止条件见 [GPU_PREFLIGHT.md](GPU_PREFLIGHT.md)。

当前两项容量预检已通过：训练Job55643、单目标梯度Job55644；均非正式实验。原始JSON/stdout已核对，COMPLETED/0:0依据用户提供的账单信息，本轮没有调用sacct或提交GPU作业。数值与SHA见 [验收摘要](report/gpu-preflight-passed-55643-55644.json)。本轮新增CPU测试后全套90项通过；四目标已完成，12目标仍未提交。

## 1. 是否逐目标，以及GPU张量何时释放

正式路径为 `formal_runner.fixed_sampling_batches → preparation.collect_plan → collect_family_gradients → adaptive.collect_gradients`。

| 对象 | 当前行为 | 审查结论 |
|---|---|---|
| 128个attention权重 | Q/K/V/O各32层，按投影/目标顺序重放 | 每次只传一个目标，不是128个一起求梯度 |
| requires_grad / mode | 基础模型先冻结；临时只解冻当前权重，切eval，finally恢复原flags/mode | 不创建基础模型optimizer；不把其他目标加入同一反向 |
| GPU inputs | formal和新预检的batch在CPU；loss callback每次搬一个微批 | callback不把GPU batch存入持久列表 |
| autograd | `autograd.grad(..., create_graph=False, retain_graph=False)` | 不构建高阶梯度图，不保留用于重复反向的saved activations |
| CPU mean | `grad.detach().to(cpu,float32)`，按监督token权重累加/归一化 | 返回均值是CPU FP32、无grad_fn；Parameter.grad不被填充 |
| 临时loss/GPU grad/CPU搬运副本 | 每微批显式del，完成后再做下一个forward | 本轮收紧释放时点，避免上一份局部对象跨入下次RHS计算 |
| 投影CPU字典 | 同一投影的所有目标梯度保留至相似度计算结束 | 处理完投影后del；不把raw gradients放进diagnostics或plan |
| RMI矩统计 | 单个梯度临时转FP64，取矩后立即删除副本 | 只在diagnostics保留Python数值/matrix |

修改前，上一微批的loss与最后一个gradient变量会保持到下次赋值，造成有界临时重叠。它不等于已经发现跨目标无限泄漏：collect_gradients返回时原局部变量会退出，saved backward tensors默认也会释放。本轮只显式缩短生命周期；loss加权、RMI、HRA、共享、冻结及初始化规则没有改变。

旧 `prepare_e2lora.py` Smoke入口会把选中的batch列表提前搬GPU，目前只有2个小batch；不能把这个旧入口直接扩大为长Sweep。正式入口与新多目标入口使用CPU batch缓存，逐微批搬GPU。若将来改为可重放流式数据加载，必须保持顺序、样本和权重一致，本轮没有猜测这种新协议。

## 2. CPU梯度缓存与主机RAM

根据真实safetensors头部形状计算，无需加载权重张量，见 [sampling-resource-audit.json](report/sampling-resource-audit.json)。

| 投影 | 单目标weight形状 | 单目标FP32 mean | 32层FP32 mean |
|---|---|---:|---:|
| Q | 4096×4096 | 64 MiB | 2 GiB |
| K | 1024×4096 | 16 MiB | 0.5 GiB |
| V | 1024×4096 | 16 MiB | 0.5 GiB |
| O | 4096×4096 | 64 MiB | 2 GiB |
| 全部128目标若同时保留 | — | — | **5 GiB = 5,368,709,120 B** |

当前实现按投影释放，因此最大持久raw mean缓存为 **2 GiB = 2,147,483,648 B**，不是每个目标算完就立即释放CPU mean。每个目标的mean不随采样steps增长；同一目标不同batch只更新一个CPU accumulator。

2GiB不是完整进程需求，还需区分阶段：

- 梯度阶段：最大单目标FP32 transfer临时量64MiB及accumulator；GPU BF16原梯度和模型激活另计。显式del减少下一微批的副本重叠。
- RMI阶段：最大单目标FP64副本128MiB，加finite-check/var_mean等内部工作区；本轮避免上一FP64副本跨到下一个转换。
- Table4替代指标：现实现会保留整个投影的FP64 flat列表；Q/O为4GiB，加原始FP32约6GiB，再加归一化/差分临时量。不能把RMI工作区估计套到所有指标。
- 输入阶段：CPU batch列表随S、B和长度增长。例如尚未确认的S512/B64、每条均2048、input_ids/mask/labels均int64，对应1.5GiB张量上限；Python样本列表、完整100K tokenization、源JSON和模型CPU加载另外占用内存。

96G主机内存申请保留，但当前只有结构字节核算，没有完整128目标/相似度的实测RSS。新预检记录VmRSS/VmHWM；CPU allocator可能保留已释放内存，因此RSS不立即下降不等于Tensor仍存活。

## 3. Sampling steps与真实计算次数

设S为逻辑采样steps，B为每逻辑步样本数，m为微批大小：每目标的微批数为 `S*ceil(B/m)`，128目标的forward/autograd.grad调用数为 **`128*S*ceil(B/m)`**，optimizer更新始终为0。B不能省略；`gradient_examples`也不能直接当steps。

| S | 若B=1,m=1，128目标反向次数 | 若B=64,m=1，128目标反向次数 |
|---:|---:|---:|
| 8 | 1,024 | 65,536 |
| 16 | 2,048 | 131,072 |
| 32 | 4,096 | 262,144 |
| 64 | 8,192 | 524,288 |
| 128 | 16,384 | 1,048,576 |
| 256 | 32,768 | 2,097,152 |
| 512 | 65,536 | 4,194,304 |

两列是给定B/m后的精确次数示例，不是已确认正式batch。CPU测试还覆盖B不能被m整除时每逻辑步各自保留尾微批的情况，不用ceil(S*B/m)替代实际循环。

## 4. 是否存在累积与如何观察

CPU测试验证旧loss/gradient对象在下一次forward前已经死亡；CPU mean无图、无Parameter.grad累积；前一投影的所有mean在下一投影开始前死亡；顺序梯度与独立加权autograd数值一致；失败/observer中止会恢复模式与flags。没有改成一次同时解冻多个目标。

这些是CPU对象生命周期与数值验证，不是GPU峰值趋势实测。GPU allocator可能让reserved逐渐增长，模型也可能建立持久小缓存；不能只看reserved或nvidia-smi判断计算图泄漏。应在每目标返回/CPU卸载后同步记录当前allocated与reserved，比较预热后的allocated是否持续随目标数上涨，并记录每投影释放后的状态。诊断不调用empty_cache掩盖趋势。

异常回溯会暂时持有出错帧；新预检失败即结束，不在同一进程内自动重试累积对象。成功路径的CPU元数据observer只接收数值/名称，不接收原始Tensor。

## 5. 下一项受限GPU预检

[multi_target_preflight.py](multi_target_preflight.py)与[独立SLURM模板](slurm/multi-target2048-preflight.sbatch)已准备，默认CPU计划。执行命令和验收项见 [GPU_PREFLIGHT.md](GPU_PREFLIGHT.md)。

| 级别 | 输入/目标 | 精确反向次数 | 当前状态 |
|---|---|---:|---|
| A first-layer4 | 2条固定合成2048序列；layer0 Q/K/V/O | 8 | 已通过Job55648受限验收 |
| B spread12 | 同样2条；layers0/15/31各四投影 | 24 | A已审阅通过；命令待用户检查，未提交 |
| C 单投影32目标 | 验证最大2GiB缓存和CPU相似度工作区 | 按届时明确微批数计算 | 尚未提供执行入口 |
| D 全128目标 | 先少量固定样本，完整路径测量 | 按确认S/B/m计算 | 尚未批准或提交 |

A/B固定B1/m1、最多2步，不创建optimizer、不安装adapter、不构造HRA，也不把合成样本当正式MetaMathQA。CLI拒绝full128和512步。申请1×A40、8CPU、96G、30分钟；600秒阶段预算只在目标边界检查，SLURM提供硬上限。进度JSONL和结果均用新目录，禁止覆盖既有结果。

A通过需要：COMPLETED/0:0、输出8次反向/0更新、每目标梯度有限、冻结恢复、无持续allocated增长或OOM。脚本成功只标COMPLETED_REQUIRES_REVIEW，不自动解除formal门禁。A/B只保存标量进度，不持久化原始梯度；旧256准备文件不能当2048缓存命中。

## 6. 实测与假设边界

唯一新增梯度耗时实测是Job55644：layer0 q_proj、一个合成2048微批，包含梯度计算和CPU mean，2.017151秒。Q/O和K/V形状/传输量不同，目标深度决定反向路径，首次kernel/缓存、样本长度、CPU统计和观测开销也不同。因此不提供 `128*2.017` 或更长steps的总耗时结论，资源报告的完整采样时间保持null。

新预检逐目标记录同步后时间与CPU卸载，完整phase还包括观测、finite-check、投影释放；模型加载单列为process成本。完成分级测量后，若建立预测模型，仍应单列假设、校准覆盖范围与误差，不能将预测改名为实测。

## 7. Partial-rank与成员4协议仍阻断

本轮meta重算E² r8/r16仍为6,810,624/13,631,488；active139候选仍为3,700,736，见 [续审计数](report/partial-rank-sampling-review.json)。当前全A/B预算与Table5约半数的表值不符，冻结策略/非零初始化及Table4统一控制仍未确认；不按论文数字反推active秩，不改变HRA公式或任意冻结矩阵。

成员4只完整确认seed42、epoch1、2048、生成512四项。实际100K/验证清单、完整训练配置、train/eval Prompt、GSM8K scorer/题集等价性及driver仍未交齐；shuffle、warmup、clip、截断、checkpoint选择和sampling steps定义差异依旧存在。两项容量通过不消除这些差异，正式协议继续pending。

CPU续审记录：[cpu-verification.sampling-review.json](report/cpu-verification.sampling-review.json)。
