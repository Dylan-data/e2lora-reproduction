# 成员 4 / 成员 5 正式实验协议对齐审查

## 2048预检通过后的续审

训练容量Job55643和单目标梯度Job55644均已通过，**均为非正式实验**。本地JSON/stdout已核对，COMPLETED/0:0依据用户提供的账单信息；见 [两项验收记录](report/gpu-preflight-passed-55643-55644.json)。此前“2048尚未执行”的准备状态由本节更新，但不把容量通过等同于正式协议通过。

当前确认范围仍仅seed42、epoch1、2048、生成512。成员4实际100K/验证清单、完整训练/Prompt/评分包仍待交接；shuffle、warmup步序、clip、答案截断、best/final checkpoint、adapter精度/初始化/缩放及采样定义差异仍在。Table4/5 partial-rank政策与预算容差未确认，未作数字匹配调整。

本轮仅收紧梯度张量/计算图释放时点，保留原loss加权、RMI、HRA、共享和冻结数学规则；CPU全套90项通过。新4/12目标预检尚未执行。完整128目标显存/RAM与耗时、真实数据端到端评分验收继续阻断正式运行，详见 [采样续审](GRADIENT_SAMPLING_REVIEW.md)。

## 2026-11-18 更新：当前范围为 Math-only

**HumanEval 因隔离执行环境限制，经组内批准不纳入本次复现范围。** Code-Feedback 同步排除。保留既有 Code 框架及12份禁用配置，活动矩阵、配置编译和执行门禁只允许 Math；Code 数据、HumanEval 环境及 Code Smoke 不再是成员5正式消融的前置条件。本节取代下方历史审查中的双任务范围和 Code 放行要求。

本轮没有 fetch 新提交；Git证据仍为 `origin/main@1c4f3bbc0f20659dac57d484d231eb84bbb3c27e`。新增设置的确认依据是用户于2026-11-18转达的成员4结论，记录在 [member4-table2-confirmed.json](configs/member4-table2-confirmed.json)，不冒充 main 中已有完整运行配置或双方已签署完整协议。

| 最新事实 | 状态 | 对成员5的影响 |
|---|---|---|
| Table2 Math seed=42、epochs=1、max_seq_length=2048、max_new_tokens=512 | **A：成员4已确认这四项** | 已记录到 Math pending 配置；seed42可作为第一重复，额外两次seed不猜测 |
| 1024会报错；正式选2048 | **A：长度选择已确认** | 下方历史C项“2048只是示例”已被本次确认更新；训练显存及2K验证集仍需核验 |
| 成员4其他两表尚未完成试跑 | **C：设置仍待确认** | 不将 Table2 设置外推成 Table14/15 的正式协议，也不要求其试跑完成才推进本次 Math-only 工作 |
| Code-Feedback/HumanEval排除 | 范围决定，不是待补文件 | 取消36次Code计划；当前用户SLURM队列只读查询返回空，未发现需scancel的Code作业，也未提交新作业 |
| 正式矩阵 | 24配置、72次三重复计划 | Table4 8×3，Table5 4×3，Figure5 Math rank 5×3、steps 7×3；只有seed42已知，另外48个seed槽位待定 |
| 实际100K manifest、完整训练配置、Prompt、GSM8K scorer/接口 | **C/D：仍未交齐** | 本地候选100K核验仍有效，但尚无成员4实物可逐项确认 |
| shuffle、warmup、clip、截断、checkpoint选择、partial-rank | **B：差异尚未解决** | 四个新参数不能消除这些更新/评估行为差异 |
| 2048 A40预检 | 已准备，尚未执行 | 训练checkpoint on/off与eval模式单目标梯度分别测；旧短序列Smoke不代替正式预检 |

旧 main 的 math入口仍硬导入缺失的 generation、generation_methods、generation_metrics 及 code/commonsense metrics。现在不要求补做Code实验；需成员4提供能独立运行的Math入口及真实数学依赖，或将非Math依赖改为按任务延迟导入。旧D清单是该固定提交的事实，不把其中HumanEval文件继续当作本次交付范围。

最新参数核验见 [Math-only参数审计](report/math-only-partial-rank-audit-20261118.json)：在真实Llama meta架构及旧Smoke缓存统计上，全A/B E² r8为6,810,624，r16为13,631,488；分别对LoRA r4/r8约多99.85%/100%。active139候选仍为3,700,736，且改变冻结/初始化。**Table5预算不匹配仍是实质阻断；不能为凑论文数字选冻结比例。** 详见 [PARTIAL_RANK_AUDIT.md](PARTIAL_RANK_AUDIT.md)。

剩余必须交接的Math材料：完整run_config与版本/运行入口；三重复seed（可先交seed42）；100K与选择验证集的有序ID/行哈希/源SHA及规范化映射；训练/评价Prompt及BOS/EOS/截断；optimizer/adapter/精度/clip/scheduler完整步序；checkpoint选择与恢复规则；GSM8K固定1319题清单及与本地HF版本的内容对照；实际numeric scorer及正反例；可加载Chloe完整adapter状态的 `config_checkpoint_output_v1` driver及环境锁。还需确认Table4机制控制、Table5预算政策与Figure5采样定义/固定值。清单见 [MEMBER4_HANDOFF.md](MEMBER4_HANDOFF.md)。

**当前仍不具备第一组Math正式消融条件。** 最短路径、资源估算及预检命令见 [FORMAL_ABLATION_READINESS.md](FORMAL_ABLATION_READINESS.md) 与 [GPU_PREFLIGHT.md](GPU_PREFLIGHT.md)。取消HumanEval消除了Code侧阻断，没有自动补齐Math侧证据。本轮只做CPU检查和预检准备，未改写已通过Smoke结果、未启动GPU训练、未Push。

## 2026-11-17 历史审查（以下保持当时证据与范围）

审查日期：2026-11-17（Asia/Singapore）。结论：**Table 4、Table 5、Figure 5 当前均未达到正式启动条件。** 成员 4 已上传训练框架、数据处理代码和长度统计，但本次审查的 main 提交缺少关键依赖模块、正式配置和实际样本清单，不能据此认定正式协议已齐备。

本报告是读取审查结果，不代表双方确认协议。本次仅新增本报告；没有修改训练、评估、配置、共享模块或既有结果，没有执行 merge/pull/checkout、删除文件、安装依赖、下载数据、提交 GPU 作业或 push。

## 1. 审查对象与证据口径

| 对象 | 本次实际审查版本 |
|---|---|
| 成员 4 | 已 fetch 的 `origin/main`：`1c4f3bbc0f20659dac57d484d231eb84bbb3c27e`，提交说明 `Update README.md` |
| 成员 5 | 当前 `chloe-ablation` 工作区，HEAD：`d13dc42784dea9c3114e66f53606874f0ab41df8`；包含未提交的 Chloe 实现，未切换分支 |
| 共享参考 | 本地与 `origin/main` 的 `references/text-generation-setup.json`、`references/ablation-setup.json` 内容相同 |
| 比较方式 | `git show origin/main:<path>`、`git ls-tree -r --name-only origin/main`，逐行静态阅读；另只读核验本地数据清单、原始文件校验值和下载元数据 |

已读取用户指定的八个文件：`experiments/s/README.md`、`README_full_suites.md`、`REPRODUCTION_AUDIT.md`、`IMPLEMENTATION_MATRIX.md`、`suite_train.py`、`train.py`、`evaluate.py`、`run_matrix.py`。同时检查 main 中实际存在的共享数据/模型模块、`inspect_token_lengths.py`、五份 preflight JSON、两个 SLURM 模板及 `pyproject.toml`；成员 5 侧检查了 `common.py`、`prepare_data.py`、`train.py`、`evaluate.py`、`formal_runner.py`、`alignment.py`、`ablations.py`、正式/Smoke 配置、交接文档和既有验收记录。另直接读取本地 e2lora conda 环境的 `transformers/optimization.py`，核实 HF scheduler 的 warmup 公式，没有导入模型或执行训练。

以下证据采用 `路径:行号`；成员 4 路径均指上述固定提交，不能误当作当前分支已集成的文件。简写：

| 简写 | 文件 |
|---|---|
| S / T / E / M | `experiments/s/suite_train.py` / `train.py` / `evaluate.py` / `run_matrix.py` |
| N / G | `src/e2lora/dataset/nlg_suites.py` / `text_generation.py`（main） |
| R | `experiments/s/README_full_suites.md` |
| C / F / A | `experiments/chloe/common.py` / `formal_runner.py` / `alignment.py` |
| P | `experiments/chloe/configs/formal-protocol.pending.json` |

状态定义：

- **A：已一致。** 仅限该项列明的配置事实或实现行为，不表示整个运行协议已确认。
- **B：不一致，需要修改。** 已看到实际实现或接口差异；双方确定共同规则后再修改，本次不改代码，也不默认成员 4 的实现就是最终标准。
- **C：组员 4 尚未确认。** 未见足以确认正式配置、来源或运行选择的交付物；有些项目同时也需要成员 5 确认。
- **D：文档或代码提到，但 GitHub 所审查提交缺少实际文件。** 指该提交的可核验缺失，不推断对方本机是否存在。

`README.md:9` 已将 `suite_train.py` 列为新主入口，将 `train.py/evaluate.py` 列为 legacy 路线。两条路线的行为不能混用。**R:162 明确说明 epochs=1、max_seq_length=2048 和生成上限只是示例选择；本报告不把它们升级为正式参数。** `slurm/train_one.slurm:32` 起同样要求外部传入 epochs/长度/生成上限，并未补齐这些决定。

## 2. 逐项对照

### 2.1 模型、数据、样本与随机性

| 检查项 | 成员 4 实际证据 | 成员 5 实际证据 | 状态与处理 |
|---|---|---|---|
| 基础模型与声明 revision | S:46、共享 JSON：`meta-llama/Llama-3.1-8B` Base，`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | P:10、Math 配置相同；本地四个权重分片、config、tokenizer 的下载元数据均记此 commit | **A**：声明与本地下载记录一致；未对全部 8B 权重重新计算哈希，亦无成员 4 实际模型加载日志 |
| Attention 投影范围 | R:17 声明 Q/K/V/O；S:492 调用目标发现函数 | C:132、F:136 明确 q/k/v/o 四投影 | **A**：目标范围一致；main 缺目标发现实现，运行时覆盖仍需验收，见 D 清单 |
| MetaMathQA 来源、版本和训练量 | N:18、N:60：`meta-math/MetaMathQA`，`aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`，100K | `prepare_data.py:41`、Math 配置相同；现存源文件为 395,000 行 | **A**：来源/revision/100K 目标一致；不等于两端实际数据内容已经逐行一致 |
| 100K 采样算法 | N:66：`Random(data_seed).sample(range(len(dataset)), 100000)`，默认 data_seed=42 | `prepare_data.py:25` 调用共享 `sample_indices`，同为 Python Random.sample；本地现存清单核验通过 | **A**：算法和默认候选 seed 一致；成立条件是源行数及行顺序相同 |
| 实际 100K 样本 ID/顺序 | S:424 会在运行时写 sample_manifest，但 Git 中没有实际文件；共享 JSON `tasks.math.training_sample_ids=null` | 本地有 `outputs/chloe/data/math-100k-seed42.json`，状态仍为 candidate | **C**：成员 4 须提供实际有序清单/摘要供比对；该运行产物的 Git 缺失另记 **D**，见第 3 节 |
| 训练前规范化文件和 manifest 格式 | suite 使用 HF split、`prompt/target/sample_id/source_index`，运行清单为 train/validation 数组 | F:33、C:239 要求规范化 `query/response` JSON 数组及 `indices/source_sha256/indices_sha256` 清单 | **B**：目前不能直接互换文件；须交付转换及原始行映射，保持同一批样本 |
| 训练集外的验证集 | S:125、N:67：默认另抽 2K、不与 100K 训练行号重叠、validation_seed=43 | Chloe 当前清单仅含 100K train；`fit` 无 held-out 选择过程 | **B**：验证集与选择用途未接齐；须确认是否采用这 2K、确切 ID 和选择规则 |
| 每个 epoch 的数据顺序 | suite S:522：PyTorch DataLoader shuffle=True，使用全局 Torch RNG；legacy T:850：`Random(seed+epoch).sample(train_indices, len(...))` | `chloe/train.py:110`：`Random(seed+epoch).shuffle(positions)` | **B**：三种方法不会因同 seed 自动同序；suite 还受此前模型/adapter 初始化消耗 RNG 影响。需冻结同一 epoch ID 序列或同一独立 sampler |
| 三次训练 seed | M:19 明确计划 42/43/44；S 默认 run seed=42，data seed=42、validation seed=43 分离 | P:27 为 null；formal/*.json 的 training_seeds=null | **C**：成员 4 的矩阵计划可作为候选，成员 5 尚无双方确认记录，不自动填入正式配置 |
| E² 梯度采样数量与顺序 | suite S:217：默认最多 64 个 examples，`Random(seed+1701).sample`；legacy T:722 则取 train manifest 前 N 项 | F:40：打乱 manifest 位置后取 S×B 个不同样本，S 是逻辑步数，B 和 seed 策略待确认 | **B**：examples 不等于 sampling steps；64 examples 不能直接解释为 Figure 5 的 64 steps。需统一数量、批次、顺序和 seed |

### 2.2 训练配置与实际更新行为

| 检查项 | 成员 4 实际证据 | 成员 5 实际证据 | 状态与处理 |
|---|---|---|---|
| AdamW、LR、betas、weight decay 的公共设定 | 共享 JSON：AdamW、5e-5、(0.9,0.999)、5e-4；S:508、S:520 以这些值调用 optimizer builder；legacy T:793 直接构建 AdamW | P:15、`chloe/train.py:68` 相同 | **A**：LoRA/E² 公共数值一致；suite 的 build_optimizer 本体缺失，不能声称已验证其实际参数组 |
| 有效 batch size | 共享 batch=64；suite 默认 micro=1×accum=64，单进程；legacy T:639 强制 world×micro×accum=64 | 正式要求 global=64；C:94 强制单卡 micro×accum=64 | **A**：正式目标和当前默认几何一致。Smoke 的 1×8 仅供验收；suite 可 CLI 改 batch，尚无正式 run_config 证明最终取值 |
| Epoch / max_steps | suite S:66 必填 epochs；legacy T:589 必须明确 epochs 或 bounded steps；无正式配置实物 | P:93、pending Math epochs/max_steps 未定 | **C**：epochs=1 仅示例；不能据此启动 |
| 不完整尾 batch | suite S:565、legacy T:858 保留；Chloe 每次取到 min(end,len) | `chloe/train.py:114` 保留 | **A**：均保留尾批。100K/global64 时每 epoch 1,563 次更新，最后一次 32 examples；这是条件计算，不是 epoch 确认 |
| loss mask / packing / 监督 token 权重 | S:187、S:196：prompt mask=-100，answer+EOS 监督，无 packing；S:613 按 label!=−100 加权 | C:205、C:225；`chloe/train.py:119` 按 labels[:,1:] 的有效 causal targets 加权 | **A**：response-only、无 packing、token 加权方向一致；保留非空且被 mask 的 prompt 时两种 token 计数等价。截断差异另列 B |
| scheduler 的名称与 ratio | 共享及 S:80：cosine、warmup_ratio=.03；Table9 .3 和 LR decay .1 单独记录 | P:24：cosine/.03；共享参考相同 | **A**：公共声明一致，不把 .1 擅自解释为 cosine 最小 LR |
| scheduler 的精确步序 | suite S:528 对 warmup 向下取整，LambdaLR warmup 用 `(step+1)/W`；legacy T:496 向上取整、同样提前 1 步 | F:134 向上取整；`chloe/train.py:76` 调用 HF cosine warmup，初始 step=0 的 warmup 值为 0 的约定需纳入对齐 | **B**：存在取整和 warmup 步偏移。若 1 epoch/100K/batch64，suite W=46，Chloe formal W=47。需统一公式、首更新 LR 及完整 LR 序列；普通 Chloe baseline 则直接读取 warmup_steps，不能只填 ratio |
| 梯度裁剪 | suite S:611 的更新函数没有 clip；legacy T:900 是否 clip 取决于缺失配置 | Chloe fit 固定调用配置 max_grad_norm 的 clip，pending Math 值为 1.0，P 中尚空 | **B**：suite 与 Chloe 的更新规则不同；需确认 clip 值或无裁剪，修改相应实现后再验收 |
| BF16 与 adapter dtype | suite S:472：CUDA 支持时 BF16，否则可回退 FP32；adapter dtype 在缺失模块中；legacy 要求 BF16 CUDA | `chloe/train.py:169` 的 GPU 检查、C:149：BF16 base、FP32 adapters、autocast；F:77 同样转 FP32 adapter | **C**：公共 BF16 声明一致，实际 adapter/optimizer dtype 尚不能对齐；需成员 4 的实现及记录 |
| alpha、dropout、初始化、scaling | S CLI alpha 默认16，但 generation/generation_methods 缺失，不能核验 dropout/初始化/缩放 | Smoke alpha16；正式 P:33 各项待定，A:91 仅列代码支持范围 | **C**：不能将 CLI/Smoke 默认提升为正式共同决定；需明确 alpha/r 与不同 group rank 的缩放 |
| Attention backend / checkpointing | suite 为可选 CLI；实际 `slurm/train_one.slurm:53` 起启用 gradient checkpointing、sdpa；无对应运行日志 | C:195 固定 sdpa，正式 checkpointing 待确认，Smoke 已启用 | **C**：有一致候选模板，尚缺正式运行选择和版本记录 |
| checkpoint 选择 | suite S:568 按 held-out validation loss 选 best；legacy T:921 按生成后的 validation numeric accuracy 选 best；E 默认 best | Chloe `fit` 仅保存训练游标 checkpoint，正式 F:196 交给外部 evaluator，没有实现相同的 best 选择 | **B**：需先选定 best-loss / best-accuracy / final 的共同规则及频率；不能随意拿各自不同 checkpoint 比较 |
| 恢复时的数据随机性 | suite epoch checkpoint 未保存 RNG，重建 DataLoader 时未恢复之前 shuffle 状态；legacy T:427 保存 RNG | Chloe 保存 Python/Torch/CUDA RNG 和确定性 epoch/offset | **B**：中断重跑 suite 不能据此保证与不中断相同样本顺序；正式协议需固定恢复约定 |
| E² HRA 与 partial-rank 口径 | suite S:234 用 shared build_plan 的逐组 round，并设置 trainable_rank=rank；legacy T:346 用精确最大余数分配 total/active 两个预算；共享 JSON 却登记 total256/active139 候选策略 | formal 调用 paper HRA；partial policy、不同 rank 的 active budget 尚待确认 | **B**：成员 4 两条入口及共享候选存在实质差异，需确定正式算法/冻参规则，详见第 6 节 |

### 2.3 训练长度、Prompt 与 GSM8K

| 检查项 | 成员 4 实际证据 | 成员 5 实际证据 | 状态与处理 |
|---|---|---|---|
| 正式训练序列长度 | S:67 必填；已上传 Llama 100K 长度统计，但 selected_max_seq_length=null | P:95、pending Math max_length/max_prompt_length 均 null | **C**：2048 是有统计支持的候选，未获正式选择；1024 会触发成员 4 已上传样本的 answer-too-long，不能照抄旧估算 |
| Math 训练 Prompt | suite N:76：`query.strip()+"\n"`；legacy G:125：`query.rstrip()+"\n"`，无 Question/Answer wrapper | 正式 template=null；Smoke 为 `Question: {query}\nAnswer:\n`，当前编码器按 template 直接格式化 | **C**：正式 template 未确认。**现存 Smoke 与成员 4 训练格式不同**，若沿用则为 B；不能因 Smoke 跑通就默认该模板 |
| 截断及 EOS | suite S:189 保留 answer+EOS，超长仅左裁 prompt，answer 独占长度则报错；legacy G:223 相同大原则 | C:209 保留 BOS+prompt 尾部，先给 response 加 EOS，再右截 response | **B**：会改变监督目标及 EOS 是否保留。即使选够长度使一批样本不触发，也需统一边界规则并验证 train+validation |
| GSM8K 测试来源/版本/题序 | suite S:322 固定 OpenAI grade-school-math commit `3101c7d5072418e28b9008a6636bde82a006892c` 的 test.jsonl；legacy E:121 读取缺失配置中的 source_url，注释说明可能跟 master | 本地 HF `openai/gsm8k/main/test` revision `740312add88f781978c0658806c59bc2815b9866`，实现要求 1,319 题 | **C**：都指向 GSM8K test 仍不足以证明内容/顺序相同；须比较完整 question/answer 有序哈希和 ID。原始 JSONL/Parquet 字节 hash 不应直接比较 |
| GSM8K evaluation Prompt | suite S:331：`Question: {question}\nAnswer:`，无末尾换行；legacy E:132 从缺失配置读取 | 旧 evaluate 复用训练 template；Smoke 是 `Question: {query}\nAnswer:\n`；正式 P:102 为空 | **C**：正式模板未定；现有 Smoke 比 suite 多末尾换行。必须按 token IDs 对齐，并分别确认 train/eval 模板，不能要求两者自然相同 |
| evaluation 输入裁剪 | S:292 左截到 max_seq_length；legacy E:136 同样保留最后 L 个 tokens | `chloe/evaluate.py:60` 保留 BOS 与尾部，限制用 max_prompt_length | **B**：输入长度上限与 BOS 处理不同；需共同规定 eval prompt cap、BOS 与超长行为 |
| GSM8K 生成长度 | suite 必填 max_new_tokens；README 512 仅示例；无完成性验收记录 | Smoke=256；正式 max_new_tokens=null | **C**：256/512 都没有被确认成正式上限；需报告达到 cap 的题数、EOS/答案完整率 |
| 解码和输出切片 | S:305：do_sample=False、num_return_sequences=1、EOS/pad 使用 tokenizer，decode 新生成 tokens 并 skip special；未显式锁 num_beams，继承 model generation config | 旧 evaluate 显式 greedy、num_beams=1，decode 新生成 tokens；正式 P 中 generation 尚空 | **A**：已见实现的 greedy 方向及只取 continuation 一致；正式 beams、EOS/stop、generation config 仍为 **C**，需完整冻结 |
| GSM8K 数值评分实现 | S:364、E:155 调用 `score_gsm8k_generations`，但模块未上传 | `chloe/evaluate.py:17` 是 Smoke proposal：最后 #### 后否则最后数字，支持 Fraction/科学记数/逗号/LaTeX frac，精确有理数相等；失败判错 | **D**：无法逐规则确认相同。README 的 exact numeric 字样不足以证明抽取/容差/格式行为一致；正式评分必须用可核验脚本及正反例 |
| 正式 scorer 接口与 checkpoint 格式 | legacy E 接收 --run-dir、best/last，读取 adapter_state；suite 在进程内 final_eval，用自己的 model_state/custom bank checkpoint | F:196 需要 config/checkpoint/output 三者，校验 protocol/task_ids/checkpoint 摘要和 score；Chloe checkpoint 含 complete_adapter_state | **B**：成员 4 的 evaluate.py 不能直接作为 Chloe formal evaluator；需桥接入口、adapter plan 重建及完整冻结分支恢复，不能只改命令行参数 |

### 2.4 Code-Feedback 与 HumanEval

| 检查项 | 成员 4 实际证据 | 成员 5 实际证据 | 状态与处理 |
|---|---|---|---|
| Code-Feedback 正式来源 | 共享 training_source=null；N:86 只接用户提供的本地 JSONL，要求过滤后至少100K train+2K validation；repo/revision 标签仍引用约66.4K公开快照 | Code protocol/source/manifest 均 pending，未见正式规范化文件或独立 Smoke | **C**：没有真实100K code-only来源、文件 SHA 和处理链；不能以重复/填充补齐，也不能将任意本地 JSONL 的标签视为该 HF revision 的证明 |
| Code 过滤、多轮展开、去重与样本身份 | N:44 读取平面 query/instruction/prompt/input 与 answer/response/output/completion；排除少数明确非code标记，其余依赖 code-only-confirmed 声明；没有 messages 多轮展开和内容去重实现 | formal 只接已规范化的 query/response JSON 与规范化行号清单；旧 common 明确不实现 Code 专用路线 | **B**：原始多轮来源到两端训练输入的适配尚未接齐；code-only flag 不能代替可复核过滤规则，需保留原始行/轮次映射及唯一性统计 |
| Code 训练参数与 Prompt | S 复用统一训练参数，Prompt 来自规范化本地文件；无正式文件/运行配置 | P:145 起均待确认 | **C**：不能从 math 配置或 README math 示例推导 Code epochs、长度、模板和 loss |
| HumanEval 题集 revision、完整 task_id | N:123 HF `openai/openai_humaneval`，revision=None，必须 allow-unpinned；S:340 同样记录 null | P:184 起 revision/count/task_ids 全空 | **C**：题集未固定，未交付完整 task_id、test/reference 内容校验；需同时锁评分器所用问题集，不能只锁生成 prompts |
| HumanEval Prompt | N:130 原样取 row.prompt，不加聊天包装；S 的通用生成可左截 prompt | 正式 evaluation_template/few-shot/EOS 为空 | **C**：成员 4 有明确实现候选；需确认保留缩进、末尾空白、BOS、截断以及原样 function completion 规则 |
| HumanEval 生成及 completion 清理 | S:305 greedy、每 row 一条 continuation；max_new_tokens 必填；是否去围栏/截后续函数等藏在缺失 code_metrics 中 | 正式 decoding、samples_per_task、stop、completion extraction 未定 | **C**：不能将单样本生成文件视作已有 pass@1；须确认停止规则、清理、每题样本数、全部任务覆盖和随机策略 |
| pass@1 计分脚本与估计 | S:371/376 引用 write_humaneval_samples / run_official_humaneval；默认分支 S:379 明记 pass@1 not computed；独立 evaluate_humaneval.py 缺失 | F 委托外部评分器，当前 path/sha/argv 未填 | **D**：官方 wrapper、本地入口都未交付；无法验证 k=1、估计公式、缺失/重复任务分母和逐题执行结果 |
| 执行环境、超时与实际执行 | S:96 timeout 默认3秒；--confirm-isolated-sandbox 只是参数确认；R:220–230 描述隔离及上游执行开关，但没有实际环境锁/镜像/验收日志 | P 的 environment_lock、timeout、isolation 为空，Code smoke 未通过 | **C**：仅有文档说明，未核验实际安装的 human_eval 版本和代码执行是否开启；需锁 Python/依赖/runner commit、CPU/内存/超时/隔离方式，并用已知正确/错误/超时 completion 验收 |

## 3. GitHub 实际文件核查

八个指定文件全部存在；但“入口存在”不等于可执行交付完整。对三个入口做 AST import 与 main tracked tree 的静态比对，以下内部依赖缺失。它们是模块级导入，因此**即使只运行 suite 的 math 子任务，也会受 code/commonsense metrics 缺失影响**。没有通过执行脚本来触发训练验证。

### 3.1 缺失的依赖模块（D）

| main 中缺少的实际路径 | 引用证据及影响 |
|---|---|
| `src/e2lora/adapters/generation.py` | S:28、T:32、E:21；adapter 安装、序列化与恢复不完整；IMPLEMENTATION_MATRIX:26 声称复用但该文件不存在 |
| `src/e2lora/adapters/generation_methods.py` | S:29；method registry、目标发现、安装、参数计数和 optimizer builder 缺失；IMPLEMENTATION_MATRIX:15 明列 |
| `src/e2lora/tasks/generation_metrics.py` | S:44、T:48、E:30；数学抽取与评分缺失 |
| `src/e2lora/tasks/code_metrics.py` | S:43；HumanEval serialization/official harness wrapper 缺失；IMPLEMENTATION_MATRIX:19 明列 |
| `src/e2lora/tasks/commonsense_metrics.py` | S:42；模块级导入阻断 suite，虽不直接属于 Chloe 消融，仍是入口依赖；IMPLEMENTATION_MATRIX:18 明列 |

### 3.2 文档/默认入口引用但缺失的配置和脚本（D）

| main 中缺少的实际路径 | 引用证据 |
|---|---|
| `experiments/s/configs/table2_nlg.json` | IMPLEMENTATION_MATRIX:12 |
| `experiments/s/configs/table14_commonsense.json` | IMPLEMENTATION_MATRIX:13 |
| `experiments/s/configs/table15_larger_models.json` | IMPLEMENTATION_MATRIX:14 |
| `experiments/s/configs/llama31_math_lora.json` | T:77 默认 load_config 路径 |
| `experiments/s/configs/llama31_math_e2lora.json` | 同上，另一 method 的默认路径 |
| `experiments/s/evaluate_humaneval.py` | IMPLEMENTATION_MATRIX:9、R:225 给出的实际命令 |
| `experiments/s/synthetic_smoke_test.py` | REPRODUCTION_AUDIT:18 声称执行过的测试 |
| `experiments/s/suite_synthetic_smoke_test.py` | REPRODUCTION_AUDIT:19、IMPLEMENTATION_MATRIX:10 |

`experiments/s/configs/` 在该提交中整个不存在。即使补上这些文档所列配置，也必须检查内部是否仍含 null/示例值；文件存在本身不等于双方确认。

### 3.3 文档承诺但尚未交付的实际运行产物（D）

R:258 声称每次运行会产生 `run_config.json`、`sample_manifest.json`、`parameter_counts.json`、`training_history.json`、checkpoints、`result.json` 及 E² 的 `adapter_plan.json`，并列出 final-evaluation 下的 dataset manifest、generations 与逐题结果。本次 main 中没有可归属于成员 4 正式文本生成实验的这些产物；也没有固定名称的数据 manifest、HumanEval samples、完整任务 ID 或环境锁文件。

这些通常可以通过双方可访问的带 SHA256 的存储交付，不要求将大数据或权重提交 Git。但当前 GitHub 文件中没有可供本次核对的清单/摘要/定位说明，不能据“运行时会生成”判定已交接。

### 3.4 确实存在的文件及其限制

- 数据模块 `dataset/{loading,text_generation,nlg_suites,commonsense_reasoning}.py`、`model/pretrained.py`、`adapters/{adaptive,lora_layers}.py` 存在；并非全部共享实现缺失。
- `tasks/` 中只有 `classification_metrics.py`，不能替代缺失的 generation/code/commonsense 评分模块。
- 五份 `experiments/s/outputs/preflight/math_*.json` 确实存在：Llama3.1 10K/100K、Llama2 100K、Qwen3 100K、GPT-OSS 100K。它们是 token 长度统计，不是训练结果或逐样本清单，全部 `selected_max_seq_length=null`；无 Code/HumanEval 验收产物。
- `pyproject.toml` 只有 `hf` optional extra（transformers/datasets 范围），没有 R:122/124 所写的 `nlg`、`humaneval` extras，也没有评分环境锁。按 README 安装这些 extra 不会得到文档所承诺的整套额外依赖。
- REPRODUCTION_AUDIT:37 仍说没有 push、远端 README 是 placeholder，与本次 main 快照不符，属过时说明。其“测试已通过”等描述在测试文件和日志未交付前，不能当作当前提交可复验的证明；语法检查也不能发现全部缺失 import。

## 4. 本地数据实物核验

### 4.1 Chloe 的 100K 候选清单

只读检查现有 `outputs/chloe/data/math-100k-seed42.json`，没有重新生成或改写：

| 项目 | 实测值 |
|---|---|
| 状态 | `candidate_pending_team_manifest_alignment` |
| 源文件 | `/home/msds/yahui003/e2lora-storage/datasets/metamathqa/MetaMathQA-395K.json` |
| 源行数 / 采样数量 / 唯一行号数 | 395,000 / 100,000 / 100,000 |
| 源 SHA256（重新读取文件计算，匹配清单及下载记录） | `fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725` |
| 原始文件下载 commit | `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18` |
| 清单文件 SHA256 | `32aa1d172728a66da282f480f7cf7d2262d8620ae606dc19d2f34f612d21d56b` |
| `indices_sha256`（Chloe digest，JSON 序列化约定） | `3c297cb3bb2908c36a7af7e2a9b98ef138f13ae0b04af00af5a4de68d390c7c6` |
| 采样算法核验 | 与 `Random(42).sample(range(395000),100000)` 完全同序 |
| 前 10 个 0-based 行号 | `[335243,58369,13112,388786,144194,128393,117026,73158,386123,53736]` |

这证明 Chloe 候选清单内部一致；**没有证明它与成员 4 的实际清单一致**。suite 的 row hash 计算对象是规范化后的 row，legacy manifest 的 row hash 是原始 row，Chloe 的 indices_sha256 又是行号列表摘要，不能把这三类 hash 直接相等比较。交接时需声明同一 canonical schema，分别比较源行顺序、完整训练 ID 顺序、验证 ID、规范化文本及其哈希。仅比较前 10 项不够。

suite 对 query 使用 strip，legacy 使用 rstrip。本地所选 100K 中，查询因这两者产生差异的数量为 0；Chloe 模板格式化未自动 strip，响应也未统一 strip，仍需冻结规范化规则，尤其 Code 不能随意破坏缩进。

### 4.2 已上传长度统计提供的证据

`experiments/s/outputs/preflight/math_llama31_8b_full.json` 报告：100K，answer 含 EOS 最大 1,705 tokens，prompt+answer 最大 1,815 tokens。候选1024时有62条 answer 单独不能容纳，132条总长度超过上限；候选2048/4096时报告的 answer/prompt 截断风险均为0。

这些数字说明不能把 Chloe Smoke 的 max_length=256 或旧资源情景的1024当正式值；2048可作为待确认候选。统计采用成员 4 的原始 query+换行模板，且 `validation_count=0`，没有覆盖训练时另抽的2K validation，也不代表 GSM8K 生成上限已经足够。正式模板、tokenizer、源行或验证集变动后需对应复核。

### 4.3 模型与 GSM8K 实物

本地 Llama 目录存在四个 safetensors 分片；四片及 config/tokenizer/generation_config 的 HF 下载 metadata 均登记模型 commit `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`。检查的是记录和文件存在性，没有重算全部权重 hash，也没有加载模型。

本地 GSM8K Parquet 存在，文件 SHA256 为 `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`。该文件位于 HF revision `740312add88f781978c0658806c59bc2815b9866` 的 main/test 缓存路径。未联网下载成员 4 指定的 OpenAI JSONL，因此本次不宣称两份数据逐题等价；需要对 canonical question/answer 及题序作比较。

## 5. 评分交接的直接阻断

Chloe 的正式入口并不是调用旧 Smoke `evaluate.py`：F:196 读取已确认的 evaluator path/SHA256/argv，要求 `config_checkpoint_output_v1`，并在 F:204 核验 `run_type/task/protocol_sha256/checkpoint_sha256/task_ids_sha256/evaluation_examples/metric/score`。

成员 4 的 legacy `evaluate.py`：

- 接收 run-dir 和 best/last，读取自己的 run_config/adapter_plan；并非 Chloe compiled config。
- 需要自己的 `adapter_state` checkpoint，而 Chloe 正式 checkpoint 用 `adapter_state` 加 `complete_adapter_state`、plan 和 fingerprint；suite 则又使用不同的 model_state/save_kind。
- 不输出 Chloe 所要求的 protocol/task_ids 摘要与统一 score 字段；suite 内置 final_eval 也没有这个独立接口。
- 缺实际 numeric scorer 和 adapter 模块，当前无法验证正确/错误题的评分一致性。

因此不能把 `experiments/s/evaluate.py` 的路径填进 P 后就宣称对齐。需要交付或实现可复核桥接器：读取同一确认协议，按 Chloe plan 重建各类 adapter，恢复包括冻结分支的完整状态，使用同一生成/评分模块，输出完整逐题记录及约定摘要。恢复部分冻结方案时，仅加载可训练参数会得到错误模型；已存在的 `train.load_complete_adapter_state` 是 Chloe 的恢复入口之一。

HumanEval 还必须核验“实际执行过”的记录。R 中关于上游执行开关和隔离的说明是文档陈述，本次没有安装或检查 human_eval 包，不替文档背书某个实际安装环境的开关状态。应由双方交接的具体版本、已知正确/错误/超时样例日志证明执行和评分有效。

## 6. Table 4 / Table 5 / Figure 5 特有阻断

1. **Table 4：机制之外的条件尚未冻结。** table4_base_rank、main_sampling_steps、LSS/HRA 的冻参与初始化控制、Cosine/L1/L2/KL 的定义仍为空。成员 4 suite 无这些替代指标的配置，不能由 baseline 默认值推断。所有方法须保持同任务训练/评价协议，只改变所研究机制。
2. **Table 5：rank 标签不能代替实际预算匹配。** Chloe 已有审计记录：LoRA r4=3,407,872、r8=6,815,744；真实 Stage A 全 A/B 可训练 E² r8=6,810,624；每投影 active139 候选为3,700,736，较 LoRA r4 高8.594%。这些为已有审计/Smoke 记录，本次未重跑。suite 采用全 rank trainable，legacy 使用另一个 total/active 精确预算策略，且 generation 实现缺失。需解释共同采用的策略、各 rank active budget、冻结分支初始化、参数去重口径和事先定义的预算差容差。共享 generation JSON 的 selected_for_reproduction 不会自动覆盖 Chloe ablation JSON 仍 pending 的状态。
3. **Figure 5：采样横轴尚不能对应。** Chloe sweep 的 steps=8…512 指 S 个逻辑 batch，总 examples=S×B；suite 的 gradient_examples 指样本数，legacy 取 manifest 前 N 项。需确认 B、微批、固定/训练 seed、nested subset、loss 权重、是否复用梯度，以及 rank sweep 固定 steps、steps sweep 固定 rank。
4. **Figure 5 Code：完整路线还缺独立验收。** 正式框架可接规范化 query/response 数据，但尚无可核验的 Code 交接数据、共同 HumanEval scorer/环境、Code Smoke。旧 `common.validate_config` 对 code 还明确拒绝；不能用 Math Smoke 替代 Code 流程验收。
5. **资源证据有限。** 既有 Math Smoke 只验证了短序列、小采样、r8 的流程。它不能证明正式长度、高 rank 或512采样步的显存、耗时、恢复能力符合资源预算。报告不采用示例 epoch/长度估算作正式开跑依据。

## 7. 需要向组员 4 确认并交付的问题

以下是待转交问题清单，本次没有自动向任何人发送消息。

1. **正式入口及版本：** Table2 对齐以 suite 还是 legacy 为准？请给固定 commit、完整依赖模块、实际配置（不能只给 README 命令）；更新过时审计说明，并交付可复验 smoke/评分验收记录。
2. **Math 数据：** 能否提供实际100K有序源行号/原始ID/行哈希、原始文件 SHA 与源行顺序证明？是否采用 data_seed42、validation_seed43 的2K不重叠验证集？请同时给验证清单，并确认是否与本报告候选清单相同。
3. **训练预算：** Math/Code 各自 epochs/max_steps、micro/accum/global batch、三次 training seeds 是否正式采用42/43/44？训练 shuffle/sampler、尾batch与断点恢复如何保证同序？
4. **更新细节：** alpha、dropout、adapter/optimizer dtype、初始化、group rank scaling、gradient clipping、packing/loss归一化具体是什么？scheduler 采用 floor还是ceil，首更新 LR、cosine floor与warmup步序如何定义？
5. **长度与预处理：** 100K长度统计 selected=null，是否正式选定2048或其他值？2K validation是否也通过同样审计？train/eval完整模板、空白/BOS/EOS、prompt cap、答案截断/保留规则分别是什么？
6. **模型选择：** 用最低验证loss、最高生成准确率还是训练结束checkpoint？验证频率、tie规则、选择集及其生成参数是什么？suite与legacy目前不一致，哪条是正式标准？
7. **GSM8K：** 请交付固定题集、完整有序ID/内容hash、完整evaluation prompt与generation config、正式max_new_tokens及触顶统计；提供实际numeric scorer、SHA256及正确/错误/无法解析/分数/逗号/科学记数等验收样例。如何证明OpenAI JSONL与本地HF Parquet内容/题序一致？
8. **Code-Feedback：** 至少100K code-only train+另2K validation的真实来源在哪里？公开约66.4K快照之外的样本如何得到？请给许可范围内可共享的定位、版本/SHA、原始行/轮次→规范化行映射、code-only筛选、多轮展开、去重及实际数量和完整manifest，说明本地文件与标签revision的关系。
9. **HumanEval 生成和题集：** 数据与harness分别用哪个commit？完整task_id/count/test/reference校验值是什么？是否直接使用原始function prompt，如何保留缩进、取completion、处理围栏/后续函数、stop/EOS？每题生成几条，采用greedy还是采样，完整参数与seed是什么？
10. **HumanEval pass@1与执行：** 请提供官方wrapper和独立入口、k/sample数/分母/缺失重复任务规则、Python/依赖锁、超时/隔离/资源配置，以及实际开启执行后正确、错误、超时程序的逐题日志。timeout=3秒是否最终确认？仅声明sandbox不能代替环境实物。
11. **正式评价接口：** 是否共同提供适配Chloe checkpoint和 `config_checkpoint_output_v1` 的driver？由谁维护完整冻结状态恢复、协议/题集/脚本hash和逐题产物？现有两套checkpoint不能直接交换。
12. **消融特有约定：** Table4的rank、主采样步数和替代相似度定义；Table5各rank实际active预算/冻结策略/容差；Figure5两个固定值、sampling batch与seed/subset/复用策略，分别如何确认？

## 8. 何时可以正式开始

| 实验 | 当前状态 | 可正式开始的必要条件 |
|---|---|---|
| Table 4（Math） | **阻断** | 成员4完整math实现/评分文件可读取；双方确认同一训练数据清单、模型/训练/生成/评分/模型选择协议；解决B项并验收；确定8方法的统一控制、rank/采样/替代指标；正式长度和adapter策略下代表配置验收通过 |
| Table 5（Math） | **阻断** | 满足Math共同条件；另解决r4↔r8及r8↔r16的真实预算/冻参问题，登记每对实际参数量与允许差距，验证完整checkpoint恢复；不能仅按名义rank宣布匹配 |
| Figure 5（Math部分） | **阻断** | 满足Math共同条件；确认两个固定值、S×B横轴、同源nested subsets与采样种子；正式长度及高rank/长采样的资源验收通过。可独立于Code按任务放行，但当前Math也未满足 |
| Figure 5（Code部分及完整图） | **阻断** | 在对应共同条件外，补齐真实Code训练来源/规范化manifest、固定HumanEval题集与官方评分执行环境、接口及正反例验收，完成独立Code Smoke。完整图需要Math和Code两条路线结果均齐备 |

放行应体现为版本化的双方确认协议、真实文件与SHA256、逐字段比较无非预期差异、`alignment.protocol_issues` 对相应任务无阻断、编译配置可核验；随后完成与该正式协议一致的必要流程/资源验收。只将status改成confirmed、只补README、只补100K数量或复用旧Smoke分数都不足以放行。

既有LoRA Job54429与E² Job54466的20步/32题结果仍只用于Smoke验收。当前没有 `outputs/chloe/formal/` 正式结果目录；`configs/formal/` 中36个待确认设置、108个重复计划不代表已有正式运行。

**当前可以继续整理交接物、补齐实现和做协议核验；本次仅完成读取审查。正式启动时间取决于上述交付与验收完成，没有证据支持现在给出确定日期或直接开跑。** 本报告不会触发GPU授权或提交作业。
