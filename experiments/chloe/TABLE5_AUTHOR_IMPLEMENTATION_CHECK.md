# Table 5作者相关仓库实现核查

核查日期：2026-11-18（Asia/Singapore）。仓库：[hhnqqq/MyTransformers](https://github.com/hhnqqq/MyTransformers)。

**结论：在本次固定检查的三个公开分支快照中，未找到对应论文的完整E²LoRA/LSS/HRA实现、Table 5四组实验配置或作者逐族Rank产物。找到了README成果条目及两条E²LoRA命令行参数声明，但这些参数没有接入变体选择、梯度准备或共享初始化路径。因此无法由该公开快照验证3.61M/7.22M，也不能把其中ShareLoRA、GoRA等实现当成Table 5的确切实验版本。**

本轮只新增本报告及[证据JSON](report/table5-author-implementation-check-v1.json)，未修改已有源码、配置、[TABLE5_PARAMETER_BUDGET_AUDIT.md](TABLE5_PARAMETER_BUDGET_AUDIT.md)或既有结果。新的GPU预检继续暂停；未下载模型权重、未加载模型、未运行正式训练、未Push。

## 一、获取方式与证据范围

独立临时目录：`/tmp/e2lora-author-check.gsNNkG/`，不在当前项目内。

- Git副本：`/tmp/e2lora-author-check.gsNNkG/MyTransformers`。
- 固定文本快照：`/tmp/e2lora-author-check.gsNNkG/snapshots/{main,glue,LoRAEden}`，源码副本设为只读权限。
- 临时检查程序：`/tmp/e2lora-author-check.gsNNkG/verify_author_sources.py`。
- 原始检查输出：`/tmp/e2lora-author-check.gsNNkG/author-check-evidence.json`。

使用不递归子模块、无初始检出的blob过滤克隆，再仅获取源码/脚本/配置文本对象。没有安装该仓库的setup.py、依赖或训练包，没有与本项目合并、切换分支或覆盖文件。PDF说明文档及pyc文件未检出；没有获取模型权重文件。

| 分支 | 固定提交 | 该分支可达提交数 | 检查文本文件数 |
|---|---|---:|---:|
| main | `4037d4d6ba67ff88de87f90b943ff4e3a3649b67` | 449 | 148 |
| glue | `1d94fb595901ddb9cb5ec1ef1ef66b5997c1a408` | 404 | 130 |
| LoRAEden | `c8027d7af545920cd40316cbe367c82f8a56b246` | 211 | 121 |

合计检查399个文件版本、272个不同文本blob，文本内容约1.92MB。主分支提交时间为2026-03-25，glue为2025-09-12，LoRAEden为2025-02-11。证据JSON保留各文件SHA256及各分支提交，所有导出内容也按Git blob对象ID校验。

另检查了全部已克隆分支共451个可达提交的元数据、222个历史路径名，没有找到带E²LoRA/LSS/HRA特征的历史代码路径。这里**没有穷尽每个历史版本的文件正文，也没有检查私有、已删除、未公布或PR专用refs**。副本中未取得tag；这不是对所有远端tag的独立穷举声明。“未找到”限定于上述明确范围，不能推出作者不存在其他实现。

## 二、已经确认的内容

### 1. README建立了关联线索，没有建立实验版本身份

主分支`readme.md:35–40`列出研究成果，E²LoRA在第40行。[固定提交README](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/readme.md#L35-L40)

该条目没有指向E²LoRA专用模块、复现命令、Table 5配置或论文实验commit。结合本轮源码检查，能确认这是作者相关工具库的关联线索；不能仅凭该条目认定它是论文确切实验版本。

### 2. 找到E²LoRA参数声明，但没有找到运行时接线

`common/parser.py:365–366`声明`--use-e2lora`和`--e2lora-similarity-threshold`，后者默认0.85。[参数声明](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/parser.py#L365-L366)

主分支相关关键词命中仅为上述两行及README第40行。其他两个分支没有E²LoRA/LSS/HRA/RMI/共享区间等对应命中；所有Python源码中均未发现对`args.use_e2lora`、`args.e2lora_similarity_threshold`或同名标识符的运行时读取。

运行路径交叉核对：

| 路径与行号 | 实际内容 | 对E²LoRA的意义 |
|---|---|---|
| `common/lora_modules/lora_set_up.py:70–346` | 47种变体的注册字典 | 没有`use_e2lora`注册项 |
| 同文件`:355–384` | 遍历该字典选择实现类，默认类为`LinearWithQLoRA` | 单独打开E²LoRA标志不会选择新实现 |
| 同文件`:563–605` | `use_lora`控制模型替换及可训练名配置 | 没有E²LoRA专用安装/分组处理 |
| `common/lora_modules/__init__.py:69–81` | 判断是否需要共享权重初始化 | 不检查E²LoRA标志 |
| 同文件`:134–182` | 梯度重初始化、rank allocator及共享权重准备 | 注册了GoRA、AdaLoRA等路径，没有E²LoRA/LSS/HRA路径 |

源码链接：[变体注册及分发](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/lora_set_up.py#L355-L384)、[共享入口](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/__init__.py#L69-L81)、[梯度准备入口](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/__init__.py#L134-L182)。

临时CPU检查抽取并执行了两个已经审阅的纯选择函数，使用桩替换实现类、注册表值及日志，不导入训练包：E²LoRA标志关闭/开启时分发结果相同；它也不会触发共享初始化。这个结果只验证分发逻辑，**不代表完整训练入口已经可运行**。

因此，不能把默认阈值0.85说成作者论文实际采用的固定阈值；它在所检查代码中没有消费者，也不能据此替换我们的自适应阈值。

### 3. 相近功能确实存在，但属于其他方法

| 文件与行号 | 确认存在的功能 | 不能据此推断的内容 |
|---|---|---|
| `share_lora.py:102–157`、`:160–171` | 按所有共享模块的最大输入/输出维度创建全局A/B，广播同一Parameter；支持A/B/AB | 不是LSS连续区间发现或HRA分配实现，不能自动归到普通E²LoRA |
| `share_lora.py:55–66` | 前向访问共享矩阵的形状/Rank切片 | 仅切片不等于物理Parameter缩小或冻结 |
| `gora.py:38–46`、`:278–279` | Gaussian梯度熵辅助函数及`grad_entropy`重要性选项 | 有熵辅助函数不等于实现了RMI、LSS及区间max-HRA |
| `gora.py:309–377` | 按模块特征维度累计参数预算，分配模块Rank，并可限制最大/最小Rank | 这是GoRA路径，不能代替论文公式（7）的E²LoRA执行证据 |
| `hira.py:1–5`、`:21–27` | ICLR 2025 HiRA的Hadamard更新 | HiRA不是论文的Heterogeneity-based Rank Allocation（HRA） |

固定提交源码：[ShareLoRA](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/share_lora.py#L102-L171)、[共享切片](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/share_lora.py#L55-L66)、[GoRA预算](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/gora.py#L309-L377)、[HiRA身份](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/hira.py#L1-L5)。

主分支未发现139-active-Rank、E²LoRA专用partial-rank或区间训练Rank预算规则。一般LoRA-FA选项确实可以只启用B，但它是独立方法：`lora_set_up.py:607–626`中普通回退列表为A/B两者，LoRA-FA条件才只取B。[可训练参数名选择](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/lora_set_up.py#L607-L626)

### 4. 没有找到Table 5四份对应配置

| 要核查的实验 | 本次实际发现 | 是否识别为Table 5配置 |
|---|---|---|
| LoRA r4 | 通用LoRA类可接受r4；不是论文运行配置 | 未找到 |
| LoRA r8 | 全局parser默认r8；Math r8示例显式启用GoRA和LoRA+ | 未找到普通LoRA Table 5身份配置 |
| E²LoRA r8 | 只有上述E²LoRA参数声明，没有实现或对应脚本 | 未找到 |
| E²LoRA r16 | 没有实现或对应脚本 | 未找到 |

具体Math脚本为`scripts/gora/llama3.1_metamathqa.sh`：第15行指定wq/wk/wv/wo，第17–29行启用LoRA+、GoRA及rank8，第36行实验名也标GoRA，第53行最大长度1024，第75行启动8卡训练。**这些是已检查的GoRA示例，不能当成E²LoRA Table 5配置。**[脚本证据](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/scripts/gora/llama3.1_metamathqa.sh#L15-L36)

旧LoRAEden分支存在DNA/GUE及pipeline的rank16示例，未标识E²LoRA或Table 5；所以本报告并不声称仓库中完全没有数字16或rank16脚本。没有发现这四项实验的完整配置组合、逐族共享区间、分配Rank JSON、精确参数统计日志或Table 5结果清单。

### 5. 一般参数计数口径可以核对

`common/utils/params_manager.py:12–22`遍历`model.named_parameters()`，仅累加`requires_grad`参数的`numel()`；第4–8行以十进制M保留两位小数。这是物理唯一可训练Parameter计数，不是逐层重复计数，也不是按前向激活坐标折算。[计数与显示](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/utils/params_manager.py#L4-L22)

该文件第36–46、63–73行按enable/disable名称设置训练标志；`lora_set_up.py:599–601`接入A/B名称列表。没有找到E²LoRA专用“约减半”统计规则。[训练标志设置](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/utils/params_manager.py#L36-L73)

此通用格式化函数支持此前审计使用十进制M和两位显示精度的解释，但**尚不能确认Table 5确实使用该函数或该提交生成**。

## 三、CPU／静态计数能确认什么

一般LoRA在`common/lora_modules/lora.py:87–95`创建A=`[r,in_features]`、B=`[out_features,r]`，二者均为可训练Parameter。[A/B构造](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/common/lora_modules/lora.py#L87-L95)

该仓库`model/llama/config.py:11–15,30–37`给出32层、hidden4096、32 query heads、8 KV heads；`model/llama/model.py:115–134`构造对应Q/K/V/O。因此在只适配这四投影且训练完整A/B的假设下，可直接静态计算：

`P = 32 × r × [(4096+4096)+(4096+1024)+(4096+1024)+(4096+4096)]`。

| r | Q | K | V | O | 合计 | 两位M显示 |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 1,048,576 | 655,360 | 655,360 | 1,048,576 | 3,407,872 | 3.41M |
| 8 | 2,097,152 | 1,310,720 | 1,310,720 | 2,097,152 | 6,815,744 | 6.82M |

模型形状来源：[Llama配置](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/model/llama/config.py#L9-L37)、[投影构造](https://github.com/hhnqqq/MyTransformers/blob/4037d4d6ba67ff88de87f90b943ff4e3a3649b67/model/llama/model.py#L115-L134)。这只是普通LoRA的形状一致性检查，不是作者Table 5运行重建；通用模型配置还允许MLP目标，实际目标仍需以实验配置为准。

**E²LoRA r8/r16的作者可训练参数量均记为null／未验证。** 没有对应共享区间、Rank或安装路径，不能合法构造作者模型并声称得到3.61M/7.22M。本轮没有自行补写缺失算法，没有用GoRA、ShareLoRA、rank减半、固定139或冻结A/B来反向拼接这两个数字。

## 四、与现有预算审计逐项对照

| 核对项 | 本项目此前已确认的行为 | 作者相关公开快照提供的证据 | 本次结论 |
|---|---|---|---|
| LSS区间 | Q/K/V/O分别8/8/5/3组，全部24组已记录 | 没有对应区间发现实现或输出 | 无法与作者逐组核对 |
| HRA预算 | 每族32×base_rank，逐区间max熵归一化及round | 没有E²LoRA分配路径；只有其他方法的rank allocation | 不能认定作者用了更小预算或K×r |
| A/B共享 | 每个区间共享同一套A/B，跨区间独立 | 一般ShareLoRA有全局A/B及切片功能，但未接E²LoRA | 不能用其全局共享拓扑替代普通E²LoRA |
| 冻结／有效Rank | 当前全部组A/B训练，无partial后缀；139仅候选 | 一般LoRA-FA/VeRA等有独立冻结规则，无E²LoRA规则 | 不支持为匹配Table 5新增冻结 |
| 参数统计 | 按唯一Parameter身份、requires_grad、numel | 通用打印器也按named_parameters/numel，M保留两位 | 没有发现已证实的特殊折半口径 |
| 数值差额 | r8=6,810,624、r16=13,631,488 | 作者E²LoRA实际计数未找到 | 原差额3,200,624／6,411,488相对论文显示中心仍未解决 |

本项目明确代码定位：[adaptive.py:86](../../src/e2lora/adapters/adaptive.py#L86)的LSS、[adaptive.py:106](../../src/e2lora/adapters/adaptive.py#L106)的Rank分配、[preparation.py:115](../../src/e2lora/adapters/preparation.py#L115)的族预算记录、[lora_layers.py:55](../../src/e2lora/adapters/lora_layers.py#L55)的安装/共享、[common.py:159](common.py#L159)的唯一参数报告。完整区间及数值仍以未改动的[TABLE5_PARAMETER_BUDGET_AUDIT.md](TABLE5_PARAMETER_BUDGET_AUDIT.md)为准。

此前关于“按当前逐族N×r预算与完整A/B时参数规模近似守恒”的数学结论不因这个README线索而改变。本轮没有获得足以确认或修订作者实际预算规则的新实现证据。

## 五、尚未确认

1. 这个公共仓库的哪个commit、分支或外部模块是论文确切实验版本；是否还有尚未公开、未检索到的实现。
2. E²LoRA参数声明是否只是不完整接口，或者实际实验代码位于其他位置。
3. 作者普通E²LoRA的预算域、Rank标签映射、A/B共享拓扑、切片/冻结策略及逐族实际Rank。
4. Table 5的精确整数参数、seed间聚合方法及计数器；通用库的打印器是否用于该表。
5. 3.61M/7.22M如何由完整可运行配置得到。此次调查没有验证这两个数字。

## 六、需要咨询作者

- 请提供Table 5对应的仓库、精确commit/tag、E²LoRA模块、LSS/HRA调用入口及必要依赖；说明`--use-e2lora`在实验版本中的实际接线。
- 请提供LoRA r4/r8和E²LoRA r8/r16的完整启动配置，特别是目标投影、rank标签与r_vanilla的关系、总预算及归一化范围。
- 请提供两种E²LoRA配置各seed的Q/K/V/O区间、区间熵、分配Rank、实际使用Rank、A/B形状、Parameter共享身份及requires_grad列表。
- 如果使用了全局共享、动态切片、mask、冻结或预算后处理，请提供对应源码及初始化规则；这些不能由现有通用ShareLoRA/LoRA-FA选项推定。
- 请提供3.61M/7.22M的原始参数统计日志及精确整数，说明是物理唯一训练Parameter、有效坐标数，还是其他统计口径，以及跨seed如何汇总。

本轮没有通过邮件、Issue或其他渠道联系作者。

## 七、验证与保留状态

9项临时静态／CPU断言检查通过：399份固定快照内容校验；精确关键词范围；47项注册表检查；Python运行时参数引用检查；带／不带E²LoRA标志的分发对照；共享初始化谓词对照；准备路径注册检查；普通LoRA形状预算核算；127个既有本项目文件SHA不变。

这不是新增的GPU实测或正式训练，也没有导入作者整个训练包。已有105项CPU测试记录未改写；本轮因无现有代码改动，没有重跑整套训练相关测试。可复核证据保存在[table5-author-implementation-check-v1.json](report/table5-author-implementation-check-v1.json)，含固定提交、源文件SHA、关键词行号、CPU检查结果及保护清单。临时目录若被系统清理，仍可用固定提交链接和SHA重新核查。

**最终状态：未找到论文对应的完整实现及Table 5配置；3.61M/7.22M来源尚未确认，需要作者源码或补充说明。现有算法、HRA、Rank、冻结规则和正式实验门禁保持不变。**
