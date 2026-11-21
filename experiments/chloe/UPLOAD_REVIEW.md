# 成员5源码交付审阅

源码包准备阶段没有启动实验或修改SSH仓库；随后成员5明确授权写回自己的分支并上传GitHub。源代码先在独立本地副本中审阅测试。当前SSH分支仍为chloe-ablation，读取时HEAD=d13dc42784dea9c3114e66f53606874f0ab41df8；GitHub main读取时为722f013e71d5e59d214cac1c4c2fcb0c069e0771。只下载源码/小配置核对，未下载其他成员checkpoint。

## 原代码缺口与补齐

1. 已完成实验的formal_math.py仅支持LoRA r4/r8和E2r8；新增suite_train.py统一覆盖LoRA、LSS-only、HRA-only、完整E2、cosine/L1/L2/KL，已配置的r4/8/16/32/64及采样8..512扫描。
2. 原批量配置与实际训练入口未完整衔接，且存在旧pending门禁；新增suite_plan.py/run_ablations.py，把明确配置编译成可执行独立run，不再要求无关任务的配置齐备。
3. 原续跑控制器绑定56320/56321；新增watch_suite.py按JSON配置串行调度，保留job状态/锁、重试上限、进展检查及异常停止。
4. 原汇总未读取实际formal_math结果；新增summarize.py读旧正式结果和新格式，按协议区分，输出表、曲线、真实参数差，排除smoke及不完整评估。新增evaluate_suite.py可独立评估实际保存的checkpoint。
5. 新增可移植SLURM入口和查询脚本、资源JSON、依赖清单、源码导出/校验脚本、CPU回归测试；更新README完整命令/协议/结果说明。历史文档保留，但过时pending结论不能覆盖当前README。

## 与最新主分支的共享改动

- lora_layers.py与main完全一致，放入包内方便审阅，不需要重复修改。
- adaptive.py仅显式释放GPU梯度/graph临时对象、指定autograd不保留graph，公式和接口不变。
- 新增公共ablation.py、grouping.py、partial_rank.py、preparation.py、similarity.py：机制开关、DP、部分冻结、逐目标梯度采集、替代相似度。不得把团队选择描述为作者公开实现。
- 没有替换main新增的dataset/loading.py等成员公共入口。集成测试使用最新main这些文件，再叠加上述明确源码。
- 不包含其他成员的实验目录，不修改共用references JSON；待审阅方案保存在自己的configs/suite-math.json。

## 实验范围及未声称完成的部分

组内原有Math-only批准继续有效。Code/HumanEval框架历史文件保留，不声称HumanEval执行器或代码曲线已完成。未运行新的Table4/E2r16/Figure5实验，不能填写分数。

Figure5固定值和高rank预算等新增配置是显式团队方案，已在README/JSON逐项标记。默认仅seed42，43/44未自动追加。既有LoRA基线结果保留原始源文件及协议hash。CPU模拟验证不能代替真实8B新增方法的GPU短跑。

## 安全合入自己的分支

1. 保留当前SSH分支，不执行reset/clean/pull覆盖。先把未提交源码和git diff备份到仓库外；模型/数据/outputs不要纳入Git。
2. 在仓库外解压源码包，先阅读README、此说明、完整文件manifest和against-main.patch。包是源码交付，不是整个Git仓库。不要盲目覆盖当前dirty目录。
3. 审阅现有改动后再保存自己的源码提交；然后在自己的分支fetch/merge最新main。避免下载历史模型大对象，可使用团队支持的partial fetch：git fetch --filter=blob:none --no-tags origin main。不要强制推送。
4. 对照manifest合入自己目录及列出的公共适配器文件，保留main其他成员更新；README与adaptive.py如有冲突逐项合并。正式已有run的源hash对应旧入口，继续保留formal_math.py，不改写历史结果。
5. 重跑README的CPU测试；学校资产审计另加--cluster-assets。真实新方法正式启动前仍走同协议短跑，本次交付没有启动这些作业。
6. 在解压包的UPLOAD-MANIFEST.json路径执行下列只读检查，逐项核对后再自行执行它打印的精确git add命令。不要git add .，不要stage manifest/patch/outputs。

```bash
python experiments/chloe/review_upload.py --manifest /path/to/extracted/UPLOAD-MANIFEST.json
git diff --stat
git diff --cached --stat
git diff --cached --check
```

将/outputs/chloe/加入自己的.git/info/exclude，或审阅后追加到仓库.gitignore，保留main现有规则。源码包本身已按扩展名/目录白名单剔除数据、模型、pt/pth/ckpt/safetensors、缓存、日志及其他成员目录，并扫描常见Token/私钥格式。已被其他成员提交的大文件不能靠.gitignore自动移除，本次也没有改动它们。

CPU验证的准确版本、通过/跳过数量及包校验见随包VALIDATION.md；GPU/SLURM新作业未执行。成员5后续已明确授权commit/push；仅允许自己的源码范围，保留其他成员更新。

2026-11-21上传前复核：main已前进到ae243032b96b175d1a9f83fe3d37e3d515449c39；新增提交仅涉及experiments/acheng/，不与本交付源码重叠。上传时仍应再次fetch并检查。
