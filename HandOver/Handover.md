# 交接索引

首要目标是推进实验、解释结果。先看本页当前状态；没有具体故障不新增框架、审计或准备轮次。旧计划已归入[历史摘要](../docs/history/README.md)，原始L2S/S2L和机器结果保留，旧轮次不重复领取。

## 当前任务

| 任务 | 状态与下一步 |
| --- | --- |
| [017请求](L2S-20261018-017.md) / [回执](S2L-20261018-017.md) | **终态回传完成：`FAILED`**（`run.json` 原件 `status=FAILED`、`failure_type=InterruptedError`、`failure="SIGTERM: preparation/fit interrupted, not completed"`）。**更正本地前提：原10小时上限并未到达**——`started_at 12:42:56Z`、终止于 **`18:10:30Z`**（文件 mtime 与 `run.elapsed_seconds` 19,644 秒一致），实际存活 **19,654 秒 ＝ 上限 35,970 秒的 54.6%**，比外层 TERM 时刻 22:42:26Z **早 16,316 秒（4 小时 32 分）**；也不是 OOM（OOM 用不可捕获的 SIGKILL）。**外层 `exit_code.txt`/`ended_at.txt`/`elapsed_seconds.txt` 三份均未生成**，说明包装 shell 与作业一同被带走，**真实退出码无法从原件取得、服务器不填推测值**。最佳解释为托管会话运行时重启（原 worker 目录与其 task 输出已不存在、机器本身未重启 uptime 26 天、无残留进程），`dmesg`/`journalctl -k` 无权限，故标注为推断。**已完成**：TCM100（1.945秒/步）、TAM100（18.374）、**TFM100（12,361.10秒、123.611秒/步、单步98.67–182.17）**、margin 一次校准 COMPLETED（1,719.93秒、`effective_margin=71.58904`）、5/6 次验证（合计2,719.76秒，**selected＝TAM@100 macro-F1 .432985**，此前 TAM@20 .409805）。`history.json` 实为 **300 条**，而 `cursor` 写 302（stage3/update2）、`coverage.JOINT` 全空——**两条 JOINT 更新未落入 history/coverage，服务器不把 302 当已验证更新数**。**未产生**：第6次验证、JOINT 余下更新、**整个完整终评**，故 `summary.json`/`evaluation-progress.json`/`predictions/`/任何逐点物理数组/任何融合误差与覆盖**一概不存在**，`summary.selection` 与终评阈值一致性核对无法执行。不变量仍成立：`test_opened=false`、`resume_from=null`、`fusion_output_parameterization=train_position_affine`。**TFM 分项值得本地注意**：总 `fusion_loss` 9.3998→4.4172，但 `lat` .4326→**.7652**、`lon` .5454→.6251、`cou` 1.0346→**1.5460** 三个分量**变差**，降幅几乎全来自 `vel` 11.2420→**0.4138**。原件已回传（`files/run.json` 542,707 B、`history.json` 323,511 B、`coverage.json` 1,121,952 B、`preparation_extract.json` 人口按实际字段聚合＝512/64/1635/29008/377/18364、缺键记未提供）；`preparation.json` 122,492,222 B 与 `checkpoint.pt` 124,049,936 B 留服务器原路径未改。**下一步归本地**：继续这 18 步 JOINT＋终评需**新授权**（017 额度与"一次无重试"已随终态关闭，服务器未 resume/未重跑/未中止他人进程）。 |
| [018请求](L2S-20261018-018.md) / [回执](S2L-20261018-018.md) | COMPLETED，600更新、2371秒；[本地分析](../docs/results/cooperative018_local_review/README.md)已完成。基础恢复/身份仍弱，效用无稳定专属收益，预算关闭。 |
| [候选修正版](../docs/proposals/candidate_revision_after018/README.md) | 用户已批准本地实现；代码/CPU验证完成。观测位置底座＋残差、log速度、圆周航向、支撑query、局部关联、四臂属性残差融合。新GPU授权0、无READY；[草案](../docs/proposals/candidate_revision_after018/config.draft.json)共2400更新、max_seconds=null，3.11h外推/4h建议不是授权。 |

**本轮只整理仓库。** 源码归入data/models/training/evaluation/utils，删除退役pilot训练、profile、旧环境/SHA验收及重复测试；原始实验配置和证据不改。四个[实验CLI](../tools/experiments/README.md)仍是现行入口。清理验证见[历史与清理记录](../docs/history/README.md)，不把代码整理或CPU检查当MTAD结果。

**当前服务器只需完成017原件回传，不是新实验。** 10月19日10:30（北京时间）fetch无新提交。原`started_at.txt=2026-10-18T12:42:56Z`，按启动标记及原命令推算，外层35970秒TERM约在次日06:42:26，含30秒清理的总上限约为06:42:56；目前仓库未收到`exit_code.txt`、`ended_at.txt`、`elapsed_seconds.txt`或完整终态结果，仅有00:39的中期快照。本地不能据缺回执宣布成功、失败或仍在训练。

服务器继续原017收尾：核本轮自有作业的实际状态，在自己的S2L-017记录真实结束原因和退出码，按原L2S复制已有run/history/coverage/summary/evaluation-progress及预测索引/metadata；缺失文件如实记未产生或未取得，不补造退出码，不加载模型补评价、不resume、不重训、不重开预算。如本轮作业异常仍在运行，执行原10小时上限的收尾要求，不继续用GPU补齐结果。只读整理现有产物已在原017请求中，不新增轮次或READY。

**下一步：** 取得017终态后分析完整任务或中断原因；新候选配置/预算决定后直接独立READY，不再重做已完成实现、原ZIP扫描或成本探测。两条线不以另一条好成绩为前提；各自预算已批才可执行，最多两单卡并行。push不会启动服务器AI。

## 评价与数据边界

017终态核命令/退出码/配置及run/history/coverage/summary：计划320更新(100/100/100/20)、6次关联验证、一次margin校准。selected可能不是末轮，明确实际选中组成；失败保留原件、不resume或重开预算。关联与完整物理误差、oracle/预测分组/单源/简单融合、全人口覆盖和共同点分别分析，不拿训练loss当验证收敛。

T_core512/V64、共享A隔离、cap128/Q64/time_unit20、显式offset/gap600、逐属性mask、有效短轨均保留。T_test封存。参考连续SmoothL1 beta .001、TCM后训练侧64×64负距离中位数一次校准。两系统训练量/loss/选模/singleton不同，不称严格等预算或作者精确复现。

[数据复审](../docs/results/dataset_followup018/README.md)、[外部处理对照](../docs/results/external_mtad_processing20261019/README.md)及[时间任务分析](../docs/results/time_task_analysis20261019/README.md)已完成。21分钟是工程备选而非论文标准，会排除37.49%原同身份对，未采用、不作为下一轮前置。数据已有支撑区恢复也差，不能把低分全归长外推。原值/标签/人口保留，不删异常提分。

实际输入支撑/最近点距离评价已接入两系统下一次正常评价，不为此重跑017/018。9001/reference复用限制降噪主张；mPCA为场景平均召回、p_fa为1−precision。CDF退役，TSA只作独立关联对照；旧A_full不是完整TMTAFF。

GPU允许兼容设备和共卡，不终止他人进程。双方不设SHA/hash门槛；活动目录不变，原L2S/S2L/失败产物保留。权限与防止过度工程的约定见[AGENTS](../AGENTS.md)。每6小时跟进，无新结果/无可推进事项安静。

## 已结束轮次

[016请求](L2S-20261018-016.md) / [回执](S2L-20261018-016.md)：零GPU取件与独立误差复算已完成，不重取。其余早期轮次如下，详细结论见[历史摘要](../docs/history/README.md)。

| 轮次 | 请求 | 回执 | 结论与当前影响 |
| --- | --- | --- | --- |
| ENV-001 | [L2S](L2S-20261016-001.md) | [S2L](S2L-20261016-001.md) | 初步环境/数据盘点完成；旧Python不兼容，后续已解决。 |
| ENV-002 | [L2S](L2S-20261016-002.md) | [S2L](S2L-20261016-002.md) | FAILED：依赖下载超时，后续已解决。 |
| ENV-003 | [L2S](L2S-20261016-003.md) | [S2L](S2L-20261016-003.md) | FAILED：下载超时；随后现场IPv4/镜像修复，测试问题由后续处理。 |
| ENV-004 | [L2S](L2S-20261016-004.md) | [S2L](S2L-20261016-004.md) | CPU通过，CUDA当时未执行；旧空闲整卡限制已失效。 |
| ENV-005 | [L2S](L2S-20261016-005.md) | [S2L](S2L-20261016-005.md) | CPU与真实CUDA合成验收通过；不重复smoke，不代表真实模型效果。 |
| GPU-POLICY-006 | [L2S](L2S-20261017-006.md) | 未收到独立回执 | 未领取查询由007承接，不再单独派发；不代填执行结果。 |
| NATIVE-PROFILE-007 | [L2S](L2S-20261017-007.md) | [S2L](S2L-20261017-007.md) | FAILED：prepare成功可复用；旧profile超时，仅9/200步。 |
| PROFILE-DIAG-008 | [L2S](L2S-20261017-008.md) | [S2L](S2L-20261017-008.md) | 6步诊断完成，均步152.61秒、峰值分配11.50GiB、每批零正例；原件已由009补齐，不外推v2全流程。 |
| EVIDENCE-009 | [L2S](L2S-20261017-009.md) | [S2L](S2L-20261017-009.md) | 原件与008预绑定hash核验通过；整理留痕缺口不影响已核原件、不阻塞010。 |
| FIRST-FIT-010 | [L2S](L2S-20261017-010.md) | [S2L](S2L-20261017-010.md) | COMPLETED：A0/CE/seed20共220更新，约2.02小时；终选micro-F1 .26058，完整V分数已由011回传并复算。 |
| COMPARE-011 | [L2S](L2S-20261017-011.md) | [S2L](S2L-20261017-011.md) | 三项完成，约3.68GPU小时；负结果已复算，额度关闭。 |
| EVIDENCE-012 | [L2S](L2S-20261018-012.md) | [S2L](S2L-20261018-012.md) | 原件收齐、manifest/分数已复算，不重取。 |
| TIME-COST-013 | [L2S](L2S-20261018-013.md) | [S2L](S2L-20261018-013.md) | 124超时、8更新；selected只有TFM2，完整评价由014补齐。 |
| TIME-EVAL-014 | [L2S](L2S-20261018-014.md) | [S2L](S2L-20261018-014.md) | 退出0、1902秒，短训物理尺度异常已定位；不重跑。 |
| COOPERATIVE-COST-015 | [L2S](L2S-20261018-015.md) | [S2L](S2L-20261018-015.md) | 退出0、292秒、12更新与四臂完成；身份近随机、位置相对改善伴随速度退化，额度关闭。 |
