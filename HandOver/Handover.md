# 交接索引

首要目标是推进实验、解释结果。先看本页当前状态；没有具体故障不新增框架、审计或准备轮次。旧计划已归入[历史摘要](../docs/history/README.md)，原始L2S/S2L和机器结果保留，旧轮次不重复领取。

## 当前任务：只认本表，不扫描旧READY

用户已同意在已讨论参考≤24h＋候选整条线≤4h范围内按独立任务合理安排，[授权原话/上下文](../docs/proposals/repair_before_retraining/authorization.json)。5种最终方法、6个GPU作业，各一次无重试，最多两个单卡并发；每项独立输出/额度，不互转结余。push不唤醒服务器AI。

| 请求 | 前置/状态与安排 |
| --- | --- |
| [019](L2S-20261019-019.md) | **FAILED（托管机制通过，payload 从未执行）— 020/021 仍不得启动**。[S2L-019](S2L-20261019-019.md)：同步`612afea`、纯CPU、一次无重试。**前置全部可用**：systemd 249、`XDG_RUNTIME_DIR=/run/user/1018`、用户管理器可访问；**Linger `no` → 授权内`enable-linger`(rc=0) → `yes`**（标记`/var/lib/systemd/linger/wangguangjie`）。**机制侧已确证**：`backend=systemd`、`unit=mtad-a30b8d0d4afb.service`、`Transient=yes`（`/run/user/1018/systemd/transient/`）、由**`systemd[1911324]`用户管理器**启动（对照：AI 自身仍在`session-9554.scope`）、PID 三分离（launcher 1915804／unit MainPID=worker 1915807／payload 1915808）、`RuntimeMaxSec=26.99s`/`TimeoutStopSec=3s` 生效、`CUDA_VISIBLE_DEVICES=""`、无残留进程。**但 `status.json=FAILED`、`returncode=1`、`exit_code.txt=1`、`stdout` 0 字节**：原因是 **L2S-019 第31行 payload 单行 python 多了一个 `)`** —— 把 `request.json` 中 launcher 实际透传的原文单独 `compile()` 得到**完全相同**的 `SyntaxError: unmatched ')' (offset 342)`，故错在请求文本、不在 launcher/systemd；`time.sleep(5)` 与两个标记写入从未发生。**`probe-start.json`／`payload-survived.txt` 未产生**（故 payload 自身 cgroup 无直接证据）；**`ControlGroup` 返回空、`MainPID` 返回 0**（unit 已退出被回收），按请求**保留实际输出、未补值**。**未 `reset-failed`**（为保存取证现场，transient 单元定义 17 行已复制回传）；**未重跑、未改命令使其变绿**；扫描唯一命中为参数名 `--no-ask-password`，无凭据。**证据层级**：已证"由用户管理器独立 transient service 承载、主进程与提交者分离、worker 自行写出记录"；**未实测跨过 AI 运行时重启**（本轮不要求、也未终止其他会话）。**建议**：删掉那个多余括号后**重新发布 019**（或明确授权修正后执行一次）；机制侧无需改动。回传 13 文件 / 4,492 B。 |
| [020参考](L2S-20261019-020.md) | READY，**019通过才启动**。TCM1000/TAM1000/TFM300/JOINT60＋完整终评，固定≤86400秒。 |
| [021共享](L2S-20261019-021.md) | READY，**019通过才启动**，可与020并行。teacher1000→一次身份V→utility200，只训练一次，固定≤9600秒，完整端点供四臂读取。 |
| [022主方法](L2S-20261019-022.md) | READY，**021完整COMPLETED才启动**。attribute-u适配300＋V，固定≤1200秒。 |
| [023普通融合](L2S-20261019-023.md) | 同一021端点，plain适配300＋V，固定≤1200秒。 |
| [024统一效用](L2S-20261019-024.md) | 同一021端点，scalar-u适配300＋V，固定≤1200秒。 |
| [025普通门控](L2S-20261019-025.md) | 同一021端点，learned-attribute适配300＋V，固定≤1200秒。 |

候选总上限9600＋4×1200=14400秒，**不是每臂4小时**。正常顺序021→022→023→024→025，优先主方法与普通融合；020占一卡时候选一次跑一项，020结束后可在两卡上跑两个已具备前置的臂。服务独立托管，AI可断开；重连读取同一任务结果后领取下一项，不重复启动。021失败则四臂不读部分产物；单臂失败其他已满足前置的臂继续，020失败不阻塞候选。

四臂共用021冻结执行目录/同一shared.pt、normalizer/S、身份records和阈值，独立初始化及seed22采样，不继承上一臂。既有full入口保留，新shared/arm入口已实现；本地CPU等价验证与完整配置见[安排与修复](../docs/proposals/repair_before_retraining/README.md)。正式任务只用上述对应JSON/L2S，不执行预算null旧草案。

**从当前待办撤销：** 006旧查询、001—018重新领取/重试/续预算、017续训/补终评或为此取checkpoint、重复012/016取件/原ZIP审计/smoke/prepare/成本测量、CDF训练及本轮额外TSA训练。保留原件，不改历史L2S/S2L的READY头部；019原请求的“无新GPU READY”是其发布时快照，后续队列以本表和新授权为准。

## 017/018已结案

[017](S2L-20261018-017.md) FAILED，已审服务器ed1c954；约5.46h时外部SIGTERM中断，未到预算，融合终评没有开始。300条history与cursor302日志落盘差异已修，原checkpoint耐久进度未本地读取不猜。25步均值位置loss总体下降，不能接受不同batch首末“位置变差”的定性。[018](S2L-20261018-018.md) COMPLETED但基础恢复/身份仍弱、无稳定效用专属收益。两轮额度关闭，详见[终态分析](../docs/proposals/repair_before_retraining/README.md)与[018报告](../docs/results/cooperative018_local_review/README.md)。

## 评价与数据边界

017原件保留，实际仅TCM/TAM/TFM各100、5次关联V、一次margin校准；外层真实退出码缺失不补造，完整物理评价未产生。失败日志缺口已在新代码修复，不能反改旧产物。新训练明确实际selected组成；关联与物理误差、oracle/预测分组/单源/简单融合、全人口覆盖和共同点分别分析，不拿变批次训练loss当验证收敛。

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
