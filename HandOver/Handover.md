# 当前交接

更新：2026-10-28。**107基线A已单次托管启动，禁止重复领取；当前无新READY。** 106/105/104/103/102已完成并核验，不重复领取。104实际范围与服务器回执过度归因的订正见[本地记录](../docs/reports/2026-10-27-three-case-chain-review.md)，105核验与深挖见[本地记录](../docs/reports/2026-10-27-raw-case-chain-review.md)。A授权不包含新增seed/步数/TAM或重新选择研究思路。

| 任务 | 状态 | 执行／回执入口 |
|---|---|---|
| 107新单位基线A | **RUNNING**；2026-10-27北京时间22:24:03单次systemd提交，冻结`e279bed`，服务`mtad-1e8d72c87900.service`、Linger=yes、Restart=no。seed20、TCM2424→TFM1000→固定056全V，各段一次，失败停后继。绑定3090物理3号卡UUID `GPU-1043f26a-d4bd-7cd8-1089-ee9066f1ffcc`、逻辑cuda:0、CPU4；提交时可用13448MiB、已有3个计算进程，共卡。**北京时间2026-10-28 00:44:45一次只读核验**：服务active/running，链与TCM阶段均RUNNING；TCM的cursor及checkpoint_cursor均记录2424/2424更新，`evaluation-progress.json`的stage为`tcm_completion`，此次未抽取场数；独立的`final_evaluation_status`为NOT_STARTED。预期路径的TFM run.json尚不存在，未核到TFM和固定056比较开始或结果；运行中的Result=success/ExecMainStatus=0不表示任务已完成。[本次只读原件](results/LEVEL1-TMTAFF-UNIT-BASELINE-A-20261027-107/read_only_status_20261028_004445.json) | [L2S-107](L2S-20261027-107.md)、[事前Idea](../docs/proposals/tmtaff_unit_baseline_A_20261027.md)、[启动原件/资源摘要](results/LEVEL1-TMTAFF-UNIT-BASELINE-A-20261027-107/)。真实run为`/home/wangguangjie/djs/association-code/runs/LEVEL1-TMTAFF-UNIT-BASELINE-A-20261027-107-attempt-1`，launcher后缀`-launch`；服务器只领取后续同一run终态核验/回传职责，禁止重建或另起TFM/评价 |
| 106新单位人口准备 | **COMPLETED，实际退出0、本地核验通过**。冻结`939bba4`，北京时间21:42:46→22:08:12；程序1526.315s／launcher1528.615s，RSS3151588KiB，7200成员、2400场两阶段完整，mismatch0。0模型/优化/权重/test；TCM eligible155134、2424步恰一遍，末批62 | [真实S2L-106](S2L-20261027-106.md)、[Record/Squeeze](../docs/reports/2026-10-27-unit-population-review.md)、[启动与终态原件](results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/)。本地经SSH读取真实终态后补写，非服务器AI提交；勿重跑 |
| 105原CSV固定链 | **COMPLETED（执行验收通过，零训练/前向/NPZ/权重）**。冻结`origin/main`＝`d185191`、CPU4、13:06:51Z→13:06:57Z退出0（launcher 13.7 s／程序5.653 s）。九个成员全部READ（3035 量测3,803,180B／参考896,377B／标签8,149B；2819 661,703／46,204／1,755；1043 见run），无成员触发64MiB上限。**六节点全部`full_canonical_exact=true`、`mismatch_count=0`**：重复0、同刻冲突0、时间逆序0、区间外行0；固定事件匹配3035 11/4、2819 11/9、1043 15/15。参考行全部与103一致（3035 191行／2819 89／1043 93），端点13785-14272、789-800、354-367、q55 370-374全match。**整参考成员`global_reference_issue_count`＝64/1/0（整成员计数，未逐条列类型）**。结论仅限实现链一致，不证明物理真值/AIS语义。见[回执](S2L-20261027-105.md) |
| 104固定三案例原链 | COMPLETED；15个固定query、980项检查0不一致；本地核验完成。缺服务器绘图，本地用脱敏回执补图。回执中“排除插值或时间问题”的推断不能由一致重放证明 | [原回执](S2L-20261027-104.md)、[Record/Squeeze与订正](../docs/reports/2026-10-27-three-case-chain-review.md)；不重复领取 |
| 102完整论文指标 | COMPLETED，本地核验300场主表、人口及056计数通过 | [L2S-102](L2S-20261026-102.md)、[S2L-102](S2L-20261026-102.md)；0训练/前向/权重/ZIP，不重复领取 |
| 103三场诊断 | COMPLETED，本地核验全部声明query、误差及尾部份额通过；主因未唯一识别 | [L2S-103](L2S-20261027-103.md)、[S2L-103](S2L-20261027-103.md)；0训练/前向/权重，原输入/参考服务器重放一致，不重复领取 |
| 099 DeepAF | HOLD | [暂停说明](L2S-20261025-099.md)；不得使用旧ZIP或旧合同启动 |

101原`request.json/status.json`已补齐，无需再补或重跑。服务器额外tail分析按事后探索记录，不是新模型任务；保留源数据及派生原件，不能直接沿用其中错误轴序、方向或排名结论。

103的私有原身份/坐标/reference端点与原NPZ继续保留在原run；105只回传脱敏汇总和相对差，完整原行/身份/坐标保留服务器。本地预测对应配准条件分析已经结束，不另交服务器重复计算。旧L2S正文的READY、探索Idea或预算草案不提供重复执行或训练许可。

## 执行边界

- 其余任务均已结案或撤回；当前领取名单只看本页。[历史摘要](../docs/history/README.md)保存结论和Git位置。
- 按[服务器规则](../docs/server_autonomy.md)单次托管、记录真实退出；活动checkout/config/runs冻结、失败保全、不自动重试。最多四单卡，现场FIFO，兼容设备可共卡，不触碰他人进程。
- T_test封存，既有056和各任务原配置保持；不设SHA/hash准入。push不代表服务器已启动。

研究目标和下一步见[研究状态](../docs/research_status.md)，保留原件见[结果索引](results/README.md)。
