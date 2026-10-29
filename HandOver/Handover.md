# 当前交接

**新授权任务116已COMPLETED并核验：** 冻结80ed87e、服务mtad-ca3fb9ef7880.service退出0，原V32八臂/0训练到固定终点；独立保存数组核验通过。113/115旧批已结束，不重跑；当前无READY或自动后继。

更新：2026-10-29。**108/109/110/112均COMPLETED且本地核验；114/115静态分组与位置小验证均已完成；113修复后固定评价已完成并核验。107原失败保留，不重训；本批无待执行后继，mtad自动跟进已删除。** 106/105/104/103/102已完成并核验，不重复领取。104实际范围与服务器回执过度归因的订正见[本地记录](../docs/reports/2026-10-27-three-case-chain-review.md)，105核验与深挖见[本地记录](../docs/reports/2026-10-27-raw-case-chain-review.md)。A授权不包含新增seed/步数/TAM或新方法训练。用户已另行明确允许推倒重来和重新选择研究思路，见[采纳流程](../docs/research_workflow.md)；这不改变107的冻结执行合同，新执行仍须在本表独立列为READY。

| 任务 | 状态 | 执行／回执入口 |
|---|---|---|
| 116固定R×query贡献控制 | **COMPLETED并核验，退出0，32场/9组件/八臂/14比较。** 程序79.940s／launcher99.683s，0训练；WS_NEAR MAE1276.328→927.435m，32场均好，对Anchor仍有MAE/尾部取舍；WS_SB无MAE收益，人口/覆盖/重复exact不变。 | [S2L-116](S2L-20261029-116.md)、[Record/Squeeze](../docs/reports/2026-10-29-query-contribution-review.md)、[L2S](L2S-20261029-116.md)。冻结80ed87e/原服务已退出，独立数组审计通过；原private留服务器，无自动扩量/重试/训练 |
| 115新组Mean位置控制 | **COMPLETED并核验，退出0，32场55.886s／launcher73.715s。** A0 exact重放全部通过，WS共同69837点MAE1219.73→1268.20m（差）、RMSE2240.36→2159.93m（好）；点覆盖97.7270→98.4417%，重复组92→217。0模型/训练，不能称整体净胜。 | [Record/Squeeze](../docs/reports/2026-10-29-static-position-review.md)、[L2S-115](L2S-20261029-115.md)。冻结cbfe97c，服务mtad-b7b3f402a312，原run115与审计/订正保留；无自动扩量/训练 |
| 114静态分组pilot | **COMPLETED，退出0，最终run118.060s／launcher122.513s，300场。** 仅WS过原V32 gate，原32不重跑、六臂均扩完268；WS decoded宏F1 76.5657→85.1820%，MSTSA mPCA77.9161→82.7318%、mPFA20.2849→14.1685%，23场/4组件F1仍差，raw不变。 | [Record/Squeeze](../docs/reports/2026-10-29-static-partition-review.md)、[L2S-114](L2S-20261029-114.md)。冻结13ecf2f，服务mtad-92e24b30be2f；原run114保留，0训练，不重跑 |
| 109固定分数决定诊断 | **COMPLETED，退出0，300场276.326s，RSS547921920B。** raw/A0完整绑定；234/300分区变化，但8随机序all宏F1全部更差/FP增加，不据敏感度认证新方法。 | [Record/Squeeze](../docs/reports/2026-10-29-frozen-decision-review.md)、[原L2S](L2S-20261029-109.md)。冻结af475e0，服务mtad-ee793415d2ba，原run109保留，不重跑 |
| 110身份容量诊断 | **COMPLETED，退出0，300场4.685s。** 纯硬单射全部跨源真边条件上界55.5212%，同源真边8202、无共同Q64正边11063；不是当前anchor或MH成绩。 | [Record/Squeeze](../docs/reports/2026-10-29-fragment-capacity-review.md)、[原L2S](L2S-20261029-110.md)。冻结af475e0，服务mtad-ff360346dae5，原run110保留，不重跑 |
| 112生成支撑报告 | **COMPLETED，退出0，8.201s，RSS540606464B。** V16/2379片段，G48.1078%、其中参考可评64.0818%；G model−hold特征MSE+.00886522，14场差/2场有益。非TAM或物理效果。 | [Record/Squeeze](../docs/reports/2026-10-29-generated-support-review.md)、[原L2S](L2S-20261029-112.md)。冻结1e55a7f，服务mtad-d8f0704d03f8，原run112保留，不重跑 |
| 113固定107末步评价 | **COMPLETED并核验，真实退出0，全300场；程序25912.028s／launcher25925.460s。** 原输入/单位/分区/ref/人口绑定通过；共同422317位置点TFM MAE42787.223m，对cap128 Mean1127.413m，位置全部300场/38组件更差；保留速度收益。0新增训练，不晋级强恢复baseline。 | [S2L-113](S2L-20261029-113.md)、[Record/Squeeze](../docs/reports/2026-10-29-fixed-endpoint-review.md)、[原L2S](L2S-20261029-113.md)。冻结af475e0、服务mtad-3d23e5ca916c已退出，原run/launcher及300NPZ保留；不重跑/加训，mtad跟进已删除 |
| 108表示依赖诊断 | **COMPLETED，真实退出0，32场15.340s，launcher15.816s。** 原Idea1.1/原T32/0前向训练执行，白名单回执与Record/Squeeze本地核验通过；不据软件/算子结果认证作者错误或EQ-Reg有效。 | [Record/Squeeze](../docs/reports/2026-10-29-pair-representation-review.md)、[L2S-108](L2S-20261028-108.md)。冻结1e55a7f，服务mtad-f6d30740bd5e；原run108与启动前路径定位纠正均保留，不重跑 |
| 107新单位基线A | **FAILED，原退出1；训练两段均COMPLETED（2424/1000），fixed056评价前预检KeyError。** 北京时间10月29日02:56:45结束，总102762.430s，非超时；0自动重试。090C旧fragment NPZ缺sources/membership，不是已观察到输入数值不一致。原code/run/checkpoint保留，不能重启原107或重训。 | [现场终态Record](../docs/reports/2026-10-29-107-failure-discovery.md)、[原L2S-107](L2S-20261027-107.md)。独立113修复评价已完成，见其Record/Squeeze；原107仍FAILED，不覆盖或重启 |
| 106新单位人口准备 | **COMPLETED，实际退出0、本地核验通过**。冻结`939bba4`，北京时间21:42:46→22:08:12；程序1526.315s／launcher1528.615s，RSS3151588KiB，7200成员、2400场两阶段完整，mismatch0。0模型/优化/权重/test；TCM eligible155134、2424步恰一遍，末批62 | [真实S2L-106](S2L-20261027-106.md)、[Record/Squeeze](../docs/reports/2026-10-27-unit-population-review.md)、[启动与终态原件](results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/)。本地经SSH读取真实终态后补写，非服务器AI提交；勿重跑 |
| 105原CSV固定链 | **COMPLETED（执行验收通过，零训练/前向/NPZ/权重）**。冻结`origin/main`＝`d185191`、CPU4、13:06:51Z→13:06:57Z退出0（launcher 13.7 s／程序5.653 s）。九个成员全部READ（3035 量测3,803,180B／参考896,377B／标签8,149B；2819 661,703／46,204／1,755；1043 见run），无成员触发64MiB上限。**六节点全部`full_canonical_exact=true`、`mismatch_count=0`**：重复0、同刻冲突0、时间逆序0、区间外行0；固定事件匹配3035 11/4、2819 11/9、1043 15/15。参考行全部与103一致（3035 191行／2819 89／1043 93），端点13785-14272、789-800、354-367、q55 370-374全match。**整参考成员`global_reference_issue_count`＝64/1/0（整成员计数，未逐条列类型）**。结论仅限实现链一致，不证明物理真值/AIS语义。见[回执](S2L-20261027-105.md) |
| 104固定三案例原链 | COMPLETED；15个固定query、980项检查0不一致；本地核验完成。缺服务器绘图，本地用脱敏回执补图。回执中“排除插值或时间问题”的推断不能由一致重放证明 | [原回执](S2L-20261027-104.md)、[Record/Squeeze与订正](../docs/reports/2026-10-27-three-case-chain-review.md)；不重复领取 |
| 102完整论文指标 | COMPLETED，本地核验300场主表、人口及056计数通过 | [L2S-102](L2S-20261026-102.md)、[S2L-102](S2L-20261026-102.md)；0训练/前向/权重/ZIP，不重复领取 |
| 103三场诊断 | COMPLETED，本地核验全部声明query、误差及尾部份额通过；主因未唯一识别 | [L2S-103](L2S-20261027-103.md)、[S2L-103](S2L-20261027-103.md)；0训练/前向/权重，原输入/参考服务器重放一致，不重复领取 |
| 099 DeepAF | HOLD | [暂停说明](L2S-20261025-099.md)；不得使用旧ZIP或旧合同启动 |

101原`request.json/status.json`已补齐，无需再补或重跑。服务器额外tail分析按事后探索记录，不是新模型任务；保留源数据及派生原件，不能直接沿用其中错误轴序、方向或排名结论。

103的私有原身份/坐标/reference端点与原NPZ继续保留在原run；105只回传脱敏汇总和相对差，完整原行/身份/坐标保留服务器。本地预测对应配准条件分析已经结束，不另交服务器重复计算。旧L2S正文的READY、探索Idea或预算草案不提供重复执行或训练许可。

## 执行边界

- 本页任务均已结案、HOLD或撤回；当前无READY后继，不重复领取旧正文命令。[历史摘要](../docs/history/README.md)保存结论和Git位置。
- 按[服务器规则](../docs/server_autonomy.md)单次托管、记录真实退出；活动checkout/config/runs冻结、失败保全、不自动重试。最多四单卡，现场FIFO，兼容设备可共卡，不触碰他人进程。
- T_test封存，既有056和各任务原配置保持；不设SHA/hash准入。push不代表服务器已启动。

研究目标和下一步见[研究状态](../docs/research_status.md)，保留原件见[结果索引](results/README.md)。
