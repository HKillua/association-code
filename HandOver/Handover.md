# 当前交接

更新：2026-10-29。**109/110/112/113为本轮一次验证READY，108仍READY；107已真实失败结案：TCM2424/TFM1000完成，固定056预检遇旧缓存字段不兼容，原运行不重试。** 106/105/104/103/102已完成并核验，不重复领取。104实际范围与服务器回执过度归因的订正见[本地记录](../docs/reports/2026-10-27-three-case-chain-review.md)，105核验与深挖见[本地记录](../docs/reports/2026-10-27-raw-case-chain-review.md)。A授权不包含新增seed/步数/TAM或新方法训练。用户已另行明确允许推倒重来和重新选择研究思路，见[采纳流程](../docs/research_workflow.md)；这不改变107的冻结执行合同，新执行仍须在本表独立列为READY。

| 任务 | 状态 | 执行／回执入口 |
|---|---|---|
| 109固定分数决定诊断 | **READY，一次CPU4/4GiB预算/AS8GiB，0前向/训练。** 原056全V300原序＋8固定排列，原raw/A0预检、decoded/组件全分母；软件与独审通过。 | [L2S-109](L2S-20261029-109.md)、[Idea](../docs/proposals/frozen_score_output_decision_diagnostic_20261029.md)。独立run109/attempt1，无自动训练后继 |
| 110身份容量诊断 | **READY，一次CPU4/4GiB预算/AS8GiB，0前向/训练。** 同056 V300缓存，完整标签等价、fragment a/b/ab/min与无共同Q64全分母；只测条件可达上界，不归罪当前非单射decoder。 | [L2S-110](L2S-20261029-110.md)、[Idea](../docs/proposals/fragment_identity_capacity_20261029.md)。独立run110/attempt1 |
| 112生成支撑报告 | **READY，一次CPU2/4GiB预算/AS8GiB，0新前向/训练。** 固定V16，只读107原TCM评估，O/G支撑和model/hold特征误差完整分母；非TAM/物理效果。 | [L2S-112](L2S-20261029-112.md)、[Idea1.1](../docs/proposals/generated_support_score_diagnostic_20261029.md)。独立run112/attempt1；评分子问题仍deferred |
| 113固定107末步评价 | **READY，一次单GPU/CPU4/RAM预算16GiB，0新增训练。** 已训练TCM2424/TFM1000不重训，修复旧NPZ绑定后重做原全V固定组评价；19软件回归/独审通过。 | [L2S-113](L2S-20261029-113.md)、[Idea](../docs/proposals/tmtaff_fixed_endpoint_evaluation_repair_20261029.md)。新run113/attempt1；原107失败保全 |
| 108表示依赖诊断 | **READY，一次CPU≤4/RAM预算4GiB/AS≤8GiB，0GPU/训练/模型前向。** 固定T_core按seed20261028抽32场，仅64个label/measurement成员；每source≤8anchor/16点窗、全部异源候选及unknown记账，0reference/V/test。对比literal/stat/pointwise/rawB四接口，不冒充作者实现或创新效果；plan1.1软件核验/独立审查通过，**尚未核到领取或真实启动**。10min仅advisory，一次失败保全不重试；无训练自动后继。 | [L2S-108](L2S-20261028-108.md)、[事前Idea](../docs/proposals/pair_representation_dependency_20261028.md)。新冻结worktree `association-code-run108`，run `LEVEL1-PAIR-REPRESENTATION-DEPENDENCY-20261028-108-attempt-1`；发布不等已启动 |
| 107新单位基线A | **FAILED，原退出1；训练两段均COMPLETED（2424/1000），fixed056评价前预检KeyError。** 北京时间10月29日02:56:45结束，总102762.430s，非超时；0自动重试。090C旧fragment NPZ缺sources/membership，不是已观察到输入数值不一致。原code/run/checkpoint保留，不能重启原107或重训。 | [现场终态Record](../docs/reports/2026-10-29-107-failure-discovery.md)、[原L2S-107](L2S-20261027-107.md)。后续仅允许另定独立的修复后零训练评价任务；未发布不领取 |
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
