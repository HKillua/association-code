# MTAD 分歧后对侧位移的去缓存诊断

## Record

`EXPERIMENT_RECORD` v1.0，`REC-ONLINE-REJOIN-FRESH-DISPLACEMENT-20261113`，completed。C-ON50，消费 `IDEA-ONLINE-REJOIN-FRESH-DISPLACEMENT-20261113` v1.0/v1.1。假设是严格批前的新对侧坐标去除旧缓存复用后，仍可增加对假边的排除证据。原 MTAD 28 场固定候选图、482 窗、416 次合法单报接回。用原生量测时刻作为理想到达，只重开原场量测，先冻结逐窗动作，再连接 C49 原匿名身份评价。零训练、选型、新场、位置参考、外部数据和封存测试。 [冻结设计](../proposals/online_rejoin_fresh_displacement_20261113.md) · [2025–2026 原文分析](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_rejoin_fresh_displacement_sources_20261113.md)

首轮 `fff8cd7` 保存了 28 份未连接标签的读出，随后在合法评价人口连接退出 1，墙时 3.89 秒。错误来自把内部原候选读出与最终合格评价窗视为同一集合。汇总出结果前的独立设计复核还指出，硬删所有分歧后报文会在任何新点有净贡献的窗移除必需括点。修订在查阅新结果前冻结 `81274d4`。R1 沿原匿名人口规则，并保留硬删作覆盖对照，另加固定括点的坐标替换。Apple M2，Python 3.12.14，线程上限 2，R1 真实退出 0，墙时 4.19 秒，峰值 RSS 239,452,160 B。154 项在线测试通过。 [复现命令](../../HandOver/L2S-20261113-online-rejoin-fresh-displacement.md) · [首轮日志](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/rejoin_fresh_displacement_attempt_1.log) · [R1 日志](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/rejoin_fresh_displacement_r1.log)

用报告者上一原生点与之前 20 秒为同一对查询端点，从两对侧严格批前历史按轨键和时刻删旧缓存原点后重新括点。两端权重之差非零的分歧后原点才算实际新贡献。沿用 45 秒最大相邻间隔、60 秒新鲜期，无评分旧缓存的 49 窗按合同单列。固定常量平移在位移中抵消。坐标替换原计划用各受影响对侧不晚于分歧起点的最后两条缓存外报文等速预测，并要求锚间隔及外推时距均不超过 45 秒。位移只直接检验报告者参与的两条候选边。独立复算用 C37 原生时钟、C42 角色键、C45 缓存支撑和冻结位移，核对 28 份文件、1,990 个逐窗位移贡献和 1,248 个新动作，贡献均严格早于接回批。 [匿名逐窗原件](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/rejoin_fresh_displacement_r1/rows.json) · [汇总](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/rejoin_fresh_displacement_r1/summary.json) · [独立核验](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/rejoin_fresh_displacement_r1/verification.json)

## Squeeze

`SQUEEZE_REPORT` v1.0，`SQ-ONLINE-REJOIN-FRESH-DISPLACEMENT-20261113`。焦点 10 窗来自 5 场、5 个共享 A 组件，均为原本正确的接回，其中 2 窗在稀疏场。旧位移实际贡献的分歧后对侧报文共 18 条，17 条在旧缓存。9 窗的原贡献全部已在旧缓存，1 窗有一条新报且无可评分旧缓存。焦点窗中 7 窗原始流两对侧都有分歧后报文，去重后还有 3 窗双侧各有一条缓存外新报，来自同一场。这 3 窗最近报文均新鲜，窗口内最大相邻间隔分别为 50.112、50.112、50.070 秒，超过事前 45 秒上限。另有 6 窗失鲜。最终 0／10 可重算位移。未剔除旧缓存时原位移支持正确接回 6 次、会误拦 4 次，同刻屏蔽支持正确接回 3 次、会误拦 7 次，当前形状 9 窗可评分、误拦 1 次。

| 人口 | 窗数 | 旧位移用过分歧后对侧报文 | 可核缓存外分歧后新报 | 去缓存后位移可算 |
|---|---:|---:|---:|---:|
| 当批推对排序，真边变远 | 131 | 10 | 22 | 0 |
| 真边变远的焦点组 | 10 | 10 | 3 | 0 |
| 当批推对排序，真边变近 | 121 | 12 | 22 | 0 |
| 真边变近且旧位移用过新报 | 12 | 12 | 7 | 0 |
| 原本正确接回 | 308 | 34 | 49 | 0 |
| 原本正确接回且旧位移用过新报 | 34 | 34 | 11 | 0 |
| 全部合法单报接回 | 416 | 45 | 71 | 0 |

416 窗中，171 窗有分歧后原始对侧报文，71 窗有可核缓存外新报，31 窗双侧都有。71 窗全部因间隔超限无法重算，逐窗最大间隔在 50.070–50.228 秒。367 个有可评分旧缓存的窗最终状态为 239 失鲜、127 间隔超限、1 缺括点。两次旧缓存新增命中都有单侧分歧后原生新报，但这些新报未进入 C49 位移，去缓存重算时也都因间隔超限。当前形状的 8 次广义误拦中，旧位移可算 6 次，其中 3 次支持正确接回。8 次误拦在去缓存后均无合法位移，原误拦仍在。

| 当批动作对照 | 主稀疏 命中／误拦（28／142） | 广义纯方向 命中／误拦（67／308） |
|---|---:|---:|
| 原 KF 接受 | 0／0 | 0／0 |
| 同刻屏蔽几何 | 13／36 | 40／89 |
| 当前形状 | 5／1 | 8／8 |
| 未去缓存位移方向 | 7／23 | 16／72 |
| 去缓存位移、可归因动作、仅撤销保护 | 各 5／1 | 各 8／8 |

三种去缓存动作在 416 窗逐项回到当前形状。稀疏广义亦为 7／32、2／148，各共享 A 组件均无净变化。主和广义共同支撑为空，无法实测坐标替换下的分差归因。固定区间的失败首先是可观测支撑不足，未测出有足够新报时的判别质量。窗可共用轨迹，局部位移不等于独立物理位置证据。没有达到事前新留场条件，持续 ID、稀疏长错链、原正确配对及首报尚无新场同延迟发布数据。`next_skill=deep-read Mode C`。 [Mode C v63](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v63-去缓存位移的覆盖回流)
