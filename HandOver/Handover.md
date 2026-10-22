# 当前交接

2026-10-23，统一协作分支`main`。执行状态以Git回执为准，不代表实时监控。**067已获用户独立批准并执行一次完毕：COMPLETED／退出码 0，300/300 场、900 项复核、47 项拒绝复现；066 保持 FAILED／REJECTED。066已结束，FAILED／REJECTED保持，不重领或追加完整V前向。**

| 当前任务 | 状态 | 下一执行方 |
| --- | --- | --- |
| [067：只读缓存数值影响／融合机会归因](L2S-20261022-067.md) | **COMPLETED／退出码 0：300/300 场、900 项重放复核、47 项拒绝逐项复现；562 秒，零前向／零压缩卷读取；不改写 066 的 REJECTED** | 本地按证据提出至多一个可证伪改动；[067回执](S2L-20261022-067.md)与[产物](results/LEVEL1-FUSION-CACHE-20261022-067/)，不重领 |
| [066：输出层验收＋全量数值审计](L2S-20261022-066.md) | **FAILED／REJECTED：300/300 场跑完、900 项审计收齐，42 分钟，退出码 1；47 项拒绝＝46×direct_only `helper_weights_close`＋1×helper_quality `physical_close`(scene 2535)，无 `summary.json`** | 本地按全量清单定位（已保存数组可无模型复审计）；[066回执](S2L-20261022-066.md)与[产物](results/LEVEL1-FROZEN-FUSION-20261022-066/)，不重领 |
| [065：ReLU重放](L2S-20261022-065.md) | **FAILED：完成2/300，scene5/direct_only两个正utility微超差，权重及物理检查通过** | [回执](S2L-20261022-065.md)已审，额度耗尽，不重领 |
| [064：保全拒绝现场](L2S-20261022-064.md) | **FAILED：首场direct_only负utility拒绝，0/300** | [回执](S2L-20261022-064.md)，不重领 |
| [063：首场helper_quality](L2S-20261022-063.md) | **COMPLETED：仅helper_quality两遍通过，未测direct_only** | [回执](S2L-20261022-063.md)，不重领 |
| [062：原冻结迁移](L2S-20261022-062.md) | **FAILED：无完整成绩，旧失败臂推断已撤回** | [回执](S2L-20261022-062.md)，不重领 |

## 本地决定

证据、勘误和判断只维护在[研究状态](../docs/research_status.md)，不在本索引复制原始回执。062／064／065／066原失败状态保留；063单臂诊断通过的边界不变。不合并服务器8倍ULP候选，不继续按本场所需数值调容差。

066未训练，人口及缓存结构核验通过；拒绝来自46场direct_only速度helper权重和scene2535/helper_quality速度输出。七臂聚合为**未验收探索性数据**，不能当正式成绩；普通控制复现及support_switch解析核验通过。067已获批只读缓存分析：A处理原OLS数值影响，B处理学习组位置缺口；代码、测试和执行命令见READY交接，尚无服务器启动回执。

研究上保留简单支撑规则为位置强控制，保留Direct-only速度收益线索；Mixed-action暂不晋升。完整数值、分层、组件异质性及限制见[研究状态](../docs/research_status.md)，执行事实只见[066回执](S2L-20261022-066.md)。

**服务器回传（067 终态）**：unit `mtad-9392f199143a`、checkout `a208767374ff…`、块 SHA256 `644ab900…` 前后一致且占位符替换 0 处；北京 00:22:56 提交、00:32:18 结束，`COMPLETED`／退出码 0／`elapsed_seconds=562.078`（自计 556.083），1 小时提醒未触发，**stdout 与 stderr 均 0 字节**。旗标实测：`checkpoint_loaded=false`、`model_forwards=0`、`optimizer_updates=0`、`archive_members_read=0`、`cached_reference_used=true`（不读压缩卷、真值取自已缓存 reference 数组）、`test_opened=false`、`gpu_used=false`、**`source_acceptance=REJECTED`、`method_result_accepted=false`、`reenactment_or_acceptance_change=false`**（066 未被改写）。

**A（原 OLS 组）**：`replay_count=900`、`rejected_replay_count=47` 逐项复现（direct_only 46＋helper_quality 1＋mixed_action 0）；映射为非互斥计数 `non_output_fragment=37`／`unselected_output_group=0`／`selected_invalid_reference_or_prediction=3`／`selected_valid_target_point=11`，落到唯一有效目标 **11 项**（状态改变 11、真值误差改变 11）；聚合影响近乎为零（九条臂／属性 MAE 差 `1e-9…2.8e-7`，direct_only 位置差恰为 `0`，`速度分解超舍入界点=0`）。**scene 2535**：拒绝映射到 `selected_valid_target_point=true`（1 有效状态点／1 误差变化点；21 片段／13 组／9 采用组／4 重复／9 目标），最大速度差 `0.0016909334517549723`（超 0.001 上限 69%），权重分解 `Δv=ΣΔw(v_helper−v_anchor)=-0.00169093`、**残差 `1.5718e-13` ≪ 累计舍入界 `4.342e-09`**——该速度差完全由 helper 权重差携带；同场 mixed_action 差 `6.16083e-04` 未超限；文件明确 `learned_group_transfer_replay_claimed=false`。

**B（056 学习组位置）**：两种分母都给——唯一目标 `442,263`、全部输出组 `472,814`；anchor `885.579/6307.49`、support_switch `729.981/3332.54`、helper_quality `801.242/6268.15`；HQ−规则均值 `+71.261 m`，改善累计 `83,611,694 m` vs 伤害 `115,127,991 m`，更好 57,988／更差 37,626／平局 346,649。互斥类别 A `675`、B `30,614`、C `37,009`、D `8,112`、仅规则平局 `4,845`、仅模型平局 `14,359`、双平局 `346,649`；**全部非平局类都落在 `anchor_outside_helper_supported`（90,769 点）**，`anchor_inside_support`（322,953）与 `predicted_group_no_support`（28,541）只有平局。单一集中点突出：`scene 3035 / target 374302000-1` 贡献 `1.83522e+07 m`（绝对）与 `1.17344e+13 m²`（平方），约为次名的 24 倍／400 倍。机制边际可重叠、不可分摊：零权重且状态不变 `221,208`、存在优于 Anchor 的有支撑 helper `70,252`、有支撑错身份 helper `65,799`、多标记点 `210,989`、`identity_unidentifiable=0`；服务器不按最大计数宣布唯一原因。

**服务器独立复核 24 项全过**（`server-consistency-checks.json`，只读、未重跑前向）：8 个层的互斥类别合计与该层点数逐层相等；067 三种输出位置 MAE 与 066 缓存值之差 <1e-6；A 侧 900／47 复现且未重新放行；8 项边界旗标符合。交付 `status=POSTHOC_DIAGNOSTIC_ONLY`、平局容差 `1e-9 m`、分位数 `linear`、Top 上限 20。服务器未把任何数字当方法成绩，未生成／覆盖 066 summary，未重试、未加前向、未改门槛、未打开 T_test；066／056／057 与 062／063／064／065 现场只读未改名。回传 15 件 1.1 MB（含 `numeric-impact.json.gz` 319,439、`fusion-opportunities.json.gz` 675,614、`report.md` 10,015、`scene-2535-numeric.json.gz` 7,988）；`scenes/` 300 个逐场证据（7,123,378 字节，run 目录 8.4 MB）留服务器登记。详见[067回执](S2L-20261022-067.md)。
## 执行与回传

服务器只领取本索引当前**获明确批准的LOCAL_TO_SERVER / READY**；DRAFT不可运行，历史批准不恢复额度。全局最多四个获批单卡任务，CPU预算按各交接；具体托管和失败规则见[服务器规则](../docs/server_autonomy.md)。一次尝试、失败停止、不自动重试，T_test封存。

本地AI负责研究、代码和决策，服务器AI负责资源确认、执行及S2L事实回传；push不会唤醒服务器，本地不SSH值守或跨聊天发指令。活动冻结checkout、数据、权重和runs保持不变。日常代码与S2L统一提交main，短期候选分支不自动成为执行来源。

## 仓库与历史

本地唯一工作目录`/Users/a1/Projects/association-code`。旧研究分支已归并退役；后续在主协作仓库保存未提交修改后同步main，不在活动冻结运行目录pull，不丢弃未推送回执。

结案材料见[历史摘要](../docs/history/README.md)，紧凑证据见[结果目录](results/README.md)。清理前完整版本为`5776cd3`，Git可恢复原件；服务器原始完整缓存和checkpoint不删、不重命名。紧凑摘要不能替代运行输入。交接索引只保留当前状态，不继续复制大段原始回执。
