# 当前交接

2026-10-22，统一协作分支`main`。状态来自Git回执，不代表实时服务器监控。**065已获批READY：用户明确批准新ReLU验收契约及一次CPU4执行。实施提交`606edd5`、356项测试通过；当前尚无启动回执。**

| 当前任务 | 状态 | 下一执行方 |
| --- | --- | --- |
| [065：按ReLU推理等价检查完成原七臂评价](L2S-20261022-065.md) | **FAILED：退出码 1，248.146 秒，完成 2/300 场后在 scene 5、direct_only 被 `utility_relu_close=False` 拒（relu 越界 2 个、均为正值，激活翻转 0），无七臂成绩；现场已保全** | 本地定下一步口径；[065回执](S2L-20261022-065.md)与[产物](results/LEVEL1-FROZEN-FUSION-20261022-065/)，不重领 |
| [064：保留拒绝现场的完整评价](L2S-20261022-064.md) | **FAILED：首场direct_only utility拒绝，0/300，无七臂成绩；现场保全成功** | [服务器回执](S2L-20261022-064.md)已审，不重领 |
| [063：首场helper_quality诊断](L2S-20261022-063.md) | **COMPLETED：仅helper_quality两遍通过，未检查direct_only** | [回执](S2L-20261022-063.md)，不重领 |
| [062：冻结融合迁移](L2S-20261022-062.md) | **FAILED：一次CPU额度耗尽，无完整成绩** | [回执](S2L-20261022-062.md)，不重领 |

## 064结论与本地决定

详细数值、证据边界和新契约只维护在[研究状态](../docs/research_status.md)。064原utility仅1元素超差，两边负值且对应helper权重为0，其它门槛通过。原始snapshot留服务器，不把摘要当完整输入。

更正此前本地与服务器共同采用的错误推断：062没有记录failed_arm；064明确为direct_only，聚合报错相同仅强烈提示同类问题。063和旧探针只测helper_quality，不能证明原故障随机消失。服务器末层抵消探针混合三个属性求和，27.365／25.3倍及据此推导的8倍ULP不作依据。原S2L和输出保持原样，供核对。

**服务器回传（065 终态）**：北京 20:21:30 提交、20:25:38 结束，unit `mtad-f152891bcdec`，checkout `606edd51dfd4…`，块 SHA256 `e42566cf…` 前后一致、替换 0 处，`--utility-replay-domain relu` 已生效并记入 `run.json`；4 小时提醒未触发，零 GPU，`test_opened=false`。`completed_scenes=2`（scene 1、4 落盘）、`last_scene_id="4"`、`failed_scene_id="5"`、`failed_arm="direct_only"`，`summary.json` 未生成 → 无七臂成绩、无 `utility_replay_audit` 总表。**新契约在 scene 1 生效**：direct_only 的 raw 域仍记 1 个超差（worst ratio 1.2066 于 `[46,8,1]`，即 062／064 同一元素），relu 域 `exceed=0` 通过并完成落盘；scene 4 三臂两域全通过（direct_only raw worst ratio 0.9869，接近 1）。scene 5 的八项检查七项为真（含 `utility_activation_equal`、`activation_flip_count=0`），只 `utility_relu_close=False`：raw `exceed=5` → relu `exceed=2`，两个越界元素都是**正值小幅值**——`[102,62,1]` `0.00516149` vs `0.00516272`、差 `1.2321e-06`、界 `1.0516e-06`、ratio `1.1717`（该元素 helper 权重两侧 `5.34e-06`，权重差 `1.27e-09`）；`[766,11,1]` `0.12807909` vs `0.12807676`、差 `2.3246e-06`、界 `2.2808e-06`、ratio `1.0192`（权重 `0.11354` 两侧，差 `1.83e-06`）。snapshot 派生：157,440 元素、两侧同为正 68,850、符号翻转 0、正值域 `max|diff|=3.05e-05`（在 `|u|≈68.3`，ratio 0.0446）、中位 `1.19e-07`；**本场若通过所需最小 `atol` 为 `1.1805125232910996e-06`**（高于现行地板 18%）。其余全在原门槛内：`helper/anchor_weights` 最大差 `1.8267e-06`、lat/lon 差恰 `0.0`、speed `8.02e-06`、heading `3.66e-05`、`mean_supported` 与三类整数数组逐位相同。保全未新增前向（`extra_partition_forwards=0`、`model_updates=0`、`comparison_error=null`）。V reference 确有读取（300／600，训练为空）。服务器未重试、未改门槛、未加探针、未跳场；056／057 只读，062／063／064 现场未动；**不再提出新的容差数字或量级依据**，口径由本地定。回传 14 件 152 KB（含两场原始逐场行与 raw／relu 越界明细）；`snapshot.npz` 3,379,479 字节、`preparation.json` 85,153,501 字节、4 个 NPZ 留服务器只登记。详见[065回执](S2L-20261022-065.md)。

**不合并**`server/064-utility-decision-gate`的全数组ULP容差候选。065显式选择新的ReLU推理等价检查；默认raw保留旧行为。正激活集合精确一致、ReLU幅度保留旧数值容差、权重和物理限值不变，原始utility差异照常记录。有限负→负差异不再阻断，不宣称raw head重现或旧失败已通过。064仍FAILED；模型／特征／分组／人口／指标不变。

## 执行与回传

服务器只领取本索引当前**获明确批准的LOCAL_TO_SERVER / READY**；DRAFT不可运行，历史批准不恢复额度。全局最多四个获批单卡任务，CPU预算按各交接；具体托管和失败规则见[服务器规则](../docs/server_autonomy.md)。一次尝试、失败停止、不自动重试，T_test封存。

本地AI负责研究、代码和决策，服务器AI负责资源确认、执行及S2L事实回传；push不会唤醒服务器，本地不SSH值守或跨聊天发指令。活动冻结checkout、数据、权重和runs保持不变。日常代码与S2L统一提交main，短期候选分支不自动成为执行来源。

## 仓库与历史

本地唯一工作目录`/Users/a1/Projects/association-code`。旧研究分支已归并退役；后续在主协作仓库保存未提交修改后同步main，不在活动冻结运行目录pull，不丢弃未推送回执。

结案材料见[历史摘要](../docs/history/README.md)，紧凑证据见[结果目录](results/README.md)。清理前完整版本为`5776cd3`，Git可恢复原件；服务器原始完整缓存和checkpoint不删、不重命名。紧凑摘要不能替代运行输入。交接索引只保留当前状态，不继续复制大段原始回执。
