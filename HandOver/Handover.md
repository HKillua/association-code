# 当前交接

2026-10-22，统一协作分支`main`。状态来自Git回执，不代表实时监控。**066已获批READY：用户批准完整审计流程及一次CPU4执行。实施提交`e6c137e`，362项测试与独立审查通过；服务器已于北京时间22:27:12提交一次，23:09:14 收终态：300/300 场、900 项审计、`acceptance=REJECTED`。**

| 当前任务 | 状态 | 下一执行方 |
| --- | --- | --- |
| [066：输出层验收＋全量数值审计](L2S-20261022-066.md) | **FAILED／REJECTED：300/300 场跑完、900 项审计收齐，42 分钟，退出码 1；47 项拒绝＝46×direct_only `helper_weights_close`＋1×helper_quality `physical_close`(scene 2535)，无 `summary.json`** | 本地按全量清单定位（已保存数组可无模型复审计）；[066回执](S2L-20261022-066.md)与[产物](results/LEVEL1-FROZEN-FUSION-20261022-066/)，不重领 |
| [065：ReLU重放](L2S-20261022-065.md) | **FAILED：完成2/300，scene5/direct_only两个正utility微超差，权重及物理检查通过** | [回执](S2L-20261022-065.md)已审，额度耗尽，不重领 |
| [064：保全拒绝现场](L2S-20261022-064.md) | **FAILED：首场direct_only负utility拒绝，0/300** | [回执](S2L-20261022-064.md)，不重领 |
| [063：首场helper_quality](L2S-20261022-063.md) | **COMPLETED：仅helper_quality两遍通过，未测direct_only** | [回执](S2L-20261022-063.md)，不重领 |
| [062：原冻结迁移](L2S-20261022-062.md) | **FAILED：无完整成绩，旧失败臂推断已撤回** | [回执](S2L-20261022-062.md)，不重领 |

## 本地决定

证据、勘误和判断只维护在[研究状态](../docs/research_status.md)，不在本索引复制原始回执。065旧契约失败事实保留；已完成2场不能作为V300成绩。不合并服务器8倍ULP候选，不继续按本场所需数值调容差。

066显式采用output验收域：内部raw/ReLU／符号变化全部记录，实际权重、state-span及物理原容差不变。有限且结构正确的数值超差在一次作业中全量收集；来源／shape／非有限／人口或缓存损坏仍立即停止。全部验收通过才写正式summary；否则FAILED并保留明确未验收汇总。每场保存完整OLS重放数组，后续复核无需重新准备数据或前向。任务不是保证放行，而是结束逐个微差重跑。

**服务器回传（066 终态）**：北京 22:27:12 提交、23:09:14 结束，`FAILED`／退出码 1／`elapsed_seconds=2522.4209690531716`（约 42 分钟，约 7.5 秒／场），`timed_out=false`、4 小时提醒未触发；**`completed_scenes=300/300`、`analysis_complete=true`、`acceptance=REJECTED`、`utility_replay_domain="output"`**，stderr 末行即工具自述 `output replay rejected: 47/900 sources; full audit saved, unvalidated-summary.json is not an accepted system result`；**无 `summary.json`、无 `replay-failure/`**（output 域不逐场中断，非保全失败）。人口与核验全数达成：`scene_count=300`、`fragment_count=24599`、`component_count=38`、`ordinary_source_replay_exact=true`、`support_control_analytical_verified=true`（与 061 解析预期 `442263`／`MAE 729.9808435828807`／`RMSE 3332.536401612317` 在 `rtol1e-10/atol1e-9` 下一致）、`threshold=1.7320103112387182` 原值、`test_opened=false`、`optimizer_updates=0`、`gpu_used=false`、容差原值未动。**47 项拒绝的构成**：`helper_weights_close` **46 项全在 direct_only**，越界元素合计 65 个、最坏容差比 min/中位/max `1.0422/1.3221/3.1805`、**65 个越界元素全部落在 speed 分量（索引 1）**；`physical_close` **1 项 = scene 2535 的 helper_quality**，逐属性最大绝对差 lat `9.792e-09`／lon `1.593e-08` 度、**speed `1.690933e-03`（上限 0.001，超 69%）**、heading `1.820e-05`；`membership_exact`／`ordinary_control_exact`／`anchor_weights_close`／`state_span_close` 在 900 项中零失败。utility 诊断照实保留不作中止：direct_only raw 140 场／929 元素、relu 107 场／475 元素、翻转 0；helper_quality raw 1 场／2、relu 1 场／1、**翻转 2**；mixed_action 全 0。幅度全部由 066 新存的 `source057_*` 数组对只读 057 缓存直接复算，**未重跑任何前向**（明细见 `rejection-margins.json` 与 47 场 `rejected-scene-rows.json`）。最坏项 scene 4771：prev `0.007905518`／cur `0.007902086`、差 `3.431970e-06`、界 `1.079055e-06`、比 `3.1805`。服务器**不建议把 atol 调到刚好通过、不提任何新容差数字**，也不引用 `unvalidated-summary.json` 里任何方法指标当成绩（七臂聚合已算出但未验收，253 个未拒场不冒充全量）。V reference 确有读取（300／600，训练为空）。未重试、未跳场、未改门槛、未加探针；056／057 只读，062／063／064／065 现场未动，只写全新 066 目录。回传 15 件 1.1 MB；`preparation.json` 85 MB、`scene-summary.json.gz` 9.87 MB、`scenes/` 300 个 11 MB、`predictions/` 300 个 2.4 GB、`system-predictions/` 300 个 316 MB 留服务器登记（run 目录 2.9 GB，磁盘余 801 GB）。详见[066回执](S2L-20261022-066.md)。

## 执行与回传

服务器只领取本索引当前**获明确批准的LOCAL_TO_SERVER / READY**；DRAFT不可运行，历史批准不恢复额度。全局最多四个获批单卡任务，CPU预算按各交接；具体托管和失败规则见[服务器规则](../docs/server_autonomy.md)。一次尝试、失败停止、不自动重试，T_test封存。

本地AI负责研究、代码和决策，服务器AI负责资源确认、执行及S2L事实回传；push不会唤醒服务器，本地不SSH值守或跨聊天发指令。活动冻结checkout、数据、权重和runs保持不变。日常代码与S2L统一提交main，短期候选分支不自动成为执行来源。

## 仓库与历史

本地唯一工作目录`/Users/a1/Projects/association-code`。旧研究分支已归并退役；后续在主协作仓库保存未提交修改后同步main，不在活动冻结运行目录pull，不丢弃未推送回执。

结案材料见[历史摘要](../docs/history/README.md)，紧凑证据见[结果目录](results/README.md)。清理前完整版本为`5776cd3`，Git可恢复原件；服务器原始完整缓存和checkpoint不删、不重命名。紧凑摘要不能替代运行输入。交接索引只保留当前状态，不继续复制大段原始回执。
