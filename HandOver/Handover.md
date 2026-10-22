# 当前交接

2026-10-23，统一协作分支`main`。执行状态以Git回执为准，不代表实时监控。**068已实现、404项测试通过，按用户“本地写完代码让服务器执行”指令发布READY。一次单GPU／CPU4：两场输入预核→位置门控1000更新→CPU初始化与终点完整V300。服务器已于北京01:23:41领取一次（RUNNING），预检已PASS。067结案；066保持FAILED／REJECTED，不再调整旧容差。**

| 当前任务 | 状态 | 下一执行方 |
| --- | --- | --- |
| [068：受限位置门控pilot](L2S-20261023-068.md) | **RUNNING：服务器已领取并提交，北京01:23:41，单元`mtad-e12d2d25ee1b`，checkout `11a70a5`，绑卡 `GPU-6811d9c0…`（物理1，共卡）** | 服务器执行和回传；[068回执](S2L-20261023-068.md)，不重复领取 |
| [067：只读缓存数值影响／融合机会归因](L2S-20261022-067.md) | **COMPLETED／退出码 0：300/300 场、900 项重放复核、47 项拒绝逐项复现；562 秒，零前向／零压缩卷读取；不改写 066 的 REJECTED** | 本地审查完成，下一步明确位置门控候选并核对极端输入；[067回执](S2L-20261022-067.md)与[产物](results/LEVEL1-FUSION-CACHE-20261022-067/)，不重领 |
| [066：输出层验收＋全量数值审计](L2S-20261022-066.md) | **FAILED／REJECTED：300/300 场跑完、900 项审计收齐，42 分钟，退出码 1；47 项拒绝＝46×direct_only `helper_weights_close`＋1×helper_quality `physical_close`(scene 2535)，无 `summary.json`** | 本地按全量清单定位（已保存数组可无模型复审计）；[066回执](S2L-20261022-066.md)与[产物](results/LEVEL1-FROZEN-FUSION-20261022-066/)，不重领 |
| [065：ReLU重放](L2S-20261022-065.md) | **FAILED：完成2/300，scene5/direct_only两个正utility微超差，权重及物理检查通过** | [回执](S2L-20261022-065.md)已审，额度耗尽，不重领 |
| [064：保全拒绝现场](L2S-20261022-064.md) | **FAILED：首场direct_only负utility拒绝，0/300** | [回执](S2L-20261022-064.md)，不重领 |
| [063：首场helper_quality](L2S-20261022-063.md) | **COMPLETED：仅helper_quality两遍通过，未测direct_only** | [回执](S2L-20261022-063.md)，不重领 |
| [062：原冻结迁移](L2S-20261022-062.md) | **FAILED：无完整成绩，旧失败臂推断已撤回** | [回执](S2L-20261022-062.md)，不重领 |

## 本地决定

证据、勘误和判断只维护在[研究状态](../docs/research_status.md)，不在本索引复制原始回执。062／064／065／066原失败状态保留；063单臂诊断通过的边界不变。不合并服务器8倍ULP候选，不继续按本场所需数值调容差。

066未训练，人口及缓存结构核验通过；拒绝来自46场direct_only速度helper权重和scene2535/helper_quality速度输出。七臂聚合为**未验收探索性数据**，不能当正式成绩；普通控制复现及support_switch解析核验通过。067已完成两项归因：原OLS组数值差的聚合影响极小，位置收益缺口同时包含漏救与保护。11项有效超限记录不等于全库只有11点变化；机会区外4845点HQ也发生变化。详细复核及候选见研究状态。

**服务器回传（068 提交与预检）**：块 SHA256 `ff91ad14…` 执行前后一致、**文本改动 0 处**（GPU 由块内声明的 `task_gpu_uuid` 注入）；选卡用 `tools/setup/gpu_allocation.py --device auto --min-free-mib 12000` → 物理1 `GPU-6811d9c0…`（free 23,261 MiB、util 13%，同卡他人 `hejingjing` 进程 980 MiB，共卡未终止），证据 `runs/…068-attempt-1-gpu.json`。worker 3535387／payload 3535390，`--max-seconds 14400 --time-policy advisory`、`--device cuda:0`＋`--cpu-threads 4`。领取前 32 核／load 17.33／available 93 GB／`/home` 可用 801 GB，我的获批单卡作业数 0（未超四个）。056 源完整（`association/` 13 文件含 `final.pt`，`system/` 两类 NPZ 各 300）。**阶段1预检 PASS**（13.180 秒，两场 3035／2535）：`data_contract_reproduced=true`、`training_gate_passed=true`、`anomalies_removed=false`、`semantic_physical_plausibility_established=false`、`checkpoint_loaded=false`、`model_forwards=0`、`test_opened=false`，压缩卷仅读这两场的量测／关联表／真实场景。**单列合法极端原值**：3035 anchor 原生速度上界 22,663.18 m/s（OLS 速度 −194.6 m/s，纬度跨 12.919–20.163°）；2535 anchor／helper 达 83,615.31／77,315.28 m/s（经度跨 123.40–140.92°）——原样保留、未删未改时钟，不据此断定根因。当前 `phase=prepare_T2100_V300`。详见[068回执](S2L-20261023-068.md)。

研究上保留简单支撑规则为位置强控制，保留Direct-only速度收益线索；Mixed-action暂不晋升。完整数值、分层、组件异质性及限制见[研究状态](../docs/research_status.md)，执行事实只见[066回执](S2L-20261022-066.md)。

067原始执行与统计事实保留在[服务器回执](S2L-20261022-067.md)及[结果目录](results/LEVEL1-FUSION-CACHE-20261022-067/)；本索引不重复复制长报告。068冻结为一个组级位置头，g初值.95，速度／航向Mean，T-only直接位置监督；step0与final同CPU完整评价。softmax仍只是机制备选，不启动第二项训练。具体预算和停止条件只见068交接，不续用067额度。

## 执行与回传

服务器只领取本索引当前**获明确批准的LOCAL_TO_SERVER / READY**；DRAFT不可运行，历史批准不恢复额度。全局最多四个获批单卡任务，CPU预算按各交接；具体托管和失败规则见[服务器规则](../docs/server_autonomy.md)。一次尝试、失败停止、不自动重试，T_test封存。

本地AI负责研究、代码和决策，服务器AI负责资源确认、执行及S2L事实回传；push不会唤醒服务器，本地不SSH值守或跨聊天发指令。活动冻结checkout、数据、权重和runs保持不变。日常代码与S2L统一提交main，短期候选分支不自动成为执行来源。

## 仓库与历史

本地唯一工作目录`/Users/a1/Projects/association-code`。旧研究分支已归并退役；后续在主协作仓库保存未提交修改后同步main，不在活动冻结运行目录pull，不丢弃未推送回执。

结案材料见[历史摘要](../docs/history/README.md)，紧凑证据见[结果目录](results/README.md)。清理前完整版本为`5776cd3`，Git可恢复原件；服务器原始完整缓存和checkpoint不删、不重命名。紧凑摘要不能替代运行输入。交接索引只保留当前状态，不继续复制大段原始回执。
