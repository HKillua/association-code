# 当前交接

2026-10-22，统一分支`main`；原研究分支已完整合入并退役。状态基于已收到的Git回执，不代表实时服务器监控。

| 当前任务 | 状态 | 下一执行方 |
| --- | --- | --- |
| [064：原七臂完整评价，增加拒绝现场保存](L2S-20261022-064.md) | **FAILED：退出码 1，265.027 秒，首场景 `direct_only` 重放被拒（`utility_close=False`，唯一越界元素 1 个），无七臂成绩；保全现场已落盘** | 本地据保全证据决定口径或论证数值契约；[064回执](S2L-20261022-064.md)与[产物](results/LEVEL1-FROZEN-FUSION-20261022-064/)，不重领 |
| [063：首场冻结融合重放诊断](L2S-20261022-063.md) | **COMPLETED：退出码 0，253.785 秒，`source_replay_passed=true`，062 的 utility 拒绝未重现（本轮 max 1.0729e-06、越界 0），无任何七臂成绩** | 本地据证据决定口径或另行论证数值契约；[063回执](S2L-20261022-063.md)与[产物](results/LEVEL1-REPLAY-DIAGNOSTIC-20261022-063/)，不重领 |
| [062：冻结057融合迁移到056学习分组](L2S-20261022-062.md) | **FAILED：一次CPU4已消耗，首场景057源重放 `utility_close=False`，不自动重试、不重领** | 本地核定复核口径或停止；[062回执](S2L-20261022-062.md)与[产物](results/LEVEL1-FROZEN-FUSION-20261022-062/) |

**064已获批READY，当前尚未收到RUNNING回执；062保持FAILED、063已结案，不重领。** 用户在本地会话明确回复“批准064，按可用CPU资源执行”。 064已实现原重放拒绝后保存现有数值现场并保持FAILED；正常推理、模型、容差与七臂协议不变，351项测试通过。本次已批准一次CPU4、零GPU／训练、完整V300七臂、4小时仅提醒，失败停止且不自动重试，不打开T_test；不转换旧pickle或增加独立探针。历史额度均不恢复。

**服务器回传（064 终态）**：北京 17:30:19 提交、17:34:44 结束，unit `mtad-ae975f904c3a`，checkout `34106d5ac32c…`，块 SHA256 `80b4a574…` 前后一致、占位符替换 0 处，`--max-seconds 14400 --time-policy advisory`（提醒未触发）、`--cpu-threads 4`、零 GPU；块内不含 `--diagnose-first-scene`／`--probe-motion`／`--probe-script`。`FAILED`／退出码 1／`completed_scenes=0`／`summary.json` 未生成，stderr 1,698 字节含完整回溯、stdout 0 字节。**保全代码生效**：`failure_snapshot_saved=true`，`replay-failure/report.json` 17,431 字节＋`snapshot.npz` 363,480 字节（29 个数组：11 类 motion、OLS 组、当前与旧 057 的 prediction/routing 各六类），`extra_partition_forwards=0`、`model_updates=0`、`comparison_error=null`，即保全未新增推理。**失败场／臂实测为 `scene_id="1"`、`failed_arm="direct_only"`**，其 `error` 串与 062 **逐字符相同**（各 885 字符）。越界只有 **1 个元素** `[46,8,1]`（属性 speed，`previous=-0.03522947430610657`、`current=-0.035231105983257294`、`diff=1.6316771507263184e-06`、`bound=1.3522947028832277e-06`、`ratio=1.2066`）；异常串里的 `max_utility_difference=1.52587890625e-05` 落在 `[11,29,1]`（`|previous|=74.177`、`ratio=0.0205`）**并不越界**，两者不能混用。其余五类数组通过（`helper_weights` 最坏 `ratio=0.7132`），物理差 position `0.0 m`／speed `6.218e-06`／heading `2.782e-05°` 均在上限内。由已保全 snapshot 只读派生：`ratio>1` 1 个、`>0.5` 5 个、`>0.2` 13 个；逐位相等 38.27%；`diff/ULP(float32)` 中位 1.0 而越界元素 438；`previous` 范围 −34.903…92.881，`|previous|<0.1` 有 96 个、最小 0.0010787；若只按 `atol=1e-6` 判则 1,617 个越界，说明小幅值处闸门实际等价于 `1e-6+1e-5|previous|`。以上只是计数与位置，不构成机制结论。V reference 确有读取（`reference_members_read=300`、`observation_members_read=600`、`training_scene_ids=[]`、`test_opened=false`），准备报告与 056 那份只差 `elapsed_seconds`，只回传摘录。**勘误**：062 回执里"失败臂 helper_quality"是我按 `LEARNED` 顺序的推断，被本轮 `failed_arm` 证伪，实际是 `direct_only`；同一段"|u|<1.4259 就会判失败"也是把最大绝对差当越界元素。063 与那三次只读探针测的都是 helper_quality，其"未重现"只对该臂成立，未测 direct_only。两处更正已分别追加到[062回执](S2L-20261022-062.md)与[063回执](S2L-20261022-063.md)末节，原记录未改写。未重试、未放宽容差、未跳场、未加诊断前向、未打开 T_test；056／057 只读，062／063 现场原样保留。回传 12 件 64 KB；`snapshot.npz` 与完整 `preparation.json`（85,153,502 字节）留服务器只登记路径与体积。详见[064回执](S2L-20261022-064.md)。

**服务器现场排查（064，用户本会话授权；只读、未改容差、未重跑、零GPU）**：057 执行 commit `3e12df3` 与 064 的 `34106d5` 之间，motion 构造／模型／普通融合控制／checkpoint 载入／dataset／physical 六个文件**逐字节相同**，`predict_partition` 函数体也逐字相同；唯一变量是设备（057 system 阶段 `--device cuda:0`，064 为 CPU）。用 snapshot 重建同一组 `[11,12,15]` 的输入后，CPU float32 重算与快照 `current` **逐位相同**（`-0.035231105983257294`），CPU **float64** 参照为 `-0.03522998870843787`，**恰好落在 CPU float32 与 057 GPU float32（`-0.03522947430610657`）之间**（距两者 `1.117e-06`／`5.144e-07`），即两个 float32 值是同一真值的舍入。utility 末层 64 项 `sum|w_i·a_i|=27.365` 而代数和仅 `1.080`（**抵消放大 25.3 倍**），该量级下**一个 float32 ULP 就是 `1.907e-06`，已大于闸门 `atol=1e-6` 地板**，故跨设备比较天然可能越界，这解释了 9,984 个元素里为何只有 1 个越界、且落在 `|previous|=0.0352` 的小幅值处（也解释 helper_quality 通过：其 utility 跨度仅 −4.37…6.30）。**决策层面无影响**：该元素 helper 权重两侧都是 `0.0`（都拒绝）、符号翻转 `0/9984`、接受集合 `relu>0` 两侧各 4,425 个且**逐元素相同**、`helper/anchor_weights` 最大差 `8.23e-07`、`prediction_learned` 的 lat/lon 差为 `0.0`、speed `6.218e-06`、heading `2.782e-05`、`mean_supported` 差 `0.0`。服务器未改任何口径；给本地三个可选（详见回执）：(A) 闸门改为"接受集合逐元素相同＋权重与物理输出沿用现容差"，utility 仅作诊断量（本例会通过，不需放宽数字）；(B) 给 utility 一个与自身量级挂钩的 ULP 地板（属放宽，需本地论证）；(C) CPU 重生成 057 参照——不可行，存档已是 float32、其 float64 值不可恢复。服务器建议 (A)，但是否采纳与是否发起 065 由本地与用户决定，064 额度已消耗、未自行重启。证据：[064回执](S2L-20261022-064.md)末节、`results/LEVEL1-FROZEN-FUSION-20261022-064/server-side-root-cause.json` 与探针脚本／输出。

062已由用户在服务器会话批准“给062分配CPU4预算并执行交接”，服务器于北京时间14:49:33提交，单元`mtad-3374238ad9d0`。本地已核对并接受授权记录；该状态来自Git回执，并非实时进程检查。

**服务器回传（062 终态，事实细节只在回执与产物，不在本索引累积）**：北京时间 14:49:33 提交、14:53:16 结束，`FAILED`／退出码 1／`elapsed_seconds=222.47886635409668`，`timed_out=false`、4 小时提醒未触发。失败在第 1 场（`completed_scenes=0`，`stage=frozen_inference_and_cache_replay`）的 helper_quality 原 OLS 组重放：七项检查只有 `utility_close=False`（`max_utility_difference=1.52587890625e-05`，恰 2^-16），`membership_exact`／`ordinary_control_exact`／`anchor_weights_close`／`state_span_close`／`physical_close`／`helper_weights_close` 均为真，物理位置差恰为 `0.0 m`。`summary.json` 与 `scene-summary.json.gz` 未生成，本轮**没有任何七臂成绩**。V reference 确有读取（`reference_members_read=300`、`observation_members_read=600`、`test_opened=false`、`optimizer_updates=0`、`gpu_used=false`）。三次独立只读重建（独立进程／线程 1…32／内存偏移 0…7／复刻单元的 `OMP/MKL/OPENBLAS=4` 环境）都得到 `max_utility_diff=1.072884e-06`、超容差元素 0/9984、最坏点只用容差的 39.3%，**作业那次的 1.53e-05 未被复现，机制未定**。服务器未放宽容差、未删难场、未提交第二次；056／057 源件只读未变。

062实现冻结于`02c0d1e`：零训练、零GPU，保留056分组/代表/完整人口，比较三个057融合模型及四个普通控制；4小时仅提醒，失败不自动重试。完整输入、命令、容差、产物和停止条件只维护在[062交接](L2S-20261022-062.md)，不在本索引复制。

**063本地审查：** 首场helper_quality旧OLS重放七项通过，utility越界0，最坏容差比0.39346；第二遍与首次输出、utility及权重逐值相等。当前CPU学习输出相对057缓存仍有微小差异，最大位置差0.98054毫米，不能写成逐位相同。完整旧产物核验已执行，仅代表旧缓存自身复算一致。[063回执](S2L-20261022-063.md)保留原记录，完整NPZ仍在服务器。

详细数值、未排除假设与下一步见[研究状态](../docs/research_status.md#063原入口首场通过但062根因未定)。首场通过不保证其余场景和另外两臂；未回传的062失败数组和057历史特征不能补造。

## 分支归并交接

用户已同意统一main并清理旧分支、旧工作目录。所有旧分支都是当前成果的祖先，无独有未合并提交；本地仅使用`/Users/a1/Projects/association-code`。历史实验通过Git提交查阅，不靠旧分支名保留。

服务器下一次回传请在**主协作仓库**先保存未提交更改，再`git fetch origin --prune`、切换`main`并`git pull --ff-only origin main`；后续S2L提交与push均到main。如果已有旧分支未推送回执，先将其提交保全并合入main，不能reset丢弃。**062失败checkout保留`02c0d1e`供复核，不覆盖失败产物；迁移不重启任务。**

## 本轮仓库精简交接

用户要求清理整个仓库。清理前完整版本为[`5776cd3`](https://github.com/HKillua/association-code/tree/5776cd3)。已结案交接、冗余结果包、旧方案及退役代码/专用测试移出当前工作树；必要事实收进[历史摘要](../docs/history/README.md)，当前判断收进[研究状态](../docs/research_status.md)。[结果目录](results/README.md)保留明确标注抽取范围的小型证据，不能冒充原始完整缓存。

维护验证：main第二轮精简后336项测试通过，15个CLI帮助入口可用，本地文档链接无断链。旧036反馈、054/058一次性流程、pair标签与独立融合包装器已退役，实际模型／指标／训练参数不变。062冻结提交仍是`02c0d1e`；当前分支的共享依赖也已通过062贯通测试。清理没有产生新的实验结果或执行授权。

交给服务器AI的维护说明：

- 本轮维护不新增执行授权；062已有独立授权并已提交，不因pull、配置或旧交接中的READY重复启动。
- 服务器`runs/`、权重、数据和已有隔离checkout完全不动；062失败现场保留原冻结版本，不在该目录pull或重跑。
- 新057/061回执的更正已保留在摘要：训练累计helper路由不是终点V路由；融合训练用身份构造污染组；真值身份支撑并集不是Anchor自身支撑。
- 已结案原S2L/L2S和完整回传从`5776cd3:原路径`查阅；本地删除副本不表示服务器原件可删除。不要将旧大文件恢复进当前树。
- 后续只新增当轮独立L2S/S2L；一轮审查结束后合并结论并清理冗余，不让本索引继续累积历史补注。

## 双端职责

本地AI负责研究、代码、结果分析；服务器AI负责资源确认、执行、监控和事实回传。push不唤醒服务器，本地不SSH值守或跨聊天发指令。当前L2S仅在状态READY且有明确授权时可领取；一次尝试、失败不自动重试。参考[模板](TEMPLATE.md)和[现场规则](../docs/server_autonomy.md)。
