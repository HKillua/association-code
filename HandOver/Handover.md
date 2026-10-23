# 当前交接

2026-10-24。本地已审服务器`9761a84`：**073A/B/C及CPU整表均COMPLETED／退出0，科研验收通过、结案。按实验前的MAE判据，接受Transformer4000为后续位置恢复底座，保留门控为小容量／尾部对照。** 两seed的点MAE、四种MAE加权及最大组件外方向全一致；点RMSE两seed均略差，P99方向翻转，不能称全面更可靠。依据与回执勘误见[研究状态073节](../docs/research_status.md#073审查与后续方向2026-10-23)／[本地复算](results/LEVEL1-POSITION-REPLICATION-20261023-073-summary/local-audit.json)。**074D已完成并完成本地结果审查，A/B/C按最近回执仍RUNNING；当前无新READY，已领取任务不得重领。**

**初始领取快照（D随后完成，当前状态以下表为准）：074A/B/C/D 已全部领取并提交（UTC 16:30:17／16:33:49／16:34:04／16:34:21），四项 RUNNING／零重试，各绑不同 UUID（A index2、B index3、C index1、D index0），A/B/C 4小时、D 6小时仅提醒。** 领取前实测 available 99 GB、PSI io／memory `avg10=0.00`、`/home` 余 774 GB；四张卡都被他人占用（util 13–100%），故把显存与耗时未知的领域臂 C／D 分到两张约 23 GB 自由的卡，A／B 依 073C 同结构实测峰值 19.5 MiB 分到各余 9.2 GB 的共卡，未触碰他人进程。服务器已按 run 自记字段核对：074A 相对 072C、074B 相对 073C 的 config 差异均只有 `scope`＋`position_loss linear→linear_relative_harm`，seed／4000／batch8／lr／clip／结构／数据字段全保持。四份执行块 SHA 提交前后各测相同（A/B `66c3f66f…`，C/D `3a45f4d7…`），块文本零替换。一次提交事故已如实登记：074B 首次执行块在块内 `git fetch` 处传输失败、`set -e` 立即中止，核实零副作用（无 checkout／输出／launch 目录、无进程）后重跑提交成功；这不是实验尝试，未使用重试额度。当时四个单卡名额已满；D已完成不表示自动授权新任务，不重领旧任务。

| 当前074任务 | 状态 | 执行范围 |
|---|---|---|
| [074A](L2S-20261023-074AB.md) | **SERVER_TO_LOCAL / RUNNING**，冻结`417c90d`，UTC 16:30:17提交，单元`mtad-05fbf89329a4`，`GPU-efbe9d07…`(index2共卡)，`profile=harm4000`，`--paired-control`为072C完整原件；config相对072C仅`scope`＋`position_loss linear→linear_relative_harm` | 服务器监督至终态，写 [074A回执](S2L-20261023-074A.md)，完成后按第二节跑一次CPU配对表 |
| [074B](L2S-20261023-074AB.md) | **SERVER_TO_LOCAL / RUNNING**，冻结`417c90d`，UTC 16:33:49提交，单元`mtad-159e011b11b4`，`GPU-1043f26a…`(index3共卡)，`profile=harm-replica4000`，对照073C；相对073C仅`scope`＋`position_loss`。首次执行块在块内`git fetch`处传输失败、零副作用，随后重跑提交成功（不是实验重试，回执已登记） | 服务器监督至终态，写 [074B回执](S2L-20261023-074B.md) |
| [074C PTFM](L2S-20261023-074CD.md) | **SERVER_TO_LOCAL / RUNNING**，冻结`ecac88e`，UTC 16:34:04提交，单元`mtad-42b633569274`，`GPU-6811d9c0…`(index1，自由23,255 MiB)，入口`ptfm_fit.py`＋`--alignment`＋`--paired-control 072C`；无profile，seed/4000/batch8/lr/clip/L2全在config | 服务器监督至终态，写 [074C回执](S2L-20261023-074C.md) |
| [074D TSA](L2S-20261023-074CD.md) | **SERVER_TO_LOCAL / COMPLETED（本地已审为负结果）**：退出0，实耗1,537.733s（16:34:20→16:59:58UTC，远未触达6h建议线），`mtad-7183cf46f69e`，`GPU-c5a00c37…`(index0)；474,624参数、峰值显存2,778,663,936B≈2.59GiB、`optimizer_updates=1000`＋`nonzero_lr_updates=999`＋ITSA/MSTSA各500、1000次调用全被裁剪。**边Macro-F1 0.361985（OLS 0.684686／056 0.806738）、分组共属 0.320892（0.670141／0.765657）、FP 726,469**；TSA自身覆盖127,899/454,952=28.1126%，与056共同点126,631=27.8339%；位置上Mean 8,178.9m但速度与Anchor航向在共同点上有改善 | 本地已审 [074D回执](S2L-20261023-074D.md)（含服务器四处勘误）与 `results/LEVEL1-TSA-NATIVE-20261023-074D/`，不重领 |

全局最多四个已发布单卡任务，准备和I/O可错峰、训练可重叠。A/B/C各4小时、D为6小时仅提醒；各一次尝试、失败保全、不自动重试或加步。各自CPU评价/缓存表预算已含。push不唤醒服务器，READY不等于RUNNING。详细研究边界见[074契约](../docs/specs/evidence-association-fusion.md#074相对初值退化惩罚与领域对照)。

## 073历史结案记录


**073三项及一次CPU缓存表全部COMPLETED／退出0（冻结代码`e08b974`）：A 3,406.8s、B 3,496.5s、C 4,076.4s、CPU表165.4s，零重试、未触达4小时建议线。** 两seed×两方法4000的表已补齐：Transformer点MAE在两个seed与四种MAE加权、最大组件及组件外全部优于同seed门控（072C 590.744／073C 606.624 vs 073A 646.750／073B 635.776），但点RMSE两seed均略差（+1.304／+44.382m），P99方向跨seed翻转（−637.0／+494.5m）；门控自身加训方向也翻转（073A比071B差+13.16m，073B比070好−25.88m）。`gate_prefixes`对两门控臂的初参数／采样器RNG／前1000实际批全`exact=true`，Transformer两臂因换seed无此前缀声明。预定MAE晋级已满足；本地决定与尾部代价一并记录，原件见[073整表回执](S2L-20261023-073-summary.md)。

| 当前任务 | 方向／状态 | 固定工作与下一执行方 |
| --- | --- | --- |
| [073A](L2S-20261023-073.md) | **SERVER_TO_LOCAL / COMPLETED**：退出0，实耗3,406.768s（10:29:02→11:25:49UTC），单元`mtad-78213406112d`，`GPU-c5a00c37…`(index0)；4000更新全成功／32000曝光／1,566,596点，7,873参数，裁剪3,160（79.000%）；step0→final MAE721.1745→646.7498，**但相对071B四项全差（ΔMAE+13.1601）** | 本地已审 [073A回执](S2L-20261023-073A.md) 与 `results/LEVEL1-GATE-LINEAR-20261023-073A/`，不重领 |
| [073B](L2S-20261023-073.md) | **SERVER_TO_LOCAL / COMPLETED**：退出0，实耗3,496.474s（10:29:15→11:27:32UTC），单元`mtad-433981ca17ec`，`GPU-6811d9c0…`(index1)；32000曝光／1,561,794点，裁剪3,137（78.425%）；final MAE635.7757／RMSE3269.4213（本表RMSE最优），**相对070三项好一项差（ΔP99+174.1795）** | 本地已审 [073B回执](S2L-20261023-073B.md) 与 `results/LEVEL1-GATE-LINEAR-20261023-073B/`，不重领 |
| [073C](L2S-20261023-073.md) | **SERVER_TO_LOCAL / COMPLETED**：退出0，实耗4,076.401s（10:29:27→11:37:23UTC），单元`mtad-56e85b7ba16f`，`GPU-efbe9d07…`(index2共卡)；105,154参数、峰值19.5MiB，裁剪3,989（99.725%）；final MAE606.6241／P95 2046.7395，**相对072C四项全差（ΔP99+774.0713）**；纯训练比072C慢29.8%，资源登记不一致，不能唯一归因共卡 | 本地已审 [073C回执](S2L-20261023-073C.md) 与 `results/LEVEL1-SEQUENCE-TRANSFORMER-20261023-073C/`，不重领 |
| [073 CPU表](L2S-20261023-073.md) | **SERVER_TO_LOCAL / COMPLETED**：退出0，一次CPU4／零GPU／零训练零前向，实耗165.404s（11:38:28→11:41:13UTC），9方法同人口、两seed配对均`PAIR_VERIFIED`、两门控前缀全`exact=true` | 本地已审 [073整表回执](S2L-20261023-073-summary.md) 与 `results/LEVEL1-POSITION-REPLICATION-20261023-073-summary/` |

本批共12000新更新、每项单GPU/CPU4/一次无重试、4小时仅提醒；CPU表1小时仅提醒。门控整链粗估约1.3h/项、Transformer约1h，准备/IO与共卡影响实速，不是完成承诺或硬时限。最多三项本批作业，全局上限四单卡，现场FIFO/CPU/内存决定错峰。活动checkout、配置、runs不改；存在输出或领取先核实际状态，不覆盖或重开。

本地去留：预定第一项已满足，采用Transformer作为后续位置恢复底座；尾部与门控加训效应翻转不构成预定MAE晋级失败。073结束，不再加步追榜或选冠军seed/终点。下一优先为领域对照及强恢复模型的可靠性增量，尚未发布新执行任务。PTFM/TSA Level1及有效完整TMTAFF仍须补，不能由本通用序列比较冒称复现。

## 最近已审证据

| 项目 | 结论／原件 |
| --- | --- |
| 072A | 1001个T常量搜索，8000次原071B曝光；唯一g=.028，T loss2.263632。V MAE873.083m，劣于固定.95与071B；曲线先降后升，不是全程单调。[回执](S2L-20261023-072A.md) |
| 072B/C | 各4000更新、32000曝光、退出0，实耗3549.45/3645.43秒；LSTM MAE661.975、Transformer590.744。clip3761/3994不等于Adam有效步长；百步窗口不证明末千步变坏。[B](S2L-20261023-072B.md)／[C](S2L-20261023-072C.md) |
| 072整表 | 13方法同442263位置点、97.21%覆盖、610重复。Transformer对071B256场/29组件MAE更好，other170亦好；RMSE/P99总体略差。[原表回执](S2L-20261023-072-summary.md)／[本地复算](results/LEVEL1-POSITION-BENCHMARK-20261023-072-summary/local-audit.json) |
| 070/071 | 两线性门控seed MAE661.655/633.590；071 LSTM/Transformer1000为684.430/691.358，已不能据此否定时序基线。[070](S2L-20261023-070.md)／[071表](S2L-20261023-071-summary.md) |

072同seed初参数、前1000实际批次和完整缓存人口核验来自服务器原件比较器；本地从300逐场统计独立重聚四权重/38组件，未下载原pt/JSONL/NPZ逐点重放。P95/P99引用服务器数组评价，未平均逐场分位数。原S2L不覆写，本地解释只维护研究状态。无新的确定生产模型或数据故障；本轮比较器中的曝光schema衔接已在本地修正并由真实小型训练检查覆盖。

## 执行／回传规则

只领取当前074表READY或其已满足依赖步骤；旧073为已结案记录，不重领。各项独立写S2L并完成即回传，不等全部。提交实际command/exit/run/config、必要端点/逐场/曝光/窗口loss/clip与耗时和原件路径；完整权重、训练JSONL、预测留服务器，不用同名摘要替换输入。成功必须由内容、命令、退出码和实际数组支持，Git版本只定位冻结代码，无SHA/hash准入。

非有限、来源/配置/人口不符即保全失败，不删点/放宽标准/自动重试。超建议时间记录实耗和进度，继续当前固定工作，不自动杀或加步。全局兼容GPU可共卡，不触碰他人进程，服务托管沿用[服务器规则](../docs/server_autonomy.md)。本地不SSH值守、不跨聊天发指令；**push不会唤醒服务器，READY不代表已启动**。

本次072自动跟进在完成本地审查、073发布与push后结束，不保留每日重复。唯一日常仓库`/Users/a1/Projects/association-code`、main；退役v2不使用。066仍REJECTED，旧待批准、旧READY与历史额度都不自动恢复。
