# 105 原始 CSV 链核查：记录、复核与深挖

## Record：问题、设计与实际执行

`EXPERIMENT_RECORD: EXP-105-RAW-CHAIN-20261027; source_idea_id: IDEA-STATE-DIAG-20261027; plan_version: 1.5-raw-csv-audit; status: completed; next_skill: squeeze-experiment`。事前[计划 §2.7](../proposals/continuous_state_reassessment_20261027.md#raw-chain-audit-105)、[执行合同](../../HandOver/L2S-20261027-105.md)、[服务器回执](../../HandOver/S2L-20261027-105.md)和[脱敏原件目录](../../HandOver/results/LEVEL1-RAW-CASE-CHAIN-20261027-105/)共同构成证据。复核代码为冻结 checkout `d185191` 的 `tools/analysis/audit_raw_case_chain.py`；原始 CSV 和 103 私有证据未回传本机，原行和绝对坐标留服务器。

**问题**：前述三个极大误差案例中，既有轨迹节点及参考端点的异常是原始 CSV 就有，还是 CSV 到 103 retained canonical 的排序、去重、解析及对应时引入？这不是对某个融合方法的性能实验。三个 V 案例在看过误差后选定，分别 3035/t23/q56、2819/t1/q60、1043/t3/q54；按事前合同固定各中心预测组的两节点，以及 1043 q55 的参考端点。六节点完整原量测序列、104 固定事件、目标全部原参考行和固定端点逐项与 103/104 的保留材料核对。该比较同源、无独立物理真值对照；未随机抽样或开展统计推断。

**执行事实**：服务器单次 CPU 托管执行，程序时间 2026-10-27 21:06:51.679–21:06:57.331（北京时间），5.653 秒；systemd launcher 13.748 秒，真实退出码 0，`COMPLETED`。4 线程上限，程序峰值 RSS 153432 KiB；本进程地址空间限额 8 GiB，非 RSS 限额。完整解压遍历 3 场各 1 个标签、量测、reference 成员，共 9 个、5,777,702 解压字节，均未触及 64 MiB 单成员上限。`optimizer_updates=0`、`model_forwards=0`、`npz_files_read=0`、`checkpoint_files_read=0`、`test_opened=false`、`groups_recomputed=false`、`threshold_reselected=false`。未读 T_test 或训练；按回执原材料未修改。

此前本地尝试领取时，启动命令的 `git fetch` 因服务器远端跟踪引用发生并发更新而提前失败；该命令未运行 105 程序、未创建此次 run。随后另一个服务器执行者完成单次实际执行并回传；本次没有重试已启动的诊断，也不能仅凭 Git 错误确定是哪一进程更新了引用。

## 实际数据与核验

本地解压三个公开 case 和 summary，并与 `run.json`、launcher request/status 及脚本定义交叉核对。核查仍使用同一次执行返回的数据，不构成独立重复运行。

| 案例 | 匿名节点及原量测行数 | 固定事件数 | 整节点 canonical、身份映射及固定事件 | 目标参考原行及端点 |
|---|---|---:|---|---|
| 3035/t23/q56 | 4: 200；133: 308 | 11 + 4 | 全部匹配 | 191 行与 103 一致；q56 两端匹配 |
| 2819/t1/q60 | 13: 694；32: 213 | 11 + 9 | 全部匹配 | 89 行与 103 一致；q60 两端匹配 |
| 1043/t3/q54 | 0: 699；14: 315 | 15 + 15 | 全部匹配 | 93 行与 103 一致；q54、q55 两端匹配 |

六节点合计 2,429 原量测行，65 个 104 固定事件，373 行目标 reference。每个节点的标签、节点身份映射、reader 可保留条件、完整 canonical 和事件匹配标记均为真；原始量测行无完全重复、同刻冲突、时间逆序、无效解析／状态或标签时间区间外行。三案 `mismatch_count=0`；每场两节点的完整数组是同一套既定 float 和航向归一语义下精确相等。目标 reference 的全部行含原字符串与 103 retained 对应一致，固定 8 个左右端点均匹配。

需要注意：`global_reference_issue_count` 分别为 3035 = 64、2819 = 1、1043 = 0；这是**整份场景 reference 成员**调用 `association/data/reference_states.py::_read_rows` 产生的 issue 条目总数，不是本目标异常计数、异常行数或位移阈值计数。代码定义的可能条目包括身份格式、非数字、非有限数和同身份同时刻重复／冲突等；本次公开结果未给出具体类型、对应行或是否属于本目标，不能据这组计数判定问题性质或成因。103 保留材料与 105 解析共用 reference reader，故一致性并非解析器或独立真值的外部验证。

## Squeeze：原 reference 邻点的局部现象

`SQUEEZE_REPORT: SQ-105-RAW-CASE-CHAIN-20261027; input: EXP-105-RAW-CHAIN-20261027; evidence_status: partial; next_skill: deep-read Mode C`。下表使用公开 case 中同目标、时间序列真正相邻的不同时刻原行；数值是按既定坐标与秒解释计算的球面位移和时间差，不是现实航速。不是原参考 CSV 物理相邻行的含义：文件可能交错存放不同目标，行号仅是定位。以下是按既定端点及其邻点中事后挑出的明显例子，不是总体分布。

| 场景 | 同目标原行号 | 相邻时差 | 相对球面位移 | 纬度差／经度短弧差（度） | 跨 180° |
|---|---|---:|---:|---|---|
| 3035 | 13775 → 13785 | 4 s | 45.90 km | −0.384859 / +0.158788 | 否 |
| 2819 | 788 → 789 | 6 s | 42.88 km | +0.231045 / +0.328790 | 否 |
| 2819 | 800 → 801 | 2 s | 41.45 km | −0.234636 / −0.308425 | 否 |
| 1043 | 367 → 368 | 5 s | 48.16 km | +0.030366 / −0.458467 | 否 |
| 1043 | 368 → 370 | 35 s | 47.22 km | −0.033300 / +0.449250 | 否 |
| 1043 | 370 → 374 | 25 s | 47.73 km | +0.034500 / −0.453983 | 否 |
| 1043 | 374 → 379 | 48 s | 47.24 km | −0.034916 / +0.449216 | 否 |

**直接观察**：三个事后选中案例的 reference CSV 都在选定端点附近出现很大的短时间相邻位置变化；1043 的四次大位移在经度方向上交替。参考全行与 103 retained 相同，因此这些具体现象不是由 103 保存 reference 时新增出来的。对六个观测节点，所核全部原行按相同规则重建后与 retained canonical 完全一致，固定事件也对应原行；这缩小了“CSV→103 canonical 增加额外变化”这一解释的范围。位移数值从 retained canonical 或 reference 坐标计算，raw 行与之匹配；未由另一个独立数据源验证。

**解释与竞争假设**：这份数据包的原量测和 reference 可以同时呈现这些大变化，故不能仅将大误差解释为网络预测或单个观测源的行为。候选解释仍包括原始材料与 reference 生成或空间/时间变换、时间单位/语义、身份关联、插值规则以及模型行为等；本次未逐一区分。已知数据集论文的逐轴 0.5° 筛选针对生成前母轨，后续仍有插值和误差生成；当前成品 reference 的血缘未知，不能以本表角差推断母轨筛选违规，亦不能凭角差未超阈值推断物理正确。

**边界及不可推出**：同源比较不能证明载入规则、字段单位、参考时间映射、真实身份及真实位置正确，也不能证明 reference 或观测中的哪一方更接近真值；不能排除较早上游处理异常或 reference 插值假设错误。事后挑选的三个案例与邻点不提供全 V 异常率、泛化或其他算法的性能证据。原始目标 reference 全行匹配并不等于全 reference 成员没有 issue，issue 具体内容也未回传。现有 evidence 不支持移除这些难例、用 reference 修补输入、推断真实船速，或据此选择新研究方向。

## 回流 Mode C 的输入

- `claim_scope`：三个事后选择的固定 V 场景、各两个节点和其目标 reference 原 CSV 对应；单次运行、无独立真值。
- `new_F`（DE）：六节点完整 canonical 与原行重建精确一致；三个目标全部原 reference 与 103 retained 一致；三个案例 reference 固定端点邻域均有大位移，1043 有经度方向交替的相邻大位移。
- `new_M`：无直接验证的生成/错误机制；数据生成、时间语义和真值来源 `UN`。不能把原行一致升级成原数据正确。
- `evidence_index`：`case-{3035,2819,1043}.json.gz` 的 `nodes`、`reference.endpoint_checks` 和 `reference.selected_neighbor_times`；`summary.json.gz`、`run.json`、launcher 原件；issue 定义见 `association/data/reference_states.py::_read_rows`。
- `open_questions`：reference 的上游生成与变换、这些行的独立真值；整成员 issue 的具体分类/目标归属未随本回执公开。下一步先将证据合并既有 Mode C 并与用户讨论，不能自动扩范围或启动新训练。
