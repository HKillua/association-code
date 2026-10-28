# 106 新单位人口准备：Record / Squeeze

**验收通过。** 106 实际完成固定 T2100/V300 的新单位准备，程序与 launcher 均为 `COMPLETED`，退出 0；全部 10 项事前指定人口计数匹配。独立重聚合 2400 场公有统计、核对 7200 个成员和 9600 条进度记录后，没有发现汇总或范围不一致。训练 TCM eligible 为 155134 条，因此 **batch64、`ordered_fragment` 下 2424 更新恰好一遍，末批 62 条且不补齐**。这回答了预算 A 的一遍规模，未证明训练充分或模型有效。

本地 Codex 经用户授权的 SSH 通道取得服务器实际终态与白名单产物，再作本地审计；**本文和 [S2L-106](../../HandOver/S2L-20261027-106.md) 是本地据实补写，不是服务器 AI 提交的回执。** 本地没有重跑准备、打开原 ZIP/封存测试或训练。当前执行安排只看[交接](../../HandOver/Handover.md)。

## 1. EXPERIMENT_RECORD

| 字段 | 记录 |
|---|---|
| 包类型 / 版本 | `EXPERIMENT_RECORD` / `research-pipeline-contract v1.0` |
| 实验 / 包 ID | `EXP-LEVEL1-TMTAFF-UNIT-PREP-20261027-106` |
| 来源 Idea / 计划 | `IDEA-TMTAFF-UNIT-PREP-20261027@1.0`；[事前计划](../proposals/tmtaff_unit_population_preparation_20261027.md)，源矩阵 `MC-RESTART-20261027@1.10`，C01 同协议基础 |
| 研究类型 | 计算研究的数据准备与人口审计；非训练、非方法效果实验 |
| 记录日期 / 状态 | 2026-10-27；`completed`，公有产物验收通过 |
| 研究问题 | 显式量测 kn→m/s、reference 保持 m/s 后，固定 T/V 的合法节点、选点、候选对和有效监督是否保持预期，准备成本是多少？ |
| 主要预测 / 估计目标 | 转换只作用于输入速度，既定标签/时间/选点/隔离规则不变，10 项旧人口应保持；新增完整正对监督分布及资源统计 |
| 对照、重复、随机化 | 旧 088/准备记录作历史数量对照，没有重跑旧单位；一次固定全量 T/V 准备，无随机抽样、训练重复或显著性检验 |
| 执行 / 审阅 | 本地 Codex 单次提交服务器托管，服务器程序实际执行；本地 Codex 读取终态并独立审计白名单 |
| 代码与入口 | 冻结 checkout `/home/wangguangjie/djs/association-code-run106`，启动记录对应 `939bba4`；[inspect_tmtaff_population.py](../../tools/analysis/inspect_tmtaff_population.py)，实际完整 argv 在 `request.json/run.json` |

### 1.1 设计与实际配置

实际读取固定 Level1 三分卷、现 manifest 和 alignment，仅 T_core2100 与 V_select300；每场标签、量测、reference 三类成员完整读取。T_test600 和 unused2000 payload 不读。沿生产 `load_timed_dataset`，Q64、cap128、真实秒归一尺度20、`shortest_arc`、原 offset/gap600，速度策略 `measurement_kn_reference_mps_v1`。配置与本地[非训练配置](../../configs/tmtaff-unit-population-inspect.json)逐项相同。

程序未构造模型、未启动 CUDA/GPU 训练、未读取 checkpoint、未拟合 normalizer；0 前向、0 优化更新。1000/4000 仅进入解析期望公式，不进入训练。新单位物理解释的来源仍是已核前8个训练场，不能由本次全量计数相同扩张为“全包每行单位和真值已独立验证”。

数据血缘：原分卷固定 T/V 成员 → 原生读取及既定隔离/选点/时钟/reference valid → 内存准备数组 → 逐场计数与正对监督直方图 → split 汇总及解析覆盖期望 → 本地重新聚合与结论。身份、节点及完整原始准备记录留服务器 `private/preparation.json.gz`；本次本地没有取得该私有文件，也没有逐数组验证数值。

### 1.2 真实执行、成本与限额

| 项目 | 实际记录 |
|---|---|
| 程序时间 UTC | 2026-10-27 13:42:46.189708 → 14:08:12.504591；1526.314939 s |
| launcher 时间 UTC | request 13:42:45.803507 → 14:08:14.417909；1528.615341 s；退出 0 |
| 北京时间 | 程序约 21:42:46 → 22:08:12 |
| 托管 | systemd `mtad-97b36945f599.service`；提交记录 `Linger=yes`、`Restart=no`；未超时、未触发 advisory、无失败或强制清理 |
| CPU / GPU | 线程上限4，OMP/MKL/OpenBLAS 均为4，`CUDA_VISIBLE_DEVICES` 为空；不把线程上限当作实测平均 CPU 利用率 |
| 峰值 RSS | Linux 原始 `ru_maxrss=3151588 KiB`＝3077.722656 MiB，约3.005589 GiB；是本进程 RSS，不是显存 |
| 内存限制 | `RLIMIT_AS=34359738368 bytes`＝32 GiB，作用于本进程虚拟地址空间；不是 RSS 上限 |
| 单成员限制 | 目录声明解压上限268435456 bytes；最大实际被读成员声明35676784 bytes，均未超限 |

`progress.jsonl` 有9600条：两阶段×2400场×`READING/COMPLETED`，每个组合各一次，时间单调，范围均为固定 T/V。观测阶段首条读取 elapsed7.854049 s、末完成977.219473 s；reference 阶段977.228792→1438.781059 s。场内日志区间求和分别967.083825/459.640205 s；它们包含阶段内计算，不能称纯磁盘 IO 或 GPU 成本。最终汇总/写盘等另占时间，不用最初几场速度外推正式训练耗时。

### 1.3 成员范围与主人口

7200 个目录成员和7200个实际读取记录逐一匹配，成员名无重复，全部属于固定 T/V，每场恰有三类成员。实际记录中标签/量测4800项带 `kind`，reference2400项由其独立 schema 标识；核验没有把这两种结构误作缺项。

| 成员类型 | 数量 | 目录声明压缩 bytes | 目录声明解压 bytes |
|---|---:|---:|---:|
| 标签 | 2400 | 2723062 | 7874575 |
| 量测 | 2400 | 1169392611 | 3412492949 |
| reference | 2400 | 131728306 | 527542805 |
| 合计 | 7200 | 1303843979 | 3947910329 |

这些是 ZIP 目录声明尺寸和成功成员范围，不是实测物理磁盘读取量。

| 人口 / 监督对象 | T_core | V_select | 定义与边界 |
|---|---:|---:|---|
| 场景 | 2100 | 300 | 与现名单完全一致 |
| 合法节点 / TCM eligible | 155134 | 24599 | 每轨原生选点至少2个 |
| cap 后原生选点 | 17932752 | 2844556 | 模型输入选点，非全部原生 CSV 行 |
| 双向相邻 transition | 35555236 | 5639914 | 每轨 `2×(n_selected−1)`，非乘特征维度后的元素数 |
| 全候选对 | 27632584 | 4439092 | 每场均为 `n(n−1)/2` |
| 正 / 负对 | 155159 / 27477425 | 24675 / 4414417 | 不等于训练呈现或独立样本数 |
| 目标 | 57359 | 9100 | 场景内 A-B 目标，非原始独立船舶 |
| 无输入目标 / missing_identity目标 | 0 / 0 | 0 / 0 | 后者只表示对应 reference 没有原行的身份数，不表示所有 query 都有真值 |
| 节点隔离 / 完全重复删除 | 0 / 0 | 0 / 0 | 量测准备计数，不是 reference 无问题证明 |
| reference 原行 / issue事件 | 7834964 / 20307 | 1277263 / 3937 | issue 是事件数，不是错误行数、坏目标数或物理异常率 |

事前 T 的场数/节点/选点/候选/正/负6项，V的场数/节点/候选/目标4项全部一致；其他量是本次实际记录，未补造事前预期。`target_without_reference_count=0` 与存在没有有效监督的正对不矛盾：`missing_identity` 定义仅为 `identity_rows==0`，合法性、query范围及 gap/mask 仍可使有效监督为零。[定义](../../association/data/targets.py)

### 1.4 目标有效点与正对有效监督分开

| 属性 | T 目标有效 query | T 正对有效 query 总次数 | V 目标有效 query | V 正对有效 query 总次数 |
|---|---:|---:|---:|---:|
| lat | 2836572 | 7695554 | 454953 | 1236078 |
| lon | 2836590 | 7695599 | 454954 | 1236084 |
| vel | 2837070 | 7696897 | 455162 | 1236659 |
| cou | 2834449 | 7689531 | 454689 | 1235350 |
| 位置 lat/lon 联合 | 2836570 | 7695554 | 454952 | 1236075 |

左类按目标计一次；右类对池内每个正对按其目标 valid mask 计数，同一目标可能有多个正对。右类是**正对池的一次枚举监督总量**，不是未来 A 实际抽样训练的有效监督，也不是独立标签数。V位置454952与既有全人口评价真值点数一致，不表明所有方法预测覆盖都相同。

每场每属性0–64个有效 query 的65格直方图都保留，格数和为正对数、加权和为上述总次数。逐场直方图、全体 scalar/nested 计数重聚合与返回 summary 一致，解析期望以独立公式复算也一致；合计60778项元数据/计数/范围/时间检查无失败。[本地复算结果](2026-10-27-unit-population-audit.json)

### 1.5 偏差、缺失与验收边界

未发现相对106预定范围、配置或终点的实质偏离；一次完成，无重试、删场、改名单、追加训练或改预算。取得的公有产物包括原始 `request.json/status.json`，不把启动时 `RUNNING` 副本当作终态。服务器 AI 尚未提交本次回执，因此由本地根据实际 SSH 读取补记；来源角色已明示。

本地审计覆盖公有2400场统计和成员/日志完整性，未重放原始数组、每个正对的身份绑定或 valid mask；后者的生成代码已静态核对，不能等同第二次独立原始数据解析。原始 reference 的生成语义、真实时钟和物理真值仍未知；无新损失、预测、训练收敛或精度证据。软件环境可定位到请求中的 Python 环境路径，但公有产物没有完整 Python/NumPy 等版本清单，不能补写具体版本。

## 2. SQUEEZE_REPORT

`squeeze_id: SQ-LEVEL1-TMTAFF-UNIT-PREP-20261027-106`；`input_experiment_record_ids: EXP-LEVEL1-TMTAFF-UNIT-PREP-20261027-106`；包版本 `research-pipeline-contract v1.0`。计数与公有血缘 `verified`，原始数组、物理真值及训练充分性 `partial/unknown`。本轮资源边界是现有公有产物和静态代码；不新增原数据扫描、随机抽样模拟、训练或效果实验。

### 2.1 主要问题已回答：A 的 2424 确实是一整遍

训练代码构造 transition 池的条件是归一后原生 mask 至少2个点；106沿相同原生 mask 计得155134条，全部合法节点 eligible。[TCM 池与有序采样](../../association/training/tmtaff.py)

`ceil(155134/64)=2424`；前2423批各64条，合计155072，末批62条，总155134。`_ordered_tcm_indices` 的短尾不回填下一轮，因此一遍没有为了凑 batch 重复前两条。若未来按该配置成功完成全部更新，轨曝光应为155134、双向 transition 呈现35555236；**这是已核人口与代码导出的计划尺度，不是106实际训练过这些样本。** 一遍只限定暴露量，不代表收敛或充分复现80 epochs。

### 2.2 有效监督并非每个正对都是64点

训练正对位置监督均值49.5979个 query，中位61、P10为18、P90为62；验证均值50.0942，中位61、P10为20、P90为62。完整64位置点的正对只有训练1101、验证163。分位数采用直方图最近秩法，是池内描述，不是训练抽样分布或置信区间。

训练155159正对中8对四属性全部零监督，51对至少一属性零监督；验证24675对中分别1/11。零监督属性的分布：T lat/lon/cou各8、vel51；V lat/lon/cou各1、vel11。所有零监督正对都保留，未据此删除、重抽或改生产采样。

**没有任何训练场或验证场的某属性正对池总监督为0。** 训练8个全空正对分别位于8场、各1对；这排除了“抽到某场就必然该属性无监督”的一种确定失败条件。它不保证未来每个随机 batch 的四属性都有监督：TFM每步先选一个场景，再从该场正对池取64对；有效 batch 任何属性总 valid 为0时现代码报错并要求不重采样。[采样](../../association/training/tmtaff.py)；[loss前置约束](../../association/training/fusion.py)。本次没有消费训练 RNG，也没有模拟或保证未来抽样路径。

### 2.3 1000 更新并不是正对池的一遍

训练2100场都有正对；场内池最小6、中位30、最大1773，1542场不足batch64。采样是场景等概率，再在场内均匀抽正对；池不足64时有放回，因此不是全库正对等概率，也不能把64000次呈现说成64000个不同正对。

| 固定更新数（仅解析） | 正对呈现 | 预期不同场景 | 预期不同正对 | 占池比例 |
|---|---:|---:|---:|---:|
| 1000 | 64000 | 795.7431 | 28452.9016 | 18.3379% |
| 4000 | 256000 | 1787.5398 | 76490.9977 | 49.2985% |

本地用本次2100个真实池规模重新计算：池大小 n<64 时，某对被一次场内 batch 包含的概率为 `1−(1−1/n)^64`；否则为 `64/n`；再乘场景选中概率1/2100。U步至少被选中一次的概率为 `1−(1−p)^U`，对全部正对求和。此处与程序使用的数值稳定形式一致。它们是单个采样设计下的期望，既非实际覆盖、泛化证据或充分训练下限，也不授权追加4000步。

### 2.4 观察、解释与替代解释

| DE观察 | AI解释 / 可以支持 | UN / 不能排除 |
|---|---|---|
| 新单位准备的10项旧人口完全相同 | 该显式转换没有改变这些指定人口规模，A一遍尺度可明确 | 同计数不证明每个状态数值或reference物理语义完全正确 |
| 正对池的逐属性监督数量不同，存在零监督对 | 训练暴露应记实际逐属性有效点，不能按64满监督估算 | 这种缺失对旧失败或未来效果的因果贡献未知 |
| 所有场景的四属性池总监督均正 | 没有“必选到无该属性监督场”的确定失败场 | 随机batch仍须沿原失败条件检查，不据统计放宽约束 |
| 1000步预期只访问约796场、18.34%不同正对 | A是固定预算可行性诊断；总步数不是全池遍数 | 多访问/多训练是否提高精度、是否过拟合、是否充分都没有效果证据 |
| CPU全量准备约25分26秒、RSS约3.01GiB | 可给后续同数据加载的资源规划提供一次实测 | 新GPU训练与端点评价成本不能由此保证；一次环境不可比较架构快慢 |

没有训练轨迹可供收敛/梯度分析，没有新预测可供误差或机制归因，相关 Squeeze 步骤不适用。异质性来自完整池直方图和场景规模，无挑选最优子集；全人口列完后不增加人为统计显著性。

## 3. Mode C 回流建议与后续路由

`mode_C_input_bundle`：

- **来源**：`SQ-LEVEL1-TMTAFF-UNIT-PREP-20261027-106`，计划 `IDEA-TMTAFF-UNIT-PREP-20261027@1.0`；本报告§1、§2及公有原产物。
- **new_P**：none。沿既有 P01 同协议比较，不能为一次准备增加独立研究问题。
- **new_F（必要的一条 DE 增量）**：显式新单位下固定 T/V 的指定人口保持；正对监督存在逐属性缺失，TCM一遍与TFM场景采样的暴露单位不同。范围仅为此次 Level1、loader、固定名单及既定mask，数值见§1.3–1.4/§2。
- **new_M**：none；采样概率→不同正对覆盖期望是已有采样代码的解析结果，不是解释模型失败的新因果机制。
- **证据索引建议**：新增一个106执行证据条目（如后续可用 E26），接 P01/C01 的执行条件；F编号由主矩阵统一分配，不在本报告另造正式矩阵或改变候选排序。
- **P×F连接**：P01 为 supported 的人口/预算单位澄清；P05 仅支持描述监督和采样分布，不证明其导致旧错误；P02/P03/P06 不增加物理真值、尾部根因或不确定性校准证据。
- **必须保留的边界**：同计数非逐数组复验、身份共享及V反复开发、reference生成未知、无训练充分性/效果、解析覆盖非实际训练。
- **next_routes / next_skill**：交 `deep-read Mode C` 作最小证据增量；后续训练由独立事前合同和当前用户决定控制，不能从本准备自动加预算、换方法或重采样。本文不改矩阵、研究状态或领取索引。

本记录使用 [record-experiment](/Users/a1/.codex/skills/record-experiment/SKILL.md)、[squeeze-experiment](/Users/a1/.codex/skills/squeeze-experiment/SKILL.md) 和 [deep-read Mode C](/Users/a1/.codex/skills/deep-read/SKILL.md)，完成事实→解释边界→回流输入，不把人口准备包装为方法创新。

证据原件：[run](../../HandOver/results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/run.json)、[launcher request](../../HandOver/results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/request.json)、[launcher status](../../HandOver/results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/status.json)、[summary](../../HandOver/results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/summary.json.gz)、[scenes](../../HandOver/results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/scenes.json.gz)、[inventory](../../HandOver/results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/member-inventory.json.gz)、[read-members](../../HandOver/results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/read-members.json.gz)、[progress](../../HandOver/results/LEVEL1-TMTAFF-UNIT-PREP-20261027-106/progress.jsonl)。本地复算只读这些公有产物和元数据，未重跑原任务。

## 4. 2026-10-28事后Squeeze：片段数量与目标级训练测度

本增量由用户要求分析矩阵时，E01的pair训练/目标评价与E26的正对人口交叉提出。**是对已完成106的事后再分析，没有原事前问题假设，不倒填Idea或预注册。** 输入仍是上面的2400场公有`scenes.json.gz`，以及已读数据/训练/评价代码；未读取ZIP、私有准备数组、107日志/权重或封存测试，0前向、0训练、0随机抽样。结果见[派生汇总](2026-10-28-fragment-weighting-reanalysis.json)。原106验收结论不改变。

### 4.1 可由三个公有计数准确恢复什么

对场景s，设T为有输入目标数，K为合法节点数，P为正对数，第j目标节点数为k_j。106全场`target_without_input_count=0`；[native.py](../../association/data/native.py:79)枚举所有i<j、身份相同即正对，故

```math
K=\sum_j k_j,\quad P=\sum_j\binom{k_j}{2},\quad
\operatorname{Var}_{j\in s}(k_j)=\frac{K+2P}{T}-\left(\frac KT\right)^2.
```

这是同场目标的**总体描述方差**，不是估计误差方差或抽样置信区间。用整数`T*(K+2*P)-K*K>0`判断k是否不全相同，避免浮点误差制造异质性。若有目标无输入、正对过滤或其他正对语义，公式须重核，不能机械推广。本次另核每场候选数为K(K−1)/2、位置监督直方图之和为P；均通过。

| 既定人口 | k不全相同的场景 | k全部相同的场景 | 同场k方差中位／P90 | 同场k变异系数中位 |
|---|---:|---:|---|---:|
| T_core2100 | **2082/2100（99.14%）** | 18 | 0.710010／1.061224 | 0.311273 |
| V_select300 | **299/300（99.67%）** | 1 | 0.719894／1.061901 | 0.315739 |

独立阅读者用同一公有JSON复算上述计数及中位方差一致，这属于独立算术核验，非独立数据复验。**k统计全部来源的片段数，尚未分离来源可见性与同源切碎程度。** 该描述覆盖现有固定人口，不是新的随机样本或独立复验；T/V有共享身份且V反复开发，不据两个分割数值相近推断外部泛化。计数只能恢复k的一、二阶矩，不能恢复每目标k、k≥2目标内部的完整分布，或与误差/机动/有效query的联合关系。

### 4.2 与TFM训练、目标评价的具体联系

当前[TFM_FROZEN采样](../../association/training/tmtaff.py:826)先均匀选有正对场景，再在场内均匀抽正对。有放回和不放回的期望身份呈现均满足

```math
\Pr(J=j\mid S=s)=\frac{\binom{k_j}{2}}{P_s},\quad
\mathbb E[N_j\mid S=s]=B\frac{\binom{k_j}{2}}{P_s}.
```

因此在该人口/采样下，片段数改变会改变被抽中的目标分布；它不是任意想象出来的接口缺项。**但这还不等于证明有害偏置。** 新正对可以增加有价值的输入组合；pair预测和一组一输出预测本来不同，训练损失与mME/MAE等也不相同，不能要求所有优化权重都等于评价权重。

仅在所有pair有相同有效query长度、先忽略不同pair预测与特征损失缩放时，场内目标权重才简化为`C(k_j,2)/P_s`。实际[fusion loss](../../association/training/fusion.py:106)按整个batch每属性有效点归一，[属性目标](../../association/models/fusion.py:247)又有不同函数及任务系数；其系数涉及`N_j*d_j / sum_l(N_l*d_l)`，不能把随机分母换成其期望，也不能将身份呈现概率直接叫精确梯度权重。TAM关联指标本来以pair计，**本问题不把pair级关联训练判为错误**；只追问TFM的目标级融合效益。

正对池位置监督枚举总量/全目标各计一次的有效query比值，T为2.712979、V为2.716935；场比值范围分别0.828283–6.000000、1.345763–5.685484，中位2.749358/2.727841。这些不是训练曝光倍数或独立标签量。T最小比小于1也说明不能把该比简单称为“监督重复倍数”：全目标分母可包含没有正对的单轨目标；具体数量尚未从每目标原件恢复。

### 4.3 观察、机制假设、反例分开

- **DE＋代数**：现人口绝大多数场的目标片段数不全相同；在当前全正对/场内均匀采样下，对应身份呈现分布随片段数改变。读取及派生范围完整，统计为确定性描述，无需制造显著性p值。
- **AI／待验**：如果片段形成主要反映报告切割而非更多独立信息，训练可能按切割粒度偏重某些目标，使pair风险与目标级风险发生不利偏离。尚未进行同一观测并集的重切片干预，也没有损失/梯度/误差的归因。
- **反例/竞争解释**：更多片段可能承载真实中断、reset或不同噪声制度，更多pair可能正当地提高覆盖；同目标监督多次不等于独立噪声重复计数。相同k的场也可能因valid长度不同而有权重差异。
- **UN**：是否造成总体或尾部退化；是否与旧失败相关；作者原采样是否同当前实现；领域是否已明确识别这个精确问题。一般类别不均衡、目标平衡采样和重复信息已有近邻，不能据本统计宣布新颖。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-106-FRAGMENT-WEIGHTING-20261028
input_experiment_record_ids: [EXP-LEVEL1-TMTAFF-UNIT-PREP-20261027-106]
source_idea_id: none_posthoc_reanalysis_of_existing_record
claim_scope: fixed_2100_training_and_300_development_public_scene_counts_and_current_code
data_quality_and_provenance_gaps: per_target_arrays_and_joint_error_distribution_not_accessed; shared_identity_and_development_use
primary_result_with_uncertainty: deterministic_2082_of_2100_and_299_of_300_nonuniform_k; no_independent_repeat_or_performance_measure
process_dynamics: no_new_training_or_optimization_trace
mechanism_and_intermediate_findings: k_changes_pair_count_and_identity_sampling_probability; performance_path_is_AI
errors_heterogeneity_and_boundary_conditions: per_scene_count_heterogeneity_only; finite_batch_valid_denominator_and_singletons_separate
robustness_sensitivity_and_alternative_explanations: exact_integer_test; full_fixed_population; no_resegmentation_intervention; informative_fragment_boundaries_possible
new_P: candidate_fragment_partition_and_target_risk_question_not_confirmed_domain_discovery
new_F: per_scene_fragment_count_heterogeneity_and_conditional_identity_weighting
new_M: AI_fragment_partition_changes_sampling_measure; harm_unknown
evidence_index: report_section4; derived_json; original106_scenes; native79; tmtaff826; fusion106
open_questions: actual_target_weights_and_error_relation; harmless_vs_informative_partition; nearest_field_problem_novelty
next_routes: deep_read_Mode_C_increment; focused_novelty_and_condition_review
next_skill: deep-read Mode C
```

复算方法完整由§4.1公式和原公有逐场字段给定：按split取全部行，分别以整数检验计数、计算上述方差及`CV=sqrt(T(K+2P)−K²)/K`；各分位采用索引`(n−1)p`的线性插值。只解析这一个公有JSON即可复核，不依赖新数据或拟合参数。该增量不提出改采样或增加107预算的执行许可。
