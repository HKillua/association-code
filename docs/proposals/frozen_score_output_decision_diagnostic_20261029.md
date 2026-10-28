# C53：固定边证据之后，分组决定是否引入可避免的身份风险

2026-10-29；`IDEA-FROZEN-SCORE-DECISION-20261029`，plan_version 1.0，模式3（假设挑战）。来源为[Mode C1.28](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-128)的单个C53。**研究设计，尚未实现、执行或发布READY。** 第一阶段只诊断固定缓存上的关系决定，不训练、重算关联分数或宣称物理恢复收益。当前安排只在[研究状态](../research_status.md)，执行许可只在[交接](../../HandOver/Handover.md)。本设计不改变107/108。

## 1. 问题是怎样从矩阵得到的

[C53六来源/20格矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/c53_stepwise_problem_matrix_20261029.md)区分两个命题：A，fragment层的候选解是否能表达合法身份关系；B，合法候选内，静态pair评分与一次硬决定是否漏掉条件组状态证据。MH-T2TA已有相容多假设，TPMBM/MeanTraj已有成熟轨迹集合输出，因而“使用多假设”本身不作为创新。Gelato的partial-decision条件化提供外域机制，但其单射、GED成本和安全约简定理不直接迁移。

本地代码提供一个更小且可以先回答的问题：**同一命名图、同一分数/阈值，改变anchor处理顺序，最终partition是否改变；改变是否影响解码后的身份关系风险？** 这是条件机制研究之前的决策接口诊断，不等于已证明静态评分不足，更不等于已选定多假设/GNN方法。

### 已有证据、推断与反例

| 类型 | 证据与含义 |
|---|---|
| DE：目标困难 | 固定056全V已有raw FP55,652；系统9,353组/610重复组、位置有效442,263/454,952点。它们表明现有完整任务尚有困难，不能单独证明分组次序是原因。详见[接口证据](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/target_interface_evidence_20261029.md)。 |
| DE：运算 | [group_predicted_tracks](../../association/evaluation/grouping.py:29)按首个未分配anchor只取仍未分配的直接接受邻居，不作传递闭包/重新分配。[raw评价](../../association/evaluation/association.py:119)直接对scores≥θ计数，与后续partition分离。 |
| AI：解析性质 | 三节点接受AB、BC、拒绝AC时，A先处理得到AB\|C，B先处理得到ABC。映回原节点后仍不同。是解析例，未运行MTAD诊断，不能判哪种分区正确。若接受图是互不相交clique，此机制不发生。 |
| UN：本设计要测 | 真实V300上发生率、decoded关系损失变化及异质性。原生产顺序是合法固定规则；对重排敏感本身不等于生产方案劣于所有重排，也不等于有更优可实现解。 |
| 反证边界 | 三场103/105极端尾部对应的既有标签组成员一致；不能把C53强行解释为那些极值的主因。它主要针对全V关联/重复/覆盖困难。 |

## 2. 三层结果必须分开

1. **Raw边指标**：固定scores、θ和候选人口，PCA/PFA/F1/AUC严格不随decoder变化。它们是本诊断的零变化控制。PFA=FP/(TP+FP)，不是FPR。
2. **Decoded关系指标**：用最终partition在同一候选对上展开共组0/1，评价PCA/PFA/F1、FP/FN与分区差异。与raw分栏，不能用decoded成绩替换原056/102口径后宣称修好了raw PFA。仓库[双报告先例](../../association/evaluation/tsactfer_level1.py:320)可复用评价方式。
3. **物理/完整目标输出**：新partition须真正重算条件状态；不能重排旧090C/TFM按旧组生成的坐标cache。第一阶段没有reference、完整target registry或新条件状态，因此**不报告全9,100目标覆盖/重复率或mME/米制收益**。输入节点的split/mixed诊断也不能冒称完整目标人口结果。

## 3. 第一阶段事前合同：只改变处理顺序

### 3.1 输入与人口

只读原056已存`association/final-scores.npz`与原summary/run及固定manifest的V_select名单。原运行根：

`/home/wangguangjie/djs/association-code/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1`

Git历史`5776cd3`的产物登记为45,271,415 bytes；原system命令/COMPLETED记录明确绑定final-scores，**不是只推测代码会保存它**。本机没有该NPZ，服务器当前是否完整可读仍UN；执行前只读预检，不自动生成/换缓存。Git版本只是来源定位，不是hash准入门槛。

schema：`scene_ids`及每场`node_ids[N]、sources[N]、pairs[E,2]、scores[E]、labels[E]、joint_support_queries[E]`。全部300场含无pair场景；每场全部合法节点、完整i<j候选均保留，同源正对/无共同query正对不剔除。分组仅接收pairs/scores/N；labels独立交评价，source仅用于同/异源分层。**不得解析node_id中的身份、batch或source编号来选分区/顺序。**

固定阈值 **θ=1.7320103112387182**，`>=`含ties；是未校准关联分数，不是概率，也不是OLS阈值−1.955476…。历史summary的selected/final均为此值，pair_count=4,439,092、TP22,666/FP55,652/FN2,009/TN4,358,765。run中`candidate_count=4,439,093`是阈值搜索的unique score数＋all-reject选项，**不是边人口多一条**，见[select_threshold](../../association/evaluation/association.py:186)。预检须核场景/节点/候选/有限值、summary绑定及原raw逐场计数，不能仅凭总数相等通过。

不读取MTAD ZIP、reference、原生时序或模型权重，不开T_test。现有V已多轮开发且与T共享身份，本分析只解释固定开发人口，不声称独立泛化。

### 3.2 干预与配对控制

- A0：原输入顺序，调用同一`anchor_direct`语义；复算原partition，并逐场与原summary的`final_result.grouping.per_scene.membership`节点组向量规范化后完全一致。旁边名为`partition`的字段是诊断统计字典，不能当成员列表解析；可另核其统计。字段已从历史原件核到，300场须按scene_id/节点序绑定，不能只核总组数或raw计数。若既有其他summary包含的是connected-component敏感性探针，不能拿它充当原anchor partition；原A0关系不符即停止。
- A1–A8：对每场按固定`random.Random(20261029)`、manifest顺序依次产生8个节点排列。固定整个命名图，将pairs两端映入新索引、排序为i<j，scores随对应边移动；节点数/边集合/分数逐值不变。调用同一decoder，再映回原始节点索引。
- 每个partition用原节点索引排序组内成员，再按最小原索引规范排序组；比较**关系集合**，不能只比较组编号。0/1节点和重复排列照样记账，报告实际唯一排列/分区数，不补抽到“有效”。
- 不以labels、reference、旧成绩选择顺序，不取8次最佳为主结果、不调θ。8次是有限算法干预，不是8个训练seed或独立样本。闭包/多假设/新打分器不混入本阶段。
- 原raw逐场混淆计数、raw接受边集合必须完全不变，作为程序控制；不变控制失败即停止，先修实现。图为clique并的场景若存在，可作自然阴性分层；不据此删掉其他场景。

### 3.3 估计对象与分析

**主要结果**：V300中至少一次逆置换后partition不同于A0的场景数/300，并列有≥2节点场景分母、各场不同partition次数/8；量化每次相对A0的共组关系异或数/全部i<j对数（N<2记未定义并保留场景）。这是固定有限人口、固定8排列下的敏感度。

**次要结果**：

- A0及每个排列在相同候选人口上的decoded TP/FP/FN/TN、宏/加权PCA/PFA和F1，按现有未定义分母规则保留null/eligible数；ITSA/MSTSA、逐场/38组件分列。分别给8个配对变化、范围/中位数，不挑最佳、不冒充显著性检验。
- 同一partition内raw拒绝边数、分区间raw接受边数；全部分母、组数、singleton数、mixed/split数量（仅在下述标签条件成立时）及耗时/RSS。保留坏结果；组数少或自洽程度高不等于身份正确。
- labels只评价原候选对。若完整i<j的正标签闭包满足等价关系，才可在**输入节点人口**恢复评价identity组件，核mixed组/拆分身份；同时核每个候选label与组件一致。不能从缺边或不一致图硬推identity；缺这些条件时停止相关派生项并报告，不将“未能计算”当0。全目标registry未开，所以无输入目标不在此项分母内。

不进行独立性不成立的目标级bootstrap或把8排列当独立重复；这是全V开发人口的描述性枚举。组件分层只解释异质性，不补造外推CI。原raw控制应严格零差，decoded变化无数值容差挑选。报告原始整数及对应百分数/百分点，禁止只报均值改善。

### 3.4 资源、终点与停止规则

一次服务器CPU≤4，RAM预算4GiB/虚拟空间≤8GiB，0GPU/模型构造/前向/优化/新normalizer；约45MB已存输入，300×9个decoder，8个重排顺序事前固定。暂估10min仅advisory，记录实际成本，不以超时自动扩量/重试；脚本应流式逐场/逐排列，避免保存9份全E级数组。完成全部300场、原/重排结果与输入验证/真实退出/成本齐备即结束。

原文件缺失、绑定/人口/有限值/原raw计数或原A0逐场partition不符、解码节点重复遗漏或逆置换控制失败：保全真实失败，不改分区方法、换缓存或自动重跑关联前向。无pair/无变化是合法结果，不是失败。不访问reference解决检查失败。现时cache缺口不妨碍研究设计，但未完成实现/预检前不发布READY。

以下只按“是否严格变化”作描述，不设效应晋级阈值或自动后继。先列全部整数和8次配对Δ、正/零/负场景数及分层，再讨论值得投入的幅度；任何后续实验另有事前合同，不能看本轮结果后声称此前已定义“明显改善”。

| 实际结果 | 支持到哪里 | 后续可检验发展 |
|---|---|---|
| 原控制正确，300场全不变 | 关闭“此缓存在指定8排列下发生顺序改变”的命题；非传递解析性质不代表实际有影响 | C53条件状态问题仍可单独研究，不能为保方向补抽有利排列；先看正确候选是否在当前合法空间及组状态是否能区分歧义。 |
| partition变化，全部decoded混淆计数不变 | 支持算法敏感度；在本轮计数层没有身份损失变化，未证明可避免风险/更优可实现方法 | 不把变动本身当改进；保留组结构差异，另论是否有可测的实际目标影响。 |
| partition变化且至少一场/排列decoded混淆计数改变 | 说明决定能改变本人口的关系风险；逐项报告正负/大小，仍未证明原顺序是最差/存在可实现收益/物理精度可改善 | 可据实际幅度讨论静态合法分区、同状态证据静态重评分和条件构造的独立对照；不自动晋级。物理收益另用新组真正前向验证。 |
| 某些重排更坏、mixed增多 | 保留负面结果，说明单纯顺序变化/闭包不是解法 | 发展拒配/组一致性/条件证据，或关闭当前规则；不删难场景、不把大组当好。 |

## 4. 对应解法：把条件建设与外域机制拆开

这是第二阶段的**条件路线**，未承诺具体架构/训练。第一阶段敏感度诊断只为降低一项不确定性，不被写成B0已生效的验证。

| 组别 | 具体改造、保留什么机制 | 价值归属与反例 |
|---|---|---|
| A：成熟静态参照 | MH-T2TA式相容解释/静态score聚合提供领域起点；另比清楚定义的静态全局分组。不是直接把MH代码称已复现 | 原MH同源单轨/局部单射前提与fragment身份域不同；朴素connected components虽然顺序不变，错误bridge会吞并，不能据此选为解法。 |
| A′：建立合法身份空间 | 同身份允许同源多fragment、保留singleton/拒配；身份R、query支撑W和报告provenance分开。signed score `s−θ`可作透明静态目标，但只称surrogate | 先核候选容量/评价究竟是全部身份边还是连通，预拼接/单窗单轨可解除冲突。此处改造收益归A′，不归Gelato或条件网络。 |
| A′+B0：透明条件证据 | 对有限partial group，使用同一原生输入的可解释条件状态/留出原生点一致性重评分，再决定候选加入/拒绝；先给非学习可审版本 | 组拟合好可能是错身份、共同偏差或自我解释；需错误partial group、held-out观测、同K与同F调用数控制。残差不是Cov/后验；不能以reference参与评分。 |
| A′+B1：后续学习版本 | B0的正面结果可支持投入Gelato式partial-decision→更新结构表示→重算动作评分，替换为海上多fragment身份/多正确动作和合法停止；B0不涨时先区分无可辨线索、规则失配与训练/表示条件 | 不套GED单射/编辑成本/安全约简；必须控制训练partial分布与推理条件化，Table4源消融同时改变两者。B0不是学习版本的逻辑必要条件，但负结果不能成为无依据自动开训的理由。 |

**第二阶段必须再拆开状态信息与条件构造，不能只用A′→B0一跳认定纯条件化有效。** 先在同一冻结完整假设库存上，比pair-only评分与使用同一F/native线索的静态完整假设重评分，识别新增状态证据的作用；再用同一观测线索、状态算子、最终cost/输出规则和候选权限，比较静态完整假设搜索与partial条件构造，匹配K与F调用预算。后者允许候选库存因构造而变，应报告实际搜索轨迹/候选覆盖；收益是条件构造/搜索的作用，不能仅凭总成绩拆成纯表示因果。冻结同一候选与条件输出再换selector，另识别输出决定的作用。同K/F只能控制成本，不自动控制信息与机制。上述对照在后续Idea具体化，本阶段不执行。

若静态A′已解决主要困难且B0无增量，可把贡献发展为有效的领域适配/决定层改进；不为包装跨域强加B1。若A′后仍有残余，而B0在相同信息/预算下改善身份与真实输出，可发展为“组条件证据驱动的身份—状态联合决定”，最终用机制、反例、消融说明贡献。**问题已知或已有相似工作不关闭方向，新增价值须真实测到。**

### 迁移A/B/C核查

- **A已知**：上述目标困难及raw/decoded接口；**A未证**：固定score缺少的究竟是条件证据还是评分、时域、数据reference等其他原因。
- **B有源证**：Gelato Fig2/每步重算及Table4；机制是partial choice改变剩余动作的表示/评分。源整体有效，不把其混杂消融当纯推理因果证明。EchoMatch辅助区分是否有对应与归一分配，错误互逆cycle也可自洽。
- **C实际经验**：MHT→T2TA在MH-T2TA已有真实应用；Gelato跨code/molecule有不同收益及跨域失效。**未找到并完整验证“Gelato条件组状态→海上物理T2T融合”的同机制成功C**，如实缺失；不无限搜索，也不虚构先例。
- **需要建立的条件**：合法关系/拒配、原生时域与延迟、同组状态模型、不会把自拟合当独立健康证据、同信息与搜索预算。真实Cov不是第一阶段先决条件；概率版须另建可用不确定性，不能随便造方差。
- **实际效果归因**：A→A′是条件建设；同完整假设库pair-only→静态状态重评分是新增证据；同线索/成本/最终规则下静态→partial构造才核条件使用/搜索；冻结候选换selector核决定。B0→B1还需匹配训练曝光/容量等。缓存诊断不能代替这些比较。

## 5. 三层质量检查及同行比较

新颖性范围：本轮复用9项完整来源，按上述命题定向回核公式/原页/可得代码；另于2026-10-29公开学术网页定向搜索track-to-track/non-overlapping/multiple-hypotheses、tracklet-fusion/Poisson及CVPR/ICCV图分组近邻。检索不是系统性穷尽，也不认证首创。新找到的[Signal Processing2025分布式密度聚类/标签共识](https://www.sciencedirect.com/science/article/pii/S0165168424003232)仅看出版页/摘要，**未作为完整机制证据或第101篇精读**；SUSHI/近期图分组线索同样只作待核近邻。已有成熟解法用于完善本路线和对照，不作为否定理由。

研究价值：正结果能定位“相同边证据→不同身份决定”的实际边界，指导是否投入更合适的决定与条件状态；负结果能关闭一个具体解释，防止不加区分上多假设。纯数学顺序反例、重排最佳分数或漂亮自洽性不是最终创新成绩。

可归因性：固定同一分数、θ、人口、decoder；唯一干预是输入处理序并逆映射。第一阶段不比较新网络/更多信息；后续按A/A′/A′+B0、同K/F和selector独立对照。全V开发身份依赖、未校准score及reference生成未知均保留。

可信同行仍须共同任务协议。TMTAFF是用户指定共同指标参照，107尚不认证复现；MH-T2TA可比较关联，不能冒称融合；MeanTraj/TM-PMBM是高质量融合起点，各自共同时间/轨迹量测等条件需合法适配，不能用失败适配当胜同行。056/090C只作诊断对照，不作为新方法必须沿用的骨架。完整方法须同时报TMTAFF共同关联/四属性口径、米制误差/尾部、覆盖/重复及成本；本阶段只有raw/decoded关系结果。

执行前独立只读复核已补：A0与旧真实分区逐场一致、无事后“明显改善”自动晋级、Gelato/Echo直接来源ID、同完整假设库存的静态状态重评分对照，以及B0失败不能逻辑上否定学习条件表示。修订均在任何实现/数据执行之前；未增加研究结果。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-FROZEN-SCORE-DECISION-20261029
idea_id: IDEA-FROZEN-SCORE-DECISION-20261029
created_at: '2026-10-29'
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.28
selected_C_id: C53
source_P_ids: [P08, P14]
source_F_ids: [F139, F140, F166, F167, F173, F174, F175, F185, F204, F205]
source_M_ids: [M35, M57, M69, M71, M79, M80]
source_E_ids: [E67, E81, E86, E87, E92, E102, E103]
research_question: '同一V300命名图/边分数/阈值下，anchor处理序改变多少最终身份分区与decoded风险'
hypothesis_and_mechanism: '非传递接受图中首anchor消耗邻居可产生不同分区；目标频率/风险未测，raw边指标应严格不变'
testable_predictions: '3.3主要敏感度及decoded变化；raw零变化阴性控制；clique并若存在应不变'
design_and_controls: '原序加8个固定RNG排列，同decoder，边随置换、输出逆映射规范化；不挑最佳，不改θ'
sample_materials_or_experimental_units: '原056 final-scores与summary/run/既定manifest；全部V300含无pair场景、全部i<j；labels仅评价'
primary_outcome_and_estimand: '300场至少一次partition改变数/300，另列eligible分母与各场次数/8、关系异或比例'
secondary_outcomes: 'decoded PCA/PFA/F1和FP/FN分场/组件/ITSA/MSTSA，raw接受跨组/拒绝同组、组数、合法输入人口mixed/split、成本'
analysis_plan_and_uncertainty: '固定开发有限人口描述统计和8次配对变化，不做IID目标CI/训练seed推断；null/全分母/坏结果保留'
resources_time_cost_and_risk: '一次CPU<=4/RAM预算4GiB/AS<=8GiB，约45MB缓存，300x9decoder，暂估10min advisory，0GPU/模型前向/训练'
ethics_safety_privacy_or_compliance: '不读ZIP/reference/test或身份编号特征；只既有授权缓存，输出脱敏统计，107/108不变'
stopping_and_interpretation_rules: '输入/绑定/原raw/逆置换不符真实失败保全不自动重跑；不变关闭指定敏感度命题；变化按3.4解释不认证方法有效'
status: diagnostic
source_transfer_id: TR-C53-CONDITIONAL-GROUP-DECISION-20261029
transfer_condition_commitments: '第一阶段只核决定接口；后续合法A-prime、条件状态、同K/F、无标签选择后才比较B增量'
adaptation_vs_method_attribution: 'A到A-prime与B0/B1收益分开；不将数学例/缓存关系结果称物理融合成绩'
evidence_status: partial
missing_evidence: [current_cache_readability, diagnostic_implementation_and_execution, target_conditional_evidence_increment, new_partition_physical_outputs, credible_peer_comparison]
next_skill: record-experiment_after_actual_execution
```
