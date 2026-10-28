# C53：合法静态分区能否利用原pair证据带来增量

2026-10-29，plan1.0，`IDEA-STATIC-PARTITION-PILOT-20261029`。消费[Mode C1.29 §34](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-129)的单个C53。用户明确授权并行小验证、初步支持后有限服务器继续；本计划把成功条件和最大后继范围在执行前写清。**这是A′静态决定诊断，不是已选创新模型、Gelato或条件状态B的效果。**

## 1. 已知困难与本轮问题

109真实结果：234/300场受顺序影响，但8种随机序的all宏F1都劣于原序，不能把随机化当解法。原序按source/batch排序，可能含有有用优先；具体质量含义未知。A0有20115/48844共组边的原分数被拒绝，切断49589/78318原接受边。110表明fragment全部身份边不能直接套单轨单射，当前anchor本身却不是单射。两路指向**在合法多fragment关系内，是否更好地利用已有完整正负pair证据**，而不是先增加训练/大搜索。

可检验假设：原star只看anchor邻边；当组内其他原评分含有有用反证时，使用完整跨簇证据的合法静态分区，可能在保持论文关联取舍的同时改善decoded身份关系。反例：评分本身错误、弱真边、合法source优先被丢失，可能使闭包/严格clique或signed surrogate更差。低surrogate成本不是身份正确，mixed/split数量少也不是充分目标。

## 2. 输入、方法与解释对照

只读同056/association的final-scores/run/summary及同manifest，真实θ=1.7320103112387182。沿109元数据、完整i<j、逐场all/ITSA/MSTSA原raw与A0原membership绑定；每个实际分析场均核。0ZIP/native/reference/物理cache/权重/模型/新score/normalizer。标签只评价及决定是否扩量，不构造任何分区/顺序/超参。

**A0及这些对照的partition本来都数学合法；本轮“全组一致”指对原评分的约束，不把raw冲突称为身份非法。** 所有方法均允许同源/异源多个fragment，同源和无共同query真边不剔除；每节点恰属于一个组，允许singleton。不假定输入Cov、pD或物理出生/死亡。

| 臂 | 精确规则 | 回答什么及局限 |
|---|---|---|
| A0 | 原input的anchor_direct，原membership逐场完全重现 | 原静态生产对照，保留其合法source/batch顺序，不宣称无信息任意序。 |
| CC | scores≥θ的无向图连通分量 | 顺序不敏感的传递闭包反对照；错误bridge可吞并，不能预称解法。 |
| CL | complete-link凝聚：初始singleton，簇间gain=min所有跨簇(s−θ)，每次合并最大gain，gain≥0才合；合并后对其他簇用min更新 | 强制每个组内原pair都接受；可避免star忽略拒绝边，也会因一个弱真pair过分拆分。不调θ。 |
| WS | signed-score凝聚：初始singleton，簇间gain=sum所有跨簇(s−θ)，每次合并最大gain，gain>0才合；合并后用sum更新 | 透明surrogate `J=Σ组内(s−θ)`的merge-only局部上升，不是posterior/全局最优；正负证据可折中，不保证身份更好。 |
| SRC_REVERSE | 按原sources首次出现顺序的块整体反序，块内原序保留；同一图重索引调用anchor，再逆映射 | 解释来源优先。只用source相等/原块序，不按真实质量选源；不是可晋级候选方法。 |
| SRC_WITHIN | 原source块序保留，每块用`random.Random(20261029+int(scene_id))`依次shuffle原节点；同图anchor后逆映射 | 解释批次/块内先后；一次固定干预，不是随机序总体效应。不是可晋级候选方法。 |

CL/WS的merge gain完全相同时，以簇的最小原节点索引组成的排序二元组、再簇ID作确定性tie-break，记录实际tie次数；不使用node_id内容、label或GT。不声称一般对称图也唯一置换等变。每臂最多N−1次merge，完整pairs输入不改；用min/sum更新和heap避免重算全部簇对。不受旧056网络骨架限制，本轮只是冻结证据的可归因内部比较。

## 3. 固定pilot与有限条件扩量

先从原manifest的V300按数值排序，用`random.Random(20261029).sample(...,32)`，再数值排序形成V32；不看成绩，不替换难例。V已多轮开发，包含109/110已诊断场景，**不是独立验证集**。这32场所有六臂全部计算/保存，raw严格不变，A0原组一致。

先计算全部pilot结果，再检查事前扩量条件。**仅CC/CL/WS可触发后继**；某一臂相对同V32 A0同时满足下列条件才扩量：

1. all场景等权macro-F1至少增加0.005（0.5百分点；本次预算筛查的实用幅度，不是普适创新门槛）；
2. MSTSA场景平均mPCA不降、mPFA不升；
3. all micro PFA不升（控制总体预测边加权误接受比例；组件/场景尾部仍须另报，不能由总体不升保证）；
4. 原raw全计数/人口/A0不变控制全部通过。

数值比较只用1e−12浮点容差，任一所需分母未定义则不满足该臂扩量条件，保留null原因。**不依据SRC对照或8种排序最佳值晋级，不选表现最好分区为模型输出。**

若任一指定方法满足，单次服务自动分析原V中剩余268场，全部六臂都完成，不仅跑“胜者”；原32不重跑，合并形成全300及新增268分别报告。若均不满足，正常结束于32场。最多300场、固定同算法/θ/预算，无调参/新seed/训练自动后继。门槛没有承诺fullV收益，也不把同开发人口扩量称独立泛化。

## 4. 主要结果、归因与完整报告

主要估计对象：pilot各臂相对A0的decoded all宏F1、MSTSA mPCA/mPFA、all micro PFA及整数TP/FP/FN/TN；如果扩量，原V300与剩余268分别同报。关联仍按既有PCA=TP/(TP+FN)、PFA=FP/(TP+FP)、场景与预测数加权/null合同，raw与decoded双列。不能拿binary共组AUC冒充原score AUC。

次要：ITSA/all/MSTSA全共同指标；每场/38组件（pilot仅实际覆盖组件及原分母）；raw拒绝同组/接受跨组、J surrogate、group/singleton/mixed/split、实际merge/tie数、运行时间/RSS。标签一致等价条件满足才记身份结构；mixed/split仅辅助，不独立定优劣。所有坏结果/无变化/空分母保留，不能只汇总触发臂或大赢家。C53 pilot关注关系风险，不报告全target覆盖、位置物理收益或mME；新组物理状态尚未重算。

同分数、θ、人口、候选输入和评价；CL/WS用了不同完整组目标与更多计算，成本必须如实比较，不能称完全等算力。此处没有partial-state B、无新增状态线索，因此改进只能归静态关系/聚合规则，不能归Gelato/条件网络。若后续研究B，仍需同状态证据的静态完整假设重评分、partial构造/selector独立对照。

描述性有限开发集单次比较，逐场/组件差异和分子分母优先；不对pair做IID CI、不把六臂当六次训练重复。多个臂是事前对照，报告全部，不以一个最佳值推断普遍优势。

## 5. 资源、终点和解释

一次服务器CPU≤4、RAM预算4GiB/AS≤8GiB、0GPU/模型/训练；建议1200s advisory，按固定32→条件300终点运行。单服务持久systemd/Restart=no，新run114/attempt1；读取原45MB scores，逐场释放，私有成员/原source/node字符串不回传。异常/OOM/人口或原raw/A0不符：真实失败停、保全不自动换算法/删场/放宽容差重试。

- **无臂过门槛**：正常完成pilot，关闭这些固定静态配方的自动扩量；不宣称全局法/条件表示都无价值，也不为追分续调阈值。
- **静态方法过门槛但扩量不支持**：如实保留选择偏差/异质性，不能挑pilot宣胜；不新增训练。
- **全V也有同向价值**：说明合法静态决定值得更完整物理/可信同行比较，增量归A′；仍不能说解决原raw FP或物理尾部。下一步再设计新partition真实恢复，非自动训练。
- **SRC对照有改变**：帮助定位来源块/批次序条件；单次结果不能认证源质量规律或以来源ID建立普遍模型。

新颖性检查复用已完整分析的MH-T2TA/Wolf/TPMBM/MeanTraj与Gelato迁移边界：分组/凝聚/相关聚类已有成熟基础，本轮是透明可归因控制与合法任务适配，不主张算法首次。若它实质改善本域未很好解决的fragment关系/风险，仍可继续发展新增价值；不能仅因“有人做过”结束，也不能仅靠数学一致性当科研效果。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-STATIC-PARTITION-PILOT-20261029
package_id: IDEA-STATIC-PARTITION-PILOT-20261029
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.29
selected_C_id: C53
source_P_ids: [P08, P14]
source_F_ids: [F204, F205, F206, F207]
source_M_ids: [M35, M79, M80]
source_E_ids: [E102, E103, E104, E105]
research_question: '固定原pair证据，合法静态分区/完整组聚合是否较原star决定有可归因增量'
hypothesis_and_mechanism: '组内负证据被star忽略；complete-link与signed-sum分别检验严格一致/证据折中，来源优先为竞争解释'
testable_predictions: '四个主要关系指标的事前32场门槛；raw/A0不变；所有反面/源块对照保留'
design_and_controls: 'A0/CC/CL/WS及source-reverse/within两解释对照，公式/tie/stop固定，不用label构造partition'
sample_materials_or_experimental_units: '同056只读cache；固定seed纯manifest抽V32，合条件才其余268，最多原V300'
primary_outcome_and_estimand: 'decoded all宏F1、MSTSA mPCA/mPFA、all micro PFA相对A0；pilot/剩余268/全300分别报'
secondary_outcomes: '全scope共同指标、逐场/组件、原拒绝同组/接受跨组、J/group/mixed/split/merge/tie及成本'
analysis_plan_and_uncertainty: '有限开发人口描述性配对，无pair级IID CI；多臂全报，不宣独立泛化/物理效果'
resources_time_cost_and_risk: '一次CPU<=4/RAM4GiB/AS8GiB，1200s advisory，32条件扩至最多300，0模型/训练/GPU'
ethics_safety_privacy_or_compliance: '不读ZIP/reference/test/权重；已有V labels仅评价/扩量，private成员与原标识留服务器'
stopping_and_interpretation_rules: '5节；门槛不满足正常停32，故障停不重试；通过仅有界扩量所有臂，无训练后继'
status: pilot
evidence_status: partial
missing_evidence: [static_partition_actual_increment, physical_outputs_under_new_groups, credible_peer_full_comparison, conditional_state_B_increment]
next_skill: record-experiment_after_actual_execution
```
