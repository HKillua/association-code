# C143 · 用成对原生指派核验逐时同船供给

`IDEA_PACKAGE IDEA-OFFLINE-PAIRED-SUPPLIER-CERTIFICATE-20261114` v0.1，待资格核查。消费 Mode C2.59／C143／M202、P02/P05/P14、F450–F454、E566/E567/E570/E578–E583。 [证据矩阵](</Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-259>)

**DE。** C141去掉最大组件后，三匹配臂各有1,291个唯一新增分区。旧分数／仅重叠／持续分离分别有79／80／80块存在事后安全且正旧J的候选，取消正J要求仍各有零块能同时过原严格供给保护。原证书要求切开的旧共组对具有相同Q支持。一个旧组内只要出现两种Q掩码，掩码不同的节点形成连通的禁切图，整组随之锁定。去最大后3,577／3,580条旧跨源假边处在锁定组，5,044条漏边中4,239条仍在现有作用域。先固定旧池和旧J，试验对象是逐Q供给保护的条件。 [C141](../reports/2026-11-14-association-native-separation-review.md)

**DE 与近邻。** C101以单条跨源边的绝对距离、运动和双侧竞争找替代供给，严格原生门未放行新增安全候选。C104以候选外共同平移反证旧供给，在少量同窗锚可用人口中实现去假；C107的新留场零动作，C113的唯一双减动作虽增加同船支持，新增源用于位置的38个Q却全增害。C82中旧056对可新增跨源边的AUC .906，高于绝对距离 .894 和相对运动 .898。2026 GMT在共同可见邻轨加入相对关系，视觉跨视图IDF1提高，去距离门时下降。DualTTrack的连续两帧资格用于潜轨确认，长程恢复另有消融。C125中单源原生方向对两次已知反转均未给出正确方向。 [C101](../reports/2026-11-12-association-native-supplier-replacement-review.md) · [C104](../reports/2026-11-12-association-registered-supplier-exclusion-review.md) · [C107](../reports/2026-11-12-association-registered-holdout-fusion-review.md) · [C113](../reports/2026-11-12-association-two-anchor-time-validation-review.md) · [C82](../reports/2026-11-12-disjoint-motion-evidence-review.md) · [C125](../reports/2026-11-13-tmtaff-native-relative-witness-review.md) · [2026原文](</Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/offline_c141_c142_sources_20261114.md>)

**AI。** 若两个来源各自同时看见两条可区分的轨迹，源内作两轨的有向位移可以消去该源共同平移。四条候选跨源边在同一时刻形成两个完整指派。比较两个源的有向向量 `d_A` 与 `d_B`，交换与直配的向量残差平方之差为 `||d_A+d_B||² − ||d_A−d_B||² = 4 d_A·d_B`。不同原生时段若重复偏向同一指派，且对时距及定位误差预算留有余量，可能给逐Q供给提供另一种经验见证。近似平行航行、四轨中含重复轨、目标特异偏差、旋转和时钟差都能制造假见证；互选或两次同号单独不构成身份认证。新增价值是在同候选池和同旧J下，以两个完整配对的比较核候选改动和实际供给，区别于逐边裸距离、外锚平移和单纯更换分数。C144独立检验固定身份下位置是否改善。

## 资格和最小对照

1. 只用 MTAD 拟合侧原1,748场／76组件。固定C141所有已生成的候选及主比较使用的匹配池，首个试验池用 C139 加匹配旧分数候选，另外两个匹配池只查可迁移覆盖。固定原WS、首组列表、原056 J、每块选择上限、原生输入和原Q64。只查保原首且正J、被原证书拦截的候选，并保留全部基准场。候选要求两源各两条不同片段，分区在此四轨上可投影为直配和交换。对两个来源各自的可分离性、四轨原生同窗、全部必要括点是否重叠单独计数。完整候选可包含额外边，对证据没覆盖的改动继续执行原保护。
2. 对相同四轨和候选，先用不复用原生行与括点的两个时段计算实际四轨可比性，两个时段都要求两源两轨的共同时间。共享原生区间的多个Q去重计资格。原C141分离包络作为可复现的源内区分对照，同源分离本身只确认两轨有区分余量。四轨都在各自时段内作合法同秒插值。只在双方源内向量及两个时段对同一完整指派一致，并在冻结的最大时距、最大括距和位置扰动下保有方向余量时形成工作证据。数值在任何新参考或身份读入前冻结。按组对、组件清点独立见证、误判、弃权与可覆盖的受影响槽。C104外锚规则若作为对照，直接复用原合同，缺外锚维持弃权。
3. 同一候选保持原组首及旧J。原严格门与原C104旧供给反证分别作对照。在C104规则保持一致时，用相同原生资格比较有向见证和只按原056排序的覆盖对照。对任一旧WS供给即将被切的Q，若接收节点自身在Q有原生支持，或原证书通过，该槽已有保护。其他槽需有至少一个仍在新组、该Q确有原生量测且获候选配对见证的供给，或所有可能失去的旧供给均有原C104反证。一个候选的未覆盖槽仍挡住该候选。全块同时评价新增和撤出边、旧首、重复及同船支持。资格与候选切换先盲冻结，缓存身份仅作事后核验。原056排序与有向余量在同场相同证据覆盖和可比候选数内对齐，单列无动作；C104另报全域条件差。所有准入后的整分区沿用旧J，禁止从身份选参或按真值挑切口。

## 去留

先查去最大后至少两个组件有合法且完整覆盖相关槽的候选，并评估双块见证能否区分四轨真实配对及被撤真供给，错误率和覆盖同时发布。资格通过才跑同池零训练的最小对照，要求去最大后净增加受保护双减块、至少两个组件有增量，四轴无净退化，原代表、重复和原同船节点×Q毛损失保持保护线，任何误切真供给在报告中完整保留。候选在新留场复核前只作开发证据。四属性传播单列原供给可用性及新增异船影响，使用同一读出作配对复算；C113的正身份、负位置反例纳入去留，融合升级需独立满足四属性和位置尾部要求。

如果去最大组件缺足够四轨见证、证据只确认单条边而覆盖不了完整动作，或同资格下不能更准确区分被撤真供给，结束此配方。当前只提出资格试验，无新运行、拟合、候选生成或封存场访问。效果归因以 C141 固定池和原旧J为基线。后续若有足够实测容量，才考虑跨C139块扩大作用域或学习评分。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-OFFLINE-PAIRED-SUPPLIER-CERTIFICATE-20261114
plan_version: '0.1'
source_matrix_id: MODE-C-OFFLINE-NATIVE-CUT-ANCHORED-MOTION-20261114
selected_C_id: C143
source_P_ids: [P02, P05, P14]
source_F_ids: [F450, F451, F452, F453, F454]
source_M_ids: [M202]
source_E_ids: [E566, E567, E570, E578, E579, E580, E581, E582, E583]
research_question: Can repeated two-by-two native assignments protect prior true support in the frozen C141 partition pool?
hypothesis_and_mechanism: Within-source vessel-pair differences cancel source translation and can distinguish competing complete assignments under repeated native support.
testable_predictions: Equal-eligibility evidence improves true supplier discrimination and adds protected FN and FP reductions beyond the largest component.
design_and_controls: Fixed C141 pools, old J and heads. Compare the strict certificate, frozen C104 exclusion, and old-056 versus paired evidence under the same native eligibility.
sample_materials_or_experimental_units: Original 1748 MTAD fit scenes and 76 components, candidate partitions, unique four-track native intervals and affected node-query slots.
primary_outcome_and_estimand: Coverage of complete candidate edits before evaluation and then protected joint cross-source FN and FP improvement.
secondary_outcomes: True support removed or restored, incorrect supplier changes, same-source errors, representative coverage, duplicates and four-attribute propagation.
analysis_plan_and_uncertainty: Freeze inputs and rules before label join, report per component and leave largest out, and separate descriptive development evidence from future holdout evidence.
resources_time_cost_and_risk: Bounded CPU qualification, then a no-training contrast only if the raw evidence qualifies.
ethics_safety_privacy_or_compliance: MTAD only, sealed test closed, native reports and identities stay on the server.
stopping_and_interpretation_rules: Stop if independent usable witnesses do not cover at least two remaining components or if complete partitions do not meet the joint protection conditions.
status: diagnostic
next_skill: record-experiment
```
