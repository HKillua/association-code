# C144 · 保持双船中心位置的相对位移约束

`IDEA_PACKAGE IDEA-OFFLINE-PAIR-DISPLACEMENT-CONSTRAINT-20261114` v0.1，待资格核查。消费 Mode C2.59／M201／F446–F448／E566、E567、E570–E576。 [证据矩阵](</Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-259>)

**DE。** C142原等权在5,580个至少一端启用、发布参考间距不超过1公里的两船对×Q上，交换代价比直配低449次。去最大组件仍174／2,896，原人口在250米内有305／1,497次交换优于直配。这些参考分层是事后定位。C142同锚相对残差使该数升至1,314，其中944次由原等权直配转为交换。两个相邻组各自去均值会同时移动两端，绝对P99和双船有向误差均退化。C125已核一个已知反转场。单源原生两船方向有一定覆盖，未找到不复用CAP128的见证，原生方向未恢复两次已知反转。 [C142复算](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261114_fusion_source_offset_motion/public-recomputation.json>) · [原等权事后近距分层](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261114_fusion_source_offset_motion/posthoc-near-gap-strata.json>) · [C125](../reports/2026-11-13-tmtaff-native-relative-witness-review.md)

**AI。** 同一来源在近共时记录两条船时，共同平移可在两船向量相减时消去。若两源都能提供不复用行的同向证据，并且该方向在第二个原生块可复核，则可试着只修正两个固定组的相对位移，保留两船中心的绝对位置。与C119按近邻风险挑位置限幅不同，这次直接改两船状态的差向量；与C142不同，两个组用同一个相对位移约束。各船独立测量偏差、时钟差、机动、近距匹配候选漏掉真实邻船和旧组错配都可产生反例。 [C119](../reports/2026-11-12-tmtaff-pair-relative-holdout-review.md) · [2026原文](</Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c141_c142_sources_20261114.md>)

## 资格先行

仅用 MTAD 原177场／4组件、C121 REG_MIX、C140原EQUAL及同一canonical_all量测。预先在共同原生支撑的组对上固定候选并集，一条来自原等权预测的近距候选，一条来自任一来源原生近共时的近距候选。范围、时间差、独有原生块及短弧坐标判据在执行前冻结。实际两组不得共用成员或同一原生位置行。每个Q仅保留不共享组的配对，按固定可见距离破冲突并记录弃选率。记录仅一源、两源矛盾、双方近时、两个原生块能各自核方向的场与组件。左右原生块须各自满足两船两源共时，端点和括点不互借。共同方向的检查只认量测，发布参考后接。报文不复用不能证明同源加工、母AIS或测量噪声独立。

门槛先检查至少两个组件有合法组对，去最大组件后仍有至少两个场和有效配对；若绝大多数潜在触发只在单一组件或两侧源方向无法互证，结束这一约束。进一步进入对照时，使用完全相同的冻结候选对、原生行、权重、时间窗与组首，保留未通过资格的全部场和Q。源顺序、船对交换及共同平移作为软件检查。只用 C151 发布缓存判真实近距及误差，不能以真近距反选作用域或门槛。

## 最小对照与去留

- 原EQUAL为完整对照。两条新臂使用同一双船状态参数化，在同一米制平面固定中心 `c = (p₁ + p₂)/2`，以 `d = p₂ − p₁` 还原 `c − d/2` 与 `c + d/2`。工作目标为 `||d − d_EQUAL||² + λρ(||d − d_native||)`。控制臂取 `λ = 0`，须重现原EQUAL；试验臂用预先固定的单个λ和径向Huber尺度。两臂共享启用域。见证只定义一次约束，原EQUAL已使用相同原生观测，解释为改变估计目标，不能当成额外独立测量。使用另一块报文的相对向量只做复核。
- 同口径报告四属性覆盖、位置全域和共同启用 P99、超过250米的逐点增害、真近距双船的有向误差和交换代价。另核候选召回、两船旧质心约束的数值误差、每场、组件和去最大，以及一船改善但另一船退化的计数。真缺口保持原等权回退，作为输出一致性检查。
- 两船启用域有向误差和交换数均须下降，新增交换不能多于恢复数，全域与启用域位置P99不升、旧非尾大增害不增、覆盖不减；去最大组件仍须看到方向改进。若只在用同一报文拟合的向量上变好，或近距样本太少、源间证据矛盾、绝对尾部恶化，停止该约束。

当前为候选，先做匿名原生资格清点再决定是否生成新预测。零训练，封存场保持关闭。相对证据如只适合解释错误，也保留其实际可用率与失败边界。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-OFFLINE-PAIR-DISPLACEMENT-CONSTRAINT-20261114
plan_version: '0.1'
source_matrix_id: MODE-C-OFFLINE-NATIVE-CUT-ANCHORED-MOTION-20261114
selected_C_id: C144
source_P_ids: [P02, P04, P05, P14]
source_F_ids: [F446, F447, F448, F449]
source_M_ids: [M201]
source_E_ids: [E566, E567, E570, E571, E572, E573, E574, E575, E576, E577]
research_question: Can source-local displacement preserve close-vessel identity geometry with the absolute pair midpoint fixed?
hypothesis_and_mechanism: Common source translation cancels in a same-source target-pair vector, under compatible time and registration.
testable_predictions: Fewer swaps and lower directed error on the same active close pairs, with no increase in position tail or large harm.
design_and_controls: Original EQUAL, identical paired geometry with zero weight, and one predeclared positive weight. Use one disjoint block to validate the witness.
sample_materials_or_experimental_units: Original MTAD 177 scenes and four components; unique group pairs at shared Q, original groups and measurements.
primary_outcome_and_estimand: Paired near directed displacement error and swap count, joint with full and active position P99 and non-tail harm.
secondary_outcomes: Coverage, pairing eligibility, new and recovered swaps, midpoint preservation, and both vessel-specific errors.
analysis_plan_and_uncertainty: All scenes and leave-largest summary. Check by component and common-native availability before reading evaluation.
resources_time_cost_and_risk: One bounded CPU qualification followed by a zero-training same-information contrast only if qualified.
ethics_safety_privacy_or_compliance: MTAD only. Native coordinates, labels and trajectories stay on server; reference after decision freeze.
stopping_and_interpretation_rules: Stop for no cross-component native witnesses or any failure of joint near-geometry, tail, harm and coverage gates.
status: diagnostic
next_skill: record-experiment
```
