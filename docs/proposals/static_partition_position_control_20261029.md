# C53：静态分区的关系增量能否传到位置与覆盖

2026-10-29；`IDEA-STATIC-PARTITION-POSITION-CONTROL-20261029`，plan1.0。消费[Mode C1.31 §36](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-131)，单个C53，P08/P14/P02、F211/M83/E109（**E109是实验114**）。按discover-innovation Pattern2/5由已观察的静态关系变化检验输出链；只诊断固定ordinary恢复器，不预选新网络或认证同行。

## 1. 问题、机制和反例

114在固定scores/θ下，WS较A0改善decoded身份关系，CL和CC有不同精度/召回取舍；所有分区本来数学合法。**尚未知道位置是否改善。** 新分组会同时改变成员平均、无支撑时的首anchor回退、首anchor身份对应、遗漏和重复。只看可评MAE可能因漏掉难目标变好，不能由F1提升推断物理收益。

假设：在同一冻结片段状态与相同mean_supported规则下，WS身份关系增量可能改善共同有效点位置，同时保留或改善目标覆盖。反例为错组、长时OLS/参考极值、均值失配或仅换首代表；CL可能精度高却漏覆盖，CC可能混组吞目标。负结果仍能区分关系规则与普通恢复规则的适用边界，不否定静态/条件方法整类价值。

## 2. 固定输入与资格

仅114**事前pilot V32**，读取114原run/summary/pilot-summary及每场public/private JSON，校验其完成、固定seed/原θ/原056来源和pilot名单；不得按WS赢家、误差或组件另抽。114全V/268成绩不作本项独立测试。A0/WS/CL/CC四臂全部评价，不重算114分区，不扩量。

源为原056 run `LEVEL1-ASSOCIATION-20261022-056-attempt-1`：

- `association/{run,summary}.json(.gz)`、`final-scores.npz`：沿109固定元数据、原config/final1000/θ=`1.7320103112387182`、完整i<j/raw各scope和原membership绑定；V32逐场node_ids/sources与system fragment完全一致，labels等于fragment身份共属，仅事后核验。
- `system/{run,summary}.json(.gz)`、`predictions/states-{sid}.npz`：原COMPLETED Level1/V/final、canonical_all、0optimizer/no-fusion及原6 ordinary methods；node_ids/identities/sources、target_identity_ids、query_seconds、native_states/support/nearest_seconds/exact_observation、原learned_assoc membership/offsets/flat和ordinary输出。
- `system/system-predictions/states-{sid}.npz`：原`learned_assoc_mean_supported_*`完整artifact。按target_ids与fragment.target_identity_ids绑定，取前T行aligned_truth/valid；保留没有输入的target，不能用unique(node identities)重建registry。每node target/valid必须对应完整target行。
- 原manifest和114 private membership长度、canonical分区、原A0及public decoded计数逐场绑定；私有membership的索引空间必须是同一原node顺序。

`native_states`实际为已存canonical_all Q64的逐片段插值/OLS状态，不是原生CSV点。无ZIP/CSV、原始reference、checkpoint、模型、新score、normalizer或单位重估。**cached_reference_read=true**；cached identity/reference仅评价和资格核验，不构造分区、平均、选择代表或query贡献。

## 3. 唯一干预和A0重放

1. 从114四个固定membership构造groups；成员按原node index排序，组按最小原index排序。每节点恰出现一次，singleton和同源多段保留。代表固定members[0]，不得按身份/误差重新选。
2. 对每臂真正调用`ordinary_fusion(native_states, nearest_seconds, support, groups)`，取`predictions['mean_supported'][members[0]]`作为每组状态。支撑内按实际成员mask平均，所有成员无支撑时保留该首anchor OLS回退；经度零合向量的anchor回退、圆周经度均值等生产规则原样保留。**绝不拼旧group坐标充作新组输出。**
3. A0先与原fragment `learned_assoc_mean_supported`逐值重放，原groups/membership相同；调用`evaluate_full_scene(..., grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True)`重放完整artifact、原population和full_target_metrics。只有通过才接受候选评价。
4. 数组沿既有`_same`：shape/value exact，浮点NaN-aware，rtol=0；原摘要计数exact，数值沿`assert_metrics`固定rel_tol=`1e-10`/abs_tol=`1e-9`。失败停止，不放宽或自适应选择容差。

全部分区/物理状态先冻结，再传身份和reference评价。使用4状态容器兼容生产接口，但**只报告位置lat/lon共同有效的米制量**，不要求速度/航向同时valid，不解释旧速度单位。球面半径6371000m；不把经纬度直接当米、也不改均值为另一个融合器。

## 4. 终点、分母和可归因报告

主要估计对象：四臂各自完整target population的position交点MAE/RMSE/P95/P99、point/trajectory coverage及整数分母；每候选**分别A0×WS、A0×CL、A0×CC**共同有效点位置误差/有益有害点数/差值。不是四臂共同交集，不把较差CC缩减WS共同集。

每场/实际覆盖组件/V32汇总全部报告：target、reference不可用、group、duplicate、anchor matched/unmatched目标，position available_target/prediction/intersection/missing/prediction_without_target；未匹配目标保留完整分母，duplicate附加行没有新真值。`all_predicted_groups`逐组误差重复目标，不能当主平均。

辅助解释：pairwise gained/lost点及目标覆盖、same/changed首anchor、固定target原输入support并集/nearest最小值的inside/outside/far>1800s/no_input分层；这些身份只用于评价。实际新group支撑作为另外的动态描述，不能混成固定人口分层。比较时保留共同valid与各自valid、全分母和所有不利结果，不仅报首代表未变或最大收益组。

point pooled MAE/均方按交点数合并，RMSE从合并MSE开方，P95/P99从实际可评距离以linear分位数汇总，不能平均场景RMSE/分位数。paired距离差沿既有比较合同以±1e−9m区分better/worse/tie，仅作解释计数，不删点或改变均值。另报场景等权MAE和有定义场景数；组件是原shared-A映射，不是独立实验重复。有限多次开发V32的单次确定性干预，无pair级IID CI/p值/独立泛化/可信同行结论；原reference生成及极端尾部仍未知。

## 5. 资源、失败和解释出口

一次CPU≤4、RAM预算4GiB/AS≤8GiB、600s advisory；逐场处理固定32，0GPU/模型构造/前向/训练/新normalizer。建议时长不是已测成本，实际elapsed/RSS/affinity记录；无自动扩量/重试。原056和114输入只读，新115目录拒绝覆盖。原成员身份/坐标/artifact留private，公共输出仅计数、误差和匿名scene/component。

缺输入/metadata或node顺序不符、原raw/A0/registry/reference不一致、分区漏重节点、非有限/非法位置或资源异常：保全FAILED和已完成部分，停止，不换场/方法/容差/数据。零共同集保留null，正常完成，不筛掉该场。

- WS共同点改善且覆盖无损：支持固定ordinary fuser下的局部位置增量；仍需看尾部/组件和成熟同行，增量不归条件B或学习融合。
- 自有集MAE改善而共同点/覆盖不支持：不能称净位置收益，报告选择/代表/覆盖取舍。
- 共同点变差但覆盖增加，或均值改善而尾部变差：保留多目标取舍，不单指标宣胜。
- 位置无增量：保留114身份增量；后续区分普通均值、query贡献和参考问题，另定合同，不自动训练。

新颖性沿既读MH-T2TA/Wolf/TPMBM/MeanTraj/Gelato和114定位：本项是已知静态机制的任务适配诊断与物理链控制，不主张聚类/均值首次；若有实质价值可继续发展，不因已有先例关闭。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-STATIC-PARTITION-POSITION-CONTROL-20261029
package_id: IDEA-STATIC-PARTITION-POSITION-CONTROL-20261029
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.31
selected_C_id: C53
source_P_ids: [P08, P14, P02]
source_F_ids: [F211]
source_M_ids: [M83]
source_E_ids: [E109]
research_question: 固定普通恢复器下，114静态分区身份增量是否带来位置及覆盖净收益
hypothesis_and_mechanism: 分组改变成员均值和首代表；需同人口位置与覆盖排除漏难目标/换代表假收益
testable_predictions: 各自完整target及分别A0共同valid的米制误差、coverage和首代表分层；不预填方向
design_and_controls: 仅114原pilot32；A0/WS/CL/CC四臂，A0先exact数组与固定摘要容差重放；新组真正ordinary_fusion
sample_materials_or_experimental_units: 原056片段/完整target缓存和114固定private membership；所有V32输入/目标及缺失/重复保留
primary_outcome_and_estimand: position各自valid及pairwise common-valid的MAE/RMSE/P95/P99；完整point/trajectory coverage
secondary_outcomes: duplicates/unmatched/prediction_without_target、gained/lost及same/changed anchor、固定支撑/远时分层和组件/成本
analysis_plan_and_uncertainty: 固定开发人口确定性配对；不做IID显著性或独立泛化；pooled MSE开方而非平均RMSE
resources_time_cost_and_risk: 一次CPU<=4/RAM4GiB/AS8GiB/600s advisory，固定32终点；0模型/GPU/训练/ZIP/新normalizer
ethics_safety_privacy_or_compliance: cached_reference_read=true；CSV/ZIPfalse；身份/reference仅事后评价；private身份坐标不回传
stopping_and_interpretation_rules: 原输入/A0/绑定或资源失败停止保全；null保留；第5节结果分支，无扩量/训练/重试
status: diagnostic
evidence_status: partial
missing_evidence: [new_partition_actual_position_and_coverage, credible_peer_comparison, reference_generation, conditional_B_increment]
next_skill: record-experiment_after_actual_execution
```
