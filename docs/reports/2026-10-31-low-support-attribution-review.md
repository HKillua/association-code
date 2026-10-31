# 177：低支持大错的候选空间与对应敏感性

2026-10-31；REC-LOW-SUPPORT-ATTRIBUTION-177 → SQ-LOW-SUPPORT-ATTRIBUTION-177；来源[Idea177/plan1.0](../proposals/decision_operating_point_20261031.md)、Mode C1.83/C53条件与F279/F280。原T32/860对象全部保留，4头WS/STRUCT/HARD/DEC，3个强对照；0V/test/训练/新部署臂。等权位置复放验证动作权限，参考只用于后验误差/纯度/候选上界，不能部署oracle或认证官方reference错误。

## Record：真实执行与边界

科学冻结c003770，独立`association-code-run177`，原run `runs/LEVEL1-LOW-SUPPORT-ATTRIBUTION-20261031-177-attempt-1`、launcher同前缀`-launch-attempt-1`，完整argv/env见[L2S177](../../HandOver/L2S-20261031-177.md)及原request.json。payload07:27:38.282276Z–07:27:56.934952Z；程序12.258121s、launcher18.795680s，RSS432119808B；CPU16–19×4/无GPU/900s advisory。科学unit `mtad-27ec89c00a14.service`真实inactive/dead/Result=success/ExecMainStatus0/Restart=no，Linger=yes。

事前要求supported/all两套候选上界，初版只按1域supported/其他all合并报告，**原输出不完整**。明确补项代码74642a0，新`association-code-audit177`、同前缀`-accounting-launch-attempt-1`，不改原科学；只读全部保存害点，用独立球面距离重算原均值/上界关系与两套候选计数。payload07:32:05.114150Z–07:32:23.165513Z，launcher18.183231s、程序5.351574s/RSS421883904B，CPU16–19×4无GPU，unit `mtad-ef157d7cf6c3.service`同成功终态。两阶段总36.978911s，非新训练或预测；补项时序不倒填。

全部57公共科学/32逐场及两个launcher文件＋1终态服务原件已回传[177证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_decoded_coverage/177)，未下传原坐标/身份/权重/大预测。READY前list/ndarray两处接口验收失败后修复，4项验收通过才提交；无真实科学失败、重试或配方变化。本实验没有新预测指标，不计方法成绩；T已参与模型训练，结果为条件诊断，不是外域确认。

## Record：原分母、全部损害与诊断空间

增害＝相对对应强控误差增加>1km，非绝对误差>1km。原175共同域均值及害点数完全对齐。来源“2”表示至少2；source支持是当前预测组内同query的观测覆盖，不等传感器健康或全局没有观测。

|T候选−对照|共同有效点|新增>1km害|候选混组害|候选纯组害|0域害|1域害|多域害|全当前组片段oracle可去害|兼容输出oracle可去害|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|HARD_vs_WS|37601|289|181|108|9|257|23|176|61|
|HARD_vs_STRUCT|37664|453|305|148|140|195|118|125|239|
|DEC_vs_HARD|37993|387|156|231|167|205|15|101|308|

两个oracle可重叠，不能相加当救回；最小误差依赖T reference，不可用作输出。第一项只看**当前组**已缓存片段状态，并单列仅受支持片段；第二项只看**已发出**且anchor身份与目标兼容的输出，不重新构组/生成状态。后者依赖身份只作对应敏感性，不证明中央协议错误。

|T对照/域|害点|仅受支持片段oracle可去害|全当前组片段oracle可去害|其他兼容输出oracle可去害|
|---|---:|---:|---:|---:|
|HARD_vs_WS/new_zero|9|0|1|7|
|HARD_vs_WS/new_one|257|0|169|37|
|HARD_vs_WS/pure_one|108|0|94|0|
|HARD_vs_STRUCT/new_zero|140|0|0|106|
|HARD_vs_STRUCT/new_one|195|0|71|23|
|HARD_vs_STRUCT/pure_zero|122|0|0|106|
|HARD_vs_STRUCT/pure_one|26|0|2|0|
|DEC_vs_HARD/new_zero|167|0|16|128|
|DEC_vs_HARD/new_one|205|0|75|167|
|DEC_vs_HARD/pure_zero|151|0|0|115|
|DEC_vs_HARD/pure_one|80|0|29|75|

WS/STRUCT/HARD/DEC分别34701/29902/33560/38026个原输出组query处于0/1域；将来源权重改等权复放，位置与原NN **exact**。这是固定组/节点状态/availability下的动作权限证明，不是所有误差已找到因果。风险头在这些时刻没有来源选择空间。

HARD−STRUCT的195个1域增害全部只有1个受支持片段；HARD−WS的257个也全部如此。不能归因于多个受支持片段平均。HARD−STRUCT支持2→0的104点有84害、1→0的111点有49害；0→0仍7害，故支持数下降也不是所有害的充分解释。

HARD−STRUCT纯0域122害：当前组所有既有片段oracle都不能去害，但已发出同目标兼容输出有106点可去害；提示候选跨组/历史覆盖与中央对应敏感性，不能只在当前组重新挑一个已有状态。纯1域26害：全当前组可去2，另24在当前状态候选中仍无这项改善空间。

HARD−WS纯1域108害：仅受支持片段0，但允许看全当前组后可去94。至少存在“已有支持外片段状态被当前读出排除”的条件空间；最佳片段可能来自当前可用源或另一源，未另核来源，不把它直接说成‘另一传感器更可信’。两种强控暴露不同边界，不能择一抹去反例。当前组没有好的缓存状态，也可能通过重新利用组内原生观测估计出更好状态；本次未检验，不能据oracle认定必须跨组。

## Squeeze：机制假设与后续条件

DE支持把困难分成三类可检条件：错组（HARD−STRUCT305/453害候选组混身份）、正确身份组但当前状态候选空间不足、已有支持外/兼容输出可用但当前读出没用。类别可重叠，是后验描述和oracle空间，不是互斥因果认证。

AI候选：当前硬support掩码将观测覆盖当成候选可用性，可能遗漏有价值的历史恢复状态；分组改变使同一目标历史分散到多个输出，也可能加剧对应/回退风险。UN：如何不依赖身份/reference识别可用历史、相对时距/模型误差是否足够、为何原生状态本身偏差、跨组链接误并风险、真实K≥3泛化。此前149全组均衡/最近/尺度失败保留，本结果没有使其自动有效。

下一应沿两种条件读成熟解法后设计：一是组内支持外历史候选的风险估计与受支持状态比较；二是跨片段/输出的可部署一致性与历史链接。需要分别建立信息/置信条件，再比较A/A′/A′+B；不是按reference挑94或106点恢复，也不直接再训同一选源头。HARD覆盖与STRUCT均值强控并存，工作点176独立运行不受177诊断选参影响。

回流Mode C1.84/F281/E184，充分107不增；无自动新恢复器/全300/新seed或扩展训练。

```yaml
package_type: SQUEEZE_REPORT
squeeze_id: SQ-LOW-SUPPORT-ATTRIBUTION-177
input_experiment_record_ids: [REC-LOW-SUPPORT-ATTRIBUTION-177]
claim_scope: original_T32_saved_outputs_all_860_targets_no_V_or_deployment
primary_result_with_uncertainty: 0_1_source_score_invariance_exact; supported_all_fragment_and_compatible_output_bounds_differ_by_strong_control; descriptive_not_independent_CI
process_dynamics: 18.795680_science_plus_18.183231_prespecified_accounting_seconds
mechanism_and_intermediate_findings: support_is_not_reliability_and_history_candidate_exclusion_are_hypotheses_not_causal_certification
errors_heterogeneity_and_boundary_conditions: complete_common_denominators_purity_anchor_support_transitions_and_all_saved_harm_cases
robustness_sensitivity_and_alternative_explanations: independent_distance_recount; diagnostic_oracles_not_deployable; central_correspondence_not_truth_certification
new_F: F281
new_E: E184
new_M: none
next_skill: deep-read_Mode_C
```
