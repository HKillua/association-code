# 163：分组输入与参考对应分离 — Record→Squeeze

2026-10-31。原T32/V32的oracle标签诊断与后补独立人口/参考核验均一次exit0。中央发现是“同一对象的多源选优”和“混入不同对象后的选择/参考对应”必须分开；本实验**没有新可部署方法分数**，不声称官方参考错误或真值净化收益能实际获得。

## Record

EXPERIMENT_RECORD `LEVEL1-GROUP-REFERENCE-20261031-163-attempt-1`，source_idea_id `IDEA-GROUP-REFERENCE-COUNTERFACTUAL-20261031`、plan1.0 / research-pipeline-contract v1.0，本聊天计算诊断。来源Mode C1.70/SR-C1/F256/E159及对应边界；[事前Idea/spec](../proposals/rare_reversal_separation_20261031.md)、[L2S与后补核验说明](../../HandOver/L2S-20261031-163.md)。问题/假设：159少数错是否在固定输入下依赖参考对象，及固定参考下受不同对象成员影响？两轴独立，不同时替换输入与目标。

冻结3d5191a、新checkout163/science/launcher attempt1。输入原127 native/125 value/114 WS/133 source包、159冻结旧10维网络/155二次头及manifest，完整绝对路径与实际argv见[run与原request](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_rare_reversal_separation/163/)。T32/12组件、V32/9组件；原官方D2 T22627/V39847，0新fit/teacher/rawCSV/test、0新正式物理预测。全部source状态/分数/可用性和first-anchor参考关系原样保留为对照。组×query人口含重复组，不等同71071去重复主评价点。

轴R固定WS源状态/得分，枚举registry存在的全部组成员目标reference，在原D2且双方reference有效处计算最佳源集合/选择excess。保留全部候选缺失/有效与所有逐query数组，不用“最好看的”替代参考更换官方评价。轴G固定原first-anchor参考，仅oracle保留与anchor同身份节点，重算源聚合/10特征并用同冻结头。纯组特征恒等后复用159原分数；混组CPUforward，另核原CPU/GPU选择0差。**净化后不足两个源的query不是可比排序域，不能称救回/正确。** 本实验使用truth只作明确oracle诊断，不能喂训练/运行策略或认证真实传感器健康。

北京时间02:08:52.754–02:09:10.128，程序17.373218s/launcher21.180402s、RSS499441664B；CPU0–3×4/RSS6GiB/GPU不可见/1800s advisory。原unit `mtad-d95acda0eec3.service`成功inactive/dead/ExecMainStatus0/Restart=no/Linger=yes，无科学重试。原件包含完整64逐场gzip/所有组件/去最大/实际命令与退出；原坐标、身份、全部每参考错误/掩码及净化状态私有NPZ留server。

| 轴G 固定官方参考，组×query | 原D2 | 净化仍多源 | 仅1源 | 0源 |
|---|---|---|---|---|
| T all | 22627 | 19566 | 2848 | 213 |
| T pure | 19108 | 19108 | 0 | 0 |
| T mixed | 3519 | 458 | 2848 | 213 |
| T rare | 154 | 57 | 19 | 78 |
| V all | 39847 | 25710 | 13612 | 525 |
| V pure | 25153 | 25153 | 0 | 0 |
| V mixed | 14694 | 557 | 13612 | 525 |
| V rare | 290 | 16 | 105 | 169 |

## Squeeze：不能把丢失多源人口当作救回

SQUEEZE_REPORT `SQ-GROUP-REFERENCE-20261031-163`，来源上列Record，中央人口/原选择错误/全部替代几何已独立核验；净化后feature/forward与选中状态误差只由科学链得到，未称独立完整复现。

| 轴R 固定状态/分数 | 原query | 有有效替代参考 | 最佳集合翻转 | 原network错 | 原错且某替代参考下正确 |
|---|---|---|---|---|---|
| T all | 22627 | 3428 | 1694 | 156 | 105 |
| T rare | 154 | 110 | 103 | 154 | 103 |
| V all | 39847 | 14555 | 4728 | 287 | 238 |
| V rare | 290 | 274 | 244 | 278 | 232 |

轴R为固定预测的标签敏感性：V有4728个query在候选参考间最佳集合不同，罕见层244/290；原278个罕见network错误中232在至少一种替代参考下可变正确。这只说明多对象组内“更接近谁”改变排序问题，**不说明官方reference错、替代reference对或232点被实际方法救回**。不同候选的存在域也不同，未覆盖query未被删除；T候选query对4126/V17076、缺registry候选0，完整逐候选有效掩码保留。

| 仅净化后仍D2 | 共同query | network原错→净化错 | 救回/新错 | 选中状态误差合计变化/m | 新增>1km |
|---|---|---|---|---|---|
| T all | 19566 | 58→44 | 14/0 | -76469.046357 | 0 |
| T rare | 57 | 57→44 | 13/0 | -28467.542629 | 0 |
| V all | 25710 | 16→16 | 0/0 | -9385.496730 | 0 |
| V rare | 16 | 16→16 | 0/0 | 0.000000 | 0 |

V290个匿名反转＝纯16＋混274。净化后混274全部掉到0/1源（105/169），剩16个纯组仍16错；因而不能说“分组修好了274个难例”。T154＝纯44＋混110，仍D2的57中原57错→44错、13救回；同时丢1/0支持19/78。全共同域T58→44/14救回，V16→16/0救回；这些仍是truth净化条件结果，不是完整coverage/尾部/真实可用算法效果。全V557个混组共同点本来network都选对，状态改变使选中误差合计下降9385.497m，不能将其误报为源排序救回。

二次头共同域对照T312→295、23救回/6新错；V155→146、11/2且2个新增>1km。可见净化并非对所有冻结评分头无害。原network完整官方错误T156＝纯44/混112，V287＝纯16/混271，仍与159绑定；未把数据构组解释写成唯一根因，也未抹去纯组反例。

去最大组件仍有参考翻转T614/V248，rare原T88/V48，净化仍多源T19/V15、仅一T4/V3、零T65/V30。现象不只来自最大组件；但数量/组/参考均来自同批开发缓存，非新场景独立复验。所有组件/全64场保留；两个轴改变了诊断目标或输入人口，故不输出oracle“全系统MAE提升”或拼159/其他方法冠军。原官方分数/所有输出覆盖保持为原证据，不修改151/152。

内部双球面公式最大差T3.485e−9/V3.642e−9m；纯组特征核484/595组、原CPU选择变化0/0。科学后为中央结论另加一次独立audit，明确是**后补核验非事前科学设计**：冻结e5ed0c4/new audit163 checkout、CPU0–3×4/600s advisory，原输入native/成员身份/source缓存及159分数独立重建原人口/反转/错误、全部替代reference距离/翻转与truth过滤后的原生支持计数。1975795 exact元素通过，几何最大差3.278e−9m；程序3.821186s/RSS83230720B、launcher4.263749s，真实COMPLETED/exit0/成功服务终态。见[independent-population-audit.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_rare_reversal_separation/163/independent-population-audit.json)及完整launcher原件。**没有独立重建净化后特征/NN输出/选中状态误差，没有认证identity/reference物理truth或部署算法**。

偏差/限制：原science仅内部检查；中央结论的独立audit是在看结果后追加并如实单列，没有倒填原计划。软件混组翻转/纯组恒等/失去D2不算救回及真实159→163链已通过，fixture缺字段已在提交前修正，不作真实成绩。单次T/V开发、K2、原registry/first-anchor协议和输入状态生成仍是条件；没有改变官方评价/删难例/访问封存测试/新训练。

DE支持需把候选是否属于同一对象与同对象的风险排序分开；AI路径是组/对应误差可能使通用风险头承担了对象判别任务，单纯换loss或训练质量不能自动解决。剩余纯组T44/V16仍错，真实质量可辨信息/局部状态/监督条件仍UN。已有MeanTraj原p6 Eq17–19在完整集合目标中联合处理匹配与状态/数量，成熟先例不否定MTAD适配价值，但也不能称国际上没人研究关联—融合耦合。下一轮基于C53×SR-C1形成**truth-free的候选对象/条件状态对照**，同时保留纯组难例路线；必须在原全人口评价覆盖/重复/关联与物理尾部，不能先筛纯组再宣提升。本轮未直接发布后继实验。

```text
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-GROUP-REFERENCE-20261031-163
input_experiment_record_ids: [LEVEL1-GROUP-REFERENCE-20261031-163-attempt-1]
claim_scope: fixed_WS_T32_V32_oracle_R_and_G_separate_not_deployable_score
primary_result_with_uncertainty: V_rare290_clean_multi16_one105_zero169_all_remaining16_still_wrong
process_dynamics: 0fit_17.373s_science_postscience_population_audit3.821s
mechanism_and_intermediate_findings: AI_object_group_reference_conditions_couple_to_source_ranking
errors_heterogeneity_and_boundary_conditions: pure44T_16V_remain_wrong_lostD2_never_rescue
robustness_sensitivity_and_alternative_explanations: all_components_drop_largest_reference_masks_CPUchoices_and_independent_population_geometry
new_P: none_existing_P05_P08_P11_P14_conditions
new_F: F260
new_M: none_M69_M108_conditions_only
evidence_index: all64_scene_summaries_all_components_private_all_reference_arrays_launcher_and_independent_audit
data_quality_and_provenance_gaps: truth_not_certified_purified_forward_not_independently_rebuilt_K2_development_domain
open_questions: truth_free_object_conditions_residual_pure_errors_full_risk
next_routes: deep_read_Mode_C1.71_then_discover_innovation_C53_x_SRC1
next_skill: deep-read Mode C
```
