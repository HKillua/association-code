# 166：集合质量上下文与同容量控制 — Record→Squeeze

2026-10-31。26个真实GPU fit/3328更新、限定输出独审/方向及四属性补项一次完成exit0。CONTEXT没有超OWN或冻结159，纯组T44/V16错误均未救回，当前配方不扩大；保留原小网络平均候选，不能据此断言所有来源上下文无效或物理信息绝对不足。

## Record

EXPERIMENT_RECORD `LEVEL1-SET-CONTEXT-RISK-20261031-166-attempt-1`，source_idea_id `IDEA-SET-CONDITIONED-RISK-RESIDUAL-20261031`、plan1.0 / research-pipeline-contract v1.0，本聊天计算pilot；Mode C1.72/SR-C1/P11/F256/F259/F260/M108。问题AI：逐源h(x)未直接读其他源质量，集合条件交互能否减少剩余纯组错误？[事前Idea/spec](../proposals/object_conditions_context_20261031.md)、[L2S166](../../HandOver/L2S-20261031-166.md)写明同容量/预算控制、竞争解释与停止条件。

冻结d379bba/new checkout166/science/launcher attempt1。复用133/155 T128、28组件/166058原监督源标签与159冻结旧10维网络；原T32的12组件留出/full共13fold，全部使用对应outerfold原159 mean/std/clip±8，不重新拟合normalizer。T128完整组/原label权，没筛pure/rare训练；0新teacher/rawCSV/分组/test，V和held组件不用于目标/权重/步数。

OWN、CONTEXT均为新增20→32→32→1、1761参数共享残差，末层零初始，基底为冻结159（另1441神经参数＋66二次系数，不称整个系统参数）。OWN输入[自身z,自身z]，CONTEXT输入[自身z,其他可用源z的均值]，无其他源时填0；全源换序等变的软件性质已核K3/缺源，真实仍K2，不认领独立健康信息。两新头fresh seed20同初始化/每步同4096源标签索引，128更新/原权MSE/Adam.001/wd1e−4/clip1，无早停/追加/恢复。26fit/3328更新/13631488源呈现，OWN为额外容量/曝光A′，CONTEXT−OWN才是条件结构的主归因比较。原159参数、物理状态、WS分组、0/1支持域/速度航向不动，全模型在V前冻结，预测先存后评价。

原T32/12组件37635/38224位置点、860目标/911组/86重复/35漏；V32/9组件71071/72196、1427目标/1605组/217重复/39漏。D2组×query T22627/V39847与主位置人口分开。匿名角色及纯混组只在评价分层；原冻结159物理输出exact绑定。新头仍是带T-reference监督的条件风险回归，不称物理可信概率/协方差。

北京时间11:03:52.218–11:06:50.076，程序177.858498s/launcher192.325988s，RSS1325379584B/CUDA88601600B；GPU1兼容共卡/CPU8–11×4/RSS8GiB/1800s advisory。原unit `mtad-69486007f794.service`及审计/补项真实inactive/dead/success/ExecMainStatus0/Restart=no/Linger=yes。[166完整脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_object_conditions_context/166/)保存全部输入/argv/env、退出、曲线/人口、64逐场及所有组件；原权重/坐标/身份/大预测留server。

| 位置/m | MAE | RMSE | P95 | P99 |
|---|---|---|---|---|
| T WS_SHARED_Q_HARD | 619.475574 | 2666.191371 | 2436.282715 | 10500.826733 |
| T WS_T128_Q_HARD | 616.573580 | 2665.648818 | 2468.555561 | 10500.826733 |
| T WS_T128_RESIDUAL_HARD | 597.178960 | 2654.205915 | 2344.415284 | 10500.826733 |
| T WS_OWN_RESIDUAL_HARD | 597.178960 | 2654.205915 | 2344.415284 | 10500.826733 |
| T WS_CONTEXT_RESIDUAL_HARD | 597.178960 | 2654.205915 | 2344.415284 | 10500.826733 |
| T WS_SOURCE_EQUAL | 1311.223497 | 2878.588762 | 2544.797341 | 10480.821469 |
| V WS_SHARED_Q_HARD | 547.721449 | 1829.356127 | 3125.980227 | 7260.393881 |
| V WS_T128_Q_HARD | 540.294588 | 1824.316638 | 3094.505647 | 7250.749899 |
| V WS_T128_RESIDUAL_HARD | 527.427043 | 1813.298607 | 2942.040400 | 7260.393881 |
| V WS_OWN_RESIDUAL_HARD | 527.424292 | 1813.298202 | 2942.040400 | 7260.393881 |
| V WS_CONTEXT_RESIDUAL_HARD | 527.430938 | 1813.300699 | 2942.040400 | 7260.393881 |
| V WS_SOURCE_EQUAL | 1276.440983 | 2168.137334 | 3097.435205 | 7194.824823 |

## Squeeze：训练误差下降没有转为纯组难例救回

SQUEEZE_REPORT `SQ-SET-CONTEXT-RISK-20261031-166`，来源上列Record；partial机制、限定输出核验通过。全部26fit末端MSE低于起点，full OWN .022456180→.022439700，CONTEXT .022456180→.022390252；更低残差回归损失没有带来hard决策优势，不把128步叫充分收敛。

| 候选−控制/m | T差 [95%组件描述区间] | V差 [95%组件描述区间] |
|---|---|---|
| WS_OWN_RESIDUAL_HARD_vs_WS_T128_RESIDUAL_HARD | 0.000000 [0.000000, 0.000000] | -0.002752 [-0.003249, 0.000000] |
| WS_CONTEXT_RESIDUAL_HARD_vs_WS_T128_RESIDUAL_HARD | 0.000000 [0.000000, 0.000000] | 0.003895 [0.000000, 0.004599] |
| WS_CONTEXT_RESIDUAL_HARD_vs_WS_OWN_RESIDUAL_HARD | 0.000000 [0.000000, 0.000000] | 0.006647 [0.000000, 0.007848] |

| 相对冻结159 | 全误选 | 救回/新增 | >1km选择excess | 纯组错/救回/新增 | 反转错 | 实际增害>1km/最大m |
|---|---|---|---|---|---|---|
| T WS_OWN_RESIDUAL_HARD | 156 | 0/0 | 90 | 44/0/0 | 154 | 0/0.000000 |
| T WS_CONTEXT_RESIDUAL_HARD | 156 | 0/0 | 90 | 44/0/0 | 154 | 0/0.000000 |
| V WS_OWN_RESIDUAL_HARD | 286 | 2/1 | 90 | 16/0/0 | 276 | 0/165.658994 |
| V WS_CONTEXT_RESIDUAL_HARD | 288 | 0/1 | 90 | 16/0/0 | 279 | 0/276.800483 |

冻结159原纯错误T44（24个选择excess>1km）、V16（5个>1km）；两个新头全部保持，**0纯组救回、0新增纯组错**。T全156错/反转154也完全不变；V OWN全287→286、救2新1/反转278→276，CONTEXT287→288、救0新1/反转278→279。小数级MAE差来自少数混组动作，不能解释为解决纯组问题。CONTEXT相对OWN V+.006647m；已有平均网络仍大幅优于二次/等权，保留原159，不为极小OWN变化选新冠军。

各新头与冻结159相比0新增>1km实际损害，主要P95/P99保持，T全部输出相同；去最大组件两新头与冻结网均相同。仍是旧0/1支持回退T1487/V2770零点，speed/heading保持；完整角色/纯混/方向/四属性、所有场和组件在原件中，不删不利证据。2000次seed20组件区间仅描述反复开发域，无新seed/独立新数据确认。

独立保存输出审计167964exact/38986numeric/252null通过，程序9.127428s/RSS109125632B、launcher9.626688s；范围是主位置/人口/尾部/组件/主要CI和不变域，**未独立重训26头或复算条件特征/选源诊断，不认证truth**。方向/角色记账9.532549s/RSS135622656B、四属性8.618399s/RSS404762624B，顺序各一次，补项launcher21.473260s，完整调用argv/returncode在accounting-calls.json。ME≠MAE/ESD未核。

计划无科学重试/改权重/V选容量；本地扩展fixture初版引用位置错误在READY前修正并通过。源context为既有信息的联合条件表达，非新物理观测；组件留出仅后端头、056前端没有新做组件留出；固定残差基底/128步/模型/正则仍限定此次负结果。当前不追加步/seed/权重枚举，不否定集合模型整体。后继纯组路线需先在T查可区别于现有统计的新证据，尤其原生时间/局部状态与参考/物理条件，不能只重复同信息换loss；未知不写成不可辨定理。

```text
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-SET-CONTEXT-RISK-20261031-166
input_experiment_record_ids: [LEVEL1-SET-CONTEXT-RISK-20261031-166-attempt-1]
claim_scope: fixed_T128_WS_frozen159_own_vs_context_residual_pilot
primary_result_with_uncertainty: context_no_increment_Vplus0.006647_vsOWN_pure44T16V_unchanged
process_dynamics: 26fits_3328updates_all_final_loss_lower_no_convergence_certification
mechanism_and_intermediate_findings: same_information_conditional_pooling_current_recipe_does_not_change_pure_errors
errors_heterogeneity_and_boundary_conditions: only_tiny_mixed_query_changes_no_new_actual_harms_over1km_zero_domains_unchanged
robustness_sensitivity_and_alternative_explanations: equal_capacity_exposure_OWN_control_component_intervals_drop_largest
new_P: none
new_F: F263
new_M: none_M108_scope_only
evidence_index: complete_166_curves_population_scores_outputs_audit_attributes_and_launcher
data_quality_and_provenance_gaps: K2_development_domain_features_fits_not_independently_rebuilt_no_physical_health_truth
open_questions: native_time_state_information_vs_reference_conditions_for_residual_pure_errors
next_routes: deep_read_Mode_C1.73
next_skill: deep-read Mode C
```
