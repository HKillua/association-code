# 189：平均运动与OU记忆项的T侧匹配干预

2026-11-01，REC-DRIFT-MEMORY-ABLATION-189 → SQ-DRIFT-MEMORY-ABLATION-189，research-pipeline-contract v1.0/workflow v2.0。来源[Idea189/plan1.0](../proposals/drift_memory_ablation_20261101.md)、MC1.98/C65/F294/E197/M119(AI)、[原生运动独立AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/native_motion_transfer_20261031.md)。探索性计算pilot，execution_status=completed，科学/规定独审各一次真实exit0，evidence_status=verified。充分108＝72+36不增，不是V/全300/作者复现或独立新域。

## Record：问题、设计与实际血缘

检验188的平均漂移PG是否需要其OU记忆项，并落实T纯组新增害输入条件。完整导入188 base/O/PH/PC/PD/PG；只新生成MC，去掉PG位移μh+(v0−μ)d中的记忆项，成为同原生端点、同已选片段/源等质量权重的μh。μ从该片段有效原生SOG/COG报告均值重算，raw kn显式一次转NE m/s、真实秒；不改选源、分组、阈值或按V设cap。缺报告/少于两有效速度时与188同节点退旧状态，不删点。

仅全组所有原生观测之前/之后且count0替位置，其他zero及单/多支持、全部速航保持O。source只作等价关系、无名字/编号优先，可变K接口；真实主要K2，canonical_all离线，不宣在线。继承178已监督风险头，本轮0新神经/前端/teacher/reference或似然优化；实际4620次输入均值计算（含两前端重复），不写零估计/零计算。原188已做的native似然成本另留原记录。

原T32/860对象、两头各七策略×Mean+NN共28臂，0V、封存test未访问；全部困难/缺失/重复保留，首anchor/first-claim参考对应固定。T前端见过训练标签、组件复用，不是全链LOCO或独立确认。命令/配置见[L2S189](../../HandOver/L2S-20261101-189.md)/[实际启动](../../HandOver/S2L-20261101-189-start.md)和原request。predict API不接reference/身份，reference/纯度仅用于评价及以下条件分析。

## Record：完整位置与主要差

单位米，全部各自/共同人口、场景/组件与全部四属性保存在原件。

|T策略|MAE|RMSE|P95|P99|
|---|---:|---:|---:|---:|
|HARD_NN|454.615138|1960.291531|1991.496178|5825.206510|
|HO_NN|392.232503|1910.707648|565.112989|6195.742358|
|HPH_NN|410.988657|2006.555132|576.814921|6968.041172|
|HPC_NN|868.711910|7335.750641|591.642516|12685.361337|
|HPD_NN|410.994320|2006.572566|576.525975|6969.309049|
|HPG_NN|393.294869|1911.869509|567.133099|6315.010233|
|HMC_NN|393.295099|1911.904010|567.856919|6313.498688|
|STRUCT_NN|430.086696|1909.961213|1551.426000|5114.230338|
|SO_NN|367.202450|1876.080258|508.503071|5251.666565|
|SPH_NN|376.111126|1973.050408|506.449022|5461.323340|
|SPC_NN|686.539198|5505.773833|520.665785|9472.822325|
|SPD_NN|376.113708|1973.077858|506.169814|5461.323340|
|SPG_NN|364.973471|1854.084509|509.774295|5238.119327|
|SMC_NN|364.977805|1854.092901|509.856901|5238.119327|

HARD家族37993/38224主点、STRUCT37745主点，原860目标/各自组与重复人口不改；普通Mean/原base/O/PH/PC/PD/PG数组exact绑定，全部coverage/raw-decoded/分组/速度航向保持。后端位置干预不认领覆盖或跨源指标改善。

|T primary共同点差|MAE差m|2000次seed20/12组件描述95%区间|
|---|---:|---:|
|HPG_NN_vs_HO_NN|+1.062365970|[-3.621377462,6.977136063]|
|HMC_NN_vs_HO_NN|+1.062595086|[-3.613942637,7.010347883]|
|HMC_NN_vs_HPH_NN|-17.693558419|[-165.379819365,49.213653551]|
|HMC_NN_vs_HPC_NN|-475.416811015|[-1013.335266776,-230.448379026]|
|HMC_NN_vs_HPD_NN|-17.699221423|[-165.382701347,49.227985525]|
|HMC_NN_vs_HPG_NN|+0.000229116|[-0.013682282,0.034653673]|
|SPG_NN_vs_SO_NN|-2.228978754|[-7.511229601,0.007414702]|
|SMC_NN_vs_SO_NN|-2.224645203|[-7.481110711,0.003677877]|
|SMC_NN_vs_SPH_NN|-11.133321520|[-186.441499248,44.914064594]|
|SMC_NN_vs_SPC_NN|-321.561392787|[-864.886684806,-32.945874531]|
|SMC_NN_vs_SPD_NN|-11.135902854|[-186.471772040,44.945057717]|
|SMC_NN_vs_SPG_NN|+0.004333551|[-0.015454484,0.034551258]|

MC−PG整体+.000229/+.004334m，区间均跨0；对178仍+1.062595/−2.224645m，区间也跨0。不能把“不显著”当严格等效，也不能用T结果给MC填V成绩；189未读V。去最大/所有组件、全部控制对比完整保留。输入统计与工作先验的简单适配仍有价值，当前只是没有看到记忆项稳定的整体净增量。

## Squeeze：主点记忆作用、纯组与可见条件

T主参考对齐eligible451/344点，与原188绑定；MC-PG位置距离均值7.735/7.229m、中位5.991/5.944m，最大339.942m。448/307点距离>1m（只作观测尺度，不是质量/部署门槛），其中误差改变量有正有负，最大恶化312.024m、最大改善236.016m；全体均值小不等每点都无影响。新增/救回>1km均0。

相对178 O，MC仍新增34/1个>1km害、全部纯组，与PG原害逐点全部保留；救3/33，最大增害3657.704/3350.635m。相对最初base门槛：H104→134、纯22→56，S75→72、纯20→21；与对O的增害定义分开。单/多与非资格域exact0，不能靠去掉衰减修复这些公里级错误。

HARD害组节点数分布为33个单节点query＋1个两节点query，所有被选节点数1；因此33点在当前组内没有换候选动作，不是“多源挑错一个”可解释或解决。34点的记忆位置差仅3.193–5.247m，去掉后误差平均变化+.782m，公里级害保持。**这些是query点，不当34独立轨迹/案例**；同一片段的多个query高度相关。

|T HARD可见条件|全部eligible451点中位|纯组害34点中位|纯组救回3点中位|
|---|---:|---:|---:|
|端点传播时距s|1316.231|3123.445|1307.899|
|报告均值速度m/s|4.086|1.081|2.542|
|报告速度离散度m/s|35.109|33.815|31.740|
|均值与全段位置secant差m/s|.150|.293|.371|
|均值与OLS速度差m/s|.219|.766|1.272|
|端点速度与均值差m/s|26.755|13.445|35.880|
|OLS位置fit RMSE m|391.057|391.057|477.132|

这些条件明显重叠：全体时距最大5231.700s，救回也需外推；害的均值-OLS差并不大于所有救回，报告与位置secant相容也不保证长期真实误差好。不据此部署“长时回退”“残差大拒绝”或认为报告噪声已识别。STRUCT唯一害在两节点/两源组、selected1，传播2836.499s、均值-OLS差1.193m/s；其33个救回里包含更长时距，不能用horizon单阈值解决。全部字段/分位/点数、纯混/before-after/害与救回在condition-profiles.json。

DE：移除记忆项没有消除原纯组害，主点操作作用已被直接测量；整体平均运动解释比“OU衰减已产生收益”更值得先保留，但不从T证明V收益唯一原因。AI/UN：平均运动估计、端点状态、坐标传播几何及跨片段历史仍需分离，不能把当前完整组纯度当参考物理真值认证。优先核代码中原生端点与拟合状态的关系，再形成新操作；不再次无依据换loss/加seed。

## Record：真实服务、规定独审、成本与偏差

科学/独审均冻结618dd1e，新association-code-run189/audit189、原run `runs/LEVEL1-DRIFT-MEMORY-20261101-189-attempt-1`。科学CPU8–11×4无GPU/1800s advisory，mtad-22f90c6b5cad.service真实exit0/inactive/dead/Resultsuccess/Restartno/Lingeryes；程序2026-10-31 19:12:51.715410Z–19:17:26.311151Z（北京时间11月1日03:12–03:17）、274.595787s、RSS2079866880B。

规定一次独审CPU16–19×4无GPU/1800s advisory，mtad-845206e1c2ee.service真实exit0/inactive/dead/Resultsuccess/Restartno；payload19:18:09.811990Z–19:19:37.427734Z。主位置/人口/coverage/关系/尾部/primary CI独核54.604480s；全部四属性同评价器保存数组完整记账12.791366s，非四属性独立双实现；独立raw native均值/球面新MC/权重与保持域及冻结原控制核13.659001s；T条件保存行/人口/预算/原34/1绑定核2.057258s。RSS1074417664B。描述量提取与条件分层共用代码，未声称物理诊断独立复现或已证因果。

科学launcher293.491648938s＋规定独审87.739351708s＝381.231000646s。含输入/全部评价/写盘，不是部署延迟。没有新科学/独审失败、训练量/数据/配方变更或V/test访问。READY前本地首次fixture缺父run元数据、并误要求圆周均值回退与原坐标bitwise相同，已改为与188实际同节点回退逐数组一致，14项检查通过后才发布；科学数值/验收未放宽。公共原件解包时系统Python不支持filter，改用项目3.12环境完成，原包未改、无科学重跑。

全部63公共原件61580385B、tar7209379B＋真实终态/传输清单回[189原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261101_drift_memory/189)，逐文件数/大小核齐。坐标/身份/报告均值/权重/完整预测/逐点条件行留server；原152及178/188和全部失败保留。当前178不替换，188仅条件候选，189无自动V/全300/新seed/后继READY。Record→Squeeze→Mode C1.99/F295/E198、SR v4.0/OD v1.13。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-DRIFT-MEMORY-ABLATION-189
created_at: '2026-11-01'
source_ids: [REC-DRIFT-MEMORY-ABLATION-189]
squeeze_id: SQ-DRIFT-MEMORY-ABLATION-189
input_experiment_record_ids: [REC-DRIFT-MEMORY-ABLATION-189]
source_idea_id: IDEA-DRIFT-MEMORY-ABLATION-189
plan_version: '1.0'
evidence_status: verified
claim_scope: T32_860targets_28arms_4620_input_means_zero_new_fit_or_V
primary_result_with_uncertainty: MC_minus_PG_plus0.000229_plus0.004334m_both_intervals_cross_zero_no_equivalence_claim
process_dynamics: no_training_curves_science293.491649_plus_audit87.739352_seconds
mechanism_and_intermediate_findings: removing_memory_retains_all_34_1_pure_km_harms_33_HARD_harm_queries_singleton_groups
errors_heterogeneity_and_boundary_conditions: input_condition_overlap_single_candidate_no_swap_action_memory_can_still_move_one_point339.942m
robustness_sensitivity_and_alternative_explanations: independent_positions_native_mean_controls_relations_and_full_same_evaluator_attributes_no_physical_cause_claim
new_P: none
new_F: F295
new_E: E198
new_M: none
evidence_index: original_run_launchers_output_audits_and_condition_profiles_linked_above
open_questions: [endpoint_and_velocity_geometry_separation, cross_fragment_history, independent_V_increment]
mode_C_input_bundle: P13_P14_F295_M119_still_AI_E198_C65
next_routes: deep-read_Mode_C_then_source_conditions_before_new_Idea
next_skill: deep-read_Mode_C
```
