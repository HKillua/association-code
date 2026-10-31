# 190：拟合端点×运动×几何的T侧因子分离

2026-11-01；REC-FITTED-ENDPOINT-FACTORIAL-190 → SQ-FITTED-ENDPOINT-FACTORIAL-190，research-pipeline-contract v1.0/workflow v2.0。来源[Idea190/plan1.0](../proposals/fitted_endpoint_factorial_20261101.md)、MC2.00/C66/F295–296/E198–199/M120 AI与[独立来源条件AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/latent_endpoint_conditions_20261101.md)。探索性计算pilot，execution_status=completed/evidence_status=verified，科学/规定独审各一次exit0；充分108不增，非V/全300/独立新域或作者复现。

## Record：研究问题、实际配置与数据

189的33个HARD害query仅一个片段，去记忆不能救，故检验潜在端点状态而不是继续换源。原生端点已是178的外推锚，不误称首次加入原生端点。固定178/189组、分数/阈值、选中节点与source等质量tie，全部原七策略base/O/PH/PC/PD/PG/MC导入exact；新GL原生端点＋geodesic OLS NE速度、FL全native位置OLS拟合端点＋同速度、FM拟合端点＋报告均值。两头各十策略Mean+NN共40臂。

每片段全部真实native时间与短弧解包lon，重算OLS slope/拟合端点；速度在native均值纬度转NE。原T32/860对象/12组件，2310次双坐标native OLS与2310次报告均值、同场模型跨前端共享；0新增神经/teacher/reference或似然优化，继承178已监督风险头不称全系统无学习。只全组单侧count0替位置，其他域与速航保持O，源仅等价类/无名字编号优先；真实主要K2/离线canonical_all、V/test未访问。

新三臂共同合法域，端点报告不足/缺失或拟合端点非法时均回MC且计数，不删点/clamp新状态。新动作可能与支持内旧插值边界不连续，不宣完整后验/在线。所有参考/缺失/重复/first-anchor对应保留，实际命令/环境见[L2S190](../../HandOver/L2S-20261101-190.md)/[实际启动](../../HandOver/S2L-20261101-190-start.md)及原request。

## Record：完整T位置结果

单位米，各自/共同点、全场/组件/去最大及四属性全部原件保留。

|策略|MAE|RMSE|P95|P99|
|---|---:|---:|---:|---:|
|HARD_NN|454.615138|1960.291531|1991.496178|5825.206510|
|HO_NN|392.232503|1910.707648|565.112989|6195.742358|
|HPH_NN|410.988657|2006.555132|576.814921|6968.041172|
|HPC_NN|868.711910|7335.750641|591.642516|12685.361337|
|HPD_NN|410.994320|2006.572566|576.525975|6969.309049|
|HPG_NN|393.294869|1911.869509|567.133099|6315.010233|
|HGL_NN|392.218631|1910.703842|565.112989|6195.742360|
|HMC_NN|393.295099|1911.904010|567.856919|6313.498688|
|HFL_NN|397.645286|1924.175830|575.027236|6342.780086|
|HFM_NN|398.507440|1924.600241|577.291214|6468.999818|
|STRUCT_NN|430.086696|1909.961213|1551.426000|5114.230338|
|SO_NN|367.202450|1876.080258|508.503071|5251.666565|
|SPH_NN|376.111126|1973.050408|506.449022|5461.323340|
|SPC_NN|686.539198|5505.773833|520.665785|9472.822325|
|SPD_NN|376.113708|1973.077858|506.169814|5461.323340|
|SPG_NN|364.973471|1854.084509|509.774295|5238.119327|
|SGL_NN|367.191668|1876.100082|508.503071|5241.967286|
|SMC_NN|364.977805|1854.092901|509.856901|5238.119327|
|SFL_NN|374.064056|1908.959411|511.691341|5432.687080|
|SFM_NN|371.679104|1884.357634|512.158876|5420.133001|

HARD37993/38224、STRUCT37745主位置点；全部原控制/coverage/raw-decoded/组/速航exact不变，不拼前端或指标冠军。本轮无覆盖/关联或新速航增量。

## Squeeze：所有primary与几何/端点/运动作用

下表共同位置点MAE差，T12组件/2000次seed20描述95%区间；T复用/身份共享、多臂探索，不当独立确认。

|T primary比较|MAE差m|描述95%区间|
|---|---:|---:|
|HGL_NN_vs_HO_NN|-0.013873|[-0.135575,0.014677]|
|HMC_NN_vs_HGL_NN|+1.076468|[-3.572671,7.041039]|
|HFL_NN_vs_HGL_NN|+5.426656|[3.217699,18.023219]|
|HFM_NN_vs_HMC_NN|+5.212342|[2.481244,17.451679]|
|HFM_NN_vs_HFL_NN|+0.862154|[-4.373486,6.270699]|
|HFM_NN_vs_HO_NN|+6.274937|[1.154067,17.312338]|
|HFL_NN_vs_HO_NN|+5.412783|[3.215999,17.892888]|
|HMC_NN_vs_HO_NN|+1.062595|[-3.613943,7.010348]|
|HFM_NN_vs_HPG_NN|+5.212571|[2.510020,17.469407]|
|SGL_NN_vs_SO_NN|-0.010783|[-0.147586,0.017108]|
|SMC_NN_vs_SGL_NN|-2.213863|[-7.499970,0.045744]|
|SFL_NN_vs_SGL_NN|+6.872388|[3.009338,20.518852]|
|SFM_NN_vs_SMC_NN|+6.701299|[3.109221,19.912960]|
|SFM_NN_vs_SFL_NN|-2.384951|[-8.219179,0.075891]|
|SFM_NN_vs_SO_NN|+4.476654|[2.168332,17.576383]|
|SFL_NN_vs_SO_NN|+6.861605|[3.009673,20.397369]|
|SMC_NN_vs_SO_NN|-2.224645|[-7.481111,0.003678]|
|SFM_NN_vs_SPG_NN|+6.705633|[3.104760,19.952123]|

GL−O完整仅−.013873/−.010783m，无新增/救回公里害，最大增害6.970/12.922m；几何在当前T数据不是公里害主要操作来源，不推广为任意区域无区别。FL−GL+5.426656/+6.872388m，FM−MC+5.212342/+6.701299m，四区间都在0上；拟合端点在两种运动条件下都更差。完整FM−O+6.274937/+4.476654m也区间在0上，RMSE/P99升，不晋级。全部组件及去最大原件保留，不能只抓某个旧害解除。

## Squeeze：完整尾部和原害/救回的保留

|T相对178 O|新增>1km害|改善>1km|纯组新增害|最大增害m|
|---|---:|---:|---:|---:|
|HGL|0|0|0|6.970|
|HFL|90|1|88|6713.135|
|HFM|125|1|123|6713.992|
|SGL|0|0|0|12.922|
|SFL|127|0|124|6713.135|
|SFM|98|4|95|6713.992|

相对最初base门槛：H原178104害/纯22，FL190/107、FM221/142；S原75/纯20，FL200/142、FM167/113。FL/FM的新害不只混组，不用纯组一词解释掉状态误差。

原PG害H34（其中33单节点query）、S1：GL在这批点均没有相对O公里害；FL仍1/1，FM仍33/0。**不能把FL的解除计给端点，因为其OLS速度控制GL本来就没这些害。** 真正同均值速度下FM−MC的端点替换只解除那个两节点害，33个HARD单节点害仍在；端点拟合不能修主体。

原PG救回H3/S33：FL保住公里救回0/0，S其中30点反而成相对O公里害；FM保住0/4、S5点转成公里害。旧救回去留与完整平均一并不利。该操作DE说明当前均值运动与原生锚的结合在部分点有价值；不证明原生端点是真值、所有拟合状态无用或真实参考错。

## Record：输入/实现核验、成本与计划偏差

全部T2310片段旧状态与重算native端点OLS绑定：最大坐标差2.842171e−13度、NE速度差1.421085e−14m/s，均在计划浮点容差内。T全部原始group-query共同fallback0；不是参数/接口不相容或全回原值掩盖效果。正常方程与独立centered-lstsq原生拟合及新球面位置重建通过，纬度/经度圆周等价不改误差判定。线性/IID误差、真实端部机动、偏差/相关与参考物理准确性仍UN。

科学/独审均6ebf497，新association-code-run190/audit190、原run `runs/LEVEL1-FITTED-ENDPOINT-20261101-190-attempt-1`。科学CPU8–11×4无GPU/RSS8GiB/1800s advisory；mtad-bb86bc494bf9.service真实exit0/inactive/dead/Resultsuccess/Restartno/Lingeryes。程序2026-10-31 19:39:46.042963Z–19:49:35.848110Z（北京时间11月1日03:39–03:49）、589.820483s、RSS3820838912B。

规定一次独审CPU16–19×4无GPU/1800s advisory，mtad-8c6989cda314.service真实exit0/inactive/dead/Resultsuccess/Restartno，payload19:49:57.433759Z–19:52:54.529481Z。主位置/人口/关系/覆盖/尾部/primary独核123.144929s、四属性同评价器完整记账21.048789s（非独立四属性双实现）、独立native-lstsq/球面动作/控制/保持域/纯混与旧base害核23.902136s、原189病例保留核1.552454s，RSS1784483840B。没有认证物理噪声、全局机理或作者完整系统。

科学launcher595.010694539s＋规定收尾177.276446586s＝772.287141125s。首场来源核验157次OLS/.143299605s另列E199，不藏成本或双计为190拟合。无科学/独审失败、改配方/删样本/重试或V/test访问。READY前17项含复用，首次独立lstsq对未中心化经度的常量引入浮点伪斜率，已改为同时中心化位置的等价QR求解、原容差不变，全部通过后才发布；未调整真实科学验收。

全部63公共原件123279821B、tar13595525B＋真实终态/传输清单回[190原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261101_drift_memory/190)，逐文件数/大小核齐。原生坐标/身份/模型参数/权重/完整预测留server，原152及178/188/189和失败冻结不动。保178与188条件候选，当前统一拟合端点配方不扩大，不再沿同一线性端点/β/窗口配方堆补丁。

DE：主要未解的单节点反转随平均运动选择而保留，几何/记忆和全段拟合端点未形成解决方案。AI/UN：下一从原生位置与SOG/COG时间关系、运动估计可靠性/可用历史条件找解法；不能因此宣任何一个为根因。C66当前配方关闭自动扩量，成熟状态估计仍有后续发展空间。回Mode C2.01/F297/E200、SR v4.2/OD v1.15；C65更简单MC在T与PG接近但V未测，若验证须新具体合同，不能直接补V成绩。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-FITTED-ENDPOINT-FACTORIAL-190
created_at: '2026-11-01'
source_ids: [REC-FITTED-ENDPOINT-FACTORIAL-190]
squeeze_id: SQ-FITTED-ENDPOINT-FACTORIAL-190
input_experiment_record_ids: [REC-FITTED-ENDPOINT-FACTORIAL-190]
source_idea_id: IDEA-FITTED-ENDPOINT-FACTORIAL-190
plan_version: '1.0'
evidence_status: verified
claim_scope: T32_860targets_40arms_2310_native_OLS_and_means_no_new_neural_reference_likelihood_or_V
primary_result_with_uncertainty: fitted_endpoints_worse_in_both_motion_conditions_all_four_primary_intervals_above_zero
process_dynamics: science595.010695_plus_audit177.276447_seconds_no_training_curve
mechanism_and_intermediate_findings: geometry_small_and_FM_retains33_singleton_harms_not_proof_observed_endpoints_true
errors_heterogeneity_and_boundary_conditions: FL_new_harms90_127_FM125_98_pure88_124_and123_95_old_rescues_lost
data_quality_and_provenance_gaps: full_T_code_cache_binding_verified_physical_noise_reference_and_manoeuvre_unknown
robustness_sensitivity_and_alternative_explanations: full_controls_native_lstsq_geodesic_and_saved_metrics_independent_four_attributes_same_evaluator
new_P: none
new_F: F297
new_E: E200
new_M: none
mode_C_input_bundle: P13_P14_F297_M120_still_AI_E200_C66_not_promoted
evidence_index: original_run_launchers_input_accounting_and_all_audits_linked_above
open_questions: [reliable_mean_motion_under_no_support, native_position_velocity_dependence, simpler_MC_V_effect]
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
