# 180：物理选择代价与高害成本监督

2026-10-31；REC-COST-ROUTING-180 → SQ-COST-ROUTING-180，research-pipeline-contract v1.0。来源[Idea180/plan1.0](../proposals/cost_routing_20261031.md)、Mode C1.89/C61、179实证及[ICML2025完整来源AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/cost_deferral_icml2025_source_analysis_20261031.md)。探索性计算实验；execution_status=completed，科学/规定独审各一次成功、证据verified。累计充分108不增；不是全T2100/全V300、作者复现或独立泛化确认。

## Record：假设、数据与匹配操作

检验“已有较好组内历史状态却误选”能否由实际位置代价监督改善，**不预设178的log回归是根因**。固定178候选/12维特征、ridge91系数与normalizer、175 HARD/173 STRUCT前端及阈值/分组；只改单来源支持query的位置。零/多支持及全部速度航向exact保持178 O，0前端更新/teacher，新头不按源名/编号优先，不用真值净化组或在推理用reference。

T128缓存按首对应参考标签，当前组≥2节点且单支持的有标签query成为决策bag；单候选不训练但完整评价，不删难例或重复输出。实际55435 bags/144061候选行，两个前端视图共同训练、不是独立样本。每fold从全部T128排除同component；原前端已见过T标签，不称全链LOCO。T32十二leave＋full共13fold，每fold三个相同178初始化的1505参数共享评分头：C继续log误差MSE、L按物理regret成本的ν1加权交叉熵、T在L成本上加1km hinge。源等总质量/源内均分用于C bag损失；可变候选L/T用N(N−1)归一化。成本从T已存y反变换构造，只是训练监督。

对成本c_j，τ_j=Σ_(i≠j)c_i、r_j=−risk_j，loss=−Στ_j logsoftmax(r)_j/[N(N−1)]；L的c=e_km−min e_km，T再加max(c−1,0)。没有20km裁剪、V选门限或源优先。各头共享seed20的同一256×128个bag抽样、全部候选、Adam.001/wd1e−4/clip1、固定末步；**39fit/9984真实GPU更新**，0新analytic fit/teacher。每种目标13fold合425984次query呈现，三个目标合1277952次；同一view/query重复呈现不当独立样本。全模型T完成后才V，封存测试未访问。

相对O，新策略同时增加一个条件风险头及训练曝光，完整部署旧＋新残差头3010参数（新增1505）；不是全系统仅1505或零增容。C/L/T同容量/信息/初始化/曝光，L−C区分目标、T−L区分hinge；固定尺度/优化器不等于各目标相同梯度动力学，更不继承来源理论的最优性或尾部保证。

原T32/860对象与V32/1427对象，HARD/HO/HC/HL/HT、STRUCT/SO/SC/SL/ST各Mean+NN，共20臂。普通Mean与原base/O均强控保留，不按V组合180与181。

## Record：完整位置结果

单位米，全部各自/共同人口、四属性、raw-decoded、组件与场景原件完整保留。

|前端/策略|T MAE|V MAE|V RMSE|V P99|
|---|---:|---:|---:|---:|
|HARD原base|454.615|452.992|1502.760|5166.150|
|HO原178|392.233|355.760|1329.100|5003.510|
|HC继续回归|391.216|355.018|1327.931|5015.063|
|HL物理成本|401.862|364.136|1351.021|5114.223|
|HT高害成本|399.074|364.321|1350.229|5110.672|
|STRUCT原base|430.087|427.241|1447.519|4842.083|
|SO原178|367.202|328.916|1267.936|4359.930|
|SC继续回归|365.975|328.442|1266.489|4355.628|
|SL物理成本|373.034|335.269|1276.794|4340.073|
|ST高害成本|371.969|335.552|1276.913|4337.847|
|HARD普通Mean（五副本相同）|1202.191|1201.217|1922.727|5147.138|
|STRUCT普通Mean（五副本相同）|1227.212|1290.534|1950.607|4918.742|

V HARD家族71396/72196点＝98.891905%、1401/1427对象、重复203，decoded跨源mPCA81.759255%/mPFA14.924344%；STRUCT家族70775点＝98.031747%、1380对象、重复104、81.765139%/15.573316%。分组/分数、覆盖、速航全部exact不变，不称前端/覆盖或四属性学习增量，不拼两头冠军。

## Squeeze：目标对照、不确定性与动力学

|V共同点MAE差/2000次seed20组件描述CI|HARD|STRUCT|
|---|---:|---:|
|C−O|−.742 [−1.032,.258]|−.473 [−1.328,.162]|
|L−O|+8.376 [6.336,40.421]|+6.353 [3.953,34.299]|
|T−O|+8.561 [6.494,40.294]|+6.637 [4.246,34.464]|
|L−C|+9.118 [7.110,41.210]|+6.826 [4.425,35.602]|
|T−L|+.185 [−.222,1.064]|+.283 [.004,.384]|
|T−C|+9.303 [7.280,40.859]|+7.110 [4.717,35.819]|

T12/V9组件、V复用开发域/共享身份，区间非独立新域确认。T L−C亦为+10.646/+7.060m，T−C+7.858/+5.994m，四区间均在0以上；不是仅V突然失效。去最大组件V L−O仍+19.964/+19.817m，T−O+20.353/+20.046m，不是单一大组件贡献的表面差。C的微小平均改善区间跨0，H P99还升，不能据此换掉178。

13fold每种目标共3328个步骤，有限loss/梯度全通过；clip前梯度>1的C/L/T为0/93/360步，最大.428/4.356/8.996。full第1→256步的随机batch loss为C .0587→.1072、L1.0908→1.9025、T1.8268→3.5366；batch不同、目标尺度不同，**这些端点不证明发散、过拟合或收敛**。没有固定探针的优化收敛认证，理论条件最优不等有限参数/预算达到最优。clip频率差异只是动力学线索，未被干预为已证失败根因。

## Squeeze：纯组误选、旧害去留与尾部

|V新策略相对178 O|新增>1km增害|改善>1km|纯组新增害|最大增害m|
|---|---:|---:|---:|---:|
|HC|18|32|5|3336.268|
|HL|214|43|89|8822.109|
|HT|220|45|87|8822.109|
|SC|13|22|7|2717.367|
|SL|189|37|72|8411.444|
|ST|196|36|69|8411.444|

所有新变化只在单支持域；该域T13492/11919点，V29883/25320点。V L−O域均值+20.011/+17.759m，T−O+20.453/+18.551m；V纯组全部有效人口49455/47964中，L平均亦+4.306/+3.565m、T+4.267/+3.424m。不是把全部退化归为混组。零/多支持及全部速航差严格0。

以最初175/173 base为共同门槛，V原178的>1km害128/107，C后135/104、L286/242、T294/245。L救旧害10/15，却引入168/150；T救8/13、引入174/151；纯组旧28/19→L95/66、T93/62。这是跨base门槛的集合变化，不等于上表相对O的增害点，定义分别保留。T原104/75→L198/138、T186/134，同样不支持高害已解。

L/T在STRUCT全体V P99略降，但均值/RMSE与新增高害更差；HARD的P99反而增，不把单列P99冠军当尾部全面改善。hinge在T有部分均值/最大害信号，V却没兑现，不能沿V再扫hinge/步数/seed。

DE：在当前信息/容量/固定预算与动作域中，成本目标比匹配继续回归更差，不能证明log回归造成原高害；C61当前配方不支持部署。AI/UN：有限模型/优化条件、按N归一化的bag权重、噪声/对应及条件信息不足仍可影响结果；本实验不判哪项根因，更不否定代价敏感选择这一类成熟方法。

## Record：核验、真实服务、成本与原件

科学fcef505，新`association-code-run180`，原run `runs/LEVEL1-COST-ROUTING-20261031-180-attempt-1`；命令/环境见[L2S180](../../HandOver/L2S-20261031-180.md)、[实际启动](../../HandOver/S2L-20261031-180-start.md)及原request。CPU0–3×4/GPU0 UUID兼容共卡；程序09:38:46.068880Z–09:50:46.059209Z，719.990863s、RSS2525286400B、Torch峰值显存38731264B；launcher payload09:38:39.665777Z–09:50:47.755971Z，728.209665s。unit`mtad-25f88de0572d.service`真实exit0/inactive/dead/Resultsuccess/Restartno，Linger=yes。

规定独审同fcef505、新`association-code-audit180`，待181独审成功结束后FIFO一次执行，CPU16–19×4/无GPU/1800s advisory。unit`mtad-142878bd8c62.service`真实exit0/inactive/dead/Resultsuccess/Restartno；payload09:51:54.563178Z–09:54:09.646898Z，launcher135.219042s、RSS975654912B。主位置/人口/coverage/raw-decoded/尾部/组件/primary CI独立复算72.899118s；全部四属性用同评价器从保存数组完整记账24.579892s，**不是四属性独立双实现**；训练/动作独立记账33.790132s。重建全部55435bag、13fold排除组件、同一256×128采样、固定底座/normalizer、39个1505参数末步、日志9984更新，NumPy评分与GPU最大差7.153e−7；全部新位置/域不变/强控exact核齐。没有优化器逐步重放或物理真值认证。

科学＋规定收尾863.428706s，不当实际部署延迟。全部99公共原件60247371B＋真实服务终态清单回[180原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_history_followups/180)，tar8905625B，逐文件数/大小核齐；private坐标/身份/权重/抽样/大预测留server。11项READY前软件验收通过；科学/独审各一次，无失败重试、配方/计划外训练量变化或T_test访问。结案后主仓仅澄清audit输出的scope文字，明确四属性同评价器记账；原fcef505代码、服务和已回传JSON不覆写，无数值改动或重算。实际K2，K3仅软件接口检查，离线canonical_all不是在线端到端。

**保178平均收益，180当前C/L/T配方均不晋级。** 不因C少量均值信号追加训练，不因L/T失败否定整类方法；先回Mode C1.91/F287/E190、SR v3.2。后继从T纯组中历史候选运动/局部观测相容性和风险分数的条件信息重新找可区分证据；与181的端点/历史链接及单侧传播问题分开，不从V错误直接设cap或拼冠军。无自动新READY/全300/seed。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-COST-ROUTING-180
input_experiment_record_ids: [REC-COST-ROUTING-180]
source_idea_id: IDEA-COST-ROUTING-180
plan_version: '1.0'
evidence_status: verified
claim_scope: fixed_T128_13fold_39fit_9984_GPU_updates_T32_V32_single_support_20_arms
primary_result_with_uncertainty: V_L_vs_C_plus9.118_plus6.826m_and_T_vs_C_plus9.303_plus7.110m_intervals_above_zero_C_vs178_crosses_zero
process_dynamics: science728.209665_plus_audit135.219042_seconds_no_optimization_convergence_claim
mechanism_and_intermediate_findings: current_cost_objective_fails_matched_regression_control_not_proof_original_log_loss_root_cause
errors_heterogeneity_and_boundary_conditions: V_L_new_harms214_189_T220_196_pure89_72_and87_69_zero_multi_unchanged
robustness_sensitivity_and_alternative_explanations: frozen_controls_complete_metrics_independent_numpy_actions_bags_folds_sampling_and_checkpoints_same_evaluator_attribute_accounting
new_F: F287
new_E: E190
new_M: none
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
