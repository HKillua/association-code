# 188：单侧速度记忆与漂移的条件收益及反例

2026-10-31；REC-ONE-SIDED-PERSISTENCE-188 → SQ-ONE-SIDED-PERSISTENCE-188，research-pipeline-contract v1.0。来源[Idea188/plan1.0](../proposals/native_motion_postqualification_20261031.md)完整继承[185科学设计](../proposals/native_motion_interventions_20261031.md#persistence-185)、Mode C1.95/C65/M119(AI)/F288–291/E191–194及[CTCRW与原生运动来源AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/native_motion_transfer_20261031.md)。探索性计算实验，execution_status=completed，科学/规定独审各一次真实exit0、evidence_status=verified；充分108＝72+36不增。不是作者完整KF/动物轨迹模型复现、全V300、独立新域或可信同行胜出。

## Record：问题、实际操作和学习范围

问题：当query在当前预测组全部native观测之前/之后且count0，恒定端点速度的长时间记忆是否合理？固定178 HARD/STRUCT组、原风险scores/选中节点/源等质量tie，不退固定源。只在上述全组单侧zero域替位置：PH原生端点保持，PC原生端点＋报告速度CV，PD零均值OU，PG原生平均漂移OU；其余zero/全部单多支持和速航exact O。178原始OLS状态及base/普通Mean均完整强控。

每原片段有效报告SOG/COG从kn一次转NE m/s，用真实秒拟合v_next−μ=exp(−βdt)(v_prev−μ)+ε的**工作似然**。PD μ0、PG μ为该片段全部native报告速度均值；q=−expm1(−2βdt)/(2β)、β0时q=dt，σ²剖面为Σ||ε||²/q/(2n)。66个预定β候选[0]+logspace(−6,2,65)/median(dt)，最小n log(max(σ²,1e−24))+Σlogq；固定网格非精确连续MLE。常速取β0并标不可辨，端点缺报告/不足两速度则四新臂同节点回旧状态，不删观测/难例。原论文估计含噪位置与潜在状态，而本轮直接拟合含噪报告速度、使用固定输入均值漂移，差异与未知测量噪声保留。

传播d=sign(h)(1−exp(−β|h|))/β、β0时d=h；PD位移v0d，PG位移μh+(v0−μ)d，原生端点球面NE指数映射。负h为明确离线时间反演工作适配，不宣严格因果/在线；μ0的条件均值位移有界不等方差/误差有界，非零μ还保线性漂移。

本轮**0新增神经/前端/teacher/reference系数拟合**，继承178已经监督训练的风险头，不是全系统无学习。实际24644 native似然工作fit＝T9240＋V15404，V只用输入报告、无参考标签，模型形式/网格/停止规则均V前固定；两前端重复每片段PD/PG，不能把调用当独立样本或写零拟合。原T32/860对象、V32/1427对象，base/O/PH/PC/PD/PG各Mean+NN两头24臂，全人口/重复/缺失不删、T_test封存。原185失败及186资格后另立188，不改原失败或公式。

## Record：全部位置结果与覆盖

单位米，所有各自/共同人口、全场/组件及四属性原件完整保留。

|策略|T MAE|V MAE|V RMSE|V P99|
|---|---:|---:|---:|---:|
|HARD base|454.615|452.992|1502.760|5166.150|
|HARD 178 O|392.233|355.760|1329.100|5003.510|
|HARD 端点HOLD|410.989|365.452|1345.992|5208.192|
|HARD 报告CV|868.712|803.331|6869.342|9003.314|
|HARD 零均值OU|410.994|365.428|1345.823|5204.960|
|HARD 平均漂移OU|393.295|353.073|1310.309|4927.703|
|STRUCT base|430.087|427.241|1447.519|4842.083|
|STRUCT 178 O|367.202|328.916|1267.936|4359.930|
|STRUCT 端点HOLD|376.111|337.492|1195.460|4684.825|
|STRUCT 报告CV|686.539|766.436|7063.118|8054.191|
|STRUCT 零均值OU|376.114|337.474|1195.362|4683.971|
|STRUCT 平均漂移OU|364.973|326.530|1251.402|4309.513|

普通Mean各副本与原HARD/STRUCT exact。V覆盖仍98.891905%/98.031747%、1401/1380对象、重复203/104；decoded跨源mPCA81.759255%/81.765139%、mPFA14.924344%/15.573316%，全部raw-decoded和速航exact不变。不能把后端位置候选称关联/覆盖/四属性学习增量，不拼两个前端的冠军。

## Squeeze：全部主要比较与不确定性

共同有效位置点、2000次seed20组件描述区间，负差更好。T12/V9组件、复用V开发域且共享身份、多臂探索，不当独立显著性确认。

|V primary比较|MAE差m|95%描述区间|
|---|---:|---:|
|HPH_NN_vs_HO_NN|+9.691137|[-20.514997,41.794234]|
|HPC_NN_vs_HO_NN|+447.570906|[45.199493,801.279127]|
|HPD_NN_vs_HO_NN|+9.667519|[-20.424356,41.762008]|
|HPG_NN_vs_HO_NN|-2.687618|[-3.224625,1.132183]|
|HPD_NN_vs_HPC_NN|-437.903387|[-800.539700,-45.486388]|
|HPG_NN_vs_HPC_NN|-450.258524|[-802.087821,-45.655737]|
|HPD_NN_vs_HPH_NN|-0.023618|[-0.032358,0.089829]|
|HPG_NN_vs_HPH_NN|-12.378754|[-40.613625,18.839342]|
|HPG_NN_vs_HPD_NN|-12.355137|[-40.581398,18.748702]|
|SPH_NN_vs_SO_NN|+8.576029|[-52.310504,17.090454]|
|SPC_NN_vs_SO_NN|+437.520138|[-10.399607,510.803441]|
|SPD_NN_vs_SO_NN|+8.558189|[-52.312775,17.069081]|
|SPG_NN_vs_SO_NN|-2.385860|[-2.760578,-0.058588]|
|SPD_NN_vs_SPC_NN|-428.961949|[-493.822558,-34.328396]|
|SPG_NN_vs_SPC_NN|-439.905998|[-513.552972,9.979157]|
|SPD_NN_vs_SPH_NN|-0.017840|[-0.021376,0.010544]|
|SPG_NN_vs_SPH_NN|-10.961890|[-19.847259,51.910105]|
|SPG_NN_vs_SPD_NN|-10.944049|[-19.825907,51.912377]|

PG对O V−2.688/−2.386m，HARD区间跨0，STRUCT仅略低0；去最大组件为+.103/−.298m，效应弱且集中度/异质性不能忽略。PG的T对O为+1.062[−3.621,6.977]/−2.229[−7.511,.007]m，HARD训练侧反例保留；不是全条件都涨。V完整RMSE/P99两头下降，但这不等每点/全尾部无损害。PH/PD/PC对完整O净均值均差，不能只比较失效CV后宣有效。PG−PD均值大幅较好但V两头区间都跨0，零均值假设和漂移操作不同，尚不宣漂移是唯一根因。

## Squeeze：尾部、纯组和旧害变化

|V相对完整178 O|新增>1km增害|改善>1km|纯组新增害|最大增害m|
|---|---:|---:|---:|---:|
|HPH|278|191|209|21340.790|
|HPC|669|29|492|222978.209|
|HPD|278|191|209|21337.692|
|HPG|11|71|11|2001.252|
|SPH|321|134|240|14731.241|
|SPC|563|27|391|325917.349|
|SPD|321|134|240|14721.217|
|SPG|0|42|0|502.112|

PG在V纯组总体均值−2.188/−1.310m，但HARD新增11个>1km害全为纯组，最大2001.252m；STRUCT无新增>1km害仍有最大502.112m退化，不称无任何新错。PG相对最初base门槛害H128→139、纯28→39，旧害0救回/新增11；S107→92、纯19不变，救旧15/新增0。相对O的>1km改善71/42与跨base旧害去留是两种定义，不混用。

**T反例不能藏：** PG相对O H新增34害/全纯、救3、最大3656.389m；S新增1/纯1、救33、最大3351.211m。相对base门槛T H104→134、纯22→56；S75→72、纯20→21。因此不能只凭V STRUCT的“0新增公里害”直接扩大或固定组合。PH/PD几十公里、PC最高325.917km增害原件全保留，不裁剪成漂亮尾部。

主参考对齐eligible T451/344点，V784/655点；PG该域V均值−244.751/−257.800m，T+89.495/−244.572m。大量非改动点稀释完整系统差，因此本域有具体改善空间，也有明确HARD纯组反转；其余域/速度航向exact，不拿raw原始group-query当主点分母。

## Squeeze：工作参数可辨性与缺少的直接控制

计划诊断聚合核T原始eligible group-query16383/15490，V19519/17022（包含未匹配/重复输出），与主点451/344、784/655分开。选中节点无缺报告回退、常速limit0、variance-floor0；全部实际24644调用都有工作拟合，不是参数默认0。

T每前端2310片段×PD/PG，V3851片段×PD/PG，两个前端重复这些模型；T grid边界582/4620、V1005/7702，每前端相同，不当独立噪声样本。PD/PG正β半衰期中位T约.16465s、V约.16460s；T高端索引65为282/297次、V503/502次，完整66格直方图保留。全部原始group-query被选端点时距中位T约2029/2037s、V1535/1525s、最大7052s，参数持续性与传播时间相差很大；但β分布是全部片段，不是仅主参考点选中的片段，不能直接据此解释均值收益或判真实运动记忆极短。

DE数学条件：当β很大且|h|远大于1/β，PG接近同原生端点＋μh，即**平均报告速度CV**。当前PC只用端点瞬时报告v0，因此PG相对PC同时改变平均运动和速度记忆；PH/PD近似并不识别真实测量噪声。**尚缺同端点/同节点/同权重的μ-CV直接控制，不能把PG收益归功于OU衰减。** T侧后续应比较同输入平均速度CV与PG，连同HARD新增纯组害核漂移/端点/候选条件；不据V挑cap/固定前端或声称协方差校准。

AI/UN：报告噪声可能令工作似然偏向短记忆、全片段平均可能比端点瞬时报告稳定，但当前未用干预区分，参考/组条件仍未证根因。即使新增价值主要来自更可靠的漂移/长期恢复适配，也值得发展，不因成熟先例或OU额外贡献尚未证就结束整条方向；需要具体匹配控制和反例修复。

## Record：服务、完整核验、成本与计划偏差

科学/独审均冻结3c3e0e3、新association-code-run188/audit188及run `runs/LEVEL1-ONE-SIDED-PERSISTENCE-20261031-188-attempt-1`，[L2S188](../../HandOver/L2S-20261031-188.md)/原request为实际完整命令和环境。科学CPU20–23×4无GPU/3600s advisory，mtad-45b78b44babb.service；程序11:02:23.911829Z–11:13:37.250997Z、673.339907s、RSS2297225216B。真实exit0/inactive/dead/Resultsuccess/Restartno/Lingeryes。

按完成顺序待187独审释放CPU后FIFO一次规定独审，CPU16–19×4无GPU/1800s advisory，mtad-38130eeb7169.service真实exit0/inactive/dead/Resultsuccess/Restartno；payload11:18:03.936570Z–11:21:50.512391Z。主位置/人口/关系/覆盖/尾部/场组件/primary独核103.464735s；四属性同评价器保存数组完整记账30.672403s，非独立四属性双实现；原生全网格剖面目标、参数选择、球面位移/权重/保持域与纯混/害独核89.257650s、RSS1189195776B。全网格最优相对浮点容差1e−10×max(1,|最小目标|)沿原计划，不放宽科学验收；不是物理β/噪声认证或作者完整KF复现。

科学launcher678.871464529s＋规定独审226.706494710s＝905.577959239s，已计划诊断只读聚合另1.143570023s，合906.721529262s。全部输入/工作拟合/评价/写盘计成本，非部署延迟。此前185失败实际314工作fit/9.215925s、184失败8.262476s与共享186资格90.650868s分别保留，不从总成本隐藏或双计共享资格。无新科学/独审失败、配方/训练量/样本排除/选模变化、神经新更新或封存test访问；参数/跨度聚合是计划内诊断的科学后只读汇总，脚本保存，不倒填为独审项目，不按其结果回写原参数。

全部96公共原件82738088B、tar10852820B＋真实服务/传输清单回[188原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_native_motion/188)，逐文件数/大小核齐；[聚合脚本](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_native_motion/analysis/native_diagnostic_summary.py)、全部直方图与范围为独有证据。坐标/身份/原生参数/权重/完整预测和scores留server。源换序/K3仅软件接口，不冒称真实K≥3泛化、在线或端到端。

**保178当前基准，188 PG保条件有效候选，不直接替换完整系统。** PH/PC/PD当前配方不扩大；PG的平均/RMSE/P99正信号与HARD纯组害/T反例/缺μ-CV控制一起回Mode C1.98/F294/E197、SR v3.9/OD v1.12。下一从训练侧分离平均运动与记忆、HARD纯组端点反转条件，再形成新具体合同；无自动全300/新seed/组合或按V设门槛。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-ONE-SIDED-PERSISTENCE-188
input_experiment_record_ids: [REC-ONE-SIDED-PERSISTENCE-188]
source_idea_id: IDEA-ONE-SIDED-PERSISTENCE-188
plan_version: '1.0'
evidence_status: verified
claim_scope: T32_V32_24arms_24644_unlabeled_native_working_fits_zero_new_neural_updates
primary_result_with_uncertainty: V_PG_vs178_minus2.688m_CI_crosses_zero_and_minus2.386m_CI_minus2.761_to_minus0.059_shared_development_components
process_dynamics: native_working_fit_boundaries_and_short_memory_not_physical_calibration
mechanism_and_intermediate_findings: drift_mean_and_memory_not_separated_without_same_endpoint_mu_CV_control
errors_heterogeneity_and_boundary_conditions: V_PG_new_harms11_0_all_HARD_pure_T_new_harms34_1_coverage_relations_speed_heading_exact
robustness_sensitivity_and_alternative_explanations: complete_controls_main_metrics_native_grid_math_and_all_strata_verified_no_causal_or_author_claim
new_F: F294
new_E: E197
new_M: none
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
