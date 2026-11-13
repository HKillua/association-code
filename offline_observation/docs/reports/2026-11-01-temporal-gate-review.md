# 194原生阶段任务验证：当前小网络门控未超过简单强控

## EXPERIMENT_RECORD：EXP-TEMPORAL-GATE-194

2026-11-01，research-pipeline-contract v1.0/workflow v2.0，computational exploratory pilot，状态 completed/current-formulation-not-promoted。来源IDEA-TEMPORAL-GATE-194/plan1.0、MC2.06/C68/F302/E205，[完整Idea与行为契约](../proposals/temporal_gate_repair_20261101.md)、[所引用193完整科学设计](../proposals/temporal_gate_20261101.md)、[实际命令合同](../../HandOver/L2S-20261101-194.md)。193导入技术失败已单独完整结案，不覆盖或重复计成194训练。

### 问题、干预与控制

问题是原178选中历史片段后，MC只保留全段报告均值，原生阶段信息/顺序是否能改善单侧zero恢复，尤其不能换源的单片段难例。由完整来源AP→Mode C→Idea形成，未预定SSM/噪声根因。194仅修复193训练入口间接导入CPU审计清空CUDA环境的问题，科学配方保持。O和MC是同一已选历史候选的两种时间状态估计，不是固定两传感器贡献。

三头MEAN/BAG/ORDER均32→32→32→1、2145参数、相同初始化/标准化/同抽样曝光，alpha读出O与MC；MEAN仅12全段/query统计、阶段20维标准化后置零，BAG把同四块池化描述数值规范排序删除块顺序，ORDER按时间排序。HALF为共同初始化的固定.5估计器控制，不部署为固定传感器配方。只原生支持全组之前/之后、count0的位置改变；其余支持域、速航、分组/阈值/原选源与全部旧28臂保留。完整44臂，不拼两前端冠军。

### 输入、学习与完整人口

复用178/189/183/127/151原T缓存和原manifest：128场全部准备；T32十二组件leave＋full＝13fold、各三头39fit。每fold造训练候选用对应旧178风险头；normalizer和新头排除held组件，前端曾见T，不能称全链LOCO或独立测试。标签绑定原first-anchor/first-claim，有效两位置query全部保留，不看V/test。

每头128步、batch1024、seed20、Adam .001/wd1e−4/clip1、固定末步，不选最好步；相同逐头初始化和抽样，实际4992次GPU更新，0新前端/ridge/teacher/似然拟合。监督目标为生产圆周位置读出后的平滑haversine米误差/1000，非内部重建proxy。原标签物理真实性/对应未知不因使用任务loss而消除。

full模型2464监督行来自80场，余48场没有合法可监督行，仍完成输入检查；不是失败剔除。13fold合29790行/原标签绑定29790次，是相关且重复的训练视图，不是29790独立对象。训练各fold实际如下，完整逐场/每前端行数与名单在原件training-population/training-folds.json。

|fold|实际有监督行的场数|行数|每头更新|
|---|---:|---:|---:|
|leave_component-13bf8d8c05af98b86d432a08|79|2369|128|
|leave_component-1b730762e8dcc866d831d9a3|79|2462|128|
|leave_component-23608a7a22c5322f7254ffc1|44|1278|128|
|leave_component-2631a2fab1075a4e376f5869|80|2464|128|
|leave_component-4eb65272866a82b14a0055d6|79|2462|128|
|leave_component-5ee24adf22c874beaf8d076d|78|2450|128|
|leave_component-75391b641c3cd7287e382451|79|2460|128|
|leave_component-805a0d28ca2328b9fac2c2f2|80|2464|128|
|leave_component-9ea441151d6468bd21826aea|78|2422|128|
|leave_component-c7082a697d934f0004db48b9|57|1669|128|
|leave_component-d784c25f6ac4208b0a38e796|77|2362|128|
|leave_component-e062a18f3a426eba715c16d2|80|2464|128|
|full|80|2464|128|

原T32评价860对象，HARD/STRUCT各自原有效位置37993/37745点、覆盖841/834对象、902/866输出组、重复61/32；所有44臂完整。原生单侧eligible group-query为16383/15490（包含未被主评价first-claim匹配的输出），主有效旧病例451/344点；不把二者混为同一人口。0V/封存test，未新增标签或重开原archive。

### 实际执行、设备和成本

科学/audit各一次独立checkout与attempt1，代码a0c245e仅定位，原活动与失败目录冻结。科学 `/home/wangguangjie/djs/association-code-run194`、audit `/home/wangguangjie/djs/association-code-audit194`；run位于服务器主仓 `runs/LEVEL1-TEMPORAL-GATE-20261101-194-attempt-1`。科学mtad-ff91c2c68e4e.service、独审mtad-7a86db18f40e.service，均真实COMPLETED/exit0/inactive/dead/Resultsuccess/Restartno/Lingeryes。完整真实argv/环境/起止在两个request/status原件，无简化替代命令。

科学GPU0 UUID GPU-c5a00c37-f181-90b3-f220-be6e7457f473兼容共卡，3090/CPU8–11×4/RSS8GiB/CUDA分配1GiB预算；audit CPU16–19×4无GPU，墙时3600/1800s advisory均未超。新冻结入口真实CUDA导入及一元素分配成功，run再次记录预检与193失败血缘；不是新训练smoke。10项READY前必要软件检查/编译/diff通过，未拿软件替代真实效果。

science内部886.428478629s，原launcher890.255668817s；独审launcher219.930464629s；设备探针3.020515648s，科学＋规定独审＋探针合1113.206649094s。science峰RSS5284745216B、Torch CUDA已分配峰21755392B（不含驱动上下文/其他进程），非全系统GPU内存或线上延迟。原193失败9.422683403s另计，193只读探针单项时间缺失保留；编辑/传输/只读查询不伪装成完整研究成本。
实际10034次native均值/阶段构造（T128准备7724＋T32评价2310），冻结风险前向3108次，39新fit/4992更新；不是0学习。科学UTC2026-10-31T22:18:27.200172至22:33:17.319624+00:00，审计22:33:27.181520至22:37:06.989968；北京日期11月1日。

### 完整T结果与不确定性

MAE/RMSE/P95/P99单位米。下面22个NN读出全部列出；22个MEAN控制按各自前端保持原数组，44臂全部四属性/逐场/组件/原始与decoded证据在原件中，没有只保存冠军。H与S人口不同，不横向拼接冠军或对论文表格宣布胜负。O为178历史恢复，MC为简单报告均值，GM/GB/GT分别为MEAN/BAG/ORDER新头；PH/PC/PD/PG是原188持续性控制，方程及全部负例见[原完整记录](2026-10-31-one-sided-persistence-review.md)。

|前端/NN臂|MAE|RMSE|P95|P99|相对O新增>1km误差增害点|
|---|---:|---:|---:|---:|---:|
|HARD|454.615138|1960.291531|1991.496178|5825.206510|—|
|HO|392.232503|1910.707648|565.112989|6195.742358|—|
|HPH|410.988657|2006.555132|576.814921|6968.041172|169|
|HPC|868.711910|7335.750641|591.642516|12685.361337|384|
|HPD|410.994320|2006.572566|576.525975|6969.309049|169|
|HPG|393.294869|1911.869509|567.133099|6315.010233|34|
|HMC|393.295099|1911.904010|567.856919|6313.498688|34|
|HHALF|392.717334|1910.702366|566.584058|6281.111919|22|
|HGM|393.107747|1911.906842|567.133099|6290.478192|25|
|HGB|393.322425|1912.750751|567.133099|6321.218344|27|
|HGT|393.342947|1913.069439|566.774065|6321.218344|28|
|STRUCT|430.086696|1909.961213|1551.426000|5114.230338|—|
|SO|367.202450|1876.080258|508.503071|5251.666565|—|
|SPH|376.111126|1973.050408|506.449022|5461.323340|132|
|SPC|686.539198|5505.773833|520.665785|9472.822325|283|
|SPD|376.113708|1973.077858|506.169814|5461.323340|132|
|SPG|364.973471|1854.084509|509.774295|5238.119327|1|
|SMC|364.977805|1854.092901|509.856901|5238.119327|1|
|SHALF|366.037991|1864.595997|508.969373|5238.119327|1|
|SGM|366.067073|1863.329931|508.969373|5238.119327|1|
|SGB|365.415904|1857.343027|508.969373|5238.119327|1|
|SGT|365.308306|1856.612511|508.969373|5238.119327|1|

增害/救回指相对同query的O误差多/少1000米，不是“绝对误差超过1000米”。—是自身或不适用，其他全部pairwise原件保留。主要预定差值candidate−baseline，负为好；12组件/2000seed20描述百分位区间，不是独立确认性检验，多比较不挑单一显著项。

|比较|完整MAE差m|95%组件描述区间m|去最大组件差m|
|---|---:|---|---:|
|HGT_NN_vs_HGB_NN|+0.020523|[-0.172379, +0.533612]|+0.361072|
|HGT_NN_vs_HGM_NN|+0.235201|[-0.253194, +1.630471]|+1.020668|
|HGT_NN_vs_HMC_NN|+0.047849|[-1.156305, +1.865337]|-0.154913|
|HGT_NN_vs_HO_NN|+1.110444|[-1.744674, +5.870217]|+3.379276|
|HGT_NN_vs_HHALF_NN|+0.625614|[-0.018973, +2.404767]|+1.676503|
|HGM_NN_vs_HMC_NN|-0.187352|[-2.776651, +2.097959]|-1.175581|
|HGB_NN_vs_HMC_NN|+0.027326|[-1.428622, +1.349355]|-0.515985|
|HPG_NN_vs_HO_NN|+1.062366|[-3.621377, +6.977136]|+3.509114|
|SGT_NN_vs_SGB_NN|-0.107598|[-0.187256, +0.354366]|-0.010523|
|SGT_NN_vs_SGM_NN|-0.758768|[-2.430956, -0.021393]|-1.670103|
|SGT_NN_vs_SMC_NN|+0.330501|[-0.095605, +1.708027]|+0.980246|
|SGT_NN_vs_SO_NN|-1.894145|[-6.675046, +0.000456]|-4.780752|
|SGT_NN_vs_SHALF_NN|-0.729685|[-2.941106, +0.512922]|-1.839446|
|SGM_NN_vs_SMC_NN|+1.089268|[-0.002545, +3.306599]|+2.650349|
|SGB_NN_vs_SMC_NN|+0.438099|[-0.002575, +1.380237]|+0.990769|
|SPG_NN_vs_SO_NN|-2.228979|[-7.511230, +0.007415]|-5.780703|

ORDER−BAG两头均跨0（H+.020523/S−.107598m），未获顺序增量；STRUCT对MEAN−.758768m区间在0下，但仍差于MC+.330501m。H对MC+.047849m、RMSE/P99更差，且对HALF+.625614m。S对O的−1.894145m区间仍跨0，且不如无需新学习的MC−2.224645m；不能只拿较弱MEAN/O宣布成功。去最大后的方向与数值全部保留，不因总体差小判严格等效。

训练39条曲线全部128步/有限loss和梯度，MEAN/BAG/ORDER首步损失范围同为5.077–12.263km，末步约4.705–10.458/4.676–10.412/4.673–10.426km（不同fold和抽样；完整曲线保留）。损失变化不证明任务增益，未据其早停/加步。

### 原难例、尾部和异质性

|前端/条件|原PG/MC公里增害|HALF保留|MEAN保留|BAG保留|ORDER保留|原公里救回|HALF/MEAN/BAG/ORDER仍救回|
|---|---:|---:|---:|---:|---:|---:|---|
|HARD|34点/2组|22|25|27|28|3点/1组|0/0/0/0|
|STRUCT|1点/1组|1|1|1|1|33点/2组|19/21/27/28|

ORDER没有把旧救回翻成新的>1km害，但H原3点全部不再达到救回1km；S原33点仅保28。H仍28个纯组/zero公里增害，S1个同样纯组/zero，最大额外误差均3275.993981m。不能将少掉6个H旧害全归功时序——无需阶段信息的HALF少掉12个，且均值更好；HALF本身亦不超过O/MC全部强控，不晋级。ORDER对BAG没有新增/救回公里点、最大增害约195.423m；对MEAN H新1害/0救回、S新1害/7救回，最大1077.016m；对HALF H新2害/0救回、S新1害/14救回，最大1608.542m；对MC虽无新公里差值点，仍可损失亚公里价值，不能称每点等效。

原害H组内ORDER给MC的alpha平均.715465，原3救回alpha平均.265089；S唯一旧害alpha.977850、33救回均值.848262。DE仅支持这组可见描述/学习在留组件条件下未正确区分部分好坏，病例仅少数组/场；不是特征完全不可辨识、噪声/参考错误或时序无价值已证。

|ORDER−O域|HARD平均误差差m|STRUCT平均误差差m|公里害H/S|
|---|---:|---:|---:|
|pure|+1.026385|-2.381994|28/1|
|mixed|+1.664988|+1.174867|0/0|
|zero|+38.076804|-73.934320|28/1|
|one|+0.000000|+0.000000|0/0|
|multi|+0.000000|+0.000000|0/0|
|eligible|+93.545673|-207.832813|28/1|

相对更早front base的旧害是另一统计：H原O旧害104点，GT新总130＝保留102＋引入28、解决2；S原75，GT新总74＝保留73＋引入1、解决2。不是把上述PG/MC34/1旧病例混成O旧害。完整原base变化及各臂/组件/纯混/支持域均在history-followup-accounting.json。

### 覆盖、关联及全部四属性

与各自178分组对应，位置点覆盖H99.395668%/S98.746861%、对象覆盖841/860与834/860，输出/重复902–61/866–32。原28控制、分组/score/raw-decoded、新增读出保持域、速度和航向全部原样；本试验没有关联或覆盖增量。raw与decoded分别列，mstsa是跨源人口，不混用微平均或all。

|前端/分组输出|all mPCA%|all mPFA%|跨源mPCA%|跨源mPFA%|
|---|---:|---:|---:|---:|
|HARD/raw|95.606441|16.063840|94.350241|17.815607|
|HARD/decoded|91.829743|7.163103|91.056951|7.325197|
|STRUCT/raw|96.318673|15.723956|94.965083|17.108027|
|STRUCT/decoded|91.346132|8.129612|89.884759|8.823268|

下列四属性是同评价器完整记账，非独立双实现；采用场等权目标均值，ME与MAE分开，SD仅residual_SD，不把有争议printed Eq29当作者ESD已知。各属性缺参考/缺预测及strict-complete null均保原件，不能拿不完整人口造作者公平名次。

|读出|属性/论文单位|ME|MAE|absolute_ME|residual_SD|
|---|---|---:|---:|---:|---:|
|HO_NN|latitude/1e-4 degree|1.069883|18.936355|8.071145|31.386669|
|HO_NN|longitude/1e-4 degree|-1.456312|20.083703|8.328821|32.410260|
|HO_NN|speed/kn|-44.288815|44.333638|44.288815|30.149626|
|HO_NN|heading/degree|-10.142827|100.243680|59.267116|109.189875|
|HMC_NN|latitude/1e-4 degree|1.174011|18.936557|8.068040|31.385759|
|HMC_NN|longitude/1e-4 degree|-1.349658|20.190733|8.453238|32.506718|
|HMC_NN|speed/kn|-44.288815|44.333638|44.288815|30.149626|
|HMC_NN|heading/degree|-10.142827|100.243680|59.267116|109.189875|
|HGT_NN|latitude/1e-4 degree|1.119774|18.958533|8.091368|31.403492|
|HGT_NN|longitude/1e-4 degree|-1.364165|20.153220|8.406767|32.480560|
|HGT_NN|speed/kn|-44.288815|44.333638|44.288815|30.149626|
|HGT_NN|heading/degree|-10.142827|100.243680|59.267116|109.189875|
|SO_NN|latitude/1e-4 degree|0.660524|18.844770|8.107490|31.031069|
|SO_NN|longitude/1e-4 degree|-1.190554|19.263756|7.297150|31.662504|
|SO_NN|speed/kn|-44.139408|44.183738|44.139408|30.051649|
|SO_NN|heading/degree|-10.052408|99.890724|59.014548|108.888704|
|SMC_NN|latitude/1e-4 degree|0.728399|18.792544|8.050098|30.985445|
|SMC_NN|longitude/1e-4 degree|-1.257018|19.179606|7.229733|31.570079|
|SMC_NN|speed/kn|-44.139408|44.183738|44.139408|30.051649|
|SMC_NN|heading/degree|-10.052408|99.890724|59.014548|108.888704|
|SGT_NN|latitude/1e-4 degree|0.698090|18.819801|8.078619|31.005678|
|SGT_NN|longitude/1e-4 degree|-1.249617|19.167264|7.207969|31.564553|
|SGT_NN|speed/kn|-44.139408|44.183738|44.139408|30.051649|
|SGT_NN|heading/degree|-10.052408|99.890724|59.014548|108.888704|

### 复核、原件、偏差和限制

主位置/人口/coverage/组件/主要CI/关联/尾部独立saved-output核验；原28臂exact、阶段和eligible与原O/MC绑定、新MLP用NumPy读权重重建，最大alpha差1.7881393432617188e−7，圆周动作/alpha保持域核齐。13fold/39fit/4992更新、同normalizer/BAG multiset/同抽样/初始化结构/全部29790参考绑定与原34–1害/3–33救回核齐。特征提取共享、没有重放优化器、四属性仍同评价器，不宣完全独立物理认证、真实Cov或作者复现。

[全部66公共原件与manifest](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261101_temporal_gate/194)共148622382原始字节，tar16110258B，数量/逐文件字节/包长度核齐：完整run、两个launcher真实命令/退出/stdout/stderr、全部T逐场/组件/主要区间/44臂、trainpopulation/folds/curves与4项审计、设备探针均保。private坐标/身份/权重/normalizer/抽样索引/大预测/原生缓存留服务器，不以公共摘要冒充完整私有输入。

相对194计划：实际样本/训练量/终点/审计均完成，无加步/seed、删点、最好步、换配方；193技术修复由独立194合同发布，历史失败不覆盖。只有原T开发/留组件后端，小样本少组、原前端见T/混组对应、真实主要K2、离线含向前/向后恢复；软件K3匿名不等真实K>2验证，0V/新域/在线端到端或可信同行排名。

## SQUEEZE_REPORT：SQ-TEMPORAL-GATE-194

消费本EXP/Idea及完整66原件，证据verified-T-exploratory。DE：同信息同容量的保顺序与去顺序没有可靠差异，新增阶段表示未超过简单MC完整风险；训练loss变化不能解释成时间语义机制成立。AI：门控可能未学到跨组件收益符号、粗四块表示不足或动作空间/参考关系仍有限；UN：真实未来机动、可观测性、参数训练条件、reference与原生报告时间对应，不能把当前阴性泛化到全部时序建模。

|深挖维度|证据及决策|
|---|---|
|强控/整体|S胜MEAN不足以胜MC，H还弱于HALF/O；不能只保一个头/一个指标冠军。|
|难例/代价|H旧害少6但救回丢3；S仍1害/救回丢5，且simple HALF可做更保守H但弱S；非新增状态估计可靠性被证明。|
|顺序作用|ORDER−BAG两头CI跨0、无公里点翻转，当前表示/预算下未获增量，不认所有时序信息无用。|
|发生条件|纯组/zero公里害保留，one/multi保持域0，少数组且标签对应未知；不退固定源优先、不能把错误全归关联。|
|技术/复现|193原因修复已实际GPU通过，科学/audit真实成功，公共原件齐；软件问题与科学阴性分别记。|

决策：**当前四块特征＋O/MC门控配方不晋级，不跑V/全300，不通过加步/seed/阈值小修续命；原178与MC/PG条件平均价值保留。** C68更广的原生时序—query相容性仍可发展，不因存在SSM先例或本次阴性否定。先回Mode C2.07/F303/E206、SR v4.8、OD v1.21；下一需利用本批反例分清可见运动条件的跨组收益符号、状态动作与组/reference对应，不重复已有端点/窗口/β小补丁；只有新的来源/证据支持具体替换后才独立Idea/READY。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-TEMPORAL-GATE-194
created_at: '2026-11-01'
input_experiment_record_ids: [EXP-TEMPORAL-GATE-194]
source_idea_id: IDEA-TEMPORAL-GATE-194
plan_version: '1.0'
claim_scope: original_T32_coarse_phase_gate_not_better_than_simple_controls_no_V
primary_result_with_uncertainty: ORDER_minus_MC_H_plus0.047849_S_plus0.330501_both_component_intervals_cross0
process_dynamics: 39fits_4992updates_fixed_final_0V_29790_bound_fold_rows
new_P: none
new_F: F303
new_M: none
new_E: E206
errors_heterogeneity_and_boundary_conditions: pure_zero_harms28_1_old_rescues_retained0of3_28of33_not_independent_cases
robustness_sensitivity_and_alternative_explanations: same_capacity_orderless_mean_half_MC_O_all_controls_no_order_increment_current_recipe_only
evidence_index: 66_public_originals_and_four_terminal_audits_private_server
open_questions: [observable_phase_information, cross_component_benefit_sign, future_motion, reference_alignment, richer_state_actions]
next_routes: deep-read_Mode_C_then_new_evidence_specific_intervention_no_automatic_run
next_skill: deep-read_Mode_C
```

