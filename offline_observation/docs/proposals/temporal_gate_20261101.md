# 193：原生阶段信息是否改善单侧恢复的任务风险

2026-11-01，IDEA-TEMPORAL-GATE-193/plan1.0，research-pipeline-contract v1.0/workflow v2.0；Mode C2.05/C68、P13/P14/F295/F298/F300/F301/M122 AI/E198/E201/E203/E204。Pattern5受控最小适配。原178历史风险选出候选后，MC只用全段报告均值；T33单片段害无其他源可换，时序信息是否有用未证。192已否定密度解释，此实验不做时间加权/端点位置拟合/β或窗口扫参。

## 来源与迁移条件

完整FlowState AP说明时间编码与query语义须共同成立、单变量/scale与条件稳定证明不能直接移植。此处先建立A′：多通道运动的无绝对坐标/身份的真实时段描述，直接用原T任务风险监督；不执行其S5/FBD/预训练，不称FlowState作者复现或B已验证。已有GPTR/Go-RIO原生时序条件及ICML两阶段读出来源均复用，充分109不增。137/156反例要求直接核目标风险，不能拿内部重建loss证明恢复正确。

## 最小干预与对照

固定178分组、阈值/前端、每fold旧风险选中片段及源等质量tie；同query的O（原OLS工作状态）与MC（同原生端点报告均值传播）为两种**时间状态估计**，不是两个固定传感器。新标量alpha∈[0,1]在它们之间做原纬度线性/经度圆周读出。任意K仍由原无源名的候选与质量权处理，不加固定源优先级/双源贡献配方。

三组相同32→32→32→1/2145参数小MLP，输出sigmoid alpha，末层零初始化＝共同HALF控制：MEAN只含12全段/query统计，新20维置0；BAG含四真实时间等长区间的5维物理描述，在原选中节点质量权池化后，以同一四块特征数值规范排序删除区间顺序；ORDER保原四区间时间顺序。BAG与ORDER含相同块特征multiset/网络容量/训练曝光，差别是块顺序。MEAN比较新增阶段信息总体价值，ORDER−BAG才检验时序排列的条件增量。不是完整原生序列编码或严格多层时间等变证明。

每块：有效报告均值相对全段报告均值方向的平行/垂直分量，位置端点secant与全段OLS差、报告均值与该secant差、两者可计算flag；未知置0并保flag，不丢片段。公共12维为真实传播时距/before、span/n/fit残差、mu与OLS模及差/报告离散、组node/source数、O–MC位置距离。选中节点按原质量权池化，所有量只有可见输入/候选，没有reference/identity/源名/绝对经纬度。块排序是同源时间信息对照，不是源编号排序。

只全组原生支持之前/之后且count0改位置，其他zero/单/多支持、全部速度航向与覆盖/raw-decoded维持O。原189全部28臂完整导入，加HALF/MEAN/BAG/ORDER×两前端×Mean+NN共44臂；原178、188各强控均保，非冠军拼接。alpha全部保存，不按T/V选阈值或剔除难点。

## 数据、学习与信息隔离

原178 T128/原T32十二组件leave＋full共13fold；只原T32/860目标评价，**0V/封存test**。复用183已完成的额外96场原生事件、151已有motion/evaluation索引，不重新打开CSV/archive，不新生成标签。每fold旧178风险模型用于本fold所有训练场候选；新gate及normalizer只用非held组件，防止用full风险模型给leave训练造额外泄漏。T32对应held评价直接绑定原178选源/189 MC数组；前端早见T标签，因此不是全链LOCO/独立确认。

块特征跨四块共享同一5维标准化统计，MEAN/BAG/ORDER共享同fold normalizer，MEAN在标准化后将阶段维置0，避免标准化差异混入顺序控制。39个新fit，每头128步、batch1024/同fold完全相同seed20抽样/初始权重；Adam .001/wd1e−4/clip1、固定末步，不选最好步/seed。共4992 GPU更新，0新前端/ridge/teacher或native似然。训练目标是球面读出后的平滑haversine绝对米误差/1000（sqrt加1e−24作可导数值平滑，零误差影响约1.3e−5m）；不是学几何误差proxy或内部自洽。原first-anchor/first-claim标签、全部有效query含混组都保，两个前端视图各.5权；query相关/原参考物理准确性未知。

## 指标、代价、停止及解释

主要ORDER−BAG、ORDER−MEAN、ORDER−MC、ORDER−O完整T共同MAE；两前端分报、12组件/2000seed20描述区间、所有组件/去最大。次要全部44臂MAE/RMSE/P95/P99、四属性/论文指标、coverage/重复、raw-decoded、纯混/单多/zero、相对O及base新增>1km害/救回/最大害、原34/1害与3/33救回实际去留。原451/344 main eligible绑定。训练loss下降或局部旧害减少均不证明机制。

若ORDER不能在匹配BAG/MEAN及MC整体/尾部保值，不扩大当前配方；BAG/MEAN若同样有效只支持任务读出或块信息，不冒认顺序收益。仅T条件候选，无自动V/全300/新seed/193重投或194。即使负结果也不否定所有SSM/连续表示，须按实际对照缩小结论。

新science193一次GPU0 UUID GPU-c5a00c37-f181-90b3-f220-be6e7457f473（现场已有计算，按规则兼容共卡，不抢占/触他人），GPU新增内存≤1GiB，CPU8–11×4/RSS8GiB/3600s advisory；原152占GPU2/3预算不改。成功后一次新audit193/CPU16–19×4无GPU/1800s advisory。独立保存主位置/人口/关系/尾部，四属性同评价器完整记账；新模型权/normalizer/fold/sampling预算、NumPy MLP与alpha/几何/保持域/全部旧控制及native块描述绑定。未重放优化器或认证物理Cov。失败保全，不换CPU/配方或自动重试，终态公共原件脱敏回传、private留server，Record→Squeeze→Mode C。

## 实现验收

新增native_phase_gate.py（无label特征/读出）、training/temporal_gate.py（匹配三组、物理loss）、temporal_gate_pilot.py与closeout；不改原活动模块。检查真实list/K3匿名/tie、日期线圆周读出/alpha两端、单位与时间平移/旋转、BAG对块置换不变但ORDER敏感、原节点MC及组O绑定、三个同初始HALF与同抽样/非heldnormalizer、物理loss有限梯度、保存模型NumPy重建与T-only入口。通过后才READY，软件不是MTAD效果。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-TEMPORAL-GATE-193
created_at: '2026-11-01'
idea_id: IDEA-TEMPORAL-GATE-193
plan_version: '1.0'
source_matrix_id: MC-MTAD-20261101-v2.05
selected_C_id: C68
source_P_ids: [P13, P14]
source_F_ids: [F295, F298, F300, F301]
source_M_ids: [M122_AI]
source_E_ids: [E198, E201, E203, E204]
research_question: native_phase_information_and_order_increment_for_fixed_one_sided_task_recovery
hypothesis_and_mechanism: phase_query_information_may_help_choose_existing_motion_estimators_no_assumed_root_cause_or_SSM
sample_materials_or_experimental_units: original_T128_training_T32_860target_evaluation_0V_no_sealed_test
primary_outcome_and_estimand: ORDER_minus_BAG_MEAN_MC_O_common_point_T_MAE_each_front
analysis_plan_and_uncertainty: all_44arms_components_tails_coverage_descriptive12component_bootstrap
resources_time_cost_and_risk: 39fits_4992GPU_updates_one_shared_GPU_science3600s_one_CPU_audit1800s_advisory
status: pilot
evidence_status: partial
missing_evidence: [target_phase_and_order_increment, full_risk_and_tail]
next_skill: record-experiment
```
