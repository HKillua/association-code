# C53：固定身份分区，单独检验每个query的贡献规则

2026-10-29，`IDEA-QUERY-CONTRIBUTION-CONTROL-20261029`，plan1.0。用户本轮授权同步挖掘候选、推进初步有效点及部分模块重构；本项只消费已完成的[Mode C1.32/1.33](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-132)中 **C53、P14/P11/P02、F212/M84/E110**。113负面不是本项的依据或前置。形式化设计先于本项任何MTAD执行；本项不是新学习方法或最终架构的选定。

## 1. 问题与来源，不把连接当作已知解法

**DE：** 114静态WS改善身份关联；115同样的普通mean_supported恢复在共同点MAE变差、RMSE与覆盖改善，且同首代表分层MAE也差。完整身份关系R并不规定每个query各成员的贡献W。**AI：** 按片段等权可能让某物理source的多片段获得更大总权，或让离query较远的有支撑成员与较近成员等权；这两个机制尚未在目标人口被隔离，不能先认定为115根因。错组、共享偏差、参考生成和均值损失仍是竞争解释。

复用完整来源分析，不新增深读计数：

- [ASF，NeurIPS2025 AP§2–5](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/xfer_fusion_asf_source_analysis_20261028.md)：正文式1–10、Table4、独立附录Tables5–12和作者`a2_fusion.py`。同物理patch内按**source**局部融合；共同query和来源粒度必须先建立。作者attention不等方差或健康概率，SCL另加监督，部分组合比单源差。本项不迁attention/SCL、不冒称ASF复现。
- [MeanTraj，TSP2026 AP§3–6](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/meantraj_tsp2026_full_analysis_20261029.md)：Eq1/15/16、Alg1–7、TableIII与技术补充。完整轨迹关系、存在域和状态质心分开；长度近似和输入重复会改变权重。FM1与FM2有取舍，批处理权限亦混杂；本项不迁完整Fréchet集合算法或全局最优保证。
- 目标代码`ordinary_fusion`中的interval-support均值、`scene.evaluate_full_scene`固定首组对应，以及115真实原件，界定当前运算和合法控制。

**迁移A/B/C：** A=原A0/WS+片段等权mean_supported；A′=保持原物理Q64、R和首代表，显式呈现support/nearest/source的预测接口，本身不改预测；B=透明的query贡献规则。ASF证明源内局部跨source机制可用，MeanTraj给领域内对象/权重分解先例，但没有已核的“本项source-balanced或supported-nearest解决MTAD”的独立C，记UN；不无限检索、不据缺C否定。来源原方法的质量和目标效果分别判断。本项首先核适配接口/简单规则有没有价值，不把普通权重包装为跨域核心B已成功。

## 2. 固定人口、信息与八个控制

只读原115使用的、114事前固定seed20261029的**V32**；不得按115收益、组件、尾部另挑。原056逐片段Q64插值/OLS状态、support、nearest_seconds、source及114的A0/WS membership；原115的完整目标artifact/reference/valid只用于绑定与评价。不是原生CSV点，不读ZIP、权重或封存测试。

R取A0和WS两种已存分区；每种分区分别比较四个W，所有八臂都报告：

| W | 预测规则 | 固定边界 |
|---|---|---|
| MEAN | 原`ordinary_fusion(...).mean_supported`，不改一个数值 | 与115相应完整artifact exact重放，是本次必要控制，不重跑原任务 |
| ANCHOR | 始终复制各组首成员全部Q64状态 | 简单强控制；相同R、首代表、fallback和完整输出，不按误差选源 |
| SOURCE_BALANCED | query内有support的物理source数为S，每source有m_s个supported成员时，每成员权`1/(S*m_s)` | 不称独立信息、真实provenance去重或误差最优权重；同source内部仍等权，可能放大差源 |
| NEAREST_SUPPORTED | 有support时，仅从supported成员中选nearest_seconds最小者；并列按原node顺序 | 不是现有全成员`nearest_observation`；时间距离不保证正确/低误差 |

SOURCE_BALANCED的lat/speed沿线性加权，经度/航向沿原圆周规则；合向量≤1e−12只对对应角度回退首anchor。两新规则在query无supported成员时复制首anchor整行OLS，**不拒绝输出、不缩人口、不删singleton**。所有组成员顺序及分区不变，每组输出一条；本项W不能减少重复组。sources只作为等价类，不依来源名称硬编码好坏。

新接口只接显式物理数组和groups，不能接fragment dict、identity、label、reference或误差。先固定八臂预测，再调用原评价函数。R改变的效应按相同W比较；W改变的效应按相同R比较，不能把二者的总变化都归B。

## 3. 预先的指标、归因和解释出口

主比较为同WS中的SOURCE_BALANCED−MEAN和NEAREST_SUPPORTED−MEAN，两个比较全部保留，不按哪个涨分决定报告。主估计量是共同有效position点的米制MAE/RMSE；P95/P99、场景等权及原共享identity组件全部报告。与ANCHOR简单控制比较，避免普通更强的单源解释遗漏；同样比较A0下各W和相同W下WS−A0。报告所有正/负场景、组件，不把pair/query当IID样本，不给独立泛化或同行胜负结论。

每臂完整目标分母/目标无参考/预测组/重复/未匹配、position覆盖、prediction_without_target、observed/missing、固定真目标输入support/远距分层均保留；每对共同集及gained/lost独立计算。**同R四W的population、对齐有效mask、首代表、覆盖应exact一致**；不一致是实现/输出故障，停止，不称收益。固定R的raw/decoded关系完全不变；引用114已核结果，不重新发布关联实验。

机制描述仅由输入算：每group-query的supported成员/source数、来源成员数不均、两新W与片段均值权重L1差、NEAREST_SUPPORTED相对全成员nearest在有支持query的选择变化及无support回退。输入条件与完整计数用于判断该人口对机制是否有辨别力；本轮位置按既定真目标support/远距及同首代表分层，不承诺新增权重条件的效应分层，更不能由相关宣布唯一因果。仅SOURCE_BALANCED变化时可归该透明规则的干预效应；不是校准/学习/完整ASF贡献。新恢复对单正权成员exact复制，其余角度作圆周运算；与旧MEAN数学相同的权重可能有微小浮点或等价角表示差，不能把这些变化当机制证据。位置收益不等完整四属性系统收益，旧speed单位不据本项解释；不宣称论文有符号ME=MAE。

- 有来源不均且规则改善MAE/RMSE/尾部、不丢覆盖：支持当前固定R下继续研究query贡献；仍需成熟同行与新开发材料验证，不自动扩量/开训。
- 无足够来源不均或权重实际不变：该人口对来源粒度假设辨别力有限，不能称整类机制失败。
- nearest改善、source-balanced不改善：支持时间选择的透明控制，不能归学习可靠性或无条件去重。
- 仅优于MEAN却不优于ANCHOR、或均值/尾部有冲突：如实保留取舍，重新比较新增价值，不叠网络救分。
- 两规则无净收益：关闭本轮配方；保留114关系收益与其他候选，回流矩阵，不自动改权/重试。

## 4. 实现、资源和终点

保留`ordinary_fusion`、旧模型、`evaluate_full_scene`、113/114/115冻结代码与数据。新增预测接口、一个八臂CPU分析入口和必要回归检查；规格见[query贡献接口](../specs/query-contribution.md)。复用原115的metadata/指标/共同集工具，不另造评价协议。

独立116 attempt-1，一次固定32场，CPU≤4、RAM4GiB/地址空间≤8GiB、600s advisory；0GPU/模型构造/前向/优化/新normalizer/阈值。正式运行前校验原114/115完成且原人口/路径/分区/均值artifact一致；新run拒绝覆盖、失败保全、无自动重试/扩量/后继训练。实际成本记run与同一launcher，原身份/坐标/weights/private artifacts留服务器；只回传计数、误差、命令/退出和匿名场景/组件。输入缺失/不一致、非法状态/分区、非有限或资源失败终止，不换规则或放宽容差。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-QUERY-CONTRIBUTION-CONTROL-20261029
idea_id: IDEA-QUERY-CONTRIBUTION-CONTROL-20261029
created_at: '2026-10-29'
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.32
selected_C_id: C53
source_ids: [E110, AP-XFER-ASF-20261028, AP-MEANTRAJ-TSP2026-20261029]
source_P_ids: [P14, P11, P02]
source_F_ids: [F212]
source_M_ids: [M84]
source_E_ids: [E110]
research_question_and_hypothesis: '身份R固定时，显式来源粒度或时间支撑的贡献W能否改善位置，且不改变覆盖/重复；目标机制未知，允许反驳'
testable_predictions: '第3节两个主比较、八臂强控制和输入机制指标；不预填效果方向'
design_and_controls: '原事前V32；A0/WS x MEAN/ANCHOR/SOURCE_BALANCED/NEAREST_SUPPORTED；MEAN exact重放115'
sample_materials_or_experimental_units: '固定32开发场景及实际覆盖共享身份组件、全目标/片段；Q64缓存不冒称原生点'
primary_outcome_and_estimand: '相同R共同有效位置点米制MAE/RMSE，完整coverage/duplicates约束'
secondary_outcomes: '第3节尾部/场景/组件/支撑/机制和代价；不报未核speed性能'
analysis_plan_and_uncertainty: '完整确定性配对开发分析；无IID显著性或独立测试；全部两主比较与负结果保留'
resources_time_cost_and_risk: '第4节一次CPU4/RAM4GiB/AS8GiB/600s advisory，固定32，0训练/模型/GPU'
ethics_safety_privacy_or_compliance: 'cached_reference_read=true且仅评价，预测无ID/ref；private原件留服务器，T_test封存'
stopping_and_interpretation_rules: '第3–4节；无自动扩量/重试/训练'
source_transfer_id: TRANSFER-QUERY-CONTRIBUTION-20261029
transfer_condition_commitments: 'Q64/support/source存在可核；source独立/真provenance与可靠性UN；独立迁移C未取得'
adaptation_vs_method_attribution: '接口A→A′保持数值；各透明W相对相同R/信息/输出的MEAN干预单独计，不冒称完整ASF或学习B'
status: exploratory-minimal-validation
evidence_status: partial
evidence_locations: '本页§1来源、Mode C1.32、114/115原run；执行前尚无116结果'
missing_evidence: [query_weight_target_effect, credible_peer_comparison, independent_exact_transfer_C, reference_generation]
next_skill: record-experiment_after_actual_execution
```
