# 关联与对象输出：先核同信息信号，再建立合法对象假设

2026-10-29，设计1.0。按项目skills及[完整采纳流程](../research_workflow.md)消费[局部Mode C §7](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/association_output_action_matrix_20261029.md#7-2026-10-29-review后修订留出一致性先与完整pair几何对齐对象化保留拒配)。这是独立研究设计，没有新实验/作者复现/完整精读计数；真正发布与执行只认HandOver。当前不因提案存在自动启动。

## 1. 已知问题和必须修订的解释

- 114的WS是signed-margin sum凝聚，decoded宏F1 .851820273，raw不变；全组静态score是必需强控，不能只比原anchor直连。
- 110容量已完成；8202同源真边全无共同Q64 support，不能把无交叠直接判负或再跑容量诊断。
- 全V1068 duplicate是额外首anchor组的评价计数，纯断轨/混组/误锚比例未知。对象化可能改善，也可能损害，不能预报“救回1068条后半轨”。
- target support用真身份构造，只能事后评价；预测组support可作合法输入，两者不混。
- 原OA-C6草案的两成员整片段留出残差逐点就是pair距离。多成员共同support的局部欧氏LOO-RMS也可由全部pair距离平方和重构。比较粗score箱不能证超出同信息静态模型的条件增量。

这些是代码/已完成记录/条件代数所支持的边界，未证明现有错误的物理根因。当前有价值的问题是：怎样在合法多fragment对象空间中利用被旧摘要压缩掉的证据，并把表示、条件运算和搜索预算贡献分开。

## 2. 最小可实施Idea：固定候选上的几何信号与恒等负控

领域为异步多源航迹关联；主模式Pattern5系统/表示审查，连接`OA-C6r`。它回答可见证据是否值得进入下一小模型，**不改变分区、不产生新恢复预测**。

### 输入、权限及固定人口

原114事前V32/9组件，全部原节点、056全部i<j分数/θ及缓存Q64位置/support、原WS membership。manifest/场景顺序与原118/121绑定，不重读ZIP或reference数组，不重算056分数/原分组/训练。缓存`native_states[N,64,4]`是Q64全窗预测，不能称真实native观测；使用原离线未来信息权限。

预测特征API只接受`states/support/pairs/scores/θ/source_equality/groups`；不接身份、labels、target、真实target support。先在单场写完并冻结所有候选及特征，再单独读取完整pair标签或节点身份进行评价。少量元数据如原汇总的样本/组件绑定须明确记为既有评价元数据使用，不声称整个进程无评价材料。T侧材料此次不读、不拟合normalizer/β/门限；V32已暴露，是开发诊断，非独立确认。

### 无标签候选与空证据处理

沿原OA-C6：保留原WS所有组，含singleton；每个至少二成员S再用组外平均`score_ij−θ`最高的两个节点形成`S∪{j}`，不足两节点全保留、并列用原节点序。规范成员集合后只去除完全相同候选。候选去重不代表物理观测去重；不删singleton、noQ、mixed或难例。所有原节点/组/候选、候选成员覆盖、大小/来源组成和无信号原因完整计数。

**所有候选仅为假设库，并不允许同时作为最终输出。** 不做纯度筛后报告“完整召回”；候选规模变化也不能充当新解码器效果。

### 三种同候选统计和负控

1. **SCORE_STATIC**：内部全部pair margin的sum/mean/min/max、中位数、负比例、候选大小、同源/跨源pair数。保留全部量，不只mean-margin分箱。
2. **PAIR_GEOMETRY_STATIC**：对每候选全部内部pair保存逐query球面距离及共同support，输出与LOO完全相同query权限下的pair共同support距离中位数/P90/RMS、支持点数和缺失分布；同时保存全Q相同统计。此为同信息静态控制，不是现有22维均已含这些统计的主张。
3. **LOO_GEOMETRY**：逐成员j整条留出，仅在`support[j,q] && any(support[others,q])`处，按既有短弧经度/纬度均值预测j；汇总残差中位数（主）、P90/RMS和有信号成员/query比例。无support为null，不赋大误差/负标签；单成员不可定义。禁止拟合被留出的j、从truth挑query或用其reference校正。

两成员候选：逐点核`LOO_haversine == pair_haversine`（只允许数值舍入容差，记录最大差）；共同support中位数等统计亦应一致。这是必需恒等负控，**不能将原22维全Q中位数换成共同support中位数的收益称条件状态收益**。

多成员候选另加**ANALYTIC_PAIR_RECONSTRUCTION**：固定query处实际有support的成员集合B_q，m_q≥2，用同一局部北东图内的欧氏点，核

`mean_i ||x_i−mean_(j≠i)x_j||² = Σ_(i<j)||x_i−x_j||²/(m_q−1)²`。

图原点/纬度只由该候选合法位置确定；两侧相同。这个局部Euclidean控制仅验证信息恒等，不替代主球面/圆周算子。球面/中位统计的差异必须分别报告，不能把非线性统计差直接称新运动信息。

### 评价、主要估计量与完整边界

特征完成后用身份给候选标pure/mixed；全部singleton、pure/mixed、无标签/无运动信号均保留各自分母。pure是候选内同身份，不等reference精度、物理健康或无绝对偏差。所有身份、坐标和候选大数组留服务器。

主结果为**候选可评覆盖、两成员恒等及多成员解析控制最大偏差、按场景/共享组件的固定残差pure/mixed排序AUROC**。定义pure为正，负残差为评分；ties=.5，单类层为null，不能删掉再宣全体效果。分别给大小2/3/4+、单源/跨源和共同support比例；保留原score完整分层。既有粗margin箱仅描述，不宣排除全部静态解释。

本轮**不以“LOO AUROC高于SCORE_STATIC”证明条件运算有效**：两者表示信息不同且无同容量学习器。它最多支持某种几何统计值得加入输入。真正方法比较须另发T监督的同信息静态组头A′与partial头A′+B，匹配全部pair几何/support、候选/求解次数、参数和成功更新。若静态头已解决，则保留有价值的轻量表示解法，不预送收益给B。

不产生新partition/预测，故原raw/decoded、位置误差、覆盖/重复均只是固定背景，不能列“新增方法提高”成绩。若以后新partition实做，必须评全部pair关联、全目标位置/尾部/coverage/duplicate及首代表变化；不能拼当前融合最佳臂的不同人口数字。

### 一次预算、终点和解释出口

建议一次原V32、CPU≤4、RSS4GiB/AS8GiB、600s advisory；0GPU/模型/前向/训练/ZIP/reference数组/测试。固定32终点，逐场写必要脱敏汇总、完整private特征和身份后评价表。墙时建议不自动杀或扩训练；输入失配、非有限或恒等超明确数值容差视失败，保全现场、不改人口/配方自动重试。数值容差须在实现规格中按实际坐标公式冻结，不能执行后放宽。

- **A：只有pair统计重组带来可辨性。** 支持表示A′，不支持条件网络归因。下一次可直接比较同信息静态/条件小头，不增加无限同类诊断。
- **B：可评覆盖过低或无跨组件可辨性。** 当前同support残差不足；转向native事件/对象假设，保留无Q正例，不能把它们当负例或增大网络补救。
- **C：来源/组大小/平行异船方向冲突。** 记录竞争解释与具体人群，不宣共享偏差根因；在新Idea中区分相对几何/共同趋势/来源误差。
- 恒等检查失败先定位实现/坐标误差，不能作为方法新信息。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-OA-SAME-INFORMATION-SIGNAL-20261029
idea_id: IDEA-OA-SAME-INFORMATION-SIGNAL-20261029
plan_version: '1.0'
created_at: '2026-10-29'
source_matrix_id: MC-ASSOCIATION-OUTPUT-ACTION-20261029-REVIEW-R3
selected_C_id: OA-C6r
source_P_ids: [OA-P7]
source_F_ids: [OA-F15, OA-F17, OA-F18, OA-F22]
source_M_ids: [OA-M7]
source_E_ids: [OA-E15, OA-E17, OA-E18, OA-E22]
research_question: 既有score和22维摘要之外的组几何可辨性，是否只是完整pair与support统计的表示变化
hypothesis_and_mechanism: 同信息pair几何可解释部分LOO统计；剩余信号仍可能只是非线性表示，条件动作增量待同信息训练比较
testable_predictions: 两成员逐点恒等；指定Euclidean多成员RMS重构；全部候选信号覆盖与分层pure/mixed排序
design_and_controls: 本文SCORE_STATIC、PAIR_GEOMETRY_STATIC、LOO及解析重构；不改分区或状态
sample_materials_or_experimental_units: 原V32/9组件完整节点/候选，056与114冻结cache；开发诊断
primary_outcome_and_estimand: 恒等最大差、信号覆盖、固定残差候选AUROC及逐场/组件异质性
secondary_outcomes: 所有null与来源/大小/support分层、候选数/状态求解数、程序和launcher时间/RSS
analysis_plan_and_uncertainty: 场景/共享组件为单位；不作候选IID显著性；V无拟合调参；null保留
resources_time_cost_and_risk: 单次CPU4/RSS4GiB/AS8GiB/600s advisory/固定32终点，0GPU训练与reference数组
ethics_safety_privacy_or_compliance: 特征先冻结再身份评价；大坐标身份留服务器；T_test封存
stopping_and_interpretation_rules: 本文A/B/C；错误保全不自动重试、扩量或训练
status: diagnostic
evidence_status: partial
missing_evidence: [implementation_and_pre_execution_numeric_tolerance, actual_target_signal, matched_same_information_training, full_method_effect]
next_skill: record-experiment_after_actual_execution
```

## 3. 对象假设路线：可发展的具体方案和未满足条件

这是`OA-C5r`的独立后续设计，**不并入上述诊断，不发布READY**。源机制分别为Draves有限假设/排他、MeanTraj集合对应/数量及TPMBM未配分支；未取得相同机制在匿名海上多fragment的独立迁移C，如实保留缺口。

**A**保留原WS partition；**A′**建立同源fragment有向可衔接候选及所有singleton，再构成有限对象假设库；**A′+B**才用明确集合目标或状态似然选择互斥对象解释。允许换整体骨架，无需把WS冻结为最终起点。候选库/搜索本身的收益先单独比较，不能冒充联合推断收益。

必须先建立的条件及可行做法：

| 条件/具体实现 | 当前状态与最小测量 |
|---|---|
| native时间和位置 | `motion_evidence.native_events`及`source_events`原生接口存在；当前056 `native_states`不是该接口。需独立native导出合同或确认既有真实事件缓存，保全部原节点/事件ID/时刻；不能用Q64首末support猜端点。 |
| 有限链接且不偷用身份 | 先用真实native末/首时间确定方向。最小假设库可保每片段时间可向后的两个最近连续性候选及singleton；排序根据原生端点恒速外推距离，平局原节点序，不读标签。候选缺口/事件覆盖与same-source真链接召回仅事后评价；候选召回不足则修改候选设计另发Idea，不隐去被截掉真链。 |
| 同源重叠与无Q | 无共同Q不能判真native不重叠。真实同时段冲突设“不能同时进入该单对象假设”的可见约束，同时保拒配/独立对象；也不预设重叠必不同船，需记录传感器是否允许重复轨这个模型假设。 |
| 身份/标签 | 可在T建立连续性误差/拒配规则并冻结到V；若T/V共享组件，披露非身份独立。identity只用于监督/评价，绝不用于候选并段、筛纯组或修复预测。当前尚未指定T真实native人口及训练配方，因此不冒称合同完整。 |
| 合法事件与排他 | 同一原event ID不能被两个同时选中的对象重复解释；不同原节点数值相同不擅自合并。一个fragment至多归一个最终对象，singleton/拒配仍保留，所有未选/未覆盖原节点计人口。 |
| 输出与评价 | 候选纯度、最大可覆盖同身份链/冲突、完整目标漏报/重复、全关联及位置尾部共同报告。继续用主评价首anchor固定对应，同时另建明确集合层诊断；不能把新Hungarian匹配结果偷换成原主表。 |
| 状态与数量成本 | 先透明固定恢复比较候选容量；再独立选择MeanTraj ED集合成本或合法似然，不凭空造Cov/检测率。集合c/p等只在T冻结或明确经验固定值，不能从V挑有利值。 |

近期可执行条件尚缺**真实native缓存/导出范围、T侧拟合及拒配协议、完整对象目标参数**；因此当前只完成源机制—条件—可执行方向设计，不声称对象化已成熟或能消除1068重复。上述缺口有具体建立路径，不因“已有轨迹链接/集合先例”关闭方向，也不把缺口变成无期限完美理论门槛。待A′有合法输入和预算即可发布一个有界候选覆盖与冲突pilot，不须先实现完整PMBM。
