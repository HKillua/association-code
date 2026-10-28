# C53-Pa：现有合法片段的身份关系与单位容量输出是否兼容

2026-10-29；`IDEA-FRAGMENT-IDENTITY-CAPACITY-20261029`，plan 1.0。复用[C33旧设计](mtad_fragment_relation_diagnostic_20261028.md)的条件数学和边界，收窄为**同一056 V300现成关联缓存**的零模型诊断；不读原ZIP/量测/reference，不重算scores，不改107/108。用户已授权四问题的小范围验证，本文件提供其中一项具体合同；实际领取/状态只在HandOver维护，本页不自行READY。

## 1. 从矩阵到可检验命题

来源为[Mode C1.28 §33](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-128) C53-Pa、P08/M35；C33旧F86/F100及MH-T2TA/TMTAFF的任务条件保留。当前任务允许同源不同fragment为同一A-B身份，但部分近邻采用每源每目标每窗一轨。若评价要求全部跨源同身份边、两源片段数为a/b，真边ab，一一硬匹配最多输出min(a,b)，条件召回上界为`min(a,b)/(ab)=1/max(a,b)`。

**H1**：当前有效输入节点人口是否存在ab>min(a,b)的identity×source-pair，以及覆盖多少identity、正边和场景？**H2**：合法正边中有多少`joint_support_queries=0`，即没有共同Q64支撑；是否同样存在于单位容量受限对象？这不代表原生时间绝不重叠，只表示现有Q64候选支撑无交点。

反例/范围：该界只约束**未经并段/闭包的双源单位容量硬边输出**，不约束软矩阵非零项、连通backbone再闭包、预合并/逐窗单轨、一般多对多身份partition或融合误差。当前anchor_direct允许多节点组，**不是单射**；本诊断不证明它的错误由容量引起，也不反推MH-T2TA作者真实人口/成绩错误。多于两source按无序source-pair分别计算，汇总是这些独立双源约束的乐观上界，不宣称一个全局多源匹配同时达到。

## 2. 输入与真实人口

仅读固定056 `association/final-scores.npz`、原association `summary.json(.gz)`、现manifest `V_select`及component映射。原路径及保存证据见[接口证据 §5](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/target_interface_evidence_20261029.md)。schema为scene_ids和每场node_ids、sources、pairs、scores、labels、joint_support_queries；无真实A-B字符串时从**完整i<j二值labels**恢复匿名等价类，每条label必须与该等价类共组关系完全一致，否则失败，不能修标签。

预检全部V300场景顺序、node ID唯一/非空、完整规范i<j候选、scores有限、labels∈{0,1}、joint-support为0..64整数；原summary final1000/θ=1.7320103112387182、逐场raw混淆计数与人口一致。全局N24,599/E4,439,092/positive24,675是已知绑定线索，仍核逐场。不得仅核总数/忽略错序。scores仅用于输入绑定，不参与容量计算或identity选择。

这是**056清洗后有效输入节点**的人口，不是全部标签声明9,100目标；没有输入的目标、quarantine丢节点及原始label时间区间不在score缓存中，明确不补造。保留所有输入节点（含singleton/单source identity）、所有source-pair的零真边对象、无pair场景；不因无共同支撑或短片段删除正边。不读T_test，不解析node_id获得身份/位置或选择样本。

## 3. 运算、结果与统计定义

1. 由正标签构造连通等价类，再对**全部候选**逐条核`same_component == label`。先验证完整候选与二值标签；非传递真边、负边落在真类内部均失败。匿名identity按其最小原node index编号，原source仅作相等类，公共输出改source_0等匿名名，不输出MMSI/node字符串。
2. 对每个identity保存节点数、各source片段数、同源真边/异源真边、无共同Q64支撑正边计数。对场景每个无序source-pair及每个identity（含a=0/b=0）保存a/b、T=ab、B=min(a,b)、U=B/T或null。单source场景仍有identity表与完整场景行，不假造另一source。
3. 逐场保存T_s=ΣT、B_s=ΣB、受限identity/source-pair数与正边份额、no-joint-support的同源/异源正边分母。源对分层、场景和原manifest共享A组件分层均保留。每个源对0正边的U=null，零分母不记0/1。
4. 汇总micro上界`ΣB_s/ΣT_s`，以及U_s有定义场景的等权均值、其实际场景数/300；另列全部场景均有定义时才存在的strict场景均值。按identity等权的条件上界与数量单列，不能与pair-weighted混名。**不叫mPCA/mwPCA**：上界的双源匹配条件和权重不等实际paper预测量权重。
5. 自检：同/异源正边合计=全部positive，源对T合计=异源positive；B≤T；equivalence/singleton/empty人口一致；no-joint-support正边不超过其正边分母。每场原整数及全人口汇总齐备后结束，所有“受限”仅指上述条件数学，不是observed model错误。

主要estimand：完整V300有效输入中受该**条件硬输出限制**影响的identity/source-pair、正边和场景份额，及其direct-edge recall上界。辅助estimand：同源多fragment构成和无共同Q64支撑正边。有限固定开发总体、共享A身份，不做IID置信区间或泛化宣称；不从现有结果挑场/换阈值。

## 4. 实现、资源与停止

入口`tools/analysis/fragment_identity_capacity.py`，仅Python标准库/NumPy；不导入模型/torch或读取物理cache。输出新目录`run.json`、`scenes.jsonl`（含匿名identity/source-pair完整计数）、`summary.json`及失败进度；禁止覆盖已有目录。与另一C53顺序诊断共享只读原文件，无新cache写入。

一次CPU≤4/RAM预算4GiB、Linux AS≤8GiB；0GPU/模型/前向/优化/normalizer；300场逐场读取，不保留全E的九份排列。暂估10min advisory，仅记录而不超时自动扩量/重试。当前只运行必要的微型软件测试，不运行真实V缓存。

文件缺失/scene错序/完整候选或等价性不符/summary绑定失败/非有限/无效support计数/自检失败：保全FAILED和已完成场景，不换数据、不修标签、不重算scores、不自动重试。输出目录已存在时拒绝而不改原产物。无受限对象、T=0或无共同支撑是合法结果，不是程序失败。

|实际结果|只支持到哪里|可发展的方案与价值检验|
|---|---|---|
|存在受限identity/source-pair|本输入的全真边任务不能由声明双源硬容量完整表达；给出真实人口/分母|合法身份partition/局部先合并/明确backbone闭包可建立A′，比较完整身份/覆盖/物理误差与成本；不自动称首次或认定当前误差根因|
|全部相关对象a,b≤1，或任务输出为合法簇闭包|关闭本人口的对应强容量命题，不追求补抽“冲突样本”|保留C53-Pb条件状态与输出损失问题，容量不再作理由|
|无共同Q64支撑的正边存在|Q64共同支撑不是全部身份真关系的必要条件|把R_identity与W_query分开，时间不重叠的identity不能仅凭mask当负例；不能据此认定所有远时fragment应合并|
|缓存/等价性无资格|材料不能支撑匿名identity计数|保留失败/缺口，另定合法导出/绑定；不从原node字符串猜真值，不开新前向补齐|

本项是问题存在范围/输出空间的资格诊断，**不是方法效应，也不自动开模型训练**。用户允许初步成功后继续小验证，具体后继仍须根线程按结果、边界与新合同安排。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-FRAGMENT-IDENTITY-CAPACITY-20261029
idea_id: IDEA-FRAGMENT-IDENTITY-CAPACITY-20261029
created_at: '2026-10-29'
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.28
selected_C_id: C53
parent_idea_id: IDEA-MTAD-FRAGMENT-RELATION-DIAG-20261028
source_P_ids: [P08]
source_F_ids: [F86, F100]
source_M_ids: [M35]
source_E_ids: [E49, E51, E53, E54, E102]
research_question: '当前有效fragment身份全边任务在什么人口上受双源unit-capacity硬输出限制，哪些合法正边无共同Q64支撑？'
hypothesis_and_mechanism: '同identity双源a/b给ab真边与min(a,b)硬容量；身份关系不等query贡献；当前anchor不受该单射界'
testable_predictions: '完整等价labels可恢复匿名输入identity；若ab>min则条件U<1；预合并/簇闭包/单轨可关闭该界'
design_and_controls: '固定同一056 V300 cache与manifest/summary；全候选/全节点、source-pair分层、零分母和无共同支撑正边保留'
sample_materials_or_experimental_units: '300scene/有效输入identity/source-pair/正边；非全9100target或真实独立物理观测'
primary_outcome_and_estimand: '受限identity/source-pair和正边人口比例；micro和场景/identity等权条件recall上界分列'
secondary_outcomes: '同源fragment数、source可用数、无共同Q64支撑正边、component分层、完整null分母和成本'
analysis_plan_and_uncertainty: '有限开发总体描述，不做IID显著性；同源真边和异源真边分开，不以U冒充paper指标或实际错误'
resources_time_cost_and_risk: '一次CPU<=4/RAM4GiB/AS8GiB，10min advisory；零模型/前向/训练/ZIP/reference/test'
ethics_safety_privacy_or_compliance: '仅匿名scene内identity序号/source等价类和计数；原node/MMSI/坐标不公开'
stopping_and_interpretation_rules: '输入/绑定/等价/自检失败保全停止，不造数据/修标签/补前向；合法零结果保留；无自动训练后继'
status: diagnostic
evidence_status: partial
missing_evidence: [actual_capacity_population, current_cache_readability, explicit_author_task_alignment, actual_method_harm_and_benefit]
next_skill: record-experiment_after_actual_execution
```
