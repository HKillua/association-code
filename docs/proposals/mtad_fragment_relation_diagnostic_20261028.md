# 片段身份与一一匹配容量：有界诊断设计

2026-10-28。**仅事前设计，尚未执行、未发布READY，不是新训练或方法选择。** 消费[Mode C1.21](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-121)的C33；107保持原冻结合同。选择此诊断因为可不训练先区分任务关系，不是断言它最有新颖性或已是根因。C34/C35/C37仍保留独立候选，当前决定只见[研究状态](../research_status.md)。

## 问题、性质与边界

IAGM原文真ID跨中断不变、实轨单位质量；MIS公开正文用全候选Hungarian；MTAD生成会把一条母轨中断成多个片段。与此同时，HTG/MVW/MH明确采用每源每目标一轨的条件。**作者是否先选轨/并段/取同期窗未知**，不得用本项目人口推翻作者成绩。

若某次评价要求两源同身份全部跨源边，目标t有a_t/b_t条独立片段，则真边数a_tb_t。只有当最终输出为一一离散轨对、没有并段或身份闭包时，直接真边最多min(a_t,b_t)。单场micro上界为

`U_s = sum_t min(a_t,b_t) / sum_t a_t*b_t`。

单位质量**软**矩阵可令全部真项非零，不受这个非零边数界约束；r条正边未归一NLL≥r log r是另一性质，不等实测训练失败。U_s不约束ITSA、组身份召回或融合误差。跨场直接累加分子/分母是micro；场景等权必须先逐场算再平均，不能混为同一指标。

## 假设与有区分力的诊断

- **H1，结构存在性**：在现有T/V全部合法片段人口、A-B任务身份下，有目标存在a_tb_t>min(a_t,b_t)，使原始跨源全真边目标无法被一一硬匹配完整表示。反例是每目标每源≤1轨，或声明任务本来只输出同期代表轨/身份簇。
- **竞争解释**：只是在比较不同任务；目标数/片段数/源不可见性混用；标签区间不等有效原生时间；作者把片段先并成实体；最终评价对组做身份闭包。以上必须分别呈现，不能用“碎片多所以网络差”概括。
- **解决路线目前只作条件参照**：局部一一时窗、时间兼容的身份分组、同源接续与全局修复均可能合理。CVPR2025 GREEDA提供不完整/稀疏图与局部重分配，但原约束每图每身份≤1，直接按source建图同样会冲突；未知证据也不能作硬禁止。不开跑任何一个新模型来预定答案。

## 材料、输入、步骤和结果对象

使用正式Level1既有固定T2100/V300名单及**关联映射CSV**，依据[数据约定](../data_contract.md)读取节点(scene,source,batch)和A-B任务身份。只读label成员与名单，不开量测/reference/预测NPZ、checkpoint，不读封存T_test。原A母MMSI不替代A-B任务身份。若节点映射冲突或缺字段，保留异常并停止解释受影响场景，不删难例或静默改标签。

1. 统计每场每A-B每源节点数、源数、singleton/无跨源真边人口、全跨源真边数和单源片段构成。来源超过两种时按无序source-pair分层，不能静默套双源全场公式。
2. 对每个双源比较对象算真边数T_s、条件一一最多真边数B_s、U_s。报告全T/V固定人口及无法定义分母的场数/目标数；零真边不自作U=0或U=1。输出场景等权上界与pair-weighted上界，标签与分母映射到[论文指标](../mtad_data_protocols.md)，只有精确相同时才使用mPCA/mwPCA名称。
3. 保留标签声明的时间区间，描述同源片段不交/交叠/接触及跨源声明支撑；**这些只是标签区间，不认证真实量测支持**。不按临时窗删片段构造一个高上界人口冒充共同协议。同期单轨任务只作明确声明的反例解释。
4. 将“原始硬边集合”与“身份组/合并后关系”分开定义。当前材料不足以认证后者时记UN；不借真实身份作为推理合并器后报模型成绩，不把oracle上界当可实现算法。

主要结果是**固定数据任务关系的结构可行性**及其人口占比，非训练效果、作者复现或新颖性。辅助结果是source构成、片段数/声明时长/场景密度分层与38个V共享A组件描述。全量固定名单是有限总体计数，不虚构IID显著性或独立身份泛化；不从现有V开发结果调整名单。

## 资源、终点和解释规则

若后续交接，目标为一次CPU标签流读：CPU≤4、RAM预算4GiB、label成员64MiB读取上限，0GPU/优化步/前向；暂估≤35分钟，正式执行须记真实成本。这里的估计不是已测性能；不增加107资源或修改活动checkout。成员超限/解析不一致/身份冲突保全并退出，不自动重试。输出脱敏每场/每身份计数和汇总，不公开MMSI、坐标或原始标签。

- **A：结构冲突存在且覆盖明显人口**：确认“本项目全片段/全边关系”不能由声明的一一硬输出完整表示；进入时间兼容/组关系方法设计和直接近邻精读。仍不证明性能损害或本领域首次。
- **B：全部多片段已被合法单轨化，或目标评价本来是簇而非原始边**：关闭本设置的强冲突命题，不为了保留想法改变数据或评价；回到C34/C35/C37。
- **C：真实节点/标签/作者预处理语义不足**：只保留可核计数，明确缺失项，不宣布成立或不成立。作者TSP2021直接近邻正文缺口与本地人口诊断可以并行，不无限等待，也不能拿摘要结案新颖性。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-MTAD-FRAGMENT-RELATION-DIAG-20261028
source_matrix_id: MC-RESTART-20261027@1.21
selected_C_id: C33
source_P_ids: [P08]
source_F_ids: [F86, F100]
source_M_ids: [M35]
source_E_ids: [E49, E51, E53, E54]
research_question: '完整同身份片段边关系与一一硬匹配的输出/评价对象是否兼容'
hypothesis_and_mechanism: '条件真边ab与硬容量min的差；实际人口/同期单轨/组闭包决定是否触发'
testable_predictions: '部分真实任务对象若a*b>min且评全边，则U<1；软矩阵非零边数和组召回不受此界'
design_and_controls: '同一固定T/V身份与source人口，分原始边/同期单轨反例/身份组语义；不改名单、不用oracle推理冒充模型'
sample_materials_or_experimental_units: 'T2100/V300合法关联映射CSV；场景/任务身份/source-pair；不读test或运动/参考数据'
primary_outcome_and_estimand: '固定全片段原始跨源真边任务的容量可行性与直接边召回条件上界；场景等权和micro分开'
secondary_outcomes: 'source可见性、片段数、声明区间/交叠、零真边人口和共享A组件描述'
analysis_plan_and_uncertainty: '全量有限总体计数，无IID显著性；定义/解析/身份不清保留UN；不给作者人口和身份独立性背书'
resources_time_cost_and_risk: '事前暂估CPU<=4/RAM4GiB/<=35min，一次标签流读；0GPU/训练/前向；未执行未READY'
ethics_safety_privacy_or_compliance: '只输出脱敏计数；T_test封存，107活动执行不变'
stopping_and_interpretation_rules: '异常保全停；A仅确认本任务结构限制，B关闭强版本，C缺语义不下结论；不自动加训练/改评价'
status: diagnostic
execution_status: not_executed_not_READY
evidence_status: partial
missing_evidence: [actual_per_identity_source_fragment_population, actual_author_preprocessing_and_discrete_output, original_edge_vs_group_evaluation, TSP2021_direct_neighbor_fulltext, harm_and_domain_novelty]
next_skill: record-experiment_after_actual_execution
```
