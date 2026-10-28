# C45输入依赖诊断：候选B的信息是否真的进入表示

2026-10-28；`IDEA-PAIR-REP-DEPENDENCY-20261028`，plan_version1.1。消费[Mode C1.23](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-123)的单个C45，模式3（假设挑战）。**这是零训练的表示/问题资格诊断，不选定EQ-Reg或最终网络，不是CA/FESR作者复现。** 真实执行、终态和回执只看[交接](../../HandOver/Handover.md)。

## 研究问题与贡献边界

用户明确认可外域机制适配解决领域已知但尚未很好解决的问题。当前“偏差鲁棒与身份区分的取舍”已被MH-T2TA等识别；具体输入删除、作者实际接口和现有方法残余瓶颈仍要核。本诊断只回答：**在定义明确的公开表示版本中，固定A和全部时间/validity，替换同场候选B时，完整可见输入是否仍保留B内容；统计恢复和逐时对齐各恢复什么？**

如果只发现印刷literal版本不含B，而可信完整方法已有B通路且未见残余瓶颈，则该发现只作接口/复现警示，不足以立EQ-Reg创新项目。若真实正负候选在该版本中确实碰撞，也仅支持该表示合同的限制，不证明作者实际网络错误、整个领域未解决或跨域方法有效。

### 证据与竞争解释

- CA Algorithm1的固定B窗均值、输出(A,N(d))与正文逐时B邻域均值不同；到C=4分类输入的装配未知。来源DE见[C45原件补核](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/c45_completion_20261028.md)。
- 对非恒定列，`N(A−c)=N(A)`是AI条件代数；常数列按下述明确研究规则处理。raw A旁路不恢复被去掉的B，但raw B、双向输入、B统计或B依赖窗口/支撑可使强命题不成立。
- 原始/恢复后输入不同不等可学身份，也不等更好泛化。m/r只能恢复已形成的差分d，不能恢复此前B整窗均值删掉的时间形状；两个适配分开，不把收益混归EQ-Reg。
- 未使用作者代码、权重或模型输出；标量航向是印刷公式表示控制，不赋予其物理圆均值的正确性。

## 事前输入、人口及时间权限

正式Level1既有三个ZIP字节分卷及固定`configs/data/mtad-level1-scene3000.json`。用Python标准库`random.Random(20261028)`对按数字排序的T_core2100场抽32场，再按数字排序处理；抽样只依据冻结名单，不依据身份组成或旧成绩。**只开这32场的label和measurement，0reference/V/test/未用场payload。** ZIP目录遍历仅作寻址，不等读取其余CSV。

复用`MTADArchive.load_scene`现行清洗/标签语义，identity为A-B，不换成母A。场景全部节点/隔离及原因计账；输出不含原identity、source/batch或绝对坐标，完整敏感debug留服务器private。沿用现行reader按缺失/歧义标签隔离形成合法人口；在该合法人口内，identity只用于诊断统计/正负分层，不参与抽样、选窗、插值或构造表示。每场每个物理source从合法tracks中以同一固定RNG最多抽8个anchor；该选择不看身份。每anchor若少于16个点记不足；否则随机取一个连续16原生时点窗口q，所有版本保持该A窗口和q不变。

对同场全部异源候选B，不限正确ID，定义eligible为其原生起止覆盖全部q（含端点）；不足记unknown/outside-support，不外推、不删除后冒充完整召回。对eligible B用同一个线性插值器在q对齐位置/速度；航向先unwrap弧度插值再映回[0,360)。B的完整时间形状只指这些已对齐query值，不宣称保留全部原生点。保持原生速度数值，不拿它作m/s物理误差。

## 计算合同：A/A′与替代解释控制

所有版本的完整接口都显式带相同A_q、q及validity；source同/异关系固定为异源，不加入B节点ID、真实身份、label区间或任何reference。

| 版本 | 精确输入/算子 | 唯一回答的问题 |
|---|---|---|
| A-literal-v1 | `c=mean(B_q,axis=time)`；`d=A_q−c`；逐列`u=N(d)`；完整接口`(A_q,u,q,validity)` | 印刷整窗mean版本是否与B内容无关 |
| A′-stats-v1 | 与A相同d/u，额外保留逐列`m=min(d), r=max(d)−min(d)` | 能否重构d和B均值；不能恢复B形状，不叫解决source bias |
| A-pointwise-v1 | 同A_q/B_q/q，以`d_t=A_q−B_q`逐列minmax | 区分逐时对齐和整窗均值，不称已忠实还原作者邻域宽度 |
| A′-rawB-v1 | A-literal完整接口额外保留B_q | 作为可辨B内容的显式旁路控制；额外信息的作用不能归B正则 |

`N(x)`以float64逐列计算；range=0时u全0（明确研究分支，非作者实现认证）。所有常数/近常数情况保留计数，不悄悄删除。A′-stats重构`d_hat=u*r+m`，报告逐属性数值误差；rawB与pointwise差异仅作内容依赖，不能当分类准确率。

**迁移B此阶段不执行。** EQ-Reg若后续值得研究，只能在同一保信息A′、相同paired增强/容量/预算下与无正则相比。共同世界旋转、source共享偏差、逐轨独立平移分别定义；不旋转时间轴、不假设正则可逆转已删信息。

## 主要估计对象、辅助结果及不确定性

主要对象是本次明确抽样与eligible规则下的**anchor条件输入依赖**，非作者关联/融合成绩。对每个eligible pair记录`max|u_literal−N(A_q)|`；预先以1e−8归一单位作数值相等诊断，同时保留最大/分位数原值，阈值不按结果改。

对同时有至少一个同A-B identity和一个不同identity候选的anchor，直接比较literal完整输入的候选间差异是否≤1e−8，并对实际正负候选向量直接求max绝对差判断是否存在碰撞；不能仅由各自到N(A)≤ε推两者间≤ε。分母分别报告：全部抽中anchor、不足16点、无eligible候选、仅正/仅负/混合标签、eligible pair与outside-support pair，不能只报成功混合子集。A′-stats、pointwise和rawB分别对同一anchor的真实正负候选记录是否存在内容变化：精确float64不等与超数值容差分别报告。u/pointwise的归一分量用1e−8；stats的m/r和rawB的各属性用原单位逐分量阈值`64*float64_eps*max(1,abs(x),abs(y))`，不把归一1e−8混套经纬/原速度/度。各属性最大差按其原单位分列；该阈值只区分浮点差异，不代表有用的物理效应或已学会判别。stats重构误差按属性报告绝对/相对值，保留零range数和潜在浮点放大情况。

统计以场景为单位保留逐场/每source方向/汇总，32场是固定随机抽样探索，不虚构IID目标独立显著性或全T发生率。结果若涉及比例，给实际分子/分母和逐场范围，不补造性能CI；不挑有利场景、不调阈值。

## 资源、终点与停止/解释规则

独立任务108；一次CPU执行，CPU≤4、RAM预算4GiB（记录实际RSS和单位）；0GPU/模型构造/前向/优化/权重/新normalizer。32场每场label/measurement声明解压各≤64MiB，预检超限不读payload；虚拟地址空间限制8GiB，已有更低限值不提高。墙时暂估≤10分钟，仅advisory；固定32场，超建议报告进度但不扩量/自动重试。沿服务器现有环境执行，独立冻结checkout/output，不改107。

- 输入、解析、非法/非有限计算、成员大小不合约或内存失败：保全并真实失败，不能删场缩人口重试；纯数值偏差超1e−8不自动当程序崩溃，而按病态range/浮点差异如实报告。
- **结果A：literal跨混合标签不变，stats可重构d，rawB/pointwise有不同依赖。** 支持这些明确表示版本的条件限制和适配分工；不支持作者真实模型失败/本领域仍普遍未解决/EQ-Reg有效。继续核可信baseline的完整输入与残余困难后才讨论新训练。
- **结果B：事前定义的pointwise/rawB控制依赖B，或本次没有混合标签eligible支持。** 前者限制将literal性质推广到所有接口，不能据此声称查到了作者真实rawB旁路；后者只记样本/支持范围不足。版本与作者真实接口分开解释，不为保住结论换窗/筛B。
- **结果C：literal差异超过预定容差。** 先查全输入依赖、浮点消减和病态range，不先判代数错；无清晰解释保留UN，不改阈值制造支持。

正常终点为32场均完成且结果/读取名单/耗时退出原件齐备；无模型收益晋级门槛。实际执行后才Record→Squeeze→Mode C回流。发布READY不等已经启动，结果以回执为准。

## 三层质量检查

| 检查 | 当前判断与限制 |
|---|---|
| 新颖性 | 归一化损可分性已由MH等识别；本诊断不声称新问题。具体EQ-Reg迁移待核，CKSMO完整PDF未得。literal算子界限不能替代方法创新；领域未很好解决程度仍需可信方法与困难条件证据 |
| 研究价值 | 支持时能区分表示已删信息与后端没用好，并区分恢复d/恢复B时间形状；不支持时限制C45强命题，避免直接投入正则训练。两种结果都有决策价值 |
| 可归因性 | 固定A/q/validity与同一候选B，label不构造特征，均值/归一统计/逐时差分/rawB各分臂；不给数学性质赋予训练成绩。同输入和预算的B增量留后续正式Idea |

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-PAIR-REP-DEPENDENCY-20261028
idea_id: IDEA-PAIR-REP-DEPENDENCY-20261028
created_at: '2026-10-28'
plan_version: '1.1'
source_matrix_id: MC-RESTART-20261027@1.23
selected_C_id: C45
source_P_ids: [P09]
source_F_ids: [F87, F104, F105, F110, F122]
source_M_ids: [M36, M47, M49]
source_E_ids: [E50, E51, E56, E58, E62]
research_question: '固定A和时间支撑后，公开表示版本是否保留候选B内容，以及统计/逐时/rawB适配分别恢复什么'
hypothesis_and_mechanism: 'literal固定B均值在窗内minmax后消B；统计可逆d而非B形状；其他接口是独立可反驳的控制'
testable_predictions: '正文主要估计对象和A/B/C解释；不使用模型成绩'
design_and_controls: 'A-literal、A-stats、A-pointwise、A-rawB；固定A/q/validity和全部eligible B，label仅评估'
sample_materials_or_experimental_units: '固定T_core按seed20261028抽32场；每场每source至多8个anchor；16点窗口；只label/measurement'
primary_outcome_and_estimand: '混合identity eligible anchor下的完整输入B内容依赖和容差碰撞；全分母逐层保留'
secondary_outcomes: '各接口依赖、stats重构误差、零range/浮点差异、support未知/隔离/scene异质性及实际成本'
analysis_plan_and_uncertainty: '实际分子分母/逐场；非作者PCA、非全T发生率、非IID身份显著性；不调阈值'
resources_time_cost_and_risk: '一次服务器CPU<=4/RAM预算4GiB/AS<=8GiB；32场/64MiB每成员；暂估10min advisory；0模型/训练/GPU'
ethics_safety_privacy_or_compliance: '只T侧既有授权数据；0reference/V/test，输出脱敏汇总；107不变'
stopping_and_interpretation_rules: '真实失败保全不重试；A仅支持接口限制，B限制强命题，C数值/输入依赖未解记UN；不自动开训'
status: diagnostic
source_transfer_id: TR-C45-REPRESENTATION-CONDITIONS-20261028
transfer_condition_commitments: '本阶段仅核A/A′输入条件，EQ-Reg及其效果未执行；方法价值须针对未很好解决的领域问题'
adaptation_vs_method_attribution: '恢复d、恢复B时形各分臂；没有B训练，所有变化不得归EQ-Reg'
evidence_status: partial
missing_evidence: [actual_execution, author_exact_input_assembly, field_residual_bottleneck, full_credible_baseline_comparison, actual_cross_domain_method_effect]
next_skill: record-experiment_after_actual_execution
```


执行前修订1.1：独立审查补清现行label清洗作用、实际正负向量碰撞比较、normalized/native逐分量数值容差，以及控制接口不等作者旁路。此修订在108首次发布/执行前完成，不是看真实结果后调整。
