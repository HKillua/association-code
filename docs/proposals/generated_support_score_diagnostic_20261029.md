# 生成支撑与全窗评分：输入资格审计及事前诊断计划

2026-10-29。**当前事前版本为末节plan1.1：只读已有TCM报告的固定V16支撑/特征误差记账，零前向/训练；完整TAM评分子问题仍deferred。** 下文plan1.0保留最初资格判断及尚未执行的备选入口。107方案不训练TAM；其检查点中的TAM权重不能当作训练完成的关联器。实际领取/执行仍由独立交接控制。

## Mode C与研究配置

- `package_type: IDEA_PACKAGE`；`package_version: research-pipeline-contract v1.0`。
- `idea_id/package_id: IDEA-GENERATED-SUPPORT-SCORE-20261029`；`plan_version: 1.0`。
- `source_matrix_id: MC-C52-STEPWISE-20261029`；`selected_C_id: C52L-Cb`。
- `source_P_ids: [C52L-Pb]`；`source_F_ids: [C52L-F1,C52L-F2,C52L-F3,C52L-F4,C52L-F5]`；`source_M_ids: [C52L-Mb,C52L-Mt]`；`source_E_ids: [C52L-E1,C52L-E2,C52L-E3]`。
- 来源：[C52完整8×2小矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/c52_stepwise_problem_matrix_20261029.md)、[107原事前计划](tmtaff_unit_baseline_A_20261027.md)、[107冻结合同](../../HandOver/L2S-20261027-107.md)。本次只选Cb；用户允许四分支并行，其他分支不并入本Idea。
- 领域/对象/任务：多源海上航迹的时域补全与身份关联。模式Pattern5：系统/接口审查。
- `evidence_status: partial`；源内运算与预算可定位，目标生成区不利后果未知。

## 已完成的本地资格核验（不是模型实验）

| 检查 | 实际本地证据 | 结论 |
|---|---|---|
| 107训练了什么 | 两份`configs/tmtaff-*-unit-draft.json`：TCM2424，TFM1000，TAM_FROZEN/JOINT均0；107合同同样明确不训练TAM | 固定预算不含已训练新单位TAM。根线程后续已报告服务器TCM/TFM完成，但本文件不把转述当独立检查点读取。 |
| TFM从TCM转移什么 | `association/training/tmtaff.py`的`initialization`只转`input_embedding/tcm_encoder/tcm_decoder`及相关buffer；语义明确`TAM unused`；fresh模型先构造，再载这些模块 | TFM检查点即使有`tam_encoder.*`，也不是本轮训练完成的关联端点；不能因字段存在就运行评分并解释错误。 |
| 正确的O支撑字段 | `association/data/timed.py:prepare_timed_scene`生成`scene['observed']`，为query在保留原生首末时刻内的插值支撑；`pack_timed_scene`故意把query部分的输入mask全部设false | O必须取准备后scene支撑，不能直接把packed mask后64位当O，否则会把全窗误算为生成。O包含确定性插值，不是全新原生观测。 |
| 实际生成范围 | `association/models/timed.py:TimedAssociationModel.complete`由原生首末时间、query时间计算before/end，只对片段两端外的query生成latent；片段内保留嵌入 | 用相同float32时间边界时，G=非O；`pack_timed_scene`已有浮点支撑一致性检查。原生点在两个query之间时，允许该片段O为0而仍有合法原生历史。 |
| 评分数学接口 | `association/models/tmtaff.py:encode_tracks/score_pairs`：TAM输出展平后负欧氏距离，非校准概率；可在同一已训练模型输出上拆OO/OG/GG平方能量 | 技术接口可行，统计解释依赖同一实际已训练TAM/TCM/normalizer；目前107不满足。 |
| 056能否替代 | 107固定056评价不执行TAM，`final-scores.npz`来自原TSA-cTFER模型，其模型/表征与107不同 | 不能把107生成特征与056分数硬拼后声称解释了其排序/错误。可各做独立来源审计，但不是本Idea的Pb验证。 |

本轮只读代码、配置和小型回执；0原始ZIP读取、0权重读取、0模型构造/前向、0GPU/训练、0reference/test。没有发现合法兼容的新单位已训练TAM端点。旧088/094/096等失败适配或旧单位模型不自动恢复为可信同行；无此端点也不意味着补全无价值。

## 问题、假设与既有正面证据

`research_question`：在固定输入人口、支撑/时间和有效训练端点下，双方均生成的区间GG是否在长缺口片段中与额外误接/拒绝相联系；若有，能否在保留TCM正常收益的同时改善这种条件性失配？

`hypothesis_and_mechanism`：两端延展扩大GG范围，完整序列编码距离可能受生成区影响；但attention可隐式识别支撑，生成运动也可能带来真实辨别力。原TMTAFF Fig9的TCM正收益必须保留：ITSA mPCA81.46→94.58，MSTSA86.41→93.91。当前没有“不利主导”的MTAD直接证据，不能先删补全再找理由。

## 最小验证及条件分支

`sample_materials_or_experimental_units`：若发布结构记账，固定原T_core按Python `random.Random(20261029)`从数值排序2100场中抽32场，抽后数值排序；Q64/cap128/time_unit20/new-unit policy与107一致。选择不看成绩，不替换空场或难场；仅标签/量测，不读reference/V/test。该分支没有模型效果指标。

`design_and_controls`：

1. **当前合法入口：结构记账。** 逐场通过生产native与timed准备接口，保留全部合法节点及全部无序pair。报告每轨O/G query数、原生精确命中/插值数量；每pair的OO/OG/GG数量及直方图，并按正负身份×同源异源分层。三类之和必须等于64，所有分层pair数之和必须等于全候选数。同源/无共同query正对完整保留，O为0不判负、不删节点。可核是否确有大量GG或无OO的目标输入条件，但不能评价生成值好坏。
2. **有合法端点才接评分。** 需要同一已训练TAM与TCM、T内normalizer、真实阈值/选择来源和训练history；不能因检查点含随机TAM字段而通过。若满足，固定最多同T32、零训练、一次小GPU推理，每轨一次completion/TAM；按全pair批量计算完整score及`d²=D_OO+D_OG+D_GG`，核与原score一致，再看分层排序/错误。当前条件不满足，因此此阶段`UN/deferred`，不自动训练TAM补条件。
3. **不能越级解释。** GG能量只是输出索引记账，attention混合使其不是独立原始观测likelihood/因果贡献。后续有信号才另定合理生成区域干预；本阶段不置零/加噪、不重训、不改threshold、不改生产模型。

`primary_outcome_and_estimand`：当前结构阶段仅估计固定T32输入的O/G及OO/OG/GG覆盖分布、合法正对在无共同query支撑下的数量/比例；分母为全部声明场/节点/候选，另报quarantine/空场，不把可比较子集冒充总体。**不报告TAM准确率、PCA/PFA、物理误差或创新成功。**

`secondary_outcomes`：每场与来源/身份分层直方图、原生选择/插值计数、读取成员数量、墙时/RSS。结构分层不使用原始身份字符串或坐标作为公有产物。

`analysis_plan_and_uncertainty`：这是32个已定训练场的描述性输入审计，非独立泛化或因果检验；按场同时保留分布，不对pair/query做伪独立置信区间。若未来有score，真负/真正、FP/FN与实际阈值绑定；0值能量的比值未定义保留null，不删除pair。改分区而固定score不能改变raw PCA/PFA。

`testable_predictions`：结构分层恒等式应成立；若确有高GG/无OO，只支持风险条件存在，不支持有害。若几乎全有OO/低GG，只缩小该数据范围的优先级；仍可研究其他gap条件。若合法TAM缺失，不将随机或异构score结果作为任何预测检验。

## 成本、停止和发展价值

`resources_time_cost_and_risk`：结构阶段CPU≤4，GPU0、训练0、一次尝试、逐场释放；32场只读64个label/measurement成员，单成员64MiB上限，预计内存预算4GiB/AS≤8GiB，实际耗时未知，10min仅规划值。未来score阶段最多一次单GPU、track microbatch4，全pair批量记账；没有合法端点不分配GPU。当前没有脚本/交接READY，不把计划冒充已执行。

`ethics_safety_privacy_or_compliance`：T_test和reference不读，原始量测/身份/坐标只留服务器；公开输出仅聚合计数和已公开场ID；不修改107活动目录、权重或失败证据。

`stopping_and_interpretation_rules`：当前完整Pb评分阶段因缺合法已训练TAM明确停止；若发布结构阶段，输入缺件/单位或准备接口不符/成员超限/恒等式失败即失败保全，不重抽/重试。结构记账成功仅证明输入风险条件与接口正确，不认证根因。高GG但后续score有益时应保留补全；无害结果仍有价值，可避免为一个错误前提开发新网络。

可发展路径：成熟部分域/联合时域方法给出了保留观测与预测类型的起点。若后续真实score证据成立，可先建立合法支撑条件接口A′，再比较候选机制B的净收益；新增价值可以是长缺口可靠性、相同质量下的补全成本或明确的适用边界，不要求“此前完全无人使用”。这一步仍由证据决定，不预选网络。

## 交接决定

- `status: deferred`，范围特指完整Pb关联评分检验；输入资格审计已完成，结构记账方案已具体化但未执行。
- `missing_evidence`：兼容的已训练新单位TAM及真实阈值/训练来源；固定T32实际结构记账；同模型OO/OG/GG与错误关联。
- `next_skill: record-experiment`，仅在独立真实执行后接入；当前资格核验不是EXPERIMENT_RECORD或新科研成绩。

主线程可独立发布CPU结构记账；不能借107已有TCM/TFM自动新增TAM训练，也不因本分支暂缺score而否定C52发展价值。

## 事前修订plan1.1：复用已有TCM原件，固定V16零前向记账

**修订原因与时点：** 计划形成后、此新诊断尚未执行时，根线程[现场确认107的TCM与TFM完成](../reports/2026-10-29-107-failure-discovery.md)；固定056子任务在旧缓存schema预检失败，比较前向为0。TCM已保存完整`completion-evaluation.json`，现场大小216,833,440B；静态writer核实其包含逐fragment的O/before/after计数及model/state_hold特征误差。因此无需原ZIP、权重或新GPU前向即可完成一个真实来源记账。该新证据使“整个分支deferred”收窄为“仅TAM评分子问题deferred”。不是根据性能选择样本/改假设。

### plan1.1交接字段（覆盖plan1.0执行范围；旧设计保留）

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-GENERATED-SUPPORT-SCORE-20261029
package_id: IDEA-GENERATED-SUPPORT-SCORE-20261029
plan_version: '1.1'
created_at: '2026-10-29'
source_matrix_id: MC-C52-STEPWISE-20261029
selected_C_id: C52L-Cb
source_P_ids: [C52L-Pb]
source_F_ids: [C52L-F1, C52L-F2, C52L-F3, C52L-F4, C52L-F5]
source_M_ids: [C52L-Mb, C52L-Mt]
source_E_ids: [C52L-E1, C52L-E2, C52L-E3]
status: diagnostic
score_subproblem_status: deferred_no_compatible_trained_TAM
evidence_status: partial
missing_evidence: [actual_V16_support_feature_summary, compatible_trained_TAM_scores, pairwise_OO_OG_GG_masks, causal_and_physical_outcome_validation]
next_skill: record-experiment
```

`research_question`：在已完成TCM原评估中，固定V16各片段的provided O与generated G分别占多少query、相对于同支撑state_hold有怎样的**冻结特征空间误差差异**？它只检查Pb所需输入/预测支撑条件和已有特征结果，不回答关联评分根因。

`hypothesis_and_mechanism/testable_predictions`：O=原始输入区间内的确定性插值嵌入，G=before+after模型延展；O与G应分割64query完整分母。若G特征MSE优于/劣于hold，仅说明这个训练端点在四属性reference均有效query上的嵌入误差相对变化。O不全是原生点，低MSE不是米制改善、身份质量或物理真值认证；保留“生成有益”的反向结果。

`sample_materials_or_experimental_units`：从既定manifest的V_select300场数值排序，用`random.Random(20261029).sample(...,16)`，再数值排序，**固定V16而非原plan1.0的T32**。不读成绩选场，不删空场/无reference片段，不补抽；实际场ID由纯manifest选择写入run。选择只依赖固定名单，不依赖已存报告内容。V已反复开发且共享身份，不称独立测试。

本地只读既定manifest预先计算的16场为：`700,1293,1582,2038,2153,2416,2479,2576,3156,3772,3835,3916,3956,3981,4272,4473`。此时未读取这些场景的TCM数值。

`design_and_controls`：

- 脚本为[generated_support_score.py](../../tools/analysis/generated_support_score.py)，仅Python标准库，单次读取已有TCM `run.json/summary.json/completion-evaluation.json`及既定manifest。完成状态/2424预算、native_time单位metadata、Q64/cap128/20s与128特征维均核；不加载checkpoint，不构造/调用模型，不读ZIP/reference CSV。**报告指标已经使用reference，不能以“本次没读CSV”声称未接触参考派生信息。**
- 一次有界解析完整已有报告（≤256MiB；现场约216.8MB），核300场完整原序、全journal fragment数/每场分母/node_index唯一连续；只选固定16场全部fragment参与主要估计。逐fragment保留所有query与缺参考query，核各区域分母/误差和、model与hold同mask；不平均各fragment MSE冒充池化MSE。
- 对model/state_hold各自按原squared_error_sum与feature_elements池化all/O/before/after/G；G合并before+after，不能平均两端MSE。比较差统一为model−hold，负值表示特征MSE较低。O和G的误差分布/参考覆盖不同，不拿两区均值直接宣称外推导致恶化。
- 可附原顶层全V300汇总，明确是**既有source输出且与V16重叠**，不是第二独立验证集，也不把它与16场差异解释成模型变化。

`primary_outcome_and_estimand`：固定V16全部fragment/query的O/G数及占比、无O fragment数，model与hold在各区**共同原有效reference分母**上的池化feature MSE和差值；all/G缺参考数量与coverage并列。空scene仍占场分母，空评价区返回null，不能过滤。

`secondary_outcomes/analysis_plan_and_uncertainty`：逐场同统计、原完整V报告人口/顶层既有汇总、读取字节与次数、真实墙时/RSS。单位是冻结T内normalizer＋时间编码＋ReLU投影后每特征元素的平方误差，非经纬度/米/速度物理误差；normalizer不refit。描述性单端点开发集记账，无pair级伪置信区间、无checkpoint/阈值重选。

`resources_time_cost_and_risk`：根线程计划独立112，一次CPU≤2、RAM预算4GiB/AS≤8GiB、0GPU/新前向/训练/ZIP/权重；建议10min，真实耗时未知。逐报告单次解析无需持久化整journal或复制私有标识；Linux进程AS上限8GiB并记录峰值RSS，解析后超4GiB则失败。外部4GiB资源托管由独立交接绑定，不把事后RSS检查冒充硬cgroup保证。单报告/metadata字节超限停止，不改范围重试。

`ethics_safety_privacy_or_compliance`：不改107文件、原失败、数据/模型；公共产物仅聚合/逐已公开scene计数，不导出node_id/identity/原坐标。T_test不读；该缓存原为V_select含reference派生评估，使用范围如实披露。

`stopping_and_interpretation_rules`：缺源/未完成/单位或人口不符/非有限或计数不一致/内存上限/异常即失败保全，0自动重试；不得替换旧单位模型或随机TAM。O/G与误差记账成功只认证本次汇总和源条件，不认证Pb成立。G优于hold保留其有效条件；G差且有大量支撑缺口仅提供后续条件化方向，不能说TCM普遍有害。**无逐query边界/latent数组，不能由区域计数重建pair OO/OG/GG或TAM距离；评分子问题继续UN，不能据112成功自动训练TAM。**

下游：真实结果先Record→Squeeze；把新增支撑/特征现象回C52小矩阵，再讨论是否需要合法已训练TAM或独立物理恢复诊断。既有TCM原文正收益及成熟支撑方法仍为可发展起点。
