# 113：复用107固定端点，独立修复评价

2026-10-29；`IDEA-TMTAFF-FIXED-ENDPOINT-EVAL-20261029`，plan1.0。继承[原A事前计划](tmtaff_unit_baseline_A_20261027.md)的共同指标诊断，不选择新方法。用户已授权四路小验证、初步支持后服务器继续；结合既有A授权与自主修复规则，完成具体计划后可单次执行，无需重复索取同一额度。**0新增训练，TCM2424/TFM1000不再运行。** 原107失败原件/冻结目录保留，113是新code/run/output。

## 问题、修复和不变的科学合同

[107真实终态](../reports/2026-10-29-107-failure-discovery.md)：两训练阶段完成，fixed056在完整输入绑定前读090C旧NPZ不存在的sources失败；旧cache也无membership。并非训练端点性能结果。[修复依据与独审](../reports/2026-10-29-fixed056-schema-repair.md)给出同COMPLETED source_system＋相同node顺序＋056真实score sources的传递来源绑定、原offset/flat唯一派生membership；所有真实已有输入/组/anchor/exact字段继续严格核。19项有区分力的软件回归通过且独立审查通过，只是软件证据。

113只回答：**原固定107新单位端点，在原056分组与相同全V人口下，真实物理恢复与共同论文指标如何？** 不把已修wrapper误称创新方法，TCM/TFM近似参考仍不是作者完整TMTAFF复现，TAM从未训练/使用；共用原raw关联计数不能归因TFM。

- 输入：原107/tfm和其初始化绑定的tcm/checkpoint、原056/system及association/final-scores、090C原final物理cache、正式Level1三个分卷、同manifest/alignment。
- 人口：全V300/9,100声明目标/原9353输出组及610重复的固定决定；逐场真实节点/query/reference/valid/分区必须复现，不按成绩删场/难例，不触T_test。
- 端点：固定107 TCM2424与TFM1000末步，不选checkpoint、不新normalizer、不变singleton/输出参数/精度，不训练TAM。按原脚本核2400准备等真实范围；不重新训练来“修正性能”。
- 对照：固定056的Anchor、Mean、attribute_hybrid等既有普通控制与同query/人口090C；完整四属性论文口径和米制MAE/RMSE/尾部/覆盖/重复/组件保留。ME不是MAE；作者mESD原实现歧义不在本次假装解决。
- 顺序：先数据/端点/单位/来源/组/全场raw计数/参考与mask精确绑定，全部通过后才固定TFM评价。0关联评分forward；新工作量仅已训TCM/TFM固定评价前向及指标，不重跑107训练链。

## 资源、终点与失败

一次单GPU兼容共卡，按现场FIFO选当前可用≥12GiB的设备；CPU≤4、RAM预算16GiB，记录实际RSS/GPU峰值/共卡。建议2h仅advisory，固定300场/末步终点，超建议写进度但不加步/场或换配方。独立worktree/run113，systemd Linger=yes/Restart=no，脱离远程连接继续；不触他人进程。

输入/来源/原raw/mask/partition绑定失败、OOM、非有限或代码故障：真实失败保全，一次不自动重试，不跳过字段/样本、不放宽容差、不缩模型/batch。需要另修复时另行记录/发布，不覆盖本次。正常终点300场全部比较、全人口/指标/逐场证据/真实退出齐备；性能不好也是正常结果，不是续训理由。

正面结果：增加固定组新单位恢复的可比较证据，仍不认证作者复现或新idea。负面结果：区分优化不足、reference/单位/表示/组条件等竞争解释，不自动加seed/步数。前置失败：仅缩小数据接口/软件缺口，不能宣称方法失败。后续研究方向由四路矩阵证据与可信同行共同判断。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-TMTAFF-FIXED-ENDPOINT-EVAL-20261029
package_id: IDEA-TMTAFF-FIXED-ENDPOINT-EVAL-20261029
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.28
selected_C_id: C01
source_P_ids: [P01, P02, P03, P05]
source_F_ids: [F01, F02, F26, F27, F28, F31, F32, F33]
source_M_ids: [M01, M02, M03, M05]
source_E_ids: [E01, E02, E11, E21, E22, E24, E25]
research_question: '原107固定新单位端点在同056全V人口/分组下的恢复与共同指标'
hypothesis_and_mechanism: '只修旧NPZ格式绑定；科学数值不变，完整预检后获得先前尚未计算的固定组比较'
testable_predictions: '真实旧输入绑定/人口保持；性能方向未知不预填'
design_and_controls: '原固定107末步与090C/普通控制；同query/组/reference/单位/全人口，0新训练'
sample_materials_or_experimental_units: '原固定V300，全目标含无输入/未覆盖/重复，原Level1分卷/manifest/alignment'
primary_outcome_and_estimand: '原公平比较合同的物理位置误差与共同论文四属性全人口指标'
secondary_outcomes: '覆盖/重复/尾部、场景/组件、固定关联计数和实际成本；不以oracle选模型'
analysis_plan_and_uncertainty: '沿原事前固定比较合同，全场/全组件并报；开发集共享身份，不声称独立泛化'
resources_time_cost_and_risk: '一次单GPU兼容共卡/CPU<=4/RAM预算16GiB，2h advisory，0训练/新normalizer，固定300场'
ethics_safety_privacy_or_compliance: '既有授权正式数据，仅全V评价；T_test封存，原坐标/身份留服务器，原107保全'
stopping_and_interpretation_rules: '绑定/代码/OOM等真实失败停且不自动重试；完成300场即终点，不因差成绩加训练'
status: diagnostic
evidence_status: partial
missing_evidence: [actual_full_V_binding, fixed_endpoint_evaluation_results, credible_author_reproduction]
next_skill: record-experiment_after_actual_execution
```
