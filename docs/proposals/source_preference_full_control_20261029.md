# 123：LOW/FixedLOW固定配方扩至其余268场

2026-10-29，执行前Idea/规格1.0。消费Mode C1.41的F219/M90/E117、融合局部SC-C3及[review](../reports/2026-10-29-proposal-precision-shrinkage-review.md)，用户本轮明确授权。预测配方完全复用121，不以尚未验证的噪声、中心方差或来源质量解释作执行依据。

## 问题与可否证预测

121原V32同71071点LOW/FixedLOW MAE527.446/447.060m，但FixedLOW较FixedMean20场更差、P99较LOW差265.615m；最大组件19场58707点。真实问题是点汇总、场景与尾部的异质性，而“低源更干净”“小样本中心过拟合”均为AI/UN。123检验同配方在其余268场、原32未见组件及全V中的适用范围。新268是反复开发材料，非独立确认；本项不是新方法或同行复现。

## 固定输入与行为

- 原056 Q64物理输入/114 WS分组/全部目标及首代表/FULL输出valid不变，T_test封存。中心、source_order、group_fold复用120保存；原32的120数组实际来自118。
- 原121的32×10=320场景臂直接导入，不能重构LOW或重拟合中心。其余268的120八臂2144场景臂直接导入，只新算LOW/FixedLOW两臂536场景臂。总3000臂中2464导入、536新算。
- LOW逐query低source有support则只取其均值，否则取高source有support成员；全无support复制首anchor。FixedLOW在相同规则前仅按冻结中心球面平移高source，空支持复制相同的校正首anchor。速度/航向沿原Mean。source字典排序仅经验约定，不冒称物理优质源。
- 十臂完整45配对，原八臂与原32十臂artifact/reference/population逐字段exact绑定；不重新运行原Huber、阈值/normalizer/分组/score。

## 估计量、分层与停止

主要结果：**新268**同点FixedLOW−LOW及FixedLOW−FixedMean径向MAE；同时LOW−Mean、全部45配对、RMSE/P95/P99、coverage/缺失/重复/成本、场景及组件好坏同。全300、原32、新268的已见/未见组件、去场景数最大组件分别报告。预测组both/low/high/none与真目标support分开；后者仅评价，不作门控。不把query当IID、不挑有利组件、不以单个均值声称整体胜出。

若效果延续但尾部或多数场景有损，保留有价值简单控制并限定范围；若不延续，收窄121结论；若FixedLOW仍逊于合理混合，为独立经验混合实验提供比较背景，不能反选参数。不存在由本项结果直接认证方差或中心误差机制的出口。

一次attempt-1，CPU4、RSS4GiB/AS8GiB、1800s advisory、固定V300终点，0GPU/模型/前向/训练/ZIP/CSV；失败保全，不自动重试或改配方。坐标、身份、中心和大预测留服务器，必要脱敏原件回传。实际结束后Record→Squeeze→Mode C；第二阶段混合/收缩先冻结独立合同，待本项回流再执行。

## 实现与验收

| 文件/行为 | 可观察验收 |
|---|---|
| `source_preference_control.analyze_scene`新增可选全臂导入，默认121行为不变 | 导入时禁止调用预测构造，全部十臂exact重放；既有源优先测试通过 |
| 新`source_preference_full_control.py` | 300有序场/38组件、2464导入/536新算、全部人口/45配对及分层 |
| 保存数组独立核验 | 复用独立球面误差/统计实现核全场全部臂/配对/组件/分层；不将该审计称物理真值或预测构造正确性证明 |

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029
plan_version: '1.0'
source_matrix_id: Mode-C-1.41
selected_C_id: SC-C3
source_P_ids: [SC-P1, SC-P2]
source_F_ids: [F219]
source_M_ids: [M90]
source_E_ids: [E117]
research_question: 固定来源优先和共享校正的更广开发适用范围与尾部代价
hypothesis_and_mechanism: 经验规则收益可延续；低源质量和中心误差解释尚未识别
testable_predictions: 新268/未见组件的两项主要MAE差及完整尾部异质性
design_and_controls: 十臂45配对；仅新增536场景臂，2464直接导入
sample_materials_or_experimental_units: 重复开发V300/38组件，原32与新268分开
primary_outcome_and_estimand: 新268共同点FixedLOW相对LOW及FixedMean的径向MAE差
secondary_outcomes: 全配对/尾部/场景组件/覆盖重复/作用域/成本
analysis_plan_and_uncertainty: 确定性配对与组件异质性，无query-IID显著性宣称
resources_time_cost_and_risk: CPU4/RSS4GiB/AS8GiB/1800s advisory/一次固定终点/0训练
ethics_safety_privacy_or_compliance: private留服务器，T_test封存，reference仅评价
stopping_and_interpretation_rules: 失败保全无自动重试；完成先Record/Squeeze/ModeC
status: diagnostic
evidence_status: partial
missing_evidence: [broader_source_preference_effect, lawful_absolute_quality, center_sampling_error, credible_peer]
next_skill: record-experiment_after_actual_execution
```
