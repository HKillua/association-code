# 127：真实原生位置的CV滤波/RTS与固定Mean/LOW

2026-10-29，执行前Idea/实现规格1.0。消费[输入/时序Mode C §8](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/input_time_action_matrix_20261029.md#8-review后修订先测时间形状的可学增量再独立检验原生状态估计)的IT-C5，承接[独立设计§3](input_time_signal_pilot_20261029.md)。来源为已完整AP的GPTR真实Δt转移/观测时刻分离与wrong-prior反例、SemiDANSE对H/观测尺度的显式依赖、原生接口与122有限负结果；本次不新增完整精读计数，不复现作者完整方法。执行许可只见主仓HandOver，本文件本身不发布READY。

## 1. 真实问题、机制及竞争解释

**DE：** 目前同Q64位置底座在support内直接插值，在外部作测量端点锚定的all-native OLS；这些值不必是无误差的物理状态。当前22维身份摘要已经含部分时序信息，122仅对Q64一阶共享时间残差得到有限负结果，未实际改变预测，不能据此否定原生状态估计。

**AI/待验：** 如果原生位置适合一个局部CV工作动态，允许观测被平滑修正可能减少reference位置风险；也可能因既有航迹已滤波/相关、非恒速机动、共同偏差、错组或reference关系而更差。Gaussian/零均值/独立噪声和真实clock均未识别。本项先核最透明的动态控制，不组合共享中心、soft权重、未知τ或新关联，避免把收益归给未验证的精度。

新增价值若存在，是在海上异步碎片的合法原生事件/完整输出下建立并量化状态—观测分离的效果与代价；Kalman/RTS/CV本身不是本项目原创，也不因有相似方法就结束研究。当前不是已完成的可信同行复现、在线端到端或概率保证。

## 2. 固定输入与数据权限

- 官方Level1三分卷、原manifest及`configs/data/mtad-level1-alignment.json`；T侧固定manifest前32个T_core，V侧固定118/121的原32/9组件。明确不读T_test任何payload。
- 每选定场景经既有`MTADArchive.load_scene → native._record(max_points128) → prepare_timed_scene(DATA) → scene_motion_evidence(canonical_all)`。原生事件由`source_events`取得uncapped canonical原事件，不把056名为`native_states`的Q64预测当原观测。
- **速度/航向不作单位改版**：使用056原factory与原生单位，未指定speed_unit_policy。位置模型只消费lat/lon/秒；T尺度选择只用位置参考。后续可让125复用完整motion字段，不改变其冻结22维normalizer的单位约定。
- 原alignment显式offset与600秒gap、shortest_arc、逐属性valid及全部label命名目标保持，不造真值、不删无参考fragment。选定T/V参考会在独立导出阶段读到并保私有缓存；T选择仅消费T缓存，V参考值只在六预测冻结后作绑定/评价，不称整个进程尚未读取V参考文件。
- 固定原056物理cache、114 WS membership、121两基线artifact：raw重建node/source/query/motion states/support/nearest/exact必须与056逐项exact；原Mean/LOW的完整artifact必须与121逐字段exact。任何差异都失败保全，不把读取链差异算动态收益。

## 3. 明确模型与查询行为

每fragment首点为chart原点，完整可见平均纬度确定east比例；经度沿既有`_short_delta`展开，恰180°保raw差符号。chart使用完整窗口，因此FWD也不能宣严格因果。

`x=[pN,pE,vN,vE]`，`F(dt)=[[I,dt I],[0,I]]`，`Q(dt)=rho [[dt³/3 I,dt²/2 I],[dt²/2 I,dt I]]`，`H=[I,0]`，`R=I m²`。

R与P0是工作尺度，不是测得1m噪声或校准协方差。P0=`diag(1,1,1/3600,1/3600)`，首位置定义prior均值、零速度，首观测不二次更新。后续用真实秒，Joseph更新及线性方程求解；预测/过滤协方差仅对称化，不jitter/裁特征值。仅输出FWD与RTS**均值**，不计算或输出RTS后验协方差，不向下游认证风险。非有限/非正定求解/不合法纬度失败保全，不更换模型或回旧OLS。

只合并观测闭区间内query与真实事件时间；同刻先观测后query，query不是新测量。所有KF/RTS转移dt非负；区间外query在端点状态确定后另作CV外推。FWD首前用首点零速度prior，末后用末前向状态；RTS首前/末后分别用平滑端点。完整64个query全保留，support仍是原真实输入支撑，外推不冒充真实观测。

## 4. T侧选择、六臂与主要估计量

1. rho固定九个`10^k, k=-10…-2`，不扩网格。每candidate的**RTS**按“场景等权→该场有有效参考的fragment等权→该fragment有效query径向MAE”计目标。无参考fragment/场保留null及缺失数，不拿点数加权替代；同人口同九候选。exact并列取更小rho。选定一个rho后FWD/RTS共同使用，FWD不称独立调优。
2. V六臂：原`WS_MEAN`、`WS_LOW_SOURCE_SUPPORTED`两臂重建后与121完整保存artifact exact；`WS_CV_FORWARD_MEAN/LOW`、`WS_CV_RTS_MEAN/LOW`四新臂。Mean与LOW只使用相同原support，LOW的合法来源权重还与121保存query_weights exact绑定。无support取相应状态模型首anchor；新四臂速度/航向全复制原WS_MEAN。
3. 两主比较为RTS_MEAN−Mean、RTS_LOW−LOW，同完整位置人口报告MAE与RMSE；全部六臂15配对、P95/P99、场景/组件/去最大组件、所有人口/coverage/重复及作用域保留。FWD/RTS是未来信息/平滑权限的比较，不叫纯模型优胜。
4. 真目标support内外、exact-native、距目标任一原片段端点≤60s/60–600s/>600s仅事后分析，由身份映射，不作为预测gate。新增状态与原固定中心不组合；Frozen FixedLOW只作历史背景，不拼不同模型冠军人口。

收益支持范围：若RTS在Mean/LOW下均有稳定净增量，支持当前离线工作模型继续发展；若仅一贡献/人群有利，保留交互；若不利或T选择不转移，结束该CV/prior/rho配方，仍不推出所有连续时间/学习状态无效。单次开发数据不证明原插值是物理根因，也不宣国际同行胜出。

## 5. 输出、共享原生cache与验收

代码为`tools/analysis/native_cv_state_control.py`。只有本次新run写盘；旧archive/cache/frozen checkout不修改。全部坐标、身份、原生事件、中心及大预测留服务器private；回传脱敏汇总、完整逐场、真实命令/退出/成本及审计。

| 产物 | schema及可供独立实验复用的边界 |
|---|---|
| `native-cache.json` | 全部64场导出完毕才写`status=CACHE_COMPLETED`，之后不改；`scene_ids={T_core:[32],V_select:[32]}`、rows含split/scene/component与事件/节点/目标计数。125/129可在root协调后稳定只读这些完成文件，不等全部rho拟合结束。 |
| `private/native-events/{split}/scene-{sid}.npz` | 无身份/参考：node_ids、sources、query_seconds、event_offsets、event_times_seconds、event_values[E,4]、event_valid[E,4]、event_node_ids、event_canonical_indices。原canonical谱系，不声称跨不同node物理事件已去重。 |
| `private/native-motion/{split}/scene-{sid}.npz` | 原完整motion字段states/support/nearest/native_count/span/fit_rmse/OLS角速度/OLS与端点米制速度/sources/query/exact，加node_ids与全部pairs。用于125冻结056评分，原单位不改；拟合残差不称协方差。 |
| `private/native-evaluation/{split}/scene-{sid}.npz` | 与物理cache分开：node_ids、identities、target_ids、target_states、target_valid、query_seconds；129只显式取前两项作事后评价，不能传入候选预测。 |
| `scale-selection.json`与private T摘要 | 九候选所有T场criterion、无参考人口、选定rho、实际解析fit/事件/更新/RTS步数；不只报最好rho。 |
| V `private/states-{sid}.npz`与逐场/summary | 六臂完整artifact，原WS_MEAN前缀schema可复用独立保存数组审计；run内training_scene_ids与scene_ids分别为T32/V32。 |

一次attempt-1；CPU≤4/RSS4GiB/AS8GiB、7200s advisory，固定T32×9及V32×6终点。0GPU/神经训练，但**不是0拟合**：每fragment/rho一次解析filter+smoother，实际fit数及事件处理量明确计入。原始读取、导出、选择、评价、写盘、launcher均计成本，超建议报告但不自动加量或杀任务；真实失败保全，不重试/调网格/删难例/换fallback/自动扩量。

| 验收 | 实际核验方式 |
|---|---|
| RTS均值和时间权限 | 独立批量Gaussian正规方程对照不复用Kalman/RTS递推；区间外query不进入Q、首观测prior不二次更新 |
| 日期线/半周与数据隔离 | 日期线/恰180°方向、native event谱系与身份/参考键拒绝 |
| T选择对象/加权 | 公开可解释fixture让fragment等权与point加权给相反选择，核RTS选择而非FWD；无参考场保留 |
| V原控制与完整人口 | 两原artifact exact；四新位置臂速度航向/valid/membership/首代表不变；6臂15配对；坏旧artifact必须失败 |

上述合成检查只认软件证据，不作为MTAD效果/作者复现。本实现的测试首轮有一个临时目录fixture缺`parents=True`，已修正并复跑通过；未发生真实研究attempt失败或改配方。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-NATIVE-CV-STATE-CONTROL-20261029
plan_version: '1.0'
source_matrix_id: MC-INPUT-TIME-ACTION-20261029@review-correction
selected_C_id: IT-C5
source_P_ids: [IT-P2]
source_F_ids: [IT-F14, IT-F17, IT-F18, IT-F19]
source_M_ids: [IT-M9]
source_E_ids: [IT-E14, IT-E16, IT-E17, IT-E18]
research_question: 原生观测与潜状态分离是否在相同Mean或LOW贡献下改善完整恢复风险
hypothesis_and_mechanism: 透明CV工作动态可能平滑带误差航迹；相关/偏差/机动/参考均为竞争解释
testable_predictions: 同完整人口RTS_MEAN-Mean及RTS_LOW-LOW的MAE/RMSE与尾部/场景组件效应
design_and_controls: T32九固定rho以RTS等场等fragment选择，FWD共用；V32六臂15配对，原两控制exact
sample_materials_or_experimental_units: 正式Level1原生事件，固定T32与原V32/9组件，重复开发且共享身份
primary_outcome_and_estimand: 两同贡献主比较的完整共同点径向MAE/RMSE差
secondary_outcomes: 全15配对/尾部/coverage/重复/支撑/原生exact/端点距离/组件/真实解析拟合成本
analysis_plan_and_uncertainty: 确定性单次开发比较，完整场景组件/去最大层，无query-IID显著性
resources_time_cost_and_risk: CPU4/RSS4GiB/AS8GiB/7200s advisory，一次固定T32x9及V32x6，0神经训练
ethics_safety_privacy_or_compliance: 物理输入与评价分离，V不选参；private留服务器，T_test封存
stopping_and_interpretation_rules: 固定终点；真错误保全，无自动重试/扩量/改模型或网格；不利仅否定此配方
status: diagnostic
evidence_status: partial
missing_evidence: [actual_native_state_effect, working_scale_transfer, noise_identification, independent_generalization, credible_peer]
next_skill: record-experiment_after_actual_execution
```
