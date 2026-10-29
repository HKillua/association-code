# 123：LOW/FixedLOW全V扩展的真实记录与深挖

2026-10-29；EXPERIMENT_RECORD＋SQUEEZE_REPORT，research-pipeline-contract v1.0。真实completed/exit0，一次attempt-1。来源Idea `IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029`/1.0，消费Mode C1.41/1.42的SC-C3、F219/M90/E117；[执行前合同](../proposals/source_preference_full_control_20261029.md)及[L2S](../../HandOver/L2S-20261029-123.md)。

## 1. 设计、实际输入与血缘

- 固定056 Q64/114 WS/manifest、FULL输出与首代表。原121的32×10＝320臂及120其余268×8＝2144臂直接导入；只新增536臂，总2464导入/536新算。旧运行不改、不重跑、不重拟合中心/Huber；全部十臂45配对。
- LOW沿原source字符串排序和support选贡献；FixedLOW仅复用原冻结中心修正高source。高源首anchor的no_support仍随相应输入校正，不能混称原始anchor回退。速度/航向复制原Mean。reference/identity只用于预测完成后的评价绑定，真实target support只作事后分层。
- 原32、其余268、其中204场/7已见组件和64场/29未见组件、全300/38组件与去最大组件分别统计。反复开发且共享身份，非独立测试；无可信同行资格、0神经训练/模型/前向/GPU/新scores/分组/阈值/ZIP/CSV/T_test。
- 原32十方法及45pair公开统计与121整个对应JSON子树全等（独立只读核16545叶项、0差异）；真实原artifact还由导入支路逐字段exact绑定。此项不等独立审过预测构造/物理语义。

## 2. 同人口主要结果（位置单位米）

### 原32：32场，同71071点

| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---:|---:|---:|---:|---:|
| MEAN | 1276.328389 | 2168.265557 | 3103.794978 | 7194.824823 | 1249.718245 |
| FIXED_MEAN | 656.041340 | 1864.654434 | 2444.380637 | 7541.587652 | 484.962267 |
| LOW_SOURCE_SUPPORTED | 527.446344 | 1813.303585 | 2942.040400 | 7260.393881 | 478.072440 |
| FIXED_LOW_SOURCE_SUPPORTED | 447.060488 | 1740.813819 | 1012.562755 | 7526.008427 | 382.696475 |
| ANCHOR | 844.759832 | 2457.606705 | 4385.888699 | 10460.984233 | 986.493191 |
| NEAREST_SUPPORTED | 927.434649 | 2173.150207 | 3873.502039 | 7301.504313 | 886.352046 |

### 新268：268场，同370073点

| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---:|---:|---:|---:|---:|
| MEAN | 1297.482127 | 3673.969221 | 2606.626670 | 8648.581954 | 1490.401755 |
| FIXED_MEAN | 602.832905 | 3492.696837 | 2064.414375 | 8732.621284 | 649.380649 |
| LOW_SOURCE_SUPPORTED | 579.613744 | 3509.063396 | 2408.549786 | 8715.862100 | 697.663114 |
| FIXED_LOW_SOURCE_SUPPORTED | 500.498213 | 3469.646494 | 796.835461 | 8753.977611 | 607.491494 |
| ANCHOR | 918.698989 | 6819.161169 | 4043.090729 | 12046.178489 | 1102.126016 |
| NEAREST_SUPPORTED | 971.698993 | 3688.155523 | 3864.850322 | 8777.496698 | 1128.715712 |

### 新268已见组件：204场，同338944点

| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---:|---:|---:|---:|---:|
| MEAN | 1252.106650 | 3566.320294 | 2513.213971 | 7405.533975 | 1419.197272 |
| FIXED_MEAN | 567.359026 | 3388.826278 | 2013.609101 | 7557.628477 | 579.681743 |
| LOW_SOURCE_SUPPORTED | 536.912213 | 3397.068430 | 2263.986708 | 7489.681329 | 621.657055 |
| FIXED_LOW_SOURCE_SUPPORTED | 460.496784 | 3363.329800 | 701.076232 | 7591.102012 | 533.884640 |
| ANCHOR | 869.874530 | 6953.764254 | 3817.572120 | 11117.687686 | 1032.581019 |
| NEAREST_SUPPORTED | 927.606212 | 3579.528640 | 3809.707792 | 7612.552282 | 1053.261661 |

### 新268未见组件：64场，同31129点

| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---:|---:|---:|---:|---:|
| MEAN | 1791.547002 | 4688.768062 | 4160.840135 | 20177.619345 | 1717.366048 |
| FIXED_MEAN | 989.085541 | 4470.100899 | 3496.983439 | 20171.405366 | 871.545913 |
| LOW_SOURCE_SUPPORTED | 1044.563719 | 4553.587129 | 4112.801906 | 20380.113846 | 939.932426 |
| FIXED_LOW_SOURCE_SUPPORTED | 936.048487 | 4466.363678 | 3007.503611 | 20200.833246 | 842.113339 |
| ANCHOR | 1450.317649 | 5129.839257 | 6933.802265 | 22577.790106 | 1323.800694 |
| NEAREST_SUPPORTED | 1451.797400 | 4711.526921 | 4473.235512 | 20311.034241 | 1369.225499 |

### 全300：300场，同441144点

| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---:|---:|---:|---:|---:|
| MEAN | 1294.074130 | 3475.751966 | 2671.411018 | 8363.726668 | 1464.728848 |
| FIXED_MEAN | 611.405110 | 3285.386489 | 2137.820266 | 8453.399103 | 631.842689 |
| LOW_SOURCE_SUPPORTED | 571.209256 | 3295.370947 | 2466.470146 | 8423.625592 | 674.240108 |
| FIXED_LOW_SOURCE_SUPPORTED | 491.889069 | 3253.797868 | 826.097676 | 8485.153495 | 583.513358 |
| ANCHOR | 906.786938 | 6323.165622 | 4103.137557 | 11749.948644 | 1089.791848 |
| NEAREST_SUPPORTED | 964.567736 | 3488.822636 | 3866.365778 | 8503.413913 | 1102.863588 |

### 去最大组件：170场，同109421点

| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---:|---:|---:|---:|---:|
| MEAN | 1671.815414 | 5013.248058 | 3756.734221 | 15939.983891 | 1683.751877 |
| FIXED_MEAN | 821.941733 | 4832.739832 | 2607.753708 | 15924.809558 | 801.580806 |
| LOW_SOURCE_SUPPORTED | 886.710159 | 4888.627527 | 3702.464141 | 16042.416325 | 868.150682 |
| FIXED_LOW_SOURCE_SUPPORTED | 780.793668 | 4840.889388 | 1824.552016 | 15935.824186 | 768.775087 |
| ANCHOR | 1365.814682 | 5418.356853 | 6754.080115 | 18928.901318 | 1375.068539 |
| NEAREST_SUPPORTED | 1311.861946 | 5032.295733 | 4245.130804 | 16100.266392 | 1310.227438 |

## 3. 全部异质性与人口

| 条件/比较：FixedLOW相对基线 | MAE差 | 场景好/差/同 |
|---|---:|---:|
| 新268/WS_FIXED_MEAN | -102.334692 | 91/177/0 |
| 新268/WS_LOW_SOURCE_SUPPORTED | -79.115531 | 252/1/15 |
| 新268/WS_ANCHOR | -418.200775 | 259/6/3 |
| 新268/WS_NEAREST_SUPPORTED | -471.200779 | 268/0/0 |
| 未见组件64/WS_FIXED_MEAN | -53.037054 | 15/49/0 |
| 未见组件64/WS_LOW_SOURCE_SUPPORTED | -108.515232 | 59/0/5 |
| 未见组件64/WS_ANCHOR | -514.269162 | 61/1/2 |
| 未见组件64/WS_NEAREST_SUPPORTED | -515.748913 | 64/0/0 |
| 全300/WS_FIXED_MEAN | -119.516041 | 103/197/0 |
| 全300/WS_LOW_SOURCE_SUPPORTED | -79.320188 | 281/1/18 |
| 全300/WS_ANCHOR | -414.897869 | 291/6/3 |
| 全300/WS_NEAREST_SUPPORTED | -472.678667 | 300/0/0 |

全300相对FixedMean：18/38组件好、20差；相对LOW：35好、0差、3同。最大组件130场/331723点（75.1961%）；去掉后LOW的MAE/RMSE/P95/P99均劣于FixedMean，未见组件64场同样如此。FixedLOW虽仍改善MAE，去最大组件RMSE4840.889388高于FixedMean4832.739832；不能把点总体优势写成每场或每组件普遍优势。

| 全300分层 | 点数 | FixedLOW MAE | FixedLOW RMSE | FixedLOW P99 |
|---|---:|---:|---:|---:|
| all | 441144 | 491.889069 | 3253.797868 | 8485.153495 |
| inside_support | 435032 | 483.446365 | 3261.806507 | 8503.703663 |
| outside_support | 6112 | 1092.812868 | 2621.663638 | 7801.778158 |
| predicted_group_both_sources | 265135 | 283.298943 | 2416.536540 | 2088.597392 |
| predicted_group_low_only | 141677 | 267.419266 | 3259.770566 | 1065.152221 |
| predicted_group_high_only | 18088 | 1428.831476 | 3988.395793 | 13751.153663 |
| predicted_group_no_support | 16244 | 4810.981468 | 9045.082891 | 35143.043240 |

- 全V9100目标/9779组/1068重复，1目标无可用reference；reference有效454952点、共同441144点、缺失13808，coverage96.964955%。新268为7673目标/8174组/851重复，未改变完整组/valid/首代表；所有配对gained/lost均0。预测组四种support分层不等真目标support内外。
- FixedLOW相对LOW的新268主MAE差−79.115531m，相对FixedMean−102.334692m；但P99分别+38.115511/+21.356327m。全300对LOW P99+61.527903m，对FixedMean+31.754392m。校正增量延续，极尾代价也延续。
- 真目标支撑外6112点FixedLOW MAE1092.813m仍高于Anchor764.687m；RMSE2621.664m低于Anchor2939.826m。不可用真身份构造部署门控，更不能只报一个有利指标。完整逐场/38组件/45配对和所有空层在原JSON，未删坏例。

## 4. 实际执行、成本与核验范围

- 冻结770a9bd、独立服务器`association-code-run123`，服务`mtad-473c05249036.service`；原run与launcher真实退出0。程序693.469159s，launcher711.930098s，RSS3071172608B；CPU0–3/4线程、RSS4GiB/AS8GiB、1800s advisory，没有超建议。完整输入/环境/真实命令/起止/退出在原request/run/status。
- 真实服务inactive/dead、Result=success、ExecMainStatus0、Restart=no、Linger=yes；冻结树干净。未保留SSH值守/守护或建立自动后继。
- 独立NumPy保存数组审核全部300场/10臂/45配对、38组件和5子集共44聚合；不导入生产评价/预测器。398770数值、1529599exact及2730null检查通过，最大MAE/RMSE差2.303e−9m/MSE差3.749e−5m²。程序75.849355s、独立launcher77.507565s/RSS644378624B，另计成本，审计服务mtad-9a221d917b72成功终态。
- 未独立审核：中心拟合/物理bias、真实reference/组件语义、support条件路由、论文坐标指标/速度航向、逐点得失次数；软件fixture只核导入/扩展/单源/空support等实现，不算科学成绩。
- 来源字段兼容说明：123逐场复用121分析器的`original118_artifacts_exact`旧字段名，新268实际校验的是传入的120八臂，并非声称118曾计算那268场；真实来源以run的control_run/pilot_run、逐场stratum及导入方法清单为准。原冻结产物不改名或重跑，这个字段不能用于推算原118人口。
- 计划与实际一致，无失败尝试、重试、选参、排除、额外训练或扩封存范围。所有十臂/45比较保留，非最佳臂报告。

## 5. Squeeze：回答到哪里，机制仍未知

FixedLOW对LOW唯一较差场3175的MAE差+41.905981m。high_only中15331点好/2679差/78同，MAE改善但P95/P99分别差434.060/702.442m；no_support另有193点好/7差/16044同。它们是保存评价统计的作用域描述，不能直接证明源噪声或中心方差。

**DE：** 经验LOW及校正LOW在更广开发人口仍有价值；但“LOW普遍胜FixedMean”不成立，未见组件及去最大组件四指标均反向。FixedLOW对FixedMean在全点和场景等权MAE有利，却197/300场、20/38组件差，P99也差；需要研究条件贡献和幅度，而非预定硬优先最终骨架。

**AI/UN：** 原始观测质量、共同偏差、错关联、运动/插值、reference生成和共享中心误差均可能参与。中心组数不是meff；本实验没有识别每源方差或“小样本中心过拟合”根因，没有建立在线或可信同行胜负。

**下一步：** 把本结果先回流Mode C，再执行已经在3de43f4、查看123结果前冻结的15臂soft/收缩2×2；0.75/0.5/m0=4不据本结果改变。保留全部强控、主效应/交互、尾部/作用域和组件。输入/时序/关联/输出按各自条件独立发展，不当作已验证权重的附属模块。

## 6. 证据与标准回流字段

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-123-20261029
source_idea_id: IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029
plan_version: '1.0'
execution_status: completed
research_question: 固定源优先及校正在新268/未见组件的范围与代价
hypothesis: 固定经验配方的效果可延续；物理机制不预设
planned_vs_actual_design_and_controls: §1及执行前Idea，十臂45配对/2464导入536新算，未偏离
actual_configuration: §1/4及原request/run，冻结770a9bd，CPU4/RSS4GiB/AS8GiB/1800s advisory
samples_materials_or_data_provenance: §1及原056/114/120/121/manifest，重复开发V300
results_with_uncertainty: §2/3全部范围/场景组件/尾部，单次确定性无query-IID推断
all_runs_or_repetitions: 一次attempt-1/exit0；一次独立保存数组审计另记成本
observations_and_deviations: §3/5，计划与实际一致，无隐蔽选参排除
evidence_locations: §6原件与服务器完整run
known_limits_and_missing_evidence: §4/5未审范围与物理机制/可信同行缺口
next_routes: squeeze-experiment
next_skill: squeeze-experiment
```

[必要脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/123/)含run/summary/全部逐场、原launcher七文件、服务终态及独立audit/launcher；原坐标、身份、中心、权重和大预测只留服务器原run，不复制到Git。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-SOURCE-PREFERENCE-FULL-123-20261029
input_experiment_record_ids: [EXP-123-20261029]
source_idea_id: IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029
plan_version: "1.0"
execution_status: completed
claim_scope: 固定WS/Q64开发V300的源优先与共享校正范围，非同行或物理机制证明
data_quality_and_provenance_gaps: 主位置保存数组独立核验限上述范围；真实参考生成/source质量未知
primary_result_with_uncertainty: 新268同370073点FixedLOW相对LOW/FixedMean MAE差-79.115531/-102.334692m，252/1/15与91/177/0场好差同
process_dynamics: 固定单次终点，无神经训练/score重算，成本见§4
mechanism_and_intermediate_findings: 校正与来源贡献的条件作用延续，真实方差与中心误差AI/UN
errors_heterogeneity_and_boundary_conditions: 未见及去最大组件LOW输FixedMean；FixedLOW多数场景/20组件差及P99代价
robustness_sensitivity_and_alternative_explanations: 原32统计exact复用；其余268/未见组件/去最大组件完整保留，非独立测试
new_P: none，沿用P02/P11/P14
new_F: 固定源优先与校正贡献的更广范围及反向层
new_M: 条件贡献与gauge作用的操作线索，物理原因UN
evidence_index: 本文§6原件、L2S/S2L及原服务器runs
open_questions: 合法条件混合/幅度、原生状态、同信息关联与对象输出、可信同行
next_routes: deep-read Mode C后独立15臂验证
next_skill: deep-read Mode C
```
