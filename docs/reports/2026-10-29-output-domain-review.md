# 119：声明输出域的实际记录与深挖

2026-10-29。一次原V32/六臂输出域控制已正常完成，独立保存数组的主要位置核验通过。**WS_NEAR的自身MAE从927.435m变为SPAN的790.264m，但位置覆盖由98.441742%降至96.120284%；共同保留点的状态和误差exact不变。** 因此这不是恢复精度增量。另一方面，SPAN相对UNION多保留1094个参考有效目标位置点，汇总MAE/RMSE反而略低，说明内部缺口与首末支撑域外值得分开研究。

本报告先按record-experiment建立实际事实，再按squeeze-experiment分析选择机制、异质性和发展价值，最后给Mode C输入。没有再次执行真实数据、训练或改变配方。独立核验范围见§4，不将其表述为全部字段或物理真值已验证。

## 1. EXPERIMENT_RECORD

| 字段 | 实际事实 |
|---|---|
| ID / 上游 | `LEVEL1-OUTPUT-DOMAIN-20261029-119-attempt-1`；`IDEA-OA-OUTPUT-DOMAIN-20261029@1.0`；`MC-ASSOCIATION-OUTPUT-ACTION-20261029 / OA-C2` |
| 主要问题 | 固定身份组与位置状态，只改变声明位置可用域，分离全部参考人口下的coverage、共同点风险与丢失点原风险 |
| 设计与实际 | 原115/116事前V32，32场/9共享身份组件；A0/WS既有NEAREST_SUPPORTED各FULL/UNION/SPAN，六臂、四个相对FULL比较；到32终点，无自动扩量 |
| 血缘 | 056 system的Q64 support→114保存A0/WS组→116保存两NEAR完整预测→117原32公开评价绑定→119只改位置mask。原32这些预测实际保存在116，117引用/重放不等于另有一套预测 |
| 输入/权限 | 读缓存reference只绑定/评价；域只读support/groups，六臂先冻结。不按身份/reference/target有效mask构域、删组；不读ZIP/CSV/封存测试 |
| 人口 | 1427个target、91328个target-query；72196个纬经联合参考有效点、19132个缺参考query。所有target都有部分参考，reference_unavailable_target_count=0 |
| 版本/命令 | 冻结`b48dae7`，cwd=`/home/wangguangjie/djs/association-code-run119`；实际argv为`/usr/bin/taskset --cpu-list 4-7 /home/wangguangjie/djs/mtad-field-20261016/bin/python tools/analysis/output_domain_control.py`及原056/114/116/117路径、manifest、全新output和cpu_threads=4，完整命令见原件 |
| 实际终态 | 程序COMPLETED/exit_code=0；launcher COMPLETED/returncode=0；exit_code.txt=0；服务`mtad-f222436a1ffb.service`随后inactive/dead、Result=success、ExecMainStatus=0、Restart=no；启动Linger=yes |
| 时间 | 程序UTC11:29:13.006990→11:30:16.221597（北京时间19:29:13→19:30:16），run.elapsed=63.214665s；summary取finalizing前63.108612s。launcher UTC11:28:57.592464→11:30:20.064413，82.472593s，不混用两个截点 |
| 资源 | CPU4/affinity4–7；RSS2006728704B、AS8589934592B（8GiB）；600s advisory未超。CUDA_VISIBLE_DEVICES为空，0GPU/模型构造/前向/优化，0新score/分组/恢复器/阈值/normalizer |
| 次数/偏差 | 唯一研究run attempt-1，32场无排除/更换域/重算。主任务现场说明：SFTP父目录未建先被拒，建目录后成功；最初taskset非绝对路径在创建record/run前被启动器拒绝，改`/usr/bin/taskset`后创建本次唯一run。这两项是传输/启动准备，不计研究实验重跑；前置事件未写入run，事实来源为主任务现场说明 |
| 缺项 | run/launcher未保存Python/NumPy/Torch确切版本，不拿本机版本补填。真实availability/物理存在及reference生成未知，尚无可信作者共同协议复现或独立泛化 |

事前[Idea](../proposals/output_domain_control_20261029.md)、[规格](../specs/output-domain-control.md)、[实际工具](../../tools/analysis/output_domain_control.py)。原件：[run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/run.json)、[summary](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/summary.json)、[全部32场摘要](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/scene-summary.json.gz)、[实际命令](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/launcher-request.json)、[launcher终态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/launcher-status.json)、[服务终态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/launcher-posthoc-service-terminal.txt)、[成本/终态汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/terminal-synthesis.json)。stdout仅完成路径，stderr/worker.stderr均0B。

每场单独文件未重复下载，全部公开内容保存在scene-summary.json.gz；原坐标、身份、membership和大数组仍在服务器原run/private。

### 1.1 六臂全部主结果

位置单位米；每行参考有效分母均72196。MAE等只在该臂预测可用且参考有效的点上计算，不同域的自身误差不能当同人口改善。完整数值表见本报告附表A。

| 臂 | 可评点/缺预测点 | 点coverage% | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| A0_FULL | 70555 / 1641 | 97.727021 | 996.707 | 2346.966 | 4084.559 | 8458.615 |
| A0_UNION | 65044 / 7152 | 90.093634 | 852.945 | 1772.318 | 3886.762 | 5837.792 |
| A0_SPAN | 65961 / 6235 | 91.363787 | 849.660 | 1764.642 | 3878.758 | 5829.907 |
| WS_FULL | 71071 / 1125 | 98.441742 | 927.435 | 2173.150 | 3873.502 | 7301.504 |
| WS_UNION | 68301 / 3895 | 94.604964 | 794.024 | 1575.740 | 3732.384 | 4706.290 |
| WS_SPAN | 69395 / 2801 | 96.120284 | 790.264 | 1567.163 | 3725.380 | 4695.086 |

| 固定R，三域均不变 | 组数 | duplicate组数 | anchor匹配target / 未匹配target | 有位置覆盖target / 缺位置target |
|---|---:|---:|---:|---:|
| A0 | 1469 | 92 | 1377 / 50 | 1377 / 50 |
| WS | 1605 | 217 | 1388 / 39 | 1388 / 39 |

所有UNION/SPAN组都至少有一个支撑query，没有整组声明域全空，没有新增整条target的位置漏报；本次仅减少每条轨迹可用时刻。duplicate组数未减少，原raw/decoded身份关系无变化。首组被部分缩域也不按真实身份转用后续重复组。

### 1.2 四个比较：拒绝点的原FULL风险

| 相对同R FULL | 丢失目标点 | 其原FULL MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| A0_UNION | 5511 | 2693.467 | 5783.305 | 11281.469 | 30354.027 |
| A0_SPAN | 4594 | 3108.006 | 6315.508 | 12833.169 | 32800.677 |
| WS_UNION | 2770 | 4216.985 | 7742.476 | 15402.667 | 33219.263 |
| WS_SPAN | 1676 | 6606.992 | 9928.280 | 19396.457 | 34134.755 |

四个比较gained_target_points=0、rescued_target_count=0、lost_target_count=0。所有共同点均tied，better/worse=0，signed change/benefit/harm均0；same-anchor为全部共同点，changed-anchor为空。因此“自身MAE下降”的直接原因是有效人口选择，不能归为状态恢复变准。参考有效分母和被拒点原误差均保留，没有从分母删除困难点。

### 1.3 四属性及论文统计

四属性的**state值**均未变；纬经两列有效mask缩域，故其自身统计可变。速度/航向mask未变，逐场all/observed/missing指标与FULL一致，共768个“场景×R×缩域×region×属性”比较块。核验块数不是独立实验数。

| R，三域相同 | 速度单位 / 可评点 | 速度MAE / RMSE | 航向可评点 | 航向MAE / RMSE（度） |
|---|---|---:|---:|---:|
| A0 | native_speed / 70574 | 52.994757 / 62.711407 | 70503 | 80.156258 / 95.402955 |
| WS | native_speed / 71090 | 52.574438 / 62.463070 | 71021 | 80.071176 / 95.386572 |

速度/航向参考有效分母分别72215/72144；native_speed保留继承缓存的评价标签，不擅写m/s或认证物理速度精度。此表由公开逐场N×MAE/MSE重聚合，独立数组审计没有重算速度/航向数值。纬经论文统计见附表B，printed_Eq29_diagnostic完整保留summary；原论文ESD实现歧义仍未解决，residual_SD不当作已核作者ESD。

## 2. SQUEEZE_REPORT：机制与发展价值

### 2.1 操作链与解释边界

`组support→有效mask→保留/拒绝哪些位置点→coverage与条件误差改变`已有直接证据。身份关系、四属性状态、恢复器都未改变。没有学习精度增量、去重、真实存在检测、在线或可信同行胜出证据。

SPAN未声明的首末支撑域外点有较高误差，这是该有限开发人口的条件风险事实，不证明传感器异常、reference错误或外推是唯一物理根因。UNION同时删掉内部无当前报告query，删得更多却不总是更好。

### 2.2 内部缺口不能一律拒绝

| R | SPAN比UNION额外目标点 | coverage增加（百分点） | 额外点MAE | 额外点RMSE |
|---|---:|---:|---:|---:|
| A0 | 917 | 1.270153 | 616.700 | 1090.088 |
| WS | 1094 | 1.515319 | 555.511 | 879.201 |

这些额外点的均值/平方均值由两嵌套域的N×MAE/MSE之差推得：`N_extra=N_SPAN−N_UNION`，`MAE_extra=(N_SPAN×MAE_SPAN−N_UNION×MAE_UNION)/N_extra`，MSE同式，RMSE取平方根。它是事后探索分析；未由原点向量直接提取，不能反推出其P95/P99、最大值或位置。可复核的[公开汇总派生脚本](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/derive-public-review.py)和[结果JSON](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/public-review-derived.json)已保存；脚本仅读公开summary/scene-summary，没有原始坐标或mask。SPAN相对UNION在两R都多保覆盖，汇总MAE和RMSE反而略低。但逐场并非总同向，见§3。

这给出有价值的发展入口：区分**当前报告存在、内部缺口可预测、首末域外外推**，再在完整轨迹方法中联合决定域与状态。不能把所有缺报告时刻都当死亡；也不能因当前只是mask控制就说整个方向无价值。若追求全覆盖，下一解法需要改善高风险域状态而非只拒绝；若允许风险控制输出，必须提前明确漏报代价。MeanTraj的域—条件质心联合优化是成熟参照，本次仅完成A′透明对照的一部分。

### 2.3 预测组域不等于真实身份支撑

评价的固定target支撑层以真实身份全部片段的support并集定义，而方法域只看预测组。UNION仍保留A0 228、WS306个“该target全部输入均无support”的参考有效点；SPAN为708/903，FULL为1045/1044，参考分母都1068。预测组可混入其他身份支撑，所以组内有报告不等于该target有报告，更不等物理存在。这不是域读取目标mask的证据；真实身份仅事后分层，不参与预测。

group-query也不能替代target可评点：A0 FULL/UNION/SPAN声明94016/69721/71362，WS为102720/79415/81090，包含重复组和无有效reference时刻。SPAN比UNION增加1641/1675个声明group-query，对应参考有效target点仅917/1094。

## 3. 异质性、全部组件与不利场景

这些是各臂**自身**误差变化，人口不同，不能写成同点恢复收益。

| R/域对FULL | 自身MAE好/差/同场 | 自身RMSE好/差/同场 |
|---|---:|---:|
| A0_UNION | 19 / 11 / 2 | 16 / 14 / 2 |
| A0_SPAN | 18 / 4 / 10 | 17 / 5 / 10 |
| WS_UNION | 17 / 13 / 2 | 15 / 15 / 2 |
| WS_SPAN | 14 / 6 / 12 | 14 / 6 / 12 |

九组件全部结果见附表C，不只保留有利组件。去最大19场组件后余13场：A0 coverage99.126496→94.419282/95.600129%，MAE1093.225→818.851/817.953m，RMSE3135.050→1735.145/1728.710m；WS coverage100→96.093497/97.177289%，MAE1079.870→844.079/842.103m，RMSE2860.438→1661.321/1655.021m（两缩域依次UNION/SPAN）。总体取舍不只由最大组件产生，仍不保证各场/组件同向。

自身MAE相对FULL变差的全部场景（Δ米）：

- A0_UNION：692(+4.734)、2153(+5.694)、2262(+9.992)、2352(+2.778)、2576(+3.760)、2874(+2.047)、3908(+8.711)、3916(+1.758)、3981(+27.067)、4272(+15.868)、4968(+11.362)。
- A0_SPAN：2576(+.472)、3916(+1.758)、3981(+14.856)、4968(+1.178)。
- WS_UNION：692(+4.734)、1187(+7.123)、2153(+5.694)、2262(+9.992)、2352(+2.778)、2502(+5.718)、2576(+3.760)、2874(+.967)、3619(+5.803)、3908(+7.642)、3981(+11.938)、4272(+27.495)、4968(+12.616)。
- WS_SPAN：1187(+.476)、1582(+.149)、2502(+.231)、2576(+.472)、3619(+.779)、4968(+1.129)。

SPAN相对UNION的自身MAE：A0 24好/4差/4同，差场864、1582、1840、3956；WS 24好/5差/3同，差场1582、2416、2479、3274、3956。RMSE为A0 25好/3差/4同、WS26好/3差/3同。尤其WS的`component-a077aa3eb774a8b670334e7c`在SPAN比UNION更高MAE/RMSE，桥接内部缺口也有局部代价。全部自身RMSE变差scene留在32场原件；共同点误差在所有32场则始终0差。

## 4. 核验范围、偏差与限制

- **运行内：** 32/32原116完整artifact exact重放并绑定117报告；非mask数组/速度航向mask和指标不变；原组、首代表、reference分母固定；共同点delta exact0、gained0。
- **本地公开数字重聚合：** 六臂8个主要位置整数分母/计数、按N加权MAE/MSE从32场汇总与summary一致；768个速度/航向比较块一致；各场population与FULL一致。该重聚合不独立重算P95/P99或原状态/mask。
- **独立原数组核验：** [独立审计原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/119/independent-saved-array-audit.json)实际一次exit0，32场/9组件/六臂/四比较，5781个数值统计。核逐scene、全32、组件及最大组件外的own/common主要位置计数、MAE/MSE/RMSE/P95/P99/coverage，gained/lost/整target变化、固定reference/结构、嵌套mask、共同点exact0及被拒点原风险。最大MAE差7.929e−10m、RMSE7.353e−10m、P99 1.795e−9m、MSE1.032e−5m²；容差rtol1e−9、米atol1e−5、MSE atol.001、数组exact含NaN。耗时2.615090s、RSS79986688B，属于事后审计成本，不冒充主程序峰值/耗时。
- **未独立核验：** source support到域mask构造、输入provenance、manifest/组件身份绑定、paper指标、observed/support/anchor等分层、逐点benefit/harm和速度航向数值，以及物理reference真实性/可信同行/泛化。不能写成“全部字段独立通过”。
- **设计限制：** 原V32多轮开发且跨场共享身份，仅确定性一次；无新训练、seed重复或收敛曲线可分析。额外点矩、跨场符号与组件讨论为事后分析，不倒填事前主假设。不按结果选最优域、改阈值、扩量或弃场。
- **在线/轻量：** 当前Q64 OLS/插值和保存分组使用未来信息，不能认证在线。0参数/0GPU只表示本控制无新增网络，不代表未来整套方法已低延迟。用户精度与真实价值优先，轻量/在线是附加价值，离线或较重成熟方法仍可比较。

## 5. 知识沉淀与具体发展价值

已回答的是输出声明的覆盖—条件风险，不是恢复器精度。值得进一步发展的，是在同一完整对象空间分开报告域、可预测域和物理存在，随后让域与状态相互约束；或者保留全域，针对高风险外推建立更合适的动态/观测条件。下一验证需要明确漏报代价或固定覆盖约束，不能仅优化缩域后的MAE。

对应成熟解法有两种不同发展：MeanTraj式域—条件质心/对应联合优化，或固定候选下的组条件状态证据和输出决定。前者可改变估计对象，后者可学习何时保留/拆分/拒配；都需将适配A′与核心机制B的增量分开。当前mask只建立透明对照，不自动等于任一成熟方法的复现。

本次到固定32终点结束，不直接挑SPAN为已证最终模型，也不自动扩量、训练存在网络或追加域阈值。新解法先由本次Record/Squeeze回流Mode C，再形成独立Idea。

## 6. 标准交接与Mode C候选

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
record_id: EXP-OUTPUT-DOMAIN-119-20261029
experiment_id: LEVEL1-OUTPUT-DOMAIN-20261029-119-attempt-1
source_idea_id: IDEA-OA-OUTPUT-DOMAIN-20261029
plan_version: '1.0'
execution_status: completed-exploratory
research_question: 固定状态只改位置声明域的完整coverage与条件风险
hypothesis: 首末域外和内部缺口风险可能不同；真实存在语义待核
planned_vs_actual_design_and_controls: 原V32/两R的NEAR/三域六臂四比较，到32终点，0恢复重算；启动准备路径处理见§1
actual_configuration: b48dae7冻结，CPU4/affinity4–7/RSS2006728704B/AS8GiB/600s advisory，0GPU训练
samples_materials_or_data_provenance: §1原056support→114组→116保存状态→117原32评价→119位置mask
results_with_uncertainty: 完整六臂/四比较/分母、九组件/负场景；一次开发集描述，无IID CI或独立泛化
all_runs_or_repetitions: 一个attempt-1，32场唯一研究run；准备路径拒绝非研究重跑；事后独立审计另计
observations_and_deviations: 所有共同点误差0，缩域自身误差下降伴覆盖损失，跨度保内部缺口有额外价值
evidence_locations: 本报告§1原件与§4独立审计；全部32场未省略
known_limits_and_missing_evidence: source域构造等未独立重审；存在语义、物理reference、可信同行/独立泛化未知
next_routes: squeeze-experiment已完成→deep-read Mode C
next_skill: deep-read Mode C
```

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-OUTPUT-DOMAIN-119-20261029
input_experiment_record_ids: [EXP-OUTPUT-DOMAIN-119-20261029]
claim_scope: 原V32/固定A0-WS_NEAR状态，只改位置声明域的coverage与条件风险
data_quality_and_provenance_gaps: 主要保存数组审计通过，具体未核范围见§4；物理reference与真实存在未知
primary_result_with_uncertainty: WS FULL/UNION/SPAN MAE927.435/794.024/790.264m，coverage98.441742/94.604964/96.120284%，共同点误差差0
process_dynamics: 一次32终点，程序63.214665s/launcher82.472593s，事后独立审计2.615090s，0训练
mechanism_and_intermediate_findings: 有效mask改变样本构成、state不变；SPAN保内部缺口拒域外，组support不等真实target支撑
errors_heterogeneity_and_boundary_conditions: §3九组件/全部自身MAE负场；去最大19场仍有总体取舍，不证明每场获益
robustness_sensitivity_and_alternative_explanations: 固定首代表/参考分母，所有共同点exact0；错组/先验/reference语义仍可混淆
new_P: none；细化已有输出风险与报告事件问题
new_F: OD-F1：固定状态三声明域产生覆盖—风险前沿；首末域外拒绝点高误差，保内部缺口比仅support保留更多且汇总条件误差略低
new_M: OD-M1：mask→保留/拒绝人口→coverage与条件误差；直接操作已测，物理存在/误差根因仍UN
evidence_index: 本报告§1–4及119全部原件；额外内部缺口矩由汇总差推导
open_questions: 如何合法估可预测域/存在并联合状态；能否改善域外而非只拒绝；如何分离错组与域风险
next_routes: deep-read Mode C回流，之后各独立Idea；无自动后继
next_skill: deep-read Mode C
```

OD-F1/OD-M1为建议整合的一条F和一条M局部编号，不与全局数重复相加；本次证据是119，不把117再算一次独立实验。

## 附表A：全组声明与无有效参考的预测点

全组query包括重复组和无参考时刻，不与目标可评点混用；“预测无有效目标参考”不能全部称false positive。

| 臂 | 全组query | 声明位置group-query | 全无声明组 | 预测无有效目标参考点 |
|---|---:|---:|---:|---:|
| A0_FULL | 94016 | 94016 | 0 | 23461 |
| A0_UNION | 94016 | 69721 | 0 | 4677 |
| A0_SPAN | 94016 | 71362 | 0 | 5401 |
| WS_FULL | 102720 | 102720 | 0 | 31649 |
| WS_UNION | 102720 | 79415 | 0 | 11114 |
| WS_SPAN | 102720 | 81090 | 0 | 11695 |

## 附表B：纬经论文口径

单位10⁻⁴度，scene等权后target平均；指标人口随域变化。ME保留符号，residual_SD不冒充已核作者ESD。

| 臂 | 纬ME | 纬MAE | 纬residual_SD | 经ME | 经MAE | 经residual_SD |
|---|---:|---:|---:|---:|---:|---:|
| A0_FULL | -19.141531 | 56.475045 | 77.871289 | -13.392828 | 59.794269 | 82.795937 |
| A0_UNION | -17.479924 | 49.150292 | 69.105000 | -18.017804 | 50.634745 | 71.200826 |
| A0_SPAN | -17.480043 | 49.107223 | 69.206071 | -17.975351 | 50.617009 | 71.295360 |
| WS_FULL | -16.117392 | 52.300791 | 73.260841 | -12.423073 | 56.953095 | 79.186844 |
| WS_UNION | -16.918731 | 47.447626 | 67.673144 | -16.003746 | 48.530689 | 68.712071 |
| WS_SPAN | -16.938712 | 47.372628 | 67.733565 | -15.991258 | 48.479817 | 68.769778 |

WS缩域后的经纬signed ME绝对值均比FULL更大，尽管自身坐标MAE更低；bias与绝对误差仍需分开判断。

## 附表C：全部九组件

表内顺序均FULL/UNION/SPAN。完整组件ID与全部scene成员保留summary；这里展示ID末8字符。组件场数不等独立seed。

**A0：**

| 组件末8字符（场数） | 可评点 F/U/S | 自身MAE F/U/S（米） | 自身RMSE F/U/S（米） |
|---|---|---|---|
| 74ed5828（1） | 248 / 221 / 221 | 837.953 / 837.134 / 837.134 | 1399.419 / 1427.532 / 1427.532 |
| 7254ffc1（19） | 58299 / 53370 / 54141 | 976.416 / 860.402 / 856.583 | 2144.764 / 1780.345 / 1772.390 |
| 7e382451（1） | 2298 / 2149 / 2183 | 810.539 / 719.856 / 714.765 | 1696.057 / 1540.035 / 1529.279 |
| 21826aea（1） | 204 / 184 / 185 | 2176.088 / 1221.616 / 1215.995 | 4237.568 / 2149.039 / 2143.264 |
| 70334e7c（1） | 713 / 677 / 691 | 1647.270 / 921.928 / 920.477 | 5641.763 / 3029.578 / 3002.108 |
| f715f00b（1） | 186 / 185 / 185 | 548.684 / 550.442 / 550.442 | 994.014 / 996.562 / 996.562 |
| 04db48b9（6） | 6463 / 6168 / 6254 | 1183.552 / 823.612 / 824.693 | 3524.017 / 1644.209 / 1640.130 |
| d2a1a9c5（1） | 2020 / 1968 / 1977 | 895.200 / 836.855 / 835.781 | 1723.550 / 1612.937 / 1609.998 |
| 1c67e3bb（1） | 124 / 122 / 124 | 1210.124 / 1225.992 / 1210.124 | 2121.949 / 2139.047 / 2121.949 |

**WS：**

| 组件末8字符（场数） | 可评点 F/U/S | 自身MAE F/U/S（米） | 自身RMSE F/U/S（米） |
|---|---|---|---|
| 74ed5828（1） | 248 / 242 / 248 | 602.099 / 598.176 / 602.099 | 912.521 / 914.440 / 912.521 |
| 7254ffc1（19） | 58707 / 56420 / 57380 | 895.331 / 783.484 / 779.410 | 1998.496 / 1557.119 / 1548.135 |
| 7e382451（1） | 2406 / 2311 / 2323 | 1105.476 / 1001.033 / 997.132 | 2122.780 / 1934.053 / 1929.246 |
| 21826aea（1） | 204 / 184 / 185 | 2176.088 / 1221.616 / 1215.995 | 4237.568 / 2149.039 / 2143.264 |
| 70334e7c（1） | 713 / 677 / 687 | 711.274 / 585.655 / 591.646 | 1390.121 / 1015.637 / 1017.505 |
| f715f00b（1） | 186 / 170 / 170 | 702.111 / 458.949 / 458.949 | 1334.709 / 733.644 / 733.644 |
| 04db48b9（6） | 6463 / 6216 / 6293 | 1121.447 / 772.402 / 771.725 | 3394.260 / 1438.854 / 1434.845 |
| d2a1a9c5（1） | 2020 / 1965 / 1985 | 1044.239 / 1002.452 / 998.898 | 2217.910 / 2151.017 / 2141.491 |
| 1c67e3bb（1） | 124 / 116 / 124 | 834.548 / 862.042 / 834.548 | 1670.101 / 1722.591 / 1670.101 |

