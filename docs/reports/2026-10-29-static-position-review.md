# 115静态分区位置链：实际记录、独立重聚合与取舍

2026-10-29。**WS的身份关系增量没有成为位置“净胜”：共同69837点的MAE由1219.731升至1268.202m，RMSE由2240.358降至2159.927m；自身覆盖提高，但重复输出92→217。** CL覆盖提高而池化MAE/RMSE更差；CC明显损失覆盖和位置质量。全部正面、不利和无变化结果保留，不据单指标否定静态分区/后续条件方法的价值。

本文按`record-experiment → squeeze-experiment → Mode C输入`处理115原件；只做本地公共产物重聚合与审计代码阅读，不读新数据、SSH、模型或更改主矩阵/状态。执行状态只见主入口。距离单位米；cached reference不是独立认证的物理真值。

## 1. EXPERIMENT_RECORD

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-STATIC-PARTITION-POSITION-20261029-115
package_id: EXP-STATIC-PARTITION-POSITION-20261029-115
source_idea_id: IDEA-STATIC-PARTITION-POSITION-CONTROL-20261029
plan_version: '1.0'
created_at: '2026-10-29'
execution_status: completed
research_question: '114固定分区经同一ordinary Mean真正重算后，位置误差/尾部/覆盖是否有增量'
hypothesis: '身份关系可改变成员平均/无支撑回退/首anchor对应；关系提升不保证位置风险同向'
planned_vs_actual_design_and_controls: '§1.1；固定114原pilot32、四臂及同源A0 exact控制，没有扩量/重抽'
actual_configuration: '§1.1–1.2；同056 canonical_all Q64/cache、固定mean_supported与anchor_identity_first'
samples_materials_or_data_provenance: '§1.2；同114原V32/9组件，056完整target/cache与114实际membership'
results_with_uncertainty: '§2–5；自身/每对共同分母、32场/9组件异质性，无IID CI/独立泛化'
all_runs_or_repetitions: '115单次attempt1；posthoc只重算已存artifact误差，0预测重跑，见§1.3'
observations_and_deviations: '§1.1无科学范围偏差；§3–5指标方向与覆盖/重复取舍'
evidence_locations: 'E115-RUN/SUM/ROWS/REQ/STATUS/EXIT/AUDIT/CODE/PLAN，见§6'
known_limits_and_missing_evidence: '旧速度单位不解释；同reference/重复开发V32；本地不持逐点向量；无可信同行或条件B证据'
next_routes: '本文件Squeeze后回Mode C；再设计查询贡献与代表控制，不自动训练'
next_skill: squeeze-experiment
```

### 1.1 计划与真实执行

消费[Idea1.0](../proposals/static_partition_position_control_20261029.md)、[L2S-115](../../HandOver/L2S-20261029-115.md)、[114记录](2026-10-29-static-partition-review.md)及Mode C1.31的单个C53。114原pilot由同manifest、seed20261029预先抽32场；本次不按WS赢的场、误差或组件再筛选。四臂A0/WS/CL/CC全部重算，没有调用六臂中两个来源顺序反控，也没有选择赢家为最终模型。

| 计划项 | 实际事实 |
|---|---|
| 固定人口/终点 | 原114事前V32，32场、9个共享身份组件、3851个输入节点；完整target registry为1427个**场内身份出现**，不是跨场唯一船数；0扩量/重抽 |
| 输入/分组 | 原056 association/system缓存和114实际private membership；原source metadata/raw各scope/node/source/A0和完整registry/ref/mask绑定通过。32/32行均声明A0 exact及raw/membership核验通过 |
| 真正恢复干预 | 各臂从同一Q64片段状态调用`ordinary_fusion`，按新groups重算mean_supported；没有拼旧组坐标。四分区均数学合法，成员/组以原索引排序，首成员作为anchor |
| 回退/对应 | 有支撑成员按mask平均，经度/航向用生产圆周规则；无成员支撑时回退该首anchor OLS；零合向量也保原anchor回退。评价固定`anchor_identity_first`，身份/reference不选组/均值/代表 |
| A0控制 | fragment原mean_supported、membership/offset/flat及完整system artifact逐数组exact；摘要沿事前rel1e−10/abs1e−9，未因结果放宽 |
| 参考/属性 | 确实读取cached reference；0原reference CSV/ZIP/原生点/权重/模型。位置只要求lat/lon有效，不要求速度/航向有效，旧速度单位不解释或换成107新单位 |
| 完整分母 | 1427×64=91328个声明target-query，位置reference可用72196、不可用19132（20.9487%）；缺预测、重复组、预测无可用target、null均保留 |
| 预算/终态 | 一次COMPLETED/exit0；0模型构造/前向/训练/GPU/新score/normalizer，原分区不重算。固定32终点，没有科学范围/样本/容差偏差证据 |

**干预同时改变成员、均值/回退和首anchor到target的对应。** 总体变化可归这套固定静态分区＋ordinary规则，不可直接说是“身份更正确”单独导致，更不能归给Gelato/条件状态B或学习融合。

### 1.2 真实命令、时间与资源

实际worktree为`association-code-run115`；解释器`/home/wangguangjie/djs/mtad-field-20261016/bin/python`，调用[static_partition_position_control.py](../../tools/analysis/static_partition_position_control.py)，输入原056/system、114/attempt1、原manifest，输出新115/attempt1。真实argv/cwd/env保留E115-REQ，不用本地命令倒填。

systemd payload **UTC2026-10-28 21:35:22.323391–21:36:35.864763**（北京时间10月29日05:35:22–05:36:35），launcher73.715483s；程序run自身记录55.885810s，summary写入前55.855168s。开始记录与结束收尾截点不同，不能把二者差值都叫算法耗时。CPU4/affinity0–3，OMP/MKL/OPENBLAS各4、CUDA不可见；RSS2,006,282,240B＝1913.339844MiB，AS8GiB、RAM预算4GiB。600s advisory未触发，退出文件0、未超时。公有原件未给完整Python/NumPy/Torch版本及code SHA，不以当前HEAD补填。

血缘：同056已存canonical_all片段插值/OLS状态＋完整target缓存 → 114实际分区 → 真正重新求各组ordinary Mean → 固定首anchor对应/完整target评价 → 公共逐场/组件/总表；原artifact、私有成员、坐标、pair mask保留服务器，不回传本报告。

### 1.3 两层独立复核及严格边界

**公共产物独立重聚合：** 本地重读E115-RUN/SUM/ROWS/REQ/STATUS/EXIT，按原manifest重建V32，名单与32行顺序一致；32行A0/raw绑定标记全真。逐场重建全4臂、全部9组件，再由组件回总表，覆盖population、8类位置计数、MAE/MSE/RMSE/coverage、固定strata、每对共同点、gained/lost、同/变anchor及signed差值。**7033项计数/数值恒等式通过**，最大浮点累加差7.451×10^-9，未平均各场RMSE或分位数；这是含重复字段的算式检查数，不是独立样本/机制证据数。还核：自身有效点＝共同＋gained/lost；MAE误差和与SSE亦能由该分割重组；inside＋outside、same＋changed均回共同all；better/worse/tie数回共同分母。

**服务器事后逐点距离核验：** 根线程另在同32场已存private artifact上执行[E115-AUDIT脚本](#6-证据索引)，没有重跑预测。直接读lat/lon、各自valid，重新构造3对共同mask/gained/lost，再重算**4个自身＋3对×2侧＝10个人口**的count/MAE/MSE/RMSE/P95/P99；60个标量与主summary一致，audit报告差值均0，0.573488s。半径6371000m，固定rel1e−10/abs1e−9；本地已核audit JSON这60字段对主表exact。因此下文分位数有逐点服务器审计支持，**不是从32场分位数平均/本地凭摘要重建**。

该审计是“不调用生产函数的单独代码重算”，不是新GT或公式多样性验证：原audit注释/JSON写生产为arcsin，但当前`fusion_controls.attribute_errors`实际也用atan2-haversine。原审计文件保留，根线程另存[明确订正](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/posthoc-distance-audit-correction.json)，数值不变/不重跑。两份代码/同一artifact/同一reference能核实现和分位数，不能独立认证reference物理真实性。固定strata与same/changed的分位数仍仅为原运行输出，十人口audit不覆盖它们；本报告对这些层只独立重聚合计数/一二阶矩。

## 2. 四臂全部主要结果与人口

位置MAE/RMSE为各方法自身有效reference交点上的条件误差；不把缺失填0。预测有效点包含重复组和参考无效query，`prediction_without_target`并非原pair FP或真实物理虚警目标数。

| 臂 | 自身交点n | MAE m | RMSE m | P95 m | P99 m | 位置覆盖% | 场景等权MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 70555 | 1215.514 | 2233.713 | 3359.146 | 7358.117 | 97.727021 | 1253.017 |
| WS | 71071 | **1276.328** | **2168.266** | 3103.795 | 7194.825 | **98.441742** | **1249.718** |
| CL | 71946 | 1278.512 | 2370.758 | 3399.791 | 9231.506 | **99.653720** | 1279.859 |
| CC | 24347 | 2060.818 | 4302.317 | 6579.074 | 20867.029 | 33.723475 | 1771.783 |

四臂的available_target_count均72196、available_target_trajectory_count均1427；场景等权MAE均定义32/32。WS自身point MAE比A0差**60.815m**、RMSE好**65.447m**，覆盖增加**0.714721个百分点**；场景等权MAE则好3.298m。两种权重对应不同估计对象，都保留，不能只选好看的均值。

| 臂 | groups | 匹配target | 未匹配target | duplicate组 | 有效预测点 | 缺预测的有效ref点 | prediction_without_target点 | target轨覆盖% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1469 | 1377 | 50 | 92 | 94016 | 1641 | 23461 | 96.496146 |
| WS | 1605 | 1388 | 39 | **217** | 102720 | 1125 | **31649** | 97.266994 |
| CL | 1762 | 1418 | **9** | **344** | 112768 | **250** | 40822 | **99.369306** |
| CC | 500 | 486 | 941 | **14** | 32000 | 47849 | 7653 | 34.057463 |

每臂`groups=匹配target+duplicate组`，target总数保持1427。WS多出136组，其中11个新增匹配target和125个新增duplicate；CL覆盖更全也伴更多duplicate。CC重复少不能单独叫改善，它同时吞并/漏掉很多目标。此处duplicate是**固定首anchor对应下对相同target的重复输出**；主误差不直接给duplicate分配新真值罚分，因此必须并报覆盖/重复，不能以一个MAE或RMSE称端到端净收益。

原误差一二阶矩（各自有效人口）亦核：A0误差和85,760,563.116m、SSE352,032,275,209.263m²；WS90,709,934.908m、334,131,459,857.525m²；CL91,983,821.483m、404,371,888,586.089m²；CC50,174,724.755m、450,661,266,877.197m²。CC误差和较小只是评价点少，绝不能以此排名。

## 3. 每候选分别对A0：共同点与gained/lost

以下三个共同人口不同；没有让CC缩减WS/CL的共同集。

| 候选 | 共同n | A0→候选 MAE m | ΔMAE m | A0→候选 RMSE m | ΔRMSE m | A0→候选 P95 / P99 m |
|---|---:|---|---:|---|---:|---|
| WS | 69837 | 1219.731→1268.202 | **+48.471** | 2240.358→2159.927 | **−80.431** | 3358.933→3057.215 / 7370.030→7161.115 |
| CL | 70377 | 1216.946→1279.470 | +62.524 | 2235.650→2374.921 | +139.271 | 3357.794→3395.205 / 7360.330→9220.773 |
| CC | 24347 | 1418.314→2060.818 | +642.503 | 2609.878→4302.317 | +1692.439 | 2825.383→6579.074 / 7337.886→20867.029 |

| 候选 | gained/lost点 | rescued/lost目标 | 新覆盖点MAE / RMSE m | 丢失点原A0 MAE / RMSE m | 共同点better/worse/tie |
|---|---|---|---|---|---|
| WS | 1234 / 718 | 33 / 22 | 1736.211 / 2596.892 | 805.271 / 1448.359 | 14874 / 16860 / 38103 |
| CL | 1569 / 178 | 46 / 5 | 1235.537 / 2175.844 | 649.067 / 1251.577 | 16006 / 16428 / 37943 |
| CC | 0 / 46208 | 0 / 891 | null / null | 1108.658 / 2007.353 | 1446 / 3844 / 19057 |

恒等式`WS交点71071=69837+1234`、`A0交点70555=69837+718`成立；其他两候选同样通过。WS覆盖净增加516点/11目标，但丢失的718点并非不存在，新增点误差也较高。**WS共同点的MAE已变差**，所以自身MAE恶化不能全部归因新增较难点；同时共同点RMSE和已审计P95/P99下降，亦不能把结果写成“WS各方面更差”。

WS共同点有益距离总量23,921,623.567m、有害27,306,695.466m，净距离和+3,385,071.899m，除69837为+48.471m。这里是逐点距离差，正负之和不是“负MAE”；±1e−9m仅分better/worse/tie，不删点或改原均值。RMSE好而MAE差表明平方加权风险与平均绝对距离发生不同取舍，不由此单独认定某种离群点根因。

## 4. 固定支撑与同/变首anchor：区分解释

固定target strata是将**所有同真实身份输入的原support并集/nearest最小值**用于评价分层，四臂一致；不能把它误当每个预测组当时真的有这些helper。真实新组支撑另外报告在target-grid上，不用于删样本。

| 候选/共同区域 | n | ΔMAE m | ΔRMSE m |
|---|---:|---:|---:|
| WS inside | 68811 | +45.835 | −85.886 |
| WS outside | 1026 | +225.263 | +112.168 |
| CL inside | 69333 | +62.424 | +142.654 |
| CL outside | 1044 | +69.174 | +21.543 |
| CC inside | 24003 | +647.147 | +1738.587 |
| CC outside | 344 | +318.484 | +345.421 |

全部方法固定inside参考可用71128、outside1068，和为72196。far>1800s与no_input的有效参考分母均0，误差保持null；它们不支持本轮已经验证长时域恢复。CL自身outside RMSE3760.627→3750.288m略好，但**共同outside** RMSE3762.394→3783.937m略差，须保留人口切换；不能用不同交点判断同一条件改善。

| 候选/首anchor关系 | 共同n | A0→候选MAE m | ΔMAE m | A0→候选RMSE m |
|---|---:|---|---:|---|
| WS same | 64483 | 1189.460→1230.570 | **+41.110** | 2121.477→2098.374 |
| WS changed | 5354 | 1584.319→1721.442 | +137.123 | 3356.255→2796.806 |
| CL same | 65056 | 1193.788→1234.511 | +40.722 | 2147.566→2253.425 |
| CL changed | 5321 | 1500.082→1829.156 | +329.074 | 3117.462→3537.710 |
| CC same | 23848 | 1405.409→1943.023 | +537.614 | 2605.379→4109.955 |
| CC changed | 499 | 2035.092→7690.422 | +5655.330 | 2816.527→9790.110 |

WS同首anchor目标1255、改变100，二者仅计两个方法都有代表的target（1355），不替代1427完整registry。**同首anchor子集仍有MAE恶化，排除了“全部变化仅由换首代表”这个强解释**；但该子集按实际分区形成，不是随机化因果中介试验。成员平均、查询支撑/权重、圆周回退以及参考/OLS特征仍需分开。

动态group支撑在完整1427×64=91328 target-grid上为：

| 臂 | anchor_support | helper_only | unsupported | unmatched_target |
|---|---:|---:|---:|---:|
| A0 | 50619 | 16738 | 20771 | 3200 |
| WS | 50658 | 19943 | 18231 | 2496 |
| CL | 51022 | 18979 | 20751 | 576 |
| CC | 17078 | 7102 | 6924 | 60224 |

这些数的分母是全部target-grid，不是72196个参考有效点。原固定no_input分层为0，并不意味着预测组都覆盖目标：CC的unmatched_target-grid仍有60224。source支撑、预测代表和目标reference可用域不能混用。

## 5. 32场与9组件：异质性不是独立重复

下表全部为各候选**分别与A0共同人口**的变化；WS净覆盖点是gained−lost。所有32场保留，包括零变化，不能只挑正面场。

| 场景 | WS ΔMAE m | WS ΔRMSE m | CL ΔMAE m | CC ΔMAE m | WS净覆盖点 |
|---|---:|---:|---:|---:|---:|
| 692 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| 700 | −137.112 | −181.171 | −228.259 | +624.387 | −31 |
| 864 | +273.056 | +404.026 | +240.671 | +909.648 | 0 |
| 1187 | −20.141 | −158.141 | −15.532 | +346.152 | 0 |
| 1293 | +117.826 | +38.785 | +117.826 | +254.996 | 0 |
| 1582 | +18.537 | +1.000 | −92.320 | +62.273 | 0 |
| 1830 | +33.726 | −88.717 | +67.737 | +1345.552 | +159 |
| 1840 | +11.320 | −142.085 | +500.474 | +1093.457 | 0 |
| 1877 | +0.000 | +0.000 | +968.800 | +196.071 | 0 |
| 2038 | +0.000 | +0.000 | +0.000 | −597.757 | 0 |
| 2153 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| 2262 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| 2352 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| 2416 | −162.687 | −295.223 | −156.813 | −62.577 | 0 |
| 2479 | +59.526 | −175.009 | +59.526 | +1404.222 | 0 |
| 2502 | +74.968 | −32.323 | +160.139 | +964.293 | +36 |
| 2576 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| 2874 | +61.223 | +60.082 | −302.835 | +176.992 | +17 |
| 3156 | +417.633 | +520.134 | +445.385 | +751.828 | +108 |
| 3274 | −952.574 | −3927.093 | −478.862 | +867.652 | 0 |
| 3619 | −147.073 | −492.950 | −120.965 | +355.880 | 0 |
| 3772 | +137.549 | +46.284 | +120.820 | +1126.491 | +30 |
| 3835 | −14.313 | −166.534 | +2.428 | +2482.082 | +197 |
| 3908 | −62.323 | −82.184 | +20.996 | −62.323 | 0 |
| 3916 | +216.938 | +405.668 | +216.938 | +711.970 | 0 |
| 3956 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| 3981 | +82.990 | +29.593 | +164.748 | +147.337 | 0 |
| 4272 | −515.963 | −431.245 | −837.770 | +0.000 | 0 |
| 4316 | −67.522 | −45.703 | −57.563 | −442.097 | 0 |
| 4393 | +82.535 | +19.680 | −70.963 | +82.535 | 0 |
| 4473 | +226.089 | +552.596 | +174.779 | +1182.306 | 0 |
| 4968 | −5.600 | −25.497 | −47.361 | −70.531 | 0 |

WS共同MAE：10场好/14差/8同；共同RMSE：14好/10差/8同。CL为MAE11好/14差/7同、RMSE12好/13差/7同；CC为MAE和RMSE均5好/20差/7同。**CC仍有5场共同人口有收益，不用总体坏结果抹掉反例**；但共同覆盖大量下降，不能据少数场选择部署。

九组件均从其场重聚合，再回总表通过。下表聚焦WS，组件ID采用原字段；不是九次独立实验。

| 组件 | 场数 | 有效ref点 | 自身ΔMAE m | 共同ΔMAE m | 共同ΔRMSE m | 净覆盖点 |
|---|---:|---:|---:|---:|---:|---:|
| component-0b9d7d23108cbf0a74ed5828 | 1 | 248 | −162.687 | −162.687 | −295.223 | 0 |
| component-23608a7a22c5322f7254ffc1 | 19 | 59832 | +59.431 | +45.921 | −48.028 | +408 |
| component-75391b641c3cd7287e382451 | 1 | 2406 | +452.919 | +417.633 | +520.134 | +108 |
| component-9ea441151d6468bd21826aea | 1 | 204 | +0.000 | +0.000 | +0.000 | 0 |
| component-a077aa3eb774a8b670334e7c | 1 | 713 | −952.574 | −952.574 | −3927.093 | 0 |
| component-c260c3bbf884e315f715f00b | 1 | 186 | +216.938 | +216.938 | +405.668 | 0 |
| component-c7082a697d934f0004db48b9 | 6 | 6463 | +10.466 | +10.466 | −71.604 | 0 |
| component-d091361fb95b75d6d2a1a9c5 | 1 | 2020 | +226.089 | +226.089 | +552.596 | 0 |
| component-fdbc675a9f9f6a7e1c67e3bb | 1 | 124 | −515.963 | −515.963 | −431.245 | 0 |

WS共同MAE为3组件好/5差/1同，RMSE5好/3差/1同；CL两指标均3好/5差/1同；CC两指标均2好/6差/1同。大组件亦保留WS的MAE差/RMSE好取舍，不能只用一个组件或一个大赢家代表总体。未做pair IID CI、显著性或独立泛化主张。

## 6. 证据索引

| E | 实际位置 | 范围 |
|---|---|---|
| E115-RUN/SUM | [run.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/run.json)、[summary.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/summary.json) | 真实终态、四臂/9组件/32场总分母与资源 |
| E115-ROWS | [scene-summary.json.gz](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/scene-summary.json.gz) | 全部32场、各自/共同/strata/gain-lost/anchor，公共重聚合原件 |
| E115-REQ/STATUS/EXIT | [request](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/launch-request.json)、[status](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/launch-status.json)、[exit_code](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/exit_code.txt) | 实际命令/环境/时间、退出0和未超时 |
| E115-AUDIT | [服务器事后审计JSON](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/posthoc-distance-audit.json)、[下载脚本副本](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/115/distance_audit.py) | 私有逐点重算十人口/六指标、common mask/gain-lost，无新预测；`code_file`原metadata名为posthoc-distance-audit.py |
| E115-CODE | [115入口](../../tools/analysis/static_partition_position_control.py)、[ordinary规则](../../association/evaluation/fusion_controls.py)、[完整target评价](../../association/evaluation/scene.py) | 实际mean重算/原态首代表、reference事后评价、pooling与mask实现 |
| E115-PLAN | [Idea1.0](../proposals/static_partition_position_control_20261029.md)、[115合同](../../HandOver/L2S-20261029-115.md)、[114记录](2026-10-29-static-partition-review.md) | 上游矩阵/固定选择、终点和结果边界 |

## 7. SQUEEZE：形成了什么问题，下一步怎样区分

**已回答：** 114的静态关系变化能沿真实ordinary恢复链改变位置/覆盖；不是只做了分组分数。WS确有RMSE/尾部分位数/覆盖收益，MAE和duplicate代价也真实存在，不能宣称净胜。CL/CC说明严格clique或传递闭包也没有普遍位置优势；反例并不否定所有全局/条件方法。

**机制边界：** 固定score身份决定→静态分区→成员query均值/首anchor回退→首anchor target对应→位置/覆盖，这条**总操作链**有DE；但还没有区分哪种质量/支撑/偏差使某helper在某query有害。分组身份正确也不保证每成员每时刻适合等权贡献，成熟多源融合本就区分这些对象。当前结果给了可发展的本域条件与新增价值入口，不主张发现全世界首次的非单调性，也不因已有先例关闭研究。

**下一项有区分度的小控制（需先独立Idea，不在本文件执行）：** 固定同A0/WS分区/完整query/reference与原态首代表，比较`anchor-only OLS`与现`mean_supported`，并按同首anchor共同点和实际query成员支撑分开。same-anchor下anchor-only应给完全相同坐标，可作负控；all-unsupported回退也应不受新helper影响。若差异主要出现在有支持helper的同anchor查询，可进一步核成员贡献/质量规则；若主要来自改变代表/覆盖，则完善输出对象/对应规则；若简单anchor控制已保留收益，应把增量归透明适配A′，不强加学习B。该控制可复用已有片段状态，不能把缓存插值/OLS点称为新held-out原生测量。

仍需可信同行、真正物理参考语义和同信息融合目标比较；新组效果只能解释为这个普通恢复器下的基准结果。原reference极端尾部/生成规则未知，不能由一个分组改善/恶化就判数据坏或真实船舶状态已改善。当前不选择新网络、不加训练，也不把112的latent风险并成115的已识别根因。

## 8. Mode C回流包（局部ID，供主线程映射）

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-STATIC-POSITION-20261029-115
package_id: SQ-STATIC-POSITION-20261029-115
input_experiment_record_ids: [EXP-STATIC-PARTITION-POSITION-20261029-115]
source_idea_id: IDEA-STATIC-PARTITION-POSITION-CONTROL-20261029
plan_version: '1.0'
claim_scope: '原114pilot32/9组件；同056缓存ordinary Mean的确定性位置/覆盖干预，不是可信同行或独立泛化'
data_quality_and_provenance_gaps: 'A0 exact32/完整registry/公共7033项重聚合通过；十人口私有逐点审计支持quantile；同reference仍非新独立GT'
primary_result_with_uncertainty: 'WS自身MAE+60.815m/RMSE−65.447m/覆盖+0.714721pp，duplicate+125；共同69837点MAE+48.471m/RMSE−80.431m；32场/9组件异质性§5，无独立CI'
process_dynamics: '单次115 exit0，程序55.885810s/launcher73.715483s，RSS2006282240B，0模型/训练/GPU/ZIP；事后审计0.573488s无预测重跑'
mechanism_and_intermediate_findings: '分组到均值/回退/首anchor对应的总链DE；same-anchor也MAE变差，不能完全归换代表；具体helper质量/时域贡献机制AI/UN'
errors_heterogeneity_and_boundary_conditions: 'CL覆盖更高但MAE/RMSE差，CC总体严重损失但仍有局部正例；固定far/no_input参考分母0；原始速度只作容器不解释'
robustness_sensitivity_and_alternative_explanations: '自有=共同+gain/lost一二阶矩通过；两臂共同不是四臂交集；权重/代表/缺失/参考/OLS条件均仍需分离'
new_P: '不增普遍新问题；细化C53关系决定与query贡献/输出风险不等价的本域实例'
new_F:
  - 'F115a_DE：WS关系增量下ordinary位置MAE/RMSE/覆盖/duplicate方向分歧，不能称净胜'
  - 'F115b_DE：WS同首anchor共同64483点MAE+41.110m；变化不全由换代表解释'
  - 'F115c_DE：CL/CC覆盖—精度取舍和全部32场/9组件正反异质性，见§2–5'
new_M: 'M115_AI：完整身份关系与每query物理贡献/首代表对应需分离；已测总链，具体因果质量路径未识别'
evidence_index: 'E115-RUN/SUM/ROWS/REQ/STATUS/EXIT/AUDIT/CODE/PLAN，§6'
open_questions: 'same-anchor成员贡献/回退、保持分区后的透明恢复对照、可信同行/参考生成、条件B的独立净收益'
evidence_status: partial
next_routes: 'deep-read Mode C；再把可区分控制形成单独Idea，无自动训练/扩量'
next_skill: deep-read Mode C
```

这次结果支持继续完善“身份关系与查询贡献分开”的明确发展路径，同时约束了效果主张：**保留WS的关系/RMSE/覆盖价值，也必须解决其MAE和重复输出代价。**

执行后范围补核：固定V32名单与原103/105三案例1043、2819、3035交集为空。原抽样不看成绩、没有替换难例；本结果不能外推三场或全V极端尾部，也不构成它们的新诊断。
