# 137：观测留出选状态尺度的执行记录及深挖

2026-10-30。**当前固定内点留出选择规则没有保持完整验证集风险：它选出rho=1e−4，同源等权下V位置MAE由原生OLS的1276.441变成1784.722m，RMSE由2168.137变成27531.051m。** 旧127以T reference选择的rho=1e−8，在本轮相同源等权控制中为1198.685/2136.307m，且P95/P99均低于原生OLS。T侧observation-RTS相对OLS有均值收益，但不优于reference-RTS的MAE，不能用T均值或观测留出loss证明尺度可迁移。

科学程序、原launcher及独立137数组审计均COMPLETED/exit0，完整T32选择、T32/V32恢复一次完成；独立核验651211项exact、175965项数值、392项null通过。审计覆盖与未审范围在下文明确，不将一致性认证扩展为物理机制证明。下文失败指这条全局内点观测选尺度配方的科学效果，没有将一次负结果推广为所有观测驱动、自监督或原生状态恢复无效。

## Record：执行事实

- package_type：EXPERIMENT_RECORD；package_version：research-pipeline-contract v1.0；experiment_id / package_id：LEVEL1-OBSERVATION-SCALE-20261030-137-attempt-1。
- source_idea_id：IDEA-OBSERVATION-SCALE-20261030；plan_version：1.0；source_matrix_id：MC-OBSERVATION-DRIVEN-20261030-v1.1；selected_C_id：OD-C2 / 全局C55。消费OD-P2/P3、F3/F4/F5/F6、M2/M3/M4/M5、E3/E4/E5/E6。
- 类型：computational exploratory pilot；2026-10-30记录，一次固定内部开发实验，无独立重复。主要问题是在相同CV/RTS核和source-equal贡献下，以T原生观测留出代替T reference选rho，能否维持/改善完整恢复风险。
- 主要估计量：完整V的observation-RTS−原生OLS等权、observation-RTS−reference-RTS等权的位置MAE差；RMSE、P95/P99、FWD、支持层、组件/去最大、覆盖与全部18臂153对作为完整边界。T不作独立测试。
- 事前依据：[Idea137](../proposals/observation_driven_exploration_20261030.md#3-idea-137原生观测留出能否替代reference选择离线状态尺度)、[行为规格1.0](../specs/observation-driven-exploration.md)、[L2S137及完整命令](../../HandOver/L2S-20261030-137.md)、[观测驱动矩阵](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md)。本轮是OD-C2实际检验，不是原文完整Q/R辨识器或Chroma/JAS作者复现。
- 冻结代码3db43c0；科学checkout `/home/wangguangjie/djs/association-code-run137`；原run `/home/wangguangjie/djs/association-code/runs/LEVEL1-OBSERVATION-SCALE-20261030-137-attempt-1`。代码历史仅用于追溯。
- 程序真实起止UTC 2026-10-29 19:21:35.940630→20:15:23.837659，即北京时间10月30日03:21:35.941→04:15:23.838。程序exit0/3227.897110s；原launcher exit0/3238.247508s，非提交耗时0.126s。
- 原unit `mtad-221298a36879.service` 已真实inactive/dead、Result=success、ExecMainStatus=0、Restart=no；证据为[run](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/run.json)、[launcher request](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/launcher/request.json)、[launcher终态](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/launcher/status.json)、[服务终态](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/observed-service-terminal.json)。

### 计划、实际、人口与成本

| 项目 | 事前计划 | 实际与限制 |
|---|---|---|
| 输入 | 127原生事件/物理状态缓存，125 T分组、114 V WS、133旧控制，原manifest | 路径和人口经生产绑定；不读ZIP/CSV/T_test，0新teacher前向；原四状态和支持mask没有换数据来源 |
| 观测选择 | T32/12组件，固定九rho，完整T结束并冻结后才读reference控制/评价 | T32全完成，2310片段均合格；580384事件中387676 retained、192708 heldout；没有不合格片段或缺失场 |
| 最终恢复 | T32/V32完整事件，两已定rho、同时输出FWD/RTS，同rho去重 | rho不同，实际两次完整恢复；T5臂/10对，V18臂/153对全保留；没有再选候选、扩网格/改mask |
| 片段/事件 | 原64场 | T2310/580384，V3851/1095780；总6161片段/1676164原生事件；重复核调用的事件工作量另计，不混作独立样本数 |
| T人口 | 固定节点、组、首代表/重复/缺失及reference mask | 860目标、911预测组、86重复，825有anchor/35漏目标；38224有效reference点、37635可评/589缺预测；位置覆盖98.4591%、轨迹覆盖95.9302% |
| V人口 | 同原V32固定人口 | 1427目标、1605预测组、217重复，1388有anchor/39漏目标；72196有效reference点、71071可评/1125缺预测；位置覆盖98.4417%、轨迹覆盖97.2670% |
| 资源 | CPU4/taskset12–15，RSS6GiB/AS12GiB，9000s advisory，0GPU | 峰值1772273664B≈1.651GiB，未超建议；科学一次，无失败/自动重试或截断 |
| 阶段墙时 | 完整选择/恢复/评价与I/O | T观测选择1709.665438s、T恢复评价459.633865s、V恢复评价1022.808910s；其余汇总/写盘等约35.789s；不是在线更新时延 |
| 真实计算工作 | 至多9×2310＋2×6161=33112次filter-smoother | 实际33112次：选择20790、恢复12322；每次同时得FWD/RTS。是解析滤波/平滑核调用，不是33112次模型训练 |
| 事件计算 | 每次核内真实工作量全计 | 累计6841412原生事件处理、6808300观测更新、8915998转移步、8915998RTS后向步、2108000内部query/414980外部query计算；后两者是核调用计数，不是独立评价人口 |
| 拟合/训练 | 观测驱动选择一个工作参数，0新神经或target-reference拟合 | teacher调用/新神经模型/optimizer/target-reference fit/解析head fit均0；选择rho仍是数据驱动参数选择，不能称无参数。旧reference-rho为历史控制，不是137新读标签调参 |
| 独立审计 | 保存九rho留出预测→MSE→选择、源等权回读、旧14臂exact、全人口/主要指标 | 一次独立只读审计COMPLETED/exit0；651211 exact/175965数值/392 null；程序55.882486s、launcher56.898101s、峰值372641792B≈0.347GiB；未重跑科学 |

T/V共享4组件，V27/32场、67780/71071点位于共享组件；原V多轮开发属性未改变。真实源仍K2，K≥3仅结构/软件性质，无外部真实传感器验证。raw关联分数、WS/decoded组、目标对应都保持，本项仅更换位置状态及rho选择方式，不是关联改进或整个深度网络端到端训练。

### 实际计算和信息权限

每fragment保首末，0-based内点i%3==1留出；仅retained位置重建局部chart、经度展开、east scale和先验，heldout时间作为query，位置仅用于预测后评分。原127 CV模型以工作rho控制积分白加速度过程项，保持原观测/先验工作尺度；它不是测得的传感器Q/R。本轮选择目标为**每场等权→场内合格片段等权→片段内heldout径向平方误差均值**，单位m²。exact平局取较小rho，实际唯一最小值，无平局选择。

| rho | T观测留出criterion，m² |
|---:|---:|
| 1e-10 | 3224698.093 |
| 1e-09 | 2834160.420 |
| 1e-08 | 2242184.663 |
| 1e-07 | 1342466.632 |
| 1e-06 | 646755.724 |
| 1e-05 | 417143.535 |
| 0.0001 | 354693.541 |
| 0.001 | 368012.290 |
| 0.01 | 430782.219 |

选中rho=1e−4，较旧reference-rho=1e−8大10000倍。相同观测criterion下，1e−4的354693.541m²比1e−8的2242184.663m²低84.181%；这只比较同一观测MSE目标，**不能和旧127 reference-MAE的m数值直接作大小比较**。32场各自最小rho分布为1e−8两场、1e−7五场、1e−5七场、1e−4七场、1e−3九场、1e−2两场，说明全局尺度不是每场最优；这只是固定结果描述，不授权改逐场选择。完整各场九候选值见[scale-selection原件](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/scale-selection.json)。

T选择于UTC19:50:05.629808（北京时间03:50:05.630）冻结，随后才读取127保存reference-rho并进行完整恢复/评价；预测先落盘再读取评价。代码和软件guard支持这个访问顺序，独立数组审计核验保存输入、索引、预测评分及声明；不能单靠数组证明进程没有额外访问。reference-rho只读历史值，137没有对reference新优化。FWD共用RTS选择的rho，不能称FWD单独调优；全窗口chart和双侧选尺度都使本实验不具有严格在线权限。

完整恢复均只改变位置，速度/航向原WS_MEAN exact；所有新臂使用同一个source-equal贡献，支持成员内均分，不依赖source编号。0源时仍保留当前CV状态Mean/首节点回退，**回退规则相同但数值不是冻结原OLS**，因此无支持域也会受rho影响。源权公共摘要相对source-equal的L1均0；新增rho效应不能归为贡献学习。旧127四个CV Mean/LOW与133十旧臂经生产exact绑定，LOW/FixedLOW只作历史控制，不是新架构。

### V完整18臂与T完整5臂

单位m，全部越低越好；V同71071点，T同37635点。原件保留全部配对、经纬论文统计及固定支持层，不用最低臂拼组合冠军。

| 方法 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| WS_MEAN | 1276.328 | 2168.266 | 3103.795 | 7194.825 |
| WS_NEAREST_SUPPORTED | 927.435 | 2173.150 | 3873.502 | 7301.504 |
| WS_LOW_SOURCE_SUPPORTED | 527.446 | 1813.304 | 2942.040 | 7260.394 |
| WS_FIXED_LOW_SOURCE_SUPPORTED | 447.060 | 1740.814 | 1012.563 | 7526.008 |
| WS_QUADRATIC_VALUE_GATE | 435.061 | 1739.320 | 1014.478 | 7526.008 |
| WS_SOURCE_EQUAL | 1276.441 | 2168.137 | 3097.435 | 7194.825 |
| WS_SHARED_Q_SOFT | 1128.755 | 2060.605 | 2990.606 | 7194.825 |
| WS_SHARED_Q_HARD | 547.721 | 1829.356 | 3125.980 | 7260.394 |
| WS_SHARED_LINEAR_SOFT | 1162.880 | 2085.495 | 3011.683 | 7194.825 |
| WS_SHARED_SHUFFLED_SOFT | 1276.493 | 2168.187 | 3101.807 | 7194.825 |
| WS_CV_FORWARD_MEAN | 1284.235 | 2504.358 | 3149.705 | 8208.787 |
| WS_CV_FORWARD_LOW | 489.263 | 2201.544 | 2971.418 | 8208.787 |
| WS_CV_RTS_MEAN | 1198.589 | 2136.436 | 2926.265 | 6224.031 |
| WS_CV_RTS_LOW | 380.127 | 1771.103 | 2700.649 | 6541.965 |
| WS_REFERENCE_SCALE_FORWARD_EQUAL | 1284.338 | 2504.241 | 3142.724 | 8208.787 |
| WS_REFERENCE_SCALE_RTS_EQUAL | 1198.685 | 2136.307 | 2913.120 | 6224.031 |
| WS_OBS_SCALE_FORWARD_EQUAL | 1828.857 | 27538.260 | 3326.260 | 10033.398 |
| WS_OBS_SCALE_RTS_EQUAL | 1784.722 | 27531.051 | 3332.561 | 9514.676 |

T侧：

| 方法 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| WS_SOURCE_EQUAL | 1311.223 | 2878.589 | 2544.797 | 10480.821 |
| WS_REFERENCE_SCALE_FORWARD_EQUAL | 1265.011 | 2867.618 | 2582.280 | 9154.836 |
| WS_REFERENCE_SCALE_RTS_EQUAL | 1244.531 | 2761.939 | 2452.560 | 9453.277 |
| WS_OBS_SCALE_FORWARD_EQUAL | 1315.791 | 2838.692 | 2743.712 | 9776.039 |
| WS_OBS_SCALE_RTS_EQUAL | 1267.062 | 2709.036 | 2657.598 | 9383.673 |

V同贡献reference-RTS相对OLS的MAE/RMSE/P95/P99分别改善77.756/31.831/184.316/970.793m；这是在不固定LOW优先时仍能保留的状态恢复信号。它仍远高于旧LOW/FixedLOW/Qhard或旧CV_RTS_LOW的MAE，不能据此宣布最终模型成熟或可信国际同行胜出。

observation-RTS的V MAE增加508.281m（+39.820%），RMSE约12.698倍，P95/P99增加235.126/2319.851m。T MAE减少44.161m、RMSE减少169.552m，但P95增加112.800m；相对reference-RTS，T MAE仍差22.532m。observation-FWD在V也为1828.857/27538.260m，不能把严重失败仅归因于RTS后向平滑。

| 两个预定主要比较，候选−基准 | T差值m；12组件描述性95%区间 | V差值m；9组件描述性95%区间 |
|---|---|---|
| observation-RTS−原生OLS等权 | −44.161；[−272.016, +35.994] | +508.281；[−2.853, +584.525] |
| observation-RTS−reference-RTS等权 | +22.532；[−254.652, +114.561] | +586.037；[+49.614, +661.129] |

2000次、seed20组件bootstrap，估计共同位置点加权MAE差。最大组件权重大、T/V共享且重复开发，区间是本批条件变异，不是独立测试置信保证；第一V区间跨0不等于“风险无变化”或等效。153/10全部配对并非153/10个独立显著性发现。

### 原件、计划偏差和缺口

完整[V summary](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/summary.json)、[T summary](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/T-summary.json)、[V32逐场](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/scene-summary.json.gz)、[T32逐场](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/T-scene-summary.json.gz)、[选择原件](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/scale-selection.json)、原run/launcher保留。原生事件、逐fragment九rho预测、完整node状态/源权、坐标/身份/reference留服务器private。报告只消费公共原件及其算术重组，没有SSH、重新算KF、改配方或读取封存测试。

科学固定终点/九rho/留出mask/新臂/预算未见计划偏离，0科学重试。9000s建议预算在发布前由原1800s小探计划修订为适应完整原生九候选成本，记录已在L2S中，不是运行后加时。相同rho去重没有触发，因为两rho不同。T记录字段称completed_training_scene_count只沿公共schema，实际是完整T恢复评价，不是神经训练。

**独立核验已完成**：[独立审计原件](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/independent-audit.json)、[审计launcher终态](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/audit-launcher/status.json)、[审计服务终态](../../../biye-ronghe/docs/evidence/20261030_observation_scale/137/observed-audit-terminal.json)均保留。审计code3db43c0、独立checkout，使用原科学绝对manifest，unit `mtad-393c494726bd.service`真实inactive/dead、Result=success、Restart=no；无失败审计或科学重试。独立NumPy从原native事件及已保存九rho留出预测重算每fragment MSE→每场→总criterion和选中rho，核源/节点等权及原节点回读、速度航向值不变、旧14臂exact、原reference/首anchor/完整人口、T5/V18与10/153全部主要配对、14/11汇总子集和各2个主bootstrap，以及20790/12322调用计数。主要MAE最大复算差约2.513e−9m；MSE最大绝对差3.052e−5m²，属当前量级下数值舍入。

**未独立核验的范围**：没有重新执行CV/RTS核，保存的node预测本身不能由本次评分复算认证为物理正确；时序访问隔离还依赖代码review/软件删除测试；不认证真实reference/噪声、速度航向误差或经纬论文指标，也不认证K≥3/跨域/在线/同行胜出。独立主要指标核没有复算附加fixed_target_strata及各层harm分布；下文支持层和由其重组的SSE份额明确属于生产公共原件及本地算术，不冒充私有条件mask独立重建。全场/组件主要SSE与主要点人口已经核验。

**当前待补**：精确坏query的原native末端速度/间隔/测地表示/同组映射与预测轨迹尚未作独立机理诊断。公共逐场/支持层足以定位风险集中，但不能反推“加速度噪声大导致外推爆炸”为已证根因，也未区分目标函数、缺口权限、相关噪声和运动模型失配各自贡献。不能在本轮自动改rho/mask、限制外推、删52个点或拼136贡献。

next_routes：squeeze-experiment→deep-read Mode C；next_skill：squeeze-experiment。

## Squeeze：收益作用域、巨额损害及可区分后继

- package_type：SQUEEZE_REPORT；package_version：research-pipeline-contract v1.0；squeeze_id / package_id：SQ-OBSERVATION-SCALE-20261030-137。
- input_experiment_record_ids：本Record137；evidence_status：partial。科学执行、限定主要数组/criterion/贡献回读已核；附加支持层未独立重建，真实物理机制/外部泛化仍UN。
- 资源边界：只读上述完整本地公共原件，作加权算术和固定层核对；不新增实验/拟合/服务器查询，不把结果后分组称预定因果消融。

### 1. 同贡献控制区分了状态价值与选择规则

reference-RTS在V30/32场、8/9组件MAE改善；T29/32场、11/12组件改善。对应observation-RTS仅V15场好17场差、3组件好6差；T18场好14差、9组件好3差。相对reference-RTS，observation-RTS在V4场好28场差、1组件好8差；T6场好26场差、3组件好9差。即使暂不解释根因，也已有理由不扩大当前观测选尺度配方。

完整V场均MAE为OLS1248.763、reference-RTS1204.349、observation-RTS2057.211m，点加权与场均都恶化；不是只改变了聚合方式。T场均依次1371.921、1278.960、1306.882m，与T点加权方向一致。reference-RTS与原CV_RTS_MEAN在V只差0.096m，但两者贡献按源/按节点不同，不能将数值接近当作同一方法。

### 2. 全组件与去最大组件，不只报告一个异常场景

以下为各组件MAE（m），完整RMSE/尾部仍在原summary：

V：

| 组件（省略共同前缀component-） | 场/位置点 | OLS等权 | ref-FWD | ref-RTS | obs-FWD | obs-RTS |
|---|---:|---:|---:|---:|---:|---:|
| 0b9d7d23108cbf0a74ed5828 | 1/248 | 787.062 | 761.670 | 735.880 | 787.888 | 768.170 |
| 23608a7a22c5322f7254ffc1 | 19/58707 | 1241.468 | 1229.764 | 1163.988 | 1877.648 | 1837.983 |
| 75391b641c3cd7287e382451 | 1/2406 | 1447.074 | 1371.056 | 1368.338 | 1438.504 | 1380.908 |
| 9ea441151d6468bd21826aea | 1/204 | 2782.578 | 2871.289 | 2831.943 | 2291.588 | 2267.614 |
| a077aa3eb774a8b670334e7c | 1/713 | 997.787 | 1311.780 | 981.488 | 1263.512 | 1227.704 |
| c260c3bbf884e315f715f00b | 1/186 | 922.333 | 822.654 | 797.852 | 1704.040 | 1680.934 |
| c7082a697d934f0004db48b9 | 6/6463 | 1498.246 | 1674.867 | 1391.551 | 1735.067 | 1645.827 |
| d091361fb95b75d6d2a1a9c5 | 1/2020 | 1424.401 | 1456.923 | 1399.243 | 1501.164 | 1478.219 |
| fdbc675a9f9f6a7e1c67e3bb | 1/124 | 1186.850 | 1242.399 | 1103.045 | 1288.739 | 1233.296 |

T：

| 组件（省略共同前缀component-） | 场/位置点 | OLS等权 | ref-FWD | ref-RTS | obs-FWD | obs-RTS |
|---|---:|---:|---:|---:|---:|---:|
| 13bf8d8c05af98b86d432a08 | 1/2812 | 2240.196 | 2015.853 | 2346.750 | 2084.666 | 2028.972 |
| 1b730762e8dcc866d831d9a3 | 1/558 | 2149.523 | 709.269 | 1918.112 | 765.267 | 821.006 |
| 23608a7a22c5322f7254ffc1 | 12/23340 | 1129.262 | 1123.612 | 1084.489 | 1173.888 | 1137.957 |
| 2631a2fab1075a4e376f5869 | 1/324 | 943.321 | 893.401 | 845.767 | 939.186 | 886.590 |
| 4eb65272866a82b14a0055d6 | 1/389 | 2503.034 | 2387.961 | 2372.241 | 2366.860 | 1982.466 |
| 5ee24adf22c874beaf8d076d | 1/318 | 1233.269 | 1120.656 | 1023.121 | 1387.053 | 1264.893 |
| 75391b641c3cd7287e382451 | 1/151 | 1622.011 | 1631.825 | 1616.099 | 1613.790 | 1617.439 |
| 805a0d28ca2328b9fac2c2f2 | 1/370 | 974.987 | 962.827 | 886.767 | 972.946 | 938.222 |
| 9ea441151d6468bd21826aea | 1/219 | 933.342 | 927.705 | 889.512 | 936.825 | 906.229 |
| c7082a697d934f0004db48b9 | 9/7714 | 1455.030 | 1347.935 | 1277.607 | 1391.464 | 1339.040 |
| d784c25f6ac4208b0a38e796 | 2/1108 | 1322.693 | 2008.197 | 1264.679 | 2077.790 | 1869.051 |
| e062a18f3a426eba715c16d2 | 1/332 | 966.390 | 950.649 | 935.917 | 967.773 | 938.625 |

最大组件23608…在V19场/58707点（82.60%），T12场/23340点。按原规则去掉整个最大组件后：

| 分组 | 原生OLS MAE/RMSE | ref-RTS MAE/RMSE | obs-RTS MAE/RMSE | obs-RTS P95/P99 |
|---|---:|---:|---:|---:|
| V余13场/8组件/12364点 | 1442.502/2860.735 | 1363.433/2483.320 | 1531.825/3495.034 | 3615.417/14479.171 |
| T余20场/11组件/14295点 | 1608.319/3609.374 | 1505.837/3513.500 | 1477.858/3124.542 | 4464.382/13606.539 |

因此V劣化不只由最大组件或4393决定；去最大仍比OLS多89.323m、比reference-RTS多168.392m。T去最大时obs优于ref的MAE又说明不能将所有场景一概判负。T的1b730…组件obs-RTS821.006m明显优于ref1918.112m，而d784…为1869.051m，明显差于ref1264.679m；这些是必须保留的异质性，不以新阈值挑选。

### 3. 固定支持层指出损害作用域，不能把不同support口径混合

`support`是每个输入片段可见原生时间的[min,max]闭区间，**不是每个query恰好存在原始观测事件**。`source_support_zero/one/multi`按被评首代表预测组中有成员时间跨度覆盖该query的源数量划分；`inside_support/outside_support`按**真实identity下各输入片段时间跨度并集**固定，可能包含没有进入该预测组的片段；它不是把该identity最早/最晚时刻之间所有间隙全部填满。二者并非同一个域：目标某个片段span覆盖该query，不代表其当前预测组有span覆盖成员。该区别来自`support.py::build_input_support`和`static_partition_position_control.py::fixed_regions`，不是观测尺度修改了人口；身份分层只用于诊断。

V：

| 固定层及点数 | OLS等权 MAE/RMSE | ref-RTS MAE/RMSE | obs-RTS MAE/RMSE |
|---|---:|---:|---:|
| source_support_zero；2770 | 4216.985/7742.476 | 3434.136/7521.079 | 18139.666/139238.271 |
| source_support_one；28454 | 591.896/1427.476 | 477.552/1421.489 | 510.979/1414.594 |
| source_support_multi；39847 | 1560.847/1661.957 | 1558.234/1662.774 | 1557.349/1654.961 |
| inside_support；70027 | 1271.209/2131.730 | 1195.674/2103.686 | 1690.587/25655.182 |
| outside_support；1044 | 1627.407/3898.870 | 1400.664/3720.354 | 8098.893/86314.101 |

T：

| 固定层及点数 | OLS等权 MAE/RMSE | ref-RTS MAE/RMSE | obs-RTS MAE/RMSE |
|---|---:|---:|---:|
| source_support_zero；1487 | 6197.185/10520.787 | 5872.190/11587.672 | 5911.102/9374.727 |
| source_support_one；13521 | 512.073/1417.313 | 391.884/1241.101 | 427.214/1387.189 |
| source_support_multi；22627 | 1467.669/2303.887 | 1449.918/1715.616 | 1463.726/2298.056 |
| inside_support；37048 | 1317.825/2890.656 | 1249.717/2769.261 | 1249.668/2663.168 |
| outside_support；587 | 894.604/1972.893 | 917.178/2252.144 | 2364.906/4784.455 |

V零源2770点仅占3.8975%，却承载observation-RTS总SSE的99.6917%；1源/多源分别仅0.1057%/0.2026%。零源相对OLS的误差和增加38565825.783m，压过1源减少2302409.274m与多源减少139380.245m；因此整体MAE增加36124036.264/71071=508.281m。这里是signed误差和分解，不等于所有零源都受害。

可用源区域仍有明确局部信号：V1源observation-RTS比OLS低80.917m、RMSE也低；多源低3.498m、RMSE低6.996m，但多源P99由2956.008增至2967.615m。reference-RTS在V1源更好477.552m；多源的ref1558.234m略高于obs1557.349m。T三个source支持层MAE均改善，但T零源分布不能保证V零源安全。没有任何一层结果授权在本次发布按层拼接方案。

按固定真实identity下片段时间跨度并集，V85.5614% SSE在inside_support、14.4386%在outside_support；因此不能把“99.69%来自预测组零源”转述成“99.69%来自target外推点”。本次far>1800s与no_input可评点均0，空层不提供长时距安全证据。真实外推权限与组缺口关系还需私有坏例区分。

### 4. 4393承担主要平方误差，但不是唯一负证据

从原逐场MSE×可评点数复算，V observation-RTS总SSE分配如下；SSE单位m²，份额不是MAE贡献或独立样本比例：

| 场景 | 点数 | obs-RTS MAE/RMSE，m | 占全V obs-RTS SSE |
|---|---:|---:|---:|
| 4393 | 1292 | 26873.311/203358.011 | 99.18525% |
| 3772 | 18799 | 1430.712/2574.896 | 0.23138% |
| 1840 | 2247 | 2206.162/6430.236 | 0.17247% |
| 3835 | 13856 | 1222.027/2497.321 | 0.16042% |
| 1830 | 8436 | 1287.721/2227.372 | 0.07769% |

4393的52个零源点MAE635761.804m、MSE约1.027440383×10¹²m²，单独占全V总SSE99.17952%；该场P95为2485.189m而P99达1227027.419m，说明常见分位也可能隐藏最极端风险。该场1292点相对OLS的signed误差增加占全部净MAE损害88.8744%。原OLS与reference-RTS在4393分别为MAE2024.213/2218.069m、RMSE5076.595/7386.263m，旧reference控制也并非在这场四指标全好，但新尺度明显放大了风险。

4393当前1源/多源分别380/860点，obs-RTS MAE331.653/1784.507m；巨额SSE集中与0源当前CV状态回退共存。**这是DE作用域定位，不是已证明的末端速度发散或rho因果链。** 公共记录没有足够的逐query原生间隔、端点速度、平面转换和成员归属来排除其它解释；报告不虚构这些细节，不自动追加诊断或补救。

T的最大单场SSE来自10号场，占23.7984%，其903点obs-RTS MAE1422.304/RMSE8531.809；其中487多源点MSE约1.27175×10⁸m²。这说明不能把所有T/V风险一律归结0源，亦不能用仅T总体均值选择一个可靠尾部策略。

V obs-RTS相对OLS有44156点改善、26915点恶化，收益和5185675.251m、损害和41309711.516m；T有23464改善、14171恶化，收益和5907601.949m、损害和4245599.632m。多数点改善与平均风险恶化可以同时成立，高代价负例必须纳入目标；不能用胜点数替代米制损害。这里未额外扫描top阈值或将坏点排除。

### 5. 观察、解释与可反驳后继严格分开

| 层级 | 当前事实或假设 | 可区分问题；本轮不执行 |
|---|---|---|
| DE：选择目标和完整任务风险不一致 | 相同T观测MSE中rho1e−4较1e−8低84.18%，固定V完整MAE/RMSE却严重更差；T尾部也不全好 | 后续需区分原观测噪声目标、MSE vs MAE、内点mask和全局聚合的共同变化，不能直接断言“平方损失就是根因” |
| DE：无源支持恢复是主要V损害域 | 0源仅3.90%却99.69% SSE，4393的52点尤其集中；去最大组件仍差 | 最小后继应先核已保存坏query的原生支持/端点状态和传播权限，判别模型外推与分组缺口；不能先把0源改回OLS再说机制已证 |
| DE：同贡献下仍有状态收益 | reference-RTS整体四指标好，obs-RTS的V1源/多源均值也有收益 | 研究能匹配完整恢复任务的观测验证目标，而非否定整个CV/RTS、观测驱动或小模型方向 |
| AI/UN：内点目标未约束无支持恢复 | 固定保留首末的选择主要考查双侧内插，最终也恢复无支持query，二者权限不同 | 与一般目标噪声/运动模型失配竞争；需分离一个权限或验证目标后再验证，不能把差异直接叫因果消融 |
| AI/UN：观测误差并不等于潜在状态风险 | 相关噪声、bias、机动可改变观测MSE排序；真实reference生成方式未知 | 不声称识别真实Q/R，也不把rho更大等同更真实噪声；相似成熟方法仍可适配，须补合法证据 |

### 6. 回流与去留

当前**不扩大本次全局内点观测rho配方、不据V改mask/rho或自动切回某源优先**；保留reference-RTS同贡献控制的有效信号，以及1源支持区观测恢复的有限收益。137不与136/138独立结果相加成联合冠军。它既没有证明国际同行胜出，也没有证明观测驱动路线整体失败。

当前限定独立审计已完成，下一步先回流Mode C，才决定是否另立“状态选择目标与实际恢复权限一致”的新Idea。最便宜的可区分入口是对已保存的高损害零源query核输入端点/时距/状态传播与组成员，先判断是否需要域匹配验证目标、不同恢复模型或贡献之外的输出规则；本批固定终点已到；后继须另立具体合同，不依据旧命令续跑。此报告不指定某个修复作为答案，不因已有类似研究否定新增价值。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-OBSERVATION-SCALE-20261030-137
input_experiment_record_ids: [LEVEL1-OBSERVATION-SCALE-20261030-137-attempt-1]
claim_scope: 固定T32/V32、真实K2、相同sourceequal与完整人口下的一次离线观测尺度选择
primary_result_with_uncertainty: V obsRTS减OLS为+508.281m，9组件区间[-2.853,584.525]；减refRTS为+586.037m，[49.614,661.129]
process_dynamics: 9个T观测MSE候选后冻结rho1e-4；33112次filter-smoother核调用；0新神经/target-reference拟合
mechanism_and_intermediate_findings: 目标和最终风险排序不同为DE；内点目标与无源恢复权限失配为AI/UN，不是已证因果
errors_heterogeneity_and_boundary_conditions: V零源3.90%点承担99.69%SSE；4393贡献99.185%；去最大仍差；1源局部改善
robustness_sensitivity_and_alternative_explanations: 全18臂/153对、T5臂/10对、全组件/去最大/尾部/完整覆盖；无新seed或独立测试
new_P: 是否应将观测验证权限与完整状态恢复域匹配，而非只优化内点预测
new_F: 同贡献参考RTS保留四指标收益；本观测选尺度配方在零源域严重失效
new_M: [内点验证与无支持恢复权限不一致_AI_UN, 噪声或模型失配竞争解释_UN]
data_quality_and_provenance_gaps: 限定审计通过；附加支持层未独立重建；真实reference/误差相关及坏query物理原因未知
evidence_index: 本文原run/selection/summary/Tsummary/逐场/launcher/服务终态链接
open_questions: 4393端点传播与组缺口的独立归因；目标函数与验证域的可分离贡献；外部K大于2效果
next_routes: deep-read Mode C→仅在新合同成熟后discover-innovation
next_skill: deep-read Mode C
```

后续回流记录：本文Record/Squeeze已被[Mode C1.57](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-157)分项消费；没有由该回流追加新实验或方法成绩。
