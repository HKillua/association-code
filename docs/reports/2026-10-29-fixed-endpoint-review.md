# 113：107固定端点全V评价的Record与Squeeze

2026-10-29。原113一次评价已真实完成；没有重新训练107、训练TAM、补seed/步数或访问测试数据。当前队列只看[交接](../../HandOver/Handover.md)，本页保留终态事实与解释。

## 1. EXPERIMENT_RECORD：问题、实际设计与证据

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-TMTAFF-FIXED-ENDPOINT-113-20261029
package_id: EXP-TMTAFF-FIXED-ENDPOINT-113-20261029
created_at: 2026-10-29
source_idea_id: IDEA-TMTAFF-FIXED-ENDPOINT-EVAL-20261029
plan_version: "1.0"
execution_status: completed
research_question: 原107固定新单位末步在同056全V人口和分组下的真实恢复表现
hypothesis: 单位和来源一致是比较条件，但不保证有限预算TFM能恢复物理状态
planned_vs_actual_design_and_controls: 见1.1；原固定V300、107末步、056阈值和分区、四普通控制及090C位置
actual_configuration: 见1.2及原request；0优化更新、0TAM评分、0选阈值/端点
samples_materials_or_data_provenance: 同原Level1分卷/manifest/alignment、056原cache和090C原末步cache
results_with_uncertainty: 见2–5；单seed与反复使用的开发V，场景和共享身份组件异质性，不作IID显著性结论
all_runs_or_repetitions: 113 attempt1；原107失败另存，未覆盖或重跑；事后只审已存结果
observations_and_deviations: 全300正常结束；2h建议墙时超出而按原advisory继续；RSS峰和作者复现证据缺失
evidence_locations: 见7，私有坐标/身份/权重/完整预测与原大summary仍在服务器
known_limits_and_missing_evidence: 非作者完整TMTAFF复现；reference生成未知；未隔离单位/优化/表示/递归的独立因果份额
next_routes: 本页Squeeze后回Mode C；后续研究讨论不自动执行
next_skill: squeeze-experiment
```

### 1.1 事前计划与真实执行

消费[113 Idea](../proposals/tmtaff_fixed_endpoint_evaluation_repair_20261029.md)、[L2S-113](../../HandOver/L2S-20261029-113.md)、[公平比较合同](2026-10-27-fair-comparison-contract.md)和[原A计划](../proposals/tmtaff_unit_baseline_A_20261027.md)。原107训练两段完成后，旧schema预检失败，原FAILED和错误产物保留；[修复](2026-10-29-fixed056-schema-repair.md)只兼容真实缺失字段的来源/分组绑定。113独立冻结af475e0、全新run/launcher，不是重启107。

| 项目 | 实际与计划 |
|---|---|
| 端点 | 原107 TCM2424/TFM1000，固定末步；没有新normalizer、训练、TAM评分或基于V选择checkpoint |
| 人口/分区 | 全V300，24,599节点/4,439,092候选对；9,100声明场内目标出现，9,353输出组、610重复、8,743首组匹配目标、357未匹配目标。不是9,100个跨场独立船只 |
| 绑定 | 全300：固定组、reference/valid、raw输入到056 motion、原score/阈值、all/ITSA/MSTSA逐场计数；旧090C的sources由同源metadata+056 score/node顺序传递绑定300场，membership从真实offset/flat派生300场，exact_observation仍核 |
| 普通控制 | canonical_all两臂使用全原生点；selected_cap128两臂使用模型可见的cap128选点。均同组/query/reference/单位；两种仅支撑内的predicted控制另列，不能与OLS完成域混写 |
| oracle/单源 | TFM oracle、simple_fusion_oracle和9001/9002仅真身份诊断，不提供预测关联同行成绩；090C仅位置内部参照 |
| 隐私/测试 | 只评价既定V；data加载重核既定T准备合同，T_test payload封存。私有身份/坐标/NPZ/权重不下载，公共逐场报告删身份/映射，仅保计数与指标 |
| 终点/偏差 | 一次全300完成、真实退出0。唯一已见时限偏差是超2h建议值，未改工作量/配方；无删场/自动重试证据 |

### 1.2 命令、资源与成本

原request记录真实argv/cwd/env：解释器`/home/wangguangjie/djs/mtad-field-20261016/bin/python`，cwd `association-code-run113`，入口`tools/experiments/tmtaff_fixed056_evaluate.py`，原107/tfm、056/system、090C、三原分卷、原manifest/alignment、cuda:0/CPU4；完整参数见E113-REQ，不用事后模板冒充原命令。

原payload UTC **2026-10-28 20:35:08.654139–2026-10-29 03:47:13.850156**（北京时间10月29日04:35:08–11:47:13）；launcher **25,925.460122s**、程序全准备/评价/写盘 **25,912.028390s**、evaluate内段 **24,231.640976s**。截点不同，不把差额全部归模型计算。原7200s advisory在UTC22:35:08触发，实际墙时约建议3.60倍；服务按合同继续固定范围，未硬停/加量。

物理GPU2/UUID `GPU-efbe9d07-0d29-edb6-825b-780e7cc40bca`，提交时可用17,573MiB/总24,576MiB、兼容共卡，CPU/OMP/MKL/OPENBLAS各4，RAM预算16GiB。他人进程未触碰；systemd `mtad-3d23e5ca916c.service`、Restart=no，终态inactive/dead、ExecMainStatus0，原launcher returncode0、无failure/timed_out，payload stderr空。原运行已核Linger=yes；终态服务记录不把缺失accounting补成0成本。

模型参数2,782,976，融合器5,937,156；本次评价模型加载后进程CUDA allocated峰57,884,672B（55.203MiB）、reserved峰81,788,928B（78MiB），不是整卡/训练峰。原wrapper未写RSS峰，终态systemd也未返回MemoryPeak、CPUUsageNSec未设置，因此**RSS峰/累计CPU成本未知**；轮询ps快照不能替代峰。

原训练曝光在E113-SUM有来源链：TCM2424更新、T2100、155,134个唯一轨/呈现、35,555,236 transition、一遍；TFM1000更新、64,000正pair呈现、29,002唯一pair、35,352唯一轨、802个训练场、42,051轨呈现，各属性有效监督3,125,062/3,125,069/3,124,286/3,122,916。64,000呈现不是64,000独立pair或充分收敛证明。原TCM/TFM run elapsed为15,285.783/85,645.776s；[107记录](2026-10-29-107-failure-discovery.md)的15,361.029/85,731.836s是阶段外层墙时，口径不同。113新增训练量为0，30745次fusion forward仅为固定评价。

### 1.3 终态与独立数值核验

原run、launcher/退出、summary、index及300NPZ齐备；index场序与固定V300绑定。服务器原summary为1,083,483,050B，含按组身份/指标且未下载；服务器一次脱敏导出的公共gz为3,866,516B，白名单删`all_predicted_groups`、`unmatched_target_ids`、`selected_group_by_target`并由原身份列表派生未匹配计数。原件未改；完整原件/完整坐标继续服务器留存，字段省略显式登记。导出25.791s，0模型/预测/优化。

**公共标量重聚合**：独立脚本`audit_public_metrics.py`从原终态公共gz/原件核V300顺序、全11方法all/observed/missing、8方法support/distance、四属性paper场景等权、raw三scope及四普通控制逐场→38组件→全局/去最大组件；090C公共材料原来只有逐场共同点数，故这层仅做组件→全局/余170场。20,117项算式/计数核验0不一致，最大浮点绝对累加差4.7684×10⁻⁶（大MSE，固定rel1e−10/abs1e−8允许范围），不平均分位数。这是重复字段/恒等式检查数，不是独立样本或机制证据数。

**真实数组重算**：服务器用独立stdlib+NumPy代码`audit_saved_arrays.py`读取原300个私有NPZ/metadata及090C原物理cache，核全部5个固定组方法的target registry、成员/首anchor/重复对应、ref/valid/完整预测有效人口；重算5×3个自身物理人口、4对×2侧×3属性＋090C两侧位置的共同误差，并核全局/全部38组件/去最大组件的count、MAE、RMSE、P95、P99。61,806项exact标量/结构、20,700项exact数组、48,335项数值对照均通过，数值最大差0；固定rel1e−10/abs1e−8，未放宽。另输出脱敏090C逐场共同位置矩，补齐公共源未保存的该层。

事后数组审计耗时35.951s，RSS峰320,147,456B，仅属于审计；0模型/优化/checkpoint/原ZIP/测试读取，原预测没有重算。它是**独立代码、同一已存reference和同一声明公式**的实现核验，不是新真值或作者复现。原source/端点/raw绑定不由此重新前向；oracle/支撑内/单源及paper/region指标这层未逐点重算，前层只从原逐场公共矩重聚合。审计范围与未覆盖项全部保留。

## 2. 原关联、完整人口与全部方法

下列raw完全继承056的score/阈值，不是113的TAM或融合贡献。accept score≥1.7320103112387182。

| raw范围 | 候选对 | TP / FP / FN / TN | mPCA % | mPFA % | mwPCA % | mwPFA % |
|---|---:|---|---:|---:|---:|---:|
| all | 4439092 | 22666 / 55652 / 2009 / 4358765 | 91.640226 | 22.486705 | 92.337147 | 71.059016 |
| itsa | 2243583 | 7704 / 9432 / 498 / 2225949 | 92.961246 | 15.788306 | 93.960105 | 55.042017 |
| mstsa | 2195509 | 14962 / 46220 / 1511 / 2132816 | 90.996983 | 24.566897 | 91.595833 | 75.545095 |

decoded分区另为TP15,394/FP33,450/FN9,281/TN4,380,967，不能把它与上表raw互换；本次未改分区，亦没有执行114的WS。

各方法**自身有效交点**如下；主误差不直接处罚duplicate，须并报完整组/覆盖。无参考/无预测不填0，位置须lat/lon同时有效。速度物理值m/s，航向物理误差为最短圆周角。

| 方法 | 位置交点n | 点覆盖% | 位置MAE m | RMSE m | 速度MAE m/s | 航向MAE° |
|---|---:|---:|---:|---:|---:|---:|
| predicted | 422317 | 92.826716 | 42787.223 | 66332.748 | 10.228 | 85.069 |
| oracle | 454919 | 99.992746 | 49629.372 | 68841.967 | 5.137 | 85.218 |
| simple_fusion_oracle | 448512 | 98.584466 | 1289.481 | 2808.111 | 24.349 | 76.442 |
| predicted_anchor | 322953 | 70.986170 | 247.448 | 2780.964 | 30.615 | 81.902 |
| predicted_mean | 413722 | 90.937505 | 1097.307 | 2988.019 | 25.781 | 77.083 |
| single_source:9001 | 433494 | 95.283458 | 236.711 | 2472.584 | 30.445 | 81.939 |
| single_source:9002 | 336599 | 73.985607 | 3090.032 | 3713.067 | 15.119 | 73.495 |
| canonical_all_anchor_ols | 442263 | 97.210915 | 885.579 | 6307.486 | 30.452 | 81.101 |
| canonical_all_mean_supported_ols | 442263 | 97.210915 | 1224.604 | 3470.074 | 26.005 | 75.936 |
| selected_cap128_anchor_ols | 442263 | 97.210915 | 884.140 | 6216.322 | 30.445 | 82.320 |
| selected_cap128_mean_supported_ols | 442263 | 97.210915 | 1222.324 | 3355.595 | 26.050 | 77.581 |

predicted位置参考有效454,952/声明582,400 target-query；其中可评422,317、缺预测32,635，完整有效预测点528,686中106,369没有对应有效参考。位置有参考目标9,099（另1个无有效位置参考），覆盖8,742、缺357；这与结构首组匹配8,743的分母不同。速度参考有效455,162、可评422,534；航向454,689、可评422,070。各OLS完整位置可评442,263、缺12,689、预测有效598,592、无对应有效参考156,329。

predicted_anchor/mean只在模型选点的支撑内输出，较小误差伴较低覆盖；不能以247m把它选为全场恢复赢家。四oracle/单源行有真实身份分组，不能拼成一个预测系统或国际同行。TFM oracle自身位置仍49.629km，但与predicted人口不同；仅说明本实现即使真组也未获得可用位置恢复，不能把两者相减当作错组的因果份额。

## 3. 主要结果：各对共同有效人口与完整尾部

四个普通控制和090C分别与TFM求共同有效unique目标点；这次五对位置共同人口恰为422,317，并非强制五方取交集。四OLS逐属性共同速度422,534、航向422,070；090C只位置，其他属性保持null。

| 对照 | 共同位置n | TFM MAE / RMSE m | 对照MAE / RMSE m | TFM−对照 MAE / RMSE m | 对照P95 / P99 m |
|---|---:|---|---|---|---|
| canonical_all_anchor_ols | 422317 | 42787.223 / 66332.748 | 774.891 / 6287.395 | +42012.333 / +60045.354 | 3375.057 / 10105.913 |
| canonical_all_mean_supported_ols | 422317 | 42787.223 / 66332.748 | 1129.928 / 3236.983 | +41657.295 / +63095.765 | 2786.276 / 5412.617 |
| selected_cap128_anchor_ols | 422317 | 42787.223 / 66332.748 | 773.256 / 6191.187 | +42013.967 / +60141.561 | 3373.686 / 10105.518 |
| selected_cap128_mean_supported_ols | 422317 | 42787.223 / 66332.748 | 1127.413 / 3107.230 | +41659.811 / +63225.518 | 2785.860 / 5403.977 |
| internal_090c | 422317 | 42787.223 / 66332.748 | 439.694 / 3071.364 | +42347.529 / +63261.385 | 1366.084 / 4701.594 |

TFM共同位置P95 **138,310.689m**、P99 **247,772.111m**；均来自该完整共同误差池，不平均场景分位数。四OLS比TFM覆盖更多而共同位置也明显更好，因此当前严重失效不能仅用“TFM评了更多难点/少覆盖”解释。cap128 Anchor/Mean共同MAE773.256/1127.413m已同选点预算，排除“全由强控制多拿原生点”这个强解释；不能据此认定唯一根因。

| 共同物理属性 | TFM MAE / RMSE | canonical_all Anchor | canonical_all Mean | cap128 Anchor | cap128 Mean |
|---|---|---|---|---|
| 速度m/s | 10.228479 / 39.306193 | 30.495518 / 112.139624 | 25.838455 / 101.927941 | 30.487327 / 97.693963 | 25.885190 / 94.595142 |
| 航向圆周° | 85.068611 / 99.455139 | 80.895585 / 96.115363 | 75.486674 / 91.329619 | 82.171363 / 97.229836 | 77.209355 / 92.845715 |

速度误差有实际正面结果；位置和航向没有同向改善。不能把速度收益吞掉，也不能用单属性改善认证完整恢复器可用；物理圆周航向与下节论文raw角差不是同一指标。

## 4. 四属性论文口径与定义边界

以下为各方法自身可评人口，按**目标内时间统计→场内目标等权→场景等权**；经纬10⁻⁴度、速度kn、航向raw度差。ME为truth−prediction的有符号偏差，接近0才是偏差小；residual SD是目标内ddof=1标准残差SD，**不是已核实的作者mESD**。作者式29literal诊断随完整公共原件保留，不用于方法优劣；它包含truth水平，不能与标准残差SD互换。

| 方法 | 属性 | 可评点 | ME | MAE | absolute ME | residual SD |
|---|---|---:|---:|---:|---:|---:|
| predicted | longitude | 422319 | 805.839516 | 3446.235491 | 3409.284233 | 497.215484 |
| predicted | latitude | 422318 | 689.018694 | 2358.327959 | 2338.220016 | 301.427552 |
| predicted | speed | 422534 | -8.279096 | 16.912198 | 16.671671 | 10.334308 |
| predicted | heading | 422070 | 145.326291 | 168.933164 | 160.138942 | 51.079943 |
| canonical_all_anchor_ols | longitude | 442265 | 4.673171 | 67.221305 | 53.937882 | 94.431557 |
| canonical_all_anchor_ols | latitude | 442264 | -3.508912 | 51.521718 | 40.825689 | 68.482972 |
| canonical_all_anchor_ols | speed | 442476 | -64.662667 | 64.741042 | 64.662821 | 54.296822 |
| canonical_all_anchor_ols | heading | 442000 | 1.447255 | 109.711288 | 71.474213 | 106.096963 |
| canonical_all_mean_supported_ols | longitude | 442265 | -2.298936 | 92.041453 | 81.118696 | 79.570116 |
| canonical_all_mean_supported_ols | latitude | 442264 | -6.640975 | 83.103550 | 75.239628 | 59.719822 |
| canonical_all_mean_supported_ols | speed | 442476 | -55.506479 | 55.537843 | 55.506633 | 53.922702 |
| canonical_all_mean_supported_ols | heading | 442000 | -1.435084 | 101.886028 | 59.128207 | 110.935282 |
| selected_cap128_anchor_ols | longitude | 442265 | 4.710660 | 67.058219 | 53.677190 | 92.677627 |
| selected_cap128_anchor_ols | latitude | 442264 | -3.570308 | 51.289816 | 40.777351 | 66.846315 |
| selected_cap128_anchor_ols | speed | 442476 | -64.555482 | 64.634570 | 64.555635 | 51.868167 |
| selected_cap128_anchor_ols | heading | 442000 | 1.645445 | 111.337087 | 72.700017 | 106.659476 |
| selected_cap128_mean_supported_ols | longitude | 442265 | -2.423673 | 91.857125 | 81.250659 | 78.211207 |
| selected_cap128_mean_supported_ols | latitude | 442264 | -6.671652 | 82.859305 | 74.967950 | 58.565010 |
| selected_cap128_mean_supported_ols | speed | 442476 | -55.498742 | 55.528342 | 55.498896 | 51.723905 |
| selected_cap128_mean_supported_ols | heading | 442000 | -1.323255 | 103.770446 | 60.757041 | 111.739355 |

所有表中方法每属性定义场景均300，但strict_complete_population各项均null：参考和预测不完整。经度/纬度各自valid点为454,954/454,953，成对位置仅454,952；不能把单坐标分母拼成位置共同人口。TFM论文速度ME−8.279kn、MAE16.912kn是场景/目标等权；物理点池MAE10.228m/s不能只乘单位因子就声称应等于16.912kn。

输入量测kn→模型状态m/s乘1852/3600一次，reference按已核范围m/s，论文速度分别转kn；新单位链不改变reference生成仍未知的事实。090C旧速度/航向没有纳入此表。8项关联论文指标仍继承056，不得与新TFM四属性指标拼成作者完整TMTAFF复现成绩。

## 5. SQUEEZE_REPORT：失败范围、异质性与替代解释

### 5.1 支撑域和距离：不是只在外推区失效

TFM observed位置417,445点MAE42,606.895m/RMSE66,192.725m；missing 4,872点MAE58,238.216m/RMSE77,395.214m。observed是全同身份原support并集，仍含确定性支撑插值，不等于原生精确量测。原实际预测组anchor_support322,953点MAE38,361.926m、helper_only90,769点57,427.276m；与cap128 Mean在相同这些分层的923.392/1716.088m对照，严重误差已在输入支撑内。unsupported与距离层因各方法valid不同，不能直接把它们作配对因果比较。

距离≤300s的TFM419,161点MAE42,714.891m；>1800s仅912点MAE49,716.416m。只修长距离外推不能解释/覆盖当前大部分误差；但此处没有干预completion/decoder，因此机制仍待辨。no_input交点0、误差null，不推断全部目标都有观测或无未匹配目标。

### 5.2 全300、38组件和去最大组件

去最大130场组件仅作事前敏感性分析，主表保留全部300。剩余170场105,745共同位置点：TFM MAE44,097.383m/RMSE65,025.295m；cap128 Mean1415.798m/4374.034m；090C619.810m/4361.790m。因此大差距并非只来自最大组件。分位数在原结果单独从剩余点池计算，不由组件分位数平均。

四ordinary及090C各自与TFM共同位置比较：**300/300场、38/38组件的MAE及RMSE均更差**；这是这次真实固定端点的全范围描述，不是独立泛化或整类方法无效。cap128 Mean场景ΔMAE最小+5,088.914m、最大+211,373.526m，组件最小+5,088.914m、最大+75,901.953m；不是只有少数尾部主导总MAE。

| 原固定案例 | 共同位置n | TFM MAE / RMSE m | cap128 Mean MAE / RMSE m | 090C MAE / RMSE m |
|---|---:|---|---|---|
| 1043 | 307 | 61777.473 / 82417.652 | 6880.136 / 36213.513 | 6330.329 / 37694.089 |
| 2819 | 627 | 73141.638 / 90971.583 | 3474.246 / 24777.648 | 2497.343 / 26462.300 |
| 3035 | 3532 | 79995.675 / 105882.074 | 1891.111 / 16359.579 | 1742.832 / 18068.249 |

这些案例沿原先103–105的固定名单保留，不用来筛选本次评价；全300更差的范围已独立核验，不能把全部差距归三场reference极端点。反面边界也保留：三场普通控制/090C仍有很大RMSE，不能因它们相对TFM更好就称物理真值或全部尾部已解决。

**有利属性与异质性**：四ordinary对的共同速度MAE均300/300场、38/38组件更好。速度RMSE对两Anchor和cap128 Mean也是全部更好；对canonical_all Mean为299场更好、1场更差，38组件均更好。航向不是所有场都差：对cap128 Mean共同圆周MAE有70场改善/230场恶化，组件7好/31差；RMSE80好/220差、组件12好/26差。完整四对的正/负/平及范围在`terminal-synthesis.json`，不把“位置全部差”扩写成“所有属性、所有场均差”。

### 5.3 过程、机制与竞争解释

- **DE**：全V固定组/输入/单位/参考绑定通过，原新单位端点位置仍数十公里；同cap控制、共同点、支撑内和去最大组件后不利信号仍在。速度误差改善与位置退化同时保留。
- **AI**：显式单位一致在这个有限预算/时间适配里不是位置恢复的充分修复；这不是精确单变量单位因果试验，不能说“速度单位不重要”。
- **UN**：有限TFM曝光/优化、观测embedding补全与物理监督、递归反馈/单例行为、作者真实实现和全场Q64适配各占多少责任仍未知；reference生成未知，不能以大误差直接判GT错，也不认证模型类别无效。
- 本次是已训末步固定评价，没有新训练动力学。原u0/u1000与历史093A的完整优化比较不由113再运行，若用于后续因果讨论须读取原107/093A相应证据；不以本页末步推造收敛曲线或充分训练结论。
- 不把本次固定anchor_direct的TFM结果当114WS分区的学习融合结果；115的ordinary Mean取舍与113是不同干预/人口，不能借此认证条件B、Gelato或某个新网络增量。

### 5.4 决定与可发展的下一问

这个固定107端点**不具备可信强恢复baseline的实测资格**，原A预算与113一次评价到此结束，不追加训练/seed/TAM或自动换配方。此结论限当前适配/端点，不否定TMTAFF整类方法；修复软件兼容性使比较可算，并不使模型可用。

下一讨论保留两个独立问题：一是如何从正确任务/充分验证的国际成熟方法建立强恢复起点，区分输入表示、物理监督和优化不足；二是114/115已测的身份关系、query贡献与最终输出风险如何一起比较。可以重选全部骨架，近邻已有类似处理用于学习和定位新增价值，不是结束方向的理由。没有单因素机制证据时不包装新理论，不先定方法再找支持。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-TMTAFF-FIXED-ENDPOINT-113-20261029
package_id: SQ-TMTAFF-FIXED-ENDPOINT-113-20261029
input_experiment_record_ids: [EXP-TMTAFF-FIXED-ENDPOINT-113-20261029]
claim_scope: 单次固定新单位端点的同056全V恢复诊断，不是作者复现或可信同行胜负
data_quality_and_provenance_gaps: reference生成未知；RSS峰/完整环境版本缺失；公共省略私有身份/坐标有明确服务器原件
primary_result_with_uncertainty: 422317共同位置点TFM MAE42787.223/RMSE66332.748m，四普通控制与090C均低很多；速度有正面收益
process_dynamics: 固定评价25912.028s/launcher25925.460s，2h advisory超出；0新增训练
mechanism_and_intermediate_findings: 支撑内也差、同cap和去最大组件保留大差；无法唯一归因单位/优化/表示/递归
errors_heterogeneity_and_boundary_conditions: 全300/38组件、各自及共同人口、四属性/覆盖/重复/尾部，详见正文与审计
robustness_sensitivity_and_alternative_explanations: 同cap/同mask/去最大组件；单seed重复开发V，不作独立泛化
new_P: none
new_F: F213 当前固定新单位端点在共同位置人口严重失效而速度误差改善
new_M: 不新增机制；复用M02/M05作为待检验解释，禁止从末步差距认定唯一根因
evidence_index: E113-RUN/SUM/REQ/STATUS/INDEX/EXIT/RESOURCE/EXPORT及独立审计，见7
open_questions: 可信强baseline/充分优化/物理监督与表示/作者实现/独立reference
next_routes: deep-read Mode C；后续具体方案再讨论，本次无自动新任务
next_skill: deep-read Mode C
```

## 6. Mode C回流

本报告事实层→深挖→[Mode C1.33 §38](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-133)。新增E111/F213，复用P01/P02/P03/P05与M02/M05；不新增因果机制/论文或自动选择新方法。计数由1.32的P14/F212/M84/E110/C53变为**P14/F213/M84/E111/C53**，充分深入100篇（70＋30）不变。

## 7. 证据索引、原件与明确省略

公共目录：[113终态原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/113/)。

| ID | 实际原件 | 用途/边界 |
|---|---|---|
| E113-RUN / STATUS / EXIT | run.json、launch-status.json、exit_code.txt、terminal-service.json | 原run COMPLETED、300场、退出0；服务状态不单独认证指标 |
| E113-REQ / RESOURCE | request.json、launch.json、field-resource.json、time_advisory.json、worker.stderr.log | 真argv/设备共卡/建议时限偏差；worker提示需要review是通用旧文案，原advisory合同已授权继续 |
| E113-SUM / INDEX | public-summary.json.gz、prediction-index.json | 11臂全部逐场/总体、成对、组件与单位/曝光；不持私有数组 |
| E113-EXPORT | export_summary.py、export-record.json | 白名单导出与省略字段；原summary未覆盖，导出不是新实验 |
| E113-AUDIT | audit_public_metrics.py、derived-public-audit.json、audit_saved_arrays.py、saved-array-audit.json、terminal-synthesis.json | 公共重聚合、真实数组/共同尾部与脱敏逐场综合；范围及未覆盖项见1.3 |
| E113-PLAN / CODE | 原Idea/L2S/公平合同；冻结af475e0的固定评价与相关metrics | 不以当前HEAD伪造冻结代码或当作者实现 |

服务器原run：`/home/wangguangjie/djs/association-code/runs/LEVEL1-TMTAFF-FIXED-EVAL-REPAIR-20261029-113-attempt-1`，launcher同名加`-launch`；原1.08GB summary、predictions中300个NPZ及metadata、身份/坐标/按组映射、原107权重/训练journal仍保留。服务器`terminal-review-20261029/`只加脱敏导出和事后核验，不修改原产物、活动或冻结代码；没有复制到Git的预测或权重。
