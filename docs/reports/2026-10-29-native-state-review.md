# 127：原生CV/RTS状态控制的Record与Squeeze

2026-10-29；单次正式Level1小范围验证。程序与launcher均COMPLETED/exit0，固定T32×9候选与原V32六臂已到终点；独立保存数组主位置/人口审计通过。本记录不扩展实验或发布READY。

**主要发现：同原WS组、同LOW贡献和同71,071个可评点，RTS LOW的MAE由527.446降至380.127m（−27.93%），RMSE由1813.304降至1771.103m（−2.33%），P95/P99亦降低。RTS Mean也有增量。但RTS LOW仅5/9组件RMSE改善，4393明显恶化；FWD LOW虽降低总体MAE，却使总体RMSE/P95/P99恶化。** 这是当前离线工作模型和重复开发V32上的条件结果，尚未证明物理噪声机制、全V效果或可信同行优势。

## 1. Record：事前问题、实际输入与终态

上游为[执行前Idea1.0](../proposals/native_cv_state_control_20261029.md)与[原生时序设计§3](../proposals/input_time_signal_pilot_20261029.md)，消费IT-C5。问题是：保持贡献规则与分组不变，改用真实原生事件的潜状态估计，是否改善完整位置恢复风险。原Q64插值/端点OLS不是原始观测；可能适合平滑修正，也可能受非恒速机动、相关航迹误差、共同偏差、错误分组或reference关系影响。后一组仍为竞争解释，未被本实验识别。

| 项目 | 实际值 |
|---|---|
| Idea / plan | IDEA-NATIVE-CV-STATE-CONTROL-20261029 / 1.0 |
| 执行 | LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1；唯一attempt |
| 服务与checkout | mtad-b7342f356c9b.service；/home/wangguangjie/djs/association-code-run127 |
| 终态 | COMPLETED；run.exit_code=0，launcher.returncode=0；Result=success，inactive/dead，Restart=no |
| 程序UTC | 2026-10-29 14:39:33.745900 → 15:23:05.898283；北京时间22:39:33→23:23:05 |
| 数据 | 正式MTAD Level1三分卷；原manifest、原alignment；T=manifest前32个T_core；V=118/121原32场、9组件 |
| 固定来源 | 056/system原物理底座；114 WS；121原Mean/LOW和LOW query weights |
| 新增计算 | T上九rho的解析filter+smoother；选一个rho后V上共同FWD/RTS状态；0新神经模型/前向/优化更新 |
| 不变项 | 原node/source/Q64秒、速度/航向旧单位、分组/首anchor/完整target registry/valid、贡献support和LOW source次序 |
| 实际读取 | 64场native及选定T/V reference导出；非整个进程V-reference-blind；T_test未读 |
| 实际终点 | 32个T场全部9rho、32个V场六臂/15对，64控制场景臂重放绑定、128新场景臂；没有自动扩量或重试 |


正式原run位于 `/home/wangguangjie/djs/association-code/runs/LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1`。原始坐标、身份、事件谱系、完整预测及T逐fragment数组留服务器。必要脱敏原件保留在资料库127目录，不在代码仓复制实时队列。命令、环境与建议墙时完整见[launcher/request.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/launcher/request.json)；真实终态见[run.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/run.json)、[launcher/status.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/launcher/status.json)和[posthoc-service-terminal.txt](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/posthoc-service-terminal.txt)。

实际payload命令（工作目录为上述run127 checkout；launcher的time-policy=advisory/max_seconds=7200，未设置硬墙时停止）：

```bash
/usr/bin/taskset --cpu-list 4-7 /home/wangguangjie/djs/mtad-field-20261016/bin/python tools/analysis/native_cv_state_control.py --archive '/home/wangguangjie/djs/data/mtad/数据集level1.zip.001.zip' '/home/wangguangjie/djs/data/mtad/数据集level1.zip.002.zip' '/home/wangguangjie/djs/data/mtad/数据集level1.zip.003.zip' --manifest configs/data/mtad-level1-scene3000.json --alignment configs/data/mtad-level1-alignment.json --source-system /home/wangguangjie/djs/association-code/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1/system --partition-run /home/wangguangjie/djs/association-code/runs/LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1 --source-preference-run /home/wangguangjie/djs/association-code/runs/LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1 --output /home/wangguangjie/djs/association-code/runs/LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1 --cpu-threads 4
```

### 原始链与权限

输入经 `MTADArchive.load_scene → _record(max_points128) → prepare_timed_scene(DATA) → scene_motion_evidence(canonical_all)`；原生位置/秒另由 `source_events` 保留uncapped canonical谱系。未把056的`native_states` Q64预测当原生事件；未启用066速度单位改版。V raw重建node/source/query/states/support/nearest/exact逐项与056绑定，LOW权重与121绑定；这些是正式执行中的生产assert，不是通用主位置审计对原始档案的独立认证。

| cache角色 | 场景 | 节点/fragment | 原生事件 | 目标registry | 有效reference位置点 | raw reference issue计数 |
|---|---|---|---|---|---|---|
| T_core | 32 | 2310 | 580384 | 860 | 38224 | 441 |
| V_select | 32 | 3851 | 1095780 | 1427 | 72196 | 529 |


上述issue计数沿原reference读取器的事件定义，未逐类或逐目标核因；不能把441/529解释成错误目标数、坏数据比例或已识别的reference故障。`native-cache.json`仅在64场导出完写CACHE_COMPLETED，此后冻结；物理event/motion与identity/reference NPZ分离，125/129后续只读复用。127程序成本包含这次共享cache的首次导出，复用不代表底座准备成本为零。

## 2. 独立核验与证据边界

主位置审计原件：[independent-saved-array-audit.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/independent-saved-array-audit.json)。COMPLETED/exit0；32场、9组件、6臂、每场15对；全量+9组件+去最大共11汇总。exact=86183，numeric=18846，null=84；覆盖/轨迹覆盖最大差0，MAE最大差2.51e-09m、MSE最大差7.05e-06m²。审计自身4.778s、RSS93,122,560B。

审计独立用NumPy单位球面叉/点积几何核完整组offset、全node分区、首anchor/duplicate/未匹配行、跨臂truth/valid/membership、all/observed/missing主位置、每对实际共同人口与尾部及各汇总。它的scope字符串仍保留旧120/121/123/124文字，但IDS[4]/M127实际分支覆盖本次六臂，原件未改写；`imported_original32_scene_count=0`指没有pilot_path继承，不表示原两控制没有重放。

**未由此审计独立核验：** 原始native/时基/reference物理正确性、T九候选选参、支持/端点/exact分层mask和相应分层指标、逐点收益/损害计数、论文坐标统计、速度/航向、LOW路由，以及原两控制对121的原始artifact exact。正式程序逐字段绑定两控制且所有场均标记通过；另只读比较127/121的两控制公共核心字段及共有固定层，250个叶值相同。不得将这些生产绑定或JSON一致性写成generic auditor额外独立验证。后文分层、逐点及坐标数值均明确消费公开原件。

原件索引：[summary.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/summary.json), [scene-summary.json.gz](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/scene-summary.json.gz), [scale-selection.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/scale-selection.json), [native-cache.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/127/native-cache.json)。完整六臂逐场、全部15对、每个作用域和组件均保存在这些原件，以下同时展示主要与不利结果。

## 3. T选择：九候选全部保留

工作模型为`x=[pN,pE,vN,vE]`，真实Δt的CV转移，积分加速度Q(rho)，H取位置；R=I、P0=diag(1,1,1/3600,1/3600)仅工作尺度。首观测定义prior、零初速，不重复更新；仅观测闭区间内query并入KF/RTS时间线，区间外另由相应端点状态外推。首前FWD使用首点零速prior。未jitter、改配方或退回OLS；只输出前向/RTS均值，不提供可用作风险权重的RTS后验协方差。

每个rho按RTS的“场景等权→该场有参考fragment等权→该fragment有效query径向MAE”选取。32场、2310 fragments全部有可用位置reference；缺参考场/fragment均0。选定rho=1e-8；FWD共同使用该rho，没有独立为FWD调优。T目标不是V的target-pooled MAE，两者不能直接比较作过拟合判断。

| rho | T等场/等fragment目标MAE（m） |
|---|---|
| 1e-10 | 2233.698376 |
| 1e-09 | 2126.678590 |
| 1e-08 | 2104.224521 ← 选定 |
| 1e-07 | 2282.118741 |
| 1e-06 | 2579.944699 |
| 1e-05 | 3224.577849 |
| 1e-04 | 4249.413876 |
| 1e-03 | 5842.643101 |
| 1e-02 | 8188.229484 |


次低1e-9目标为2126.678590m，与选定目标2104.224521m相差22.454069m；未给出选择不确定性或外层交叉验证。事后只从已保存T目标计算每场局部最小rho的分布为1e-10:6场、1e-9:11场、1e-8:9场、1e-7:4场、1e-6:1场、1e-4:1场，提示工作尺度需求异质；没有据此重新选局部rho，也不能据此认证噪声协方差。

T逐场在固定1e-8下的目标与参考fragment人口（完整每场九候选见scale-selection原件）：

| T场景 | fragment | 参考可用/缺失 | 固定rho目标（m） |
|---|---|---|---|
| 0 | 157 | 157/0 | 3225.939 |
| 3 | 626 | 626/0 | 1535.578 |
| 7 | 21 | 21/0 | 2890.574 |
| 9 | 99 | 99/0 | 2277.399 |
| 10 | 41 | 41/0 | 1772.846 |
| 11 | 27 | 27/0 | 1252.313 |
| 14 | 18 | 18/0 | 1235.896 |
| 17 | 165 | 165/0 | 2353.047 |
| 19 | 14 | 14/0 | 1492.807 |
| 21 | 29 | 29/0 | 1617.689 |
| 22 | 41 | 41/0 | 4631.559 |
| 23 | 16 | 16/0 | 1131.838 |
| 24 | 52 | 52/0 | 2736.938 |
| 26 | 40 | 40/0 | 2240.077 |
| 29 | 37 | 37/0 | 4639.007 |
| 31 | 10 | 10/0 | 1877.433 |
| 34 | 20 | 20/0 | 2366.425 |
| 35 | 222 | 222/0 | 1323.633 |
| 40 | 36 | 36/0 | 1030.013 |
| 45 | 31 | 31/0 | 1975.145 |
| 47 | 90 | 90/0 | 1293.071 |
| 48 | 43 | 43/0 | 2101.433 |
| 50 | 48 | 48/0 | 2545.882 |
| 56 | 11 | 11/0 | 988.642 |
| 58 | 18 | 18/0 | 1474.947 |
| 59 | 122 | 122/0 | 1657.379 |
| 60 | 17 | 17/0 | 1981.382 |
| 61 | 78 | 78/0 | 1669.993 |
| 62 | 19 | 19/0 | 2694.619 |
| 66 | 121 | 121/0 | 1669.370 |
| 68 | 18 | 18/0 | 2751.604 |
| 72 | 23 | 23/0 | 2900.708 |


## 4. V全人口与六臂结果

下表径向误差单位m，所有六臂使用相同71,071个reference∩prediction位置点。总registry1427目标×64=91,328声明query，72,196有效reference位置点、19,132缺reference点；未覆盖39个目标/1125个reference有效点均保留，不能把缺失行算作零误差。

| 共同人口字段 | 六臂相同的实际值 |
|---|---|
| 目标/组/重复组 | 1427 / 1605 / 217 |
| anchor匹配/未覆盖目标 | 1388 / 39 |
| reference不可用目标 | 0（但存在逐query reference缺失） |
| 有效reference / intersection / 未预测 | 72196 / 71071 / 1125 |
| 预测可用 / 无目标预测点 | 102720 / 31649（含duplicate及无reference对应点） |
| 点coverage / 轨迹coverage | 98.4417419% / 97.2669937% |
| 15对coverage转移 | gained/lost/rescued/lost-target全部0；首anchor未改变 |


| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---|---|---|---|---|
| Mean | 1276.328 | 2168.266 | 3103.795 | 7194.825 | 1249.718 |
| LOW | 527.446 | 1813.304 | 2942.040 | 7260.394 | 478.072 |
| FWD Mean | 1284.235 | 2504.358 | 3149.705 | 8208.787 | 1271.779 |
| FWD LOW | 489.263 | 2201.544 | 2971.418 | 8208.787 | 457.853 |
| RTS Mean | 1198.589 | 2136.436 | 2926.265 | 6224.031 | 1205.278 |
| RTS LOW | 380.127 | 1771.103 | 2700.649 | 6541.965 | 359.854 |


方法对应：Mean=`WS_MEAN`；LOW=`WS_LOW_SOURCE_SUPPORTED`；其余分别为`WS_CV_FORWARD_MEAN/LOW`与`WS_CV_RTS_MEAN/LOW`。LOW只是冻结经验source次序和支持条件贡献，不等于已识别的低噪声源。四新臂只改位置，速度/航向沿原Mean；关联分数、114 WS组与θ未变，不构成本次关联指标增益。

两项事前主比较：

| 比较 | ΔMAE | ΔRMSE | ΔP95 | ΔP99 | 改善/恶化/持平点 | 净绝对误差变化和（m） |
|---|---|---|---|---|---|---|
| RTS Mean − Mean | -77.739 | -31.830 | -177.530 | -970.793 | 47174/23897/0 | -5525023.460 |
| RTS LOW − LOW | -147.320 | -42.201 | -241.392 | -718.429 | 62899/8172/0 | -10470145.778 |


RTS Mean的MAE/RMSE/P95/P99分别降低6.09%/1.47%/5.72%/13.49%；RTS LOW分别降低27.93%/2.33%/8.20%/9.90%。LOW组合收益与Mean组合收益不同，说明当前贡献规则与状态底座存在交互；本次没有共享中心校正、125收益head或129新分组，不能把它们的单项成绩拼为联合模型成绩。

六臂也没有同一方法占据全部最优指标：RTS LOW的MAE/RMSE/P95最低，但RTS Mean的P99为6224.031m，低于RTS LOW的6541.965m。当前只隔离了同贡献规则下的状态底座增量，尚未验证相对既有FixedLOW等强控制的联合新增价值。

### 全部15对共同点结果

每对n=71,071，候选减基线；负值为该指标改善，点计数以原1e-9m容差分改善/恶化/持平。逐点计数来自生产公开统计，未被generic audit独立覆盖。

| 候选 − 基线 | ΔMAE | ΔRMSE | ΔP95 | ΔP99 | 改善/恶化/持平点 |
|---|---|---|---|---|---|
| LOW − Mean | -748.882 | -354.962 | -161.755 | 65.569 | 39392/702/30977 |
| FWD Mean − Mean | 7.907 | 336.092 | 45.910 | 1013.963 | 43243/27786/42 |
| FWD LOW − Mean | -787.065 | 33.278 | -132.377 | 1013.963 | 62283/8747/41 |
| RTS Mean − Mean | -77.739 | -31.830 | -177.530 | -970.793 | 47174/23897/0 |
| RTS LOW − Mean | -896.202 | -397.163 | -403.146 | -652.860 | 66132/4939/0 |
| FWD Mean − LOW | 756.789 | 691.054 | 207.665 | 948.393 | 23551/47491/29 |
| FWD LOW − LOW | -38.183 | 388.240 | 29.377 | 948.393 | 54626/16415/30 |
| RTS Mean − LOW | 671.143 | 323.132 | -15.776 | -1036.362 | 27278/43793/0 |
| RTS LOW − LOW | -147.320 | -42.201 | -241.392 | -718.429 | 62899/8172/0 |
| FWD LOW − FWD Mean | -794.973 | -302.814 | -178.287 | 0.000 | 39449/620/31002 |
| RTS Mean − FWD Mean | -85.647 | -367.922 | -223.440 | -1984.756 | 43605/26319/1147 |
| RTS LOW − FWD Mean | -904.109 | -733.255 | -449.056 | -1666.822 | 62690/7235/1146 |
| RTS Mean − FWD LOW | 709.326 | -65.108 | -45.153 | -1984.756 | 23766/46159/1146 |
| RTS LOW − FWD LOW | -109.136 | -430.441 | -270.769 | -1666.822 | 55074/14850/1147 |
| RTS LOW − RTS Mean | -818.462 | -365.333 | -225.616 | 317.934 | 39571/499/31001 |


## 5. 场景、组件与平方误差风险

以每个新臂对相同贡献旧臂为基线；以下方向计数按1e-9m容差，由全部已完成场景/组件公开结果计算，未挑选有利部分。

| 新臂 | 场MAE好/差/平 | 场RMSE | 场P95 | 场P99 | 组件MAE | 组件RMSE | 组件P95 | 组件P99 |
|---|---|---|---|---|---|---|---|---|
| FWD Mean | 20/12/0 | 12/20/0 | 14/18/0 | 19/13/0 | 4/5/0 | 3/6/0 | 2/7/0 | 3/6/0 |
| FWD LOW | 25/7/0 | 16/16/0 | 11/21/0 | 19/13/0 | 5/4/0 | 3/6/0 | 2/7/0 | 3/6/0 |
| RTS Mean | 30/2/0 | 26/6/0 | 20/12/0 | 27/5/0 | 8/1/0 | 5/4/0 | 5/4/0 | 7/2/0 |
| RTS LOW | 30/2/0 | 27/5/0 | 21/11/0 | 27/5/0 | 8/1/0 | 5/4/0 | 6/3/0 | 7/2/0 |


两RTS主比较均30/32场、8/9组件MAE改善；但RTS LOW的RMSE只有27/32场、5/9组件改善。唯一组件MAE不利的是2038；4393属于最大组件，明显恶化被同组件其余场景的收益抵消。

| 组件 | 场数/共同点 | Mean MAE→RTS Mean | LOW MAE→RTS LOW | RTS Mean ΔRMSE | RTS LOW ΔRMSE |
|---|---|---|---|---|---|
| component-0b9d7d23108cbf0a74ed5828 | 1/248 | 821.917→769.884 | 287.685→164.435 | -31.827 | -57.837 |
| component-23608a7a22c5322f7254ffc1 | 19/58707 | 1241.122→1163.666 | 494.452→343.511 | 63.966 | 75.778 |
| component-75391b641c3cd7287e382451 | 1/2406 | 1446.795→1368.044 | 717.377→590.533 | -48.034 | -64.804 |
| component-9ea441151d6468bd21826aea | 1/204 | 2782.578→2831.943 | 1560.540→1587.082 | 173.899 | 193.409 |
| component-a077aa3eb774a8b670334e7c | 1/713 | 997.787→981.488 | 433.472→390.215 | 40.346 | 42.820 |
| component-c260c3bbf884e315f715f00b | 1/186 | 922.333→797.852 | 460.206→261.674 | -294.101 | -368.586 |
| component-c7082a697d934f0004db48b9 | 6/6463 | 1498.914→1392.228 | 712.518→555.846 | -644.556 | -701.639 |
| component-d091361fb95b75d6d2a1a9c5 | 1/2020 | 1424.401→1399.243 | 636.087→545.029 | 102.314 | 99.674 |
| component-fdbc675a9f9f6a7e1c67e3bb | 1/124 | 1186.850→1103.045 | 468.324→353.753 | -7.949 | -24.870 |


**关键不利事实：** 最大19场组件RTS LOW MAE494.452→343.511m，但RMSE增加75.778m，平方误差和增加约1.4536×10¹⁰m²；6场`component-c7082a697d934f0004db48b9`的平方误差和下降约2.5764×10¹⁰m²，才抵消前者等损害。不能将全量RMSE略好写成各组件尾部均被改善。4393的RTS LOW MAE增加113.132m、RMSE增加2380.885m；2038分别增加26.542m、193.409m；4473与3274的LOW MAE改善但RMSE分别增加99.674m、42.820m。未取私有数组定位这些极值的真实片段/机动/参考原因。

去场景数最大19场组件后，保留13场/8组件、12,364共同点；原registry、覆盖与duplicate在各臂仍相同：

| 方法 | MAE | RMSE | P95 | P99 |
|---|---|---|---|---|
| Mean | 1443.496 | 2861.164 | 3527.567 | 10324.144 |
| LOW | 684.111 | 2609.221 | 3509.098 | 10324.144 |
| FWD Mean | 1544.438 | 3559.011 | 3521.818 | 13357.517 |
| FWD LOW | 769.949 | 3358.107 | 3507.011 | 13195.983 |
| RTS Mean | 1364.411 | 2483.806 | 3407.144 | 9890.816 |
| RTS LOW | 553.989 | 2182.394 | 3407.144 | 9972.466 |


因此RTS增量不只来自最大组件；FWD LOW在去最大后MAE反而684.111→769.949m。当前FWD不能被当作可在线替代RTS的等效方案。

逐场两主比较全表（候选减旧同贡献基线，单位m；完整六臂及15对逐场值见scene-summary原件）：

| V场景 | 共同点 | RTS Mean ΔMAE | ΔRMSE | RTS LOW ΔMAE | ΔRMSE | RTS LOW ΔP99 |
|---|---|---|---|---|---|---|
| 692 | 129 | -15.512 | -11.061 | -86.920 | -24.464 | -100.013 |
| 700 | 3331 | -58.839 | -9.693 | -129.085 | -18.394 | -163.197 |
| 864 | 843 | -53.074 | -27.698 | -78.183 | -43.156 | -31.506 |
| 1187 | 2652 | -22.393 | -9.731 | -132.748 | -37.907 | -97.484 |
| 1293 | 169 | -38.176 | -19.239 | -143.518 | -40.880 | -74.032 |
| 1582 | 1054 | -42.880 | -18.755 | -130.583 | -61.890 | -134.986 |
| 1830 | 8436 | -90.227 | -105.650 | -154.458 | -140.537 | -820.516 |
| 1840 | 2247 | -185.999 | -901.481 | -231.193 | -983.188 | -3773.824 |
| 1877 | 134 | -27.494 | -3.843 | -102.969 | -1.677 | -25.752 |
| 2038 | 204 | 49.365 | 173.899 | 26.542 | 193.409 | 918.817 |
| 2153 | 310 | -61.039 | -14.960 | -147.460 | -63.847 | -111.192 |
| 2262 | 365 | -51.439 | -10.304 | -131.708 | -49.545 | -7.237 |
| 2352 | 289 | -8.243 | -2.077 | -137.341 | -39.404 | -28.511 |
| 2416 | 248 | -52.033 | -31.827 | -123.250 | -57.837 | -180.531 |
| 2479 | 851 | -3.375 | 18.176 | -113.402 | 17.461 | -2.303 |
| 2502 | 3133 | -53.357 | -7.935 | -133.030 | -26.546 | -71.569 |
| 2576 | 589 | -42.565 | -7.392 | -129.266 | -25.284 | -36.130 |
| 2874 | 1073 | -59.511 | 0.224 | -125.729 | -12.125 | 37.068 |
| 3156 | 2406 | -78.751 | -48.034 | -126.843 | -64.804 | -93.080 |
| 3274 | 713 | -16.300 | 40.346 | -43.257 | 42.820 | 133.396 |
| 3619 | 754 | -22.369 | -22.353 | -68.344 | -65.380 | -2.469 |
| 3772 | 18799 | -92.923 | -125.805 | -162.010 | -161.687 | -979.267 |
| 3835 | 13856 | -116.566 | -166.263 | -181.218 | -209.709 | -1115.925 |
| 3908 | 455 | -28.908 | -23.299 | -136.285 | -37.089 | -169.753 |
| 3916 | 186 | -124.480 | -294.101 | -198.532 | -368.586 | -1661.023 |
| 3956 | 941 | -12.712 | -14.462 | -136.610 | -56.272 | -88.928 |
| 3981 | 759 | -41.637 | -20.293 | -137.247 | -48.216 | -187.921 |
| 4272 | 124 | -83.804 | -7.949 | -114.571 | -24.870 | -26.091 |
| 4316 | 1431 | -112.567 | -973.802 | -156.683 | -1002.633 | -8270.700 |
| 4393 | 1292 | 193.856 | 2309.668 | 113.132 | 2380.885 | 11526.240 |
| 4473 | 2020 | -25.158 | 102.314 | -91.058 | 99.674 | -23.922 |
| 4968 | 1278 | -42.979 | -9.528 | -139.173 | -50.724 | 79.992 |


## 6. 支持、原生exact与端点距离：条件性，不冒充机制

这里的support/exact/端点距离按真实身份汇集**同一目标任一输入fragment**，仅事后分析，不作为预测gate；不是选中预测组的纯输入属性。endpoint>600s表示query距任一fragment端点较远，常见于长fragment内部，**不等于外推超过600s**。target-union support外仅1044可评点；逐fragment运行的inside/outside query计数122240/124224有另一分母，不可与目标评价层混用。

全部六臂条件位置数值如下，单位m；作用域mask与分层统计属于公开生产证据，未由本次generic主位置审计独立核验。

| 作用域 | 可评/有效reference点 | 方法 | MAE | RMSE | P95 | P99 |
|---|---|---|---|---|---|---|
| all | 71071/72196 | Mean | 1276.328 | 2168.266 | 3103.795 | 7194.825 |
| all | 71071/72196 | LOW | 527.446 | 1813.304 | 2942.040 | 7260.394 |
| all | 71071/72196 | FWD Mean | 1284.235 | 2504.358 | 3149.705 | 8208.787 |
| all | 71071/72196 | FWD LOW | 489.263 | 2201.544 | 2971.418 | 8208.787 |
| all | 71071/72196 | RTS Mean | 1198.589 | 2136.436 | 2926.265 | 6224.031 |
| all | 71071/72196 | RTS LOW | 380.127 | 1771.103 | 2700.649 | 6541.965 |
| inside_support | 70027/71128 | Mean | 1271.094 | 2131.863 | 3014.416 | 7182.773 |
| inside_support | 70027/71128 | LOW | 511.183 | 1763.711 | 2775.653 | 7229.704 |
| inside_support | 70027/71128 | FWD Mean | 1276.975 | 2477.629 | 3005.799 | 8197.485 |
| inside_support | 70027/71128 | FWD LOW | 470.297 | 2166.243 | 2753.466 | 8197.485 |
| inside_support | 70027/71128 | RTS Mean | 1195.576 | 2103.820 | 2836.927 | 6157.693 |
| inside_support | 70027/71128 | RTS LOW | 365.050 | 1725.520 | 2522.707 | 6466.466 |
| outside_support | 1044/1068 | Mean | 1627.407 | 3898.870 | 4679.348 | 12883.033 |
| outside_support | 1044/1068 | LOW | 1618.326 | 3897.027 | 4679.348 | 12883.033 |
| outside_support | 1044/1068 | FWD Mean | 1771.207 | 3899.227 | 4900.203 | 8600.364 |
| outside_support | 1044/1068 | FWD LOW | 1761.406 | 3897.295 | 4900.203 | 8600.364 |
| outside_support | 1044/1068 | RTS Mean | 1400.664 | 3720.354 | 4486.524 | 8535.535 |
| outside_support | 1044/1068 | RTS LOW | 1391.399 | 3718.566 | 4486.524 | 8535.535 |
| exact_native | 78/78 | Mean | 690.980 | 2702.281 | 831.827 | 15721.303 |
| exact_native | 78/78 | LOW | 664.461 | 2691.478 | 577.612 | 15721.303 |
| exact_native | 78/78 | FWD Mean | 736.315 | 3050.383 | 1364.974 | 18750.039 |
| exact_native | 78/78 | FWD LOW | 711.491 | 3041.463 | 1104.205 | 18750.039 |
| exact_native | 78/78 | RTS Mean | 662.096 | 2973.824 | 1364.974 | 18028.869 |
| exact_native | 78/78 | RTS LOW | 637.273 | 2964.674 | 1104.205 | 18028.869 |
| endpoint_le_60s | 4952/5070 | Mean | 1277.677 | 2458.616 | 3641.149 | 8518.252 |
| endpoint_le_60s | 4952/5070 | LOW | 790.001 | 2258.337 | 3629.560 | 8518.252 |
| endpoint_le_60s | 4952/5070 | FWD Mean | 1268.690 | 2697.134 | 3651.819 | 7412.480 |
| endpoint_le_60s | 4952/5070 | FWD LOW | 755.439 | 2516.036 | 3645.672 | 7412.480 |
| endpoint_le_60s | 4952/5070 | RTS Mean | 1191.194 | 2496.394 | 3577.341 | 7229.360 |
| endpoint_le_60s | 4952/5070 | RTS LOW | 660.575 | 2298.544 | 3572.456 | 7229.360 |
| endpoint_gt60_le600s | 25989/26610 | Mean | 1348.773 | 2350.660 | 3525.015 | 7711.099 |
| endpoint_gt60_le600s | 25989/26610 | LOW | 650.544 | 2050.701 | 3514.825 | 7766.704 |
| endpoint_gt60_le600s | 25989/26610 | FWD Mean | 1351.036 | 2608.603 | 3538.220 | 7739.566 |
| endpoint_gt60_le600s | 25989/26610 | FWD LOW | 609.963 | 2339.737 | 3531.837 | 7874.833 |
| endpoint_gt60_le600s | 25989/26610 | RTS Mean | 1268.526 | 2340.507 | 3493.242 | 6446.977 |
| endpoint_gt60_le600s | 25989/26610 | RTS LOW | 504.509 | 2036.345 | 3479.316 | 6586.979 |
| endpoint_gt600s | 40130/40516 | Mean | 1229.245 | 2000.454 | 2345.332 | 6963.653 |
| endpoint_gt600s | 40130/40516 | LOW | 415.327 | 1571.752 | 539.870 | 7029.583 |
| endpoint_gt600s | 40130/40516 | FWD Mean | 1242.892 | 2408.919 | 2330.682 | 8699.412 |
| endpoint_gt600s | 40130/40516 | FWD LOW | 378.249 | 2063.320 | 761.312 | 8702.042 |
| endpoint_gt600s | 40130/40516 | RTS Mean | 1154.209 | 1940.851 | 2287.728 | 5837.522 |
| endpoint_gt600s | 40130/40516 | RTS LOW | 264.967 | 1489.261 | 535.787 | 6319.066 |


`far_gt_1800s`与`no_input`的有效reference/可评点均为0，全部指标null，不能据本V32讨论远距离外推或无输入目标效果。inside/outside有效reference为71128/1068，可评为70027/1044；endpoint≤60/60–600/>600s有效reference为5070/26610/40516，可评为4952/25989/40130。

RTS LOW在support内净绝对误差下降10,233,234.098m，在外下降236,911.679m；约97.74%的净收益来自support内，所以本次支持的是当前状态重建在既有支撑内也有收益，不能只包装成修复外推。support内LOW MAE511.183→365.050m；外部1618.326→1391.399m，但仍有332/1044外部点评价误差恶化。

**exact-native反例必须保留：** 只有78个可评点，RTS LOW MAE664.461→637.273m，但RMSE2691.478→2964.674m、P95 577.612→1104.205m、P99 15721.303→18028.869m；57点好、21点差。RTS Mean也MAE略好、RMSE/P95/P99更差。这不支持“平滑修正所有原观测更准确”，也不能证明坏reference或机动就是根因。端点≤60s层LOW RMSE亦2258.337→2298.544m，虽然MAE/P99改善。

## 7. 论文坐标统计完整保留

沿现有合同的scene-equal target-mean聚合，单位1e-4 degree；ME有符号，MAE、absolute_ME、residual_SD各有定义，不能用ME代MAE，不能把residual_SD或印刷Eq29诊断称本表径向RMSE。每属性1427目标、91328声明query、72196有效reference、71071评价点；1388目标有评价、1385目标有至少2点。未与TMTAFF作者数据窗口/协议建立同口径比较；这些生产坐标统计未在本次generic审计范围。

| 方法 | 属性 | ME | MAE | absolute_ME | residual_SD | 印刷Eq29诊断 |
|---|---|---|---|---|---|---|
| Mean | latitude | -29.868418 | 77.134087 | 73.005607 | 39.233743 | 203999.921686 |
| Mean | longitude | -25.865700 | 81.390119 | 74.857731 | 46.241390 | 1332991.208329 |
| LOW | latitude | -2.614399 | 26.195071 | 16.905809 | 40.176409 | 203972.187494 |
| LOW | longitude | 0.594569 | 31.228885 | 19.940786 | 46.831591 | 1332964.281338 |
| FWD Mean | latitude | -30.363286 | 77.227612 | 73.966810 | 40.039170 | 204000.402545 |
| FWD Mean | longitude | -25.318006 | 83.604940 | 77.245468 | 49.631081 | 1332990.605431 |
| FWD LOW | latitude | -3.206644 | 23.783507 | 17.858430 | 39.319762 | 203972.776867 |
| FWD LOW | longitude | 1.051203 | 31.285459 | 22.169489 | 48.656306 | 1332963.770249 |
| RTS Mean | latitude | -30.274722 | 74.877410 | 72.693146 | 36.425128 | 204000.324105 |
| RTS Mean | longitude | -26.916510 | 78.450436 | 73.821362 | 42.575246 | 1332992.237953 |
| RTS LOW | latitude | -3.115132 | 19.678780 | 16.354348 | 34.670966 | 203972.691880 |
| RTS LOW | longitude | -0.534245 | 23.946949 | 18.490636 | 40.607925 | 1332965.389174 |


## 8. 成本、计划偏差与权限

| 项目 | 实际 |
|---|---|
| 程序完整墙钟 | 2612.152429s（43.536min），含64场native/reference导出、T九候选、V和写盘 |
| launcher墙钟 | 2616.685933s；与程序分开，不相加成执行时长 |
| 峰值RSS / AS | 2,012,348,416B / 8GiB；未超4GiB resident合同 |
| CPU / GPU | CPU4，affinity4–7；CUDA_VISIBLE_DEVICES空，0 GPU |
| 神经成本 | 本次0新神经模型/前向/optimizer更新；上游056关联器仍属于完整链，不能宣整个方法零参数 |
| T解析工作 | 2310×9=20790 fragment-rho fits；每次共享FWD+RTS，不把四个融合输出重复算四次滤波 |
| V解析工作 | 3851 fits；总24641；并有64旧控制场景臂重放与128新融合场景臂 |
| 预算偏差 | 7200s advisory未触发；无超时、重试、fallback、加rho/seed/训练或test读取；phase保留finalizing与COMPLETED共存仅末阶段标签 |
| 共享成本 | 64场cache被125/129复用；各实验保留自己的实际成本，不把读取/教师成本隐去 |
| 实时资格 | 未测目标设备逐更新时延/内存；整体墙钟不是在线每步延迟 |


| 计数口径 | 原生事件访问 | 观测更新 | 转移步 | RTS回扫步 | 区间内query | 区间外query |
|---|---|---|---|---|---|---|
| T九rho累加 | 5223456 | 5202666 | 5783184 | 5783184 | 581166 | 749394 |
| V一次 | 1095780 | 1091929 | 1214090 | 1214090 | 122240 | 124224 |


T事件访问5,223,456=580,384×9；每fragment首点定义prior，因此更新数=事件访问−解析fit数。T/V转移与RTS回扫在观测区间内进行，外部query另作端点状态外推，没有用负dt注入Q。上述计数来自原执行审计字段，并未由独立保存位置auditor重跑滤波验证。

FWD只在递推中不用未来位置更新，但chart比例、原WS分组、完整窗口输入、原OLS底座及rho选择仍有离线权限；RTS明确利用完整后续观测。FWD和RTS还共同使用RTS目标选的rho，所以本结果既不能认证严格在线资格，也不能据FWD负结果否定独立调优的全部因果滤波。未建立联合可训练或在线端到端融合。

## 9. Squeeze：事实、解释和下一讨论项

| 证据级别 | 发现与边界 |
|---|---|
| DE-127-1 | 同固定WS/贡献/人口，RTS Mean与RTS LOW全量MAE/RMSE/P95/P99均好；30/32场和8/9组件MAE好，去最大组件仍有增量。支持保留该离线原生状态控制。 |
| DE-127-2 | FWD Mean整体更差；FWD LOW MAE好而RMSE/P95/P99差，去最大后MAE亦差。当前RTS结果不能由FWD直接替代。 |
| DE-127-3 | RTS LOW最大组件SSE恶化、4393极端损害，exact-native/近端点层存在MAE与尾部冲突。全量净好不代表风险被校准或普遍改善。 |
| DE-127-4 | 收益主要位于true-target支撑内，而非只在外推；端点距离层不是外推时长层。 |
| AI-127-1 | 当前动态约束/后向信息或许改善了原native轨迹的reference风险；贡献规则与状态模型有交互。仍不能将动态收益唯一归为独立噪声平滑。 |
| UN-127-1 | 原观测是否已滤波、相关噪声/偏差、机动、错误分组、reference生成关系及4393失效机制未识别；R/Q/P不是实测协方差。 |
| UN-127-2 | 固定rho在其余268场/其他组件的适用性、对可信同行的同协议效益、严格在线与完整轻量链成本未知。 |


后续讨论应分开两件事：第一，若发布新的明确合同，可在不改rho/模型/人群的前提下检验更广开发场景的收益及尾部迁移；第二，利用原失败场景的保存数组分离动态/初速/未来信息与reference/分组等竞争解释，再决定有必要的最小替换。当前不按V调rho、不直接组合125门控或129分区、不把两个独立成功点的数字相加；尚无联合模型成绩。

Mode C建议回流：将“状态—观测分离在固定贡献下存在离线开发收益”加入IT-C5的DE；同时加入“前向/平滑权限、组件SSE反转、exact-native尾部”反例，收窄工作模型适用范围。问题不是已证实“原生噪声需要CV”，而是哪些合法可观测条件能识别有益状态约束、控制整条恢复链的尾部损害；候选方案须再由矩阵形成可核因果分离。

## 10. 机器可读Record / Squeeze交接

```yaml
package_type: EXPERIMENT_RECORD
experiment_id: EXPERIMENT-127-NATIVE-STATE-20261029
source_idea_id: IDEA-NATIVE-CV-STATE-CONTROL-20261029
plan_version: "1.0"
execution_status: COMPLETED_exit0
research_question_and_hypothesis: 固定WS及Mean或LOW贡献时真实原生CV/RTS能否改善完整位置恢复风险
planned_vs_actual_design_and_controls: T32固定九rho以RTS等场等fragment选择，FWD共用；原V32六臂15对按计划完成
actual_configuration: rho=1e-8; working_CV_R=I_P0=diag(1,1,1/3600,1/3600); CPU4; 0_new_neural_updates
samples_materials_or_data_provenance: 正式Level1三分卷与原manifest/alignment；056物理底座/114WS/121两控；T_test未读
all_runs_or_repetitions: [LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1]
results_with_uncertainty: 同71071点RTS_LOW_MAE527.446_to380.127_RMSE1813.304_to1771.103；单次重复开发/共享身份，无IID显著性或独立泛化保证
observations_and_deviations: 无执行配方偏差；全量尾部好但最大组件SSE与exact_native尾部不利；完整负面见正文
evidence_locations: library/docs/evidence/20261029_small_validation/127; server_original_run_private
known_limits_and_missing_evidence: T选参与分层mask不在generic主位置独立审计范围；原两控artifact_exact仅生产绑定；物理机制/全V/可信同行/在线未知
next_skill: squeeze-experiment_completed_then_ModeC_increment
```

```yaml
package_type: SQUEEZE_REPORT
report_id: SQUEEZE-127-NATIVE-STATE-20261029
source_experiment_ids: [EXPERIMENT-127-NATIVE-STATE-20261029]
source_idea_id: IDEA-NATIVE-CV-STATE-CONTROL-20261029
source_connection_C_id: IT-C5
execution_evidence: COMPLETED_exit0_and_independent_main_position_population_audit_passed
main_finding: fixed_contribution_RTS_has_development_increment_under_same_full_population
counterevidence: [FWD_tail_failure, largest_component_SSE_harm, scene4393_harm, exact_native_tail_harm]
mechanism_status: candidate_AI_not_identified
new_findings_for_mode_c: [DE-127-1, DE-127-2, DE-127-3, DE-127-4, AI-127-1, UN-127-1, UN-127-2]
interpretation_boundary: no125_RTS_combined_score_no_peer_victory_no_online_end_to_end_claim
followup_authorization: none_created_by_this_report
```
