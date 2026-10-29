# 留出共享相对分量的绝对效果：118实际记录与深挖

2026-10-29。**预先声明的“固定低source、修正高source”控制获得了值得继续发展的正向结果：同WS、同71071有效位置点，Mean的MAE1276.328→656.041m，较Anchor844.760m与NEAR927.435m也更低；RMSE亦改善。** 结果没有靠删组、改覆盖或reference拟合取得。但P99相对Mean/NEAR更差，对称gauge的MAE反而1716.225m，加入Huber没有总体增量。故支持的是当前经验锚定下的相对修正价值，**不是已辨识绝对sensor bias、完整方法净胜、作者复现或可信同行胜出**。

本报告先按record-experiment记录执行与完整结果，再按squeeze-experiment分析机制/异质性，最后提供Mode C输入。没有新训练、重跑或新增实验。初稿写入后，主代理完成一次独立保存数组的主要位置数值审计并通过；**范围限于第8节，不能称全字段、机制或物理真值均已审计**。

## 1. EXPERIMENT_RECORD：问题、实际输入与执行

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-SHARED-COMPONENT-CONTROL-118-20261029
source_idea_id: IDEA-SHARED-COMPONENT-CONTROL-20261029
plan_version: '1.0'
execution_status: completed-exploratory
research_question_and_hypothesis: 三场已见的留出相对结构能否在原V32改善绝对恢复；共享校正与局部限影响是否各有增量
planned_vs_actual_design_and_controls: 原固定V32/WS/八臂22对比/三fold留出/两gauge/Anchor与NEAR；主要设计完成，无研究样本排除
actual_configuration: 低source固定或两源对称修正；delta1000m/5轮Huber-distance重加权；不是精确Huber最优解
samples_materials_or_data_provenance: 原056 Q64 fragment→114冻结WS→115完整artifact及原manifest；32开发场景/9共享身份组件
all_runs_or_repetitions: 唯一研究run 118 attempt-1，一次正常退出0；确定性控制，非多seed重复
results_with_uncertainty: 本文全八臂/22对比/32场9组件描述；无独立测试CI，不将query当IID
observations_and_deviations: taskset相对路径在创建run/service前被拒；绝对路径修正后唯一研究执行；回传父目录缺失修正非重跑
known_limits_and_missing_evidence: gauge依赖/reference生成未知/重复开发V/非身份独立/离线输入/独立主要数组审计范围有限
next_routes: squeeze-experiment then deep-read Mode C
next_skill: squeeze-experiment
```

上游为[事先Idea1.0](../proposals/shared_component_control_20261029.md)、[融合局部矩阵SC-C1](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/fusion_action_matrix_20261029.md)、[行为规格](../specs/shared-component-control.md)及[原交接](../../HandOver/L2S-20261029-118.md)。冻结代码 **b48dae7**、独立checkout `/home/wangguangjie/djs/association-code-run118`；该版本仅用于历史定位，不是科学准入。

输入血缘：原056 `/system` 中完整物理状态/support/nearest/source → 114预测WS分组 → 三fold只用其它fold训练组估共享相对分量 → 主gauge或对称gauge校正 → 原Mean或固定5轮重加权 → 全组输出冻结 → 原115缓存reference及身份绑定与全人口评价。没有按reference选择源、gauge、组、阈值或迭代参数。每fold chart纬度也只从其训练组有support输入确定。

server原run：`/home/wangguangjie/djs/association-code/runs/LEVEL1-SHARED-COMPONENT-20261029-118-attempt-1`，launcher同路径加`-launch`。完整坐标、身份、源中心/权重和预测留该run的`private/`，本报告不复制它们。必要脱敏原件在[资料库118目录](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/118/)：`run.json`、`summary.json`、`scene-summary.json.gz`、launcher request/launch/status/退出码/日志及事后服务终态。全部32场的八臂、22比较与参数作用计数保存在原`scene-summary.json.gz`，正文不重复粘贴整份JSON；没有只保留好场景。

实际payload来自原`launcher-request.json`：

```bash
/usr/bin/taskset --cpu-list 0-3 /home/wangguangjie/djs/mtad-field-20261016/bin/python \
  tools/analysis/shared_component_control.py \
  --source-system /home/wangguangjie/djs/association-code/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1/system \
  --partition-run /home/wangguangjie/djs/association-code/runs/LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1 \
  --control-run /home/wangguangjie/djs/association-code/runs/LEVEL1-STATIC-PARTITION-POSITION-20261029-115-attempt-1 \
  --manifest configs/data/mtad-level1-scene3000.json \
  --output /home/wangguangjie/djs/association-code/runs/LEVEL1-SHARED-COMPONENT-20261029-118-attempt-1 --cpu-threads 4
```

cwd为上述冻结checkout；`CUDA_VISIBLE_DEVICES=''`，OMP/MKL/OpenBLAS线程均4。systemd服务`mtad-c347e9074412.service`，Linger=yes、Restart=no；主运行、launcher returncode、`exit_code.txt`均0。事后服务为inactive/dead、Result=success、ExecMainStatus=0。stdout有COMPLETED，stderr与worker.stderr为空。

程序UTC **2026-10-29T11:29:12.999946+00:00→2026-10-29T11:30:44.896055+00:00**，完整run单调计时 **91.896156s**；summary写入前快照为91.711402s，不能把它当最终总成本。launcher请求开始至终态 **108.528399s**。峰值RSS **2007023616B**，AS8GiB，CPU亲和0–3；600s advisory未触发。0GPU、0模型构建/前向/optimizer、0新score/分组/阈值/normalizer、0ZIP/CSV/T_test。该91.896s是整次离线缓存实验成本，不是在线每次更新时延，也没有GPU模型速度数据。

执行偏差完整记录：主代理执行记录显示，首次payload的`taskset`相对路径被launcher在mkdir/创建service前拒绝，没有读取研究数据或创建118 run；改成`/usr/bin/taskset`后才有上面唯一研究运行，原L2S已同步绝对路径。初次SFTP回传因本地父目录不存在被拒，创建新目录后回传成功，亦非实验重跑。主代理运行前修正了不足拟合fold的chart零哨兵统计：没有合法chart时相对残差为None并保留未定义分母，**没有改变预测**；必要软件检查7项通过，不冒充MTAD结果。上述启动前/回传操作的依据为主代理操作记录；下载目录的JSON是实际唯一run原件。

## 2. 固定完整人口与全部八臂

八臂都是同一WS、同顺序/首代表、同物理输入/support与有效mask：3851节点、1605组、1427目标；217个重复预测组、1388个首代表匹配目标、39个未匹配目标、无参考目标0。声明target-query91328，参考有效位置点72196，实际共同可评71071，缺预测1125。位置点覆盖98.441742%，轨迹覆盖97.266994%；102720个组-query位置声明、31649个prediction-without-target点。**22个配对的gained/lost/救回/丢失目标全部0；same-anchor共同点均71071，changed-anchor为0。** 覆盖、重复不是本项贡献，也没有解决既有WS全V覆盖/重复问题。

原WS_MEAN完整artifact重放115 exact；所有其它臂的组/代表/valid及人口不变由实际运行检查。它是实现内绑定证据，独立数组审计状态另列，不能混称两次独立实验。

| 方法 | MAE(m) | RMSE(m) | P95(m) | P99(m) | 场景宏MAE(m) |
|---|---:|---:|---:|---:|---:|
| WS_MEAN | 1276.328 | 2168.266 | 3103.795 | 7194.825 | 1249.718 |
| WS_HUBER | 1275.620 | 2168.558 | 3114.053 | 7194.825 | 1249.739 |
| WS_FIXED_MEAN | 656.041 | 1864.654 | 2444.381 | 7541.588 | 484.962 |
| WS_FIXED_HUBER | 656.374 | 1866.778 | 2455.881 | 7541.588 | 485.064 |
| WS_SYMMETRIC_MEAN | 1716.225 | 2300.417 | 2390.411 | 7008.621 | 1595.336 |
| WS_SYMMETRIC_HUBER | 1717.793 | 2301.627 | 2395.856 | 7008.621 | 1595.791 |
| WS_ANCHOR | 844.760 | 2457.607 | 4385.889 | 10460.984 | 986.493 |
| WS_NEAREST_SUPPORTED | 927.435 | 2173.150 | 3873.502 | 7301.504 | 886.352 |

FIXED由事前稳定字符串排序选低source不动、高source减去估计中心；SYMMETRIC各移半量。两者都未通过reference选取。Huber臂仅重加权位置，速度/航向保留相应普通Mean；Anchor/NEAR是原普通强位置控制。参数不是经V搜索的最佳值。

## 3. 全22比较：不能只看对Mean

以下差均为candidate−baseline；场景和组件列为MAE改善/恶化/相同，阈值1e−9m；全部比较共同点71071。P95/P99是全共同误差向量的分位数差，不是逐点误差差的分位数。

| candidate vs baseline | ΔMAE(m) | ΔRMSE(m) | ΔP95(m) | ΔP99(m) | 场景好/差/同 | 组件好/差/同 |
|---|---:|---:|---:|---:|---:|---:|
| WS_HUBER_vs_WS_MEAN | -0.709 | 0.292 | 10.258 | 0.000 | 11/21/0 | 3/6/0 |
| WS_FIXED_MEAN_vs_WS_MEAN | -620.287 | -303.611 | -659.414 | 346.763 | 31/0/1 | 8/0/1 |
| WS_FIXED_HUBER_vs_WS_MEAN | -619.955 | -301.488 | -647.914 | 346.763 | 32/0/0 | 9/0/0 |
| WS_SYMMETRIC_MEAN_vs_WS_MEAN | 439.897 | 132.151 | -713.384 | -186.204 | 2/30/0 | 0/9/0 |
| WS_SYMMETRIC_HUBER_vs_WS_MEAN | 441.465 | 133.361 | -707.939 | -186.204 | 2/30/0 | 0/9/0 |
| WS_ANCHOR_vs_WS_MEAN | -431.569 | 289.341 | 1282.094 | 3266.159 | 26/6/0 | 5/4/0 |
| WS_NEAREST_SUPPORTED_vs_WS_MEAN | -348.894 | 4.885 | 769.707 | 106.679 | 32/0/0 | 9/0/0 |
| WS_FIXED_HUBER_vs_WS_FIXED_MEAN | 0.332 | 2.123 | 11.500 | 0.000 | 11/7/14 | 4/2/3 |
| WS_FIXED_HUBER_vs_WS_HUBER | -619.246 | -301.780 | -658.172 | 346.763 | 31/0/1 | 8/0/1 |
| WS_SYMMETRIC_HUBER_vs_WS_SYMMETRIC_MEAN | 1.568 | 1.210 | 5.445 | 0.000 | 9/9/14 | 3/3/3 |
| WS_SYMMETRIC_MEAN_vs_WS_FIXED_MEAN | 1060.184 | 435.762 | -53.970 | -532.967 | 0/32/0 | 0/9/0 |
| WS_SYMMETRIC_HUBER_vs_WS_FIXED_HUBER | 1061.419 | 434.849 | -60.025 | -532.967 | 0/32/0 | 0/9/0 |
| WS_HUBER_vs_WS_ANCHOR | 430.860 | -289.049 | -1271.836 | -3266.159 | 6/26/0 | 4/5/0 |
| WS_FIXED_MEAN_vs_WS_ANCHOR | -188.718 | -592.952 | -1941.508 | -2919.397 | 28/4/0 | 8/1/0 |
| WS_FIXED_HUBER_vs_WS_ANCHOR | -188.386 | -590.829 | -1930.008 | -2919.397 | 28/4/0 | 8/1/0 |
| WS_SYMMETRIC_MEAN_vs_WS_ANCHOR | 871.465 | -157.190 | -1995.478 | -3452.364 | 4/28/0 | 3/6/0 |
| WS_SYMMETRIC_HUBER_vs_WS_ANCHOR | 873.033 | -155.980 | -1990.033 | -3452.364 | 4/28/0 | 3/6/0 |
| WS_HUBER_vs_WS_NEAREST_SUPPORTED | 348.185 | -4.592 | -759.449 | -106.679 | 0/32/0 | 0/9/0 |
| WS_FIXED_MEAN_vs_WS_NEAREST_SUPPORTED | -271.393 | -308.496 | -1429.121 | 240.083 | 28/4/0 | 7/2/0 |
| WS_FIXED_HUBER_vs_WS_NEAREST_SUPPORTED | -271.061 | -306.373 | -1417.621 | 240.083 | 28/4/0 | 7/2/0 |
| WS_SYMMETRIC_MEAN_vs_WS_NEAREST_SUPPORTED | 788.790 | 127.267 | -1483.091 | -292.884 | 0/32/0 | 0/9/0 |
| WS_SYMMETRIC_HUBER_vs_WS_NEAREST_SUPPORTED | 790.358 | 128.477 | -1477.646 | -292.884 | 0/32/0 | 0/9/0 |

主2×2的事实：无校正Huber相对Mean只−.709m MAE、RMSE+.292m，场景宏MAE还+.021m；固定gauge校正Mean相对Mean−620.287m MAE（48.599%）、RMSE−303.611m，是主要正向信号；校正后再Huber反而MAE+.332m、RMSE+2.123m。相同Huber下加入固定校正仍−619.246m MAE，说明主要变化来自共享修正这条干预，而非该局部重加权。不能据此证明所有鲁棒损失无价值。

FIXED_MEAN同时优于Anchor/NEAR的整体MAE与RMSE，已经超过本次简单强对照，值得继续发展；但对Mean P99+346.763m、对NEAR P99+240.083m，仍有尾部代价。对Anchor四项位置指标均更好，却有4场、1组件MAE更差。对NEAR有4场、2组件MAE更差。因此不能称所有场景/全部风险一致胜出。

SYMMETRIC_MEAN相对Mean MAE+439.897m、RMSE+132.151m，9/9组件MAE均差；它的P95/P99更低，说明平均误差与远尾也不完全同向。FIXED比SYMMETRIC的MAE在32/32场和9/9组件更好，**绝对gauge是实质条件，不是可忽略的坐标选择**。两源对齐的经验收益不能反推哪个源物理无偏，亦不能在新场景按reference选gauge。

## 4. 场景、共享组件和支撑异质性

全部9组件如下，C编号只为本表阅读，完整component ID保留。每格为位置MAE(m)；每组件完整RMSE/P95/P99及22比较在原summary的`components`中。

| C/原component ID | 场景数 | 点数 | Mean | Huber | FixedMean | FixedHuber | SymMean | SymHuber | Anchor | NEAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C01 `component-0b9d7d23108cbf0a74ed5828` | 1 | 248 | 821.917 | 831.725 | 155.301 | 155.301 | 862.522 | 862.522 | 1333.727 | 602.099 |
| C02 `component-23608a7a22c5322f7254ffc1` | 19 | 58707 | 1241.122 | 1240.156 | 663.209 | 663.660 | 1678.965 | 1680.835 | 745.256 | 895.331 |
| C03 `component-75391b641c3cd7287e382451` | 1 | 2406 | 1446.795 | 1446.581 | 627.947 | 626.766 | 2133.319 | 2134.022 | 710.585 | 1105.476 |
| C04 `component-9ea441151d6468bd21826aea` | 1 | 204 | 2782.578 | 2782.721 | 2222.062 | 2222.144 | 3131.623 | 3131.706 | 2187.423 | 2176.088 |
| C05 `component-a077aa3eb774a8b670334e7c` | 1 | 713 | 997.787 | 997.826 | 345.056 | 345.056 | 1303.067 | 1303.067 | 2550.029 | 711.274 |
| C06 `component-c260c3bbf884e315f715f00b` | 1 | 186 | 922.333 | 922.334 | 435.164 | 435.164 | 1087.148 | 1087.148 | 582.920 | 702.111 |
| C07 `component-c7082a697d934f0004db48b9` | 6 | 6463 | 1498.914 | 1499.569 | 605.342 | 605.342 | 1877.188 | 1877.184 | 1511.891 | 1121.447 |
| C08 `component-d091361fb95b75d6d2a1a9c5` | 1 | 2020 | 1424.401 | 1424.468 | 644.255 | 644.246 | 1929.912 | 1929.904 | 872.834 | 1044.239 |
| C09 `component-fdbc675a9f9f6a7e1c67e3bb` | 1 | 124 | 1186.850 | 1186.768 | 1186.850 | 1186.768 | 2091.296 | 2091.215 | 2729.546 | 834.548 |

FIXED_MEAN对Anchor的恶化场为1293(+278.813m)、1830(+193.356m)、2038(+34.638m)、3835(+179.176m)；对NEAR为1293(+207.675m)、2038(+45.974m)、3835(+173.979m)、4272(+352.302m)。这些场景保留。对Mean没有场景MAE恶化，4272相同；FIXED_HUBER在这一场微改善使其对Mean变为32/32好，但总体反而略差，不能用“好场更多”掩盖加权净效应。

最大组件C02占19/32场、58707/71071点（82.603%）。本次记录阶段另用公开逐场的点数×MAE/MSE重构“排除该组件”敏感性，**仅事后分析同一次结果，没有重跑或排除主要人口**；计算直接来自原`scene-summary.json.gz`每场的`methods[arm].position.all`：`MAE=Σ(n_s·MAE_s)/Σn_s`、`RMSE=sqrt(Σ(n_s·MSE_s)/Σn_s)`，其中`n_s=intersection_count`；未调用预测器、未读取新数据。P95/P99不能由场景分位数合成，故不填此子集尾部。剩13场/12364点：

| 方法 | 排除最大组件MAE(m) | 排除最大组件RMSE(m) |
|---|---:|---:|
| WS_MEAN | 1443.496 | 2861.164 |
| WS_HUBER | 1444.008 | 2861.492 |
| WS_FIXED_MEAN | 622.008 | 2555.913 |
| WS_FIXED_HUBER | 621.777 | 2555.783 |
| WS_SYMMETRIC_MEAN | 1893.144 | 2916.625 |
| WS_SYMMETRIC_HUBER | 1893.277 | 2916.667 |
| WS_ANCHOR | 1317.228 | 3921.579 |
| WS_NEAREST_SUPPORTED | 1079.870 | 2860.438 |

正向信号不只来自最大组件；FIXED_MEAN在剩余子集的MAE/RMSE也好于Mean/Anchor/NEAR。该子集由公开逐场MAE/MSE和点数重构；逐场主要统计随后获独立数组核验，但审计没有单列这个排除子集，不能把它另算独立重复。

支撑分层按完整真实身份的原输入支持定义，只用于评价，未参与选法；inside_support共同70027点、outside_support1044点，far_gt_1800s/no_input可评点均0（不得把无样本当效果）。全部八臂如下：

| 方法 | 支撑内MAE / RMSE(m) | 支撑外MAE / RMSE(m) |
|---|---:|---:|
| WS_MEAN | 1271.094 / 2131.863 | 1627.407 / 3898.870 |
| WS_HUBER | 1270.375 / 2132.164 | 1627.406 / 3898.870 |
| WS_FIXED_MEAN | 644.636 / 1819.269 | 1421.068 / 3832.988 |
| WS_FIXED_HUBER | 644.973 / 1821.477 | 1421.068 / 3832.987 |
| WS_SYMMETRIC_MEAN | 1708.443 / 2264.821 | 2238.210 / 4023.922 |
| WS_SYMMETRIC_HUBER | 1710.034 / 2266.068 | 2238.210 / 4023.922 |
| WS_ANCHOR | 844.528 / 2439.535 | 860.342 / 3460.636 |
| WS_NEAREST_SUPPORTED | 917.013 / 2136.838 | 1626.465 / 3901.317 |

FIXED_MEAN支撑外MAE1421.068m仍差于Anchor860.342m，虽优于Mean/NEAR约1627/1626m；支撑外只占1044点，不能用整体收益隐藏该条件。支撑外RMSE3832.988m亦高于Anchor3460.636m。这为后续判断何时保留anchor提供条件证据，但本轮没有据它学习或拼接oracle。

## 5. 中间机制与参数作用率

- 1605组中1040组有共同支撑跨源差，565组没有；全部组照常输出。96个fold估计中92启用、4不足最小2训练组而零校正。按组rank映射有1598组落在可用估计fold、7组回退，**这是共享估计可用率，不等于1598组的输出位置均发生变化**；固定低source的单源低source组可能不动。
- 原共同支撑跨源pair-query43004；合法chart可定义的留出相对残差覆盖1034组/42779 pair-query，6组/225 pair-query未定义。1293和2038各2组未定义，4272的2组全未定义；4272场均相对残差为None，未用零chart假装可估。
- 对上述1034个可定义组按组等权汇总，相对残差3084.587→1186.024m；计算为`Σ(scene_relative_mean × relative_residual_defined_group_count)/Σ(relative_residual_defined_group_count)`，None场不填零、未定义组不混入分母。它是**输入局部equirectangular差减冻结center**的剩余范数，既不是校正后球面两轨迹实际差，也不是真实bias误差。坐标近似、运动/时差和错匹配仍可产生这一结构。
- 92个启用fold的中心范数min/median/max为1343.041/2855.421/4552.722m；没有用reference筛掉大中心、重估gauge或挑健康组。

| 重加权臂 | 末轮权重<1的node-query | 非均匀权重group-query | 非均匀占全部102720组-query |
|---|---:|---:|---:|
| WS_HUBER | 71236 | 35777 | 34.830% |
| WS_FIXED_HUBER | 19746 | 9997 | 9.732% |
| WS_SYMMETRIC_HUBER | 19747 | 9998 | 9.733% |

node-query总可能人口为3851×64＝246464，但公开原件未单列总supported node-query分母，故不把上述计数说成“支持观测降权率”。末轮被降权也不等其输出一定改变：所有成员同权因子、单成员和无support都复制相应Mean。非均匀权重显著减少与共享修正后的成员差较小一致，但**不证明偏差物理类型已识别或Huber因果失效**；总体新增无益是该固定δ/五轮配方的实际结果。

## 6. 观察—解释—尚未证明

| 层级 | 当前内容 | 对应反证/边界 |
|---|---|---|
| DE | 不用当前fold拟合共享相对分量，固定低source校正Mean在既定32场获得明显绝对误差收益，超过本轮Anchor/NEAR强控制 | 仍有对强控制恶化场、P99代价，材料重复用于开发且共享身份，不是新独立测试 |
| DE | 固定校正与对称校正在全部场景/组件有显著方向差异 | gauge并非可忽略；相对结构没有独立决定绝对零点，不能称无锚绝对校准 |
| DE | 局部重加权在校正之后没有总体额外收益 | 本配方不需保留Huber分支；不能据一次δ/次数决定所有鲁棒/异常模型无价值 |
| AI | 跨组共享相对分量携带了有用的源间差异，经验固定低源使当前reference误差更小 | 物理偏差、生成位移、时差、机动/插值和错配仍可解释部分结构；未分离这些原因 |
| UN | 能否从合法输入可靠建立/选择绝对锚点、能否跨新场景/源语义泛化、原生因果时序下能否保持收益 | 不能靠reference选source、变gauge或删坏场；后继须独立Idea与真实对照 |

旧090C在原A0全V上的约565m或固定cap128共同点上的数字，与这里新WS/V32不属同人口，不能直接说118胜过或输给旧模型；应另作同条件比较。TMTAFF、PTFM原表也不是本轮同协议基线。本项0神经网络参数与短CPU成本是实际工程性质，**尚不证明在线端到端、特定设备实时或效果—模型大小前沿**。

## 7. SQUEEZE_REPORT与Mode C输入

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-SHARED-COMPONENT-CONTROL-118-20261029
input_experiment_record_ids: [EXP-SHARED-COMPONENT-CONTROL-118-20261029]
claim_scope: 原V32/WS/同71071点，离线固定低source的跨组相对校正改善主要误差；gauge和尾部仍限制
primary_result_with_uncertainty: FIXED_MEAN MAE656.041/RMSE1864.654，Mean1276.328/2168.266；31场/8组件好、各1同；对强控制仍有恶化
process_dynamics: 唯一一次0训练运行完成32终点，程序91.896s/launcher108.528s/RSS2007023616B；非在线延迟
mechanism_and_intermediate_findings: 相对留出残差下降、固定与对称gauge风险不同、Huber总体无增量；物理bias/绝对锚点未识别
errors_heterogeneity_and_boundary_conditions: P99对Mean及NEAR变差；支撑外仍输Anchor；32场/9组件及排最大组件保留
robustness_sensitivity_and_alternative_explanations: 事前两gauge/2x2/强控制；事后去最大组件仍有主要收益，不能构成独立重复
data_quality_and_provenance_gaps: reference生成未知、Q64全窗未来信息、共享身份与V反复开发；独立主要数组审计已通过声明范围，其余字段/机制未核
new_P: none；细化既有SC-P1/P2与C51/C10条件
new_F: 固定低source共享修正有明显绝对收益，但对称gauge恶化、Huber无增量且尾部有代价
new_M: AI/限定操作链：目标绝对风险对相对修正的绝对gauge敏感；分工中的共享项有效，当前局部重加权无净增量
open_questions: 合法锚定与source泛化、同条件旧学习恢复器/可信同行、新场景及原生在线接口、尾部来源
next_routes: deep-read Mode C；后继不自动发布/扩量/训练
next_skill: deep-read Mode C
```

回流建议只新增一条有边界F和一条操作机制M，不把八臂拆成八个创新、把gauge差异宣布成新领域定理，或把相对结构解释为唯一根因。**当前固定gauge的绝对收益值得保留和深入，Huber组合本轮不晋级；是否扩量、建立锚定或训练新模块，须先由矩阵形成下一份具体Idea。** 初稿写入后的独立审计追加见下一节，未倒称其在初稿前已完成。

## 8. 后补：独立保存数组主要位置审计

原件为资料库118目录的`independent-saved-array-audit.json`；一次执行COMPLETED/exit0，32场、9组件、8臂、22配对。审计从保存的预测/reference/valid与结构重建主要位置统计，核逐场及整体/组件的own和pair-common人口、MAE/MSE/RMSE/P95/P99、覆盖；另核gained/lost/救回/丢失目标计数、固定保存人口/reference/结构。共有22232项numeric统计、26015项exact array/value检查，条件检查129422项。它是同一118结果的独立数值核验，不是新实验、独立数据重复或模型复现。

容差为rtol1e−9，米制atol1e−5、MSE atol.001m²，arrays按含NaN exact；最大绝对差MAE1.710e−10m、RMSE1.501e−10m、P95 1.760e−9m、P99 2.015e−9m、MSE1.237e−6m²，覆盖差0，均通过。审计成本4.922326s、RSS111775744B，另有user CPU5.203018s/system CPU.357030s；**不并入91.896s主运行峰值或假作GPU成本**。

未独立核：拟合中心/三fold/Huber机制、输入来源与原115绑定、独立manifest/真实身份component定义、support/observed/anchor分层、论文指标、逐点benefit/harm计数、速度/航向数值、reference物理正确性以及同行/泛化主张。原通用审计JSON中的119域检查说明不属于118任务，本报告不借它给118增加核验范围。相对残差与作用率依然是原生产工具记录，不能写成独立机制认证。
