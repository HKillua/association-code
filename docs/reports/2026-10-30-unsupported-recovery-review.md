# 146：支持外端点/速度分离 — Record→Squeeze

2026-10-30。科学及限定独立输出审计均COMPLETED/exit0。支持“当前观测RTS的大速度外传可被局部替换修复”，不支持“完整融合系统已经稳定超越强基线”。原145接口失败另保留。

## Record：计划与真实执行

- EXPERIMENT_RECORD：`LEVEL1-UNSUPPORTED-RECOVERY-20261030-146-attempt-1`；source_idea_id `IDEA-UNSUPPORTED-RECOVERY-CONTINUATION-20261030`、plan_version1.0；[原科学Idea](../proposals/unsupported_recovery_20261030.md)、[独立修复Idea](../proposals/unsupported_recovery_continuation_20261030.md)、[L2S146及补项合同](../../HandOver/L2S-20261030-146.md)。来源Mode C1.59 C55/F242/M107/E141、技术1.60/E142。
- freeze fb821d9，新checkout `association-code-run146`，原run/launcher新目录一次。固定T32/12组件、V32/9组件、Q64/WS组；T14/V21臂、91/210两两比较、全部五主量完成。完整位置T37635/V71071，零支持T1487/V2770，无人口删除/改valid。0新全局参数拟合/rho选择/神经/teacher/组/T_test。
- 科学UTC06:16:29.265141→06:31:48.416905（北京时间14:16→14:31），程序919.151809284s/launcher922.699939933s，峰值2344538112B；CPU4/taskset8–11、RSS6/AS12GiB，1800s advisory未超建议。T287.327s、V631.807s，包含缓存/复放/恢复/完整评价。
- 6142个必要节点的原obs-rho KF/RTS核，1670167原生事件处理、1849991转移和RTS步；节点由**所有预测组的零支持query**决定，不由reference可评点或142已知坏点选择。所有复放Q64与原137保存值绑定。不是0计算，不是新参数训练，也不是单query线上延迟。
- science unit `mtad-fb7cedadb32a.service`，真实inactive/dead success、MainPID0、Restart=no、Linger=yes。[run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_unsupported_recovery/146/run.json)、[argv/env](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_unsupported_recovery/146/launcher/request.json)、[退出/服务](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_unsupported_recovery/146/observed-services-terminal.json)。原坐标/身份/端点/权重/大预测仅留服务器。

### 实际干预与明确偏差

所有新策略只换零支持query的位置，原支持内和全部速度/航向输出精确不变。因子为CV工作端点/原native端点 × CV速度/原全片段OLS速度；OLS以原native端点锚定，非拟合截距。另设CV端点静止HOLD；强Qhard支持内＋CV端点/OLS速度；最后在该Q恢复基础上，将零支持改为组内所有片段按source等总质量、源内均分的恢复。来源仅定义相等关系，无source编号/命名优先。

原145在首T场157核后list接口失败，程序14.077s/launcher17.496s、RSS408137728B；0完整评价，原代码/目录/服务不重启，不能倒填本146成功。详见[失败Record/Squeeze](2026-10-30-unsupported-recovery-entry-failure.md)。146科学不改，仅修容器、将含truth artifact读取后移；新预测先存再打开评价字段。

共享评价没有输出计划要求的anchor-before/after层，终态后按L2S补充一次只读固定64场记账，另存[additional-zero-support-accounting](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_unsupported_recovery/146/additional-zero-support-accounting.json)。code577251d，13.301073591s/RSS210722816B；0新预测/fit/阈值，全部方法共用原零支持/旧anchor时序层、组件/去最大。再对原142已经暴露的4393最坏anchor-after固定27点绑定旧误差并列新结果，明确为事后Squeeze，不是V选参。原summary/预测未改，不能把补项倒填为原输出audit范围。

## 完整位置强对照

单位m。缩写CVP/NP为CV/native端点，CVV/OLSV为CV/OLS速度；Q开头臂保持Qhard支持内，其余保持OBS-RTS支持内。SOURCE_ALL为来源均衡全组候选。历史LOW类只作强控，不是新方案。

| T完整臂 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| SOURCE_EQUAL | 1311.223 | 2878.589 | 2544.797 | 10480.821 |
| SOURCE_NEAREST | 987.325 | 2883.289 | 4116.754 | 10406.995 |
| Qsoft | 1175.045 | 2810.466 | 2434.711 | 10406.995 |
| Qhard | 619.476 | 2666.191 | 2436.283 | 10500.827 |
| REFERENCE_FORWARD | 1265.011 | 2867.618 | 2582.280 | 9154.836 |
| REFERENCE_RTS | 1244.531 | 2761.939 | 2452.560 | 9453.277 |
| OBS_FORWARD | 1315.791 | 2838.692 | 2743.712 | 9776.039 |
| OBS_RTS | 1267.062 | 2709.036 | 2657.598 | 9383.673 |
| OBS_CVP_OLSV | 1278.069 | 2870.137 | 2529.231 | 10498.146 |
| OBS_NP_CVV | 1268.157 | 2710.695 | 2660.941 | 9396.365 |
| OBS_NP_OLSV | 1278.366 | 2870.503 | 2526.916 | 10495.446 |
| OBS_HOLD | 1231.055 | 2576.496 | 2528.824 | 8272.118 |
| Q_CVP_OLSV | 619.179 | 2665.797 | 2441.184 | 10498.146 |
| Q_SOURCE_ALL | 628.156 | 2679.088 | 2469.147 | 10840.233 |

| V完整臂 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| MEAN | 1276.328 | 2168.266 | 3103.795 | 7194.825 |
| NEAREST_SUPPORTED | 927.435 | 2173.150 | 3873.502 | 7301.504 |
| LOW历史控制 | 527.446 | 1813.304 | 2942.040 | 7260.394 |
| FixedLOW历史控制 | 447.060 | 1740.814 | 1012.563 | 7526.008 |
| Qgate历史控制 | 435.061 | 1739.320 | 1014.478 | 7526.008 |
| SOURCE_EQUAL | 1276.441 | 2168.137 | 3097.435 | 7194.825 |
| Qsoft | 1128.755 | 2060.605 | 2990.606 | 7194.825 |
| Qhard | 547.721 | 1829.356 | 3125.980 | 7260.394 |
| LINEAR_SOFT | 1162.880 | 2085.495 | 3011.683 | 7194.825 |
| SHUFFLED_SOFT | 1276.493 | 2168.187 | 3101.807 | 7194.825 |
| SOURCE_NEAREST | 926.659 | 2172.333 | 3872.747 | 7301.504 |
| REFERENCE_FORWARD | 1284.338 | 2504.241 | 3142.724 | 8208.787 |
| REFERENCE_RTS | 1198.685 | 2136.307 | 2913.120 | 6224.031 |
| OBS_FORWARD | 1828.857 | 27538.260 | 3326.260 | 10033.398 |
| OBS_RTS | 1784.722 | 27531.051 | 3332.561 | 9514.676 |
| OBS_CVP_OLSV | 1241.072 | 2151.799 | 3123.313 | 7249.745 |
| OBS_NP_CVV | 1787.312 | 27571.591 | 3339.257 | 9514.969 |
| OBS_NP_OLSV | 1242.084 | 2161.748 | 3116.900 | 7190.458 |
| OBS_HOLD | 1297.411 | 2389.159 | 3268.067 | 8867.580 |
| Q_CVP_OLSV | 546.710 | 1817.590 | 3130.808 | 7313.695 |
| Q_SOURCE_ALL | 551.752 | 1756.077 | 3152.028 | 7216.507 |

## 五项预定主差与完整异质性

差candidate−baseline；2000次seed20组件描述区间，不是独立确认性CI。最大组件按场景数；不按V删坏点、选速度上限或rho。

| 分区/比较 | MAE差m [95%] | 场景好/差/平 | 组件好/差/平 | 去最大差m |
|---|---|---|---|---:|
| T CVP_OLSV−OBS | +11.007 [−72.806,236.926] | 21/8/3 | 6/4/2 | +96.234 |
| T NP_OLSV−OBS | +11.303 [−73.560,237.392] | 20/9/3 | 5/5/2 | +95.795 |
| T HOLD−OBS | −36.008 [−152.366,31.292] | 16/13/3 | 7/3/2 | −45.444 |
| T Q_CVP_OLSV−Qhard | −.296 [−.858,1.999] | 20/9/3 | 6/4/2 | +.439 |
| T SOURCE_ALL−Q_CVP_OLSV | +8.977 [1.890,11.114] | 8/21/3 | 2/8/2 | +4.432 |
| V CVP_OLSV−OBS | −543.650 [−620.087,−32.839] | 25/5/2 | 7/2/0 | −123.422 |
| V NP_OLSV−OBS | −542.638 [−618.903,−33.185] | 24/6/2 | 7/2/0 | −123.193 |
| V HOLD−OBS | −487.311 [−577.195,48.083] | 18/12/2 | 7/2/0 | +7.813 |
| V Q_CVP_OLSV−Qhard | −1.011 [−2.744,.809] | 22/8/2 | 6/3/0 | −.229 |
| V SOURCE_ALL−Q_CVP_OLSV | +5.042 [−29.407,9.929] | 8/22/2 | 4/5/0 | −18.358 |

V相对原OBS，CVP_OLSV有2260点改善/510点增害/68301平，益40009239.321m、害1371525.253m；**232点增害>1km，最大32327.462m**。去最大仍有79个>1km增害，完整平均改进不表示各点安全。T该臂反为+11.007m，913好/574坏，435个>1km新增害、最大36272.270m，不能隐藏T反向。

Q_CVP_OLSV相对原Qhard：V1465好/1305坏，0个>1km新增害、最大293.288m；T809好/678坏，0个>1km、最大891.528m。但V P953125.980→3130.808m、P997260.394→7313.695m更差，不能用MAE−1.011m或RMSE小降称稳定整体增量。

来源均衡SOURCE_ALL相对Q_CVP_OLSV：V771好/1068坏，438个新增害>1km、最大7819.460m，T273个>1km、最大7511.480m；完整MAE更差而RMSE/P99有益、去最大方向反转。无source优先不等所有片段等权就可靠，未因此采用盲均衡作为最终贡献。

完整原14/21臂、91/210比较、覆盖/重复、固定支持层、主差positive/negative及所有场组件见[T汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_unsupported_recovery/146/T-summary.json)、[V汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_unsupported_recovery/146/summary.json)及同目录逐场gzip，不只保留本表。

## 支持外before/after与原27点反例

各臂共用原anchor原生跨度定义，限原零支持可评人口；不等真实identity总跨度，不按方法重新划层。

| 分区/侧与点数 | 原OBS | CVP_OLSV | NP_CVV | NP_OLSV | HOLD | Q_SOURCE_ALL |
|---|---:|---:|---:|---:|---:|---:|
| T before 728 | 5612.037 | 7551.813 | 5603.724 | 7560.658 | 6219.688 | 7742.670 |
| T after 759 | 6197.953 | 4883.183 | 6260.208 | 4889.400 | 3829.675 | 5145.264 |
| V before 1655 | 6103.858 | 4555.263 | 6113.396 | 4551.600 | 6542.590 | 4553.773 |
| V after 1115 | 36004.473 | 3650.402 | 36155.385 | 3720.313 | 4291.679 | 3974.012 |

T before原OBS的收益被OLS速度替换损害，V before却改善；HOLD在原27点更好，不意味着完整人口更好。零支持V的MAE18139.666→4191.032m，RMSE139238.271→7671.044m，但旧reference-RTS零支持MAE3434.136m仍更好；完整V reference-RTS1198.685m也优于本OBS修复1241.072m。

原142已暴露4393固定27个after点：原OBS MAE1222553.040m/max2389909.389m；只换native端点保留CV速度仍1224914.420m/max2392304.978m；只换OLS速度保留CV端点降至31059.340m/max32081.499m。HOLD该分支MAE15848.087m/max30747.109m，但完整V较CVP_OLSV差。原Qhard在这27点33372.499m，说明替换后**仍是三万米级错误，未解决物理参考/输入异常**。原真实船速、native异常来源、reference物理准确性仍UN。

## 限定独立核验与Squeeze

独立输出审计code cfcb687，checkout `association-code-audit146-outputs`、unit `mtad-0cad72c6d74b.service`，COMPLETED/exit0、inactive/dead success、Restart=no；CPU2/600s advisory，16.779006251s/RSS439578624B。78742712数组/计数元素、8265数值标量通过，不是独立样本数。[audit原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_unsupported_recovery/146/independent-output-audit.json)及audit-launcher保留真实命令/退出。独立NumPy核native参考/valid、首anchor/重复、全主指标/覆盖/场组件/去最大、主actual harm/bootstrap及支持内/非位置exact。

6142核Q64绑定为科学生产过程；output audit不重放全部核、不核真实噪声或全部支持层；额外before/after和27点记账另记13.301s成本与范围。原145失败、preflight错误标记、实际容器缺口均如实保留，未当状态方法负结果。

`SQ-UNSUPPORTED-RECOVERY-20261030-146`消费本Record。DE：在固定原模型/输入/评价下，主要灾难分支对传播速度替换敏感、对端点位置替换不敏感；操作层证据从142过程定位推进为真实干预效果。AI/UN：不能推成真实高速、物理故障原因、普遍最优OLS或支持外全部应停止传播。T before、部分大新增害、HOLD/来源均衡的相反取舍及旧reference状态强控共同约束该机制。

去留：保留支持域与传播规则分离的**局部修复证据**；不把它叫完整模型稳定提升，不自动扩268/训练/改rho/删点。后继可研究同合法信息下before/after/时距与组候选条件，以及如何保住支持内收益并改善强贡献控制；须先回Mode C再立新Idea。拟回流F245/E145、限定增强M107，论文/精读104不增。真实K2、共享T/V组件/历史开发V、source输入语义及可信同行缺口保持；无在线/端到端/异构K≥3认证，无自动后继。

回流已完成：[Mode C1.61](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-161)、[OD v1.6](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md#unsupported-recovery-feedback)登记F245/E145，M107只限定增强操作层证据，物理来源仍UN。原anchor沿冻结系统作归因对照，不据此宣最终多传感器优先规则或完整旧系统任意片段置换等价。执行回执[S2L146](../../HandOver/S2L-20261030-146.md)。
