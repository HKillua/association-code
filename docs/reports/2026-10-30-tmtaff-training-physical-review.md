# 153：冻结107训练侧物理恢复 — Record→Squeeze

2026-10-30。科学COMPLETED/exit0。固定训练T8/16正对的保存位置输出在全部1024个查询上只有**一个不同的经纬度二元组**；换序、自配对、state_hold也完全相同。它已在T暴露严重恢复失效，不能以113的V失败判赢TMTAFF原方法。

## Record：输入、实际执行与完整结果

EXPERIMENT_RECORD `LEVEL1-TMTAFF-T-PHYSICAL-PROBE-20261030-153-attempt-1`，source_idea_id `IDEA-TMTAFF-T-PHYSICAL-PROBE-20261030`、plan1.0、research-pipeline-contract v1.0，computational/定位探针。本聊天执行，[事前Idea/spec](../proposals/tmtaff_training_physical_probe_20261030.md)、[L2S153](../../HandOver/L2S-20261030-153.md)。源自Mode C1.62/F247/E149关于原107未核T物理输出的缺口，不是新方法成绩。

- 冻结bb388d4、`association-code-run153`。只读原107末端TFM1000步checkpoint；没有新训练/normalizer/TAM/V/测试/CSV或checkpoint选择。TCM由原模型complete生成，state_hold只用可见观测/端点保持。
- 原127首T8，每场输入身份确定的首2个跨源正对，16对/27个节点出现、Q64/1024个声明对查询。原生cache重建与实际timed reader在选点、单位、query、normalized pack逐值一致的软件检查已通过；输入m/s，参考m/s，不把legacykn直接当m/s。
- 预测先保存，再读取相应训练参考；732个有效位置/航向点、729个速度点，其余无参考保留。按CAP128两源支持：双源301位置点、一源299、零源132；是本小样本正对域，不能替代全T/全V人口。
- 六臂为原TFM、交换左右、左自配对、state_hold绕过学习外推、同cap128普通Mean、canonical_all Mean；每臂相同目标/人口，未挑好例或删点。原107/113不改。
- 北京时间21:29:58–21:30:42，科学44.447878s、launcher49.156432s；RSS1070030848B、CUDA峰已分配51331584B（不含全部context/他人或硬件总占用）。GPU3 UUID1043f26a…兼容共卡、CPU0–3×4/1800s advisory未超；非在线延迟。
- 同一`mtad-3aecaf93769f.service`终态inactive/dead、success/ExecMainStatus0、Restart=no、Linger=yes，实际argv和退出完整保存。

完整脱敏原件：[153目录](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_temporal_unsupported_full_peers/153/)，含六臂全部支持层三属性/逐场潜特征变化、run/原launcher、原失败审计与新数值复核。坐标/身份/模型/大预测留服务器原run/private。

| 同732位置点，float64重新核算m | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| 原TFM、换序、自配对、state_hold（四者位置完全相同） | 36634.927 | 42787.658 | 85374.119 | 93025.975 |
| CAP128 Mean | 1887.023 | 3884.933 | 5122.240 | 10189.589 |
| Canonical Mean | 1887.336 | 3887.976 | 5099.462 | 10248.128 |

原TFM速度MAE2.969578m/s，CAP128 Mean23.363417m/s；航向MAE106.375908° vs75.798511°。速度局部好不改变位置崩溃/航向差的事实；没有四属性整体胜出。完整层内数量/原混合dtype及double数值在原件，不从速度挑冠军。T8没有独立样本或CI，不补造泛化区间。

## Squeeze：操作区分、数值复核与未知

SQUEEZE_REPORT `SQ-TMTAFF-T-PHYSICAL-PROBE-20261030-153`，输入上列Record。0更新，无收敛曲线，本轮不重复107训练轨迹。追加数值分析没有新forward/fit。

**DE：** 四路各1024个保存位置的unique rows均1，CAP128/Canonical Mean各881。换序/自配对/state_hold相对TFM两位置列最大差均0；速度/航向最大差分别不超过4.768372e−7m/s/1.549721e−6°。这是保存精度下的输入干预不敏感，不只是某个误差均值碰巧相同。

**DE：** 双源支持的301点TFM MAE约40.47km、CAP128 Mean1.559km；单源约31.35km vs1.289km，零域约39.87km vs3.989km。TCM与state_hold在所有场支持内latent MSE0、支持外非0（逐场约.0104–.0287），但TFM位置不变。因此“只由支持外TCM传播导致”不足以解释所见。不能把非物理latent MSE乘尺度直接叫米误差。

**AI/UN：** 后段表征、递归解码、投影/输出量化或训练目标的失配仍可成立；本轮没有保存全部中间decoder/FC活动或干预其参数，未定位哪个权重/层造成常值，也不证明作者原算法有缺陷。换序/自配对/state_hold可能离训练分布；即便如此，原训练正对本身的常值输出已是直接失效事实，不等全部训练人口都相同。

数值偏差与失败如实保留：首只读audit1e9743a在`all/TFM/position/mae`以exit1拒绝，没有完成结果文件；原153直接对float32输出/float64参考进行三角计算，而原unit-vector也未统一double，算术路径不等价。未放宽容差或覆盖原summary；原失败记录补保存，原调用缺程序计时，成本明确未知。

新合同/checkout0a23fc8追加一次`numeric-output-review-v2.json`：先按原混合dtype复现360项统计，再独立float64 haversine/unit-vector交叉核对，二者范围明确分开。0forward/fit，.291137s/RSS44781568B，caller原命令/退出也留存。原与double的所有属性误差最大差.100174（单位按属性），全位置MAE36634.938579→36634.926855m；不能以此数值差解释数万米失效。正对身份/跨源绑定与六臂数组有限/shape通过，不是独立重建TCM/TFM或物理真值认证。

本次不重训107或凭该失败宣同行净胜。后继应先在冻结T侧区分常值产生在哪段及训练目标/物理输出梯度，再明确一个修正的训练合同；不能把换normalizer/加步或残差输出直接当已证解决办法。PTFM/DeepAF另走全量训练比较，不继承这一失败排名。

回流Mode C1.64：F250/E153，限定更新M02物理监督/恢复边界；无新机制/论文/C。下一skill deep-read Mode C；需要的是区分证据，不是重复失败配方。
