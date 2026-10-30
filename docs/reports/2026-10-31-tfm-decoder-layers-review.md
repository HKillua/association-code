# 161：冻结decoder逐层与memory幅度 — Record→Squeeze

2026-10-31。科学及原launcher一次exit0；部分路径已定位，缩小memory没有恢复定位精度，训练优化唯一原因仍未知。

## Record

EXPERIMENT_RECORD `LEVEL1-TFM-DECODER-LAYERS-20261031-161-attempt-1`，source_idea_id `IDEA-TFM-DECODER-LAYERS-20261031`、plan1.0/research-pipeline-contract v1.0；计算/冻结操作定位，本聊天执行。[事前Idea161](../proposals/counterexample_factorial_20261031.md)、[L2S161](../../HandOver/L2S-20261031-161.md)。来源Mode C1.66/1.68的M02/F251/E154，非新增论文精读。

冻结194584f、新checkout161/run各一次；原107末步/normalizer、原127 T首2场各2正对，4对/6节点出现/Q64/256查询、共同204有效位置。只读身份选固定正对，0更新/V/test；不重训107。6层decoder保存64次自回归末查询的norm1/cross/norm2/norm3/layer，FC2 memory倍率1/0/.01/.1事先固定。首步真实cross输入与Q/K投影用float64重放logits/softmax，只定位首步注意力分布，不替换实际forward。正常倍率同原154物理输出最大差0m，两场均成立。

北京时间01:08:51.973–01:09:27.073；程序35.100244s/launcher44.734744s、RSS1109008384B、CUDA峰值49682432B。GPU0 UUIDc5a00c37-f181-90b3-f220-be6e7457f473兼容共卡、CPU16–19×4/RSS8GiB/1800s advisory未超；原unit `mtad-6f99e74bf3f9.service`真实inactive/dead/success/ExecMainStatus0/Restart=no/Linger=yes。完整[161原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_counterexample_factorial/161/)含全部路径汇总/四臂同点指标/实际请求/退出/服务；完整中间数组/原权重/坐标留private。

| memory倍率 | 同204位置MAE/m | RMSE/m | P95/m | P99/m |
|---|---:|---:|---:|---:|
| memory_1.0 | 58291.802 | 62211.113 | 92115.956 | 94290.793 |
| memory_0.0 | 58917.706 | 62782.403 | 92640.832 | 94808.442 |
| memory_0.01 | 58551.924 | 62452.326 | 92365.598 | 94540.298 |
| memory_0.1 | 58295.183 | 62214.210 | 92119.009 | 94293.843 |

## Squeeze：同查询跨输入的差异与时间变化分开

SQUEEZE_REPORT `SQ-TFM-DECODER-LAYERS-20261031-161`。DE：正常memory两场首步最后两层的attention最大概率均值约1，末层entropy约2.2e−24；重放logit范围约512。仅支持**这些首步头的分布极端集中**，不自动证每一步同样、错误attention或训练原因。

原summary的range混合了查询时间与输入样本变化；不能用零memory仍有时间变化称输入响应恢复。因此终态后明确追加一次只读`response-accounting.py`，同一自回归查询索引跨4个输入比较（不要求不同场景的绝对物理时间相同），原预测不变/0fit。正常倍率各层输出cross-input最大差约.001599/.006043/.006875/.002198/.0000826/.00000143：主要在最后两层进一步收缩；位置同query索引跨输入差为0。倍率0的所有层cross-input差均0，原summary中时间变化来自固定BOS递归，**不是**输入信息。

倍率.01末层cross-input差约.0000441，位置lat/lon差约1.91e−6/1.53e−5度；首步末层entropy约3.916而非极端集中。但MAE58551.924m仍差正常58291.802m；.1也没恢复定位。DE支持memory尺度对注意力分布/响应有操作作用，不能把“响应变大”当精度恢复，或将一项尺度操作当训练成因证明。

倍率同时改变cross-attention的key与value路径，未分离“softmax分布变化”和“值幅度变化”，故不能把输出差全部归于attention集中。若后继继续，应先分离K/V操作或训练梯度/监督证据，而非仅按当前倍率选模。

后验补项程序.235906s/RSS39845888B，真实argv/退出0与脚本原文另存，未在原计划中预定此分解，按计划偏差记录，不伪称事前指标或独立数组audit。没有独立重放全部decoder、保存梯度或新训练对照，未认证原论文实现/同行排名。层级范围混合自回归反馈，层5/6操作的唯一因果份额仍UN。

交接记录偏差：具体L2S161与Idea已事先发布，主Handover领取索引未同步，在终态时补登记；不倒填原索引、科学实际范围和attempt未变。

当前只能保留“末两层收缩、首步分布极端集中、尺度改变不足修复”的操作路径，不据此给107加步/调倍率部署。后继若需训练修正，须先核训练期梯度、memory归一化与监督路径，另定新合同；本次不自动开启。交Mode C1.69 F258/E161，M02部分增强，无新M/C，下一skill deep-read。
