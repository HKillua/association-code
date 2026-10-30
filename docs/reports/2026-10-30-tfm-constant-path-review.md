# 154：冻结TFM内部路径 — Record→Squeeze

2026-10-30。科学和同一launcher均COMPLETED/exit0；已把“零投影/纯物理输出量化”与“正常memory下解码响应近乎常值”部分分开，尚未定位训练成因。

## Record

EXPERIMENT_RECORD `LEVEL1-TFM-CONSTANT-PATH-20261030-154-attempt-1`，source_idea_id `IDEA-TFM-CONSTANT-PATH-20261030`，plan1.0/research-pipeline-contract v1.0，computational/冻结定位探针，本聊天执行。[Idea/spec](../proposals/counterexample_driven_pilots_20261030.md#154常值产生在哪段)、[L2S154](../../HandOver/L2S-20261030-154.md)，承接Mode C1.65/F250/M02。

冻结a57b690/new checkout154，只读107最后TFM1000端点。原127首T2（0/3），每场首2个输入身份跨源正对，4对/6节点出现/Q64/256查询，204个有效参考位置；原153的输入/正对和CAP128 Mean固定控制复用。TCM→左右encoder→FC→递归decoder→projection→affine；原输出、fc2 memory置零、同latent/权重double投影三路先保存再评价。0训练/优化/V/test/CSV，不修改107/153/152或权重，hook finally移除。

北京时间22:56:37–22:56:57，程序19.683077s/launcher29.861463s、RSS1068773376B、CUDA峰已分配49682432B；GPU1 UUID6811d9c0…兼容共卡、CPU0–3×4/RSS8GiB/1800s advisory未超。原unit `mtad-e5d2741ba76a.service`以实际launch原件为准（见下），真实服务inactive/dead、success、ExecMainStatus0、Restart=no/Linger=yes。

全部脱敏[原件154](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_temporal_unsupported_full_peers/154/)含run/完整summary/原命令/状态/退出与真实service；原latent/FC/raw投影/完整预测及权重仍server private。零memory的中间完整数组未另存，summary有hook统计、保存了最终干预预测；不冒称已保存或独立复算该路全部中间层。无科学重试或计划参数更换。

| T2同204位置点，m | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| 原TFM | 58291.802 | 62211.113 | 92115.956 | 94290.793 |
| ZERO_MEMORY | 58917.706 | 62782.403 | 92640.832 | 94808.442 |
| DOUBLE_PROJECTION | 58291.039 | 62210.398 | 92115.195 | 94290.033 |
| CAP128 Mean | 1816.416 | 3052.920 | 8536.751 | 10874.462 |

没有CI/新seed/全T/V泛化，本表不与153的T8不同人口直接计算提升率。零memory是离训练分布的干预，不当新正式baseline。

## Squeeze

SQUEEZE_REPORT `SQ-TFM-CONSTANT-PATH-20261030-154`，输入上列Record；本轮0更新，不重新声称核了完整107优化轨迹。已知常值投影与hook清除的软件回归通过，不是MTAD成绩认证。

| 直接观察DE | 范围与条件 | 可以排除/仍未知 |
|---|---|---|
| 完成特征有变化，最大channel range .719/.962；FC2对应存储位置跨两样本差最大.00302/.00951 | 两场、各2样本；FC2 `[2,8192]`，其他层具体shape见原件 | 不是所有输入或FC2数组完全相同；不证明它保留了对目标有用的信息 |
| 正常memory下decoder最后latent跨样本/时间最大range均9.5367e−7，raw位置列range约4.47e−8–8.94e−8 | decoder `[2,64,128]`；不同统计轴已明示，不能将range比值当严格Jacobian增益 | 近常值在affine/物理输出之前已有，唯一层级/训练原因仍未知 |
| 最终位置投影两行L2 .34436/.41000；最后LayerNorm尺度abs .7398–.9242 | 原保存权重统计 | 不能解释为投影全零或最后norm权重全零；权重非零不保证输入响应 |
| fc2置零后decoder range .03639、raw最大变化.08366、位置最多改变939.896m | 相同冻结权重、同4正对，零memory跨两场也相同 | 通路没有完全断开；但位置MAE更差，不是修复成功，不能据此认定“memory过大”是根因 |
| double投影后位置range约1.36e−8–2.34e−8度；相对原输出最大变化.7704m | 只重放最后线性/affine运算，不重跑全模型double | 量化能抹去极小变化，但不能解释58km误差；全double模型/中间量化尚未核 |

机制假设AI/UN：递归解码的输入条件化、FC memory共同模式/尺度、attention选择或层间变异收缩均可成立；本轮未保存各decoder层/attention logits或梯度，不能把“解码器响应小”直接当已证优化根因。原TFM输出与153两位置列差为0，增强复用/重建绑定；不是作者核心算法失效的证明。

最低成本后继是冻结权重的decoder逐层与memory尺度分离，检查何处收缩、attention是否进入极端集中，而非立即重训107或以投影残差包装新方法。后继必须新合同/独立run，效果与原因分开。回Mode C1.66新增F251/E154，M02限定增强，无新论文/机制/C。下一skill deep-read Mode C。
