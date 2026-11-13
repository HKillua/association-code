# C99 · 固定真对的编码差异收缩

## Record

`REC-TMTAFF-ENCODER-CONTRACTION-20261112`，research-pipeline-contract v1.0，completed。消费 Mode C2.36/C99、P04/P14×F345/F351/F357、M157、E316/E329/E320/E321。实测定位 TFM 目标差异的逐层变化，并以一次冻结边界干预检查局部条件差异可否传到物理输出。 [事前 Idea](../proposals/tmtaff_encoder_contraction_20261112.md) · [合同](../../HandOver/L2S-20261112-c99.md) · [终态](../../HandOver/S2L-20261112-c99.md)

复用原 T8 的 16 真对、1,024 个 pair×Q、共同 u0 和 C93 原图／截断历史梯度两份 u32。三个异目标场 384 个盲有向点、198 个共同有效、90 个共同有效且双支撑。五个同目标场分别为 640、444、0。共同物理位置误差样本为全域 732、双支撑 301。原始 Q、参考、掩码、支撑和三个普通控制按原 C91/C93/C96 绑定。三快照原输出、两次残差加法以及以原出口替换的对照重放均逐值核齐。 [输入核验](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_tmtaff_encoder_contraction/science/input_audit.json)

左右六层逐层保存 self-attention、FFN、残差加法和 LayerNorm，另保存 FC 拼接输入、FC1、激活、FC2 与逐时记忆。预定的 48 个同维边界在原图 u32 异目标盲层排序，唯一最小为右侧第六层 attention 残差 .428081，次小为同层 FFN 残差 .433315。三快照固定同一边界。反事实只在同快照同场两 pair 的真实出口中点两侧，传递按本地幅度缩放的原入口差值。中点误差最高 1.192e−7，其他网络照常前向。 [事前规则的实际选择](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_tmtaff_encoder_contraction/science/selection.json)

SSH MCP 一次托管，科学 55.859 秒、worker 58.426 秒，真实退出 0。0 优化、0 新原始档案、0 验证或封存测试。独立复算从保存 trace 和原/反事实状态重新计算 64,059 项，最大绝对差 2.212e−9。 [科学终态](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_tmtaff_encoder_contraction/science/run.json) · [复算](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_tmtaff_encoder_contraction/science/independent_audit.json)

| 快照 | 选中出口相对差 原→替换 | FC memory 相对差 原→替换 | 有效异目标平均分离 原→替换，m | 全域位置 MAE / P99 原→替换，km |
|---|---:|---:|---:|---:|
| u0 | .592879→.600148 | .553321→.549916 | 36,660.619→25,382.087 | 117.637 / 229.224 → 115.174 / 197.177 |
| 原图 u32 | .001094→.002555 | .00004128→.00005730 | 0→0 | 36.022 / 90.352 → 36.022 / 90.352 |
| 截断历史梯度 u32 | .001743→.003723 | .00010695→.00010244 | .025708→0 | 36.067 / 77.438 → 36.067 / 77.438 |
| CAP128 Mean | — | — | 48,118.956 | 1.887 / 10.190 |

原图 u32 的全域及双支撑 MAE/P99 均逐值保持。截断 u32 的全域、双支撑 MAE 分别增加 .003464 m 和 .007719 m，两层 P99 均保持。共同有效异目标点的标准化经纬响应 RMS，原图／截断 u32 为 1.785e−7／2.674e−7，物理平均响应 0／.012854 m。BOS 六个异目标盲有向点的物理响应均为 0，后段 192 点为 0／.012151 m。同目标盲控制两个 u32 的位置响应也为 0。u0 物理误差与分离方向相反，不能以其均值下降单独判断条件恢复。 [完整分层和控制](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_tmtaff_encoder_contraction/science/summary.json)

## Squeeze

`SQ-TMTAFF-ENCODER-CONTRACTION-20261112`，verified。原图 u32 的左右编码器层尾相对差由输入 .772/.862，经过首层降至 .338/.375，末层降至 .000695/.000458。截断 u32 由相同输入降到 .001005/.000786。两个 u32 各侧的 attention 与 FFN 残差在六层均降低平均相对差。LayerNorm 单步相对差比约 .946–1.000，离 1 的差小于残差步。 [层级原件](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_tmtaff_encoder_contraction/science/summary.json)

完成后按同一私有 trace 把相对差拆为绝对差与两边能量。两个 u32 的全部 attention 残差，绝对差比为 1.049–1.192，能量比为 1.698–2.681。FFN 残差绝对差比 1.134–1.284，能量比 1.582–2.789。选中层的 attention 分支与入口差向量余弦均值为 −.00715／−.01374。直接测得的相对衰减主要来自幅度增长更快，选中层的分支差向量未见一致反向对齐。LN2 随后把绝对差和能量同时缩到约 .35–.63 倍。重复的残差幅度增长和归一化对应跨层绝对差累积变小。该分解为事后描述性诊断。 [幅度和方向分解](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_tmtaff_encoder_contraction/science/decomposition.json)

FC1 的进一步相对衰减也伴随绝对差上升。原图 u32 从 FC 输入到 FC1，绝对差 .000620→.004055，能量 1.022→78.546，相对差 .000607→.0000516。截断 u32 对应 .000944→.007609、1.014→70.524、.000931→.0001079。FC 记忆的目标可分性仍需用明确读出测量，绝对差和相对差的变化不能独立证明信息可识别性。

局部出口确实恢复了本层入口量级的相对差，两个终点的下游记忆响应方向不同，最终物理输出均未恢复目标分离。该结果约束了单层末端恢复的作用，仍保留跨层累计、FC 幅度、decoder/投影低敏感和真实时间窗/配准的竞争解释。局部替换用现有快照自身状态，组合仍可能离开训练分布。观察限固定训练 16 对、原 seed20 和三快照。新的干预需先回到 Mode C，并独立检验信号保留能否转为物理定位收益。
