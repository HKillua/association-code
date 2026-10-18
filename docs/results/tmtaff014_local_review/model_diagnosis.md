# 014 融合量级异常：独立模型与优化诊断

2026-10-18。范围为 013/014 已回传的小型原件、实际代码与 CPU 数学核验。没有读取原 ZIP、T_test、服务器 checkpoint；没有模型前向、训练、GPU、代码修改或新实验。`SC`=论文直接陈述，`DE`=代码/产物/可复算推导，`AI`=有证据的解释，`UN`=尚未确认。数学界采用实数算术，不是 FP32 舍入的逐位形式化证明。

**结论：未发现把标准化预测错当物理值、漏反归一化、度/弧度或米/千米混淆的确定实现错误。发现了更实质的参数化与更新量不匹配：参考 TFM 用末层 LayerNorm 后的随机线性头直接输出物理经纬度，没有地理中心偏移。它必须通过优化从约零的输出尺度走到约 131° 经度。014 使用的 TFM 只有两次更新；当前代码及优化器甚至使未批准的“TFM100＋JOINT20”方案有可推导的输出范围限制，不能把该方案直接当作足够的完整参考训练。**

这既不是 TMTAFF 充分训练的效果，也不能仅写成“多训一点即可”。本诊断针对014执行时的raw模式（Git历史eb014bc，仅作来源记录）。随后用户明确授权“找到问题直接改代码”，本地按[输出尺度规格](../../specs/tmtaff-output-conditioning.md)实现可选train_position_affine；以下原始诊断与数学界仍只针对raw模式，历史结果不变。

## 1. 首先确定在评价哪份模型

**DE：** [014 summary](../../../HandOver/results/TMTAFF-TIME-EVAL-20261018-014/files/summary.json.gz) 的 `selection` 为 `stage=TAM_FROZEN, update=2, fusion_stage=TFM_FROZEN, fusion_update=2`。不是 JOINT2 权重。013 的 TAM2 macro-F1 为 0.37077052529632837，JOINT2 为 0.3628350200715334，没有超过原选中值。

[complete_training.py:493](../../../association/complete_training.py#L493) 在 TFM 入口恢复选中的关联模型；[586 行](../../../association/complete_training.py#L586) 只按关联 macro-F1 更新完整快照；[597 行](../../../association/complete_training.py#L597) 在 TFM 阶段末把已训练 fusion 加入选中快照。014 零优化、零重选阈值。故本次物理评价使用 **TCM2、TAM2、TFM2 的完整快照**，之后发生过的 JOINT2 不能算进这份 TFM 的训练量。TFM2 看到 128 次正对呈现、96 个独立正对、两个训练场景；这是所选模型的覆盖，不能用所有四阶段的总量替代。

## 2. 用 oracle 二轨组定位到学习融合路径

以下由完整逐场景 `all_predicted_groups` 按组大小复算，误差按实际有效交点加权，不平均场景 RMSE：

| 014 方法与分组大小 | 组数 | 交点数 | 位置 MAE | 组级 MAE 范围 |
| --- | ---: | ---: | ---: | ---: |
| oracle，单轨透传 | 19 | 826 | 1.61×10⁻⁹ m | 0～2.97×10⁻⁹ m |
| oracle，恰好两轨 | 42 | 2,008 | 13,497,878.7 m | 13,340,081.3～13,591,911.9 m |
| oracle，三轨 | 56 | 2,550 | 13,498,586.3 m | 13,320,510.7～13,625,896.1 m |
| oracle，四轨 | 95 | 4,620 | 13,513,650.6 m | 13,325,909.8～13,743,381.9 m |

**DE：** 358 个 oracle 多轨组均在千万米量级。两轨只调用一次 TFM，尚未把融合 memory 再喂给下一轮。因此，错误身份或多轨递归不是这个量级的必要原因。所有分组大小的完整统计在 [model_geometry_diagnostic.json](model_geometry_diagnostic.json)。

**DE：** 单轨走 [complete_evaluation.py:220](../../../association/complete_evaluation.py#L220) 的输入支持查询值透传（片段端点范围内可含插值，并不全是原始实测点）；多轨走 228 行 TFM。单轨正确而多轨失败，把问题收窄到学习融合路径及其权重/表示，不支持全局参考坐标或统一指标单位错位。其他同指标对照：9001 单源 MAE 0.348 m、9002 为 4,357 m、简单 oracle 融合 1,797 m。它们覆盖率不同，不能直接作为同覆盖总排名，但足以反驳统一把米误算为千米导致全部结果异常。

预测组全量 MAE 12.230 Mm 低于 oracle 的 12.901 Mm **不表示错误关联改善了融合**：预测组只覆盖 6,184 个有效交点，而 oracle 为 18,347；单轨占比、对应与漏目标也不同。两轨/multi-track 的条件结果才是这里的定位证据。

## 3. 物理单位链路是自洽的，但输出起点很差

**DE：代码链路。**

1. [timed_inputs.py:295](../../../association/timed_inputs.py#L295) 对输入真实观测做 train-only Z-score；TCM 输出为 latent，不是可直接逆标准化的四维预测。
2. [complete_training.py:295](../../../association/complete_training.py#L295) 从 `fusion_targets` 直接取物理 `states`，没有归一化目标。[fusion_training.py:174](../../../association/fusion_training.py#L174) 把 TFM 输出直接与这些目标计算损失。
3. [fusion.py:124](../../../association/fusion.py#L124) 线性输出四维；lat/lon/cou 保持 raw，vel 用 softplus。它的契约从一开始就是度、原生速度、度，不存在“网络输出的是 Z-score、评价忘了 inverse”的代码证据。
4. [complete_evaluation.py:200](../../../association/complete_evaluation.py#L200) 与 231 行分别用同一物理 reference 和 TFM physical states；[fusion_metrics.py:122](../../../association/fusion_metrics.py#L122) 仅在 Haversine 内将度转为弧度，并乘 6,371,000 m 地球半径。

CPU 合成核验：同一点误差为 0；赤道相差 1° 为 111,194.9266 m；原点到本次输入 normalizer 的均值位置 `(19.996895°,130.990043°)` 为 14,238,784.0 m，和异常的地理量级一致。最后一项只是尺度示例，**不是从 MAE 反推出真实预测在原点**。

**SC：** 原文 TMTAFF 第 13–14 页 TFM 段落及 Eq.18–21 明确讨论原始物理特征的数值尺度，以多任务损失权重处理；不能把“没有输出反归一化”直接定性为漏实现。全文定位：[tmtaff.txt](../../../../biye-ronghe/papers/mtad_sources/tmtaff.txt)，当前本地行 672–684。论文描述不等于当前离散时间/递归接口与作者代码完全一致；TFM 输出端与递归细节仍属披露过的实现选择。

## 4. 输出范围推导：解释为什么两步不可能完成地理定位

**DE：结构条件。** [fusion.py:68](../../../association/fusion.py#L68) 的 PyTorch decoder layer 默认 `norm_first=False`；最后一层以 `norm3` 结束，decoder 容器 `norm=None` 不会取消该层归一化。评价时 output dropout 关闭。线性头 Xavier uniform，bias=0；末层 LayerNorm 初始 gamma=1、beta=0。其参数不在前两个阶段的优化器中（[complete_training.py:276](../../../association/complete_training.py#L276)）。

设最后一次 LayerNorm 标准化后的向量为 `z`，维数 `d=128`。实数算术下 `sum(z)=0`、`||z||₂≤sqrt(d)`。某一 raw 输出坐标为

`y = b + wᵀβ + (w⊙γ)ᵀz`。

因而对**任何输入**，固定权重都有外包区间

`|y − (b+wᵀβ)| ≤ sqrt(d) ||w⊙γ − mean(w⊙γ)||₂`。

它是外包界，不保证区间每一点可达，也不是模型永远无法学习大经度。它同样适用于每次自回归解码、每次递归融合，不能靠换输入或多次递归绕过。

本次 CPU 只按生产配置/构造顺序（seed20，先 TimedAssociationModel、后 PairFusion）初始化了模块，**没有 forward 或训练**。本地 fresh 头的经度区间为 **±15.5793°**、纬度为 **±14.5886°**。这是本地随机初始化说明，不冒充服务器实际 checkpoint 权重界。无论随机种子如何，Xavier 的逐元素界 `a=sqrt(6/(128+4))` 又给出保守初值界 `|y|≤128a=27.2897°`。初始网络不能直接表达约 131° 经度。

### 不把 Adam 步长误当作固定 `lr`

当前 [_optimizer](../../../association/reference.py#L185) 为 Adam `(β1,β2)=(.85,.99), eps=1e-6`；本完整 trainer 调用不传 weight_decay，故为 0。TFM 和 JOINT 各自创建新 optimizer。对任意有限的实际梯度（包括整体裁剪后的梯度），对 Adam 加权一、二阶矩用 Cauchy 可得第 `t` 步单坐标更新量不超过 `lr_t C_t`：

`C_t = (1−β1)/(1−β1^t) × sqrt[(1−β2^t)/(1−β2) × Σ(k=0…t−1)(β1²/β2)^k]`。

正的 epsilon 只会缩小该上界。令所有有效阶段的累计单坐标漂移上界为 `Δ=Σlr_t C_t`；新阶段重新计 `t`，参数的漂移继续累计。由 `|w_i|≤a+Δ, |γ_i|≤1+Δ, |β_i|,|b|≤Δ`，得到更保守、与所有随机种子无关的评价坐标界：

`|y| ≤ Δ + d(a+Δ)(1+2Δ)`。

| 有条件的训练方案 | 累计 Δ 上界 | 任何输入下TFM网络生成 raw 经/纬度绝对值上界 |
| --- | ---: | ---: |
| fresh TFM2，lr .002 | .0040057763 | 28.0292° |
| 假设 fresh TFM100，lr≤.002 | .3479863257 | 122.1730° |
| 假设 TFM100＋新 Adam JOINT20，后者 lr≤.0002 | .3524326573 | 123.7865° |

**适用边界：** fresh 当前初始化/结构、现有 Adam 超参、这些阶段/更新数与学习率上限、前序 TCM/TAM 不更新 fusion、有限梯度、实数算术。不是恢复任意旧权重时的通用结论，不是 FP32 逐位保证，也不能据此反推更长训练要多少步才收敛。改学习率、初始化、输出映射、优化器或训练阶段后必须重新评估。该结论已由独立审阅者按 PyTorch post-norm 与 Adam 公式复核。

**含义：** 现有未批准的 TFM100＋JOINT20 配置，即使梯度方向理想，保守界仍到不了本次输入经度中心 130.9900°；不能把它解释为“只差采样覆盖/再等一次训练”的问题。若采用衰减学习率且不超过上述上限，界不会变宽。此处不是精确性能预测，而是对该短训练配置的必要表达范围检查。

## 5. 损失和历史支持什么，不支持什么

| 013 阶段/更新 | lat 分量，度尺度 | lon 分量，度尺度 | speed 分量 | heading 分量 | 裁剪前总梯度范数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TFM1 | 20.0396 | 131.2230 | 11.3863 | 1.0385 | 76,648.34 |
| TFM2 | 20.2171 | 128.6820 | 3.3168 | 1.2021 | 74.16 |
| JOINT1，未选中 | 15.4057 | 126.4175 | 2.1888 | .9995 | 493.996 |
| JOINT2，未选中 | 13.6090 | 125.0663 | 1.8498 | 1.0058 | 339.265 |

**DE：** [013 history](../../../HandOver/results/TMTAFF-TIME-COST-20261018-013/files/history.json) 的大经度分量与需要学习地理偏移一致。各行采样场景和有效点不同，不能把这四行称为固定验证集学习曲线。TCM 没有物理四态输出，前面两次 TCM 更新不能视为已经预热了物理 TFM 头。

[FusionLoss](../../../association/fusion.py#L161) 的位置项为 raw-degree SmoothL1(beta=.001)，远离真值时单点导数绝对值为 1，而非随 131° 误差增大；初始多任务权重再乘 .5。增大的物理误差不会给位置头提供同比增大的梯度。learned log-variance 可以调整损失权重，但不提供缺失的地理中心偏移；大 loss 会推动对应权重降低。这是优化语义，不证明其最终不能收敛。

**不可夸大：** TFM1 的巨大梯度范数说明发生过强裁剪，但不能把裁剪比直接等同 Adam 的有效参数更新缩小比。也不能从 heading MAE 87.6° 单独断言预测服从随机分布。末层/递归数值敏感性可能影响优化，但现有结果不足以把它定为唯一根因。

## 6. 递归反馈和候选公平性

**DE/AI：** 当前多轨推理把 FC fusion memory 当作下一对的左输入（[fusion.py:102](../../../association/fusion.py#L102)、[complete_evaluation.py:224](../../../association/complete_evaluation.py#L224)）；训练正对的两侧则都是 TCM 完成特征。这有表示分布变化，而且原文递归接口并不充分明确。它仍需作为近似参考限制披露，但 oracle 两轨已经失败，不能用它解释全部异常，更不能擅自换反馈方式后冒称同协议。

**DE：** 候选 [cooperative.py:264](../../../association/cooperative.py#L264) 使用 `state_projection(hidden) * train_std + train_mean`，并约束纬度、环绕经度/航向。其初始输出天然在训练地理中心附近；它不是“同样 raw 头只多了 utility”。所以候选相对当前参考的巨大改善可能主要来自输出参数化、时序 decoder、训练目标/量等差别。固定候选底座的四臂比较可检验 utility 的额外作用；跨底座总分差不能单独证明该机制。

## 7. 当前可采取的动作与证据缺口

- 014 如实记录为短训完整评价，不能给出完整 TMTAFF 训练效果结论。更正“TFM2＋JOINT2 的模型被评价”为“仅 TFM2 的 selected”。
- 不直接采纳原raw参考100/100/100/20模板。用户随后授权定位问题后直接改代码，因此按公开规格实现输出尺度适配；正式更新量与GPU预算仍未批准。候选已批准的测量不据此暗改。
- **UN：** 014 没保存物理预测数组，服务器权重未回传；故没有实际预测经纬度直方图、selected 的精确末层参数区间或同输入的未训练/已训练对照。后续若需要确认，最小证据是读取现有 selected 的 `decoder.layers[-1].norm3.{weight,bias}` 和 `state_projection.{weight,bias}`，计算固定权重界；无需新训练、原数据重扫或完整评价重跑。实际逐点原因仍需真实预测，而不是用 MAE 反演。
- 当前最支持的解释 **AI**：raw 地理坐标输出的冷启动/尺度门槛与过短优化共同导致融合尚未到达目标地理区域；关联错误和递归可能造成其他损害，但不是千万米误差的首要必要原因。不存在足够证据宣称 TMTAFF 方法本身不可行。

复算命令（CPU，约 1 秒；只写同目录 `model_geometry_diagnostic.json`）：

```bash
/Users/a1/Projects/association-code/.venv/bin/python docs/results/tmtaff014_local_review/model_geometry_diagnostic.py
```

脚本包含指标单位合成断言、013/014 原件统计、本地 fresh 全尺寸模块初始化和公式计算；**没有 optimizer.step、模型 forward、真实数据或测试集读取**。本次退出 0。
