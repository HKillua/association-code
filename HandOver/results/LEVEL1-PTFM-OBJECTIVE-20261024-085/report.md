# 085：PTFM 既有缓存与日志资格封口

当前两项 PTFM 配方继续关闭。本次仅核现有输出的目标与物理误差，未重新训练、前向或构造原生双源子任务。

完整 V：300 场；原六臂物理人口、分组、对应和有效 mask 已重放。

| 方法/端点 | 唯一采用组数 | 有位置监督组数 | 标准化位置目标（组等权） | 位置 MAE m（点等权） | 位置有效系数（组均值） |
|---|---:|---:|---:|---:|---:|
| direct074C/initial | 8743 | 8742 | 1.5504151 | 16808.071 | 0.29707516 |
| direct074C/final | 8743 | 8742 | 0.12603388 | 1845.3843 | 0.35385602 |
| residual076B/initial | 8743 | 8742 | 0.029449127 | 721.17448 | 0.29707516 |
| residual076B/final | 8743 | 8742 | 0.038023005 | 1609.0601 | 0.35360665 |

上述组等权目标与点等权 MAE 分母不同；summary 同时给出组等权物理误差、点等权原目标和逐组符号对应表。
全部输出组诊断含重复组，按首代表身份取已有参考；它不是新增的唯一目标成绩。无真值组仍保留计数，原目标总项为零，不被当成零误差样本。

## 解释边界与出口

- 原位置目标下降而物理 MAE 上升，仅支持目标与指标的取舍线索；不能据此直接确定训练失败根因。
- 若二者都变差，也不能仅凭 V 输出判断优化失败或 T/V 分布差异。
- 固定 T 同批 step0/final 未保存在本轮允许的缓存中，结论保持 UNKNOWN；需要新冻结前向，085 不执行。
- 训练日志按 mode 与固定 100 步窗口汇总；批次不同，不能代替固定 T 端点比较。
- 任务有效系数为每组、每任务有效点上权重的几何均值。它不是 helper 权重或可信概率。
- 每批只保存 task loss 和 sigma 的均值，无法恢复历史每样本加权任务项。不得用均值乘积替代。
- V 缓存没有 LSTM 参数；V 目标不含 L2。训练日志 data loss 与 L2 单列，不能把两者之和同此 V 数据目标混算。
- 不自动修复、重训、挑终点或晋升该适配；发现具体可区分问题后再单独设计。

## 其他限制

- V reused, shared identities; group diagnostics are not independent training repeats.
- All-output loss maps each anchor identity to its existing reference, including duplicates; not formal unique-target system scoring.
- Unique selected-group loss is group-equal; point-equal physical metrics are separately reported.
- Effective task coefficients are geometric mean weights on each task mask, not helper reliability.
- Training logs contain batch means on changing samples; no exp(mean sigma)*mean loss reconstruction is valid.
- Objective/metric disagreement is descriptive, not proof of the cause of training failure.
- This neither constructs the original PTFM dual-source window task nor certifies author reproduction.
