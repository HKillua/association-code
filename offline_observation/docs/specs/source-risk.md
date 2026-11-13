# 133共享源误差得分：实现规格

对应[事前Idea1.0](../proposals/source_risk_20261030.md)。仅新模块和新runner，旧预测算子/权重/冻结run不修改。科学终点与解释以Idea为准。

## API与数值行为

新纯计算模块`association/evaluation/source_risk.py`：

- `build_source_features(*, states, support, nearest_seconds, sources, native_count, span_seconds, fit_rmse_m, ols_velocity_mps, groups)`：有限states[N,Q,4]、bool support[N,Q]、nearest非负、三项质量数组[N]、速度[N,2]、source非空字符串[N]和合法全节点partition。来源只按等价类归并，按组内首次出现组织源行，不按名称排序。返回pack：`features[S,Q,10]`、`source_states[S,Q,4]`、`available[S,Q]`、`group_offsets[G+1]`、`node_to_source_row[N]`、`supported_counts[S,Q]`、`source_group_indices[S]`。S为所有组内来源行数，非全场物理来源数。保留每源原成员数供支持比例。输入不修改；任意Q>0、任意有限K，每组源数可不同；N=0允许空输出。
- 源位置、航向的圆周均值与现有恢复器规则相容，源内及源间圆周零合量统一回退固定组anchor对应角度并计数（即使anchor当前不支持）；本源无support时使用该源首成员作为只为有限存储的填充值，但available=false，不能参与权重/监督/其他源上下文。其余可用源的等权角度若圆周零合量，回退固定组首anchor相应角度；保留回退计数。特征顺序、尺度及定义逐项按Idea，所有无available行features置0。
- `fit_score_models(x, y, sample_weight)`：x[L,10]、y[L]、正权[L]均有限；按权加权mean/std；返回LINEAR/QUADRATIC/SHUFFLED三个字典，含head、mean、std、beta、training_label_count、training_weight_sum、no_labels_fallback、置乱索引。λ=1，正则不罚截距，目标按sum(weight)归一化。空L返回零head，预测无信息。`predict_score(model, x)`输出[L]，不得接来源/标签。
- `source_weights(scores, available, group_offsets, *, kind)`：scores[S,Q]有限；kind为equal/soft/hard。每组每query仅available源参与，equal均分，soft稳定softmax(-score)，hard精确最小值平局均分；零可用源全0。允许重新排列源行并相应排列available；输出等变。不把输出称校准概率。
- `recover_source_policy(*, states, support, groups, pack, scores, kind, mean_states)`：将源贡献均分给本源当前支持成员，再调用现有recover_group_states；无support输出与Mean一致，最后所有新输出[...,2:]覆盖为mean_states[...,2:]。返回`group_states[G,Q,4]`、`node_weights[N,Q]`、`source_weights[S,Q]`。预测入口无reference/identity。
- `source_labels(pack, groups, identities, target_ids, truth, valid)`：单独监督API，调用者在合法pack完成后才读取T评价输入。按Idea首anchor/首组绑定且至少两源available；返回`y[S,Q]`（未赋标签NaN）、`labeled[S,Q]`、`sample_weight[S,Q]`（未赋标签0）、`errors_m[S,Q]`、计数audit。每query标签权总1。不可据标签修改pack/预测/分组。

## Runner与证据

`tools/analysis/source_risk_pilot.py`独立新输出目录；绑定127 completed cache、125 completed T/V名单/组、114 V WS及manifest。逐场读取物理allowlist，预测之后才读独立评价文件。旧5臂仅读125保存artifact，新5臂全部预测后统一评价并核人口/valid/分组相同。

保存T feature/label/source结构、39组解析拟合参数及OOF，V各臂完整artifact和贡献数组到private；公网JSON只留scene/component和聚合，不带原source名、身份、位置或模型参数。完整45配对、9组件、去最大组件、主比较组件bootstrap及来源贡献动态摘要。绑定旧Mean预测和所有旧导入artifact；开销记录真实拟合次数、耗时、峰值和各阶段进度。无新teacher/神经训练/关联阈值选择。

## 有意义的验收

| 要求 | 检验 |
|---|---|
| 来源身份不控制贡献、K可变 | K=1/2/3/4及不同组源数；源重命名/源行排列与同物理结果对照 |
| 非固定两动作 | 手算3源soft权、hard平局、缺源归一化、无source fallback；常量分数与源等权相同 |
| 正确空间/时间语义 | 日期线/航向圆周、单源完整复制、相同源多片段内部等权、方向有定义时不可用源不污染上下文；无定义时显式anchor回退及计数 |
| 无真值进入预测 | 显式接口、feature冻结后改标签只改变监督不改变预测；无输入修改 |
| 训练权与T隔离 | 加权小例独立解、空训练回退、折外样本不进normalizer/拟合、置乱复现 |
| 位置单因素 | 所有新臂速度/航向与Mean相等，固定人口/有效性不变 |
| 真实输出可信 | 完成后独立从保存数组复算主位置/人口和关键T标签/参数；不以自身summary作唯一证据 |

必要小例只检查软件性质，不算MTAD或K>=3真实科学成绩。参数/输入非法时抛ValueError，数值非有限抛错误并保全；不自动降配重跑。
