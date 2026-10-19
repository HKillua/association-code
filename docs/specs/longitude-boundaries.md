# 经度周期边界：显式兼容修复

2026-10-19。用户要求先修复已知实现问题、不确定研究选择另行讨论。本页只处理±180°日期线的算术一致性；不改变时间窗口、训练样本、数据划分、身份标签、目标覆盖、速度单位、损失权重、指标或模型规模。未扫描原ZIP、读取T_test或启动GPU。017正式配置、活动冻结执行和历史产物保持原样。

## 可复现的问题

1. reference经度179°→−179°时，原普通线性插值在中点生成0°且valid=true；量测支撑插值在相同位置生成−180°。两者相差180°。
2. 原经度SmoothL1直接相减：预测−180°、目标+180°是同一位置，却得到359.9995损失；预测179.9°、目标−179.9°得到359.7995且梯度+1，最短经度差应为−0.2°。

这些是合成反例确认的边界缺陷。**尚无已审真实MTAD样本触发证据，不能将其写成018低分的已证实原因，也不据此停止、重训或改写017结果。** 当前全场景Q64与长时外推是否合适仍是另一个研究问题；本修复不解决或暗改它。

## 修复规则与接口

- `align_reference_states`、`build_fusion_targets`、`build_scene_fusion_targets`新增`longitude_policy='linear'|'shortest_arc'`，默认`linear`。新策略仅在相邻参考经度差的绝对值**大于180°**时沿最短弧插值并wrap到[-180,180)；普通区间保留原线性算术。恰180°保留原有符号线性方向，不新增无效mask；精确参考时刻直接复制原值，+180仍为+180。
- 原始reference读入、重复冲突、逐属性有效mask、最大gap及不外推真值规则不变。非法/不一致参数仍报错。
- `load_timed_dataset`的数据配置新增可选`reference_longitude_policy`，默认`linear`。数据准备报告保存实际策略；新策略逐场景写入`reference_alignment.longitude_policy`。默认linear不向旧逐场景契约新增字段，保证历史checkpoint的严格人口/装配对比继续工作。输入归一化和native/query适配不变。
- `FusionLoss`新增`longitude_loss_mode='linear'|'shortest_arc'`，默认`linear`。shortest_arc仅将经度差中绝对值>180°的项替换为周期差，再送入原位置损失；标准模式仍为SmoothL1 β=.001。纬度、速度、航向、学习权重、微批分母与regularizer完全保留。恰±180°的残差保留原符号。**这不是新的大地线距离loss，也不引入纬度余弦权重。**
- 候选与完整参考训练配置均接受可选`longitude_loss_mode`，缺失表示linear；训练记录/checkpoint配置保存实际值。完整参考checkpoint恢复按保存值重建loss，旧文件缺字段时仍linear。
- 完整参考fit接受可选reference策略；evaluate-only从保存的全部场景契约还原同一个策略后装配，并继续严格比较数据契约。不允许不同场景策略混合，不在评价时猜新策略或重选阈值。

## 启用范围与兼容

旧调用、旧正式JSON、旧checkpoint缺新字段均保持linear；接口没有自动升级历史实验。017的正式JSON未改。单独选择shortest_arc会改变跨日期线样本的监督或目标函数，须在报告中明确，不把修正前后分数当完全相同处理。

当前020参考与021—025候选正式配置均显式启用两个shortest_arc选项，见[执行安排](../proposals/repair_before_retraining/README.md)。合成原生ZIP检查覆盖参考装配和新loss；历史linear路径继续兼容。

## CPU验收

针对性用例覆盖：

- 跨日期线两方向、等价±180点、精确参考点、普通区间及恰180°方向；非经度属性、mask、gap和原输入不变。
- 经度边界零损失及梯度方向；普通区间float32/float64、两种位置公式的损失值与预测/目标梯度逐值等价；含NaN mask的微批累加等价。
- 单对及全身份目标装配都贯通新策略，缺参考身份仍保留；未知选项在开archive前拒绝。
- 候选v2合成CLI、四臂checkpoint和运行记录保存新loss/reference选项；完整参考合成ZIP四阶段fit、checkpoint载入亦贯通。
- 旧checkpoint删除两个新字段后仍能载入与evaluate-only；默认新装配与旧data_contract完全一致。新shortest checkpoint的评价确实复用日期线合成reference目标，不回落linear。

实际执行：

```bash
PYTHONPATH=tests python -m unittest test_longitude_boundaries test_cooperative_revision_cli test_cooperative_revision_training test_cooperative_cli test_fusion test_fusion_targets test_fusion_training test_timed_dataset test_complete_cli test_complete_evaluate_cli test_complete_training test_complete_snapshot -q
```

121项针对性CPU测试通过；随后全仓480项CPU测试通过（21.369秒），原生审计15项通过（0.041秒），均退出0。独立审查确认旧配置/契约兼容与新策略贯通，并独立复算日期线目标和双侧梯度，通过且无阻断问题。这里包含微型合成网络/合成ZIP测试，不是实际MTAD训练、CUDA效率或方法效果证据。
