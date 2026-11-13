# 在线观测

本目录维护在线观测的输入、运行、状态、关联、融合和评价代码。当前工程控制仅依赖 Python 标准库。MTAD 原包、冻结名单和数据定义沿用项目约定。方法和下一步见[研究状态](../docs/research_status.md#下一步)，历史 Q64 计时见[原工具入口](../tools/online/README.md)。

## 工程回放

从仓库根目录运行：

```bash
python -m online_observation causal-replay --help
python -m online_observation causal-replay \
  --archive PART1 PART2 PART3 \
  --manifest configs/data/mtad-level1-scene3000.json --scene 692 \
  --start 0 --stop 7200 --stale-after-seconds 60 \
  --output outputs/online-692.jsonl
```

- ZIP 分卷按文件字节顺序提供。回放只打开冻结开发名单内指定场景的量测成员。也可通过 `--measurements` 提供同格式的开发量测 CSV。JSONL 含片段键和预测坐标，按私有产物保留。
- 回放从 `--start` 冷启动，按量测时刻安排理想到达，逐秒输出到含 `--stop` 的最后一轮。片段以 `(source, batch)` 标识。最新整行的航速和航向用于位置外推，超过配置的陈旧时长停止输出。这个按 CSV 整行交付的运动控制用于检查回放、生命周期和输出格式。航速与航向的实际可用时刻见[数据约定](../docs/data_contract.md#在线字段可用时间)。
- JSONL 记录查询时刻、量测时刻、数据年龄、生命周期事件和逐秒计时。服务时间含交付、更新、预测和状态序列化。预读和排序单计，文件写入与设备传输在计时范围之外。排队是固定每秒释放一轮的串行模型。按协议的 `ticks` 核对结果完整性。

```bash
python -m unittest discover -s online_observation/tests
```

## 严格输入与开发诊断

这些诊断从已到的位置、时间和片段等同性计算，量测速航既不参与预测，也不筛选位置。训练场的 A-B 标签和参考位置只进入评价。输入字段权限、真实回放结果和方法资格见[在线研究报告](../docs/reports/2026-11-11-online-field-and-method-review.md)。

|入口|用途|
|---|---|
|`python -m online_observation.field_audit --help`|核对字段与相邻区间、采样时距和批号等同性|
|`python -m online_observation.association_probe --help`|出生事件的候选门与排序诊断|
|`python -m online_observation.bias_probe --help`|事后核对共时真对应的相对位移|
|`python -m online_observation.registration_probe --help`|先无标签估计平移，再对晚到片段作共同候选比较|
|`python -m online_observation.rejection_probe --help`|固定候选与首选，按相同接收数比较首报拒配风险|
|`python -m online_observation.confirmation_probe --help`|同一后续报告时刻拆分过期、更新、续鲜与开放候选|
|`python -m online_observation.stale_history_probe --help`|固定首报核超龄候选可达、入门、排序和误伤|
|`python -m online_observation.peer_activity_probe --help`|固定首选比较局部报告活动和几何拒配|
|`python -m online_observation.held_gate_probe --help`|固定候选比较留场中的原始／校正距离和 1／4 km 门|
|`python -m online_observation.prefix_gate_probe --help`|同场比较前缀拟合分流和固定 1／4 km 门|
|`python -m online_observation.continuous_factorial --help`|严格位置输入下比较非等间隔 KF、局部及带空配全局分配，逐秒评分持续 ID|
|`python -m online_observation.recent_motion_support --help`|对固定 Level1 开发场核原生首报与二报的运动历史、候选和邻轨支持|
|`python -m online_observation.nemot_experiment --help`|组件隔离的 NEMOT 运动三臂，原生事件回放、固定候选与独立事后评分|
|`python -m online_observation.nemot_association_diagnostic --help`|固定模型和候选，重放关联间隔、空配和错漏链寿命诊断|
|`python -m online_observation.causal_recovery --help`|组件隔离的原几何、宽门、因果外锚与共同过去位移，评门外补配及确认时差|
|`python -m online_observation.expansion_confirmation --help`|固定 4–8 km 候选，分开核对双端续报、外见证冲突暂缓、相符提前和确认时差|
|`python -m online_observation.neighbor_witness_experiment --help`|稀疏首报的共时邻轨证据、同航竞争和历史见证写入资格|
|`python -m online_observation.interval_assignment_experiment --help`|固定原候选图和发布时刻，比较整组端点与区间评分，分别评等量换配和留空|
|`python -m online_observation.interval_assignment_squeeze --help`|从私有冻结回放分解长错段和不同改配动作的后验结果|
|`python -m online_observation.hypothesis_gap_diagnostic --help`|在原候选和发布上核新涉事报文、同刻屏蔽和自然无新报|
|`python -m online_observation.hypothesis_gap_later --help`|只在已冻结的分歧窗内核撤边后到达的报告|
|`python -m online_observation.relative_motion_gap --help`|在原候选和冻结分歧窗上分解位置反转、共时相对位移和连续新报|
|`python -m online_observation.shared_innovation_gap --help`|用冻结图核四端报告、失鲜与原窗终止，再以同刻屏蔽和等幅衰减对照同源邻轨创新|
|`python -m online_observation.async_report_gap --help`|在原图对照严格接回前及当批的实报、屏蔽和各端预测方差，核误拦与提前量|
|`python -m online_observation.cross_source_calibration_gap --help`|在原图分解跨源均值与中心离散，核严格过去的外部锚、正确接回与屏蔽|
|`python -m online_observation.single_anchor_transfer_gap --help`|从冻结的跨源残差核单锚的跨目标误差、真假候选排序和弃权|
|`python -m online_observation.disjoint_report_closure_gap --help`|从冻结的两批报告核四端点覆盖、异源互指及分别遮蔽早报和晚报|
|`python -m online_observation.reciprocal_report_downdate_gap --help`|从前态重建晚报对侧预测，隔离一条已进入预测的前报更新|
|`python -m online_observation.past_motion_shape_gap --help`|在原图和同一已到历史上比较净位移、两段运动变化及同刻屏蔽|
|`python -m online_observation.rejoin_batch_diagnostic --help`|固定原接回批，对照当前形状、同刻遮蔽和严格较早的缓存形状|
|`python -m online_observation.rejoin_cache_diagnostic --help`|固定原接回批，核缓存时距、严格批前配对持续和当批几何对照|
|`python -m online_observation.rejoin_geometry_diagnostic --help`|固定原接回图和时刻，比较原几何增量、候选间距、预测名义方差和逐边残差|
|`python -m online_observation.fusion_probe --help`|固定身份的逐秒位置控制及跨源权重比较|
