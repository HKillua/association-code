# 在线路线代码入口

在线与[离线](../offline/README.md)持续并行。当前目标与待办只看[研究状态](../../docs/research_status.md)，执行和冻结范围看[交接](../../HandOver/Handover.md)。

## 入口与兼容

从仓库根目录查看命令：

```bash
python -m tools.online --help
python -m tools.online pipeline-timing --help
python -m tools.online causal-replay --help
```

[pipeline_timing.py](pipeline_timing.py) 是原 `tools/analysis/deployment_pipeline_timing.py` 的真实迁移，参数、场景、模型和输出格式保持不变。原脚本命令、`python -m tools.analysis.deployment_pipeline_timing` 与旧模块导入继续可用；新旧导入指向同一模块对象。

实际计时需原协议指定的服务器目录和冻结权重，使用 `--root`、`--output`、`--time-policy advisory`；`--completed-run` 仅接收原失败前四条有效记录。本次迁移没有新开计时任务，[原报告](../../docs/reports/2026-11-05-pipeline-online-timing.md)保留既有证据。

## 因果回放

从有序 ZIP 分卷读一个已冻结的 Level1 开发场景，也可用 `--measurements` 输入同格式的开发量测 CSV。输出文件须是新路径，JSONL 含原始片段键和预测坐标，按私有产物保留。

```bash
python -m tools.online causal-replay \
  --archive PART1 PART2 PART3 \
  --manifest configs/data/mtad-level1-scene3000.json --scene 692 \
  --start 0 --stop 7200 --stale-after-seconds 60 \
  --output outputs/online-692.jsonl
```

- `--start` 至 `--stop` 为闭区间整秒，每秒查询一次，区间起点冷启动。回放器预读并按量测时刻排程，采用理想到达时钟。状态只接收当前秒及此前已到且尚未交付的记录，每秒输出对应当前查询时刻的状态，含空秒。读 ZIP 时只打开指定场景的量测成员。
- [StreamState](../../association/online/stream.py) 按 `(source, batch)` 维护片段状态。一点即可输出，随后用最近量测的速度和航向推进位置，速度按 Level1 量测 kn 显式转为 m/s，航向输出最近量测值。每条输出同时给出量测时刻、到达时刻及年龄。超出显式陈旧时长的片段停止输出，再有足够新的量测时以原键出现。非法状态或同刻冲突从收到时起隔离该片段，已发布秒保留。回放保留已到合法行用于识别旧时间戳上的重复或冲突。
- JSONL 包含协议和逐秒结果。服务时间覆盖交付新行、更新、预测和状态序列化，输入预读和排序单独计时，文件写入与设备传输在计时范围之外。串行排队按固定每秒释放一轮，用实测服务时间推算等待和结果可用时间。核验完整性时对照协议中的 `ticks`。当前回放提供片段级运动控制，跨源身份由后续关联器提供。

## 原前缀计时

- `Pipeline.call` 从已接收的 CSV 字节开始，覆盖清理、运动拟合、关联、分组、融合恢复和 JSON 序列化。固定工程协议比较全窗口与相邻两个整秒前缀，每轮重算前缀并输出历史 Q64。
- [replay.py](../../association/online/replay.py) 保留原 `csv_rows`、`ReceivedMeasurements`、`preprocess`。原计时用收到的来源和片段边界生成匿名元数据，拒绝未来量测，要求至少两个有效片段。
- PTFM/DeepAF 组预测复用 [fusion_peer_inputs.py](../../association/models/fusion_peer_inputs.py)。`association/online` 可随包安装，`tools/online` 从源码仓运行。

## 评价边界

用户指定每秒更新一次，最大延迟尚未指定；更新频率、计算延迟、确认等待和稳定吞吐须分开报告。两个前缀的工程计时不证明持续 1 Hz 或在线精度。离线可共享解析、单位、模型与适用指标；未来信息权限、查询时刻、窗口和延迟/规模协议需分别定义。

TSA-cTFER 是在线关联候选，现有离线 encoder 适配不包含原在线决策；固定关联后的融合另设控制，不能将局部对照称为全系统胜负。对照资格与在线缺口见[路线审查](../../docs/reports/2026-11-10-offline-online-route-review.md)，后续工作统一维护在研究状态。
