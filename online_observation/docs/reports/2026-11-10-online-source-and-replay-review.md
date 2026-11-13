# 在线来源、观测权限与早期工程回放

2026-11-10 的历史证据摘录。更新频率和复用设想按当时原文留档，当前实验以[在线研究状态](../research_status.md)和在线独立实现为准。 [冻结审查全文](https://github.com/HKillua/association-code/blob/e10a176f2f1a81486e37464feae7ace6d8ffe949/docs/reports/2026-11-10-offline-online-route-review.md)

## 在线对照：已找到真实来源，但资格分层

|候选|可比较任务|输入和证据边界|
|---|---|---|
|**TSA-cTFER**，Remote Sensing 2024；直接在线关联候选|按新到数据更新轨迹/特征，处理轨迹出现/消失，困难对延后确认。[原刊§4–5](https://www.mdpi.com/2072-4292/16/18/3380)|同样使用MTAD，原实验20秒更新、最近128点、3/6确认；不是已证1Hz。实际网络用经纬/时间，MSTSA关联GT还要求重叠超过2分钟；没有状态融合误差。公开作者代码本轮未确认。原文的确认延迟包含等待观测，不能当计算耗时。[已存全文分析](/Users/a1/Projects/biye-ronghe/docs/analyses/2026-10-16_method_neighbors/direct/tsactfer_analysis.md)|
|**IFNet**，Information Fusion 2025；固定关联后的学习融合候选|递归融合局部估计，处理未知噪声相关。[原刊](https://www.sciencedirect.com/science/article/pii/S1566253524005281)|需各源后验均值/协方差/先验；原文同步固定源，不解决匿名多目标关联。具体输入条件复用[既有原文分析](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/recent_domain_sources_20261027.md)，本轮补核原刊来源。不能只取融合头耗时与完整系统比|
|**GCI-PMB**，IEEE TSP 2026；多目标关联＋融合条件候选|逐时融合PMB后验，处理匹配/未匹配假设；有[作者论文](https://arxiv.org/abs/2506.18397)和[MATLAB代码](https://github.com/Agarciafernandez/MTT)|MTAD没有现成PPP、存在概率和状态密度；原示例共同时间轴，不自动处理异步/轨迹碎片身份。需合法建立概率输入，不用真值为推理造协方差。[原文/代码核查](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/gcipmb_source_analysis_20261028.md)|
|**Target Tracking in Asynchronous Sensor Networks Under Temporal Misalignment**，IEEE TSP 2026；条件候选|量测/后验到达驱动，联合相对时间偏移估计和后验融合。[原刊](https://ieeexplore.ieee.org/document/11421001/)|需后验/运动模型；已知不同采样时刻不等于未知时钟偏移。只有目标任务确有这一问题时才需此专门对照，不为套论文追加问题。[已有原文与补充分析](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/recent_domain_sources_20261027.md)|

直接在线关联、固定关联融合、多目标后验融合应分别核资格，再组合完整系统比较。上述“在线/递归”只说明处理机制，不证明相同机器上可稳定1Hz。原DeepAF也可作为点迹滤波机制参照，但需要先处理输入层级差异。

## 现有在线差距及公平测量对象

**DE—已完成工程样本。** [11月5日整链计时](https://github.com/HKillua/association-code/blob/e10a176f2f1a81486e37464feae7ace6d8ffe949/docs/reports/2026-11-05-pipeline-online-timing.md)只有3场、相邻两个prefix、每条件2次；虽然不读截止后量测，但输出仍是历史Q64，未强制估计当前截止时刻。没有在线精度、长期排队、稳定吞吐或P95延迟结果。

87片段prefix下，PTFM为0.895–1.004s，我们HARD为1.974–2.301s、STRUCT为2.234–9.931s。这组既有实测不支持泛称“我们更新更快”。DeepAF当前逐组64步全量重算的耗时不能代表原论文的增量实现。

**分析判断：** 在线路线的公平问题应是“给定到达流与资源，算法能在允许时限内准确输出多少目标的状态与身份”。至少分开记录：

1. 收到新数据→结果可发送的计算/排队延迟及P50/P95/P99、超时比例；1Hz持续输出是否积压，不能只重复发布旧结果。
2. 输出状态对应的物理时刻与最新观测时间；没有新观测时的当前状态预测与有新观测时的更新分报。
3. 首次见到轨迹→确认跨源关联的等待，含最低历史长度及延迟决策；与单次运算秒数分开。
4. 精度/误关联/漏目标/重复/轨迹身份持续性；同一截止和可见信息评价。离线未来信息结果可作参照，不冒称等权限在线对手。
5. 并发目标/片段数、源数K、到达记录数每秒、历史长度四个负载维度；片段多不等于船多，源数多也不等于记录吞吐大。

1Hz输出本身不会创造每秒的新传感器观测；若使用参考插值形成评价时刻，须声明参考构造和支持范围。初始积累、网络、乱序与时钟偏差是否包含必须写清。采用1秒截止可以作为后续明确的工程协议选择，但当前用户只指定了更新频率。
