# TSA-cTFER 离线表征与成对评分适配

本实现已完成011离线baseline首训，完整V分数已复算；本页记录当时适配，不是新的READY交接，也不是作者原代码或完整在线系统复现。来源：*Contrastive Transformer Network for Track Segment Association with Two-Stage Online Method*, Remote Sensing 2024, 16, 3380，[全文](https://doi.org/10.3390/rs16183380)，主要依据 §3.4–3.5、§5.4、§6。本地已核过全文，不以当前 TCM/TAM contrastive 分支替代它。最新比较边界与下一轮输入协议问题见[2026-10-18审计](experiment_reaudit_20261018.md)。

## 保留的核心

- 经纬度 z-score → 连续线性嵌入，加离散 timestamp lookup；learned padding → ReLU → Linear → 正弦位置编码。
- 最长128槽、4层、64维、4头、FFN256/GELU，pre-LN，dropout0.1，真正排除padding keys，masked mean。
- 共享encoder、ITSA/MSTSA各一个无bias的64→64线性头；同源/异源分别路由，余弦作为原始分数。
- B个正对产生2B条轨迹；每个anchor指定另一个正端，其余非self项进入InfoNCE分母，双向平均，温度0.07。
- ITSA裁young轨迹尾部；MSTSA按同一真实时间窗同步裁剪。Adam β=(0.9,0.999)、eps=1e-6。

## 明确的适配选择与局限

原文没有公开时间词表、初始LR、两任务混合比例、裁剪范围及假负例处理，本轮未获得作者源码。以下规则在看成绩前固定，不宣称来自作者：

| 项目 | 首轮实现 |
| --- | --- |
| 共同输入 | 复用007划分与既有scene64在支持内注册点；各轨迹的有效点按时间压到128槽，剩余padding。不新增采样、不丢尾部、不读取reference。相同原始时间范围，TSA只使用lon/lat/time，不用速度/航向。128是模型容量，实际有效点最多64；**这是共同输入适配，非作者原始最新128量测预处理**。 |
| 归一化 | 仅T_core有效注册点的全局经纬度总体均值/方差；std<1e-6置1，不用验证集拟合。 |
| 时间编码 | 实际observation_time减scene起点，以20秒向下取整，固定4096词表，越界夹边；记录clipped数量。位置编码是压缩后点序号。该选择保留场景内时间关系，但非已知作者编码。 |
| 训练正对 | T_core中AB同身份；ITSA同源、两段各>120秒、不重叠且gap<1200秒，old→young；MSTSA异源且交叠>120秒。条件来自原文任务适用范围。保留全量评价标签，**不把不适用正例改为负例**。 |
| 批采样 | 分任务batch，ITSA/MSTSA交替；每批从该任务可用AB身份不放回均匀选512个，再每身份均匀选一个正对。避免同身份成为假负例。原文未给此处理，身份均匀也不是声称作者的pair均匀。少于512个合格身份明确失败，不自动缩batch。 |
| 增强 | 保留比例Uniform[0.5,1]。ITSA只裁young尾部，至少2点；MSTSA交集内随机同一时间窗口，任一端少于2点则两端均不裁，记录fallback。保持原scene-relative timestamp。 |
| 有界首训 | seed20、FP32，1000次更新（两任务各500）、batch512正对=1024轨迹、峰值LR1e-4、weight decay0、grad clip1。线性warmup67步后半余弦衰减。不是原文150epochs/3次重复，不能据此称收敛复现。 |
| 验证选模 | 每100步完整V_select，原统一macro-F1精确阈值搜索；epoch/update间相同F1保留较早模型。同源/异源共享一个选定阈值。不用原在线阈值，不读取T_test。 |
| 覆盖范围 | 验证对所有原候选评分，包括原文在线适用范围外的片段；同时报告原范围的覆盖数，不能将子集结果混入全量排名。没有在线Hungarian、3/6次确认或延迟指派，不报告在线系统效果。 |

原文masked mean分母存在表述歧义；采用有效点数。无bias投影＋余弦下与除128仅差正比例缩放。attention mask采用文字要求的排除语义，不照公式中“0乘logits”实现。

当前配置实际474,624参数（其中时间词表262,144），原文joint模型表中为0.674M；词表/完整实现未公开，因此不宣称精确参数量复现。训练记录1000次optimizer调用，余弦终点最后一次LR=0，999次非零LR；summary同时报告这两个数，不把短训称为原150epochs。

这些差异使本轮只能回答“独立的论文核心离线适配在共同协议下表现如何”，不能回答是否达到作者原始在线分数。TSA用T_core，CDF1用T_core+T_head；与A0是相同训练池，与B/A_full比较必须披露额外训练数据和不同优化量。

汇总第一张表时，除了全候选主表，还须对**所有参比方法**在上述论文适用范围的同一个候选子集，用各自全量V_select已选阈值重算指标并报告覆盖率；不单独对TSA删难例或重选阈值。本入口保存完整分数缓存与范围覆盖，子集对比在各方法缓存到齐后统一计算。

## 入口与可观察验收

`tools/experiments/tsactfer_fit.py`接受archive、007 manifest、原protocol-config、独立tsa-config、显式device及全新output。沿用现有数据、指标、Run与缓存工具。两端不再用SHA或fingerprint判别执行资格；旧hash字段仅为历史元数据，缺失或变化不阻塞。实际配置、划分、角色/种子、模型与缓存内容继续检查，010的已完成缓存可直接复用。

| 行为 | 验证 |
| --- | --- |
| mask/双头/InfoNCE真实有效 | padding扰动不改有效输出；task head隔离；手算loss与有限梯度 |
| 输入、标签与时间窗明确 | 经纬度/时间与共享注册记录逐点对应；正对边界、身份唯一、同时间增强测试 |
| 能实际训练并选模 | 小型原生格式合成archive的双任务更新、完整验证、checkpoint重载和缓存复算 |
| 数据与失败边界 | 只加载T_core/V_select；旧输出拒绝；不足身份不降规模；超时/失败保留产物且不标完成 |

011已完成：TSA外层170秒、退出0，1000次更新；macro-F1 .520546、micro-F1 .340053、precision .218620。此前6小时单项/12小时总额度属于已关闭011，结余不转移、不重跑。完整结果及共同适用子集见[011分析](results/compare011_interim/README.md)。T_test继续封存，新训练须有新协议和预算；一次有界开发结果不证明作者150epochs/在线系统复现或收敛。
