# 2025–2026 在线观测基线筛选

此前 24 个 T_core 场的四臂中，非等间隔 CV-KF 对位置保持少 37,865 个错漏配对秒，去掉最大共享 A 组件后反增 737。局部与全局分配逐场相同，因为候选子图没有双侧竞争。更早的 128 场首报对照仍有 255 个真边在门内的错配、218 个从未见真对侧的错配，后者有 109 个只有一个门内假候选。前两种错误所需的证据不同，且首报只有一个本地原始点。 [持续身份结果](2026-11-12-online-continuous-factorial.md) · [首报结果](2026-11-12-online-prefix-gate-qualification.md)

定向核对 12 篇 2025–2026 原文及可见作者资产。11 篇有正式出版记录，Fusion-Poly 的 IROS 接收有作者声明与会议索引转录。证据进入 Mode C v20 的 F-ON69–73、M-ON32–34 和 C-ON27–29。 [逐篇章节、原表、反例及公开资产](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_literature_recent_2025_2026_20261112.md) · [P×F 证据矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v20-近期学习基线和可获得的运动信息)

## 筛选结论

| 优先级 | 近期原文的机制与对照 | MTAD 的下一项可归因比较 |
|---|---|---|
| C-ON27 运动预测 | IEEE TSP 2026 NEMOT 把非 Markov 历史、邻居和学习量测更新分开。真值对齐的一步车辆位置 MSE，CV／个体历史 RNN／加入邻轨 .434／.299／.285。同一原检测输入仅学运动的开放跟踪 AMOTA 75.5→76.6。CVPR 2026 HyperSSM 固定检测和关联，MOT17-val KF／独立 SSM／邻轨 SSM HOTA 64.5／63.4／65.5，DanceTrack-val 47.0／53.7／59.7。AAAI 2026 PlugTrack 同时有 KF 胜出的轨迹段和混合预测有利的跨域场。 [NEMOT 表 III–V](https://arxiv.org/html/2506.18124v3#S7.SS2) · [HyperSSM 表 5](https://arxiv.org/html/2604.12665#S4.SS3) · [PlugTrack](https://ojs.aaai.org/index.php/AAAI/article/download/37491/41453) | 冻结位置/时间输入和候选，CV-KF 对时间感知的个体历史预测，再加同源邻轨上下文。NEMOT 的运动部分是首选机制对照，原文全系统依赖视觉量测和已估速度，公开实现和权重未核得。HyperSSM 有作者源码，跟踪器短历史可补重复框和预测框。MTAD 对照应显式区分原始报告和模型自生状态。 [HyperMOT 作者源码](https://github.com/SkyeSong38/HyperMOT) |
| C-ON28 身份待定 | NeurIPS 2025 DualMOT 支持弱更新不宜直接改写身份，IROS 2026 作者稿 Fusion-Poly 显示直接加异频观测也会使漏检和换手变差。AAAI 2026 SparseCoop 在真实 V2X-Seq 用投影和保留未配查询。FUSION 2026 的仿真中，GNN-PMB 自起始便较差，其余几种方法的主要差异出现在目标消失附近的假目标代价，未测持续 ID。 [DualMOT 表 4–5](https://proceedings.neurips.cc/paper_files/paper/2025/file/07054a34f24ac3ef64c5f2fdf571b8c0-Paper-Conference.pdf) · [Fusion-Poly 表 III](https://arxiv.org/html/2603.08199v2#S4) · [SparseCoop](https://ojs.aaai.org/index.php/AAAI/article/view/37952) · [V-PMB 表 I](https://arxiv.org/html/2605.07309#S5) | 先检验同门和年龄下，保留到下一份实际报文是否打开双侧竞争，是否新增可辨识证据。可解竞争足够时在同代价和预算下比较单解与有限多假设。1 Hz 保留首版，另记后续修订。 |
| C-ON29 概率与空配 | ICLR 2026 MOSES 支持不规则查询的联合概率，但同一个 MIMIC-IV 的边缘 CRPS 改善伴联合 Energy Score 变差。AAMAS 2025 MFPD 和 IEEE T-ITS 2026 DI-MTP 有真实 AIS 多模态预测，依赖多点历史、额外属性或航迹预处理。 [MOSES 表 2–4](https://proceedings.iclr.cc/paper_files/paper/2026/file/899c6a43b9976e1077522fe5a39cafa3-Paper-Conference.pdf) · [MFPD](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1408.pdf) · [DI-MTP](https://doi.org/10.1109/TITS.2025.3635392) | 保留同预测均值的高斯／混合似然对照，另估空配所需证据。报告校准与关联风险。缺真片段时只观察已到分母；原文的最优样本误差无空配风险解释。 |

## 对照的边界

第一项冻结为 C-ON27 的历史支持诊断，随后训练比较的顺序为 CV-KF 工程控制、个体时间感知 RNN、同样的 RNN 加同源邻轨。完整试验应另冻结 T_core 内组件分隔的训练、选型和新留场，所有臂用同一位置/时间、候选池、空配及确认规则。先固定候选看真边排序和预测误差，再接入同一持续系统看逐秒错漏配、ID、首次正确覆盖、确认等待、场景/组件和实际服务时间。五条实际报文是此次诊断口径，后续模型可以按掩码处理较短历史。样本量、训练预算和终点须等诊断回流后独立冻结。 [冻结诊断](../proposals/online_recent_motion_support_20261112.md)

候选边、联合约束和短等待各自需要可观察机会。带视觉外观、置信度或检测前端的 LA-MOTR、DualMOT、SparseCoop、Fusion-Poly 完整系统保留作者输入条件。连续概率滤波需额外核检测机会与存在状态。TSA-cTFER 保留为同数据比较路线，作者精确名单和资产缺口仍按原文记录。可比位置融合误差要另有独立参考。 [TSA-cTFER 作者协议](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_tsactfer_protocol_20261112.md)
