# MTAD 在线系统基线重选

2026-11-14。按用户修订标准，在线主基线来自 2025–2026 年多传感器在线跟踪或融合，原任务有可核的竞争结果，能从授权作者代码或完整算法独立重建，且核心机制可在 MTAD 的因果输入下运行。文末给出深核后的选型。MTAD 尚无同协议的近期完整同行结果。 [首轮原文与代码审查](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_peer_baseline_sources_20261114.md) · [Mode C v76](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v76-持续在线系统基线资格)

## 既有比较能回答的问题

- C-ON24 的 24 场四臂实验测了位置保持与 CV-KF。局部和全局指派在原图上逐场相同，图分量只有星形或孤边，未暴露有备选的双侧竞争。去掉最大共享 A 组件，KF 的错漏配对秒反增 737。该实验的状态输出使用共同的位置保持。C-ON28 后续工作臂用另一套在线重配入口，报告继续标出具体实现。两者提供工程参照。 [C-ON24 结果](2026-11-12-online-continuous-factorial.md) · [C-ON28 入口](../../online_observation/nemot_replay.py)
- NEMOT 已执行时间感知运动预测适配。固定候选时，预测位置 MAE 由 549.829 m 降到 537.104 m，错漏配对秒由 91,737 增至 91,865。HyperSSM 有公开作者代码，MTAD 适配尚未执行。两者用于预测模块对照。 [NEMOT 结果](2026-11-12-online-nemot-motion.md) · [HyperSSM 资产状态](2026-11-12-online-fusion-transfer-review.md)
- C-ON56 从原系统已接回的 31 个双报窗开始，当批重解后停止。它验证了坐标干预如何改变指派。完整比较需要从所有首报开始，追踪各系统后续状态、拒配和确认。 [C-ON56 记录](2026-11-14-online-rejoin-global-assignment.md)

## 第一轮主系统候选

| 原顺序 | 候选及入选依据 | 进入 MTAD 主表前的验证 |
|---|---|---|
| 1 | **CDP-WGL，Signal Processing 2025。** 本地航迹状态进入密度峰聚类，维护加权标签图，并持续输出融合状态和全局 ID。论文有完整算法 1–3、历史标签消融及两组真实出租车路线上的模拟观测对照。相对 TC-OSPA²，两组轨迹 OSPA² 小幅下降，融合时间降至约四分之一。 [刊版原文](https://hoa-van.nguyen-au.com/files/hoa_j9_distr.pdf) | 依公式和伪代码独立实现。前缀只能产生当时可用的本地状态。查清作者式 (6) 与文字的密度排序口径、同密度、零／单状态、同源多片段冲突和历史边重复累计。先在作者图 3、图 5 可核条件及反例上验证分簇和标号，再在 MTAD 开发场验证单源、错配、持续 ID 和真实服务成本。 |
| 2 | **Draves 等的异步航迹级多假设系统，IEEE MFI 2026。** 以局部轨迹组合维护互斥假设，随新报文扩展和评分，子假设继承身份，按当前分支输出。原模拟评价以终局最优历史对单帧 GNN，且最优分支吸收不足八成局部轨迹。 [作者全文](https://arxiv.org/html/2610.20290v1) · [会议日程](https://www.mfi2026.org/program.html) | 按原文复建分区、扩展、评分、剪枝和输出。由 MTAD 的已到二维位置初始化单源状态并允许拒配，用冻结的当前发布规则评价。双视线启动、局部噪声及终局回写属于已知协议差异。先验证分支独立、重复报文、剪枝和 ID 冲突，再与同延迟当前发布比较。 |
| 待补全文 | **CFAG-GLM，IEEE Sensors Journal 2026。** 论文摘要直接针对遮挡后断轨重接，融合组与持续全局标号均有定义方向。 [IEEE 出版页](https://ieeexplore.ieee.org/abstract/document/11738847) | 补得全文、算法或作者资产后，再和 Draves 比较适配代价与协议。当前只有摘要证据。 |

TSA-cTFER 保留为同 MTAD 的作者协议参照。2024 原文有 20 秒顺序递送和两阶段确认，原代码、权重、精确名单及难配超时未核得。Wolf 的 2025 稿有可追溯 Python，适合当批跨源似然与空配的组件对照。MH-T2TA（2025）和 T2T（2026 在线、2027 刊期）分别提供窗口多假设和带空配学习关联，优先级低于上述持续系统。ICCV 2025 CoopTrack 有完整多代理跟踪框架和作者代码，原运行需要 MTAD 未提供的视觉特征及相对位姿。现有门宽、最近邻、KF＋GNN 和同刻屏蔽均留在工程或机制对照。 [TSA-cTFER](https://doi.org/10.3390/rs16183380) · [Wolf 作者代码](https://github.com/Fusion-Goettingen/TAES_2025_Wolf_T2TA) · [MH-T2TA](https://jzus.zju.edu.cn/article.php?doi=10.1631/FITEE.2400340) · [T2T](https://doi.org/10.1016/j.patcog.2026.114650) · [CoopTrack](https://github.com/zhongjiaru/CoopTrack)

## 共同回放与入表证据

1. 每臂收到同一 MTAD 原生前缀、来源／片段等同性和同刻批。未来派生速航、整段末时刻、关联真值与参考轨迹不进入运行时。可使用各自因果的候选、状态、空配和发布规则。逐个核验截断或改变未来后，已发布的前缀结果一致。各臂保留实际首版与后续更新，比较同一已到信息及同一延迟预算。具体预算按开发合同事前固定，1 秒只作为可比评分网格之一。 [数据字段](../data_contract.md#在线字段可用时间)
2. 用完整场景自首报起评分，报告错配对秒和漏配对秒、纯且唯一 ID、持续错链及保住原正确配对、稀疏首报、从未见真对侧的错接、正确确认覆盖及等待。未确认保留分母。单报、失鲜、双侧竞争及弱几何分层均报场与共享 A 组件。推理与更新耗时、积压和信息等待分别计。固定图、固定分数或固定运动头可作为归因对照；完整比较保留各法实际关联输出。融合误差另按相同参考支撑报覆盖和尾部，区分数据集参考和独立位置真值。 [共同评价定义](../data_contract.md#评价覆盖与信息使用)
3. 已用 28／48 场用于开发与历史比较。只读当前 Level1 manifest 与五份在线名单，合并所有角色和两份明确历史排除 ID 列，已知在线场 440（437 在 T_core）。它们触及 T_core 全部 84 个共享 A 组件，2,100 场中虽有 1,663 个 ID 不在这份历史并集，完全未见组件余量为零。原 3,000 场名单以外有 25 场／24 组件与该名单及已登记预览不交，仅是元数据线索，当前协议未使用。先核档案权限、既有暴露和足量性，再定新留场。训练、校准、选型及评分仍按共享 A 分离，T_test 继续封存。 [分割约定](../data_contract.md) · [48 场原留场说明](2026-11-12-online-interval-assignment.md)
4. 每个候选的入表证据包括来源版本、可独立复建的核心行为、需要改变的观测与时钟条件、端到端开发回放以及按同一协议冻结的输出。原文数值、独立复建和 MTAD 适配分别标明。实现统一在 `online_observation`，不调用离线实现。

## 双源首报条件复核与扩查

用户要求先评估再实现。本轮只读原 28 场冻结输入和论文／作者代码，未运行候选、训练或启用留场。原图的 102,404 批中有 1,932 批两源各一条新鲜轨，涉及 13／432 个首报。另有 80 个首报时刻只有一源新鲜，其中 29 个当时全图只有一条轨。这些数字刻画输入条件，按既有新鲜口径计算。 [逐来源和输入人口](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_peer_baseline_fit_sources_20261114.md)

| 原候选 | 适配性结论 | 影响结论的具体证据 |
|---|---|---|
| **CDP-WGL** | 保留历史标号和融合比较资格，降低整链首选优先级。 | 原文和式 (6) 的密度排序符号相反。两源各一条时两点的局部密度相等，算法没给合并或空配裁决。当前簇先融合，历史加权仅决定标签。一旦假关联累计成零权，原规则无过期和撤销，当前分簇后也可能复用同一标签。两源稀疏与互斥规则必须独立披露。 [原算法](https://arxiv.org/html/2401.00605v2#S3) |
| **Draves 2026** | 保留为异步多假设系统候选，完整资格取决于双源身份管理。 | 原根由两源组合启动，子分支继承父 ID。双源下没有第三源扩展，换本地片段会开新根。原实验用终局最佳组合对逐帧 GNN，组合覆盖不到八成。适配保留独立分支滤波、按冲突分区剪枝及互斥读出，单源、空配和接续另列。 [原算法和评估](https://arxiv.org/html/2610.20290v1#S3) |

| 近两年补查候选 | 复现依据 | 当前优先级及原因 |
|---|---|---|
| **Lindenmaier 等，IEEE Sensors Journal 2025** | 算法 1 和式 29–51 可复建，实测雷达／智能相机直接供局部轨，有全局状态反馈及唯一 ID。对 JIPDA 的 F1/MOTA 优势不足 0.5 个百分点，后者召回更高。未核到作者代码。 | **优先做完整系统资格核验。** 比 CDP 更直接以源局部轨和全局轨为接口。全生命周期需要本地存在概率和状态协方差，MTAD 无原生值。开发时先核历史状态与误差校准能否支持其关联，并独立说明存在近似。 [原文](https://doi.org/10.1109/JSEN.2025.3548899) |
| **ModTrack，ECCV 2026** | 作者代码含纯空间跟踪入口、GM-PHD/HMM 和雷达适配，当前树未见项目级软件许可。论文非真值辅助雷达 LSTQ 42.1，对 Radar Tracker 66.8；76.9 来自真值辅助输入。固定版本在整场结束后拼接 ID。 | **优先做公开代码和位置模式资格核验。** 共同评价需要逐时前向读出、因果误差校准，以及有依据的检测机会近似。保留使用已到成员报文评分的 Draves 一并核验。 [论文](https://arxiv.org/html/2603.15812v3#S4.SS7) · [作者评估源码](https://github.com/JackRobs25/ModTrack/blob/17afb547fa5f8341743595598fe269768aa28637/modtrack/core/evaluate_modtrack.py#L2115-L2130) |
| **Three-Level MIFT，Electronics 2025** | 海事 LiDAR/AIS，显式输出融合轨和两类单源轨。算法可核，关键代码需索取。仿真中 IDF1 89.80→90.82%，位置 RMSE 1.93→2.19 m。 | 海事单源覆盖及拒配对照候选。原关联系统用尺寸和 AIS 身份，先核去除这些信息后还保留哪些机制。 [原文](https://doi.org/10.3390/electronics14214344) |
| **MV3DT，2026 作者稿** | 算法 2 给出轨片的时移距离和逐时身份合并、纠错，工程有 DeepStream 参考应用。 | 位置轨片关联和生命周期备选。原系统依赖相机置信、重叠视场和共帧支撑，当前无严格同输入完整系统资格。 [原文](https://arxiv.org/html/2606.13127v2) · [工程入口](https://developer.nvidia.com/blog/build-a-multi-camera-3d-tracking-application-with-nvidia-deepstream-9-1-skills/) |
| **MH-T2TA，FITEE 2025，同数据家族学习对照** | 公开冲突簇枚举和训练算法。补充材料测试 20／40／60 s 窗，AF1 为 .8327／.8526／.8888；作者称 20 s 时一侧可只有 1 点。 | 在固定等待下比较学习的成组关联。单点时端点斜率的处理、逐时全局 ID 和接续策略仍需明确，论文窗末成绩保留原协议。 [原文](https://www.fitee.zjujournals.com/en/article/doi/10.1631/FITEE.2400340/) · [补充材料 §2.8](https://cdn.academax.com/videos/suppl-3a2ce598732f437bbd7d8a9956f55c41-FITEE.2400340suppl001.pdf) |
| **SeaCrossTrack 2026、HiLO 2025、Castelino 等 2026** | SeaCrossTrack 给原生／统一生命周期结果，作者仓是数据；HiLO 有 BSD 源码，但每样本重置对象集；Castelino 等给异步轨级状态融合公式，作者实验重心为单目标状态。 | 分别支撑生命周期归因、学习融合及状态融合比较。SeaCrossTrack 的 IDF1 优势在统一生命周期后从 7.47 pp 变为 3.63 pp，ID 切换仍增加。 [SeaCrossTrack](https://doi.org/10.3390/jmse14212097) · [HiLO](https://arxiv.org/html/2506.02554v1) · [Castelino 等](https://doi.org/10.3390/s26062003) |

优先核验 Lindenmaier 的局部轨输入与 ModTrack 的可运行空间后端，并保留 Draves 作为机制不同的系统候选。首轮资格用同一 MTAD 已到报文核单源首报、两源各一条、双侧竞争、失鲜与断轨续接。达到全场在线输出资格后，报各法自身合法生命周期下的系统结果，再用同生命周期的对照定位收益来源。用同一延迟预算报告正确确认覆盖，当前发布和事后修订分别评分。CDP-WGL 的主要价值是历史身份稳定性与当前错聚类之间的分离。 [Mode C v77](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v77-双源首报与基线适配复核)

## 多基线名单与标准流程

复核飞书《视频文案》。第 1 章先列“复现论文”，后列“制造意外实验”“深挖意外现象”“构建 baseline 骨架全景图”。在复现论文一节，原文建议尽量选近两年引用量最高的论文，以便获得源码、找到后续改进并开展组合实验。第 3 章补充，后继较少主要影响骨架改造记录，探索仍可进行。项目选多套基线来自用户明确要求。本轮形成候选名单，已经核了原任务效果、复建材料和 MTAD 输入条件，尚未完成同口径引用比较，也尚无共同 MTAD 整链结果。 [视频文案第 1 章](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#Wq7gdDT11otUQsx36z6co1qhnjb) · [第 3 章](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#QxYsdMheSoUZfGxPbYecO8Din7b) · [主仓 AGENTS](../../AGENTS.md#科研方法) · [完整流程](../research_workflow.md#2-必须遵守的研究顺序)

| 选入的基线候选 | 首轮精读和复建要判清的条件 |
|---|---|
| **Lindenmaier 2025 与 Draves 2026，并列的完整系统候选** | 分别比较全局轨反馈与跨批互斥假设。前者的算法可独立复建，实测优势有限，局部协方差和存在概率待核。直接后继 Bóka 2026 增加视场与出生模型，可追踪生命周期的改造证据。后者需从原文确定保留分支机制的双源单轨启动、空配和换接，区分当前发布与终局最优。 [Lindenmaier](https://doi.org/10.1109/JSEN.2025.3548899) · [Bóka](https://eprints.sztaki.hu/11134/) · [Draves](https://arxiv.org/html/2610.20290v1) |
| **ModTrack 2026，第三条完整系统候选，限空间模式** | 公开代码可追溯，软件许可待核。核对其几何跟踪能否在取消终局拼接、没有视觉外观与真实检测机会时保住核心机制。原雷达非真值输入低于论文雷达对照，现有证据不足以认定为 MTAD 强基线。 [ECCV 论文](https://link.springer.com/chapter/10.1007/978-3-032-37490-5_24) · [固定源码](https://github.com/JackRobs25/ModTrack/tree/17afb547fa5f8341743595598fe269768aa28637) |
| **MH-T2TA 2025、GMvA 2026，固定等待的学习关联候选** | MH-T2TA 有 MTAD 家族的 20／40／60 秒结果及公开算法。GMvA 为 IEEE T-ITS 海事图关联，有作者代码，原入口在窗内重采样并跳过单侧窗。两者共同协议需核单点、单侧、显式空配及窗结束时的可见信息；持续 ID 另核。 [MH-T2TA](https://www.fitee.zjujournals.com/en/article/doi/10.1631/FITEE.2400340/) · [GMvA](https://ira.lib.polyu.edu.hk/handle/10397/119909) · [GMvA 固定源码](https://github.com/LouisYxLu/GMvA/tree/a7207aa645f9a78e255248d299ccfb814e4bb54b) |
| **CDP-WGL 2025，历史标号与融合对照** | 对照原文中当前分簇、状态融合和历史标签三处行为，重点核同密度、同源互斥与错误历史标签是否可撤销。原比较对象 TC-OSPA² 有作者实现，可核源协议中的差异。 [原文](https://arxiv.org/html/2401.00605v2#S3) · [TC-OSPA² 作者实现](https://github.com/AdelaideAuto-IDLab/Distributed-limitedFoV-MOT) |

扩查保留 AAF-AALM 的空配后验补充、ICAS 2026 SV-PFA-AWTF 的低置信缓冲和统计成组校验。海事三源级联有实测线索，当前只取得出版社摘要。DetectorContext 有 FUSION 2026 海试与公开提交。它与已有候选 CD-PMBM／TM-PMBM 都值得核观测模型，当前 MTAD 报文无法区分未探测与未观测，真实扫描及完整检测集仍缺。 [扩查证据](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_peer_baseline_fit_sources_20261114.md#多基线选型的后继与独立对照) · [PMBM 输入与代码](2026-11-12-online-fusion-transfer-review.md)

1. **选定并复建多套基线。** 从候选原文读清机制、输入、原评价和后继，保留已有原协议证据及未知。独立复建各法，在原 28 个 MTAD 开发场按共同因果输入和延迟预算从首报跑到持续输出，建立各法可比较的实测起点。作者原文结果与 MTAD 适配结果分别报告。仅在 MTAD 验证。 [视频文案第 1 章](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#Wq7gdDT11otUQsx36z6co1qhnjb)
2. **做探索，保留和深挖实际现象。** 逐篇来源分析，结合基线运行和历史 MTAD 证据做 Mode C 的 P×F 矩阵。原文允许根据矩阵组织组合实验，也允许独立来源的发现性组合。项目按 Idea 写清范围和出口后执行，先 Record，再 Squeeze 并回填矩阵。对反常或失败现象反复走 1a–2b，从观察、归类、追问到可证伪因果链。随证据完善基线骨架全景和后继改造位置，按潜在收益和可解决性选研究方向。 [第 1–3 章](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#Wq7gdDT11otUQsx36z6co1qhnjb) · [飞书五技能](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#S2RFdTEteoRR0mxr52dcsM0SnKb)
3. **针对已定位问题验证解法。** 先查已有解法和生效条件，再在 Idea 定对照、指标、代价和回退线。按 3a–4b 替换部件、分离改动、逐个消融、核对因果预测；跨域时用 A／A′／A′＋B 拆开适配与新增机制的收益。全系统比较和固定组件归因依照上面的共同回放，结果按 Record→Squeeze→Mode C 回流。原 28 场仅供开发，留场须先核共享 A 组件独立性。 [标准流程](../research_workflow.md#2-必须遵守的研究顺序) · [飞书跨域迁移](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd#QifSdePVUomGQwxR0npcYUe8nCf)
