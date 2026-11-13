# 180：共享可变候选选择器的物理/高害代价监督

2026-10-31，IDEA-COST-ROUTING-180，plan1.0。来源Mode C1.89/C61、P13/F284/F285/M115/E187/E188、ICML2025完整来源AP。Pattern4跨域成熟机制适配；理论与loss差异已核，不宣当前log目标为根因或新方法已有收益。179 T中所有104/75新害均有较好候选，纯单19/20，age/margin不是已证门限。

固定178特征/候选/分组/原前端与阈值、原175 H/S及178 O强控；**只在单来源支持query改变位置选择**，零/多支持位置及全部速航保持178 O原输出exact，coverage/raw-decoded不变。0前端/teacher forward，无固定源身份/源优先/真值组净化，不跨组/不删输出。

训练原T128的178缓存，合法首对应参考标签、source-balanced节点权重继承；仅valid单支持query且当前组≥2片段（有实际选择）参加三个新头训练，单候选无选择的query仍完整评价。两个前端视图各.5，不当独立样本。每fold从全部T128排除同component；原前端见T标签，不称全链LOCO。T32十二leave＋full十三组，全部从对应178末步相同残差网权重出发，**ridge91系数、12维normalizer固定**，新训练不接触V。1505参数共享12→32→32→1，任意候选数/ragged mask，同一评分器用于两头及全部候选，不设置固定J输出层。

每fold三个头：C继续178 log-error MSE（source-balanced bag均值）；L按ν1成本加权交叉熵；T同L但增加高损害代价。对候选j，从T已保存y反变换出e_j米；先减当前候选最小误差，d_j=(e_j−min e)/1000，L成本c_j=d_j，T成本c_j=d_j+max(d_j−1,0)。1km来自既有损害定义，不沿V选择；球面距离自然有界，无另加20km截断或样本裁剪。oracle仅建立T标签，不在推理使用。它是相对候选oracle的regret成本，不等实际相对178的增害；实际新害必须另核。

取r_j=−(原ridge评分＋新残差)，τ_j=Σ_(i≠j)c_i，loss=−Σ τ_j logsoftmax(r)_j/[N(N−1)]。这个正的按候选数因子是ragged query尺度适配，保持条件最优排序但不自动继承原受限假设类/IID/固定J定理。只用ν1，不用已发现问题的ν>2分支。三个头共享相同query抽样/初始化/曝光；C是新增训练与抽样改变的匹配控，L−C区分目标，T−L区分高害hinge。原178 O仍主强控；不把C未超O或T未超L藏掉。

每fold固定256更新，每步128个有放回group-query、同seed20样本索引给三头，全候选均参与；Adam.001/wd1e−4/clip1，原mean/std与ridge不重拟合。13×3＝39神经fit/**9984真实GPU更新**，0新analytic fit，末步固定，不early stop或换seed。保全部抽样索引、训练人口、每步曲线与模型。各fold权重不共享，所有模型T训练完才V；无预算后续追加。

容量边界：每个新策略单头1505参数，C/L/T匹配；完整部署为保留零支持178 O，需要保留旧178头并增加一个单支持条件头，风险残差头合计3010、相对178新增1505，ridge/normalizer可共享。不是全流水线零增容或整系统仅1505参数；相对O还含条件头/曝光变化，只有L−C/T−L的对照直接隔离目标。当前保存原输出缓存仅用于固定对照，不将缓存读出耗时冒充实际部署延迟。

原T32/V32，HARD/HO/HC/HL/HT、STRUCT/SO/SC/SL/ST各Mean+NN＝20臂；Mean是对应原普通Mean副本。NN新头输出仅单支持动作，保原完整两个强控，不拼不同前端覆盖/精度。primary为新三头各对O、L−C、T−L及T−C，固定2000次seed20 T12/V9组件描述CI、去最大/各场组件；完整/共同MAE/RMSE/P95/P99、全部属性、覆盖/重复/raw-decoded all/ITSA/MSTSA、0/1/多域、纯混/anchor、相对O与原base新增>1km害和最大害一起报。复用V为开发探索，不独立新域确认；真实K3仅软件接口测试。

若T/L不超C/O，当前cost配方无增量、不自动加步/seed/扩大人口；若平均收益被损坏或尾部代价大，只保条件信号，不按V拼门限；若整体与尾部改进，冻结后另立扩范围合同。非有限/OOM/绑定错误一次失败保全，不改batch/损失/配方重试。新180 checkout/run/launcher attempt1，CPU0–3×4/GPU0 UUID兼容共卡≥4GiB余量、RSS8GiB/3600s advisory；最多四自己单卡含152。成功后独立audit180 checkout/launcher CPU16–19×4无GPU/1800s advisory，一次保存输出/全部属性/39fit及索引/参数绑定/单支持动作域/尾部核算。原178/152/107/132不动。

验收READY前：可变N/mask/换序及成本梯度方向；MSE与L/T共同采样/初始化/冻结底座；真实小fit/权重往返无标签推理；原178动作与保存→独立评价链。模拟只验软件。Record→Squeeze→Mode C后再决定，不因相似委托已有先例关闭有新增价值的适配。
