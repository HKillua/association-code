# 165–167：无真值分组修正、集合条件风险与冻结梯度

2026-10-31，IDEA_PACKAGE / research-pipeline-contract v1.0，plan1.0。本聊天依用户新授权执行；每项独立checkout/run/launcher一次。152活动合同/107权重/159基准冻结。来源Mode C1.72执行前（继承1.71），不把oracle净化变成运行输入，不退源编号优先。研究范围为Level1原T32/V32开发pilot及冻结T2定位；T_test封存。

## 来源、观察与候选取舍

F260/E163已区分固定输入换参考与固定参考净化，证明可比人口与对象条件的关系，不证明官方参考错或可部署净化收益。C53沿2/2/1/2/1/1＝9，SR-C1沿1/2/1/1/1/2＝8。原MeanTraj完整AP及作者p6 Eq17–19提供匹配/数量/状态的集合目标先例；当前signed surrogate不等该Fréchet目标/原算法。114操作和完整负例已知：WS merge-only可容许反证、未作split/move；更高J不保证物理更好。F259/E162表明当前目标/难例加权没稳定超159，不能同信息无依据重复换loss。

代码新增可定位观察DE：`source_risk.py::build_source_features`的前7维是自身质量，末3维是与其他源状态/运动的差；159预测为逐源h(x)，未直接读其他源的质量向量。**AI假设**：同一自身质量的评价可能取决于其他可用来源的质量，这不总能由独立潜势h(x_i)的差表示；不声称所有44T/16V纯组错误均因此产生，也不声称增加物理独立信息。集合均值/共享残差是成熟结构，当前不主张首创。新增价值只有在合法输入下减少实测错误并保完整风险才成立。真实K≥3/新域、完整相同海上传感器迁移案例仍缺；不新增论文精读。

## 165：IDEA-TRUTH-FREE-PARTITION-REFINEMENT-20261031

模式5，C53/P05/P14/F260/M69/E163，partial。问题：固定056完整成对证据，WS仅合并的搜索边界是否留下可用反证？假设AI：允许成员移至另组或独立成组，若提高同一signed within-group J，可能改善对象关系及实际状态；竞争解释是原分数失准、更高J不对应物理风险、过拆/首代表变化引起coverage和重复损害。

**运行只读物理/score/node索引，无identity/reference/label。** T原125 `T-features`的scores＋127原pairs；V原056 final-scores的node_ids/pairs/scores，仅取这三类，节点与127绑定。原WS必须逐关系绑定保存membership。候选REF从WS出发，对全部node×现存组及新singleton求精确ΔJ：sum到新组−sum到原组其他成员；每次取最大正增益（>1e−10），tie按原节点/组最小输入索引，不依赖来源名字；最多4N次move，另记是否达到局部最优/预算与ties，不自动加迭代。保留每次move/J轨迹；不是全局优化保证。CL用既有114确定规则作保守强控，不选新阈值。

原T32/V32六臂 `WS/CL/REF × MEAN/NN`：每种分区真实重建source pack/普通Mean、同冻结159 T组件留出或full头；网络不训练，0teacher/rawCSV/test。WS_NN输出绑定原159，CPU重算选择差异另记；分组变化后的特征/状态必须重算，不能拼旧组坐标。每场所有预测/partition存盘后才读评价身份/reference。仍沿官方anchor_identity_first，只用于评价，不据目标换anchor。全部节点恰好进一组、单源/零支持也输出。

主要：原全target人口、各自及两臂共同位置MAE/RMSE/P95/P99，覆盖获失/重复/漏轨迹，raw不变/decoded F1/PCA/PFA及组件/去最大，完整四属性/支持内外/代表改变与实际新增>1km。比较REF_NN−WS_NN、REF_MEAN−WS_MEAN，以及CL两强控；同分区NN−Mean检验是否保平均收益，不能将分组与风险头收益混归一个机制。全64逐场/组件与2000次seed20组件描述区间；V重复开发非确认。新groups变动使错误query分母改变，选源错误只分层描述，不将不再D2算救回。可保留局部候选但平均/覆盖/尾部不通过不晋级；失败不重试/换recipe。

CPU0–3×4/RSS8GiB、GPU0兼容余量6GiB（仅评分，可与已知兼容任务共卡）、1800s advisory，0更新。科学成功后一次新audit165 checkout核保存membership合法/原score J单调及所有输出的主位置/人口/尾部/组件/主要CI；不认证truth/作者复现；同一限定补项launcher另顺序运行一次saved_attribute_accounting，完整四属性/实际argv/退出/成本单列。私有坐标/成员/分数留server。

## 166：IDEA-SET-CONDITIONED-RISK-RESIDUAL-20261031

模式1，SR-C1/P11/F256/F259/F260/M108/E159/E162/E163，partial。问题：保持159平均收益时，联合来源质量上下文能否帮助纯组反转，而非单纯再训练原潜势？假设与竞争解释见上节；不可只因额外训练有效就归因集合条件。

固定155T128/28组件/原166058监督源标签、原T32的12组件留出/full共13fold。复用133/155 feature/label缓存与159冻结旧10维网络，不新teacher/CSV/分组/标签。两个新20→32→32→1、1761参数、末层零初始化残差头：OWN输入 `[自身z,自身z]`；CONTEXT输入 `[自身z,其他可用源z的均值]`。z沿对应159 outerfold T-only mean/std与clip±8，无新normalizer拟合；无其他源则context0，不含source名/slot/identity/绝对坐标，均值随可用源置换等变，真实仍K2。以冻结159预测为基底学习同一log误差残差；原159参数不更新。

两个新头fresh seed20同初始化/同batch4096源标签索引、同原query权加权MSE/Adam.001/wd1e−4/clip1，各128更新；26fits/3328更新/13631488源呈现，无早停/追加/恢复/V选步数。冻结159为A，OWN为额外容量/曝光控制A′，CONTEXT为条件结构B；比较CONTEXT−OWN才有条件增量归因，不能靠赢冻结弱曝光控制下结论。完整模型在V前冻结，训练全T128不筛pure/rare，不使用V或held组件标签定策略。

原T32/V32六臂：旧T32Qhard、T128二次、冻结159网、OWN残差、CONTEXT残差、源等权。同WS/状态/0与1支持域/速度航向保持；预测存盘后评价。主要平均/高害选择excess/实际新增损害、纯组与混组、稀有/常见救回新增分报；重点T纯44/V纯16原错误同query救回与新增，但不筛纯组作为训练/总体主成绩。完整人口/属性/尾部/组件/去最大/主要差CI同165。若只有纯层改善却整体/尾部明显退步，当前配方不扩大；若平均有益但少数反转未解，保条件候选不称健康已辨；两个新头都不稳定超冻结网则不加步数/seed或V挑配比。

GPU1兼容余量6GiB、CPU8–11×4/RSS8GiB/1800s advisory。一次科学成功后规定一次新audit166 checkout核完整保存输出/主要指标与不变域，及一次原数组方向/四属性记账；不独立重训新头或认证物理健康。私有权重/坐标/identity/大预测留server。

## 167：IDEA-TFM-FROZEN-GRADIENT-FLOW-20261031

M02/F261/E164，模式5、diagnostic。K-only改变集中却没恢复精度，原训练的有效监督/梯度仍UN。固定107末步/原T2四跨源正对、全部四属性有效mask；读取原normalizer、fusion objective及log_variance/clip配置。0optimizer/训练更新/V/test，不改变checkpoint。

每场完成TCM eval后detach，固定两输入latent；TFM分别eval和train(dropout seed20，新诊断RNG，非原训练RNG复放)，调用**原FusionLoss**与原物理mps reference，不新增归一化/目标。各一次forward，分别对total和原加权position部分求梯度，记录每个参数的finite/None/zero/norm、encoder/FC/各decoder层/BOS/projection、left/right/memory/states，以及原uncertainty权/损失分量/总梯度和原clip下比例。可见有梯度不证明原训练充分/可学，近零也不直接证明唯一死层；首层/全链与loss尺度分开解释。所有参数前后逐tensorexact检查未更新，普通eval输出绑定164正常，train模式不称同原训练随机批次。

GPU0、CPU16–19×4/RSS8GiB/1800s advisory；等待165释放槽位后FIFO执行，最多四个自己单卡任务，其他进程不动。原数组私有，公共完整参数/模块梯度统计/命令/退出/成本回传。非有限/OOM/错误保全、无自动重试/改batch，后继训练仍须另据证据设计。

## 行为验收

| 验收 | 核验 |
|---|---|
| 分组不读truth且完整合法 | 纯score函数签名/同score置换关系、每node一次、分裂及迁移反例、J单调/预算；真实链预测先落盘再开评价 |
| 条件结构而非额外训练混杂 | OWN/CONTEXT同参数/批次/初始零残差、可用K≥3置换/缺源检查；不筛pure训练、原159基准绑定 |
| 全population与风险 | 真实保存→共同评价→独审链，小fixture仅测软件，真实64场才是科学 |
| 冻结梯度定位 | 原loss/两模式/无optimizer，所有参数前后exact、eval基线绑定；软件梯度finite/模式恢复 |

软件核验完成后才发布L2S165/166/167 READY，再提交新冻结目录一次。全部真实结果和偏差严格Record→Squeeze→Mode C后定下一步，不自动扩300/新seed/加步或改152。


执行前核验记录：5项算子/条件置换/真实梯度测试通过；真实159→166→165保存/评价/独审/属性链通过，另以合法但覆盖改变的fixture分组核lost-target，不作真实CL效果。扩展fixture初版插入位置使expression变量未定义，已在发布READY前修正并通过；无服务器科学启动/失败重试。167实际状态数组沿164原run登记的161正常预测绑定，另与164正常公共指标对照，不假造164未保存的物理数组；启用autograd可改变attention数值kernel，差异记录而非倒改原数组。所有规则在实际执行前确定。
