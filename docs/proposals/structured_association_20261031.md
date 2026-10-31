# C57：对象损失核查与训练侧结构监督

2026-10-31，IDEA_PACKAGE plan1.0。来源[Mode C1.77](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-177)、[独立AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/structured_association_sources_20261031.md)。P05/P14×F270/F271/M111，C57候选10/12；LA-MOTR仅边界对照，不是本方法成功先例。原168/170条件平均信号保留，不以旧方法为必须骨架。

领域/任务：统一物理输入的多源多片段关联→构组→状态输出；当前K2/离线。目标是coverage与跨源mPCA/PFA及整体物理尾部共同改善；参数/延迟是次要量。隐私坐标、身份、权重只留服务器，不访问测试；不改152活动合同。

## IDEA-OBJECT-FOREST-QUALIFICATION-20261031-171

**问题/假设**：170T丢13/救17目标，候选解释是逐对训练/构组未承担对象错并代价。边负类仍约束singleton，不能说BCE无法学单片段。先核原T32丢目标是否被异类anchor吸收、片段数/正边数量、直接森林梯度能否触及相应错误跨对象边；这是结构监督可行性检查，不是唯一根因证明。

输入仅170原T32保存partition/score/output、127原T motion/eval与主manifest；无V/test/teacher/训练/新物理预测。旧WS与CTXCAL所有目标（包括未丢者）按同规则统计，完整匿名逐目标/逐场及累计分布，核目标获失和有效query获失。训练目标已用过，非独立样本；不做事后显著因果认定。

机制资格：对每T场，用真实**T**类数k，仅训练诊断，最大k森林和完整T类内MST，同一上三角对称高斯扰动ε.1、B4、seed20。保存value gap、rank、错误选边、其关联的丢目标数；固定rank平移不变，忽略数值tie的梯度为两森林指示之差。若全部gap为0或所有丢目标都不与可纠正错误森林边相接，则不沿此具体配方训练；有非零且相关只说明可测入口，不保证下游改善。T/真实k不能进入推理接口。

spec：新增通用upper-edge完整图Kruskal/完整类约束及资格CLI；输入finite分数/完整唯一i<j边/k合法，否则明确失败；不改原WS/168/170/159。返回selected edge bool及校验rank；只输出匿名汇总，逐目标仅局部序号无原identity。软件验收用N4枚举森林最优、完整标签特例最优、有限差分梯度/平移不变、空rank与节点重排；CLI fixture验证全目标获失/同分布对照和拒V。原始run写真实阶段/退出/命令与成本。

一次171新run/冻结checkout/launcher attempt1，CPU16–19×4/RSS4GiB/无GPU，600s advisory/Restart=no/Linger=yes；不重跑170、不自动重试。完成先Record→Squeeze→Mode C，再决定是否具体发布172；这里未发布172或授权无界枚举。

竞争解释：现有特征不足/参考对应/后端状态不相容、loss失配不等唯一根因。损失的forest–WS部署失配、样本扰动方差、目标大小加权可能影响结果；本次不据V调系数或阈值，也不把组纯度当coverage。成熟解法适配有价值即可，不要求绝对首创。

## IDEA-OBJECT-FOREST-QUALIFICATION-REPAIR-20261031-172（plan1.1）

源SQ171/Mode C1.78：171在首场空参考row查表失败，0科学终点，不视方法失败；原件保留。新172只修中央人口映射：未知target且eligible query全false则不映射，无效参考输出仍保原组统计；未知target有任何eligible query明确报错。4项unittest涵盖修复/反例及原森林保证，已通过。其余T32、ε.1/B4/seed20/0训练/0V、输出与资格判据**逐项同171**，不是扩大或替换科学配方。

172新冻结checkout/run/launcher attempt1，CPU16–19×4/RSS4GiB/无GPU/600s advisory，一次故障保全，不自动重投；CLI显式run-id172区分记录。全部T资格成功后Record→Squeeze→Mode C，之后才具体发布新训练Idea；未发布173。

## IDEA-STRUCTURED-ASSOCIATION-20261031-173（plan2.0）

源SQ172→Mode C1.79，13丢T目标均有可测错误森林梯度，但314对象也触及，机制尚AI。假设：在相同输入/容量/完整场曝光/T工作点下，辅助簇关系监督可减少错并丢对象并保持跨源关联与物理平均；竞争解释为额外训练/采样改变、原特征不可辨、forest–WS目标失配和状态不相容。失败须按联合终点关闭当前配方，不以loss下降称成功。

**A/A′/A′+B**：A原168冻结CTX＋170同T规则工作点及原056 WS；A′从原CTX权重继续**全场平衡BCE**；A′+B同起点/信息/容量/曝光，仅多partial FY项。三个CTX头都6209参数，共同复用冻结056 5697、上下文归一化/邻边缓存，原159后端固定。A′两类BCE各取均值后平均，有单类则对该类均值；避免全场负类数量淹没。无新输入/源角色/一对一容量/后端GT拟合。全场曝光不同原168抽256 pair，所以A′是必要强控，不能只与冻结A比较。

训练仅原T128/775705唯一all-pair（读取168现有cache）；完整T身份仅建标签/类内树，T32用原127eval、余96用151Tindex，绑定node_ids。两头各**512步＝4轮128场**，seed20每轮相同permutation、同全部pairs，共1024真实GPU更新，每头3102820对呈现；Adam .001/clip1，optimizer fresh。均从168末CTX初始化，不改原件、不earlystop/看V选步。FY系数λ1、ε.1/B4上三角高斯/共同噪声，按max(n−k,1)归一，固定先验不扫λ；k=完整T类数仅训练，rank0结构项0/BCE仍训练。与原文B100、负距离嵌入/批标准化不同，明确适配而非作者复现；MC方差未保证，保存全部512场训练日志/损失/梯度/曝光/起终权重。

**部署完全不读k/identity/reference**：只上下文分数→原signed WS，仍自主对象数/多片段；forest–WS不一致为主要适配未知，理论不跨用。T-only工作点复用170原raw跨源mPFA预算、257分位＋旧θ/固定tie，对原056、冻结CTX、BCE、STRUCT全部同规则，先保存全网格/模型再开V。任一T无可行点仅T_NOT_QUALIFIED结束，不放宽/改λ/加步。所有推理API无GT标签。

原T32/V32、`WS/CTX/BCE/STRUCT × Mean/159NN`八臂，全保存raw score/分组/预测后评价；冻结WS/CTX分别绑定原170 WS/CTXCAL分数/分区/物理输出（容差沿原）。T159头LOCO、V full不变；前端T本身已训练，不能称前端LOCO。主要对比STRUCT−BCE（辅助目标）、BCE−CTX（继续训练/曝光）、STRUCT−CTX/WS（实际增量），不拼不同人口或不同候选冠军。

评价：raw/decoded all/同源/跨源完整PCA/PFA/TP/FP/FN/TN，全部target coverage/重复与新丢query/目标，完整及共同位置MAE/RMSE/P95/P99、逐场/组件/去最大/2000次seed20描述区间、实际新增>1km/最大增害/0单多支持与same-anchor及纯度；四属性同原输出核算，不称速度航向新学习。V反复使用/9组件且T共享身份，不是独立泛化。只有整体、覆盖、跨源双指标和尾部一起支持才晋级；混合取舍保条件信号并关闭当前配方自动扩大，不扫V阈值/加训练。

spec：新增173脚本复用168 cache/predict/evaluate/finish与170标定，通用训练函数两个头同场序，保存heads/normalizer/sampling；标签只训练分支，预测函数不接受GT，V标签在raw落盘之后加载。验收4项森林测试＋两步真实优化同曝光/不同loss、保存head往返/无标签推理、既有变组保存→独审→四属性fixture；CLI完整数据与终点校验。原有接口不改。

新173 checkout/run/launcher attempt1，GPU0 UUID共卡余量6GiB/CPU0–3×4/RSS8GiB/3600s advisory/Restart=no/Linger=yes；含152最多四自己单卡。实际完整V32结束后一次新audit173 checkout/launcher CPU16–19无GPU、保存位置/人口/raw-decoded独审＋四属性、以及T网格独立sorted-prefix核；原标签/坐标/权重不回传。一次科学故障保全，不自动重试/变方/更多seed/全V300；Record→Squeeze→Mode C后再决策，不自动扩大。
