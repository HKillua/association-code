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
