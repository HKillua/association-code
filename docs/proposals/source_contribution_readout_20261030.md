# 140：共享平方风险与集合贡献读出的分离验证

2026-10-30，设计及行为规格 **1.0，IMPLEMENTED_LOCAL_VERIFIED**。来源和Mode C交接、root事前设计审查、本地实现及窄测试已完成；**尚未发布READY、没有真实运行结果，限定独立审计仍待完成**。当前目标/执行许可仅见研究状态和HandOver，本文不复制实时队列。按项目discover-innovation及spec-driven-coding先冻结可否证合同，再实现。

## 1. Idea交接与选题

- `package_type=IDEA_PACKAGE`，`package_version=research-pipeline-contract v1.0`；`idea_id=IDEA-SOURCE-CONTRIBUTION-READOUT-20261030`，候选实验140。
- 来源矩阵：[MC-SELECTION-READOUT-20261030-v1.0](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md)，消费**SR-C2**、SR-P2/P3、SR-F1/F3/F5/F6/F7/F8、SR-M2/M4、E1/E2/E4/E5/E6/E7/E8。全局上游为Mode C1.57；不增加论文/实证计数。
- 研究对象：统一经纬度/速度/航向、时间戳与来源归属下的多源位置融合；源数可变的算法接口、真实MTAD当前K2、固定组/候选/物理恢复/人口。
- 主模式：Pattern 2，已有排序到真实混合损害的现象驱动；跨域迁移作为辅助，不能以硬化控制本身冒充完整创新。
- `evidence_status=partial`。已经验证的是133/134/136/138固定开发材料；机制、可学习性、几何读出净增量及跨传感器泛化均未知。

**具体问题：** 已有评分能区分部分端点，但按分数直接soft或倒数加权未必得到有益的物理混合。能否先合法学习平方位置风险，然后在**同一个评分器**下利用当前源间几何选择贡献，减少真实输出损害？这与139“同hard下把log监督改为米制监督”不同；140明确新增一个平方风险评分器A′，其总变化不能都归给集合读出。

### 已有证据与迁移边界

| 来源 | 已核事实（DE） | 对140的作用及不能支持的结论 |
|---|---|---|
| [133](../reports/2026-10-30-source-risk-review.md) | 同Q分数V32 soft MAE1128.755m、hard547.721m；Q训练的是log误差 | 分数与读出有可分离影响；不能将`expm1(q)^2`宣称条件平方风险 |
| [134](../reports/2026-10-30-mixture-opportunity-review.md) | T32最佳端点以外的4097格空间均值39.600m、中位0.346m | 允许探索条件混合，但不是新方法可达到的保证；本次不重新网格或拿oracle做标签 |
| [138](../reports/2026-10-30-observation-conflict-review.md) | C2997点中LOO端点credit96.263%，实际混合仍差；D8056点排序本身常错 | 同分数hard是必要强控；不能只报C、借Q支持侧决定权重或把所有点统一锐化 |
| [Chroma完整AP§2](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_transfer_sources_20261030.md)及原文p5/p9/p15–16 | 使用共同历史真实目标，直接比较成员加入后的集合WQL；简单性能权重较差，单选也有优于greedy的反例 | 借鉴集合目标而非单源分数即贡献；140不复刻其历史窗口、专家训练或greedy，不将自身残差当共同真值 |
| Mode C的CoCo/MeanTraj/ESCI来源 | 任务目标、几何对象、有效矩条件分别决定可解释范围 | 建立监督A′有来源依据；不声称预测风险为Cov、保守界或Bayes最优。下述恒等式是AI独立推导，非这些论文的新定理 |

**未选的最小变化：** 单纯LOO-hard或固定温度/稀疏化只能检验动作幅度，修不了排序错误；原135完整嵌套温度校准保留但本次不恢复。当前选择有物理量纲的平方风险A′与集合读出，代价是13次合法T监督拟合，符合“尽量少依赖拟合”而非“禁止拟合”的用户偏好。

## 2. 固定输入、权限与终点

1. 原127 `native-motion`、125的T组及V旧控制、114 V WS、133 `training-{sid}.npz`/`oof-{sid}.npz`/V十臂缓存；原manifest **T_core前32场/12组件及既定V_select32场/9组件**。输入路径在正式READY逐项绑定原run，不能以新生成缓存默默替换。133/125/127/114均须COMPLETED/exit0、列表/父路径/计数一致。
2. 只读已保存运动、评价和133合法标签；**不读native-events/ZIP/CSV，不做teacher前向、神经更新、时序重建、重新分组、平移配准、网格oracle或T_test**。上游模型曾用T、T/V共享身份及V已开发的边界沿原133，不称全系统独立泛化。
3. T标签只进入明确训练折；每外层组件的normalizer、ridge系数全部由其余组件拟合。所有外层T预测只用本轮对应LOCO模型；全T模型只用于V。没有第二层校准或T/V挑参，因而不复用含外层组件训练的base分数去拟合读出。
4. 源名只构造等价类，不当特征/排序优先；reference身份只建立训练标签及预测后评价。不得读138的C/D、reference最佳角色、Q支持侧来构造任何新权重。所有reference分层只在预测冻结后解释。
5. 新臂仍仅改变位置；速度/航向逐元素保持原WS_MEAN，0可用源位置回退原Mean；单源贡献1。保持全部组、首代表、重复、缺目标、无支持和同一四状态输出。离线输入仍有未来信息，不能称在线或端到端联合训练。
6. 一次新attempt固定完成13拟合、T32六臂、V32十三臂及约定评价，随后Record→Squeeze→Mode C回流。失败保全、没有自动重试/替方/加seed/扩268。无阈值搜索，不按V结果改loss、floor、solver或人口。

## 3. A′：合法的共享弦长平方风险评分器

采用原133的10维source-query特征、权重`a=1/K_available`、labeled mask、first-anchor且去重复训练标签。直接读原T的`errors_m`，不是从旧预测Q逆变换。球半径固定沿原评价 **R=6371000m**；真实径向误差e转换为

`y = [2 R sin(e/(2R)) / 1000]^2`，单位km²。

这等于原source endpoint和同一reference在3D球面嵌入后的弦长平方。使用km是预固定数值单位，不从数据选尺度。原有效e应位于[0,πR]；标签finite/范围/mask错误即失败，不能删样本。

每折仅拟合**一个66系数二次ridge**（1截距+10一次+55二次）：训练权先归一化和为1；weighted mean/std仅本折训练样本，std<1e-6置1；标准化特征沿133 clip[-8,8]；非截距λ=1，截距不惩罚。解线性系统而非显式求逆，不调用会同时拟合三头的`fit_score_models`。13次=12组件留一+全T；保存每折训练组件/场、标签数、权重和、mean/std/beta及无标签回退标记。真实规定人口须有合法标签；软件空标签接口可返回0头并记录，不把回退称训练成功。

ridge优化的是风险预测与chord²标签之差的平方，不是直接优化最终融合坐标的MSE；它会显著强调大风险标签，不能混淆两个损失。raw预测可以负：保存原值，三个新读出**共同用`r=max(raw,0)`**，报告负值数/占比/幅度及clip引起的平局。不能把clip后非负称为校准。此有限10维加权投影不等于给定全部决策几何的真实条件二阶矩；极端标签可支配平方回归，常量偏差/错误组和少数角色也可能仍不可辨。这些为竞争解释，不靠训练loss下降确认风险正确。

## 4. 同risk下的三个贡献规则与数学边界

在当前组/query的可用源上，用原133 source endpoint经纬度作

`x_s = (R/1000) [cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)]`，单位km；`D_st=||x_s−x_t||²`，单位km²。

这是合法当前预测几何，不读取reference。D用嵌入差向量逐项平方求和，避免大半径norm²相减的消减误差。源内位置仍按既有支持节点构造。这里只将endpoint嵌入作为权重证据；**最终位置仍把源质量均分给本源支持原节点，再调用原恢复器**，不能先将圆周均值归一化后加权冒充原算法。

| 新臂 | 固定行为 | 归因 |
|---|---|---|
| `WS_CHORD_RISK_HARD` | 取r最小源；exact平局源间均分 | A′单源风险强控；与旧Qhard差异包含新目标和clip，不是集合读出增量 |
| `WS_CHORD_RISK_INVERSE` | `w_s∝1/max(r_s,1e-6)`，1e-6km²即1m²；归一化 | 同r的透明普通风险权重；floor为工程常数，报告命中数，不宣精度校准 |
| `WS_CHORD_RISK_GEOMETRY` | 在simplex上最小化`F(w)=r·w−0.5 wᵀD w` | 同r下引入集合几何的候选B，系数1由恒等式固定，不再拟合/选温度 |

### 恒等式、求解与实际恢复之间的边界

对固定欧氏候选x_s、同一目标y及simplex权w，有

`||Σw_s x_s−y||² = Σw_s ||x_s−y||² − 0.5 Σ_st w_s w_t ||x_s−x_t||²`。

该式不要求源误差独立，也不要求无偏。只有候选/权重在相同条件信息下固定且r是对应真实条件平方风险时，F才等于条件期望风险。本实验r仅有限估计；**F可能为负，应原样保存并报告，不夹零掩盖不相容**。理论对象还是未归一化的3D加权向量，既不等于实际原节点纬度/圆周经度恢复，也不直接优化径向MAE。即使F数值求解最优，实际物理输出也可能变差。

一般K：对和为0的方向v，`vᵀDv=−2||Σv_s x_s||²≤0`，故F在simplex上凸。求解器支持实际任意K，而不是只写两源公式：

- K0：所有源质量0，物理层原Mean回退；K1：质量1；K2且d=D01>0：第一源质量`clip(0.5+(r1−r0)/(2d),0,1)`，包含两端点。d=0时退同r hard，exact风险tie均分。
- K≥3：确定性Frank–Wolfe，从源等权起，线性子问题对梯度最小的exact平局**均分**；沿方向作解析[0,1]精确线搜索。最多256步，不因效果加步。停止阈值`gap≤1e-8*(1+max|r|+max D)`，单位km²；gap=`w·grad−min grad`。报告实际步数、未收敛数、最终gap和目标值，不把未收敛静默换hard。理论FW gap仅证F代理最优差，不证实际误差界。
- 支持/源换序或重命名应数值等价；新增/移除源允许改变答案，不能称集合增删不变。浮点舍入按预定数值容差验证，真实K≥3效果没有数据支撑。

## 5. 对照、完整人口和预固定判断

T侧原133没有十臂完整artifact。本次用旧OOF Q分数**新重放**`WS_SHARED_Q_SOFT/HARD`及`WS_SOURCE_EQUAL`，再加三个新臂，共**6臂/15对**；明确是本次恢复计算而非历史T artifact。V侧原133十臂全部exact绑定，加三个新臂，共**13臂/78对**。十旧臂包括Mean/nearest-node/LOW/FixedLOW/旧Qgate、sourceequal/Qsoft/Qhard/linear-soft/shuffled-soft；固定源优先只留强对照，不能作为最终方法。

四个主要差（candidate−baseline）：

1. CHORD_RISK_HARD−SHARED_Q_HARD：A→A′的完整适配差；不是只读出改变。
2. CHORD_RISK_INVERSE−CHORD_RISK_HARD：同新risk普通连续权重相对硬选择。
3. CHORD_RISK_GEOMETRY−CHORD_RISK_HARD：集合读出是否超同risk硬选择。
4. CHORD_RISK_GEOMETRY−CHORD_RISK_INVERSE：在同risk下几何项相对普通风险权重的增量。

主结果为完整同人口径向位置MAE；同时保留RMSE/P95/P99、正负差数量/和/最大及>1000m损害、逐场/组件/去最大、0/1/2+支持、覆盖/重复/无输出/速度航向。四主差在T和V分别用2000次seed20组件bootstrap，报告描述性95%区间而非新独立测试显著性；不为所有78对扫描CI。

评分诊断只在预测后：新raw/clipped风险对真实chord²的误差，旧Q与新risk的端点排序及错选米损害、完整可用source-query与多源有标签域两种clip/floor分母。保存每query贡献熵/最大值、F负值和gap/迭代。源级端点损害与实际混合误差分开；hard平局的实际混合不等随机选一个端点的期望损害。原reference角色/纯混组及138六格如做复用，必须明确为T事后诊断、保留全部格，不进入模型或选参数；本轮不必为延续138专门重造一套条件分层。

**支持与失败：** B须在完整人口相对同risk hard及inverse获得有意义的MAE改善，并检查RMSE/尾部/组件代价，才能支持“集合读出增量”。只优于旧soft、只改C有利子集或仅训练风险下降，不满足该主张。若新hard改善而B不改善，只支持评分目标适配；若B局部改善但总体/尾部更差，保留作用域及损害，不直接扩量。若总体无增量，记录critic估计、条件信息缺口、3D代理/真实恢复差异与小oracle空间等竞争解释，不否定成熟集合方法整个类别。是否超历史LOW/FixedLOW/旧Qgate另报，不能以内部胜负代替可信同行。

## 6. 行为规格和实现文件

独占三个实现入口及针对性测试；不改133冻结API/公共helper/旧run：

- `association/evaluation/source_contribution_readout.py`：`chord_targets(errors_m)`；`fit_risk_model(x,y,sample_weight)`；`predict_risk(model,x)`返回raw；`contribution_weights(risk,available,group_offsets,source_states,kind)`返回source_weights及solver/floor诊断；`recover_contributions(states,support,groups,pack,source_weights,mean_states)`沿原node恢复。物理API不接受truth/identity/role。
- `tools/analysis/source_contribution_readout_pilot.py`：CLI `--native-run --value-run --partition-run --source-risk-run --manifest --output --cpu-threads --time-policy advisory`。校验完成输入/原名单/同node/query/mapping/support/counts/feature；新目录`exist_ok=False`，输出与输入不得重叠。只调用本模块一个critic的13fit。复用`observation_common`的显式method_names评价/聚合，不调用锁定十臂的旧评价函数。
- 原T训练pack/旧Q分数与当前physical重建绑定；保存训练标签与折模型。V必须在reference或旧控制artifact读入前冻结新权重/物理状态；模型文件已冻结，全Tfit完成后才进入V。旧十臂及人口exact绑定。

私有输出建议固定为`models.npz`、`T-fold-index.json`、`training-{sid}.npz`（合法特征/标签/mask/权重/组件）、`policy-{split}-{sid}.npz`（pack、old_Q_scores、risk_raw/risk_clipped、三臂source/node weights与group_states、floor/solver诊断）、`states-{split}-{sid}.npz`（method_field完整评价artifact）。场文件`scenes/{T_core|V_select}/scene-{sid}.json.gz`；公开`run.json`、T及V场表/summary；全部原身份/坐标/权重/逐点误差留private服务器，回传脱敏完整聚合。

元数据须记`analytic_head_fits=13`、实际线性系统solve数、T/V完成场数、每阶段耗时/RSS/输入读取/方法与对数、T旧Q新replay、全T模型冻结、0新增neural/teacher/nativeevents/grid/test；中途失败计数仅累计已完成阶段/场，当前部分场工作未知，不冒称未计算。

### 验收与必要软件检查

| 行为 | 针对性证据 |
|---|---|
| 球面径向e→真实弦长平方与几何单位 | 同位置/日期线/极区手算或独立3D距离；非法finite/范围拒绝 |
| 新fit只一个头，训练统计无heldout泄漏 | 小手算ridge；独立正常方程残差；修改heldout标签不改变对应模型；13fold清单 |
| K2端点/内部混合，K≥3可用 | 手算两端、对称抵消、同向共偏无保证、三角/非对称多源凸目标与FW gap；不同源顺序/命名一致 |
| source质量→原node恢复 | 同源多片段、support0/1/multi、非结合圆周例、speed/heading和Mean回退exact；NaN/错误mapping拒绝 |
| 全链合法人口与旧控制 | 人工极小T32/V32列表fixture测试13fit/T6/V13/四bootstrap及输出；篡改一个输入绑定失败并保全现场 |

这些fixture只认证软件行为，不当MTAD成绩/作者复现。生产实际结果另需一次有界执行及限定独立审计；不为测试数量扩建模拟训练链。

## 7. 成本、停止与记录

CPU≤4、RSS6GiB、AS12GiB、0GPU；建议墙时1800s，超过只记录/报告，继续既定终点而不自动加数据/训练。13个66×66系统；设计矩阵分折/分场读，保留必要训练10维数据，不创建所有场×所有方法巨大张量。复杂度含数据设计`O(L·66²)`及solve`O(66³)`，不是仅O(d³)；真实K2闭式权，泛K最多256次O(K²)更新/query，另有全部T15/V78对的评价与压缩IO成本。

尚无本次实测成本；原133全39头约141.506s仅作宽估参考，不承诺新run同速。单attempt的模型、预测、计数、command/launcher/退出和偏差全部记录。真实科学与审计终态后按Record→Squeeze→Mode C回流；没有结果前不把本文假设写成机制成立，不加新数据集/K≥3/在线/可信同行胜出主张。

## 8. 本地实现核验（软件证据）

已实现本合同两个生产文件及两份针对性测试。主仓`.venv`为已有环境，无新安装：

- `.venv/bin/python -m unittest tests.integration.test_source_contribution_readout -v`：6项通过，测试体0.033s。覆盖独立弦长/日期线、加权ridge正常方程、K2端点和负F、K4数值证书与换序、support/floor/原节点恢复，以及同源多片段圆周非结合反例。
- `.venv/bin/python -m unittest tests.integration.test_source_contribution_readout_pilot -v`：1项完整小fixture通过，22.371s。先用原133 writer建立4节点的人工T32/V32列表缓存，再执行140全链；核13fold的真实输入/normalizer与训练列表、T6/V13臂、四主bootstrap、完整人口/属性、篡改一处源feature绑定失败并保留旧完成run。仅软件fixture，生成的旧/新拟合不计真实科学运行。
- 四个Python文件`py_compile`及`git diff --check`通过。之后仅补显式scene `split/node_count`字段与source_weights维度拒绝；新增圆周反例随最后6项core测试执行，未为新增字段重复整个fixture。
- 初次误用系统`python3`缺numpy、bundled Python缺torch，均在导入阶段失败；随后采用已有`.venv`，没有将环境导入失败写成科学失败或删除记录。

当前核验不替代服务器真实64场、13fit的运行与原件检查，也不替代独立不导入生产实现的数值审计。
