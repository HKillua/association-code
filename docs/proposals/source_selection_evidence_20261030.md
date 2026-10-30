# 139候选：固定共享证据下的米制选源目标

2026-10-30；plan_version 1.0，**IMPLEMENTATION_SPEC，已由root及独立来源代理review，未READY、未运行**。本文件同时承载Idea与本次可执行规格，不另建平行规范。原133/134/136/138终点不重开；本项只有独立新合同发布后才能执行。

## 1. Idea来源与待解决问题

- package_type：IDEA_PACKAGE；package_version：research-pipeline-contract v1.0。
- idea_id / package_id：IDEA-SOURCE-SELECTION-EVIDENCE-20261030；候选实验139。
- source_matrix_id：MC-SELECTION-READOUT-20261030-v1.0；selected_C_id：SR-C1；已消费[事前局部矩阵](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md)24格及其完整来源/数学条件，不新增论文或实证计数。
- source_P_ids：[SR-P1, SR-P3]；source_F_ids：[SR-F1, SR-F2, SR-F3, SR-F4, SR-F6, SR-F8]；source_M_ids：[SR-M1, SR-M3]；source_E_ids：[E1, E2, E3, E5, E8]。SR-C2只提供需分离的读出边界，不混入本次目标干预。
- 上游：[Mode C1.57](../../../biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-157)、[133 Record/Squeeze](../reports/2026-10-30-source-risk-review.md)、[134 Record/Squeeze](../reports/2026-10-30-mixture-opportunity-review.md)、[136 Record/Squeeze](../reports/2026-10-30-observation-quality-review.md)、[138 Record/Squeeze](../reports/2026-10-30-observation-conflict-review.md)。
- 模式：Pattern 2，现象驱动；evidence_status：partial。原Q有选择信号但仍有少数高代价误选；本项目标改变能否减少完整风险、是否覆盖稀少角色反转及外部数据泛化，均未知。

**本次只问：保持合法输入、共享头容量、训练权重和hard物理读出不变，把回归标签从对数误差换成米制误差，能否减少选错源造成的实际位置损害？** 这是一项可归因的解决方案检验，不继续只做诊断，也不预定最终模型。

直接事实：134原T多源22627点中QOOF误选602点，平均endpoint选择损害47.264m，471点损害超过1km；138中154个匿名角色反转点仅8个选对，但另456次误选仍在多数角色。不能将主要问题全部归为角色反转，也不能由98%以上选择准确率推断米制损害已足够小。原133使用`log1p(e_m/1000)`标签；其高误差差异被压缩，是需要隔离检验的目标因素，**尚未证明是这些误选的根因**。

原计划优先核原生速度/航向的跨属性信息。定向复核[数据约定](../data_contract.md)及[单位/评价协议](../mtad_data_protocols.md)发现：已有Level1前8个T场中295148/295191相邻区间在kn→m/s后与位置差分相对误差小于1%；协议明确其接近“带噪位置前向差分速度”。因此不能将该变量包装为独立传感通道或健康证据，也不能直接搬用IMU/接触运动学的独立约束。更丰富的时序表征可能仍有价值，但本次不加变量，避免与目标变化混改。

来源机制只作有条件参照：[Chroma完整来源分析](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_transfer_sources_20261030.md#2-lf-b1-完整来源分析包chroma)正文§3.4/p5、附录D在同一合法历史目标上比较集合损失，消融显示单体分数加权不等集合效果；它不提出本文的径向风险回归，也不保证本数据上的选择。CoCo等任务监督方案提供“用任务误差训练后再进入结构化推理”的可行范式，不提供本项目传感器真实协方差或健康校准。没有核得该精确MTAD适配的独立跨领域C，不冒称无人做过，也不为补C无限搜索。

## 2. 干预、理论含义与竞争解释

固定原133每个合法source-query的10维特征`x`、径向误差`e_m`和训练权`a=1/K`。旧目标与新目标分别为：

`y_log = log1p(e_m/1000)`；`y_meter = e_m/1000`。

对训练折内所有合法行，令`ã=a/Σa`，新模型仅最小化：

`Σ ã_i [g_β(x_i) − y_meter,i]² + Σ_{j>0} β_j²`。

这里平方的是**误差分数的回归残差**，不是物理位置残差；本项不是条件位置MSE/二阶矩critic。公里是预先固定数值单位，不搜索单位缩放或λ。除标签变换外，λ1、模型结构、训练权和读出均保留原133，因此实际比较包含该固定目标尺度下的相同正则配方，不声称已分别找出两目标各自最佳正则。

**AI数学解释：** 若无正则、容量足够，在该加权分布下平方回归的最优量是`E[a·e_m/1000 | x]/E[a | x]`，旧目标对应`E[a·log1p(e_m/1000) | x]/E[a | x]`。当前合法标签实际K=2、a恒定，可简化为相应未加权条件均值。两者的排序在重尾条件分布下可能不同；对任意K不能直接搬用未加权条件均值结论。本次只有有限66维ridge投影，单源特征还可能缺少共同决策信息，故**不保证Bayes最优选源**，更不证明物理可信度、Cov、无偏性或概率校准。

可反驳预测：若原对数压缩确实妨碍当前有限模型对高代价选择的拟合，同容量的新米制头应在未参与该折训练的T组件和固定V完整人口上减少实际位置MAE及高代价误选；不要求选择准确率必然上升。若仅训练误差下降、OOF或V恶化、去最大组件后反转、或均值收益伴显著尾部恶化，则不能宣目标改变解决问题。

竞争解释保留：已有特征仍可能主要识别生成角色，缺乏区分反转的证据；原误选可能主要来自分组污染、有限容量或监督稀少；直接米目标可能被共同高误差/极端点支配，导致过拟合。新目标不会增加输入信息；对稳定共同偏差的可辨识性也没有因此建立。相对query代价、鲁棒损失、重加权、新特征、温度与集合风险读出本次都不加入。

## 3. 固定输入、监督权限与拟合

仅Level1原T32/12组件、原V32/9组件；复用127原生状态缓存、125的T membership、114的V WS、133的训练pack/原模型与OOF、136源nearest控制以及134的T诊断键。无需原生事件归档、ZIP/CSV、原始teacher权重或新teacher前向；T_test封存。原数据T/V共享身份及反复开发限制仍成立。

特征完整继承133的10维与原顺序：最近原生点时距均值、OLS fit RMS均值、native点数均值、片段span均值、支持片段数、支持比例、本源成员位置RMS、其余可用源等权位置距离、OLS速度向量差、航向圆周差；沿原`log1p`尺度与圆周退化规则。source ID/编号/匿名role、真实identity、reference、绝对坐标不进入特征；不能将native计数等角色相关proxy自动认作跨传感器不变。

训练标签完全沿133：组first-anchor目标已知、同目标仅首组、位置reference有效且当前至少两源available；每query总权1、每源1/K。沿同一源行映射读取`errors_m`形成新标签，核旧`y=log1p(errors_m/1000)`、labeled/weight、source/query/组对应一致。重复/未知/无reference/0或1源query仍输出和评价，只不赋新训练标签。混组label相对anchor目标，不是源自身物理健康。

只拟合QUADRATIC：加权训练均值/方差；std<1e−6置1；标准化z裁[-8,8]；截距＋10线性＋55上三角二次项，共66系数；λ1且截距不罚。12个组件LOCO各仅使用其余T组件拟合normalizer与β，再拟合1个fullT模型，共**13次新解析拟合**，不是神经训练。没有重新拟合旧log模型、LINEAR或SHUFFLED，没有新增target专用置乱控制；原置乱臂仅作历史背景，不据它证明新目标的特有信号。

旧log OOF分数仅读133已保存分数。新T分数用对应heldout-component模型；fullT模型保存并冻结后才进入V预测；V特征先形成、新旧预测固定后才打开V reference作评价。OOF只隔离本次共享头，不把上游全T teacher变成全链独立泛化。空训练集沿原零系数fallback并计数；真实固定折若标签数/组件对应不符则失败，不删场继续。

新分数可以为负，**不clip、不加常数来宣非负、不改变排序**。报告可用行负分数比例/min/分位、选中负分数比例、精确tie数及z裁剪数；仅称“米制目标得分”，不能直接称合法非负风险估计。非有限分数直接失败保全。

## 4. 物理读出与固定对照

新头只改变`source_weights(...,kind='hard')`的输入分数。可用源取预测最小，精确平局均分，不新增epsilon tie；不可用源权0。每源质量均分给本源当前支持片段，使用原节点四状态恢复经纬度；无支持query沿原Mean first-anchor回退；速度/航向最终逐数组覆盖为旧Mean。圆周无定义沿133同group-anchor退化，计数并披露。不能直接混已经归一化的source圆均值替代原节点恢复。

| 分区 | 固定臂 | 计算/导入性质 |
|---|---|---|
| T32，4臂/6对 | `WS_SOURCE_EQUAL`、`WS_SOURCE_NEAREST`、`WS_LOG_Q_OOF_HARD`、`WS_METER_Q_OOF_HARD` | 前两臂从136完整T artifact导入exact；旧log从133既存OOF分数恢复完整T，D2与134既存真实hard核对；新meter用新LOCO分数恢复 |
| V32，12臂/66对 | 133原10臂＋136 `WS_SOURCE_NEAREST`＋`WS_METER_Q_HARD` | 前11臂完整artifact逐数组绑定；仅最后一臂是本项新目标政策 |

133原10臂名称原样保留：`WS_MEAN`、`WS_NEAREST_SUPPORTED`、`WS_LOW_SOURCE_SUPPORTED`、`WS_FIXED_LOW_SOURCE_SUPPORTED`、`WS_QUADRATIC_VALUE_GATE`、`WS_SOURCE_EQUAL`、`WS_SHARED_Q_SOFT`、`WS_SHARED_Q_HARD`、`WS_SHARED_LINEAR_SOFT`、`WS_SHARED_SHUFFLED_SOFT`。源nearest与旧node-nearest明确不同。LOW/FixedLOW/旧Q保留强反例，不输入新模型、不作为最终固定源策略。

原133没有保存全T的QOOF完整artifact，故本项旧log全T恢复是**旧政策的新完整执行/评价**，不是伪称全部旧结果直接复制。其D2真实位置/误差必须与134绑定；0/1源域新旧hard按定义同恢复，不能将这些位置称新学习增益。T新旧拟合目标的MSE/MAE分别按自身标签单位报告，不能直接比较log域MSE与km²回归残差来宣布改善。

## 5. 估计量、完整风险与固定诊断

**唯一主比较：** V原完整共同71071有效位置点，`mean(e_WS_METER_Q_HARD − e_WS_SHARED_Q_HARD)`，负数有利。保持完整目标、anchor首代表、有效mask、覆盖/重复/漏目标与原人口绑定。所有66对保留供完整强控解释，不按V挑最佳对。

V主要比较同时报告RMSE/P95/P99、点级改善/恶化量与数量、32场和9组件、场宏平均、按原场数定义去最大组件、0/1/2+可用源、原固定支持/时距层及成本。seed20、2000次组件bootstrap只用于主MAE差；9组件且V已反复开发，不称独立测试显著性。源作用域与真实identity输入片段span并集分开，不能把0源等同无原生观测/纯外推。

T保留完整37635位置点及全部58304预测组query计数；新旧QOOF完整MAE差及同样12组件bootstrap为预定辅助量，不能把训练内fullT分数成绩替换OOF。训练标签45254行及D2 22627query是另两个分母。记录重复anchor、未知、无reference、未覆盖、0/1/2+和源数；源行不能当独立query增加样本量。

T的D2仅复用134已核query/source-row/endpoint/角色/纯混键，不重算4097网格，也不把reference-best作为政策。每个新旧头保存：

- 实际hard径向误差；`selection_cost=Σ_s w_s e_endpoint,s`、`selection_regret=selection_cost−min(e_endpoint)`和`selection_credit=Σ_s w_s1[e_endpoint=min(e_endpoint)]`。前者是真实几何输出，后三者是端点选择诊断；平局可能不相同，不互相替换。
- 新旧正确/误选的固定2×2计数（tie的credit单列，不强作严格正确）；严格救回/新引入错误的数量与米制regret总量；全部点的新减旧signed误差、正害和负收益分别保留。>1000m损害及0<损害≤1000m沿134/138既有描述尺度，不变成规则阈值。
- 固定匿名role_0最佳/role_1最佳/参考精确平局三层、纯/混组两层、12组件/去最大；不继续全部交叉筛选。关注已存在154反转点的完整结果，同时保留多数角色中的456旧错误，不能只报道救回少量反转而忽略全局代价。
- top1%集中度使用`ceil(.01×D2点数)`，按各头positive selection regret排序；分母是该头总regret，0总量时null。新旧同一人口，不用“受损点的1%”替换。

角色表只在评价阶段从134 private mapping读取，绝不进入训练/分数/读出。源换名/换行在合法group成员保持时应等价；现真实数据K2，K1/K3/K4仅必要软件验收，不宣已经实测任意多传感器。

## 6. 可执行输入与产物合同

拟CLI：`--native-run --value-run --partition-run --source-risk-run --quality-run --mixture-run --manifest --output --cpu-threads 4 --time-policy advisory`。输入均固定为已完成原run，输出新目录；不可根据结果换数据/阈值/模型。

| 参数 | 绝对路径 |
|---|---|
| native | `/home/wangguangjie/djs/association-code/runs/LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1` |
| value | `/home/wangguangjie/djs/association-code/runs/LEVEL1-CONDITIONAL-SOURCE-VALUE-20261029-125-attempt-1` |
| partition | `/home/wangguangjie/djs/association-code/runs/LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1` |
| source-risk | `/home/wangguangjie/djs/association-code/runs/LEVEL1-SOURCE-RISK-20261030-133-attempt-1` |
| quality | `/home/wangguangjie/djs/association-code/runs/LEVEL1-OBSERVATION-QUALITY-20261030-136-attempt-1` |
| mixture | `/home/wangguangjie/djs/association-code/runs/LEVEL1-MIXTURE-OPPORTUNITY-20261030-134-attempt-1` |
| manifest | `/home/wangguangjie/djs/association-code-run136/configs/data/mtad-level1-scene3000.json` |

固定科学输出建议为`/home/wangguangjie/djs/association-code/runs/LEVEL1-SOURCE-SELECTION-EVIDENCE-20261030-139-attempt-1`。具体checkout/launcher/CPU核与审计命令由实现review后的L2S发布；本表不是READY。

输入先核各run完成、原名单/组件/lineage及source/query/membership完整对应。允许由相同成员集确定唯一源行双射并保存，不用source排序/误差排序对应；任何多义/缺失/非有限失败，不静默删query。主要产物：

- `run.json`：绝对输入、真实命令、固定T/V名单/完成数、阶段、13fits计划/实际、预算/成本/退出/失败。成功才标完整；失败当前场部分工作未知须保留。
- `T-fold-index.json`、`private/models.npz`：每`full`或`leave_{component}`的mean/std/beta/训练行数/权总量/训练组件/target_kind；normalizer与旧133对应fold一致，参数不可要求与旧目标相等。明确target_kind=`radial_error_km`，head=`QUADRATIC`，而非MSEcritic。
- `private/training-{sid}.npz`：原pack/映射/合法标签mask/权重、`errors_m`、`y_log`与`y_meter`、component；完整标签与特征不公开。`private/oof-{sid}.npz`保存新旧分数与fold绑定。
- `private/policy-{split}-{sid}.npz`保存pack、源/节点权重、得分与新旧group_states；`private/states-{split}-{sid}.npz`保存T4/V12标准完整artifacts。`private/diagnostic-{sid}.npz`保存D2原键、endpoint/实际误差/credit/regret/role/purity及原134核对字段。
- 公共T/V逐场、汇总、OOF/D2诊断：`T-scene-summary.json.gz`、`T-summary.json`、`scene-summary.json.gz`、`summary.json`、`T-selection-diagnostics.json`，按`all_components/by_component/without_largest_component/largest_component_id/primary_bootstrap`现有结构。公开不含source原名/身份/坐标/权重。

## 7. 验收、成本、停止与结果路由

实现文件仅新`association/evaluation/source_selection_evidence.py`、`tools/analysis/source_selection_evidence_pilot.py`与针对性测试。可只读复用原source_risk/common的预测、物理恢复及评价接口，不改原133/136/137模块。单Q拟合避免调用原三头fit后丢弃两头却少算成本。

| 验收项 | 证据/预定检查 |
|---|---|
| 只有目标改变、13fits | 新Q用原log目标时与旧QUADRATIC回归数值一致；raw目标独立加权法方程重算；每折训练组件、normalizer与原133绑定 |
| 不读reference构造新特征或V政策 | 访问顺序fixture；T标签沿原pack独立核；full模型先冻结再V预测/评价 |
| 权重与物理读出 | K1/3/4、源行置换/重命名、无支持、exact tie、负score、圆周退化；实际恢复走原核，速度/航向旧Mean exact |
| 强控与完整人口 | T两个/V11个导入臂逐数组exact；旧log T D2对134真实hard；0/1/多源、完整target/coverage/重复、主要误差和所有pair绑定 |
| 科学主量可复核 | 独立NumPy从保存训练数组重算13fits、OOF得分、source权重/得分、完整位置指标/主bootstrap和D2代价；不导生产predictor/evaluator冒充独立 |
| 真实链路软件验证 | 小fixture走runner全T/V合同和审计；必要篡改标签/score被拒绝。只认证软件，不代替真实MTAD结果 |

CPU≤4、RSS6GiB、AS12GiB、1800s advisory，GPU0；13解析fit、0神经更新、0新teacher、0新状态估计、0新分组、0oracle搜索。逐场处理，保存大数组在server；报告实际输入读数、预测/恢复次数、wall time/RSS、13fit真实成本。1800s非删场或截断预算；超建议告警后只完成固定终点。科学一次attempt，非有限/绑定失败/OOM保全，不自动重试/改目标/降样本。

固定终点为T32/V32全部输出、限定独立审计、Record→Squeeze→Mode C回流。主量、尾部、组件/去最大和强控共同支持时，才考虑后续独立扩展；V不利即结束此配方，不按V换loss/λ。若T OOF有益但V不利，优先报告分布/目标适配边界；若均值改善但角色反转或尾部恶化，保留明确代价，不包装普遍可靠选源。若各层均无增量，不能因此否定更多合法信息或其他适配，也不再重复调这个固定配方。

这是一项低成本的任务目标适配，不主张回归/平方损失/hard本身原创。它和140拟议的平方风险critic/集合读出是独立问题，结果不拼成联合冠军；即使成功，也只建立当前内部共同协议的增量，不能宣真实K≥3、在线、换传感器或可信国际同行胜出。

## 8. 发布前软件证据（不含真实MTAD结果）

2026-10-30，主仓`.venv/bin/python`：新目标3项单元、真实runner的1项完整缓存fixture、复用读出路径的原source-risk12项均通过。fixture从133原拟合→136原控制→134原D2→139新目标实际运行，T4/6对、V12/66对、13fits及旧标签不匹配即失败保全均覆盖；该fixture是人工小数据，只核软件。

独立审计由另一代理实现，不导生产预测/评价计算；2项审计单元与同一已完成fixture上的3项集成方法通过，分别核完整数组及标签、β、source权重、实际误差、regret和role篡改拒绝。最终软件审计COMPLETED/exit0，331718 exact、108969 numeric、20592 null；D2=1952/标签3904是**人工fixture计数**，不得写成139真实数据结果。最初加载旧字段定义导致的软件审计拒绝已在同一缓存重载最终代码后核验，不重跑真实科学任务。

独立覆盖：13模型/原normalizer/fold、全OOF及V分数、原节点hard恢复、T2/V11旧控制exact、D2对应/全部预定诊断、主要位置/人口/组件/去最大/主bootstrap。限制：不认证原reference物理正确性、原127/teacher、真实健康/因果/外部泛化；时间访问权限由代码结构和软件检查确认，不能从存盘数组单独证明；速度航向核原值相等，附加支持条件的pair层不冒充全部独立重建。后续真实运行和实际成本仍须单独Record→Squeeze，不由本节认证。
