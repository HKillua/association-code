# 当前研究状态

更新：2026-10-28。本页是**当前研究目标、结论与下一步的唯一入口**；执行状态只看[HandOver](../HandOver/Handover.md)。正式数据MTAD Level1：T2100/V300/封存测试600，固定056组及共享身份限制见[数据约定](data_contract.md)。

## 当前要解决什么

近期先形成一项可信的**位置融合／恢复贡献**，与已发表同行在同任务下比较，再接回补全—关联—融合完整系统。接受成熟机制和跨领域海上适配；090C是内部强对照，不绑定未来架构。按用户最新指导，以具体问题和可验证改善组织研究，允许成熟机制及跨域适配先形成解法，再依据机制、消融与必要推导提炼论文贡献；文献数量不作为开始有界探索的门槛。当前**没有稳定胜过可信同行的证据，也没有有效完成TMTAFF复现**。[用户目标订正原记录](reports/2026-10-26-goal-reconciliation.md)。

## 现在做什么

0. **继续扩大高质量来源与baseline筛选，不限原候选。** 用户要求本领域七八十篇、跨域几十篇，优先2025–2026；海上直接任务、一般航迹融合与相邻估计分别记，不能相互充数。[67篇逐项判断](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/screening_20261028.md)保留筛选当时的来源与边界；当前[登记156条](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/README.md)为领域及相邻99、跨域57，仍非156篇高质量全文或完成本领域70–80篇筛选。用户已提供原先缺正文的GD-IMM、MGFormer和GA-STR正式版并完成来源分析，均为已有候选升级而非新题名。选型继续比较新来源及输入/代码/效果证据，不因目录数收窄。
1. **Mode C更新到1.16，累计32篇核心来源：30篇正文分析＋2篇定向。** [矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-116)有6问题、63现象/数学性质、25机制、39证据；正式2025–2026刊会元数据23篇，实际阅读版本逐项保留。新增[GD-IMM](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/gdimm_source_analysis_20261028.md)、[MGFormer](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/mgformer_source_analysis_20261028.md)、[GA-STR](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/gstr_source_analysis_20261028.md)正式PDF；前两篇是单船AIS重建，第三篇是局部概率后验下单目标时空配准仿真，不能把论文成绩直接当TMTAFF同任务比较。MGFormer公开代码验证序列使用未mask位置、训练COG标签取time，公开CSV train/val有26共同MMSI；快照是否就是发表实验版本未知。GA-STR未知时间参数恒速分支估组合偏差而非分别识别，印刷公式存在可定位疑点，代码未定位。扩展批次累计13份来源分析、9份作者核心代码静态核查、4份代码未定位、0作者复现；其余候选含摘要、局部正文和历史分析复用，不混称精读。C21–C23仅是条件连接，尚无最终方法或MTAD效果。本包reference生成/变换未明确，限制物理真值主张。
2. **102/103已执行回传，本地核验通过；三场主因未唯一识别。** [完整实验记录/深挖/回执订正](reports/2026-10-27-returned-diagnostics-review.md)保留全部人口、对照和不利结果。103的5504声明query中4548可评；共同P95尾195点/15目标占090C SSE98.86%，现有标签下组成员一致，且部分近query原生量测相对reference已出现极大误差。090C三场MAE优于Mean而RMSE劣于Mean；不能只归因错helper、网络外推或直接判reference错误。C03旧GNC方案所需的可识别健康约束未建立，不进入该pilot。原任务结束，不重跑102/103。
3. **相对配准C10获得有限的预测输入条件支持，尚无绝对精度收益证据。** [完整Record/Squeeze](reports/2026-10-27-predicted-registration-audit.md)：不看真实身份估计中心，三个场景留出组等权相对残差3179.501→629.063、3937.242→244.724、2903.963→621.317m；3035两个组恶化，1043仅两组留出。早晚query检查是离线插值条件，非在线预测。方向多样性/绝对锚点/全V效果仍未知；与C08/C09[重新比较](reports/2026-10-27-candidate-feasibility.md#candidate-comparison)，不因已有结构或消融容易先定方法。
4. **104及105完成，原始reference已有短时大位移，根因仍未知。** [104记录](reports/2026-10-27-three-case-chain-review.md)的三场极值各±2 query、980项重放检查无不一致。[105记录](reports/2026-10-27-raw-case-chain-review.md)核对三个事后选定场景的六节点：原量测按既定规则重建后与103保留的完整五维canonical一致，三个目标的reference原始行也一致；三场所选目标的reference原始相邻时间点均可见数十公里短时位移。范围只限所查节点及目标，未证明真实航迹、时间或生成过程，亦不能推断全数据异常率。实际状态以[交接](../HandOver/Handover.md)为准。
5. **106人口核验通过，用户已批准A，107按固定额度推进。** [106 Record/Squeeze](reports/2026-10-27-unit-population-review.md)核完固定T2100/V300的7200成员，真实退出0、全部事前人口一致；T eligible155134，TCM2424步恰一遍、末批62。训练8个全空监督正对/51个至少一属性空监督正对保留，无整场某属性池全空；抽样batch若无监督仍按既定规则失败保全。[107事前计划](proposals/tmtaff_unit_baseline_A_20261027.md)固定seed20、fresh新单位TCM一遍→fresh TFM1000→固定056全V评价，oracle仅0/末步且固定使用末步。独立顺序入口与故障边界检查通过，领取/真实启动只看[交接](../HandOver/Handover.md)。旧流程约20h09m，新增评价和共卡使总成本未知；这是可行性诊断，尚无新端点效果或训练充分性证据，不用失败适配认定同行成绩。
6. **数据选取审计完成：保留现名单做开发比较，不能称作者同协议或身份独立泛化。** [审计与复算](reports/2026-10-27-dataset-selection-review.md)确认名单按既定规则重建一致，没有按模型成绩挑场；T/V共享6053个原始A键，V最大两个组件占63%。整场Q64与论文SL64的真实时间窗未对齐；旧适配在10183个共同位置点上TFM MAE59040.15m、Anchor583.53m，巨大差距不能全归因选错数据。当前090C仅学习位置，速度/航向沿用Mean。保留难例与封存测试，不据成绩重抽；身份泛化需另定协议，不从unused直接换一批冒称独立测试。

**用户当前决定边界：** 用户已明确回复“同意按草案A执行”，106前置核验已通过；无需再次索取相同预算许可。107只执行已定2424/1000及指定评价，不自动加步/seed/TAM或更换方法。用户另授权扩大论文、跨域组合和baseline研究；不受旧候选限制，但具体新方法与训练额度仍需形成可讨论的证据和方案。C08/C09/C10的历史证据见[105后讨论材料](reports/2026-10-27-candidate-feasibility.md#post105-discussion)。C09潜特征通道不能误称为物理属性独立权重。

## 保留的结论

| 对象 | 已核事实 | 决定／证据 |
|---|---|---|
| 056与090C/097A | 056固定组位置覆盖97.2109%、9353组/610重复；两seed位置MAE564.834/583.976m | 内部比较起点，不是可信领域同行；[方法](specs/evidence-association-fusion.md) |
| 098关系偏置 | 双seed均未过原晋级规则，最终关bias几乎不改变MAE | 当前配方结案；不代表训练期间bias完全无影响。[历史](history/README.md) |
| 100原生事件残差 | T0继续训练564.968m，T1固定汇总566.092m，T2 attention汇总567.187m，均未胜090C | 不晋级、不自动加步/seed；两臂都有真实query decoder。[完整分析](reports/2026-10-26-experiment-review.md) |
| 101缓存分析 | 090C对Anchor的绝对误差收益96.23%来自helper机会区；少数场景主导平方误差，普通控制亦有大错 | 现象已核，根因未定；不能直接归因错helper、数据坏或需要平滑。[完整分析](reports/2026-10-26-experiment-review.md) |
| TMTAFF/PTFM/TSA适配 | 当前实验未建立可信复现；096物理恢复和关联均弱 | 不用失败适配认证“赢同行”；[历史](history/README.md) |
| DeepAF | P/V核心存在，真实MTAD适配未完成；与TMTAFF所用四属性改版有差别 | 099暂停；不扩建模拟资格链。[实现边界](reports/2026-10-25-deepaf-method-audit.md) |

## 判断边界

- 本轮遵循[完整科研流程](research_workflow.md)：103、104、105、106及本地C10诊断各自完成Idea→实际执行→Record/Squeeze→Mode C；102和额外tail分别按指标合同/事后探索记录。107已有事前基线诊断计划与用户授权，结果待实际执行；C09方法Idea未执行。软件验收不是科研成绩。
- 无效配方缺少新证据就换，不因沉没成本续训；单次负结果也不否定整类模型。旧099的10/28初判、11/1去留节点不因换候选重置；3%等是具体旧合同门槛，不是普适创新定义。
- 论文指标按[统一定义](mtad_data_protocols.md)逐项补齐，另保留米制误差、尾部、覆盖/重复、场景/38组件。ME不是MAE；Level1量测kn/reference mps的已核范围见数据约定，历史原生速度不能冒称单位已统一。
- V反复用于开发，且有共享身份；不声称独立测试泛化。T_test保持封存，不删难例、不拼不同人口或单模块冠军。

详细历史不在本页追加；已执行事实进[结果分析](reports/2026-10-26-experiment-review.md)，旧阶段进[历史摘要](history/README.md)。
