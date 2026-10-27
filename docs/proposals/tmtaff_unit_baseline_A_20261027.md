# TMTAFF新单位基线A：事前诊断计划

2026-10-27。用户已明确同意：**106人口核验通过后，执行单seed20、TCM有序一遍2424步、TFM1000步，以及固定056全V评价的方案A**。本计划在该训练执行前形成；首次书写时尚未核到106最终终态，后续同日已独立核验通过，事实追加见末节。执行状态与领取安排仍只见[研究状态](../research_status.md)和[交接](../../HandOver/Handover.md)。本文件不发布READY，也不声称已有新单位权重或效果。

## 交接元数据

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-TMTAFF-UNIT-BASELINE-A-20261027
idea_id: IDEA-TMTAFF-UNIT-BASELINE-A-20261027
plan_version: '1.0'
created_at: '2026-10-27'
source_matrix_id: MC-RESTART-20261027@1.10
selected_C_id: C01
selection_scope: existing_same_protocol_baseline_foundation; no_new_C_selected
source_P_ids: [P01, P02, P03, P05]
source_F_ids: [F01, F02, F26, F27, F28, F31, F32, F33]
source_M_ids: [M01, M02, M03, M05]
source_E_ids: [E01, E02, E11, E21, E22, E24, E25]
source_ids:
  - MC-RESTART-20261027@1.10
  - IDEA-TMTAFF-UNIT-PREP-20261027
  - docs/reports/2026-10-27-fair-comparison-contract.md
  - docs/reports/2026-10-27-tmtaff-speed-unit-verification.md
  - docs/reports/2026-10-27-dataset-selection-review.md
status: diagnostic
authorization: user_approved_budget_A_conditional_on_106_population_verification
execution_gate: 106_population_verified_execution_requires_separate_handover
evidence_status: partial
missing_evidence:
  - new_unit_TCM_and_TFM_fixed_endpoint_results
  - new_unit_fixed056_full_V_results_and_cost
  - author_exact_preprocessing_split_stage_budget_and_ESD_implementation
  - dataset_reference_generation_and_independent_physical_truth
  - sufficient_training_budget_and_multi_seed_stability
next_skill: record-experiment
```

`evidence_locations`为下文来源表、比较合同§8的088/093A历史原件定位、106[事前准备计划](tmtaff_unit_population_preparation_20261027.md)及其真实回执；尚未得到的106事实不预填。按[完整流程](../research_workflow.md)、[discover-innovation](/Users/a1/.codex/skills/discover-innovation/SKILL.md)的Pattern 5（系统/接口假设核查）和[experimental-design](/Users/a1/.codex/skills/experimental-design/SKILL.md)形成诊断设计。训练后保留本版本，以实际配置和偏差进入Record→Squeeze→Mode C。

## Mode C入口与研究配置

领域为多源海上轨迹恢复与融合；研究对象是TMTAFF时间适配在固定MTAD Level1人口、显式物理速度单位和受限历史曝光下的可用性。**selected_C_id=C01仅引用已有同协议比较基础，不是本轮选择新的创新连接。** 原矩阵C01为P01×F01/F02、M01/M02、E01/E02/E11，状态supported，六维2/2/2/0/0/2；其新颖性为0，矩阵明确“不独立包装创新”。C08/C09/C10保持未选择，本计划不实现其网络或改变优先级。

来源均定位到[Mode C 1.10及其保留增量](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md)。下列额外P/F/M/E为设计边界，不改写C01原定义，也不把当前未完成106补造为新E。

| 已有矩阵项 | 来源定位与本计划使用方式 |
|---|---|
| P01/F01/M01/E01/E11 | 矩阵§4.1–4.3、TMTAFF原文式与表、[指标](../mtad_data_protocols.md)/[数据约定](../data_contract.md)：同数据名不等于同任务，ME/MAE/ESD、单位、人口和聚合须分开 |
| P02/F02/M02/E01/E02 | 矩阵§3、§4及TMTAFF/PTFM原文分析：观测嵌入补全与物理状态监督不同；已有方法的属性间代价不能省略。不因单位修正预设恢复成功 |
| F26/E21、F27/E22 | 矩阵§11.2–11.3、[102/103 Record/Squeeze](../reports/2026-10-27-returned-diagnostics-review.md)：速度偏差/航向误差与位置继承关系明确，MAE改善可能伴RMSE恶化；必须报告四属性、全人口及尾部 |
| P03/F28/M03/E22/E24 | 矩阵§11、§13：共同极尾已存在于量测—reference关系，原链重放一致不证明物理真值，不据尾部标签删数据或认定单位是唯一原因 |
| F31/E11、F32/F33/E25 | 矩阵§14–15、[105 Record/Squeeze](../reports/2026-10-27-raw-case-chain-review.md)：数据生成及reference物理含义仍未知，六节点一致与原reference短时大位移均局限于固定选例；不推广为全V异常率 |
| P05/M05 | 矩阵§4.2–4.3及§11：oracle正对训练与真实预测组不同，故oracle诊断不能替代固定056预测组结果；训练呈现、共享身份与部署条件分别披露 |

DE：单位转换链、配置/接口与历史曝光有可核实现和软件证据；旧093A在oracle条件下的严重位置失败已记录。AI：单位一致是改善基线可信度的必要修正，但能否显著改变位置恢复仍待本轮检验。UN：作者结果复现、充分训练、reference独立物理真实性以及新单位实际效果。本诊断使用包内reference作为既定评分对象，不把它提升为已独立认证的真实航迹。

## research_question 与 hypothesis_and_mechanism

研究问题：在与旧088/093A尽量相同的结构、损失、采样和曝光下，从头采用`measurement_kn_reference_mps_v1`的TCM/TFM，能否形成有限预算下可解释的恢复基线，并在固定056组的全V300上与普通控制和090C位置完成同任务比较？

机制假设为有限命题：旧量测kn与reference m/s混用会改变模型输入/监督的物理关系；显式统一后，T内normalizer、新TCM表示与TFM优化目标处于声明一致的单位路径，原部分失配可能减轻。**单位一致不保证位置精度、训练充分或真实reference成立。** 替代解释包括：仅1000步场均匀正对抽样覆盖有限、TCM观测嵌入目标与真值恢复有差别、递归/单例处理和整场Q64窗口适配、既定reference/场景生成条件、优化随机性及实现差异。

testable_predictions：

1. 106确认固定人口与监督后，新训练/评价应保持相同名单、query、目标、reference valid及固定056分组；若改变则当前方案不可按原比较解释。
2. 新TFM u1000与u0的oracle位置MAE、RMSE及各属性误差给出真实优化方向；若u1000仍差于u0或保留数十公里误差，这个额度下的单位修正没有解决恢复失效。u0仅诊断，绝不被选为最终模型。
3. 新旧093A oracle在可绑定的同人口/同指标下报告变化；若大幅改善，仅支持“当前新单位路径与有限配方更可用”，不把全部变化归因一个单位开关；若仍失败，削弱“单位是充分修复”的解释。
4. 固定056全V的主要比较不由oracle变化预定：改善、无改善及MAE/RMSE相反均完整报告，特别检查覆盖、极尾和最大身份组件外。

## design_and_controls

流程固定为新TCM→新TFM→固定056评价。精确JSON及CLI见[比较合同§8.3](../reports/2026-10-27-fair-comparison-contract.md#83-待用户选择的具体a预算新单位基线可行性诊断草案)；文件名保留`draft`历史命名，但经本计划冻结预算并获条件授权，不修改其内容。真实设备、来源路径和独立输出目录由独立交接具体绑定，禁止自行替换为其他人口或旧单位权重。

| 阶段/臂 | 固定设计 |
|---|---|
| 启动前条件 | 核106真实完成、退出/配置/范围、T2100/V300及各项人口/监督无未解释差异；特别确认TCM eligible=155134。仅“进度完成”或总节点相同不足以通过。条件未通过不开始训练，不重复106 |
| TCM | [配置](../../configs/tmtaff-tcm-unit-draft.json)：seed20，fresh模型及T内normalizer；`ordered_fragment`一遍、batch64、2424成功更新，最后62条，预期35555236双向transition；每个实际曝光量回传。固定最终checkpoint |
| TFM | [配置](../../configs/tmtaff-tfm-unit-draft.json)：仅转移本次完成TCM的冻结completion/normalizer，新fusion/optimizer；seed20、1000成功更新×64正对=64000呈现。按有正对场均匀再场内抽样，场内不足64有放回；不训练TAM/JOINT，不resume |
| 结构/优化 | model/fusion/training/train_splits全部保持对应历史维护配置；128维、8头、6层、dropout0.1，track_microbatch4、pair_microbatch8、常数LR0.002、clip1、`standard_smooth_l1`/`shortest_arc`、`train_position_affine`。两配置相对level1文件只新增速度单位policy，不调loss权重/采样 |
| Oracle监测 | TFM仅u0/u1000各一次完整V评测并保存预测，`--pairing-audit --save-predictions`。选定端点始终u1000，不按V挑最优，不追加检查点或更换训练量 |
| 固定056 | 完整TFM最终checkpoint；056原final1000分数、阈值、完整node/pair/分组、Q64/reference绑定，TAM前向0、阈值重选0。真实失败组、单例、缺失、重复全部保留 |
| 普通控制 | `canonical_all_anchor_ols`、`canonical_all_mean_supported_ols`使用全原生点；另给`selected_cap128_anchor_ols/mean_supported_ols`区分输入点数效应。相同组/query/reference和单位，同一评价器；不以cap控制取代强控制 |
| 090C内部参照 | 仅位置、固定056、相同成对有效点；其速度/航向来自旧Mean，不当成本轮新单位学习输出；不是可信同行替身 |
| 历史/外部参照 | 旧093A oracle是受限历史诊断；论文原表是外部参照，不将有符号ME换算成米制MAE胜负，也不声称本轮复现作者完整系统 |

单位转换仅对输入量测速度kn乘1852/3600一次，reference沿既定已核范围保持m/s，论文速度报告转kn。不得缩放旧神经输出/权重冒充重训。历史运行与本次还存在单位标记、评价修订、pairing-audit记录、固定056 wrapper及共同mask统计等变化，软件环境和浮点轨迹未承诺逐值重放；**本设计不是纯单变量因果单位消融**。除声明变化外不增新模块、方法干预或数据修补。

## sample_materials_or_experimental_units

固定Level1原分卷、既有3000名单及alignment，只读T_core2100/V_select300；T_test600及未用2000场payload封存，不因当前成绩重抽场。Q64为既定整场真实秒query，原生cap128/time_unit_seconds20、reference offset/gap600/shortest_arc和隔离规则不变。106预期人口及确认依据见[其计划](tmtaff_unit_population_preparation_20261027.md)，本文件不预填实际通过。

训练轨/正pair是曝光单位，不是独立船只；64000次pair呈现不等于64000个独立pair或完整epoch。评分对象为每场声明目标/query/逐属性valid，先报告全目标/缺预测/缺真值/重复，再报误差。V300有38共享身份组件，T/V共享身份，且V已反复开发；不将query、场景或随机种子1次结果解释为独立泛化实验。

## primary_outcome_and_estimand

主要结果是**固定056组、全部V300的完整人口覆盖，以及新TFM与每个强普通控制在成对共同有效位置点上的位置MAE和RMSE差**。差值统一为“新TFM−控制”，负值表示该指标较好；每个对照独立报告其共同有效点数量/比例。RMSE从该共同人口的平方误差重算，不平均各场RMSE；缺预测不写成零误差，不把不同有效人口的平均直接作胜负。对090C以同规则另报位置差，不合成“最好模块”主表。

覆盖是解释误差的必要条件而非事后过滤器：误差下降同时覆盖下降，不能直接宣布更好；需同时看各自全有效人口与共同mask。不同指标冲突则并列，不合成为单一排名。此诊断不预设“提高X%即可认证同行”的门槛，也不支持从一个seed宣布稳定优势。

secondary_outcomes：位置P95/P99、坐标MAE、经纬10^-4度ME/标准残差SD、速度m/s及kn误差、航向raw/circular误差、各属性覆盖/重复、16项论文维度（8项关联继承056），支撑/外推距离分层，TCM/TFM训练损失和梯度/实际曝光/成本。作者mESD实现仍UN，标准残差SD明确标识。oracle u0/u1000和旧093A比较为诊断层，不替代主要结果。

## analysis_plan_and_uncertainty

保留每场原预测、真值valid、目标绑定、逐属性共同mask、源单位及各臂来源；全V300、逐场、38组件及剔除最大130场组件后的结果均报告，后者只是预定敏感性分析，不能删掉主表中的难例。报告场景/组件间分布和配对差异，不将query级标准误当独立样本不确定性。单seed20不做稳定优势或统计显著性结论；本轮不新增bootstrap/seed训练来制造置信度。

训练过程保存真实成功更新、TCM轨/transition曝光，TFM有序pair draw、unique场/轨/对、替换次数、每属性有效query监督。逐阶段训练与评价时间、参数、峰值显存分开列；种子相同不证明浮点全程重放。所有不利、缺失和失败结果保留，不挑场、不挑checkpoint、不由看见结果临时改主要估计量。正文或图表新增事后分析时显式标记。

## resources_time_cost_and_risk

各阶段单次尝试，显式单卡与CPU4，按服务器FIFO/最多四单卡规则安排，不触碰他人作业；TCM/TFM依赖顺序不能并行。8h是各阶段配置的advisory墙时，不是自动硬停或可扩步数的额度。旧088＋093A在3090的墙时约20h09m，此外还有本轮固定056全V额外评价；新单位、准备/缓存和共享负载使新总成本未知，不能保证仍20小时或评价免费。用户批准的是固定2424/1000和指定评价的工作量，不是自动增加到4000步。

风险包括长时间oracle评价、准备/缓存内存、非有限梯度/OOM、有限曝光仍无法收敛，以及好均值掩盖坏尾部。启动后不因耗时或负结果改batch、跳评价、加步、补seed、重启或resume；advisory超时仅记录并继续本次原固定范围。保存真实设备/软件/命令、阶段时间与失败原因；不以历史进程PyTorch allocated当作当前整卡可用显存。

## ethics_safety_privacy_or_compliance

原ZIP、reference、既有056/090C及历史运行不修改。身份/坐标、真实预测NPZ与完整有序draw等可能泄露原始身份的产物仅留服务器private；Git仅回传脱敏计数/指标/必要日志及来源，traceback也先核敏感字段。无外部受试者或新数据采集；不得依据标签或reference在推理前向配准、筛成员或挑query。

## stopping_and_interpretation_rules

1. **未满足106前提：** 不启动；记录差异并讨论，不改样本/预算来凑155134，不复跑106。106通过只确认准备人口与监督范围，不认证单位全部物理语义、数据真实性或训练充分。
2. **工程/绑定失败：** 输入缺件、单位/人口/reference/056绑定不符、非有限梯度、OOM、异常或用户停止，终止该次并保留全部已有日志/checkpoint/退出；不自动重试、缩batch、删数据或加训练。未完成端点不能包装为完成比较。
3. **正常固定终点、oracle仍差：** 该有限配方仍不可作为可信同行胜负依据；区分损失/梯度、属性尺度、曝光和支撑层证据，削弱“单位足以修复”的解释，停止该配方，不自动追加训练。
4. **Oracle改善但固定组无改善：** 保留改进及无改进，检查错组/单例/覆盖/分层差异；不能以oracle宣布部署效果或立即归因TAM，更不启动TAM。
5. **固定组平均改善但覆盖/尾部/其他属性恶化：** 完整报告取舍；不以MAE单项晋级、不删难点，未来是否改目标或机制另行讨论。
6. **各项表现可用：** 支持本设置、单seed、固定预算下的同任务参考可行性；仍非作者精确复现或稳定优势。未来正式多seed、充分训练、完整系统或方法候选需独立事前方案，不由本轮自动触发。

## 三层质量检查与下游

新颖性：本项不提出创新主张，不再为“单位修复”搜新文献背书，C01新颖性0不提升。研究价值：成功建立可解释的比较端点；失败也缩小旧适配为何不可用的解释范围，避免把弱baseline当同行。可归因性：固定数据/显式结构/损失/曝光和末步，区分统计与工程变化；因未重做历史环境的纯单变量配对、多seed或独立测试，因果和泛化结论均受限。

本计划的状态为`diagnostic`，已获方案A的条件授权。独立执行合同核106终态后才能实际启动。执行后首先生成`EXPERIMENT_RECORD`，用`source_idea_id=IDEA-TMTAFF-UNIT-BASELINE-A-20261027`及`plan_version=1.0`逐项核计划与实际，并包含失败；再Squeeze、回流Mode C。新研究方法选择仍留待讨论。

## 同日启动前追加：106人口前置已完成独立核验

此处是计划形成后、A训练启动前的前置事实更新，**不改变plan1.0的训练量、配置、目标、端点或停止条件**。106真实终态为`COMPLETED`、launcher退出0、`mismatch_count=0`；本地经已授权SSH取得服务器原件后完成独立审阅。来源是[据实补写的S2L-106](../../HandOver/S2L-20261027-106.md)、[Record/Squeeze](../reports/2026-10-27-unit-population-review.md)及[60778项公有产物复算](../reports/2026-10-27-unit-population-audit.json)，不是服务器AI已提交或本地重新执行。记录ID为`EXP-LEVEL1-TMTAFF-UNIT-PREP-20261027-106`，分析ID为`SQ-LEVEL1-TMTAFF-UNIT-PREP-20261027-106`。

固定2400场/7200成员和9600条进度记录完整，10项预设人口全部匹配；T_core eligible155134、选点17932752、双向transition35555236、正对155159，V_select300场/24599轨/9100目标。故2424更新确为batch64的有序一遍，最后62条；这是已核人口导出的未来曝光尺度，不是106已经训练。

正对监督存在缺失：T的8对四属性全空、51对至少一属性空；V分别1/11。所有T/V场的每属性正对池总监督均大于0，但不保证未来每个随机batch可用；全部pair保持，实际batch若任一属性总监督为0，仍按原损失报错并整链失败保全，不重采样或自动重跑。已核统计不认证私有数组逐值、全包单位、reference真值或训练充分性。

**同日增量来源：** 106的Record/Squeeze另回流[Mode C 1.11，§16](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-111)，新增E26（106实际执行证据）及F34（新单位固定人口、正对监督与曝光单位），沿用P01/M01/C01的共同评价基础，不新增机制或选择C08/C09/C10。`followup_source_ids=[MC-RESTART-20261027@1.11, F34, E26, EXP-LEVEL1-TMTAFF-UNIT-PREP-20261027-106, SQ-LEVEL1-TMTAFF-UNIT-PREP-20261027-106]`。上文`source_matrix_id=@1.10`保留本计划首次形成时的真实来源，不能把后来增量伪装成当时已经存在；此次只闭合原已冻结A的启动前置，不事后修改假设或额度。

因此本计划的**106人口前置已满足**；实际A启动仍由独立[L2S-107](../../HandOver/L2S-20261027-107.md)与领取索引控制。方案A用户授权无需重复询问，4000步、额外seed、TAM和新研究方法均未因前置通过获得授权。
