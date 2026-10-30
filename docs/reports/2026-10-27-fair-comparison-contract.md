# 固定056下的TMTAFF同任务比较：实现合同与预算边界

**2026-10-27实现增量：** 下文保留实施前合同；现已新增[固定056评价入口](../../tools/experiments/tmtaff_fixed056_evaluate.py)、[共同控制](../../association/evaluation/fixed056_controls.py)和必要软件检查。入口核对新单位固定TFM/TCM端点、完整056组与分数、原可见量测重放、Q64和完整reference，输出四种普通控制、成对共同mask、38组件及090C位置共同人口；分阶段记录来源曝光和评价成本。真实新单位TFM权重尚不存在，完整V300端到端运行和最终成绩仍未完成，不能把合成检查称为同行比较。§4/§7过去时态的“缺口”保留为开发前基线，不是当前代码状态。当时用户要求先讨论再授权；后续已明确同意106人口核验通过后执行方案A，见§8.3末尾授权追加和正式事前计划，未授权选择新研究方法。

**关联人口口径修正（2026-10-27）：** 独立审查发现最初实现的`inherited_056_association`误用了另一任务的时间窗/轨长筛选，不等于102的全同源/全跨源候选人口。该实现尚未运行真实TMTAFF比较，没有错误实测结果。本次改用`evaluate_scenes`原完整人口，并对056原摘要的all/ITSA/MSTSA各场混淆计数与四种论文指标逐项核对；短窗口径不再混入该字段。旧056原始300场汇总独立复算与102的ITSA 2243583对、MSTSA 2195509对和八项指标一致；新增短轨/零间隔人口与逐场计数篡改回归，相关软件检查45项通过。新接口的真实逐场绑定需实际端点运行时通过，不能把本次软件验证称为V300实跑。

入口示例仅说明参数，**不构成运行授权**：

```bash
python tools/experiments/tmtaff_fixed056_evaluate.py \
  --tfm-run <new-unit-completed-tfm> --source-system <056-system> \
  --reference090c-run <090c-run> --archive <level1-part-1> <level1-part-2> <level1-part-3> \
  --manifest <level1-manifest> --alignment <level1-alignment> \
  --output <new-output-dir> --device <explicit-device>
```

090C对照仅比较位置共同有效点；其历史速度/航向沿用旧Mean，四属性论文数值仍见既有102，并标明旧单位解释，不混成新单位训练结果。作者ESD实现、充分训练量和独立泛化仍未知。

软件验收命令（2026-10-27，本地`.venv`，退出0）：

```bash
.venv/bin/python -m unittest tests.evaluation.test_fixed056_controls tests.evaluation.test_complete_evaluation tests.data.test_timed_speed_units tests.evaluation.test_level1_system_contracts tests.evaluation.test_three_case_chain -q
```

44项通过。新增检查覆盖旧缓存速度复制转换不修改原数组、cap128与canonical_all分离、组顺序/单位拒错、冻结TCM/TFM端点与曝光、验证原生输入检查、固定reference/分数/组绑定、漏目标/重复、预测callback不接收reference，以及合成小样本从CLI到控制/共同mask/090C对照的连接。CLI小样本在文件来源/训练checkpoint绑定处使用测试替身，**不是完整真实数据或权重端到端验证**；真实接口仍需新单位端点和服务器原056/090C缓存。开发中发现测试fixture的错误字段和不完整训练配置并修正，未改变真实研究数据或训练记录。

2026-10-27。本文核查现有代码，给出下一次实现必须满足的接口与评价对象；**没有运行训练、准备全量数据、打开T_test或发布训练READY**。已有接口为DE，最小实现设计为AI，充分训练量与作者ESD实现为UN。当前执行状态只读[交接](../../HandOver/Handover.md)。

来源：Mode C 1.6 的P01/P02、F26–F29及E21/E22；[102/103实测](2026-10-27-returned-diagnostics-review.md)、[速度单位验收](2026-10-27-tmtaff-speed-unit-verification.md)、[TMTAFF维护契约](../specs/tmtaff-reference.md)、[数据约定](../data_contract.md)与[论文指标](../mtad_data_protocols.md)。本文件是比较实现合同，不是已选创新Idea，也不把软件通过算科研收益。

## 1. 要回答的比较问题

先固定056的final1000预测组，比较给定相同组成员时的**恢复与融合输出**。这一层TMTAFF不执行TAM、不开新阈值；ITSA/MSTSA列继承056并标明“固定关联”，不能说TMTAFF改进了关联。所有方法仍保留错误组、单例、漏目标和重复输出。

待恢复/融合已有可信表现，才另立自己的关联—补全—融合完整系统任务。完整系统的组、阈值、覆盖和错误传播重新报告，不能把固定056结果与该层结果拼成一行。090C只作已有内部强控制，历史训练量与速度/航向来源披露；它不限制候选架构。

本次名称统一为“**TMTAFF同任务时间适配**”。论文的Level、名单、SL64物理窗口、截短分布、80 epochs阶段分配和mESD实际程序未公开；用户没有作者代码。即使同一张16项指标表齐全，也不称精确作者复现，不按论文数值直接判断本项目胜负。

## 2. 固定人口、单位与输出

| 项目 | 固定内容与可核验条件 |
|---|---|
| 数据与划分 | Level1现manifest，T_core 2100、V_select 300；scene/component顺序按manifest；T_test 600封存。V反复开发、共享A身份已知，报告仅为开发比较 |
| 056来源 | `LEVEL1-ASSOCIATION-20261022-056-attempt-1/system`及其原`sibling association/final-scores.npz`、final1000阈值；不选新checkpoint、不改阈值 |
| 分组 | 原`anchor_direct/input`，逐场和056保存的`learned_assoc_group_offsets/group_node_indices/membership`完全核对；枚举既定分组规则仅作一致性检查，不重新拟合或修组 |
| 节点与候选 | 完整原node顺序、source、intervals、全i<j候选及标签一致。身份只核人口/监督/事后目标绑定，不进入前向、配准筛选或query选择 |
| Query与reference | 同一个真实秒Q64、原reference offset/gap600、`shortest_arc`、完整目标registry和各属性valid；目标按`anchor_identity_first`绑定。相同目标/真值数组逐项比较，不按误差重新匹配 |
| 单位 | 新TMTAFF输入和物理输出速度m/s；量测kn仅转换一次，reference保持m/s；T内重建normalizer、从头训练新权重、重建新TCM的latent cache，禁止加载旧速度训练的权重/normalizer/cache或靠缩放权重冒充修复。论文速度分别把预测和reference转kn；历史090C的速度/航向仍标来自旧Mean，不能改名为单位修复后学习结果 |
| 单例与递归 | 保留维护契约中的`observed`单例旁路、`fc_memory`递归、input顺序；单例支撑外missing保留。不要为了凑覆盖，把OLS临时塞进TMTAFF单例并称原方法 |
| 统计人口 | 全声明目标/缺预测/缺真值/重复输出先报；误差同时报各自有效人口及每一成对比较的共同有效人口。不同mask的MAE不直接作胜负依据；错误不随重复输出增加目标数 |

056历史人口（V300的9100目标、9353组、610重复等）是回执核验线索，实际绑定仍核数组与原记录；不靠这些总数替代逐场检查，也不设Git SHA或文件hash准入。

## 3. 可见观测与普通控制不能混为一谈

合法原生量测范围、Q64、分组和reference相同，不等于各方法实际消费的观测点数相同：TMTAFF编码器最多128点，090C及当前普通OLS使用`canonical_all`。比较必须同时列出以下两类控制，且不能用同一个“Mean”名字掩盖策略差异。

| 控制 | 位置与其他状态规则 | 用途 |
|---|---|---|
| `canonical_all_anchor_ols` | 全原生点；支撑内分段插值，支撑外以实测端点锚定的全轨OLS速度外推；速度/航向插值并在区间外hold | 与056/090C相同可见范围的强普通控制；“Anchor”就是单轨OLS控制，勿另列一行相同“OLS”冒充独立方法 |
| `canonical_all_mean_supported_ols` | 支撑内对实际有支撑的成员等权；组无人有支撑时回到该输出anchor的OLS。经度/航向圆周均值，歧义沿用既有anchor回退并计数 | 固定056下与现有Mean一致的普通融合控制 |
| `selected_cap128_anchor_ols` / `selected_cap128_mean_supported_ols` | 相同公式，只使用TMTAFF实际输入的选点；速度已为m/s | 分离“用更多原始点”与学习机制差异，不作为降低强控制标准的替代 |
| 现有`predicted_anchor` / `predicted_mean` | 仅cap128支撑内插值，无支撑外预测；圆周歧义missing | 保留为支撑内诊断，不能当全query的共同OLS控制 |

最小实现可直接读取056冻结的`native_states/support/nearest_seconds/query_seconds`，复制速度列乘1852/3600，再调用`ordinary_fusion`；其位置、支撑、OLS参数和分组不重算。reference独立从既定评价artifact读取，绝不能供控制函数前向。该路径不需要放开`load_timed_dataset`当前对`speed_unit_policy + motion/native_events`的拒绝。

同时从新TMTAFF的`native_values/native_times/native_observed`生成cap128控制。常规控制都经同一个`evaluate_full_scene`绑定，输出单位、目标、分组和支撑字段显式记录。若未来候选消费全原生点，它与TMTAFF的表只能支持“各自已披露输入策略下的同任务效果”；具体模块净贡献需候选自己的同输入A/A′/A′+B，而非把cap差异归给模块。

## 4. 已有接口与最小缺口

| 能力 | 现有证据 | 需做的具体改动 |
|---|---|---|
| 物理速度贯通 | `physical_units.py`、timed loader、normalizer、TCM/TFM参数标记、评价和NPZ已做必要软件检查 | 生成新运行配置并从头训练；不得缩放旧checkpoint充当新单位训练 |
| 固定分数评价 | `evaluate_complete(...association_records=..., select_on_validation=False, threshold=...)`逐场核scene/candidate字段且跳过TAM前向 | 绑定056原final-scores和阈值，核原保存组及完整V；结果写明`tam_scoring_forwards=0` |
| 固定TCM/TFM端点 | `tmtaff_complete_fit.py --phase tcm/tfm`、`--initialize-tcm-from`、固定TFM监测点/末步及真实曝光日志已存在 | 增加一个评价wrapper读取新TFM末步与原056缓存；无需修改训练阶段来选择关联 |
| 当前TFM评测 | `train_complete.evaluate_tfm_endpoint`强制`oracle_only=True`；现093类输出只是oracle分量诊断 | wrapper额外产出固定056预测组结果；不能把oracle终评改标签冒充预测组 |
| 旧096组合脚本 | [completion_system_evaluate.py（历史源码）](https://github.com/HKillua/association-code/blob/cba60543a329e1c67007217101b9ae5bdf39f082/tools/experiments/completion_system_evaluate.py)严格绑定093/094及原completion-policy合同 | 不直接改参数冒用096；提取必要绑定/评价逻辑即可，不继承旧source资格或旧训练额度 |
| canonical_all共同控制 | `ordinary_fusion`已有；056缓存已有；新单位timed loader会丢弃raw tracks并拒绝motion混用 | 独立复制056控制速度列，保留原数组；添加cap128可见性匹配控制；禁止原地改缓存 |
| 公平比较输出 | `evaluate_full_scene`、`ScenePredictionWriter`、论文指标聚合、102补算程序已有 | 对固定056各臂做逐属性共同mask、全人口、P95/P99、38组件、支撑/距离分层及成本汇总，不重复模型前向来补指标 |

最小wrapper接口应为：`--tfm-run <new-unit completed tfm>`、`--source-system <original056/system>`、`--manifest`、`--alignment`、`--archive <Level1三卷>`、`--output <new attempt>`、显式`--device`、CPU4。绑定TFM的真实配置/末步/normalizer和原056缓存，再验证新准备输入人口、Q和reference；没有`--best`、新阈值或挑场参数。当前尚无这个新CLI，因此本合同不能宣称端到端命令已可运行。

**必要软件验收仅四组：** (1) 固定056缓存→完整分组和目标绑定不变，打乱node/pair/order必须报错；(2) 控制速度仅一次转换，原缓存不变，位置/航向/支撑不变；(3) cap128与canonical_all控制在有意不同的观测下确实分开，前向不可读取reference；(4) 同一小样本端点评价包含错误组/单例/无参考/重复/缺预测，完整与共同mask指标可独立复算。合成fixture只验证这些接线，不认证同行效果；不追加模拟训练资格链。

## 5. 训练量与端点必须按曝光记录

当前已知旧088是一遍155134条合法轨，batch64共2424更新；末批62条，不是`2424×64`条真实曝光。`ordered_fragment`的完整epoch为`ceil(Neligible/64)`步，轨曝光为`Neligible`，双向原生transition曝光为`2∑(n_selected−1)`。新单位路径若节点与mask不变，应复核这些数，但不能用总更新数替代实际曝光。

TFM当前采样是“等概率选一个有正对的场景，再在该场选正pair”，不足batch时有放回；不是全库pair均匀。1000×64=64000次pair呈现，不是64000个独立pair，也不等于完整epoch。必须保留unique场/轨/pair、总pair、有效逐属性query监督和替换次数，TCM预训练与TFM成本分开加总。`--pairing-audit`能保存实际有序draw及有效监督；其不改变训练目标。

上述2424/1000是已知历史量，**不是新训练预算**。单位改变后输入、normalizer和实际损失尺度改变，旧额度不自动足够。本合同冻结固定末步、不按V选最好点、预先列全seed的规则；具体TCM epoch数/更新数、TFM pair曝光和学习率须在新Idea中定值。实现准备时先核新单位路径的eligible轨、pair池、有效监督和预期成本，再写整数预算；不能为“可执行”硬编一个充分训练量。公开材料也不能告诉我们“80 epochs”应怎样分配给各阶段。

新任务应把受限适配可行性与后续正式比较分清。若首个预定端点在同人口下仍远逊普通控制，须依据位置/速度/航向损失、梯度与支撑分层证据判断适配哪里失效，不自动加epoch/seed，不能用这个弱适配宣布候选胜过TMTAFF。若普通控制可用性建立，正式比较仍须独立冻结总TCM/TFM曝光、seed列表、验证频率、固定端点、停止条件与成本上限。最终判断须按预定端点的全人口覆盖、MAE/RMSE/尾部作出；不能事后为一个正指标补晋级门槛。

候选A/A′/A′+B可以配平自身的采样序列、有效监督、更新数、seed和输入；不同架构的TMTAFF参考不强行声称相同步数=同预算。参考主表报告TCM/TFM各阶段总曝光和GPU时，另外给固定计算额度的比较视图；只在实际可匹配的量上称公平。小候选训练4000步与TMTAFF含TCM的1000步不具可比总成本。

## 6. 输出与去留解释

每次固定端点评价至少保留：原真实命令/退出、真实配置、checkpoint末步、normalizer及单位、056缓存来源/阈值/组、每场原预测/真值valid mask、选定目标映射、全部V300场与38组件、参数量/训练和推理耗时/峰值内存。每个臂报告16项论文维度，其中8项关联重复继承056而非本臂提升；融合经纬10⁻⁴度、速度kn、航向度的mME和**标准残差SD**，作者mESD另标未知。并报米制位置MAE/RMSE/P95/P99、坐标MAE、速度m/s与kn误差、航向原始/圆周误差、覆盖与重复。

主要比较是固定组位置恢复：完整人口先看覆盖，成对共同有效点看位置MAE与RMSE；同时给P95/P99和最大130场组件外的结果。MAE变好而RMSE恶化必须并列，不能用平均抵消103所见的极尾代价。对场景和38共享身份组件的配对差异可做描述性区间，但不能把query当独立样本、把组件bootstrap称独立测试。首个单seed可行性不支持稳定优势。

执行失败、非有限、输入/单位/组/人口绑定不符或reference影响前向即终止并保留原件；一次尝试，不自动重跑。按服务器规则墙时为建议，只继续原固定更新，不扩额度。固定终点后无正证据也完整回传，不以表现选择输出行或换checkpoint；不删难点、不从oracle挑对应、不将缺预测错误写零。是否进入后续训练的当前决定只放研究状态及独立READY合同。

## 7. 本次核查结论

DE：物理单位链、固定TCM/TFM端点、缓存分数API、普通控制和完整人口评价都有可复用实现。DE：056绑定wrapper、两种可见性控制和共同mask汇总尚未贯通；当前phase=tfm是oracle评价，096 wrapper绑定其他任务。UN：充分训练量、论文未公开预处理及ESD实际公式仍未知。

因此最小工程路径明确为“新单位TCM/TFM端点 + 固定056评价wrapper + 缓存canonical_all控制/新cap128控制 + 同一人口汇总”。本轮核查没有生成科研性能结论；后续代码完成不自动授权训练，也不阻止依据新证据重选候选。

## 8. 新单位训练预算准备：人口、曝光、成本与待讨论方案

**同日软件复核订正：** 固定056比较中，完整及汇总速度数值已是m/s，但`common_target_metrics`的逐场共同有效人口报告仍沿默认标为`native_speed`。现保留输入artifact的显式速度单位，并拒绝两臂单位/策略不一致；仅修报告声明，不再次换算数值，旧无单位artifact仍保留历史标签。相关26项软件检查通过，不涉及任何真实预测或模型训练，也不改变既有科学结果。

2026-10-27。本节是**基线实施和预算草案**，不是选择新研究方法、训练Idea、服务器READY或任何新实验结果。用户同意先准备比较预算；本文不启动训练，不预先选择C08/C09/C10。以下历史资料均为旧单位路径与已完成端点：从`git show f7923eb:HandOver/S2L-20261024-088.md`及同提交的088 `run.json/coverage.json.gz`，和`git show d15b92a:HandOver/S2L-20261024-093A.md`及093A `completion-value-verification.json`、`run.json`提取；不将旧结果冒充新单位结果。共同评价入口按本文开头最新修正版。

### 8.1 已核人口与新单位仍需确认的地方

| 对象 | 旧Level1实际记录（DE） | 对新单位的约束和未知 |
|---|---|---|
| TCM训练 | T_core 2100场、155134可用片段；已选原生点17932752；双向相邻transition呈现35555236 | 旧088有序1遍，batch64共2424更新，最后62条。速度乘正系数而不改时间/标签/选择规则，故数量**预期不变**；新准备输入必须核实，不把预期写成新运行事实 |
| 候选对 | 训练总27632584对，正155159、负27477425；2100场都有正对 | 正对数、场景清洗/隔离及reference有效mask须以新单位准备结果核对；不把155134轨当155159正对或二者当独立船只 |
| 正对池分布 | 2100场正对数最小6、中位30、最大1773；1542场不足batch64 | 按现实现，先等概率选正对可用场，然后该场取64对；不足64时有放回。不是全库正对均匀抽样，场间/目标间概率不同 |
| V评价 | 300场、24599片段、4439092候选对；声明目标9100，38共享身份组件 | 反复开发、跨划分共享身份；必须完整人口、重复和缺失点先报，不能当独立测试或把query当独立样本 |

按旧088各场实际正对池及当前采样代码的**解析期望**（不是新训练实测）：1000步的64000次正对呈现，期望访问约796场、28453个不同正对（约18.34%）；4000步的256000次呈现，期望访问约1788场、76491个不同正对（约49.30%）。假设每步独立等概率抽取上述2100场，选中场内`n<64`时按64次有放回抽样，否则无放回，某对单步被选概率分别为`[1-(1-1/n)^64]/2100`或`64/(2100*n)`；累加`n*[1-(1-p)^U]`。实现见[TFM采样](../../association/training/tmtaff.py)；期望不是实际覆盖、独立观测数或泛化证据。旧093A实际1000步有29002个不同正对（18.69%），与该期望仅作数量级对照，不能推断新单位采样轨迹。

**最低准备检查，不训练：** 用明确新`speed_unit_policy=measurement_kn_reference_mps_v1`的临时完整配置，运行现有`tmtaff_complete_fit.py --phase tcm --inspect-only`，读取固定T_core/V_select并生成scene准备报告；0模型构造/0GPU/0优化/不碰T_test。汇总T/V场数、节点/选中原生点、逐场正对/负对池、目标及逐属性有效参考，比较旧数据目录、隔离结果和预期155134/155159/17932752。每个正对可用监督的实际query分布，单靠scene汇总不能精确推导，须在准备数组中按目标与正对记录或在未来`--pairing-audit`中按实际抽样记录，不能用目标有效点冒充正对训练监督。准备是对全部固定T/V场景读取，不是三场诊断；ZIP成员数量、耗时、峰值内存记录于新输出，不能事先保证只读少量数据或沿用新GPU训练成本。此处只是指出已有入口；是否发布单独准备任务由主仓交接决定。

### 8.2 可供讨论的固定单seed候选预算，不是训练许可

两个方案都先完成上述新单位人口核验，使用固定Q64、原生cap128、`shortest_arc`、T_core训练/V_select开发、`measurement_kn_reference_mps_v1`，不可加载旧单位normalizer、checkpoint或缓存。保持两阶段分别从头：TCM fresh初始化，TFM仅从该新TCM转移冻结completion、使用fresh TFM和optimizer；不训练TAM，不以oracle结果代替固定056预测组结果。最终固定末步，u0/末步监控只是诊断，不按V挑更优模型；0/1000或0/4000按选定方案**开跑前**配置。seed20、batch64、microbatch/模型结构、常数学习率0.002、梯度clip1.0、损失与采样采用当前对应[TCM](../../configs/tmtaff-tcm-level1.json)/[TFM](../../configs/tmtaff-tfm-level1.json)的历史配置，仅加入新单位及单独明确的阶段更新/监测点；**这些是为检验单位纠正后能否形成可用基线的预备选项，不是充分训练或精确作者复现的依据**。若检查与预期人口不符，停止并先分析，不改batch/任务或临时按V选预算。

| 可审选择 | TCM建议 | TFM建议 | 用途／直接限制 |
|---|---|---|---|
| A：重做已知曝光的单位可行性诊断 | 1个有序epoch；若155134轨确认，2424更新，batch64尾62 | 1000更新×64正对＝64000呈现；seed20，固定0/1000 | 与旧088+093A的曝光量可直接对照新单位影响；旧093A这一预算oracle位置MAE约49.6km，不能拿它当充分同行，也不能将旧失败直接归因单位 |
| B：增加正对场景覆盖的候选 | 同一1个有序epoch | 4000更新×64正对＝256000呈现；seed20，固定0/4000 | 解析期望覆盖更多池，但梯度尺度已变、训练时间显著增加，仍不能保证收敛或可信同行；这不是A跑完后的自动追加或resume，须先定预算并独立合同 |

两者的1个TCM epoch和1000/4000 TFM更新只是具体可复核的**初始候选**，不是推断出的最佳预算；用户要求充分对比，只有单seed不足讨论稳定优势。若先做A，其结果无论好坏都不能改变原终点或据此自动发B，若进行多seed则未来须事先固定seed列表、完整训练/选择规则与成本；优先考察误差、完整覆盖、尾部和最大身份组件外，而不是用损失下降宣布有效。若方法只在少数子集好而整体、其他属性或尾部恶化，完整并报；若固定端点明显差于合理普通控制，记录具体失败、检查实现与失配后停止该配方，不将其冒充可信同行胜负。

**历史资源事实和不确定性：** 088旧单位单卡3090 TCM launcher 10315.590s（2h51m56s），程序10268.729s，其中固定V特征评价7036.270s、2424次训练步骤累计约1924.3s；两者相减余约1308s，不是088原回执声称的3344s，且记录的计时边界可能重叠，余数不能直接称纯准备耗时。旧093A单卡3090 TFM1000步launcher 62226.770s（17h17m），训练步骤累计30989.5s、u0评估14123.9s、u1000评估15715.4s；含准备/缓存/评价且旧8h advisory已超时但固定任务继续。旧088+093A顺序执行墙时之和约20h09m，**不包括新固定056评价**；新单位、共享设备/缓存等会改变时间，不能当保证。若仅为规划以旧093A训练步均速线性外推4000步约34.4h训练，另加旧两次端点评价约8.3h和准备；这是未验证的数量级假设而非实测或授权预算。旧088进程peak allocated约1.85GiB、reserved约3.28GiB，旧093A约1.38GiB allocated；仅所选进程PyTorch allocator、不是整卡可用显存。服务器按已有FIFO、最多四单卡、不触碰他人作业，资源实时状况另核。

**固定停止与回传规则：** 0新增阈值选取/0T_test、无源代码伪装；输入/单位/组绑定不符、异常终止/非有限梯度/OOM、数据缺件或用户停止时，保留完整日志和失败状态，不自动重试、缩batch、删样本或加步。墙时为advisory，超建议时间仅记录并继续已冻结的同一次额度；不能将8h写成自动硬终止或借超时扩额度。u0、固定末步以及固定056预测组的完整人口和普通控制均需记录；新训练曝光按成功更新、唯一场/轨/对、对呈现、每属性有效监督、TCM/TFM各自GPU时/峰值分别记录，训练期`--pairing-audit`记录有序对抽样但不能证明原始训练浮点逐值重放。执行以后才有Record/Squeeze/Mode C；当前无新模型效果记录。

设计参照[experimental-design技能](/Users/a1/.codex/skills/experimental-design/SKILL.md)中的实验单位、重复和避免伪重复原则；技能所属参考：Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, 2026, [arXiv:2609.00065](https://arxiv.org/abs/2609.00065)。它是研究流程工具来源，不增加本项目目标领域核心论文数量或作者复现证据。

### 8.3 待用户选择的具体A预算：新单位基线可行性诊断草案

**阅读时序：本节标题、下列DRAFT及“待用户决定”措辞保留获授权前的草案历史；用户后来已明确选择A，条件及事前计划见本节末尾追加说明。** 本节仍不是服务器领取入口，106最终人口核验不得跳过。

**原草案：DRAFT，未获训练授权，不是READY。** 本节把§8.2方案A落实为可审配置和命令形状，便于讨论计算投入；不选择新的研究机制，不替代后续事前Idea/独立交接，也不启动任何作业。对应[TCM草案](../../configs/tmtaff-tcm-unit-draft.json)和[TFM草案](../../configs/tmtaff-tfm-unit-draft.json)可被现有配置解析器读取，但文件存在不提供执行许可。当前训练权重不存在；106的最终人口核验、用户对预算的明确选择和后续独立执行合同均不得跳过。

**要区分的三种差距：** （1）旧093A与新单位TFM的u0/u1000 **oracle**结果，是相近结构、损失和曝光下的修复前后诊断，用于观察原49.6km级失败是否仍存在；（2）新TFM固定末步在**056原预测组、全V300**上与普通控制及090C位置的比较，是本合同定义的同任务比较，包含错组、单例、缺预测和重复；（3）TMTAFF**论文原表**只作外部参照，因作者划分/窗口/阶段80 epochs分配及mESD实现未闭合，不能把其有符号ME与本项目米制MAE相减或计算提升率。三层分开列，不训练TAM，也不称完整TMTAFF系统复现。

| 供决定的项目 | 草案A固定值与约束 |
|---|---|
| 目的与预测 | 检验从头纠正物理速度路径后，这个有限预算的时间适配能否产出可用的恢复结果。若oracle仍严重落后普通恢复控制，原失败不能仅以单位问题解释；即便改善也不能据此确定单位是唯一原因 |
| 数据与确认条件 | 固定Level1 T2100/V300、Q64、cap128、reference gap600/shortest_arc、原056及090C来源；T_test不读。仅在106最终确认T可用轨155134、既定人口/选点和监督无未解释变化后，2424才代表完整1个有序epoch；不符则先分析，不默默改步数 |
| TCM | 从头初始化模型及T内normalizer；`ordered_fragment`、batch64、2424成功更新，尾批62条；预计曝光155134轨及35555236双向transition，实际数单独记录。只取固定末步，不按V选择 |
| TFM | completion仅转移上述新单位TCM并冻结；新TFM及optimizer，从头1000成功更新、batch64＝64000次正对呈现；seed20。仍按场均匀、场内抽正对，不足64有放回；记录实际unique场/轨/对及逐属性有效监督，64000不是独立样本或完整epoch |
| 结构与优化 | 两份草案的model/fusion/training/train_splits逐项等于对应维护中的历史配置。128维、8头、6层、dropout0.1；TCM track_microbatch4、TFM pair_microbatch8；常数LR0.002、clip1、现有`standard_smooth_l1`及`shortest_arc`损失、`train_position_affine`输出；不另调超参、loss权重或采样 |
| 端点与监测 | TCM2424末步；TFM仅u0及u1000的完整V oracle监测并保存预测，**固定使用u1000**。完成后额外做一次固定056全V评价及全部声明控制；不依据u0/u1000优劣换checkpoint，不新增阈值 |
| 评价与统计 | 本文§6全部指标与人口报告；先覆盖/重复，再同一成对有效mask的MAE、RMSE、P95/P99，逐属性ME/标准残差SD，38共享身份组件及最大组件外。seed只有20，反复使用的V300是开发集；不声称独立泛化或稳定优势 |
| 终止与出口 | 输入/单位/组/人口不符、数据缺件、非有限梯度、OOM或异常即停并留原件；不自动重试、缩batch、resume、追加seed/步数。正常结束后即使无改善也完整回传，依据失败层级再讨论；不自动转B或换研究方法 |
| 计算投入 | 旧088＋093A单卡3090顺序墙时约20h09m，**另加尚未实测的新固定056全V评价成本**；新单位准备/缓存/共享负载可能不同，新总成本未知。配置8h是各阶段advisory而非8h硬预算，继续的只是预定2424/1000步；不是授权无限训练。一个TCM、一个TFM和一次固定组评价各一次尝试，按服务器FIFO安排，CPU4/显式单卡 |

**配置差异和因果边界：** 两份draft相对各自当前`*-level1.json`只增加`data.speed_unit_policy=measurement_kn_reference_mps_v1`，其他显式字段无改动。这使输入量测kn转换m/s、reference保持m/s；T内重新估计normalizer、新TCM latent缓存和新权重均重新生成，不能加载旧单位权重再缩放输出。与真正历史088/093A运行相比，还存在当前单位标记/评价修订、额外`--pairing-audit`记录、固定056 wrapper与全人口/共同mask统计等工程和统计变化，运行环境及浮点轨迹也未声明完全重放；**不是只拨动一个开关的纯因果单位干预**。新旧oracle指标须核相同scene/target/query/valid人口后解释；固定056结果是新比较层，不冒充旧093A已有的预测组结果。保留相近预算是为定位基线可行性，不是为了认定旧预算充分，也不是作者80 epochs复现。

以下为**不可直接复制执行的审阅模板**：尖括号是尚未在训练交接中冻结的来源/新目录/设备占位符；不得自行替换后开跑。实际路径、运行环境、106证据、用户选定A及新Idea/停止条款须进入以后独立合同。本轮不创建交接编号、不改领取表。

```text
python tools/experiments/tmtaff_complete_fit.py \
  --phase tcm --config configs/tmtaff-tcm-unit-draft.json \
  --archive <level1-part-1> <level1-part-2> <level1-part-3> \
  --manifest <frozen-level1-manifest> --alignment <frozen-level1-alignment> \
  --output <new-unit-tcm-attempt> --device <explicit-single-device> \
  --cpu-threads 4 --time-policy advisory

python tools/experiments/tmtaff_complete_fit.py \
  --phase tfm --config configs/tmtaff-tfm-unit-draft.json \
  --initialize-tcm-from <new-unit-completed-tcm-attempt> \
  --archive <same-level1-part-1> <same-level1-part-2> <same-level1-part-3> \
  --manifest <same-manifest> --alignment <same-alignment> \
  --output <new-unit-tfm-attempt> --device <explicit-single-device> \
  --cpu-threads 4 --time-policy advisory --pairing-audit --save-predictions

python tools/experiments/tmtaff_fixed056_evaluate.py \
  --tfm-run <new-unit-completed-tfm-attempt> \
  --source-system <original056-system> --reference090c-run <original090c-run> \
  --archive <same-level1-part-1> <same-level1-part-2> <same-level1-part-3> \
  --manifest <same-manifest> --alignment <same-alignment> \
  --output <new-fixed056-evaluation-attempt> --device <explicit-single-device> \
  --cpu-threads 4
```

命令参数核对现有[训练入口](../../tools/experiments/tmtaff_complete_fit.py)及[评价入口](../../tools/experiments/tmtaff_fixed056_evaluate.py)；每个输出必须是新目录，TFM来源必须是本次完整TCM，评价来源必须是本次完整TFM末步。`--reference090c-run`是实际必需参数；TAM阶段与`--resume`均不在草案中。此处未执行这三条命令，未构建模型、读取原数据或消耗GPU预算。

**需要用户决定的具体事项：** 是否选择“单seed20、TCM1遍＋TFM1000步＋0/末步oracle＋一次固定056全V”的A诊断，接受历史约20小时加新增评价且新成本未知。也可选择暂不训练，先讨论不同预算或目标；不能把“继续准备”、此前106授权或本草案被提交等同于选择A。选择后先完成正式事前记录和独立交接，再按该固定额度执行；结果回来后按Record→Squeeze→Mode C回流讨论。

**同日独立软件核验补记：** 两份草案均通过模型/融合配置数据类、训练`_configuration`及输入`_settings`解析；逐字段核对仅新增速度policy。独立提取真实CLI的argparse声明段，三个完整模板的必需参数和阶段选项解析通过，未加载数据/模型/设备或执行训练。这只确认配置与接口，不是实际端到端验证。

**同日用户决定追加（原草案之后、训练之前）：** 用户已明确同意在106人口核验通过后执行上述固定A：seed20，TCM eligible155134时有序一遍2424步，fresh新单位TCM/normalizer，fresh TFM1000×64，oracle仅u0/u1000且固定末步，以及一次固定056全V和声明控制。正式事前包为[IDEA-TMTAFF-UNIT-BASELINE-A-20261027，plan1.0](../proposals/tmtaff_unit_baseline_A_20261027.md)。因此原文“未获授权/待选择”不再是当前授权状态，也不需要再次索取同一预算许可；**106真实终态和独立人口/监督核验仍须先完成**，本段不凭进度宣布通过。预算、失败停止、历史约20h加新增评价成本未知等约束不变；4000步、额外seed、TAM、完整系统或C08/C09/C10新方法均不在此授权内。实际路径/设备/新目录及领取只由独立交接具体绑定，当前本文件的编写不等于已启动。

## 9. 2026-10-30：三种同行共同任务比较的后继设计

本节记录用户要求优先落实TMTAFF/PTFM/DeepAF公平比较后的设计，**不是训练READY或已经完成的实验**。优先级只在[研究状态](../research_status.md)维护，发布/领取只在HandOver。§1–8是原107/113合同，保持原A0分组和终态事实；不重训、续跑或覆盖107/113，不将旧074C/099配置当作新运行许可。

### 9.1 比较对象与共同条件

- **先比较融合，再比较系统。** 第一层共用056分数生成的冻结WS预测组，保留错误组、单例、漏目标和重复；这是比较条件，不规定最终骨架。旧A0只作历史桥接，不与WS新结果混排。第二层才比较实际系统：TMTAFF可用其TAM；PTFM本来只做已关联融合，记作“共同前端/WS＋PTFM”；DeepAF轨迹输入适配也注明实际前端，不能伪造全片段关联模块。
- **同一数据、时间和信息权限。** Level1 T2100/V300、Q64、offset/gap、目标对应及逐属性valid保持，T_test600封存；normalizer只拟合T。共享身份和反复使用开发V的限制照报。模型保留各自内部时间表示/补全/递推，披露实际点数/cap策略，提供selected_cap128匹配普通控制及canonical_all强控制，不把更多观测的效果全算网络贡献。改变输入政策须作为新版本训练，不能临时换旧checkpoint输入。
- **同一物理单位。** 新四状态输入速度为m/s，量测kn只转一次、reference按既定证据解释为m/s，原文件不改；论文速度另转kn。DeepAF内部P/V不等于对外四属性；速度/航向沿用Mean时标明。旧权重、normalizer和cache不得混入新单位训练。
- **全输出域和原适用域分别报告。** 全Q64保留缺预测/覆盖/全目标计数，共同有效点误差另列。双源共同支撑可作PTFM等的原适用条件补充，但所有方法用同一事前子域。若补公共回退，须事先固定，并将“原核心＋公共回退”整链另命名，不能偷偷补齐后称原方法。
- **我们的候选也到同一V300。** Qhard/146的V32不能和同行V300排名；先冻结实际组合，再评价原32和其余268。不拼模块冠军，不以固定源优先控制替代通用方法。

### 9.2 已核代码差异与修订对象

以下是实现边界，不是已证明的性能根因；修订前按`spec-driven-coding`写行为规格和必要检查，新入口/目录独立发布。

| 方法 | 可复用能力 | 必须核清/修订的具体差异 | 首个真实判断 |
|---|---|---|---|
| TMTAFF | 完整AP、`tmtaff-reference.md`、107训练及113原数组/记录；TCM/TFM阶段与显式速度单位入口 | 物理输出恢复、normalizer/输出affine、监督/多任务尺度、teacher/递归和单源输出域；现fixed056 wrapper锁A0，不能换WS名字冒用；原有限训练不等于充分训练 | 先用既有T日志/预测或固定端点作T侧定位，分有支撑双源/长缺口；缺少T缓存如实记，不造已有产物。修订有依据后另立新训练，原107不动 |
| PTFM | 原20页完整AP；双分支增强、特征门控/LSTM和任务loss核心 | `ptfm_inputs`现为anchor/helper汇总而非原两传感器分支；旧loader/normalizer未形成新单位四状态共同合同；`ptfm_fit.py`硬绑074C/历史4000步配对来源 | 建立明确的双源分支适配、来源换序报告与缺失策略，再用真实T核物理目标/loss/恢复；helper汇总版不能改名作者原版 |
| DeepAF | 原P/V核心和真实时间/局部米制转换存在 | `deepaf_peer_fit.py`硬绑099及旧072/090抽样链；TBPTT1、desired increment/dt监督、初始化/虚拟候选和离线权限均须对照原文与当前任务 | 主比较用原始P/V版本，不用TMTAFF四Transformer改版代替；真实T核自身历史闭环、监督语义、dt和无观测行为，再冻结新训练入口 |

依据：[TMTAFF完整复读](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/tmtaff_reaudit_20261027.md)、[PTFM完整复读](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/ptfm_reaudit_20261027.md)、[DeepAF实现审查](2026-10-25-deepaf-method-audit.md)。本轮定向读源码/config和真实记录，无新增全文精读/科学试验。2026-10-30公开检索仍未取得作者完整训练代码，部分出版社页面429；不据此断言代码不存在。已有全文与独立实现可继续推进，作者未公开名单/代码不是无限等待条件；未解处作有理由、事前固定的适配并公开。

### 9.3 训练、资源与报告边界

先完成接口和真实T侧有界验证，核输入/监督/物理恢复/闭环及训练曝光，不扩建自建模拟资格链，也不设“必须先赢Mean”作为正确性证明。每项T侧具体场景/批次、是否更新参数、预算和终点在新合同发布前固定，本节没有立即可执行的训练命令。

首seed20作可用性比较；实施/适配解释清楚后，正式稳定性比较预定seed20/21/23，全部报告。每模型的整数训练量、学习率/调度、早停或固定端点、验证/选模和调参额度，依据原文、合法样本池及真实吞吐量写入独立合同后才READY。PTFM原150 epoch/20轮早停等是原文依据；TMTAFF阶段分配未知，不硬称还原80 epoch。不同模型相同步数不等于公平预算，分别报告预训练/下游监督曝光、搜索资源和总成本。

按现场最多四单卡：三个同行就绪可各一路，第四路供已具备独立Idea的科学验证；未成熟不占卡等候。CPU拟合/数组统计有界运行。一次尝试、失败保全、不自动重试/加步，原132现场不动。新训练须针对可核差异另立来源/Mode C/适配设计，不是给原失败续训；差结果既不认证胜同行，也不触发无限追数值。最终冻结后再按项目边界处理封存测试，本节不自动开启。

共同位置主表为球面MAE/RMSE/P95/P99、完整人口/覆盖/漏目标/重复及逐对共同人口；速度/航向物理绝对误差另列。论文维度继续ITSA/MSTSA各四项及四属性ME/明确定义SD，raw/decoded/系统分表；ME不当MAE，残差SD不冒充作者未知ESD。逐场/共享组件、去最大组件、支撑层、before/after/距观测时长、>1km新增损害与训练/推理成本保留；缺预测不能从主表消失。

三问题从最新Mode C1.61分别交独立Idea：选源错误研究合法证据能否区分高损害；混合研究实际恢复相容性、细小收益分辨率和可学性；尾部同时保住T-before恢复和强Qhard整体/尾部。方法不在比较合同预定，不重跑143/144/146，不把候选解释写成已证机制。
