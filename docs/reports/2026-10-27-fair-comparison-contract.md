# 固定056下的TMTAFF同任务比较：实现合同与预算边界

**2026-10-27实现增量：** 下文保留实施前合同；现已新增[固定056评价入口](../../tools/experiments/tmtaff_fixed056_evaluate.py)、[共同控制](../../association/evaluation/fixed056_controls.py)和必要软件检查。入口核对新单位固定TFM/TCM端点、完整056组与分数、原可见量测重放、Q64和完整reference，输出四种普通控制、成对共同mask、38组件及090C位置共同人口；分阶段记录来源曝光和评价成本。真实新单位TFM权重尚不存在，完整V300端到端运行和最终成绩仍未完成，不能把合成检查称为同行比较。§4/§7过去时态的“缺口”保留为开发前基线，不是当前代码状态。用户要求在讨论并同意前不选择新思路、不发布训练任务。

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
| 旧096组合脚本 | `completion_system_evaluate.py`严格绑定093/094及原completion-policy合同 | 不直接改参数冒用096；提取必要绑定/评价逻辑即可，不继承旧source资格或旧训练额度 |
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
