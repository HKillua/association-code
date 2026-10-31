# 168–169：关联上下文与按query支持的分离验证

2026-10-31，IDEA_PACKAGE / plan1.0，Mode C1.74先于本设计，承接P05/P13/P14、F262/F265–267、M69/M107/M110、E165/E168–170、C56及C53。用户本轮明确优化覆盖、跨源mPCA/PFA并包含前端；允许执行新有限实验，152活动合同、107/132现场、159强控不改。

## 证据与机制假设

独立[来源AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/association_context_sources_20261031.md)核STAR的节点/边消息、TrackTrack联合候选/初始化与不利消融/失败图/代码差异。原方法缺外观/框/置信度到海上接口，不直接复现，不引一对一约束。STAR对其他视觉tracker集成是有限C经验，完整海上传感器迁移缺失如实记；B=关系条件机制，不是KAN或QP包。

DE：056独立22维pairwise网络，114后置WS仅合并；165同score REF提高32场跨源指标却救5丢2，新增大害全在新0/1支持域。AI分别为①第三节点关系可帮判断模糊边，②搜索输出anchor和query支持被破坏会造成可避免代价。竞争解释包括原score失准/密度捷径/噪声传播、anchor冻结保住原错、相同支持数但状态更差/对应变化；不先写根因。原159的源质量上下文166无效，不等于前端关联上下文已经测过。

## 168：IDEA-ASSOCIATION-EDGE-CONTEXT-20261031

主模式4，C56。A=冻结056；A′=同参数新增OWN残差，A′+B=CTX残差。冻结056原normalizer、22→64→64隐藏及原score；每边h为第二SiLU后64维。w=sigmoid(s−原θ)，不是已校准概率；节点u为邻边h加权均值，读出某边时**从两端聚合中移除该边自身**，无邻边用0。OWN输入[h,h,h]，CTX输入[h,u_i+u_j,|u_i−u_j|]；二者192→32→1、6209参数、末层0，共享初始化seed20。T128每个原pair拟合各自mean/std，clip±8；上下文只运动/score，既不读身份/reference/source名字，也不以源slot选权。维持对称与节点重排等变，软件K≥3仅接口不是异构泛化。

T128固定151名单前128，前32沿127原motion，新增96由151显式mps→kn一次转换，endpoint/OLS mps保持。冻结teacher输出与T125/V056原scores绑定并记最大数值差；原A输出直接沿保存score。hidden不同kernel/批大小最大abs差容许1e−4，超过或原WS成员不符即失败，不放宽继续。全部配对不裁候选，T标签只在特征冻结后读取，V不训练。实际原056已全T训练，此次T32是训练诊断而非LOCO前端验证；159后端仍原T组件留出/full，不能称整链T独立。

两个新头各1000步、batch256（128pos/128neg）、同BalancedScenePairSampler seed20同批，Adam lr.001/wd0/clip1，无早停/恢复/追加seed/加步；共2000optimizer更新/512000pair呈现。BCE用原score+新残差；推理θ固定1.7320103112387182，不V选阈值。原score/teacher隐藏不反传；只新增前端打分残差真实GPU训练，非完整新骨架全T重训。全T模型冻结后一次T32/V32，`WS/OWN/CTX × MEAN/NN`六臂，所有新分区真实重建Mean/source pack，159不重训。新打分与169不叠加挑冠军。

主要结果：raw与WS decoded分别报告cross mPCA↑/mPFA↓、all/cross F1/TP/FP/FN、joint-support分层，全target覆盖率/获失/重复、物理MAE/RMSE/P95/P99、实际新增>1km损害、四属性、场景/组件/去最大和2000次seed20组件描述区间。CTX−OWN作关系条件增量，二者对原A分别报容量/额外曝光效果。损失下降不证明机制；若只提高mPCA却加误报/丢目标或伤尾部，不晋级为联合升级；未稳定超OWN则关闭此配方，不追加训练。32场不冒称300场/论文胜负。

CPU0–3×4、GPU0兼容余量≥6GiB、RSS8GiB、3600s advisory；新checkout/run/launcher attempt1，最多四个自己单卡含152A/B；无SSH值守/他人进程操作。保存预测/score/membership先于V真值，私有数组/权重留server，所有公共逐场/组件/属性/曲线/训练量/实际命令/退出/成本回传。

## 169：IDEA-ANCHOR-QUERY-SUPPORT-20261031

主模式5，C53×P13。固定056分数/θ、159、原T32/V32，0训练/teacher/CSV/test。四分区×Mean/NN八臂：WS、原165 REF、ANCHOR、SUPPORT。WS/REF绑定165输出；ANCHOR从WS出发，只移动非原anchor到另一原组，目的组原anchor须仍为最小节点，不生新组/删原组。这样隔离“原输出对象代表保持”条件，仍可能保错组、不能修回原来没有的anchor。

SUPPORT同ANCHOR搜索与4N次正ΔJ预算，但要求每次move后，受影响两组在**每个query可用的不同来源个数均不低于move前**。只读node索引/来源相等关系/support，不读名字排序、参考或身份。每个组的来源身份可变，不按特定源优先；对所有query逐项检查，不以平均支持抵消损失。ANCHOR−REF拆输出代表限制，SUPPORT−ANCHOR拆观测数约束；WS强控保留。ΔJ>1e−10，tie按原输入索引，记录拒绝候选数量、J/每次move及停止预算；只在合法候选中找最佳，不是先找最佳失败就停。

可检验预测：ANCHOR应保持原输出代表集合；SUPPORT在原query网格保持原组支持源数下界；是否减少实际>1km新害并保关联/mPCA/PFA要看真数据，不能从约束推导物理保证。仍保所有节点一次，原组/anchor数固定、无truth修组，覆盖/重复必须按完整官方anchor_identity_first重算。无候选/全未变是有效负结果，不改约束续跑；不按V设支持阈值/时长cap。四属性、组件/覆盖获失/全人口与共同域/实际大害同168，支持源数和anchor独立重算；不将support数不降说成支持质量已保。

CPU8–11×4、GPU1兼容余量≥6GiB（仅冻结评分）/RSS8GiB、1800s advisory，独立一次。成熟与否按关联、覆盖、物理整体/尾部共同决定，不自动扩V300；若无净益保留反例、结束当前配方。

## 实现与验收（先通过再READY）

| 文件/行为 | 可执行核验 |
|---|---|
| `association/models/association_context.py` 图聚合/共享头 | 自身边移除、2节点空邻域、节点置换/来源换名无影响、0残差exact、OWN/CTX同参数/初始化/同批真实梯度 |
| `association/evaluation/partition_refinement.py`受限搜索 | 原refine默认输出不变；anchor不动、合法最佳候选搜索、source支持数逐query不降、source重命名不影响、J单调与全node一次 |
| `tools/analysis/association_context_pilot.py` | 真实保存→variable-group评价→独审/四属性fixture；基准WS159及REF165绑定；标签不进预测API |
| 完整科学后一次限定audit/属性 | 各arm保存输出独立重算位置/人口/coverage/尾部/组件/主要CI，关联由纯NumPy标签×pairs×membership重算计数/宏指标，支持与anchor逐项核；不独立重训或认证物理truth |

成功后Record→Squeeze→Mode C回流再决定下一步；科学故障保全，不自动重试/换配方。新独审技术错误可另立明确修复检查，不能倒填为原科学成功。

发布前核验：5项算子/梯度/受限搜索测试、原159→165→169真实接口保存/评价/独审/完整属性fixture，以及168冻结teacher→真实2步头训练→变组保存→独审/属性链均通过（共7项测试）。初版本地sampler记账方法名不符、约束fixture误设目的anchor次序，均在READY/服务器科学前修正，保留为实现偏差，不是科学重试。支持proof还独立检查全部query而非平均数；浮点聚合分母≤1e−12仅数值零邻域置零，不解释为科学支持阈值。没有把软件fixture当MTAD成绩。
