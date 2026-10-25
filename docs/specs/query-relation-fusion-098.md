# 098：查询时刻的运动关系能否改善位置融合

## 本轮决定与研究主张

用户2026-10-25选定A并要求快速推进主线。本轮直接实现并发布四项固定实验，不等待093A／094A／096A，不改其活动checkout或接续权。

**问题：固定预测关联组后，现有可观测运动关系作为集合注意力的软偏置，能否在两个seed上超过通用集合融合及同预算强Transformer？** 这是缺测／混入错误成员条件下的**位置融合与恢复**候选；速度／航向保持Mean，不据此主张完整多状态融合创新。

078/081已试逐helper编码与学习加权；087增加片段内时间编码；090的冻结readout未胜追加预算。098保留每helper的现有27维证据，加入同query集合交互，并用相同结构的通用set对照隔离任务关系偏置。本轮**不单独证明早聚合是瓶颈**，不把历史4000步078/081当作本轮8000总预算的容量消融。

集合交互本身已有[Set Transformer，ICML2019](https://proceedings.mlr.press/v97/lee19d.html)。这里的generic是任务适配的set-attention对照，非作者全协议复现。候选主张仅为可见运动关系在本任务中的可检验增量；未完成全面文献排重，未声称提出新注意力算子或已经取得正结果。

## 输入、机制与可证伪预测

- 固定056关联、组成员顺序/首成员输出目标、全部V300人口与既有匹配；GT只进T损失和预测后的评价。
- 原76维sequence输入、fixed.95物理底座、原二维切平面残差、linear Haversine/km损失不改。新分支只复用原27维helper输入及当前query的support mask；不加入ID、关联分数、原生重叠观测历史或新标签。
- 每helper `phi:27→32→16`；同query Q/K/V集合注意力和残差LayerNorm；用原76维query特征生成16维pool query，得到16维分支表示。supported mask之外不参与聚合，空集输出零；注意力权重不是物理置信度。
- generic与relation全部参数结构、数量与初始化相同。只有relation在QK/sqrt(16)上减去固定偏置：取已有压缩特征 `z=x[[16,17,20,21]]`（helper OLS两方向速度、相对anchor的两方向位置），`b_ij=min(mean((z_i-z_j)^2),4)`，强度1。generic的b为0。不扫描尺度/强度。
- 该差异是signed-log特征中的helper彼此一致性先验，**不是精确米制距离**，也没有直接建模anchor/helper可靠性或错误身份概率；不把一致性等同真值、不硬剔除少数片段。学习QK可以抵消偏置，是否有益由对照决定。
- 原76列投影维度不改；新增16→64无bias投影初值为0，在原首层Linear之后相加，再经过原LayerNorm/SiLU。完整初始输出逐值等于来源final，避免扩宽GEMM引入初值差异。原Transformer与所有新参数全部训练。
- 支撑H≤1的当前query分支不受bias开关影响；后续时间Transformer允许其他query的影响传播。只有整组每个query均H≤1时，同权重开关的完整预测必须相同。按该组级机会分层，不把所有单helper时刻收益都判成机制无效。

若relation无法超过同输入/参数generic，关系先验主张失败；若二者都无法超过追加预算Transformer，关闭这一配方。仅generic提高则记录更强任务基线，不能将其容量/表征收益记为关系创新。

## 四项固定训练与基线

| 任务 | seed | 模式 | 初始模型 | 同追加预算基线 |
|---|---:|---|---|---|
| 098A | 20261023 | generic | 072C final4000 | 090C final |
| 098B | 20261023 | relation | 072C final4000 | 090C final |
| 098C | 20261021 | generic | 073C final4000 | 097A final |
| 098D | 20261021 | relation | 073C final4000 | 097A final |

各臂fresh Adam 4000，batch8、lr.001、clip1，严格重放各自原Transformer的4000个成功batch／RNG／曝光；不恢复原optimizer。branch seed为训练seed+98000。共16000新更新；已有来源与baseline不重训。两阶段总8000更新不等于连续optimizer8000步或已充分收敛。generic/relation参数相同但FLOPs存在偏置计算差异；与Transformer不声称等参数/等GPU时。

所有臂只取step0与固定final做CPU全V300评价；relation臂额外对**同一final参数关闭bias**做一次全V300 `relation_off`评价，0新训练，无checkpoint选择。不开T_test、不改阈值/normalizer、不删难例、不挑seed。

主比较复用完整预测缓存：seed23 090C MAE564.834006m；seed21 097A MAE583.975593m。072C/073C只作为起点和预算增量参照，fixed.95/Mean/Anchor保留。

## 运行前冻结的去留标准

每个seed单独判定；**两seed均通过**才能进入独立确认阶段，单seed正结果不晋级：

1. relation相对同seed强预算Transformer点MAE至少降低3%（约547.889m／566.456m），RMSE与P99不增加；其余三种MAE加权和最大组件外点MAE不恶化，过半场景和组件MAE降低。
2. relation相对同seed generic点MAE至少降低1%；RMSE、P99、四种MAE加权和最大组件外MAE不恶化。这一更严格增量门槛在098数据返回前固定，避免把数值微小差异写成贡献。
3. relation相对同权重relation_off的全人口点MAE严格更低；存在组级multi-helper作用人口，且无作用组的全预测逐值不变。完整报告开关影响、helper 0/1/≥2和组级机会层，不能把开关结果当独立训练基线。关闭偏置后退化只支持功能依赖／适应性，不单独证明模型理解了正确的运动机制；匹配训练的generic才是关键增量对照。
4. 执行与数据边界全部通过：同源final初值、batch/RNG/曝光、全参数更新、完整人口/覆盖/重复不变、Mean速度航向不变、无T_test。

浮点比较保护为1e-6m；3%/1%是实际幅度门槛，不用epsilon代替。固定final，不自动第三seed、延长步数、宽度/偏置扫描。无论正负，四项及两seed表完成即结束这一合同。generic单独胜出仅列为下一轮可用强基线；默认模型不因训练完成自动改变，T_test保持封存。V300反复开发与共享身份限制必须随结果报告。

## 实现与验收矩阵

| 行为 | 实现位置 | 可观察验收 |
|---|---|---|
| 同参数set交互、偏置开关、空集/mask/顺序不变 | `association/models/query_relation.py` | 有限梯度；同初值；单/多helper与组级noop测试 |
| final暖启动及精确重放 | `association/training/query_relation.py` | tiny真实训练、保存加载、失败保全与实际batch核验 |
| 固定入口、全人口端点与off干预 | `tools/experiments/query_relation_fit.py`、`association/evaluation/query_relation.py` | CLI生产contract；tiny贯通，初值同来源final |
| 强预算/通用对照、尾部/组件/机制判定 | `tools/analysis/compare_query_relation.py` | 逐场缓存重算、人口篡改拒绝、gate否决测试 |
| 双seed唯一结案 | `tools/analysis/summarize_query_relation.py` | 两张完成表/逐场数据重算全部门槛；单seed、缺seed、伪改pass均不晋级 |
| 四臂资源与来源 | `HandOver/L2S-20261025-098.md` | 独立冻结checkout、每项一次、全局四单卡，真实S2L才算领取 |

不复用或改写旧078/090 scope，不放宽旧step0防线。只用现有launcher发布这四项，无第二个与096抢卡的自动队列。正文合同、代码、必要检查与READY同一发布批次；源文件/缓存缺失则记录BLOCKED，不换来源续跑。

旧控制器的全部GPU子任务必须已确认提交或不可逆终态，098才可进入互斥提交窗口；不需要等待已提交子任务／CPU表的结果。该约束只用于资源协调，不改变研究问题或旧任务。

## 本地实现证据

两模式各109954参数（原Transformer105154＋新分支4800），全部可训练。新训练每100个成功更新写轻量`fit/heartbeat.json`；实际逐步样本保留在`fit/training.jsonl`。运行中不把`fit/run.json`旧累计快照当作实时进度。

本地CPU环境torch2.5.1、numpy2.3.5，已运行新模型／训练／比较／完整入口贯通及所复用旧sequence/helper/090/097回归，首批56项通过（18.112s）。随后独立审查发现评价后处理失败可能残留端点COMPLETED，已修正并新增失败状态回归；另新增双seed汇总的缺seed/混seed/单pass/改写pass拒绝测试。累计覆盖58项；最终入口贯通与比较8项重测通过（6.047s），并将真实产生的单seed表送入结案读取校验。四个生产config均以真实固定manifest/alignment检查通过；三个CLI help与diff检查通过。

这些检查证明实现与协议链条，不代表正式Level1效果。完整数据、GPU、来源权重和NPZ在服务器；只有接收端原件预核与实际S2L才建立正式执行事实。初值对照直接复用真实来源final的CPU缓存，省去重复teacher前向；不会省略候选的任何输出组。

## 导入端对抗性复核补充

本地导入后保留原包提交；评价拒绝重复目录时必须原样保留既有结果，不能把已完成端点改成FAILED。旧096控制器持有的是整个队列生命周期的锁，原“现场协调”文字不能排除查卡与提交竞态；098增加只读GPU提交排空证明，四臂再以独立提交锁串行检查和提交，训练阶段仍可并行。新检查不停止或修改旧服务。

集合注意力与偏置均为O(QH²)，关系偏置临时量为Q×H×H×4；实际最大helper组、显存、完整端点耗时待服务器记录。4000更新与匹配曝光不等于相同FLOPs／墙时；4h仅提醒，不能因包内小型CPU检查断言会更快。对新臂使用相同预声明配方，失败保全，不事后缩小batch或截断helper。

最终本地验证、来源范围及修复证据见[导入审查](https://github.com/HKillua/association-code/blob/1a25183/docs/reports/2026-10-25-098-import-review.md)。软件检查不代表正式实验；两seed结案重新计算已完成单seed表的门槛，不声称再次独立重放服务器原始NPZ。
