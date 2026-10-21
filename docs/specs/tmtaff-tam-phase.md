# 033之后的独立TAM阶段与047训练策略修复

状态：**035已完成并复核，原授权一次1000步已用完，不重领。** 结果见[035分析](../proposals/repair_before_retraining/035-analysis.md)，下文保留已执行规格。

问题：固定033的TCM4000端点，学习新的TAM，检验补全特征改善能否支持关联。此阶段没有TFM或JOINT，不能宣称完整TMTAFF已复现。补全缺陷分析与034候选独立进行。

## 输入与行为

在已有`tmtaff_complete_fit.py`增加`--phase tam --initialize-tcm-from <033-run目录>`。源必须是COMPLETED的TCM-only端点；目标输出目录必须新建。核对模型配置、分支、原生时间适配、cap、完整T/V数据契约与仅T拟合的normalizer，normalizer数值相同后使用源副本。

仅复制并冻结`input_embedding`、`tcm_encoder`、`tcm_decoder`及继承的`positions`缓冲。native时间编码是无参数函数，没有遗漏的时间投影；`positions`在native模式不使用，仍复制以保留状态。TAM按当前seed重新初始化，不载入源TAM、融合、optimizer、RNG、history、coverage或累计耗时。不恢复COMPLETED033，也不恢复FAILED020。真正中断的新TAM轮次仍沿用原INTERRUPTED恢复约定和同配置检查。

固定配置：[`configs/tmtaff-tam-035.json`](../../configs/tmtaff-tam-035.json)，TAM1000，其他阶段0；seed20、lr0.002/constant、Adam参数、batch/microbatch、全候选pair、正负平衡采样及T_core负例中位数margin校准均沿用020。每100步用完整V做原macro-F1阈值扫描与最优快照选择。预计约5–7小时，8小时仅提醒，不自动停止、加步或重试。固定T_core512/V64、cap128/Q64、20秒单位与offset/gap/mask；T_test封存。

## 输出与比较

每次已有V推理的全部原分数复用`runtime.save_records`保存在`association-scores/update-XXXXXX/`，不新增推理。这样保留300步同预算比较、1000步端点和selected最佳V三个口径。summary列出每次分数路径与选择结果；最终指标从selected分数缓存计算，包含overall、同源ITSA、跨源MSTSA及paper_metrics，不执行任何物理融合推理。

新run/checkpoint记录初始化源及其TCM更新量；本轮coverage/history只计TAM。保存的fusion/objective状态只是原checkpoint容器中的未训练占位，不计为已训练或已评价。原full/tcm流程不变。不能把035@1000对020@300的差异全归于TCM；035@300可比较原020@300，但单seed且训练RNG经历不同，仍不构成严格单因素因果证明。

## 必要验收

| 行为 | CPU证据 |
| --- | --- |
| completion准确转移且冻结，TAM独立初始化并更新 | 微型native ZIP，逐tensor检查，故意改变源未训练TAM以排除整模型误载 |
| 全V所有pair，空场景保留，T_test不读取 | 原生ZIP fixture封存test；V评分人口及缓存逐值比对 |
| 原校准/阈值，300/端点/selected可追溯，无多余推理或融合 | 原评分函数调用次数、summary及原分数缓存；禁止fusion训练/终评 |
| 新预算独立，INTERRUPTED仍精确恢复 | 源耗时不继承；分段TAM与一次运行参数/loss一致 |
| 非COMPLETED源、不同数据契约、跨phase混用被拒绝 | 在目标目录创建前拒绝；CLI无源/多源参数拒绝 |

检查入口：`.venv/bin/python -m unittest tests.integration.test_tcm_fit_cli tests.training.test_complete_training tests.training.test_complete_snapshot`。CPU检查只验证执行语义，真实MTAD关联收益待已批准服务器实验。

## 047：TAM专属dropout适配（已完成，完整V未改善）

042同输入诊断已完成，依据见[042分析](../proposals/repair_before_retraining/042-analysis.md)。[047配置](../../configs/tmtaff-tam-no-dropout-047.json)相对035只新增`model.tam_dropout=0`；缺省None继承公共dropout，旧配置/权重行为不变。该参数覆盖TAM各层nn.Dropout和MultiheadAttention.dropout，以及CE output_dropout；CL实际不使用CE head。TCM/TFM仍使用各自原dropout，不能改全局dropout冒充只改TAM。

TCM-only转移只豁免TAM专属dropout差异，补全架构、表示、公共dropout、normalizer、数据及其余约束仍严格核对；target TAM独立初始化，不读035权重。相同种子下参数初值/RNG状态相同；训练移除随机mask后的torch RNG经历不同，scene/pair sampler仍用独立NumPy原种子。INTERRUPTED恢复必须保留同一语义，不能边恢复边切dropout；旧checkpoint缺字段按None兼容。selected/final加载保留新参数。

原margin校准规则和4096个T_core展示不变，报告新增实际TAM dropout；移除随机性后margin数值自然重新得到，不强沿用75.26。正d、负hinge²的尺度改变还会改变两类对距离的损失导数平衡（不等于直接测到参数梯度），因此不能按原始loss简单排名或宣称完全隔离了数值margin。每步`history.tam_train_batch`保存同一次前向的正负均距、两类含0.5系数的CL类内均值、负例margin内比例、两类count及有效margin；无新增前向、随机数消费或loss改动。

047仍一次TAM1000、每100完整V与原macro-F1选模/全分数保存，无TCM/TFM/JOINT更新；具体范围、8小时提醒及回传在[047交接](../../HandOver/L2S-20261021-047.md)。参数更新/转移/精确恢复等本地检查及独立审查通过，不等于真实关联改善；无dropout也不保证不同设备/fastpath逐位相同。若确定性距离仍近零或完整V未改善，再据分项诊断定位下一因素，不自动加训或改loss。

## 可选目标尺度诊断：仅负项除固定margin

047的负结果见[分析](../proposals/repair_before_retraining/047-analysis.md)。当前本地实现可选`model.contrastive_negative_scale="inverse_margin"`：单例`L*=0.5*[y*d+(1-y)*relu(m-d)^2/m]`，批次取均值；m仍由原训练侧负距中位数一次校准并固定，不由V调节。正项对距离导数保持0.5，负项为`-relu(m-d)/m`，范围[-1,0]；这些不是参数梯度。它等于margin归一化目标乘m，不是把原总loss除m，亦不是论文公式修bug。缺省`"paper"`保留原式；该候选只支持原线性正项。

[提案配置](../../configs/tmtaff-tam-inverse-margin-proposed.json)相对047只新增上述开关，继续冻结033、TAM无dropout、原初始化/采样/校准、lr和完整V选模。1000步只是待批准实验配额，没有新增READY或启动训练，也不切换到新3000场景协议。TCM-only移交允许该TAM损失差异；已训练TAM的resume不能切换目标。校准和训练分项报告使用实际负项权重，selected/final保持同一配置。正式原式baseline仍须另行对齐论文公开的数据与训练呈现，不能用本诊断替代。

训练器以每场景`split_unit`区分协议：缺省`shared_a_component`继续拒绝T/V共享组件；仅一致显式`scene`允许共享真实组件。所有协议仍拒绝重复/跨角色重叠scene_id，混协议拒绝。新checkpoint记录协议和完整场景契约，旧缺字段按组件协议解释；数据或协议变更不能resume或转移旧TCM。此兼容准备不授权新数据训练。
