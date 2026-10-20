# 033之后的独立TAM阶段（035准备）

状态：**代码及配置准备，1000步GPU预算待用户确认，不构成开跑授权。**

问题：固定033的TCM4000端点，学习新的TAM，检验补全特征改善能否支持关联。此阶段没有TFM或JOINT，不能宣称完整TMTAFF已复现。补全缺陷分析与034候选独立进行。

## 输入与行为

在已有`tmtaff_complete_fit.py`增加`--phase tam --initialize-tcm-from <033-run目录>`。源必须是COMPLETED的TCM-only端点；目标输出目录必须新建。核对模型配置、分支、原生时间适配、cap、完整T/V数据契约与仅T拟合的normalizer，normalizer数值相同后使用源副本。

仅复制并冻结`input_embedding`、`tcm_encoder`、`tcm_decoder`及继承的`positions`缓冲。native时间编码是无参数函数，没有遗漏的时间投影；`positions`在native模式不使用，仍复制以保留状态。TAM按当前seed重新初始化，不载入源TAM、融合、optimizer、RNG、history、coverage或累计耗时。不恢复COMPLETED033，也不恢复FAILED020。真正中断的新TAM轮次仍沿用原INTERRUPTED恢复约定和同配置检查。

草案配置：[`configs/tmtaff-tam-035.json`](../../configs/tmtaff-tam-035.json)，TAM1000，其他阶段0；seed20、lr0.002/constant、Adam参数、batch/microbatch、全候选pair、正负平衡采样及T_core负例中位数margin校准均沿用020。每100步用完整V做原macro-F1阈值扫描与最优快照选择。预计约5–7小时，8小时仅提醒，不自动停止、加步或重试。固定T_core512/V64、cap128/Q64、20秒单位与offset/gap/mask；T_test封存。

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
