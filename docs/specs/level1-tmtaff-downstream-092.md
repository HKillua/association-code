# 092：同一 Level1 TCM 来源上的独立 TAM / TFM 阶段（SUPERSEDED）

**2026-10-24：088回传后，本未发布/未领取计划已被[093/094成对设计](post088-completion-value.md)替代，禁止领取或执行下列历史命令。** 两个旧单learned臂不计新尝试；只认当前HandOver。以下保留当时方案与验收记录。

本合同把 088 的基础端点推进到两个可审的学习阶段，补齐独立任务基线资格。它们均属于 **time-adapted approximate TMTAFF**，不是原作者协议的完整复现、充分训练的竞争结果或完整系统。执行状态以[092 交接](../../HandOver/L2S-20261024-092.md)及交接索引为准；本文件不自动增加已领取任务。

## 问题、依赖和范围

092A 回答：固定当前 TCM 表征后，原 TAM 关联训练在新 Level1 全人口上是否能形成可用的分数和关联输出？092B 回答：同一固定 TCM 上，TFM 在真实分组条件下的物理输出、共同点误差和覆盖如何随固定预算变化？两项可以并行，互不依赖对方结果，也不要求 TCM 特征误差或其中某项先胜过物理控制。

唯一训练来源为服务器 `runs/LEVEL1-TMTAFF-TCM-20261024-088-attempt-1`。只有 `run.status=COMPLETED`、最终完整 V 特征评价也为 `COMPLETED`，且实际 checkpoint、配置、T/V 场景顺序、片段人口、normalizer 和成功更新记录一致，才可领取。088 尚在训练或终评时只记 `WAITING_DEPENDENCIES`，退出码 75；不读取活动 checkpoint，不创建本次输出/launcher，不消耗尝试。源任务失败或来源矛盾须保全并回传阻断，不能自动换旧来源。条件满足且有 FIFO 资源时可按本合同自主领取，无需重新请求用户批准。

两项各自新 checkout、新输出、新优化器与随机流；仅转移 `input_embedding / tcm_encoder / tcm_decoder / positions`，继承 T-only normalizer，并冻结整个补全模块。不得继承源运行中未训练的 TAM/TFM 参数、优化器、历史或已用墙时。现有训练器在重读真实输入后逐项比较数据契约及重新计算的训练 normalizer，不靠文件 hash 接纳来源。

## 固定输入与两个端点

共同输入为新 Level1：T_core 2100 场 / 155134 片段、V_select 完整 300 场 / 24599 片段，T_head 空，T_test 600 场保持封存。原生 cap128 保首尾、Q64、时间单位 20 秒、参考经度 shortest_arc；单位不代表每 20 秒重采样。网络沿用 088：d128、8 heads、6 层、FF256、dropout .1；seed20、Adam .002、constant、clip1。单 GPU / CPU4；整链 8 小时为提醒，提醒后继续同一次固定更新和终评。各一次，不自动重试、缩 batch、加步或删样本。

| 项目 | 092A TAM | 092B TFM |
| --- | --- | --- |
| 配置 | [tmtaff-tam-level1.json](../../configs/tmtaff-tam-level1.json) | [tmtaff-tfm-level1.json](../../configs/tmtaff-tfm-level1.json) |
| 唯一训练阶段 | TAM_FROZEN 4000 更新 | TFM_FROZEN 1000 更新 |
| 其他阶段 | TCM / TFM_FROZEN / JOINT 全 0 | TCM / TAM_FROZEN / JOINT 全 0 |
| 采样 | 原 balanced scene-pair：每批 32 正 + 32 负 | 正对可用场景均匀，再均匀真实正对 |
| 曝光 | 256000 对呈现，正负各 128000 | 64000 真实正对呈现 |
| 实际计算 | TAM 实际整批 pair64；TCM cache / TAM 编码 batch_tracks64 | pair batch64 / microbatch8；track microbatch4 |
| 目标 | 原 contrastive p=1 / 原负项比例 | 原 standard_smooth_l1，beta=.001 度，train_position_affine，shortest_arc |
| 评价端点 | 仅最终 4000；完整 V 所有候选边 | 0 与 1000；完整 V oracle-only |
| 选择 | 最终分数按原 V Macro-F1 协议选阈值 | 固定 final，不选阈值或 best checkpoint |

TAM 的 `train_negative_median_v1` margin 只由 T_core 首次校准：独立 seed21、64 个 batch64、2048 正 / 2048 负曝光，恢复 Torch RNG 且不消耗正式采样流；保留校准细节与实际 margin。配置中的 `pair_microbatch=8 / track_microbatch=4` 仅作用于 TFM 路径，不缩小当前 TAM 的实际整批 64。4000 更新不是训练 epoch，也不是已充分收敛的证据。只在第 4000 次评价，不搜索中间 checkpoint；但 **同一 V 选阈值和报告 F1 仍存在选择偏差**，且该 V 已反复用于项目开发。完整候选分数、逐场/ITSA/MSTSA 关联指标必须回传必要摘要并保存原始记录，不能只回传一个 F1。TAM 不运行未训练 TFM 来制造物理分数。

TFM 只使用真实分组，评价包含原 simple-fusion / single-source 控制、完整人口与 TFM/simple/truth 共同支持点；同一评价端点保留完整预测数组。第 0 次是该新初始化的诊断，1000 为固定终点；二者不作 V 最优选择。TFM 不调用 TAM、关联阈值或 TAM margin 校准，不把 oracle 成绩称为预测关联系统成绩。64000 为采样曝光，不是唯一正对数或 epoch。旧 Level2 049 的 TFM 退化不能忽略；换 Level1 本身不构成修复证据。

## 入口、状态和机器检查

新增窄入口 [tmtaff_downstream_fit.py](../../tools/experiments/tmtaff_downstream_fit.py)，先检查源运行终态，再检查 088 配置、完整人口与 T-only 来源，并复用现有 `tmtaff_complete_fit.py --phase tam/tfm --initialize-tcm-from`。`--check-source-only` 不写文件、不构造训练模型、不占 GPU。正常执行仅在来源就绪后进入现有训练算法；源/目标完整配置须分别等于本仓库 088 / 092 对应配置，不能用 CLI 偷改科学预算。

运行结束补写 `downstream-verification.json`：核同源 normalizer / 数据契约、TCM 冻结逐值一致、只有指定阶段按顺序完成全部更新、完整 V、TAM 所有候选边数/节点数、TFM 两个 oracle 端点及预测索引。检查失败时入口非零返回，并将检查失败记录独立保留；即使底层训练 `run.json` 为 COMPLETED，也不能把该作业作为通过验收的 092 端点。入口不重写已经产生的训练原件。

验收必须包括来源正确、有限数值、真实成功更新/曝光、完整评价与有效权重；成功退出不等于性能成功。所有无输入场景、缺真值、未匹配目标、无预测和 coverage 均须保留，不筛容易样本。报告 loss/梯度/预测分布、完整人口误差与共同点误差，可据结果判断训练量、表征/头或支持域问题；不以相对于物理控制的胜负阻断本次基线资格推进。

## 测试映射与明确的下一接口

新增真实 tiny Level1 ZIP 贯通测试保留 Q64/cap128/time20，仅缩小网络、人口和更新：完成 TCM 后分别运行 TAM/TFM，检查无输入验证场、缺参考目标、冻结参数、完整候选、真实 oracle 预测及禁止读取 T_test。状态测试保证 pending/活动来源不读 checkpoint、不创建输出、不调用训练。真实数组/normalizer 不一致仍由现有训练器在模型构造前拒绝。软件测试仅证明入口/契约，不是正式 GPU 实验结果。

**完整学习关联 + 学习融合的组合评价接口尚缺。** 当前 `tmtaff_complete_evaluate.py` 接收已完成 full 阶段且含共同 selected snapshot 的 checkpoint，不能把独立 TAM 或独立 TFM checkpoint 直接冒充 full 模型。092 之后另立组合接口：验证双方同一 TCM / normalizer / 数据契约，分别装载已训练 TAM 与 TFM，冻结既定 TAM 阈值，在同一全 V 上产生预测分组融合、oracle 分组对照、覆盖和错误传播。若要联合微调或扩展训练量，也必须另写具体合同；本次不增加 JOINT，不拼接模块独立数字成系统分数。
