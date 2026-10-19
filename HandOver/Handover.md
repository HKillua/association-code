# 当前服务器任务

只从当前队列领取；旧L2S的READY不是新任务。操作见[027修订](L2S-20261019-027.md)，配置与[本地评审](../docs/proposals/repair_before_retraining/README.md)以最新用户决定为准。push不会唤醒服务器。

**2026-10-19最新用户决定：按补全→关联→融合逐阶段解决问题。022—025即刻HOLD，不因021成功自动领取；027旧启动块也不能绕过本次HOLD。** 先核补全，再确定后续阶段。020/021是否停止正在向用户确认；本条不授权对活动作业发停止信号，不热改其代码。不把已停/已完成写成未启动；若四臂已经领取，回报实际状态，不另开或重跑。

**时长只作建议，超时由AI询问，不自动杀训练。** 固定更新、实验次数、最多两单卡和T_test封存不变。新任务两层advisory；在途020/021仍有旧固化截止，不靠Git热取消、不为此重启。

| 项目 | 最新状态与下一步 |
| --- | --- |
| [019](S2L-20261019-019.md) / [026](S2L-20261019-026.md) | 019转录错误FAILED；026退出0及独立服务托管已核，两项不重跑。 |
| [020参考](S2L-20261019-020.md) | RUNNING。a221083原件已核TCM1000/TAM200共1200更新；selected=TAM200。V64场景/1635轨/29008对/377目标符合协议，macro-F1 .425259（较TAM100＋0.07个百分点），P .165678/R .686078，TP2262/FP11391/FN1035/TN14320。逐场景计数与公式复算一致，无逐边scores；尚无TFM/JOINT及物理终评。旧截止约北京时间10月20日14:46:55/14:47:25。 |
| [021共享](S2L-20261019-021.md) | RUNNING attempt3。17:40文字快照teacher854/1000；最新已收原history为17:19的703步，二者不混称独立核验。仍待teacher1000→身份V→utility200→shared_export，尚无该次身份分数；022—025尚未领取。旧截止约北京时间10月19日19:25:41/19:26:10。 |
| [027修订](S2L-20261019-027.md) | COMPLETED，旧校准取件已核，不重复。四臂独立代码目录已备，尚未启动。 |
| [022主方法](L2S-20261019-022.md) | HOLD_COMPLETION_REVIEW；暂停领取，原计划attribute-u300＋全V，按027两层advisory，建议20分钟。 |
| [023普通融合](L2S-20261019-023.md) | HOLD_COMPLETION_REVIEW；暂停领取，原计划plain300＋全V，同上。 |
| [024统一效用](L2S-20261019-024.md) | HOLD_COMPLETION_REVIEW；暂停领取，原计划scalar-u300＋全V，同上。 |
| [025普通门控](L2S-20261019-025.md) | HOLD_COMPLETION_REVIEW；暂停领取，原计划learned-attribute300＋全V，输出仍COOPERATIVE-LEARNED。 |

**本地结论与下一步：**

- 020首两次关联验证仅小幅改善，误关联仍多。相较同场景“全接受”macro-F1 .347362有提升，不称全随机。p_fa=1−precision；无纯负场景只使该特定指标为null，FP仍可测。计数/人口、训练覆盖和成本详见[复核](../docs/proposals/repair_before_retraining/README.md)。
- 原人口与协议一致，不能扩展为已排除所有时间/标签/监督问题。当前无已证实需停训的实现故障；研究推进顺序已由用户改为补全优先，020/021的停止方式待明确。不为本次回执重扫数据、重选阈值或另开诊断。
- 020 checkpoint元数据、run/history/coverage已收齐本次中期需要，**不重复取同一快照**。以后正常落盘更新再回，selected须标真实阶段/步数；CPU只读元数据导出已获准，不需再次审批。权重留服务器、不构造模型或额外forward。
- 新四臂启动配置已设为advisory，时间是建议；在途020/021旧限制仍存在。成本外推不是失败概率；不从头重跑、不缩步、不注入进程。接近旧截止且未完成时按下文报告。

**正常运行产生后再回，不为了取件追加计算：**

- 021完整shared阶段结束：终态run/progress/history、已有execution及launcher退出码，identity-records/index.json和全部场景NPZ（可打包），shared.pt只登记实际路径和存在情况。identity-records在teacher1000→身份V→utility200→shared_export之后才写，不要求teacher刚结束就有。
- 020终评：summary.json、evaluation-progress.json、predictions/index.json和场景metadata。022—025各臂：route-evaluation.json、route-scores.npz、route-predictions/index.json和场景metadata，route为attribute-u/plain/scalar-u/learned-attribute。这些报告已有预测/oracle分组、单源/简单融合、覆盖、观测区/缺失区及输入支撑/距离分层。
- 为核实际数据和恢复轨迹，终评先回各自预测index中场景序号0—3的metadata＋NPZ，按固定序号、不按结果好坏挑；四臂和参考用相同场景核对。其余完整数组留服务器，后续需要时只复制已有文件。这些数组包含真实输入/时刻、query、标签、reference/mask和预测，足以先查具体错位/外推案例，但4场景不能冒充全V逐点复核。

020中期没有逐对score缓存，不能要求不存在的identity-records；缺少的字段标“未保存/尚未产生”，不临时加载模型补算。preparation.json和exit_code等终态文件在RUNNING期间不存在不算失败。用新快照子目录避免覆盖inspection-027/原失败产物；本清单只是现有任务结果回传，不是新READY轮次，不改活动代码、阈值、训练步数或T_test封存。

四臂当前HOLD；原共享前置和027命令仅保留供后续恢复规划，不构成启动许可。先完成补全分析，再明确恢复哪个阶段；不自行删难例、改阈值、增训练或重启任务。

建议参考24h、021单次4h、候选整线6h（含失败）、四臂各20分钟。超过建议或接近旧截止仍未完成时，先说明实耗/进度/剩余/检查点并询问；当前不提前接受失败、不从头重跑、不缩步骤。若后续实耗确有跨截止风险，先审已有检查点的接续可行性再给方案，不能把“只能重头再跑”当既定事实；当前失败参考不具备自动resume授权，不篡改旧状态。

001–018已结案，不重领/续训/补评；CDF退役，本轮不额外训练TSA。[历史摘要](../docs/history/README.md)。
