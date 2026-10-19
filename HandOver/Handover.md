# 当前服务器任务

只从当前队列领取；旧L2S的READY不是新任务。操作见[027修订](L2S-20261019-027.md)，配置与[本地评审](../docs/proposals/repair_before_retraining/README.md)以最新用户决定为准。push不会唤醒服务器。

**时长只作建议，超时由AI询问，不自动杀训练。** 固定更新、实验次数、最多两单卡和T_test封存不变。新任务两层advisory；在途020/021仍有旧固化截止，不靠Git热取消、不为此重启。

| 项目 | 最新状态与下一步 |
| --- | --- |
| [019](S2L-20261019-019.md) / [026](S2L-20261019-026.md) | 019转录错误FAILED；026退出0及独立服务托管已核，两项不重跑。 |
| [020参考](S2L-20261019-020.md) | RUNNING。72b3bad已回传tam100原run/history/coverage；本地核1100更新=TCM1000/TAM100、全部loss/梯度有限、一次V完成708.240秒、history记录F1 .4245588143/64场景、TAM均时23.164249秒、test_opened=false。没有逐边scores，不称独立重算F1；仍缺checkpoint.selected.validation的P/R与V人口摘要，下面已明确允许CPU只读导出。尚无物理终评。旧截止约北京时间10月20日14:46:55/14:47:25。 ✅ **TAM@200 已落盘并按 `48046d6` 授权完成 checkpoint 的 CPU 只读元数据提取**（`torch.load(map_location='cpu', weights_only=True)` 一次成功，未回退 `weights_only=False`，未构造模型／未 forward／未读 ZIP，`states` 与 `optimizer` 未导出）。新目录 `results/REFERENCE-REPAIRED-20261019-020/snapshot-20261019-0935Z/`：`checkpoint-metadata.json` 175,698、`run/history/coverage` 的 gz（原 542,042／466,541（**1,200 行＝TCM1000＋TAM200**）／2,510,826）、launcher 四件与 `SOURCE_AND_TIME.txt`。**首个可复核 P／R**：`selected` 明标 **`stage TAM_FROZEN／update 200`**、`macro_f1 = 0.42525871846207886`、`threshold −0.000253647`（`selected_snapshot_copied=true` ⇒ 确实超过 TAM@100 的 .4245588，＋0.00070）；**tp 2,262／fp 11,391／fn 1,035／tn 14,320**，**micro precision 0.165678 ＝ 1.4577 倍先验（先验 3,297／29,008＝0.113658）**、recall `p_ca` 0.686078、`p_fa`（FDR）0.834322、FPR 0.4430、micro-F1 0.26690；`negative_scene_count 0` ⇒ 过并仍 **null 不可测**；`per_scene` 64 项全量回传。人口逐项对上预期：**V 64 场景／1,635 轨／29,008 对（正 3,297／负 25,711）／377 目标**，T_core 512／54,792／18,249,384／12,699。边界：017 的可核摘要无 tp／fp／tn，**不得把 1.4577 倍说成优于 017**。实测第二单价：**`t_TAM(100→200) = 22.027 秒/步`**（比第一百步快 5.1%）、验证 **694.653 秒**。条件情景更新：按实测 TAM/验证＋017/013 的 TFM/JOINT 原单价总需 **85,527 秒（余 813 秒）**；若 TFM/JOINT/终评按 1.1988 放大则 **97,123 秒（超 10,783）** ⇒ 仍取决于未被 020 实测的 TFM/JOINT 单价。照 `48046d6`：不提前接受失败、不从头重跑、不缩步骤、不为略超重写授权。 |
| [021共享](S2L-20261019-021.md) | RUNNING attempt3。17:19快照原history703行teacher、progress完成703/下一步704、elapsed6814.155秒；run仍是阶段开始0步快照，不能据此说未训练。尚无该次身份V；teacher1000→身份V→utility200。旧截止约北京时间10月19日19:25:41/19:26:10。 `09:40:32Z`：teacher **854/1,000**、`elapsed 8,085.326`、阶段均 9.283 秒/步，而 `history` **最近 20 步 8.073／最近 50 步 8.064**（在加速）。余需 **2,177～2,502 秒** ⇒ 内部累计 **10,262～10,587 秒 ＜ 14,400（余 3,813～4,138）**，预计 **`10:17～10:22Z`** 出 `shared.pt`。不重复已收快照，待终态一次性回传 run/progress/history、identity-records（index.json＋全部场景 NPZ）与 `shared.pt` 路径登记。 |
| [027修订](S2L-20261019-027.md) | COMPLETED，旧校准取件已核，不重复。四臂独立代码目录已备，尚未启动。 |
| [022主方法](L2S-20261019-022.md) | WAITING_021；attribute-u300＋全V，按027两层advisory，建议20分钟。 |
| [023普通融合](L2S-20261019-023.md) | WAITING_021；plain300＋全V，同上。 |
| [024统一效用](L2S-20261019-024.md) | WAITING_021；scalar-u300＋全V，同上。 |
| [025普通门控](L2S-20261019-025.md) | WAITING_021；learned-attribute300＋全V，输出仍COOPERATIVE-LEARNED。 |

**本地对六项复核请求的结论：**

- 首分暂未显示改善：020报告.42456，017已核TAM100为.432985，约低0.84个百分点。但TCM已从100变1000、校准值71.589→75.693、验证频率/批次等也不同，不能称同总预算或单因素对照。两轮都做校准，不等于校准数值不影响结果。selection.candidate_count=28737是阈值候选数，不是关联人口，需原记录核实际对数及P/R。
- 时间投影只作条件情景。按回执列出的单价复算总24.093h；把旧TFM/JOINT及终评同时乘1.264得27.912h，不是概率界限。基础情景的训练结束约83527秒，早于旧86340秒截止，可能截在终评而非训练；不能宣称“最可能在TFM/JOINT被杀”。TFM/JOINT还未实测、新微批会改变其成本，不据此从头重跑或缩步。
- 补全LRU缓存驻CPU，不会涨满额外占约900MiB显存；nvidia-smi空闲值已扣当前占用，不再减一次本进程峰值。TAM不使用pair/track_microbatch，不能以其变更解释TAM慢。共卡成本不当独占基准。
- 100步/阶段末才写checkpoint确有可观测性限制，旧history不能定位实时阶段。重复SIGTERM打断终态的修复已在9b06f80并用于新四臂，不重复实现，不热改活动020/021。真实故障照原件处理。
- authorization.json已是advisory，86400/1200为建议值，不是待补的新额度。四臂20分钟也只是旧速估计，不能保证；无需为了推算略超重写授权或申请28小时从头重跑。

**当前回传清单：72b3bad的020 run/history/coverage和021 run/progress/history已收到，不重复同一快照。现在优先补020检查点元数据；明确允许以下CPU只读提取，它不属于被禁止的模型推理，无需再等用户审批。**

| 来源 | 现在可返回的已有内容 | 用途 |
| --- | --- | --- |
| 020：runs/REFERENCE-REPAIRED-20261019-020-attempt-1 | run.json、history.json、coverage.json，可gzip；对应-launch的status.json及有内容的stderr/stdout | 核阶段、loss/梯度、验证曲线、真实训练覆盖和异常；只报F1不足以定位问题。 |
| 020：同目录checkpoint.pt | 权重留服务器；只用CPU torch.load(map_location='cpu', weights_only=True)导出checkpoint-metadata.json：cursor、elapsed_seconds、config、normalizer；selected去掉states后原样保留，尤其stage/update/threshold/validation；data_contract只导出下面的人口摘要 | 中期完整P/R、TP/FP/FN/TN及逐场景指标实际在selected.validation，不在run/history。必须标明被选快照的stage/update，若已到后续步，不冒充TAM100。 |
| 021：实际attempt3 | run.json、progress.json、history.json；对应-launch/status.json和有变化的stderr | 核teacher/utility实际更新、身份BCE与物理分项loss、当前进度和失败。run可能停在阶段边界，应结合progress/history。 |

020人口摘要直接读取checkpoint的data_contract=[训练场景列表,V场景列表]：每场景保留scene_id/split及节点数、pair_count、positive_count、target_ids数量、native_observed_counts之和，再汇总。预期V为64场景/1635轨/29008对（3297正、25711负）/377目标；若不同如实返回，不改筛选凑数。正常配置/normalizer保留原值，不能重新fit。CPU只反序列化已有检查点，不构造模型、不forward、不读ZIP；无须上传.pt权重。每包注明采集时间和实际源路径；文件可能分别处于不同更新游标，如实注明即可。

**正常运行产生后再回，不为了取件追加计算：**

- 021完整shared阶段结束：终态run/progress/history、已有execution及launcher退出码，identity-records/index.json和全部场景NPZ（可打包），shared.pt只登记实际路径和存在情况。identity-records在teacher1000→身份V→utility200→shared_export之后才写，不要求teacher刚结束就有。
- 020终评：summary.json、evaluation-progress.json、predictions/index.json和场景metadata。022—025各臂：route-evaluation.json、route-scores.npz、route-predictions/index.json和场景metadata，route为attribute-u/plain/scalar-u/learned-attribute。这些报告已有预测/oracle分组、单源/简单融合、覆盖、观测区/缺失区及输入支撑/距离分层。
- 为核实际数据和恢复轨迹，终评先回各自预测index中场景序号0—3的metadata＋NPZ，按固定序号、不按结果好坏挑；四臂和参考用相同场景核对。其余完整数组留服务器，后续需要时只复制已有文件。这些数组包含真实输入/时刻、query、标签、reference/mask和预测，足以先查具体错位/外推案例，但4场景不能冒充全V逐点复核。

020中期没有逐对score缓存，不能要求不存在的identity-records；缺少的字段标“未保存/尚未产生”，不临时加载模型补算。preparation.json和exit_code等终态文件在RUNNING期间不存在不算失败。用新快照子目录避免覆盖inspection-027/原失败产物；本清单只是现有任务结果回传，不是新READY轮次，不改活动代码、阈值、训练步数或T_test封存。

四臂等021完整COMPLETED（teacher1000/utility200，共1200更新，shared.pt、identity-records及唯一阈值齐），读取同一实际成功端点（预期attempt3），不读失败attempt1。复用已备独立冻结目录和027命令；科学配置、normalizer/S、身份阈值、初始化/seed22、新Adam不变。每臂一次无重试，优先022/023，最多两单卡，可共卡，不触碰他人。

建议参考24h、021单次4h、候选整线6h（含失败）、四臂各20分钟。超过建议或接近旧截止仍未完成时，先说明实耗/进度/剩余/检查点并询问；当前不提前接受失败、不从头重跑、不缩步骤。若后续实耗确有跨截止风险，先审已有检查点的接续可行性再给方案，不能把“只能重头再跑”当既定事实；当前失败参考不具备自动resume授权，不篡改旧状态。

001–018已结案，不重领/续训/补评；CDF退役，本轮不额外训练TSA。[历史摘要](../docs/history/README.md)。
