# 016执行证据、回执勘误与017状态边界

016是015已保存物理预测的CPU分析，不是又一次训练。原命令、退出码、起止和所交文件支持本轮COMPLETED；没有发现需要重跑016的执行失败。017的当前状态只有服务器文字回执，不能与016同等强度地独立复算。

## 016原件核验

核验范围：`HandOver/results/COOPERATIVE-ARRAYS-20261018-016/`、`HandOver/S2L-20261018-016.md`、原L2S、`tools/analysis/inspect_cooperative015_arrays.py`及其复用的`compare_physical_predictions.py`。直接读文件和数组，不设置SHA/hash准入。

| 项目 | 直接核验结果 |
| --- | --- |
| 实际命令 | `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/wangguangjie/djs/mtad-field-20261016/bin/python tools/analysis/inspect_cooperative015_arrays.py --run runs/COOPERATIVE-COST-20261018-015-attempt-1 --output HandOver/results/COOPERATIVE-ARRAYS-20261018-016/files` |
| 退出码 | `exit_code.txt`为0；stderr为空 |
| 时间 | 2026-10-18 20:43:07至20:43:41（北京时间），起止相差34秒，与`elapsed_seconds.txt`一致 |
| 状态 | stdout、inspection、progress均COMPLETED；64场景，6份比较，test_opened=false |
| 文件 | 47文件，14,411,293字节；六比较合计11,779,080字节；16个NPZ与16个metadata齐全 |
| 场景/目标人口 | 六报告均64场景、377目标、64条per_scene记录 |
| 案例选择 | 重新对完整64场景报告执行`select_cases`，结果与inspection的规则、理由、场景、差值完全一致 |
| 案例可读性 | 16个NPZ均以allow_pickle=False打开，各83个数组；文件大小与copied_cases记载一致 |
| 案例来源 | 16份metadata与015此前回传的相应metadata内容完全一致；本地没有015全部256份物理NPZ，不能冒称对全量原NPZ逐点复算 |

34秒是墙钟时长，不是CPU核秒；线程上限4不意味着使用了136核秒。命令中CUDA_VISIBLE_DEVICES为空；结合脚本只读保存数组、不构造模型或调用训练/推理，支持本轮零GPU用途。原请求的“GPU0”指0 GPU预算，不是授权使用物理第0张卡。L2S规定570秒TERM及15秒清理，命令文件记录内部实际命令而非timeout包装；本轮34秒结束，未触发限时。

脚本逐一检查四臂完整评价场景顺序，比较器核对完整输入/候选/标签、目标身份与参考mask；只写独立诊断输出、复制少量原文件。该代码路径符合“无模型、无训练、无推理、无ZIP、无重新选阈值”的范围。015原目录“534文件未修改”是服务器陈述，本地不能直接审计远端完整目录；代码检查未发现对源目录的写入行为。没有为补证再次扫描原数据。

## 不能沿用的服务器文字结论

### 1. original与common不是全部数值相同

六份报告逐字段比较均有36个数值不同，且全部属于`available_prediction_count`或`prediction_without_target_count`。误差、真值交集与目标覆盖一致，但共同点分析会排除没有对应有效真值的预测；不能抹去原人口中的这些预测。

| 分组/区域 | original预测点 | common预测点 | original无真值预测点 | common无真值预测点 |
| --- | ---: | ---: | ---: | ---: |
| predicted全域 | 4096 | 3191 | 905 | 0 |
| predicted已有观测支持区 | 2907 | 2906 | 1 | 0 |
| predicted缺失区 | 1189 | 285 | 904 | 0 |
| oracle全域 | 24128 | 18364 | 5764 | 0 |
| oracle已有观测支持区 | 17071 | 17062 | 9 | 0 |
| oracle缺失区 | 7057 | 1302 | 5755 | 0 |

这些计数在比较左右两臂以及位置、速度、航向均出现，因此每报告是2侧×3区域×3属性×2字段=36处不同。正确表述是：四臂有效误差比较的支撑相同，共同点没有进一步损失可评价的真值交集；原完整人口及无真值预测仍须另报。四臂还具有不同门控规则/结构，不能把差异严格归因于“仅适配权重不同”。

### 2. 速度“无重叠区间”错误

完整参考速度范围为0.196437至9010.053936，预测全有效域范围约252.370至1705.795，区间存在重叠。能够成立的是：四臂所有预测都超过参考99分位138.730；参考的主体分布与预测明显错位。参考仍有少量大值，不能据此把全部真值当作小值，也不能未经单位核验宣称物理速度达到某个km/h。

回执将四臂的reference∩prediction统一写为3191点，准确范围应限定为predicted分组；oracle对应18364点。四臂之间相同，不表示predicted和oracle之间相同。

### 3. 案例选择符合预定规则

| 场景 | scene_index | 选择理由 | utility−identity位置MAE（米） |
| --- | ---: | --- | ---: |
| 1587 | 34 | 最小差值 | −31798.07749 |
| 270 | 51 | 最大差值 | +4299.70783 |
| 2211 | 47 | 下中位差值 | −13429.79829 |
| 4 | 7 | 最小数值scene_id | +478.38485 |

恶化场景270与4均保留，每个场景四臂齐全；案例选取并非只挑好例。但四场景仅供说明，不能当全验证集统计；局部目标最大恶化不等于整场景平均最大恶化，不能混用两种排名。

## 017：可知与不可知

本地没有`HandOver/results/TMTAFF-TIME-DEVELOPMENT-20261018-017/`原始结果目录，亦没有其run/history/command/exit文件。以下只依据`S2L-20261018-017.md`截至北京时间2026-10-18 21:43:35的监督记录，未SSH或另起监控。

- 已在独立`association-code-run017` worktree领取，状态RUNNING；启动时间20:42:56，内部35940秒、外部35970秒TERM加30秒清理，额度仍10 GPU小时。
- 最新监督快照已报告TCM100与TAM20，共120更新；不是Handover顶部仍写的TCM100/校准未开始。该快照的run文件时间21:33:43、作业内elapsed3037.1秒，查询约21:40:41；不能把回执更新时间当作新的实时训练进度。
- 训练侧64批margin校准已报告COMPLETED，实际margin71.58904266；首次TAM20完整验证已报告COMPLETED，587.6868秒。没有回传其指标，不能报告新的F1改善。
- 已报告采用train_position_affine、新鲜训练、test_opened=false；这些是服务器记录，待终态原件验证。
- 初选时GPU0空闲，21:41:08起记录共卡；允许共卡，不能把之后耗时当独占benchmark。没有明确OOM、非有限值或预算超限的证据，不应因此终止、重启或更换模型。
- “8.0–8.2小时能完成”仅按旧013两步TFM/joint与014终评作外推。尚未测到本轮TFM100真实成本，也有缓存、分组、写盘与共卡影响；不能保证完工或据此追加预算。
- 服务器记录会话终止可能中断子进程，是监督方式的风险，不是已经发生的失败。当前没有确定新失败需要本地立即干预，也不允许创建后台守护或重新训练。继续原任务与现有六小时检查，等待终态原件。

需要更新当前索引以免把旧TCM快照当最新，并修正上述016文字解释；不改服务器原S2L、不重跑016、不动017活动执行文件。

## 是否应因016异常立即打断017

**当前建议继续017。** 这是针对已定位代码路径的判断，不保证017收敛或达到论文成绩。两条线共享原生输入装配与训练侧`fit_normalizer`，却没有共享016暴露的两个候选输出/身份路径：

| 机制 | 015自有候选 | 017参考开发训练 |
| --- | --- | --- |
| 物理输出 | `cooperative.py`先对四属性做`raw * state_std + state_mean`，速度再softplus；训练速度std约415.27会参与输出缩放 | `fusion.py:PairFusion`的`train_position_affine`仅恢复lat/lon尺度；速度是`softplus(raw_speed)`，不乘速度std、不加速度mean |
| 身份关联输入 | `CooperativeModel.score_pairs`将first/last/span等时间特征及`sl*sr`直接送身份MLP | `TimedAssociationModel`继承`AssociationModel`，在TCM/TAM表征后计算负欧氏距离；无该时间乘积MLP |
| 时间表征 | 有连续编码，同时有上述原始tau的附加支持特征 | `continuous_time_encoding`用sin/cos输出有界编码；没有候选手工时间乘积通道 |
| 融合控制 | identity/utility/learned门控参与各臂 | `PairFusion`固定成对融合结构，无候选身份概率floor与效用门控 |

依据为已批准017配置、`association/fusion.py`输出代码、`association/timed_models.py`及`association/models.py:score_pairs`，并与`association/cooperative.py`直接对照。共用输入normalizer意味着仍应在终态审查异常参考大值与学习效果；它不等同于共用候选的速度反归一化输出或身份乘积。当前并无017的非有限值、OOM、配置跑错或其他确定失败原件。为016问题停止017会丢掉已完成的TCM/校准/TAM开发进度，且现有授权不允许同轮自动重试，未有证据支持这样做。保留原预算、原监督、原代码，候选诊断/方案在隔离主仓库独立推进。

### 限定静态复核：归一化—潜特征—物理监督链

追加检查仅阅读当前017执行实现，没有GPU或真实训练。`timed_inputs.py:296/300`仅归一化观测四属性；`timed_models.py:143`的complete输出[N,64,128]潜特征；`fusion_training.py:154/174`将该潜特征送TFM；`fusion.py:163-165`只对输出经纬度恢复一次训练尺度，速度用raw softplus；`complete_training.py:295-306`直接读取原生物理fusion_targets，没有再归一化监督；`fusion.py:243-260`先按逐属性mask取值，再计算物理SmoothL1/对数速度/周期航向loss。没有把128维潜特征当作4维物理量反归一化，也未发现重复恢复尺度。`fusion_training.py:175-177`使用完整有效batch各属性计数累计微批，uncertainty正则只加一次；训练和终评都使用同一TCM潜特征接口。

本次限定审查没有发现足以确定017训练无效的共享归一化、维度或mask bug。TCM嵌入目标随输入embedding学习、损失权衡、关联单指标选模、极端参考速度和短训覆盖仍是科学与优化风险，需按现有协议看真实结果，不能从静态代码将这些风险认定为本轮已发生的失败。
