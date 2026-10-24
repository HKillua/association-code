# 088：Level1 TCM基础端点合同

本项补齐当前正式Level1来源的TCM训练端点，复用已有时间适配实现，不重复旧Level2训练，也不作为087方法探索的前置。它是 **time-adapted approximate TMTAFF的基础阶段**，不是作者原版复现、充分训练的强基线或完整系统成绩。执行状态只认[交接索引](../../HandOver/Handover.md)，本文件本身不产生领取任务。

## 固定执行范围

现有[入口](../../tools/experiments/tmtaff_complete_fit.py)使用`--phase tcm`和[Level1配置](../../configs/tmtaff-tcm-level1.json)，无需修改生产算法：

- T_core2100场、预期155134合法片段；V_select完整300场、预期24599片段。T_test封存，normalizer仅拟合T输入。
- 原生片段至少两个合法时间点；cap128均匀选点保首尾；全场Q64；时间数值单位20秒，不能解释为每20秒重采样或已复原作者窗口。
- seed20，Adam .002，constant，clip1；`ordered_fragment`按场景/节点顺序覆盖一次；2424次成功更新＝2423个64片段batch＋最后62片段短尾。`track_microbatch=4`不是TCM实际batch，TCM仍整批计算。
- fresh运行，TCM以外阶段更新全0，无下游、阈值选择、物理融合或best checkpoint选取。只保存固定final及完整V特征诊断。
- 一次单GPU/CPU4，整链8小时仅提醒。失败保全且不自动重试、缩batch或删输入。服务器核真实CPU、显存、内存与I/O；本地小型CPU测试不提供正式显存/时长保证。

## 重要纠正：这里不存在随训练移动的目标投影

附件088写的“目标表征随模型学习”与当前代码不符。[`_stage_parameters`](../../association/training/tmtaff.py)在TCM阶段显式冻结`input_embedding`；[TCM损失](../../association/models/timed.py)又对下一状态的投影目标`detach`。当前目标是在固定T归一化、固定时间编码及固定随机输入投影下得到的特征；TCM encoder/decoder被训练，输入投影不会因本轮更新移动。

这消除了“通过共同收缩输入目标投影降低损失”的解释，但不证明TCM自然查询外推有用、所有输出都保留区分度或完整系统有效。学习特征MSE仍不是位置米制误差；一步teacher误差、自然query误差及其分层须分别报告。若输出近常量，需用实际输出与固定目标的分布辨识，不能仅凭低平均MSE宣布无坍缩或宣布坍缩。

## 终态和后续边界

必须同时具有2424真实成功更新、完整一遍曝光、完整300场终评、可加载checkpoint与实际阶段游标，才形成可审基础端点。训练结束但终评中断仍不得写COMPLETED。回传`run/summary`、`coverage/history`必要聚合及逐场人口、`completion-evaluation`，原始完整文件留服务器并登记路径；不以裁剪摘要替换后续输入。

下一阶段可独立准备TAM、TFM或一个明确的TCM训练充分性问题，但本轮不自动追加。不得设置“TCM特征误差先胜物理米制控制”这一不相容门槛。2424只来自一遍训练人口覆盖，不等价于论文80 epochs；原文未公开窗口、Level/名单和阶段训练分配的限制仍见[论文协议核查](../mtad_data_protocols.md)及[参考契约](tmtaff-reference.md)。旧Level2 optimizer、未训练TAM/TFM参数不能作为已训练对照结果。

## 本地可执行性证据

2026-10-24使用仓库`.venv/bin/python`、PyTorch2.10.0 CPU，新增[真实CLI贯通测试](../../tests/test_level1_tcm_qualification.py)。测试构造Level1原生ZIP并生成fresh scene manifest，保留Q64/cap128/time20及实际生产TCM阶段设置，仅缩小网络和训练规模：两训练场8片段、batch6、两更新的[6,2]短尾。验证T-only normalizer、训练投影逐值不变且TCM encoder更新、完整验证人口含无输入场景和缺真值片段；拦截所有封存test payload读取及TAM/TFM/阈值路径。该测试通过，说明入口可执行，不是正式Level1/GPU实验。

文档引用的历史测试数不视为本次重跑证据；实际本轮检查列表以代码审查/发布记录为准。
