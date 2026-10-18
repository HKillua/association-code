# 输出尺度修正与本地验证

2026-10-18。用户本轮授权：“分析实验，我去吃饭，如果你觉得找到了问题及直接改代码，如果需要和我讨论，就把能先做的做了。”本地据此实施已定位的输出参数化修正，未增加GPU实验或替用户决定正式训练预算。

## 最小改动

- `PairFusionConfig.output_parameterization`默认raw，历史输出与state键保留。新参考开发模板显式train_position_affine，预算仍null。
- train_position_affine仅增加两个固定buffer，使用既有训练原生normalizer，仅将经纬度线性投影做mean/std物理映射，速度与航向保持原实现。没有新增可训练参数，未改物理loss、TCM/TAM、采样、递归、分组、指标或选模。
- trainer、新/旧selected/last快照及恢复入口传递同一normalizer。旧配置缺字段解释为raw，不自动升级；统计缺失、非有限、非正std或快照统计与normalizer冲突均拒绝。
- 新run明确记录`fusion_output_parameterization`，原生affine名称为time/output-scale-adapted approximate reference。013/014原始配置和结果、已批准候选代码与配置未变。

这属于已披露的近似参考数值适配，不声称原论文必然要求该映射，更不作为候选创新。它只修正从零尺度追到地理中心的优化障碍；修正后的真实关联/补全/融合性能尚无结果。

## 验证

| 实际检查 | 结果 |
| --- | --- |
| 新增9项CPU测试：已知/零投影物理值、有限前反向、固定统计、非法统计、raw兼容、原生四阶段、selected/last和精确恢复、统计冲突 | 通过；原8项曾独立复跑，最终9项及收窄改动独立复核 |
| 新测试及fusion/complete_training/snapshot/timed_pipeline/frozen_cache相关52项 | 通过，7.55秒 |
| `.venv/bin/python -m unittest discover -s tests` | **433项通过，19.369秒，退出0** |
| `.venv/bin/python -m unittest discover -s tools/pilot -p test_audit_mtad_level2.py` | **15项通过，0.062秒，退出0** |
| `git diff --check` | 退出0 |
| 014原阈值完整分数/分组/统计CPU复算、数学上界独立复核 | 通过，见同目录报告和两个脚本 |

合成CLI测试中出现的预期FAILED日志用于检验非法设备和输出路径冲突，完整unittest终态为OK，不是真实MTAD运行失败。

另从Git历史直接载入修改前PairFusion比较：生产配置259个state条目/5,937,156参数及初始化RNG完全一致；小型raw模型eval/train输出和反向梯度逐位相同。最终收窄改动再独立静态复核通过。

独立review覆盖输出映射位置、不可训练buffer、历史raw键与RNG、checkpoint与resume、数学推导及结果支持域。没有用改断言、改评价人口、删难例或重选阈值取得通过。CPU验证不等于CUDA实测或收敛证据；本地没有新增真实数据训练、GPU或T_test操作。

## 交接

015命令由第二位审阅者对照当前CLI/trainer/已批准配置核对，Bash语法检查退出0。累计更新验收用`completed_updates=12`，不误读末评价阶段的`total_updates=0`。参考本地修正不改变015候选配置；新交接与代码分别提交。

正式参考的更新数、选择策略和总预算仍未批准。已批准的015只用于候选12次更新及完整四臂输出的实际成本与数值诊断，不将它作为效用收益或充分训练比较。
