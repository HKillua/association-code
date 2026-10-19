# 当前研究目标与路线

2026-10-19。目标是**不完整量测下，补全、关联、融合一起做**：用完整TMTAFF时间适配近似参考与自有候选在相同数据和评价条件下比较，判断课题是否值得继续。baseline与候选并行推进，各自的配置、预算和领取状态只以[交接索引](../HandOver/Handover.md)为准。

## 当前完成与尚缺

| 对象 | 当前角色与证据 |
| --- | --- |
| 完整TMTAFF近似参考 | 原生时间输入、TCM/TAM/TFM四阶段、恢复与全人口评价已实现。017已外部SIGTERM中断，未到预算、完整终评未开始；[终态复算与修复](proposals/repair_before_retraining/README.md)纠正首末batch归因并交付独立托管/日志修复，不宣布融合胜负。 |
| 018共享补全/融合候选 | 600更新和四臂全V完成。身份弱、物理恢复误差大，效用没有稳定专属收益；[完整负结果](https://github.com/HKillua/association-code/blob/612afea/docs/results/cooperative018_local_review/README.md)保留，额度关闭。 |
| 当前候选v2 | [规格](proposals/candidate_revision_after018/README.md)、[本地实现/验证](proposals/candidate_revision_after018/README.md)已完成；保留可见位置底座，修正速度/航向表示，加入支撑补全、局部关联、属性残差融合。共享/独立臂已实现，021—025在候选合计4h内安排，须先通过019，尚无v2真实MTAD效果。 |
| TSA-cTFER | [独立离线关联诊断](tsactfer_adaptation.md)，不是完整补全/融合baseline，也未复现原在线指派。保留既有结果，不扩建新融合管线。 |
| CDF1、A0/A_full | CDF活动训练退役，[负结果与理由](https://github.com/HKillua/association-code/blob/612afea/docs/decisions/retire_cdf1_and_focus.md)保留；A_full不是完整TMTAFF。旧关联分数不与新原生时间完整系统混排。 |

用户接受跨领域成熟机制加海上任务具体适配，不要求三个阶段各造一个新模型。按[Ponytail](https://github.com/DietrichGebert/ponytail)和第一性原理，先完成必要数据/实现检查，再用一次完整比较检验假设；不把所有baseline、多种子、全组合消融或“海上首次”穷尽证明放到开跑前面。已有近邻方法仍诚实引用，数值修正不计创新。

## 当前方法假设与必须回答的问题

同一条同船辅助片段，在不同查询时刻、不同物理属性上的恢复价值可能不同。v2以真实观测支撑形成补全查询，以全局＋局部证据判断身份，再按属性效用汇聚辅助残差；三个阶段共用原生输入与共享编码解码，不堆三套模型。借鉴来源与海上近邻区别见[候选规格的来源表](proposals/candidate_revision_after018/README.md)。这些是待验证的适配假设，018负结果并没有自动证明新结构必要或有效。

需要两张表：

- **完整任务表**：参考与候选共用划分、查询、参考监督和全目标人口，报告关联P/R/F1、预测分组位置/速度/航向误差、覆盖、缺失区、错并拆分、参数与实际成本。两系统优化器、loss、mask、正对量、选模和singleton策略不同，披露这些差异，不称严格等训练预算。
- **候选机制表**：v2的plain、scalar-u、attribute-u、learned-attribute共享teacher、身份/阈值、初始化和采样；同新残差结构比较属性路由及效用监督。新旧018同时改变表示、结构和训练量，不能逐项归因。更多模块消融在总体出现信号后考虑。

单源、简单融合和oracle身份分组是诊断：简单方法仅有支持区输出，oracle使用真实身份，不能与预测身份完整系统直接混排。保留各自全人口覆盖与共同有效点，两者缺一不可；零交集误差为null，不删难例或重选子集阈值。完整V只作开发，T_test封存。

## 数据、时间任务与下一步

已批准T_core512/V64、共享A隔离、cap128/Q64/time_unit20、已核offset/gap600；保留所有有效短轨。20秒是编码尺度，当前Q64实际间隔随场景跨度变化，非原采样周期。全部576场景数据与监督审计已完成，见[数据契约](data_contract.md)和[冻结审计](https://github.com/HKillua/association-code/blob/612afea/docs/results/frozen_input_audit_20261018/README.md)，不重复扫描。

[外部处理核查](https://github.com/HKillua/association-code/blob/612afea/docs/results/external_mtad_processing20261019/README.md)未证明我们读错原包；TMTAFF未公开可还原的窗口秒数，整场景Q64是适配假设。[时间任务分析](https://github.com/HKillua/association-code/blob/612afea/docs/results/time_task_analysis20261019/README.md)显示21分钟窗会排除37.49%原正对，因此当前不把切窗重构设为v2前置。长时无辅助预测的风险仍在，但018已有支撑区仍远差于简单融合，先验证基本恢复能否改善。

[实际输入支撑/距离评价](specs/input-support-evaluation.md)已接入正常完整评价，不改主指标、选模或模型调用；没有新真实全量分层结果，不为它另开GPU诊断。9001/reference数值复用和速度单位冲突见[数据复查](https://github.com/HKillua/association-code/blob/612afea/docs/results/dataset_followup018/README.md)：保留原值和尾部，不擅自换单位或用已有观测零误差证明降噪。

本地模型及评价工作已收尾，CPU验证与独立复核完成；最新仓库清理的验证范围见[历史整理记录](history/README.md)，这些检查只证明软件行为。用户已同意在参考24h＋候选合计4h上限内拆任务；候选teacher1000、utility200只训练一次，四臂各300独立运行。分拆后的保守旧速代理约3.355h，4h是上限而非完成保证。下一步019核实CPU服务托管后，按020—025依赖队列执行，不挪旧额度、不重复成本测量或制造新框架；旧预算null草案保持历史，不执行。

旧环境、013/014/015/016及被替代计划统一见[历史摘要](history/README.md)。旧回执和机器证据保存在Git历史，取消SHA门槛、动态GPU/共卡、每6小时跟进等现行权限不变。
