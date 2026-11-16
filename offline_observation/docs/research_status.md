# 离线观测研究状态

更新于 2026-11-16。执行状态见[交接](../HandOver/Handover.md)，旧结果按[研究历史](history/README.md)查阅。

## 目标与协议

在同一冻结系统上提高同源接续（ITSA）、跨源关联（MSTSA）及经纬度、速度、航向的融合与恢复精度，同时报告覆盖、重复、位置 P99 和场景／组件差异。TMTAFF 是论文主对照，目标是用双方完整输出同场比较。当前只用 MTAD；Level1 的 T2100、V300 共享部分身份，测试 600 场封存。源换序和可变 K 是设计目标，真实证据主要来自 K=2。 [数据约定](data_contract.md) · [指标定义](mtad_data_protocols.md) · [TMTAFF 原文](https://www.mdpi.com/2076-3417/16/14/7251)

## 当前证据

- **TMTAFF 作者协议。** 论文表 1–3 是复现目标；场景划分、配准细节及表 3 mESD 实现仍有未决项。 [协议核验](reports/2026-11-12-tmtaff-protocol-gap-audit.md) · [C147 作者协议材料](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c147_tmtaff_author_protocol_20261114.md)
- **TMTAFF 的本地整链适配。** C147 在原 V300 运行自身 TCM、TAM 和 TFM，与 Qhard 同场。新算的 4,439,092 个 TAM 分数逐值重现旧判定，原阈值接受 99.973% 的候选边，290 场只产出一组，位置覆盖 3.356%。这是现有适配的退化证据。 [C147 同场结果](reports/2026-11-14-offline-tmtaff-own-chain-c147.md)
- **配套对照。** C145 保留传统接续／DTW 和 WS 对照；C146 保留同配额全局指派对照，显式空配的误差和覆盖取舍留在原报告。PTFM／DeepAF 的实测范围是固定 WS 组后的融合，152A/B 待按原合同收尾。 [C145](reports/2026-11-14-offline-baseline-common-c145.md) · [C146](reports/2026-11-14-offline-baseline-assignment-c146.md) · [152A/B](../HandOver/Handover.md#待核验与保全)

## 下一步

从[已定位的基线选择断点](reports/2026-11-14-offline-baseline-restart-audit.md)继续。先证据化 TMTAFF 的训练侧窗口及配准、TAM 分数分离度及阈值、TFM 原组位置读出，每次只判别一个可观测解释。并行核同源和跨源候选论文实际任务、输出与协议，保留能做可信共同实测的路线。验证前按完整流程形成来源分析、矩阵连接和 Idea，服务器执行使用对应合同。 [完整流程](research_workflow.md) · [Mode C2.64](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-264) · [离线资料库](/Users/a1/Projects/biye-ronghe/offline_observation/README.md)
