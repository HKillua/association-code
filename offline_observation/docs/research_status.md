# 离线观测研究状态

更新于 2026-11-18。执行状态见[交接](../HandOver/Handover.md)，旧结果按[研究历史](history/README.md)查阅。

## 目标与协议

在同一冻结系统上提高同源接续（ITSA）、跨源关联（MSTSA）及经纬度、速度、航向的融合与恢复精度，同时报告覆盖、重复、位置 P99 和场景／组件差异。TMTAFF 是论文主对照，目标是用双方完整输出同场比较。当前只用 MTAD；Level1 的 T2100、V300 共享部分身份，测试 600 场封存。源换序和可变 K 是设计目标，真实证据主要来自 K=2。 [数据约定](data_contract.md) · [指标定义](mtad_data_protocols.md) · [TMTAFF 原文](https://www.mdpi.com/2076-3417/16/14/7251)

## 当前证据

- **TMTAFF 作者协议。** 论文表 1–3 是复现目标；场景划分、配准细节及表 3 mESD 实现仍有未决项。 [协议核验](reports/2026-11-12-tmtaff-protocol-gap-audit.md) · [C147 作者协议材料](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c147_tmtaff_author_protocol_20261114.md)
- **TMTAFF 的本地整链适配。** C147 在原 V300 运行自身 TCM、TAM 和 TFM，与 Qhard 同场。新算的 4,439,092 个 TAM 分数逐值重现旧判定，原阈值接受 99.973% 的候选边，290 场只产出一组，位置覆盖 3.356%。这是现有适配的退化证据。 [C147 同场结果](reports/2026-11-14-offline-tmtaff-own-chain-c147.md)
- **配套对照。** C145 保留传统接续／DTW 和 WS 对照；C146 保留同配额全局指派对照，显式空配的误差和覆盖取舍留在原报告。PTFM／DeepAF 的实测范围是固定 WS 组后的融合，152A/B 待按原合同收尾。 [C145](reports/2026-11-14-offline-baseline-common-c145.md) · [C146](reports/2026-11-14-offline-baseline-assignment-c146.md) · [152A/B](../HandOver/Handover.md#待核验与保全)
- **C148 端点选择诊断。** 原 fit 的 76 个组件各首场，同图同 K 的行列势使跨源漏接/误接从 465/419 降至 445/401，但缺测处原有同船跨源支持毛丢失 121，代表身份数少 1；去最大后仍毛丢失 76。本冻结参数未通过事前保护条件，结案、不扩大。未测实际四属性及其覆盖，不作为融合改善或 TMTAFF 胜出的证据。 [Record / Squeeze](reports/2026-11-17-offline-endpoint-qualification-c148.md)
- **C149 航迹向回退诊断。** 全部原 fit 1,748 场／76 组件完成。复建 C74 B 的模型低速回退分支占共同评价点 13.05%，贡献绝对角误差 42.21%、超过 150° 大误差点 62.01%；去掉最大组件后份额仍为 32.08%／49.26%。去最大后有 39 个组件各至少 10 个此类大误差点，具备事后确认同身份的非首成员未复用区间。事前机会条件通过，但没有替换读出、没有证明方向证据正确，也不是新方法精度收益。 [Record / Squeeze](reports/2026-11-17-offline-heading-fallback-c149.md)

- **C150 单区间替换关闭。** 全域圆周 MAE 27.7310°→27.7322°，救回 47,874 尾点、新增 48,112，去最大仍净增 117。保留原航迹向 B。 [结案](reports/2026-11-18-offline-heading-interval-c150.md)

## 下一步

- **C151P，关联前的局部位置平滑。** 先在 C148 的拟合侧 76 个组件各首场上固定候选、同源链和各块匹配数，比较原 DTW 与关联专用三点平滑。报告原分区指标、曲线及原同船支持毛损失。 [代码与原文复核](reports/2026-11-18-association-preprocessing-review.md) · [事前设计](proposals/association_position_smoothing_c151p_20261118.md)
- **P1，轨对评分重选。** 保留 C148 的原图、链、K及其已见端点空间，原行列势参数继续结案。新评分与可见支持保护须分离变量；可见支持不减少不等于真同船供给零丢，不把原毛丢失守线改成净增即可。当前仅建立后续连接，尚无新关联 Idea 或 READY。
- **P2，基线账本与断点。** [原件待办已核](reports/2026-11-17-offline-baseline-ledger-next-actions.md)。152A/B 复用原日志、末轮权重、逐场数组和终态补齐原合同收尾，不重训；旧代码每轮覆盖同一个检查点，不能假定历史各轮权重都存在。TMTAFF 先查原训练账本，新增训练侧分离度、配准或时窗实验另立 Idea，不在 V300 选参。

C150 已在原 fit 1,748 场结案。C151P 使用该拟合侧固定 76 场。来源与 P/F/M/E/C 见[矩阵](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-11-16_maritime_mtad_reading/six-paper-next-step-audit.md)。

TMTAFF 仍从[基线选择断点](reports/2026-11-14-offline-baseline-restart-audit.md)继续核训练侧窗口、配准、TAM 分数分离度和 TFM 原组位置读出。当前C147退化适配不能代表可信作者复现。各次运行遵循[完整流程](research_workflow.md)，经 Record→Squeeze 后回填矩阵。
