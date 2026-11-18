# 离线观测研究状态

更新于 2026-11-18。执行状态见[交接](../HandOver/Handover.md)，旧结果按[研究历史](history/README.md)查阅。

## 目标与协议

在同一冻结系统上提高同源接续（ITSA）、跨源关联（MSTSA）及经纬度、速度、航向的融合与恢复精度，同时报告覆盖、重复、位置 P99 和场景／组件差异。TMTAFF 是论文主对照，目标是用双方完整输出同场比较。当前只用 MTAD；Level1 的 T2100、V300 共享部分身份，测试 600 场封存。源换序和可变 K 是设计目标，真实证据主要来自 K=2。 [数据约定](data_contract.md) · [指标定义](mtad_data_protocols.md) · [TMTAFF 原文](https://www.mdpi.com/2076-3417/16/14/7251)

## 当前证据

- **TMTAFF 作者协议。** 论文表 1–3 是复现目标；场景划分、配准细节及表 3 mESD 实现仍有未决项。 [协议核验](reports/2026-11-12-tmtaff-protocol-gap-audit.md) · [C147 作者协议材料](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c147_tmtaff_author_protocol_20261114.md)
- **TMTAFF 的本地整链适配。** C147 在原 V300 运行自身 TCM、TAM 和 TFM，与 Qhard 同场。新算的 4,439,092 个 TAM 分数逐值重现旧判定，原阈值接受 99.973% 的候选边，290 场只产出一组，位置覆盖 3.356%。这是现有适配的退化证据。 [C147 同场结果](reports/2026-11-14-offline-tmtaff-own-chain-c147.md)
- **配套对照。** C145 保留传统接续／DTW 和 WS 对照，C146 保留同配额全局指派和显式空配对照。152A/B 已从原训练日志、末轮权重和 V300 数组闭环。同 441,144 个位置点，147、PTFM＋Mean、DeepAF 的 MAE 为 590.760、738.682、1,361.695 米。PTFM＋Mean 相对 147 的 RMSE 低 437.139 米，位置误差增加超过 1 公里的有 20,750 个点。 [C145](reports/2026-11-14-offline-baseline-common-c145.md) · [C146](reports/2026-11-14-offline-baseline-assignment-c146.md) · [152A/B 独立复算](reports/2026-11-18-peer-endpoints-and-tam-fit.md)
- **C148 端点选择诊断。** 原 fit 的 76 个组件各首场，同图同 K 的行列势使跨源漏接/误接从 465/419 降至 445/401，但缺测处原有同船跨源支持毛丢失 121，代表身份数少 1；去最大后仍毛丢失 76。本冻结参数未通过事前保护条件，结案、不扩大。未测实际四属性及其覆盖，不作为融合改善或 TMTAFF 胜出的证据。 [Record / Squeeze](reports/2026-11-17-offline-endpoint-qualification-c148.md)
- **C149 航迹向回退诊断。** 全部原 fit 1,748 场／76 组件完成。复建 C74 B 的模型低速回退分支占共同评价点 13.05%，贡献绝对角误差 42.21%、超过 150° 大误差点 62.01%；去掉最大组件后份额仍为 32.08%／49.26%。去最大后有 39 个组件各至少 10 个此类大误差点，具备事后确认同身份的非首成员未复用区间。事前机会条件通过，但没有替换读出、没有证明方向证据正确，也不是新方法精度收益。 [Record / Squeeze](reports/2026-11-17-offline-heading-fallback-c149.md)

- **C150 单区间替换关闭。** 全域圆周 MAE 27.7310°→27.7322°，救回 47,874 尾点、新增 48,112，去最大仍净增 117。保留原航迹向 B。 [结案](reports/2026-11-18-offline-heading-interval-c150.md)
- **C151P 关联前局部位置平滑关闭。** 在拟合侧 76 个组件各首场固定同源链、候选和配额，跨源漏接／误接从 465／419 降到 461／412。原有同船支撑毛丢失 71 个缺测查询，恢复 160 个，去最大仍丢失 67 个。当前保留原跨源输入。 [结案](reports/2026-11-18-association-position-smoothing-c151p.md) · [代码与同行原文](reports/2026-11-18-association-preprocessing-review.md)
- **C151S 整块供给硬门关闭。** 667 个可见归零槽中，56 个原有同船支撑、611 个原先只有异船供给。另有 15 个同船损失被异船接替掩盖。整块否决拦下全部四次改配，包括场 1290 的安全改进。 [逐边诊断](reports/2026-11-18-c151s-swap-supply.md)
- **094F 拟合侧分离弱。** 在两臂训练见过的八个组件代表场，冻结端点对跨源自然候选的 AUC 为 0.48999 / 0.48056，去最大场为 0.49045 / 0.50202。两种人口的 AP 均接近先验。训练模式、时窗和配准的独立贡献待核。 [拟合侧分离度](reports/2026-11-18-peer-endpoints-and-tam-fit.md#094f--record)

## 下一步

- **P1，先查同一静态边承担的时段能否分开。** 在场 1229 的同一 K=1 配额下，撤出旧边损失 44 槽，新边能补 52 槽。下一项先固定原图、K 和代价，核旧／新边的需供时段是否冲突，再查所有改配块。场 0、1290 留作供体可信性的反例。验收同时看同船毛丢失、异船供给、漏接／误接和放过的安全改配。保护问题可辨后再单独测 ITSA 可达门或 MSTSA 时间门，短轨和下游原生观测全部保留。 [C151S 机制与反例](reports/2026-11-18-c151s-swap-supply.md)
- **P2，固定同批输入定位 TAM 的模式差异。** 下一次对照只改变训练态／推理态，保持权重、拟合侧自然配对和输入不变，同时看距离尺度、AUC、AP。若仍弱，再分别审查输入时窗、无标签配准或表示目标的生效条件。原 152A/B 留作固定 WS 的条件融合基准。 [断点及对照](reports/2026-11-18-peer-endpoints-and-tam-fit.md)
- **P3，先核航迹向方向是否可辨。** 生产读出保留 C74 B。新诊断先按报文能否夹住查询、时间偏移和位移相对同源位置噪声的大小分层，再记录多点弦向角的一致性与多源速度矢量是否相互支持。持续小位移且跨源一致列为低速候选，有显著位移和连续转向列为机动候选，跨源矛盾、尺度估不准或证据不足单列。先比较各层样本量及原 C74 B 尾点，保留纯组／混组、去最大组件和两方向越过 150° 的计数。未来门控若引入新尾点即触发原非尾点零增害保护。噪声尺度取自可见观测，真实船况和身份只作事后核验。本轮仅定义诊断口径，航迹向代码未改动。 [C149](reports/2026-11-17-offline-heading-fallback-c149.md) · [C150](reports/2026-11-18-offline-heading-interval-c150.md) · [127 固定组位置对照](reports/2026-10-29-native-state-review.md)

三项实际回填见 [C151S](reports/2026-11-18-c151s-swap-supply.md#mode-c-回填) 和 [152R / 094F](reports/2026-11-18-peer-endpoints-and-tam-fit.md)，历史来源见[六文献矩阵](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-11-16_maritime_mtad_reading/six-paper-next-step-audit.md)。后续实测按[完整流程](research_workflow.md)另立合同。
