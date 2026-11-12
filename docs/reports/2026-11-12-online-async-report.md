# 异步实报与预测不确定性的接回诊断

## Record

`EXPERIMENT_RECORD` v1.0，`REC-ONLINE-ASYNC-REPORT-20261112`，completed。`IDEA-ONLINE-ASYNC-REPORT-20261112` plan v1.0、C-ON40。问题是四端尚未齐时，单条实报对两组对侧预测的时距方差，能否在 KF 接回前区分错接与正确接回。固定同候选、同报文、同均值，拆开实报坐标、候选方差与高斯归一项。近期现场研究有源内校准、真实到达日志或检测范围等条件，分别保留其来源和适用范围。 [计划](../proposals/online_async_report_20261112.md) · [近期纯传感器原文分析](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_async_report_sources_20261112.md)

冻结提交 `4ac3f0f`。原 28 场、28 个共享 A 组件，24 场稀疏。逐场对账 C28、C36、C37、C39，原生事件 102,404、实报 102,485、候选边事件 940,693、风险窗 482 个，稀疏 236 个。评分只用批前四端预测、实报与因果前缀配准。无涉事新报不追加分数。捕获起点、严格窗内、原接回批，所有 28 份评分写盘后才读评价身份。原候选和发布保持。原生量测时刻作为理想到达，物理网络延迟未知。

```sh
.venv/bin/python -m online_observation.async_report_gap \
  --archive PART1 PART2 PART3 --manifest configs/data/mtad-level1-scene3000.json \
  --graph-run PRIVATE_C28_OUTPUT --gap-run PRIVATE_C37_CAPTURE \
  --timing-run PRIVATE_C39_CAPTURE --output-dir NEW_PRIVATE_DIRECTORY
.venv/bin/python -m online_observation.async_report_squeeze \
  --summary NEW_PRIVATE_DIRECTORY/summary.json --output NEW_PRIVATE_DIRECTORY/aggregate.json
```

Apple M2、Python 3.12.14，CPU 线程上限 2。北京时间 20:53:15–20:53:23，一次真实退出 0，墙时 8.80 秒、程序 7.334 秒、最大 RSS 245,710,848 B。零训练、选型、新场、V、位置参考和封存测试。112 项在线测试通过。对 1,477 个批次逐项核实每批恰一条涉事报文、全部时距严格大于 0 且不超过 60 秒，以及批次、窗边界和焦点对账；复算实报和屏蔽高斯分差共 2,954 项。 [真实日志](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_online_mtad/async_report_gap_attempt_1.log) · [匿名逐场](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_online_mtad/async_report_gap_attempt_1/summary.json) · [复核原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_online_mtad/async_report_gap_attempt_1/verification.json)

## Squeeze

`SQUEEZE_REPORT` v1.0，`SQ-ONLINE-ASYNC-REPORT-20261112`。56 个稀疏焦点边来自 28 个四端窗和两个组件，各窗接回了两条错边，另一组两条边均真。原窗中位 3.882 秒。起点均有一条涉事报文，严格起点后、接回前有新涉事报文的为 19 组。正确接回主对照 142 个四端窗，来自 9 个组件，67 组有严格后续实报；其中 48 个正确窗来自同两个目标组件，21 个有严格后续实报。较宽的正确接回口径共 146 窗。 [完整汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_online_mtad/async_report_gap_attempt_1/aggregate.json)

### 原接回前已有的最后一批证据

| 同刻读出 | 提前识别错接（28组） | 误拦正确接回（142组） | 平衡正确率 |
|---|---:|---:|---:|
| 继续保留 I22 | 0 | 0 | 50.0% |
| 继续保留接回前 KF 的另一组 | 28 | 142 | 50.0% |
| 四端同刻几何 | 17 | 56 | 60.6% |
| 四端同刻几何，屏蔽新报 | 13 | 27 | 63.7% |
| 单报，对侧共用方差 | 10 | 89 | 36.5% |
| 单报，对侧各自方差，无体积项 | 9 | 87 | 35.4% |
| 单报，对侧各自方差，高斯分差 | 9 | 87 | 35.4% |
| 单报高斯分差，同刻屏蔽新报 | 10 | 65 | 45.0% |

使用候选各自方差相对共用方差，在目标中增加 3 组、减少 4 组警报，在正确对照中增加 2 组、减少 4 组误拦。体积项对目标最终方向无影响，在对照各增减 1 组。实报相对高斯屏蔽少识别 1 个错接，多误拦 22 个正确接回。单报时共用方差只是正比例缩放，方向与单报欧氏逐批一致。

9 组有效警报涉及 18 个焦点边，最新证据到接回的中位时间 1.084 秒，合计 30.318 焦点边秒的提前量。若采用曾经出现正分即一直警报，命中可升至 22 组，误拦同时升至 110 组。只看起点后确有新报的组，最后一批识别 5／19 组、误拦 39／67 组。早期暂时正分和末次仍有正分分别计数。这些时长表示读到分数的时间，未实测缩链。

两个目标组件分别识别 6／21 与 3／7 组，对同组件的正确接回分别误拦 23／41 与 6／7。去掉前缀配准平移后仍是 9／28，误拦降至 74／142。宽口径为 9／28 与 89／146。单报跨源误差强依赖对齐，解除平移影响也未恢复提前辨别力。

### 接回批才到的新证据与方差失配

接回当批，高斯分差识别 16／28、误拦 36／142，共用方差为 16／28、37／142。相对严格接回前，高斯有 13 组目标新获正确方向、6 组失去，正确接回有 76 组撤销此前的误拦、25 组新遭误拦。28 个目标中 20 个的接回批来自与最后早期证据不同的源。共用方差也能读出大部分当批增量。报文更换和方向变化同时出现，来源切换的因果贡献未单独检验。接回当批的提前量为零。

稀疏同轨创新在 10–30 秒档有 689 条，15.1% 超过理想二维高斯的 95% 分位。跨源真边对已到报文的归一化创新，在预测时距 ≤10 秒的 375 条中有 61.9% 越界，在 10–30 秒的 193 条中有 64.2%；更长档只有 1 条。现有方差来自同一源内模型，跨源残差可能同时受偏差、航迹处理、预测均值及误差相关影响。该统计核模型适用性，无独立物理位置或分源噪声校准，不能把差异归到时距方差本身。

**结案。** 最新严格可用读出低于同刻屏蔽、共同方差和保留 I22。两目标组件均有触及，但分别被更多同组件误拦抵消。原 28 场证据未达到新留场门槛。本轮未开启新场或同延迟多假设发布，原正确配对和首报保持、长错链真实变化仍待实际验证。 [Mode C v43](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v43-异步提前证据回流)
