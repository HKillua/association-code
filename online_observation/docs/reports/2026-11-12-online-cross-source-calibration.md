# 接回换源的预测均值、方差与历史参照

## Record

`EXPERIMENT_RECORD` v1.0，`REC-ONLINE-CROSS-SOURCE-20261112`，completed。`IDEA-ONLINE-CROSS-SOURCE-20261112` plan v1.0、C-ON41。检验20个接回批换源组的有符号跨源误差、以及独立于涉事四端的历史参照能否在原接回前改善决策。 [冻结计划](../proposals/online_cross_source_calibration_20261112.md) · [近期纯传感器论文分析](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_cross_source_calibration_sources_20261112.md)

冻结提交 `8bf8396`。原28场、28个共享 A 组件，24场稀疏。逐批对账 C28 候选、C37 原窗和 C40 的实报及屏蔽分差。102,404 原生事件、102,485 实报、940,693 候选边事件、482 个窗（稀疏236）。1,477 个带符号报文读出先冻结再读 A-B，原候选和发布保持。时钟使用原生量测作为理想到达。Apple M2、Python 3.12.14、CPU 线程上限2，21:19:41–21:19:49 一次真实退出0，墙时7.95秒、程序6.880秒、最大 RSS 278,380,544 B。零训练、选型、新场、V、位置参考或封存测试。117项在线测试通过。 [真实日志](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/cross_source_calibration_attempt_1.log) · [匿名逐场](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/cross_source_calibration_attempt_1/summary.json)

```sh
.venv/bin/python -m online_observation.cross_source_calibration_gap \
  --archive PART1 PART2 PART3 --manifest configs/data/mtad-level1-scene3000.json \
  --graph-run PRIVATE_C28_OUTPUT --gap-run PRIVATE_C37_CAPTURE \
  --async-run PRIVATE_C40_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
.venv/bin/python -m online_observation.cross_source_calibration_squeeze \
  --summary NEW_PRIVATE_DIRECTORY/summary.json --output NEW_PRIVATE_DIRECTORY/aggregate.json
```

独立复核2,954个带符号分解、2,244次图锚引用和272次共同位移锚引用。锚资格批与写入实报严格分开，锚早于读取，配准阶段、报告方向、两候选时距差和片段排除均符合冻结规则。标签仅在私有评分落盘后用于评估。 [复核](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/cross_source_calibration_attempt_1/verification.json)

## Squeeze

`SQUEEZE_REPORT` v1.0，`SQ-ONLINE-CROSS-SOURCE-20261112`。20 个目标窗来自两个组件（16／4），覆盖40个焦点边。14组换源时仍在观测同一真目标，6组同时换目标。接回前最新报告只来自4个真片段对，当批来自5个。匹配76个换源正确接回（8个组件），其中22个来自目标同组件。8／66个不换源窗和原28／142总体另列。 [聚合原件](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/cross_source_calibration_attempt_1/aggregate.json)

### 跨源误差以分层均值差为主

| 20个换源错接组 | 最后严格接回前 | 接回当批 |
|---|---:|---:|
| 真对侧预测时距中位 | 18.702秒 | 0.817秒 |
| 真对侧残差模长中位 | 1,623 m | 1,702 m |
| 源内创新模长中位 | 325 m | 255 m |
| 名义每轴标准差中位 | 203 m | 172 m |
| 归一化跨源创新中位 | 66.26 | 90.18 |

将源方向统一后，两目标组件均值指向近乎相反。按组件和报告方向分层，均值能量占真边平方残差97.7%／97.7%，对应分层均值 RMS 1,629／1,676 m、中心化 RMS 251／258 m。统一方向整体去均值仍有19／20越过名义二维95%分位；各组事后各自去均值仅1／20越界，两时点相同。两源批前预测位置差解释主要均值，源内创新的均值向量明显更小。去中心化是同样本描述，未验证方差已校准。正确接回同组件的真边残差中位还达1,691／1,829 m，故偏移大小不能识别错接。原坐标只提供合成跨源误差，单源物理偏差、参考点和相关噪声仍不可拆。

### 锚的供给、留出误差和接回对照

全场共写入47,240个历史单报残差，418个具共同位移资格。20目标的38个严格接回前读出中，36次各有一个外部原图唯一锚，2次没有，始终无两个可用。其中1个较早批有一个共同位移锚，提前2.507秒；最新和接回批的共同位移锚均为0。最新有原图唯一锚18／20，均在60秒内，事后未见错锚或与涉事身份重合。接回批20／20有一个原图锚，仍无双锚或方差样本。76个正确对照最新有40组达双锚、5组达三锚。其双锚均值臂使留出真报 RMSE 770→527 m，误拦26→23，目标同组件双锚只有2个正确组且误拦0→1。目标没有资格复现这个收益。

按原方案的最新严格证据，实报识别7／20、误拦47／76，同刻屏蔽为8／20、33／76，继续保留 I22 为0／20、0／76。接回批实报为10／20、19／76，提前量为0。没有目标双锚或三锚评分，故无法确认事前的均值或方差臂。

**只读事后单锚探针。** 把冻结池里唯一图锚的有符号残差作为相对均值，原方差保持。18个有锚错接组的留出真边 RMSE 1,618→410 m，18组全部改善，命中7→13，警报中位提前0.550秒。只来自两对外部片段，分别用于16和2个目标窗，新增命中均在第一个组件。32个同样仅有单锚的正确接回组，RMSE 1,555→559 m，误拦18→15；同组件20个同资格正确窗误拦11→8。将对照扩展到所有76个正确接回、凡有锚都用最近一个且无锚保留，误拦38／76，仍超过13／20的目标命中数。这说明已有锚可部分预测相对均值，关联保护仍不足。探针读取固定结果，未回放、重选原图或改主判据。 [事后单锚探针](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261112_online_mtad/cross_source_calibration_attempt_1/single_anchor_probe.json)

**结案。** 原双锚和中心化方差资格均未达到。单锚是后验候选，独立见证数少且正确接回保护不足。隔离的新留场未启用，同延迟发布、原正确配对、首报和真实长错链变化尚未验证。 [Mode C v45](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v45-相对均值与单锚证据回流)
