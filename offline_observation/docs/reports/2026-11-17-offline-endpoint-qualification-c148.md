# C148 · 固定跨源链配额的端点资格

## Record

`EXPERIMENT_RECORD REC-OFFLINE-ENDPOINT-QUALIFICATION-C148-20261117`，`research-pipeline-contract v1.0`。包 ID 与实验 ID 相同，生成于 2026-11-17，执行状态 `completed`，证据状态 `verified`（本轮执行与回传统计）。上游 `IDEA-OFFLINE-ENDPOINT-QUALIFICATION-C148-20261116` 计划 1.0，`MODE-C-OFFLINE-20261116-QUALIFICATION` 的 C148／P1／F1、F2、F4／M1／E1、E2、E4。未取得独立逐块复核及四属性融合效果。下一环节为本页 Squeeze。 [Idea](../proposals/association_endpoint_qualification_c148_20261116.md) · [READY](../../HandOver/L2S-20261116-c148.md) · [来源矩阵](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-11-16_maritime_mtad_reading/six-paper-next-step-audit.md)

研究问题是固定跨源候选和匹配数后，空槽的行列势能否换入和换出链端点，同时减少漏接和误接并保留原有同船供给。一次确定性配对诊断。实际从 MTAD Level1 原 C92 fit 1,748 场／76 共享 A 组件中，按冻结原名单取每个组件第一场，共 76 场。最大组件在完整拟合侧含 966 场，此处只占场 3 一场。逐场绑定输入端 C151 `T_core` 量测缓存与原量测，重建 C145 `END_DTW` 同源链和 C146 原 DTW 图。所有场沿用至少 2 个共同查询、同步距离不超过 10 km 与 DTW 不超过 5 km，逐候选连通块用原贪心的 K。`RAW_FIXED` 用原 DTW，`POTENTIAL_FIXED` 在可换块用同一精确求解器加空槽行列势。温度 500 m，空槽分数 −5,000 m，固定 20 次迭代。求解使用转为实数质量和 S01 末步按行归一之前的对数行列势。全部 76 场预测与哈希落盘后，才打开 C151 同场评价身份。 [冻结输入与预测清单](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/freeze.json) · [代码 2008088](https://github.com/HKillua/association-code/blob/20080885fa85313b7a40934369e335aee47c05c6/offline_observation/tools/analysis/association_endpoint_qualification.py)

实际版本 `20080885fa85313b7a40934369e335aee47c05c6`。Campus CPU23 单线程，空 CUDA，systemd 用户服务 `mtad-f8878ea8d93c.service`，Linger=yes，Restart=no。北京时间 2026-11-17 00:02:45.963 至 00:03:16.594，科学用时 30.631 秒，托管用时 33.048 秒，退出均为 0。一次尝试，76 场全部预测及评分，未触发 1,800 秒时限提示。已回传记录中未见合同偏差或样本排除。原生点迹、身份和逐块索引留 Campus，回传匿名逐场统计。训练、参数搜索、校准、原 held、V300 与封存测试均未执行。四属性误差未测，原生支持和首组代表数属于关联诊断。 [实际命令](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/launcher/request.json) · [科学终态](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/run.json) · [托管终态](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/launcher/status.json)

| 指标 | 全部 76 场，原 DTW → 行列势 | 去最大后 75 场，原 DTW → 行列势 |
|---|---:|---:|
| 候选块 / 可换端点块 / 实际换端点块 | 312 / 41 / 12 | 294 / 35 / 10 |
| 有端点变化的组件 | 10 | 9 |
| 同源 TP / FP / FN，两臂一致 | 677 / 179 / 99 | 501 / 104 / 63 |
| 跨源 TP / FP / FN | 1,090 / 419 / 465 → 1,110 / 401 / 445 | 878 / 213 / 260 → 886 / 206 / 252 |
| 跨源微 F1 | .71149 → .72407 | .78780 → .79462 |
| 代表身份数 / 重复组 | 802 / 51 → 801 / 52 | 581 / 39 → 581 / 39 |
| 任意跨源供给节点×Q，全节点查询 | 81,267 → 81,604 | 61,781 → 61,783 |
| 任意跨源供给节点×Q，本节点缺观测 | 35,219 → 35,166 | 25,821 → 25,635 |
| 同船跨源供给节点×Q，本节点缺观测 | 24,047 → 24,489 | 19,843 → 19,988 |
| 同船跨源供给，本节点缺观测，毛丢 / 恢复 | **121 / 563** | **76 / 221** |
| 固定 K 的最大可选真片段对，标签上界 | 1,373 | 1,018 |

同船供给以另一来源的同身份片段在该节点对应查询有原生支持来计。毛丢是原 DTW 有此支持、行列势没有的节点×Q。毛丢与恢复分别报告，不用净增抵消。跨源 TP/FP/FN 计最终分区内由链对展开的原片段对，和匹配链的数量不同。观察为单次确定性结果，只描述每组件首个拟合场，不估计总体效果、随机种子方差或统计显著性。 [终态统计](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/summary.json) · [76 场逐场原件](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/scenes.json.gz)

## Squeeze

`SQUEEZE_REPORT SQ-OFFLINE-ENDPOINT-QUALIFICATION-C148-20261117`，`research-pipeline-contract v1.0`。包 ID 与深挖 ID 相同，生成于 2026-11-17，输入 `REC-OFFLINE-ENDPOINT-QUALIFICATION-C148-20261117`，证据状态 `partial`。证据覆盖这次固定参数、两源、固定配额的拟合侧诊断。缺失证据包括跨参数或独立人群验证、逐候选原件的独立复核、同场融合四属性与位置 P99，及作者协议复现。未开原数据补跑。下一步 `deep-read Mode C`。

- **数据与账本（DE）。** 从本地回传逐场记录重汇总 76 场及去最大 75 场，61 项整数计数与摘要一致，核对了支持得失和换边恒等式。原 manifest 和 C92 `frozen-split.json` 的 SHA-256 等于冻结清单，按名单复算每组件首场与逐场顺序一致。量测、预测和评价三份哈希各有 76 项与场景清单对应。私有预测与原生点迹仍由 Campus 留存，复核范围是回传记录。各场边际绝对残差的最大值是 .002159（场 845），为固定 20 次迭代的实测残差。 [冻结清单](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/freeze.json) · [逐场原件](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/scenes.json.gz)
- **可作用范围与机制（DE/AI）。** 41/312 块（13.14%）存在同配额端点可换空间，来自 32 场，含 143 个可换入端点，涉及 219 原片段及 5,931 链×Q 支持。固定行列势最终改变 12 块（全部 3.85%，合格块 29.27%），移入和移出均为 22 个端点。换入、换出各 31 条链链接，换入展开 31 真边和 38 假边，换出展开 11 真边和 56 假边。去最大后仍换 17 条，换入 15 真／18 假，换出 7 真／25 假。可换空间和实际效应都存在，作用范围受同 K 和候选形状限制。真边上界比行列势实际真边多 263 条（去最大 132 条）。这是允许伴随假边且不约束支撑的标签上界。 [终态统计](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/summary.json)
- **原生供给的增害有跨组件证据（DE）。** 有端点变化的 10 场中，8 场漏误同降，1 场同升，1 场两项净值均持平。4 场共出现 121 个缺观测处的原生同船支持毛丢失，其中 3 场在漏误同步改善时仍丢支持。场 3 少漏 12、少误 11，同时丢 45 个同船支持并少 1 个代表。去掉场 3 后仍有场 0、91、3055 的 9、44、23 个毛丢失。场 3055 同时增加 4 个漏接和 5 个误接，没有对应同船支持恢复。全场任意跨源支持净增 337，但在本节点缺观测处减少 53，去最大后该缺观测区减少 186。这些方向分化支持保留逐点毛丢和代表守线。 [逐场原件](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/scenes.json.gz)
- **边界与后续回流（AI/UN）。** 相同 K、相同图和同一求解器让这一次变动可归于加入行列势及其端点交换。原 DTW 的链距离和行列势均不直接评价逐时同船供给。观测与这一目标错位相符，尚无独立证据将增害进一步归因于源偏、时距或个别势值。拟合侧最大组件其余 965 场和其他角色都未覆盖，结论限于预先固定的一个温度／空槽组合。 [计算入口](../../tools/analysis/association_endpoint_qualification.py) · [已定停止线](../proposals/association_endpoint_qualification_c148_20261116.md)

**按事前停止线，C148 当前配方结案。** 去最大后 9 个组件有实际替换，且全域和去最大均减漏、减误。然而缺观测处同船毛丢失要求为零，实测为 121 和 76，且全域代表身份数少 1。`qualifies_for_larger_fit_diagnostic=false` 与事前规则一致。回填矩阵的新现象是同配额净减误漏可在同场伴随原有同船供给丢失。新问题是轨对评分能否识别换出边供给的不可替代性。待验证机制为行列势和原距离目标无法区分供给的时间与身份条件。保留这些证据，当前配方不扩大到全拟合侧、校准、留出或 TMTAFF 对比。 [停线与假设](../proposals/association_endpoint_qualification_c148_20261116.md) · [结案标志](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261117_offline_c148_endpoint_qualification/summary.json)
