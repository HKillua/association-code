# 2026-11-16 离线研究清扫

本次以飞书《视频文案》和新链接所指的补充页里的诊断、实验对照与复盘要求回看已有工作。后者第 2、5 节要求先用可观测量呈现异常，再用对照区分解释；实验明确控制变量，控制诊断代码及运行的投入。采用日期及项目落地见[离线完整流程](../research_workflow.md)。 [视频文案](https://my.feishu.cn/wiki/OHs3w9meHiQ3OXk4XC7ctFPXnxd) · [飞书目录](https://my.feishu.cn/wiki/YOcWwbmpHihq8NkAMKWcfBWXnO7) · [诊断与实验补充页](https://my.feishu.cn/wiki/TWXBw9US4iloe8kN2IpcAFiBnyc)

| 核查对象 | 可证范围及处理 | 原件 |
|---|---|---|
| 10 月 17–18 日基线选择 | 最早可证断点。当时先做 TSA-cTFER 适配和 TMTAFF 训练入口，尚未有多套可信基线的共同协议整链实测。回退到来源筛选、复建和公平实测起点；10 月 27 日采用的完整流程用于回看。 | [基线回退审计](../reports/2026-11-14-offline-baseline-restart-audit.md) |
| C46 迁移筛选 | 缺与 C45／C47 的可比投入筛选、完整六维判断和最终 Idea／训练合同。作为未执行的条件分析留档。 | [迁移流程复盘](../reports/2026-10-28-cross-transfer-workflow-review.md) |
| 102、C09 | 102 是指标合同，追加尾部属于事后探索；C09 Idea 未执行。原口径、证据及合同保留。 | [10 月 30 日研究过程](2026-10-30-research-process.md#判断边界) · [102 合同](../../HandOver/L2S-20261026-102.md) |
| 096A、107、113、152 | 分别保留自身 TAM 退化、开发诊断、借用旧关联、固定 WS 的融合证据。152A/B 完成执行，规定收尾待核。 | [基线审计](../reports/2026-11-14-offline-baseline-restart-audit.md) · [152 交接](../../HandOver/Handover.md#待核验与保全) |
| C128 Autoferry | 外部数据资格运行先于当天 MTAD-only 约束落地，作为外部历史证据。 | [C128 原回执](../../HandOver/S2L-20261113-c128.md) |
| C143、C144 | 两项 Idea 未执行，已退出活跃队列。 | [C143](../proposals/association_paired_supplier_certificate_20261114.md) · [C144](../proposals/fusion_pair_displacement_constraint_20261114.md) |
| C145–147 | 事前 Idea、执行合同和事后回流可追溯。C147 的启动失败和后续科学结果各有记录；共同协议退化和作者论文证据按原报告使用。 | [C145](../reports/2026-11-14-offline-baseline-common-c145.md) · [C146](../reports/2026-11-14-offline-baseline-assignment-c146.md) · [C147](../reports/2026-11-14-offline-tmtaff-own-chain-c147.md) |

其他历史记录沿用各自证据边界。事前材料无法核实时保留未知，欠缺原记录的步骤保留缺项，不补造预注册。开始一次新研究时，从[当前状态](../research_status.md)和[交接](../../HandOver/Handover.md)进入；按当前问题选读历史证据。

## 原件与恢复

清扫前的代码版本为 `20b263fd732022f456bb0891a4eb8f2e4250cb4b`。研究推导、矩阵、Idea 正文、失败和研究报告保留各自入口。旧导航的全文、数据口径和链接可在冻结版本按原目录浏览。 [旧研究状态](https://github.com/HKillua/association-code/blob/20b263fd732022f456bb0891a4eb8f2e4250cb4b/offline_observation/docs/research_status.md) · [旧合同总索引](https://github.com/HKillua/association-code/blob/20b263fd732022f456bb0891a4eb8f2e4250cb4b/offline_observation/HandOver/README.md) · [旧结果总索引](https://github.com/HKillua/association-code/blob/20b263fd732022f456bb0891a4eb8f2e4250cb4b/offline_observation/HandOver/results/README.md)

退役 16 个只服务于结案实验的分析脚本及 14 份专用测试。冻结原件逐文件与清扫前一致。原合同、命令、研究报告及结果原件仍可查；需要重跑历史现场时，从冻结提交另开工作目录。 [脚本、测试与复现定位](../../tools/analysis/archive-20261116.md)

本地可用 `git show 20b263fd732022f456bb0891a4eb8f2e4250cb4b:offline_observation/HandOver/README.md` 读取完整合同导航；其他原件按同一提交和原路径查阅。

资料库首页原件在原目录的[压缩副本](/Users/a1/Projects/biye-ronghe/offline_observation/README-2026-11-16-pre-cleanup.md.gz)，原相对链接以原目录为基准。离线资料库继续负责保全原文和独有证据。
