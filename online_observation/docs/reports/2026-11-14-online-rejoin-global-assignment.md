# MTAD 原接回批的局部翻转与全局指派

## Record

`EXPERIMENT_RECORD` v1.0，`REC-ONLINE-REJOIN-GLOBAL-ASSIGNMENT-20261114`，completed。C-ON56，消费 `IDEA-ONLINE-REJOIN-GLOBAL-ASSIGNMENT-20261114`。原 28 场 MTAD，482 风险窗、416 合法接回、31 原合法双报窗。依 C55 原缓存分差固定 13 次局部翻转及 18 次未翻转，叠加原 5 次新报推错、5 次原屏蔽几何已错和唯一新增真命中。原候选、同批时钟和非触发边缓存代价保持。按 C28 原源对全局部分指派、双侧空配、同优解处理及分组发布重解。实报重解逐窗匹配全部原发布边。以严格批前预测替换触发坐标，分别重算全部触发关联边、仅重算固定两边，另有相同空配的四端局部对照。原图不重新门控。全场先冻结路径及边，再按 C55 匿名标签和 MTAD A-B 连接身份。 [冻结设计](../proposals/online_rejoin_global_assignment_20261114.md) · [近期文献分析](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_rejoin_global_assignment_sources_20261114.md)

事前合同 `11b8d92`，冻结实现 `3dc12f2`。Apple M2，Python 3.12.14，四个数值线程环境变量均为 2。一次本地 CPU，退出 0，墙时 5.14 秒，峰值 RSS 254,164,992 B。186 项在线测试通过，含 4 个新增专项。内部冻结形成 32 窗，按原 C55 资格评价 31 窗。未训练、选型、读取外部数据、新留场或封存测试，也未推进下一批状态或改真实发布。 [复现命令](../../HandOver/L2S-20261114-online-rejoin-global-assignment.md) · [运行日志](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261114_online_mtad/rejoin_global_assignment_attempt_1.log) · [匿名逐窗](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261114_online_mtad/rejoin_global_assignment_attempt_1/rows.json) · [汇总](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261114_online_mtad/rejoin_global_assignment_attempt_1/summary.json)

近期原文将局部代价、互斥选择、空配和身份保留分开。Tao 等的多雷达方法以互斥指派优化总成本，Shi 等在实测 AIS／雷达上使用带虚拟雷达的匹配，两者的成本和空配约定有别于 MTAD。IEEE TSP 的目标到雷达单元边缘概率和 IEEE TAES 的类别辅助全局关联都区分真假、漏检及定位。C56 的同图对照保持原求解，检验局部代价干预是否传到发布边。 [Tao 等](https://doi.org/10.3390/app16073159) · [Shi 等](https://doi.org/10.3390/s26072255) · [Kropfreiter 等](https://doi.org/10.1109/TSP.2026.3659742) · [Bai 等](https://doi.org/10.1109/TAES.2024.3491936)

独立求解复核重新读入 7 场原始报文，在 40,340 批中恢复严格批前预测。用部分匹配动态规划重解 96 次全图，均与冻结边集一致，无精确最优并列。独立身份复核重新连接原 MTAD A-B，复核 648 次边标签、2,892 次新动作与回退、1,008 个聚合统计，结果一致。 [求解复核](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261114_online_mtad/rejoin_global_assignment_attempt_1/solver_verification.json) · [身份与汇总复核](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261114_online_mtad/rejoin_global_assignment_attempt_1/outcome_verification.json)

## Squeeze

`SQUEEZE_REPORT` v1.0，`SQ-ONLINE-REJOIN-GLOBAL-ASSIGNMENT-20261114`。所有 31 窗的冻结图都有两源，原实报均选回两条原配对。触发边 83 条，全部重算代价。1 条外围边因代价过双侧空配价失去竞争资格，原 HOLD 图边仍保留。四端所在原图连通分量有 4–22 个顶点，中位 6。局部翻转组有 4–9 个顶点，中位 6，分布在 4 个共享 A 组件。

| 固定分层 | 窗数 | 原焦点接回中止 | 完整换成另一组 | 完整保留原组 | 仍接回但另一原边变化 |
|---|---:|---:|---:|---:|---:|
| C55 局部排序翻转 | 13 | 13 | 13 | 0 | 0 |
| C55 局部排序未翻转 | 18 | 0 | 0 | 17 | 1 |
| 其中 5 次新报推错 | 5 | 1 | 1，错误 | 4 | 0 |
| 其中 5 次原几何已错 | 5 | 2 | 2，错误 | 3 | 0 |
| 其中唯一新增真命中 | 1 | 1 | 1，正确 | 0 | 0 |

**局部翻转 13 窗。** 四端局部求解、只遮固定两条边的全图求解、全触发边遮蔽三者在这 13 窗均给出同一个完整备选。换配 3 次为真，10 次把两条原本正确的边换成假边。按原图连通分量计，增加 6 真边和 20 假边，移除 20 真边和 6 假边。13 窗没有外围发布边变化或四端空配。C55 的分差翻转确实传到了全局选择，身份真值主要向坏处变化。

**局部未翻转 18 窗。** 全触发边遮蔽保留全部焦点边。1 个混合真值窗保留唯一原焦点边，非焦点原边换掉，连同一条完全外围的原假边一起移除。新配两条接外围的边，1 真、1 假，连通分量净多 1 真、少 1 假，四端无空配。该结果在只遮固定两边时同样出现，说明原一对一约束能把局部成本干预传到外围。另一个稀疏窗只遮两条固定边会失去一条原边并牵动外围，遮全部触发边时完全恢复实报边集。该窗仅遮两边会额外移除 1 真并新增 2 假，说明只改组内边会遗漏部分外围代价联动。两类遮蔽在 31 窗对是否仍有焦点边的判断逐窗相同，完整边集只在这一个稀疏窗不同。

**重点案例。** 5 次新报推错在实报和当前形状下都正确接回，屏蔽全局后 1 个稀疏窗改成两条假边。另 5 次原生屏蔽几何已错在实报和当前形状下也都正确接回，全局屏蔽打断其中 2 个非稀疏正确接回。唯一新增真命中在原实报和当前形状下漏掉，全图屏蔽完整恢复两真边、移除两假边，没有外围变化。另 2 次正确完整换配同样来自当前未否决的窗。此处“真命中”是 C52/53 重点病例，另 2 窗为本轮全局对照的新增命中。

| 同原接回批动作 | 双报 命中／误拦（8／22） | 稀疏主（28／142） | 广义（67／308） |
|---|---:|---:|---:|
| 当前形状／全弃权 | 4／2 | 5／1 | 8／8 |
| 原实报重解，覆盖外沿当前 | 0／0 | 3／1 | 4／6 |
| 原接回批几何同刻屏蔽，限双报覆盖 | 7／17 | 5／7 | 11／23 |
| C55 当前形状的几何与插值同时遮蔽 | 6／5 | 5／1 | 10／11 |
| 全图屏蔽、仅遮固定触发边、完整备选、主读出 | 各 3／10 | 各 3／4 | 各 7／16 |

主读出相对当前双报新增 3 命中、失去 4 命中，新增 9 误拦、消除 1 误拦。共同支持稀疏主 2 个目标、9 个正确对照，当前 2／0，主读出 0／3。共同支持覆盖全部双报真值明确窗，其中 1 混合窗只计边及结构变化。31 窗主读出主动 30 次，1 次回退。0 个共享 A 组件净改善，4 个净退化。上述边变化按窗计，组间共享 A，属于原场配对诊断。既有原实报全人口对照在所有原合法窗均沿原接回，表内同覆盖臂只替换双报窗动作。未过预注册晋级，未冻结新 MTAD 留场，无新的持续 ID、稀疏长错链、原正确配对或首报发布结果。 [Mode C v75](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v75-同批全局指派与身份回流)
