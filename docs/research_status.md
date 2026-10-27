# 当前研究状态

更新：2026-10-27。本页是**当前研究目标、结论与下一步的唯一入口**；执行状态只看[HandOver](../HandOver/Handover.md)。正式数据MTAD Level1：T2100/V300/封存测试600，固定056组及共享身份限制见[数据约定](data_contract.md)。

## 当前要解决什么

近期先形成一项可信的**位置融合／恢复贡献**，与已发表同行在同任务下比较，再接回补全—关联—融合完整系统。接受成熟机制和跨领域海上适配；090C是内部强对照，不绑定未来架构。当前**没有稳定胜过可信同行的证据，也没有有效完成TMTAFF复现**。[用户目标订正原记录](reports/2026-10-26-goal-reconciliation.md)。

## 现在做什么

1. **Mode C已更新到1.7，完成预测对应条件核查→Record→Squeeze→矩阵回流。** [矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-17)仍为19篇独立论文、17篇正文分析＋两篇定向，正式2025–2026共10篇；现为6问题×30现象/数学性质、12机制、23证据。新增E23来自同一103数据的有界分析，不是新论文或独立算法重复。T-FoT-SP正式12页稿、部分作者实现等缺口仍按来源登记，不把预印本风险推给未读正式稿。
2. **102/103已执行回传，本地核验通过；三场主因未唯一识别。** [完整实验记录/深挖/回执订正](reports/2026-10-27-returned-diagnostics-review.md)保留全部人口、对照和不利结果。103的5504声明query中4548可评；共同P95尾195点/15目标占090C SSE98.86%，现有标签下组成员一致，且部分近query原生量测相对reference已出现极大误差。090C三场MAE优于Mean而RMSE劣于Mean；不能只归因错helper、网络外推或直接判reference错误。C03旧GNC方案所需的可识别健康约束未建立，不进入该pilot。原任务结束，不重跑102/103。
3. **相对配准C10获得有限的预测输入条件支持，尚无绝对精度收益证据。** [完整Record/Squeeze](reports/2026-10-27-predicted-registration-audit.md)：不看真实身份估计中心，三个场景留出组等权相对残差3179.501→629.063、3937.242→244.724、2903.963→621.317m；3035两个组恶化，1043仅两组留出。早晚query检查是离线插值条件，非在线预测。方向多样性/绝对锚点/全V效果仍未知；与C08/C09[重新比较](reports/2026-10-27-candidate-feasibility.md#candidate-comparison)，不因已有结构或消融容易先定方法。
4. **104已准备为唯一新READY：三个固定极值及各±2 query原链核查。** 3035 target23/query56、2819 target1/query60、1043 target3/query54，仅读103已留私有/公开证据，0训练/前向/ZIP/NPZ；[事前plan1.4](proposals/continuous_state_reassessment_20261027.md#chain-audit-104)。软件检查通过，尚无真实104回执。核reference原端点/重复、时间映射、canonical步长和标签解析；量测CSV→canonical原行缺口保留，不能仅凭一致性判现实谁错。按回执决定修实现或明确unknown后结束，不扩扫、不删难点或用reference修输入。C08/C09保持独立研究资格，GNC无优先级，090C不绑定新架构。
5. **公平比较实施合同已明确，全链仍未贯通。** [合同](reports/2026-10-27-fair-comparison-contract.md)固定056组/Q64/完整人口，同时保留canonical_all强控制与cap128可见性匹配控制，明确新单位、TCM/TFM实际曝光与末步。缺口是固定056评价wrapper、共同控制和共同mask汇总；现TFM终评为oracle，不能冒充预测组成绩。显式单位链已有[软件验收](reports/2026-10-27-tmtaff-speed-unit-verification.md)，但新单位未训练、充分训练量未定，不发布训练READY。16论文维度与MAE/RMSE/尾部/覆盖/38组件/成本并报，作者ESD仍未知；不用失败适配或论文报数宣布胜负。

## 保留的结论

| 对象 | 已核事实 | 决定／证据 |
|---|---|---|
| 056与090C/097A | 056固定组位置覆盖97.2109%、9353组/610重复；两seed位置MAE564.834/583.976m | 内部比较起点，不是可信领域同行；[方法](specs/evidence-association-fusion.md) |
| 098关系偏置 | 双seed均未过原晋级规则，最终关bias几乎不改变MAE | 当前配方结案；不代表训练期间bias完全无影响。[历史](history/README.md) |
| 100原生事件残差 | T0继续训练564.968m，T1固定汇总566.092m，T2 attention汇总567.187m，均未胜090C | 不晋级、不自动加步/seed；两臂都有真实query decoder。[完整分析](reports/2026-10-26-experiment-review.md) |
| 101缓存分析 | 090C对Anchor的绝对误差收益96.23%来自helper机会区；少数场景主导平方误差，普通控制亦有大错 | 现象已核，根因未定；不能直接归因错helper、数据坏或需要平滑。[完整分析](reports/2026-10-26-experiment-review.md) |
| TMTAFF/PTFM/TSA适配 | 当前实验未建立可信复现；096物理恢复和关联均弱 | 不用失败适配认证“赢同行”；[历史](history/README.md) |
| DeepAF | P/V核心存在，真实MTAD适配未完成；与TMTAFF所用四属性改版有差别 | 099暂停；不扩建模拟资格链。[实现边界](reports/2026-10-25-deepaf-method-audit.md) |

## 判断边界

- 本轮遵循[完整科研流程](research_workflow.md)：103及本地C10诊断各自完成Idea→实际执行→Record/Squeeze→Mode C；102和额外tail分别按指标合同/事后探索记录。104只有事前设计/实现/软件核验，未有真实回执；C09方法Idea未执行，无新训练合同。软件验收不是科研成绩。
- 无效配方缺少新证据就换，不因沉没成本续训；单次负结果也不否定整类模型。旧099的10/28初判、11/1去留节点不因换候选重置；3%等是具体旧合同门槛，不是普适创新定义。
- 论文指标按[统一定义](mtad_data_protocols.md)逐项补齐，另保留米制误差、尾部、覆盖/重复、场景/38组件。ME不是MAE；Level1量测kn/reference mps的已核范围见数据约定，历史原生速度不能冒称单位已统一。
- V反复用于开发，且有共享身份；不声称独立测试泛化。T_test保持封存，不删难例、不拼不同人口或单模块冠军。

详细历史不在本页追加；已执行事实进[结果分析](reports/2026-10-26-experiment-review.md)，旧阶段进[历史摘要](history/README.md)。
