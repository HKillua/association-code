# DeepAF实现边界：尚无真实MTAD同行成绩

**099 HOLD。** 自建模拟训练已退役，只保留历史排错事实；不再用合成误差认证DeepAF、判断真实任务竞争力或要求先赢模拟控制。原生DeepAF、作者协议复现和本项目任务适配是三个不同层次。

## 保留的生产实现

原文：[Cui等，DeepAF，Aerospace 2025，12(3)，194](https://doi.org/10.3390/aerospace12030194)，Eq.(9)–(14)、§3.1–3.3。当前代码为[模型](../../association/models/deepaf_peer.py)、[训练](../../association/training/deepaf_peer.py)、[评价](../../association/evaluation/deepaf_peer.py)。

- 独立P/V两网；每网6 encoder＋6 decoder、d32、4头、FF64；共265732参数。
- 候选集合无位置编码，历史有编码；cross-attention保留`2qᵀk−qᵀq−kᵀk`，self-attention为scaled dot product。
- 上一步自身V作为虚拟候选，P/V历史来自预测，不读取真值历史。
- 已确认并修复初始化坐标混用：native OLS经纬角速度转换到实际初始化位置的球面切向坐标。构造高纬反例证明旧近似有偏差，不说明Level1普遍影响量级。修复保留，不撤回有效修复。

## 与原方法不同的明确适配

| 项目 | 本项目实现及限制 |
|---|---|
| 任务 | fixed056离线已分组位置恢复，未复现在线门控、起始、终止 |
| 时间/坐标 | 全场Q64真实间隔、球面局部米制，不是原1秒二维模拟协议 |
| 初始化 | 一条有效OLS历史，q0虚拟前驱；不是作者已明确的原始起始流程 |
| 无观测 | anchor插值/外推始终进入候选，不能声称验证了原在线完全空门 |
| 监督 | P为相对上步预测的纠偏增量/dt；V为相邻位置差分/dt，而非已知瞬时真速度 |
| 优化 | scale10、dropout0、TBPTT1；位置项隐含1/dt²权重，后续位置误差不反传到此前V |
| 信息 | 离线预处理可用query后的量测，网络又按时间顺序递推；不能叫纯online |
| 输出 | 正式比较只限位置；对外速度/航向仍为Mean，内部V不是完整四属性效果 |

原文P标签允许true或desired，不能直接认定纠偏监督错误；以上选择须在真实任务中审查，不能因已披露就自动判为合理。模型具备这些核心结构，不等于可靠复现作者性能。

## 删除与保留

自建256/1024合成训练、影子诊断及重复JSON/报告不进入当前证据表，也不作为恢复099的门槛。它们只能支持有限的实现排错与可学习性观察。原件可在Git `e7e4a11`以及本地`runs/DEEPAF-NATIVE-QUALIFICATION-20261025-local-attempt-1`查阅，用户ZIP和模型权重未删除。

保留必要的单位/坐标、真实参数更新、输入不读评价真值、完整人口及缓存比较检查。下一步直接围绕真实Level1适配设计有限执行，不继续优化自建模拟分数。原099范围仅作[历史合同](https://github.com/HKillua/association-code/blob/903f679/docs/specs/peer-feasibility-sprint-099.md)留存；它已HOLD，后续具体任务只认[交接](../../HandOver/Handover.md)。
