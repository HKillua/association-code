# 当前研究状态

更新：2026-10-30。本页只维护当前目标、结论与下一步；执行只看[HandOver](../HandOver/Handover.md)。整理前的问题推导、矩阵/Idea连接、取舍与执行沿革已[全文保留](history/2026-10-30-research-process.md)，不以摘要替代研究材料。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。以本领域真实存在、尚未很好解决的问题为起点，允许更换整体骨架；成熟方法的合理跨域适配、有意义的新增价值均可研究。按[AGENTS](../AGENTS.md)、[完整飞书采纳流程](research_workflow.md)及[项目skills](../.agents/skills/README.md)推进，不先定方法再找依据。

最终方案不依赖源编号优先、仅两源平移或固定双源混合配方；可变K、来源换序/增减是研究要求，真实数据目前仍主要K2。精度优先，轻量/在线/端到端是加分目标。用户接受合理T侧监督拟合及新数据集重拟合，但倾向少依赖后端reference系数拟合，不等于禁止学习。

## 现在做什么

**优先落实TMTAFF、PTFM、DeepAF的共同任务公平实测，同时推进三个未决问题。** [比较设计§9](reports/2026-10-27-fair-comparison-contract.md#9-2026-10-30三种同行共同任务比较的后继设计)区分固定WS组的融合比较与实际完整系统；PTFM/DeepAF注明共同前端，不伪造其全片段关联能力。

1. 核/修三者与原文的明确适配差异及共同评价入口，用既有证据和真实T侧有界验证固定新合同。统一T2100/V300、Q64、单位、信息权限和完整人口；保留强普通控制和输入点数匹配控制，我们的通用候选也扩到同一V300，不拿V32与同行V300排名。
2. 首seed20作可用性比较，正式重复预定20/21/23；每模型整数训练量、选模/调参额度和成本须先冻结。最多四单卡，三个同行就绪可各一路，第四路给成熟科学Idea。现仅有设计，尚无新READY或训练，旧任务不自动重开。
3. 三问题从Mode C1.61分别生成新Idea，不能重复同信息换loss或拼接模块冠军：

| 问题 | 已有证据与下一项要区分的内容 |
|---|---|
| 高损害误选 | 138/139/143表明只改监督尺度/相对目标未改善强Qhard；研究何种合法时序/组证据能区分少数高害，同时核救回与新增错误 |
| 混合收益未兑现 | 134训练侧存在不均匀条件空间，140/144未稳定兑现；区分实际恢复相容性、细小收益分辨率和可学性，不把条件空间叫方法成绩 |
| 零观测支持时的尾部大错 | 142核清数值放大路径、146取得局部干预效果；研究如何保住T-before收益并改善强Qhard整体/尾部，物理异常来源仍未知 |

输入、原生时序、关联和对象输出仍在研究范围；已有候选/未采纳分支留在原矩阵与Idea中，不因近期优先融合而删除。

## 保留的结论

| 对象／实际范围 | 当前能支持的结论 | 完整证据 |
|---|---|---|
| WS分组，V300 | 固定056分数，decoded宏F1 76.5657→85.1820%；跨源mPCA77.9161→82.7318%、mPFA20.2849→14.1685%。raw不变；23场/4组件F1变差，完整覆盖/重复仍有代价。是内部增量 | [114](reports/2026-10-29-static-partition-review.md)、[115](reports/2026-10-29-static-position-review.md)、[117](reports/2026-10-29-query-contribution-full-review.md) |
| 通用来源贡献，T32/V32 | V同71071点源等权MAE1276.441m、Qsoft1128.755m、同评分Qhard547.721m。T参考监督共享二次头有排序信号，不是校准物理健康；当前soft未优于hard | [133](reports/2026-10-30-source-risk-review.md)、[134](reports/2026-10-30-mixture-opportunity-review.md)、[138](reports/2026-10-30-observation-conflict-review.md) |
| 139/140/143/144，T32/V32 | 米制、相对代价、平方风险及集合gain当前配方均未稳定超Qhard。143为579.466m；144仅−.038579m、区间跨0、动作全0/1，不能宣连续融合成功；当前配方不扩大，不否定整类方法 | [139](reports/2026-10-30-source-selection-evidence-review.md)、[140](reports/2026-10-30-source-contribution-readout-review.md)、[143](reports/2026-10-30-relative-source-decision-review.md)、[144](reports/2026-10-30-actual-mixture-gain-review.md) |
| 状态恢复，T32/V32 | 146只换零支持传播速度，OBS-RTS V MAE1784.722→1241.072m、RMSE27531.051→2151.799m；但T+11.007m且V新增232点损害>1km。接Qhard仅547.721→546.710m，区间跨0、P95/P99差，未获稳定系统增量 | [137](reports/2026-10-30-observation-scale-review.md)、[142](reports/2026-10-30-state-recovery-tail-review.md)、[146](reports/2026-10-30-unsupported-recovery-review.md) |
| 历史条件有效点 | 125/131条件贡献、127原生RTS有真实收益及完整反例，但依赖已被排除的固定源角色条件，不能作为通用多源最终方案或相加冠军；观测免reference拟合136仅小幅均值收益、强控/尾部未过 | [125/131](reports/2026-10-29-conditional-source-value-full-review.md)、[127](reports/2026-10-29-native-state-review.md)、[136](reports/2026-10-30-observation-quality-review.md) |
| 可信同行 | 尚无共同协议下稳定胜过TMTAFF/PTFM/DeepAF的证据，TMTAFF有效复现未完成。107有限训练/113评价位置严重失效、速度有收益；不以失败适配判赢原方法 | [113](reports/2026-10-29-fixed-endpoint-review.md)、[DeepAF边界](reports/2026-10-25-deepaf-method-audit.md)、[比较合同](reports/2026-10-27-fair-comparison-contract.md) |

最近143/144/146均到可核终态，完成Record→Squeeze→Mode C1.61；145原技术失败另保留。详细人口、成本、偏差和不利结果见原报告；整理不改变实验、论文或矩阵计数。

## 研究材料与推导入口

以下资料库链接为本机原件路径，GitHub不包含对应全文；本轮保持原位、内容不变。矩阵、Idea与工作流分析保留完整正文，摘要只作导航。

| 需要什么 | 入口 |
|---|---|
| 全局P/F/M/E/C与回流 | [Mode C1.61](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-161)，累计P14/F245/M109/E145/C55 |
| 多源贡献范围与条件 | [多源可靠性矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/multisource_reliability_matrix_20261030.md) |
| 选源与集合读出 | [SR v1.3，36格](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#relative-actual-feedback) |
| 观测驱动与状态恢复 | [OD v1.6，42格](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md#unsupported-recovery-feedback) |
| 逐篇阅读范围、来源与代码覆盖 | [深入阅读审计](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/deepread_coverage_100_20261028.md)，充分深入104＝72领域/相近＋32跨域；不是作者复现 |
| Idea完整设计、假设与否证 | [提案目录](proposals/)；各Record链接对应Idea，未执行/未采纳不删除，135仍仅保留草案 |
| 历史推导、连接与取舍原文 | [完整研究过程](history/2026-10-30-research-process.md)、[结案摘要](history/README.md)；历史“下一步”不自动成为任务 |

## 判断边界

- 正式Level1 T2100/V300/封存测试600；名单不按模型成绩挑选。T/V共享6053个原A键，V最大两组件占63%，不能称身份独立或独立确认性泛化。[数据约定](data_contract.md)与[指标定义](mtad_data_protocols.md)是唯一规则入口。
- ME不是MAE；作者ESD实现未知。raw/decoded、固定组/完整系统、各自/共同有效人口分别报告，保留覆盖、重复、组件和尾部，不删难点。当前后端位置实验的速度/航向仍沿用Mean，不宣四属性学习增量。
- 支持域、reference生成/物理准确性及长尾来源未完全查明。三场原始短时大位移仅支持所查案例，不据此认定全数据坏。[三场原链](reports/2026-10-27-three-case-chain-review.md)、[原CSV核验](reports/2026-10-27-raw-case-chain-review.md)。
- 132在SUMMARY完成1000次GPU更新后被oomd终止，完整评价及另三臂未完成；用户要求保留现场，管理员日志暂缺，不自动重启/重训。[失败记录](reports/2026-10-29-ordered-input-continuation-failure.md)。原107/113及143/144/146已结案，不续跑旧额度。
- 换loss、局部修复或软件通过不证明机制，不因单次失败否定整类方法，不把真实K2接口能力说成K≥3/异构泛化或在线端到端。
