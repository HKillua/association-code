# 当前研究状态

更新：2026-10-30。本页只维护当前目标、结论与下一步；执行只看[HandOver](../HandOver/Handover.md)。整理前的问题推导、矩阵/Idea连接、取舍与执行沿革已[全文保留](history/2026-10-30-research-process.md)，不以摘要替代研究材料。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。以本领域真实存在、尚未很好解决的问题为起点，允许更换整体骨架；成熟方法的合理跨域适配、有意义的新增价值均可研究。按[AGENTS](../AGENTS.md)、[完整飞书采纳流程](research_workflow.md)及[项目skills](../.agents/skills/README.md)推进，不先定方法再找依据。

最终方案不依赖源编号优先、仅两源平移或固定双源混合配方；可变K、来源换序/增减是研究要求，真实数据目前仍主要K2。精度优先，轻量/在线/端到端是加分目标。用户接受合理T侧监督拟合及新数据集重拟合，但倾向少依赖后端reference系数拟合，不等于禁止学习。

## 现在做什么

**按用户最新要求，先把自己的方法把握好。统一评价入口与Qhard全V300基准已完成，独立输出复算通过；PTFM/DeepAF真实T8训练已接通，但尚非全量公平成绩。** [147结果](reports/2026-10-30-source-risk-full-benchmark-review.md)为冻结原参数扩量，不拼146；[148/原107痕迹](reports/2026-10-30-fusion-peer-connection-review.md)保留真实GPU更新和未解决的恢复误差。[比较设计§9](reports/2026-10-27-fair-comparison-contract.md#9-2026-10-30三种同行共同任务比较的后继设计)仍区分固定WS组与实际完整系统，PTFM/DeepAF注明共同前端。

本轮已从Mode C1.63形成并实际运行[149零支持候选/150时序信息](proposals/temporal_source_and_unsupported_candidates_20261030.md)，科学和限定v2输出复算通过，原审计split入口失败另保留，正在Record/Squeeze，不提前写成成功机制。[151/152全量比较设计](proposals/fusion_peer_full_training_20261030.md)的数据准备已在服务器运行，152具体整数曝光待准备终态后发布；[153原107固定端点T物理探针](proposals/tmtaff_training_physical_probe_20261030.md)独立发布，不重训旧107。

1. 固定147六臂全量基准，优先形成高害误选与支持外恢复的独立新Idea；保留整体/覆盖/重复/尾部和原32之外的结果，不在V追温度/阈值。当前零支持仍继承首anchor OLS与圆周歧义约定，尚不满足最终无首代表依赖的整体要求；不退回源编号优先，也不直接重用已失败的SOURCE_ALL配方。
2. 在148真实T接线基础上接全T2100/V300公平比较，先明确样本/共同支持原域与公共回退域、点数匹配、整数曝光和选模预算，另立具体合同；不能把8步T探针当完整训练。首seed20，正式重复预定20/21/23，最多四单卡/FIFO。TMTAFF先区分T物理恢复、特征/输出/递归失配，旧107不续训。147/148已终态收尾，本轮新任务只看HandOver。
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
| 通用贡献冻结基准，V300 | 共同441144点，Mean/Nearest/Qsoft/Qhard MAE1294.074/964.568/1154.759/590.760m；余268 Qhard599.026m。300场/38组件相对Mean和Nearest场MAE均改善，但对Mean3039点增害>1km、P99略差；零支持3.682%点承担27.666%SSE，回退仍待替换 | [147 Record/Squeeze](reports/2026-10-30-source-risk-full-benchmark-review.md) |
| 同行接线/失效定位 | PTFM/DeepAF各真实GPU8×4更新，固定T探针u8仍差Mean，未完成T2100/V300排名。原107保存normalizer/affine/单位吻合、1000步仅覆盖802T场/29002unique pair；输出尺度大不等失效根因 | [148与原107痕迹](reports/2026-10-30-fusion-peer-connection-review.md) |
| WS分组，V300 | 固定056分数，decoded宏F1 76.5657→85.1820%；跨源mPCA77.9161→82.7318%、mPFA20.2849→14.1685%。raw不变；23场/4组件F1变差，完整覆盖/重复仍有代价。是内部增量 | [114](reports/2026-10-29-static-partition-review.md)、[115](reports/2026-10-29-static-position-review.md)、[117](reports/2026-10-29-query-contribution-full-review.md) |
| 通用来源贡献，T32/V32 | V同71071点源等权MAE1276.441m、Qsoft1128.755m、同评分Qhard547.721m。T参考监督共享二次头有排序信号，不是校准物理健康；当前soft未优于hard | [133](reports/2026-10-30-source-risk-review.md)、[134](reports/2026-10-30-mixture-opportunity-review.md)、[138](reports/2026-10-30-observation-conflict-review.md) |
| 139/140/143/144，T32/V32 | 米制、相对代价、平方风险及集合gain当前配方均未稳定超Qhard。143为579.466m；144仅−.038579m、区间跨0、动作全0/1，不能宣连续融合成功；当前配方不扩大，不否定整类方法 | [139](reports/2026-10-30-source-selection-evidence-review.md)、[140](reports/2026-10-30-source-contribution-readout-review.md)、[143](reports/2026-10-30-relative-source-decision-review.md)、[144](reports/2026-10-30-actual-mixture-gain-review.md) |
| 状态恢复，T32/V32 | 146只换零支持传播速度，OBS-RTS V MAE1784.722→1241.072m、RMSE27531.051→2151.799m；但T+11.007m且V新增232点损害>1km。接Qhard仅547.721→546.710m，区间跨0、P95/P99差，未获稳定系统增量 | [137](reports/2026-10-30-observation-scale-review.md)、[142](reports/2026-10-30-state-recovery-tail-review.md)、[146](reports/2026-10-30-unsupported-recovery-review.md) |
| 历史条件有效点 | 125/131条件贡献、127原生RTS有真实收益及完整反例，但依赖已被排除的固定源角色条件，不能作为通用多源最终方案或相加冠军；观测免reference拟合136仅小幅均值收益、强控/尾部未过 | [125/131](reports/2026-10-29-conditional-source-value-full-review.md)、[127](reports/2026-10-29-native-state-review.md)、[136](reports/2026-10-30-observation-quality-review.md) |
| 可信同行 | 尚无共同协议下稳定胜过TMTAFF/PTFM/DeepAF的证据，TMTAFF有效复现未完成。107有限训练/113评价位置严重失效、速度有收益；不以失败适配判赢原方法 | [113](reports/2026-10-29-fixed-endpoint-review.md)、[DeepAF边界](reports/2026-10-25-deepaf-method-audit.md)、[比较合同](reports/2026-10-27-fair-comparison-contract.md) |

最近147/148A/148B与147独立输出审计均到可核终态，完成Record→Squeeze→Mode C1.62；原107痕迹只读再分析另记，不重训。详细人口、成本、偏差和不利结果见原报告；143/144/146与145原失败保留。

## 研究材料与推导入口

以下资料库链接为本机原件路径，GitHub不包含对应全文；本轮保持原位、内容不变。矩阵、Idea与工作流分析保留完整正文，摘要只作导航。

| 需要什么 | 入口 |
|---|---|
| 全局P/F/M/E/C与回流 | [Mode C1.62](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-162)，累计P14/F247/M109/E149/C55 |
| 多源贡献范围与条件 | [多源可靠性矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/multisource_reliability_matrix_20261030.md) |
| 选源与集合读出 | [SR v1.4，39格](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#full-benchmark-feedback) |
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
