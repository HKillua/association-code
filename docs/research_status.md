# 当前研究状态

更新：2026-10-31。本页只维护当前目标、结论与下一步；执行只看[HandOver](../HandOver/Handover.md)。整理前的问题推导、矩阵/Idea连接、取舍与执行沿革已[全文保留](history/2026-10-30-research-process.md)，不以摘要替代研究材料。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。以本领域真实存在、尚未很好解决的问题为起点，允许更换整体骨架；成熟方法的合理跨域适配、有意义的新增价值均可研究。按[AGENTS](../AGENTS.md)、[完整飞书采纳流程](research_workflow.md)及[项目skills](../.agents/skills/README.md)推进，不先定方法再找依据。

最终方案不依赖源编号优先、仅两源平移或固定双源混合配方；可变K、来源换序/增减是研究要求，真实数据目前仍主要K2。精度优先，轻量/在线/端到端是加分目标。用户接受合理T侧监督拟合及新数据集重拟合，但倾向少依赖后端reference系数拟合，不等于禁止学习。

## 现在做什么

**完整V300基准仍为固定T128头582.808m；新159只评价原T32/V32，不能替换全300成绩。** [159表征×表达](reports/2026-10-31-local-expression-review.md)已完成6656次真实GPU更新：V32旧特征残差网540.295→527.427m、误选761→287，但罕见反转229→278；保留平均收益候选，不称高害解决。局部网527.399m不支持稳定额外增量。[157完整结果](reports/2026-10-30-t128-full-scope-review.md)与旧[147基准](reports/2026-10-30-source-risk-full-benchmark-review.md)保持；[比较设计§9](reports/2026-10-27-fair-comparison-contract.md#9-2026-10-30三种同行共同任务比较的后继设计)仍区分固定WS后端任务与完整系统。

本轮新验证按[165–167独立Idea/spec](proposals/object_conditions_context_20261031.md)推进：不读truth的固定分数split/move、冻结159上的集合质量上下文与同容量OWN对照、原TFM损失的冻结梯度定位。软件及真实缓存/覆盖变化链已核验，执行事实只见HandOver；结果尚未产生，不预宣机制/精度。原平均候选与全300基准保留。

本轮162–164已完成服务器真实验证、限定核验及Record→Squeeze→Mode C1.71。[162目标×学习质量](reports/2026-10-31-objective-difficulty-review.md)：四新臂52fit/6656GPU更新，V32为527.447/527.987/535.208/576.356m，均未稳定超过冻结159的527.427m，当前四配方不扩大。旧网256步与新128步曝光分列，不把新弱控胜负当升级。原159平均候选保留，不按V小数挑冠军。

[163分组/参考分离](reports/2026-10-31-group-reference-review.md)给出新的任务条件证据：V32反转290个组×query，真值净化后仅16仍多源、105仅一源、169零源，剩16仍选错；**274个失去多源资格不能记作救回**。固定原状态换候选参考，最佳源集合4728个query发生变化，但不证明官方参考有错。两轴均为oracle诊断，不是新部署成绩。后补独立核中央人口/原错/全部替代几何通过；净化后的特征/forward及物理truth不在独审范围。T纯组44/V纯组16难例仍保留，不能把全部错误归因构组。

[164冻结K/V](reports/2026-10-31-tfm-key-value-review.md)分开尺度操作：K-only放松首步attention集中，T2同204点MAE却58.292→58.373km，V-only/both也更差；当前简单尺度配方不接方法，训练梯度/监督/归一化成因仍未知，107不重训。[160端点×速度](reports/2026-10-31-endpoint-separation-review.md)单独端点局部信号、速度/联合灾难及全部旧反例完整保留，不拼冠军。

[151完整数据准备](reports/2026-10-30-full-peer-data-review.md)已完成T2100/V300、46126可训练组。152A/B仍真实systemd RUNNING并实际优化，fresh seed20、固定150/20epoch对应427800/80800更新，可能跨多天；尚无末轮全V公平分数，实际训练进度只看HandOver。冻结活动合同仍比较原147，不改成157/159；旧148不续跑。

1. 保留157全量基准及159平均收益，下一轮优先由**C53×SR-C1**区分“同对象来源风险排序”和“对象/参考条件”：用合法可观察证据形成有限一致对象候选/条件状态，再在同候选与状态调用预算下测试决策增量。不用truth净化作为运行输入，不更换官方参考/筛掉丢支持点；全人口的关联raw/decoded、覆盖/重复、物理平均与尾部都必须核。成熟先例已有，创新看MTAD适配的实际新增价值，不因先例结束研究。
2. 纯组T44/V16仍错，与混组路线分开研究额外上下文/局部状态/学习条件；不把参考敏感性叫数据错误或“信息绝对不足”。当前目标/质量四配方不无依据加权/加步枚举；真实K≥3/外域仍缺。零支持仍继承旧首anchor回退与圆周歧义，输入/原生时序/状态路线未关闭，但不重复已失败同窗速度配方或退固定源优先。
3. 跟进152固定全T训练/末轮V300同WS任务比较：完整曝光/物理学习、强控/覆盖/尾部/成本，PTFM核心缺域与公共Mean扩域分报、DeepAF P/V适配权限明确。不能把共享WS比较称原完整系统复现。TMTAFF若再研究训练须先核真实梯度/监督传播与尺度路径，不能直接加107步数。首seed后继另定具体合同，最多四单卡/FIFO，无SSH值守或自动新训练。

| 问题 | 新证据之后仍要解决什么 |
|---|---|
| 高损害误选 | 保住159平均收益；分别处理对象/对应条件及纯组真难例，当前162目标/质量配方未完成这一步。163只是明确oracle与可比人口，不是方法升级。 |
| 混合收益未兑现 | 134训练侧不均匀条件空间仍在；先确认同对象条件/实际状态相容性，再区分细小收益分辨率与可学性，不能用对象变化假冒混合增量。 |
| 零支持尾部 | 160端点局部信号仍保留，速度/联合约20km RMSE不扩大；需新的传播条件/风险信息，不能V设horizon/cap、删难例或拼局部冠军。 |

输入、时序、关联、融合与输出仍在研究范围，原矩阵/Idea/未采纳理由保留。新后端增量仍主要位置，速度/航向完整评价；真实K2/离线/共享开发域及V重复使用不称独立泛化。162–164本轮各一次科学已收尾，无自动新READY；后继从Mode C1.71进入具体Idea后再执行。

## 保留的结论

| 对象／实际范围 | 当前能支持的结论 | 完整证据 |
|---|---|---|
| 目标×训练质量，T128/原T32-V32 | fresh四臂52fit/6656GPU更新，V527.447/527.987/535.208/576.356m均未稳定超冻结159527.427m；平衡局部救回不能抵消整体新增错误/尾部，当前配方不扩大 | [162](reports/2026-10-31-objective-difficulty-review.md) |
| 固定输入/固定参考双轴oracle，T32-V32 | V反转290净化后16多/105一/169零，剩16仍错；274丢D2不是救回。参考候选可改变最佳集合4728query；中央人口/参考几何独审通过，不认证truth/可部署提升 | [163](reports/2026-10-31-group-reference-review.md) |
| 冻结TFM K/V分离，T2四正对 | 单改K放松attention却MAE58.292→58.373km，四臂无精度恢复；正常/both与161 exact，训练原因未唯一认证，不重训107 | [164](reports/2026-10-31-tfm-key-value-review.md) |
| 局部表征×残差网，T128训练/原T32-V32评价 | V540.295→527.427m、误选761→287，但反转229→278，P99+9.644m；两种网各3328更新、共6656，局部网没有稳定额外收益。保留平均候选，未解决稀有高害 | [159](reports/2026-10-31-local-expression-review.md) |
| 端点×速度分离，T32/V32 | 端点ONLY V548.006m/RMSE1712.592m，联合908.219m/RMSE19967.257m；新增1163点增害>1km/最大增害1695.760km，当前速度/联合不扩大。局部尾部信号不等稳定系统升级 | [160](reports/2026-10-31-endpoint-separation-review.md) |
| 冻结TFM层级，T2四正对 | 正常输出与154 exact；同query跨输入差最后两层收缩，首步末层attention极集中；memory×.01改变分布却MAE更差。K/V/梯度/优化成因未分离 | [161](reports/2026-10-31-tfm-decoder-layers-review.md) |
| 固定T128贡献头，V300 | MAE590.760→582.808m、RMSE3303.235→3299.900m，P99不变；总误选7438→5212、>1km选择excess4466→2890。罕见反转1868→1940、实际仍231点对旧增害>1km，不能宣高害解决。余268 MAE599.026→590.973m；零/一支持保持 | [155](reports/2026-10-30-source-coverage-review.md)、[157](reports/2026-10-30-t128-full-scope-review.md) |
| 同源历史选择/当前多尺度，T32/V32 | 156历史CV V MAE641.236m/RMSE10068.222m不扩大；158r1 V547.819m基本持平旧547.721m，RMSE2654.187m差于1829.356m。较匹配r2/CV有局部信号，仍15点>100km/数值回退/after反例，当前不接主方法 | [156](reports/2026-10-30-within-source-forecast-review.md)、[158](reports/2026-10-30-robust-endpoint-review.md) |
| 冻结TFM内部路径，T2四正对 | TCM/FC有变化、decoder末端约1e−6变化，投影位置权重非零；零memory改变输出但更差，最后投影double仅约.77m差。部分操作定位，不是唯一优化原因认证 | [154](reports/2026-10-30-tfm-constant-path-review.md) |
| 新时序/零支持pilot，T32/V32 | 149三候选无稳定MAE升级，最近V RMSE/P99较好却391点对Qhard增害>1km；150 V小益/T变差，罕见反转未解。原审计失败与新限定复核完整保留 | [149/150](reports/2026-10-30-temporal-unsupported-review.md) |
| 冻结TMTAFF，T8物理定位 | 四种输入干预1024查询位置均同一常值；732同点TFM MAE36.635km、Mean1.887km，支持内也差。不是唯一根因认证，也不是可信同行胜负 | [153](reports/2026-10-30-tmtaff-training-physical-review.md) |
| 通用贡献冻结基准，V300 | 共同441144点，Mean/Nearest/Qsoft/Qhard MAE1294.074/964.568/1154.759/590.760m；余268 Qhard599.026m。300场/38组件相对Mean和Nearest场MAE均改善，但对Mean3039点增害>1km、P99略差；零支持3.682%点承担27.666%SSE，回退仍待替换 | [147 Record/Squeeze](reports/2026-10-30-source-risk-full-benchmark-review.md) |
| 同行接线/失效定位 | PTFM/DeepAF各真实GPU8×4更新，固定T探针u8仍差Mean，未完成T2100/V300排名。原107保存normalizer/affine/单位吻合、1000步仅覆盖802T场/29002unique pair；输出尺度大不等失效根因 | [148与原107痕迹](reports/2026-10-30-fusion-peer-connection-review.md) |
| WS分组，V300 | 固定056分数，decoded宏F1 76.5657→85.1820%；跨源mPCA77.9161→82.7318%、mPFA20.2849→14.1685%。raw不变；23场/4组件F1变差，完整覆盖/重复仍有代价。是内部增量 | [114](reports/2026-10-29-static-partition-review.md)、[115](reports/2026-10-29-static-position-review.md)、[117](reports/2026-10-29-query-contribution-full-review.md) |
| 通用来源贡献，T32/V32 | V同71071点源等权MAE1276.441m、Qsoft1128.755m、同评分Qhard547.721m。T参考监督共享二次头有排序信号，不是校准物理健康；当前soft未优于hard | [133](reports/2026-10-30-source-risk-review.md)、[134](reports/2026-10-30-mixture-opportunity-review.md)、[138](reports/2026-10-30-observation-conflict-review.md) |
| 139/140/143/144，T32/V32 | 米制、相对代价、平方风险及集合gain当前配方均未稳定超Qhard。143为579.466m；144仅−.038579m、区间跨0、动作全0/1，不能宣连续融合成功；当前配方不扩大，不否定整类方法 | [139](reports/2026-10-30-source-selection-evidence-review.md)、[140](reports/2026-10-30-source-contribution-readout-review.md)、[143](reports/2026-10-30-relative-source-decision-review.md)、[144](reports/2026-10-30-actual-mixture-gain-review.md) |
| 状态恢复，T32/V32 | 146只换零支持传播速度，OBS-RTS V MAE1784.722→1241.072m、RMSE27531.051→2151.799m；但T+11.007m且V新增232点损害>1km。接Qhard仅547.721→546.710m，区间跨0、P95/P99差，未获稳定系统增量 | [137](reports/2026-10-30-observation-scale-review.md)、[142](reports/2026-10-30-state-recovery-tail-review.md)、[146](reports/2026-10-30-unsupported-recovery-review.md) |
| 历史条件有效点 | 125/131条件贡献、127原生RTS有真实收益及完整反例，但依赖已被排除的固定源角色条件，不能作为通用多源最终方案或相加冠军；观测免reference拟合136仅小幅均值收益、强控/尾部未过 | [125/131](reports/2026-10-29-conditional-source-value-full-review.md)、[127](reports/2026-10-29-native-state-review.md)、[136](reports/2026-10-30-observation-quality-review.md) |
| 可信同行 | 尚无共同协议下稳定胜过TMTAFF/PTFM/DeepAF的证据，TMTAFF有效复现未完成。107有限训练/113评价位置严重失效、速度有收益；不以失败适配判赢原方法 | [113](reports/2026-10-29-fixed-endpoint-review.md)、[DeepAF边界](reports/2026-10-25-deepaf-method-audit.md)、[比较合同](reports/2026-10-27-fair-comparison-contract.md) |

149–151/153–164已完成记录与对应回流；151数据准备不新增方法成绩。159/160按事前安排从完整冻结数组补齐四属性/方向，未倒填独审；155–158此前漏项的订正保留，速度航向无新学习增量。161后验同query跨输入分解及交接索引迟记单列偏差。162 READY早于本地fixture通过的时序偏差保留，实际提交前修正并通过；163中央人口/参考独审为科学后追加，未倒填事前设计。详细人口、成本、不利结果和失败原件保留。152仍训练中，无新公平排名。

## 研究材料与推导入口

以下资料库链接为本机原件路径，GitHub不包含对应全文；原分析完整保留，新结果按版本追加。矩阵、Idea与工作流分析保留完整正文，摘要只作导航。

| 需要什么 | 入口 |
|---|---|
| 全局P/F/M/E/C与回流 | [Mode C1.72执行前](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-172)，累计P14/F261/M109/E164/C55 |
| 多源贡献范围与条件 | [多源可靠性矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/multisource_reliability_matrix_20261030.md) |
| 选源与集合读出 | [SR v1.9，57格](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#objective-reference-feedback) |
| 观测驱动与状态恢复 | [OD v1.10，54格](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md#endpoint-separation-feedback) |
| 逐篇阅读范围、来源与代码覆盖 | [深入阅读审计](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/deepread_coverage_100_20261028.md)，充分深入104＝72领域/相近＋32跨域；不是作者复现 |
| Idea完整设计、假设与否证 | [提案目录](proposals/)；各Record链接对应Idea，未执行/未采纳不删除，135仍仅保留草案 |
| 历史推导、连接与取舍原文 | [完整研究过程](history/2026-10-30-research-process.md)、[结案摘要](history/README.md)；历史“下一步”不自动成为任务 |

## 判断边界

- 正式Level1 T2100/V300/封存测试600；名单不按模型成绩挑选。T/V共享6053个原A键，V最大两组件占63%，不能称身份独立或独立确认性泛化。[数据约定](data_contract.md)与[指标定义](mtad_data_protocols.md)是唯一规则入口。
- ME不是MAE；作者ESD实现未知。raw/decoded、固定组/完整系统、各自/共同有效人口分别报告，保留覆盖、重复、组件和尾部，不删难点。当前后端位置实验的速度/航向仍沿用Mean，不宣四属性学习增量。
- 支持域、reference生成/物理准确性及长尾来源未完全查明。三场原始短时大位移仅支持所查案例，不据此认定全数据坏。[三场原链](reports/2026-10-27-three-case-chain-review.md)、[原CSV核验](reports/2026-10-27-raw-case-chain-review.md)。
- 132在SUMMARY完成1000次GPU更新后被oomd终止，完整评价及另三臂未完成；用户要求保留现场，管理员日志暂缺，不自动重启/重训。[失败记录](reports/2026-10-29-ordered-input-continuation-failure.md)。原107/113及143/144/146已结案，不续跑旧额度。
- 换loss、局部修复或软件通过不证明机制，不因单次失败否定整类方法，不把真实K2接口能力说成K≥3/异构泛化或在线端到端。
