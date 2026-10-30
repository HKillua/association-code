# 当前研究状态

更新：2026-10-31。本页只维护当前目标、结论与下一步；执行只看[HandOver](../HandOver/Handover.md)。整理前的问题推导、矩阵/Idea连接、取舍与执行沿革已[全文保留](history/2026-10-30-research-process.md)，不以摘要替代研究材料。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。以本领域真实存在、尚未很好解决的问题为起点，允许更换整体骨架；成熟方法的合理跨域适配、有意义的新增价值均可研究。按[AGENTS](../AGENTS.md)、[完整飞书采纳流程](research_workflow.md)及[项目skills](../.agents/skills/README.md)推进，不先定方法再找依据。

最终方案不依赖源编号优先、仅两源平移或固定双源混合配方；可变K、来源换序/增减是研究要求，真实数据目前仍主要K2。精度优先，轻量/在线/端到端是加分目标。用户接受合理T侧监督拟合及新数据集重拟合，但倾向少依赖后端reference系数拟合，不等于禁止学习。

## 现在做什么

**完整V300基准仍为固定T128头582.808m；新159只评价原T32/V32，不能替换全300成绩。** [159表征×表达](reports/2026-10-31-local-expression-review.md)已完成6656次真实GPU更新：V32旧特征残差网540.295→527.427m、误选761→287，但罕见反转229→278；保留平均收益候选，不称高害解决。局部网527.399m不支持稳定额外增量。[157完整结果](reports/2026-10-30-t128-full-scope-review.md)与旧[147基准](reports/2026-10-30-source-risk-full-benchmark-review.md)保持；[比较设计§9](reports/2026-10-27-fair-comparison-contract.md#9-2026-10-30三种同行共同任务比较的后继设计)仍区分固定WS后端任务与完整系统。

本轮159–161科学、规定独审/补项及Record→Squeeze→Mode C1.69收尾。[160端点×速度](reports/2026-10-31-endpoint-separation-review.md)：单独端点V MAE548.006m/RMSE1712.592m有局部尾部价值，短窗速度/联合却导致约20km RMSE，当前不扩大。[161冻结decoder](reports/2026-10-31-tfm-decoder-layers-review.md)进一步定位最后两层跨输入收缩与首步attention极集中，缩小memory并未恢复精度；K/V同时改变、训练原因仍未知，107不重训。原154–158完整结果与反例保留，不拼冠军。

[151完整数据准备](reports/2026-10-30-full-peer-data-review.md)已完成：T2100/V300、46126可训练组、0缺声明缓存。152A/B已核真实systemd RUNNING并实际优化，T-only normalizer/固定T初始探针均完成；fresh seed20，固定150/20epoch对应427800/80800更新，可能跨多天。尚无新全量公平分数，详细实际进度只看HandOver；旧148不续跑，107不重训。

1. 保留157全量基准及159真实平均收益；下一连接优先区分新非线性表达下的风险目标/稀有监督交互、分组/参考对应及缺失上下文。159不是高害完成方案，不按V挑两网小数冠军或自动扩量。不能无新依据重复同信息换loss；但新的表达已有实证，原143固定二次头负结果不永久排除新交互。零支持仍继承旧首anchor回退与圆周歧义，尚不满足最终完整匿名恢复；不退源编号优先。
2. 完成152固定全T训练/末轮全V300同WS任务比较，核实际曝光/物理学习与全部强控/覆盖/尾部/成本；PTFM核心缺域与公共Mean扩域分报、DeepAF P/V适配权限明确。活动合同比较原147，不修改其代码/对照；终态若补比较157或后继须另读冻结输出核同人口。不能把共享WS比较称原完整系统复现。首seed20，后继重复另定具体合同；最多四单卡/FIFO，无自动新训练。TMTAFF后继须分离K/V尺度或核训练梯度/监督，首步集中与层内收缩不是唯一训练成因，不直接加步。
3. 三问题从Mode C1.69继续形成可分离的新Idea：

| 问题 | 已有证据与下一项要区分的内容 |
|---|---|
| 高损害误选 | 159旧特征网常见V错误532→9，罕见229→278，新增49错均在混组但纯组仍16错。表达/表征有平均价值，稀有监督、组/对应及上下文未分离；不认证健康/信息绝对不足，救回新增与完整尾部一起核 |
| 混合收益未兑现 | 134训练侧存在不均匀条件空间，140/144未稳定兑现；区分实际恢复相容性、细小收益分辨率和可学性，不把条件空间叫方法成绩 |
| 零观测支持时的尾部大错 | 160单独端点局部有益，短窗稳健速度/联合无未收敛或饱和仍灾难；不再无依据枚举同窗统计。需新的传播条件/风险信息，不能只V设horizon/速度cap，保住方向/强控/完整尾部；真实物理原因未定 |

输入、原生时序、关联和对象输出仍在研究范围；已有候选/未采纳分支留在原矩阵与Idea中，不因近期优先融合而删除。

10月31日讨论已落实为[159/160独立Idea和真实验证](proposals/counterexample_factorial_20261031.md)，不是只整理；局部表征因151新96未存events明确使用支持网格，派生速度不当独立健康通道。已有网络为研究候选，不预定最终骨架。当前新后端增量仍只有位置，速度/航向同时保存评价；K2/离线/共享开发域及V重复使用不称独立泛化。三项本轮各一次验证已收尾，无自动新READY或扩大训练。

## 保留的结论

| 对象／实际范围 | 当前能支持的结论 | 完整证据 |
|---|---|---|
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

149–151/153–161已完成记录与对应回流；151数据准备不新增方法成绩。159/160按事前安排从完整冻结数组补齐四属性/方向，未倒填独审；155–158此前漏项的订正保留，速度航向无新学习增量。161后验同query跨输入分解及交接索引迟记单列偏差。详细人口、成本、不利结果和失败原件保留。152仍训练中，无新公平排名。

## 研究材料与推导入口

以下资料库链接为本机原件路径，GitHub不包含对应全文；原分析完整保留，新结果按版本追加。矩阵、Idea与工作流分析保留完整正文，摘要只作导航。

| 需要什么 | 入口 |
|---|---|
| 全局P/F/M/E/C与回流 | [Mode C1.69](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-169)，累计P14/F258/M109/E161/C55 |
| 多源贡献范围与条件 | [多源可靠性矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/multisource_reliability_matrix_20261030.md) |
| 选源与集合读出 | [SR v1.8，51格](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#local-expression-feedback) |
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
