# 当前研究状态

更新：2026-10-31。本页只维护当前目标、结论与下一步；执行只看[HandOver](../HandOver/Handover.md)。整理前的问题推导、矩阵/Idea连接、取舍与执行沿革已[全文保留](history/2026-10-30-research-process.md)，不以摘要替代研究材料。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。以本领域真实存在、尚未很好解决的问题为起点，允许更换整体骨架；成熟方法的合理跨域适配、有意义的新增价值均可研究。按[AGENTS](../AGENTS.md)、[完整飞书采纳流程](research_workflow.md)及[项目skills](../.agents/skills/README.md)推进，不先定方法再找依据。

最终方案不依赖源编号优先、仅两源平移或固定双源混合配方；可变K、来源换序/增减是研究要求，真实数据目前仍主要K2。精度优先，轻量/在线/端到端是加分目标。用户接受合理T侧监督拟合及新数据集重拟合，但倾向少依赖后端reference系数拟合，不等于禁止学习。

## 现在做什么

[178历史候选动作验证](reports/2026-10-31-history-candidate-review.md)已完成科学/规定独审，Record→Squeeze→Mode C1.87/SR v2.9：原T128/13ridge＋13神经fit/3328GPU更新、原T32-V32/24臂，总1800.894s，98公共原件与清单完整回传。原HARD/STRUCT/159强控及分组不变，同一新评分器M/O与均衡E、零Z/单1/联合O分离，来源充分107不增。

**保留历史候选风险读出的真实平均收益，尾部未解。** V HARD MAE452.992→355.760m、RMSE1502.760→1329.100/P995166.150→5003.510；STRUCT427.241→328.916m、1447.519→1267.936/4842.083→4359.930。O−M及O−E共同均值组件描述区间均在0下，去最大仍改善；全部分组/raw-decoded/coverage/速航exact不变，不称覆盖/前端提升。原两头不同coverage/强控不拼成单方法，仍仅V32，不能替换全300/同行。

V联合相对对应base仍新增128/107个>1km增害，纯组28/19、零支持32/47、单支持96/60；最大增害6.782/16.984km，T P99反而变差。单支持同人口下，相对全历史oracle差>1km的M→1点数V2655→337/2582→297，但这含动作合法性差距，不是M在合法集内排错；oracle不部署。下一从T的纯单支持风险误选与零支持状态传播分别结合来源形成新干预，保护当前平均收益，不直接沿V设cap/拼策略/加步。跨片段历史链接与原生状态重估独立保留；发生域不等根因。178无自动全300/新seed或后继READY，执行终态只看HandOver。

[176最终指标/覆盖工作点](reports/2026-10-31-decoded-coverage-review.md)与[177低支持来源分离](reports/2026-10-31-low-support-attribution-review.md)均已完整收尾，Record→Squeeze→Mode C1.85/SR v2.8。176原T32/V32十八臂、0训练/2240T场候选，R/D/C全选回原HARD/STRUCT阈值；约束未激活，无新增收益，当前网格/预算配方结案。科学＋规定收尾634.112s、96公共原件及清单回传，事后等价核3.359s另记。

177原T32/860对象、0V/训练/新部署，36.979s成功、57公共原件＋终态记录回传。HARD−STRUCT195个单源害全部单受支持片段；纯0域122害当前组已有状态oracle无改善、其他兼容输出oracle106点有空间。HARD−WS纯1域108害，受支持片段oracle0、全当前组含支持外历史可改善94；不同强控保反证。均是动作空间/对应敏感性，不能部署oracle、判reference错或宣根因。

围绕**组内原生时序/支持外历史状态的恢复与风险读出**、**跨片段/输出的历史一致性**的首个最小干预178已完成，正负证据如上；不是只加旧选源头训练或重跑149均衡/最近。HARD覆盖、STRUCT均值强控保持，C59当前配方不继续扫更细θ。不自动全300或新seed；充分107不增。

[174 T分离](reports/2026-10-31-association-decision-review.md)→SGC/OHEM独立AP→Mode C1.81/C58→[175真实训练与完整Record/Squeeze](reports/2026-10-31-decision-error-mining-review.md)→**Mode C1.82/SR v2.6已完成**。原T128两头各512步/共1024GPU更新、原T32/V32十二臂；原四强控exact、1548T候选/512步mask及全部人口/关联/四属性/尾部独核，科学＋收尾863.005s、99公共原件回传。充分107＝72+35不增；新增NeurIPS2025来源在冻结后partial，非事前依据。

**保HARD一般难例训练作为覆盖—关联折中候选。** 对原WS V32：点覆盖98.442→98.892%、对象1388→1401（救22丢9）、decoded跨源mPCA78.989→81.759%/mPFA15.550→14.924%、MAE527.427→452.992m；共同均值−75.273m区间不跨0，去最大组件仍改善。但关联区间均跨0、1499新增>1km害，对STRUCT共同MAE+24.721m/区间不跨0，不能替换全部强控或宣稳定全胜。raw召回89.710→89.646%略降，与decoded改善分开；总真跨边少1，宏平均不等总边增量。

**DEC构组错例反馈当前配方不扩大。** 与HARD同逐类辅助数/曝光，覆盖再多8目标但decoded mPCA−6.193pp/PFA+.909pp、重复203→320；1123新增害中1122在新0/1支持域。T也较HARD弱，不是只有V失效；发生域不是因果认证。下一从HARD保存T救回/新增害分离按query支持、源内片段历史/代表对应及状态回退条件，再形成独立干预；不退源优先，不沿V调λ/θ，不自动新seed/全300。

本轮[独立结构监督AP](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/structured_association_sources_20261031.md)→Mode C1.77/C57→171技术失败保全/172明确修复资格→Mode C1.79→新[Idea173](proposals/structured_association_20261031.md)→实际1024GPU更新与原T32/V32八臂→[完整Record/Squeeze](reports/2026-10-31-structured-association-review.md)回[Mode C1.80](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-180)，P14/F275/M111/E178/C57。全部原launcher/科学/限定独审/四属性及1032T候选独立核完成，科学＋收尾671.161s。充分深入107＝72领域/相近＋35跨域（2023机制基础例外，LA-MOTR2025补充/代码partial不计），7项软件验收在READY前通过。

**173保条件增量，不整体晋级。** 同信息/容量/初始化/曝光/工作点下，STRUCT对继续BCE的V有效点coverage96.888→98.032%、MAE438.835→427.241m，共同差−13.827m描述CI[−89.959,−4.464]；但decoded跨源mPCA82.939→81.765%、mPFA14.486→15.573%，933对BCE新>1km害。对原WS覆盖仍低98.442%、救12丢20/1131新害；对冻结CTX平均区间跨0且对象少1。raw召回对BCE有增量却未同向兑现decoded；forest–WS/工作点/桥边失配与状态相容均候选，**未证唯一根因**。未知K部署仍WS，不用训练GT k/固定源。当前λ1/B4/4轮配方关闭自动扩大/全300；其后T分离已由174完成，新175为单独C58实验，不是续173配方。

用户明确优先提高覆盖率与跨源mPCA/mPFA，并包括前端。前批STAR NeurIPS2025/TrackTrack CVPR2025正文、补充、反例及TrackTrack核心实现的[完整AP与阅读边界](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/association_context_sources_20261031.md)保留，STAR源码未获；当批充分106，当前累计107，非作者复现。[168](reports/2026-10-31-association-context-review.md)实际T128/2000GPU更新、T32/V32六臂：CTX有共同域均值/召回信号，但覆盖下降和1027新大害，未联合升级。[169](reports/2026-10-31-anchor-support-review.md)0训练八臂：固定代表仍伤位置，硬保每query支持几乎冻结搜索，当前配方关闭。全部真实原件/独审/四属性与后验记账完成，保反例，不退固定源。

**上述工作点强控已由[170](reports/2026-10-31-association-operating-point-review.md)实际完成，不再待设计。** 0训练/teacher、原T32/V32八臂，同一T raw跨源mPFA预算/774候选独立核、V前冻结。V CTXCAL跨源mPCA80.684%/mPFA15.369%、MAE432.801m，但coverage97.694%仍低原98.442%，救17/丢24目标及1001新大害；共同域较同规则OWN有平均增量，跨源指标变化区间均跨0。原scorer标定不变；OWNCAL覆盖略升却mPCA/尾部更差，不能拼冠军。168–170全部原launcher/限定独审/四属性及后验记账成功收尾，回[Mode C1.76](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-176)，P14/F270/M110/E173/C56，106篇不增。当前无覆盖＋跨源双指标稳定联合升级；保关系条件平均信号，下一由T反例核错并对象的输出代价与pair/构组目标接口及成熟解法，不沿V扫阈值或自动加同BCE步数。

**完整V300基准仍为固定T128头582.808m；新159只评价原T32/V32，不能替换全300成绩。** [159表征×表达](reports/2026-10-31-local-expression-review.md)已完成6656次真实GPU更新：V32旧特征残差网540.295→527.427m、误选761→287，但罕见反转229→278；保留平均收益候选，不称高害解决。局部网527.399m不支持稳定额外增量。[157完整结果](reports/2026-10-30-t128-full-scope-review.md)与旧[147基准](reports/2026-10-30-source-risk-full-benchmark-review.md)保持；[比较设计§9](reports/2026-10-27-fair-comparison-contract.md#9-2026-10-30三种同行共同任务比较的后继设计)仍区分固定WS后端任务与完整系统。

本轮165–167服务器科学、限定核验与补项全部完成，并回流Mode C1.73。[165无真值分组](reports/2026-10-31-truth-free-partition-review.md)固定056分数/159头，允许split/move：V32完整MAE527.427→517.947m，P99 7260.394→7166.747m；但共同域区间跨0、救5同时丢2目标、T完整597.179→597.588m，仍有140个新增>1km增害，当前仅保条件候选、不扩大。**140个新大害全部在新组零/单源支持域（22/118），且都same-anchor，pure→pure也有60个**；这部分不能继续归为小网络多源选错。

[166集合质量上下文](reports/2026-10-31-set-context-risk-review.md)冻结159，加OWN/CONTEXT同容量残差作控制，26fit/3328GPU更新；V527.424292/527.430938m，T纯44/V纯16原错均未救回。CONTEXT无稳定增量，当前配方不扩大、保原159平均收益；同信息换loss/头不再无依据枚举，不写物理信息绝对不足。原[163参考对应诊断](reports/2026-10-31-group-reference-review.md)仍不证明官方reference错，也不把真值净化丢D2当救回。

[167原TFM损失梯度](reports/2026-10-31-tfm-gradient-review.md)固定107 T2/0更新：输入梯度非全断，但当前加权position左输入约1e−8、memory约1e−6、输出投影约5–6，eval/train模式方向相近；量纲/参数数不同、历史训练未复放，不认证唯一根因。正常输出与164/161 exact仍约58km；107不重训。旧160端点/速度和其他反例保留，不拼冠军。

[151完整数据准备](reports/2026-10-30-full-peer-data-review.md)已完成T2100/V300、46126可训练组。152A/B仍真实systemd RUNNING并实际优化，fresh seed20、固定150/20epoch对应427800/80800更新，可能跨多天；尚无末轮全V公平分数，实际训练进度只看HandOver。冻结活动合同仍比较原147，不改成157/159；旧148不续跑。

1. 176–178均完成，不重复旧工作点/上界核查或无依据续训同头。保178历史候选平均收益，由T新增害分别针对纯单支持风险误选、零支持原生状态/历史传播及跨片段一致性读充分来源/公式/消融/代码，回矩阵后形成可区分干预；原组内状态无空间与有好状态却选错不可混为一类。HARD/STRUCT强控保留，不用oracle部署、不退固定源，不沿V调参。
2. 纯组T44/V16难例与分组支持路线分开，先查现有统计/网格编码没保留的原生时间、局部状态和reference条件，形成可区分的新证据再定实验。166当前联合质量表达没有救回，不自动加步/seed/换loss；0支持旧首anchor回退/圆周歧义未解。真实异构K≥3/新域仍缺，不退固定源优先。
3. 跟进152固定全T训练/末轮V300同WS任务比较：全部曝光/物理学习、强控/coverage/尾部/成本，PTFM核心缺域与公共Mean扩域分报、DeepAF P/V适配权限明确。不能把共享WS比较称原完整系统复现，活动合同仍比较147。TMTAFF若再做新训练干预须据原loss几何/梯度路径制定可分离具体合同，不能直接续107或以梯度大小选方案；最多四单卡/FIFO，无SSH值守或自动新增训练。

| 问题 | 当前分离出的待解条件 |
|---|---|
| 高损害误选 | 175 HARD保覆盖候选，1499新害1311在0/1域；DEC相对HARD1123害1122在0/1域，发生域非根因。159平均收益保留；165的新大害全在0/1域，先解决分组后的支持/回退；166纯组44/16仍错，原生信息/状态/对应条件另查，不能混归一个根因。 |
| 混合收益未兑现 | 134训练侧空间未变，165仍hard/普通Mean比较，不是连续混合效果；对象/按query支持和实际状态相容性需先明确，再研究风险读出。 |
| 零支持尾部 | 165即使pure→pure也可从1支持变0后大错，组关系/状态回退耦合需核；160短窗速度/联合灾难不重复，无新证据不V设时长/速度cap或拼局部冠军。 |

输入、时序、关联、融合与输出仍在研究范围；原矩阵/Idea/未采纳理由保留。165改变分组使速度/航向和population也变，完整属性已核；166无新属性学习。真实K2/离线/共享开发域/V重复使用不称独立泛化。165–167本轮各一次科学已收尾，无自动新READY；后继从Mode C1.73进入具体Idea后执行。

## 保留的结论

| 对象／实际范围 | 当前能支持的结论 | 完整证据 |
|---|---|---|
| 历史候选动作，T128训练/原T32-V32 | 3328GPU更新，匹配M/E及Z/1/O分离；V HARD355.760m/STRUCT328.916m，保平均/RMSE/P99信号且coverage/关联exact不变，但128/107新>1km害、T P99反证保留，不称尾部已解 | [178完整记录](reports/2026-10-31-history-candidate-review.md) |
| 最终指标/覆盖工作点、T低支持分离 | 176十八臂/0训练，R/D/C均选原点，当前配方无增量；177证实单支持害多无选源动作，组内历史与跨输出候选空间不同，oracle不可部署 | [176完整记录](reports/2026-10-31-decoded-coverage-review.md)、[177完整记录](reports/2026-10-31-low-support-attribution-review.md) |
| 匹配难例HARD/DEC，T128/原T32-V32 | 1024GPU更新/12臂，HARD对WS覆盖/decoded双指标/均值点估计同向，MAE452.992m；关联区间跨0/1499新害、对STRUCT均值差，保折中候选。DEC覆盖最高但未超匹配HARD，当前配方不扩大 | [174分离](reports/2026-10-31-association-decision-review.md)、[175完整记录](reports/2026-10-31-decision-error-mining-review.md) |
| 结构监督同曝光，T128/原T32-V32 | 两头各512步/共1024GPU更新，STRUCT对BCE coverage96.888→98.032%、共同MAE−13.827m描述CI不跨0；但decoded PCA82.939→81.765/PFA14.486→15.573%、933新>1km害，原WS覆盖仍98.442%更高。保条件信号，当前固定配方不扩大；原171技术失败保全/172资格成功 | [171/172](reports/2026-10-31-object-forest-qualification-review.md)、[173](reports/2026-10-31-structured-association-review.md) |
| 同T工作点控制，原T32/V32 | 0训练、三scorer同T规则/774候选独立核；CTXCAL跨源mPCA80.684%/mPFA15.369%、MAE432.801m，但coverage97.694%低于原98.442%，救17丢24/1001新大害；当前无稳定联合升级 | [170](reports/2026-10-31-association-operating-point-review.md) |
| 前端上下文，T128训练/原T32-V32 | 2000真实GPU更新、同容量OWN控制；CTX跨源mPCA78.989→82.183%、MAE527.427→436.131m，mPFA基本不变且覆盖下降，保平均/召回条件信号 | [168](reports/2026-10-31-association-context-review.md) |
| 代表/支持硬约束，T32/V32 | ANCHOR保人口但位置更差，SUPPORT T0/V2moves、无实质联合增量；两当前配方关闭，不永久保原错组 | [169](reports/2026-10-31-anchor-support-review.md) |
| 无真值分组修正，T32/V32 | 同score/159头，V完整527.427→517.947m/P99下降，但救5丢2目标、共同差区间跨0、T完整略差；新增140个>1km增害全新0/1支持，pure→pure60。保条件信号，不自动扩大 | [165](reports/2026-10-31-truth-free-partition-review.md) |
| 集合质量上下文，T128/原T32-V32 | OWN/CONTEXT同容量，26fit/3328GPU更新；CONTEXT V527.430938m不超OWN527.424292m或原159527.427043m，纯T44/V16均未救回；当前配方不扩大 | [166](reports/2026-10-31-set-context-risk-review.md) |
| 冻结TFM原loss梯度，T2四正对 | total/position梯度分开、0optimizer/更新，输入非全断但量级小、输出投影较大；TFM/损失权重exact未变/eval绑定164，仍58km。非唯一训练成因，107不重训 | [167](reports/2026-10-31-tfm-gradient-review.md) |
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

149–151/153–167已完成记录与对应回流；151数据准备不新增方法成绩。159/160按事前安排从完整冻结数组补齐四属性/方向，未倒填独审；155–158此前漏项的订正保留，速度航向无新学习增量。161后验同query跨输入分解及交接索引迟记单列偏差。162 READY早于本地fixture通过的时序偏差保留，实际提交前修正并通过；163中央人口/参考独审为科学后追加，未倒填事前设计。165支持/纯度/anchor深挖为科学后只读分析，实际成本/原代码单列；167新dropout/当前梯度不等历史训练复放，入口读取时序不当科学故障。详细人口、成本、不利结果和失败原件保留。152仍训练中，无新公平排名。

## 研究材料与推导入口

以下资料库链接为本机原件路径，GitHub不包含对应全文；原分析完整保留，新结果按版本追加。矩阵、Idea与工作流分析保留完整正文，摘要只作导航。

| 需要什么 | 入口 |
|---|---|
| 全局P/F/M/E/C与回流 | [Mode C1.85](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-185)，累计P14/F282/M114/E185/C59 |
| 多源贡献范围与条件 | [多源可靠性矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/multisource_reliability_matrix_20261030.md) |
| 选源与集合读出 | [SR v2.4，87格](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#structured-supervision-feedback) |
| 观测驱动与状态恢复 | [OD v1.10，54格](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md#endpoint-separation-feedback) |
| 逐篇阅读范围、来源与代码覆盖 | [深入阅读审计](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/deepread_coverage_100_20261028.md)，充分深入107＝72领域/相近＋35跨域；不是作者复现 |
| Idea完整设计、假设与否证 | [提案目录](proposals/)；各Record链接对应Idea，未执行/未采纳不删除，135仍仅保留草案 |
| 历史推导、连接与取舍原文 | [完整研究过程](history/2026-10-30-research-process.md)、[结案摘要](history/README.md)；历史“下一步”不自动成为任务 |

## 判断边界

- 正式Level1 T2100/V300/封存测试600；名单不按模型成绩挑选。T/V共享6053个原A键，V最大两组件占63%，不能称身份独立或独立确认性泛化。[数据约定](data_contract.md)与[指标定义](mtad_data_protocols.md)是唯一规则入口。
- ME不是MAE；作者ESD实现未知。raw/decoded、固定组/完整系统、各自/共同有效人口分别报告，保留覆盖、重复、组件和尾部，不删难点。当前后端位置实验的速度/航向仍沿用Mean，不宣四属性学习增量。
- 支持域、reference生成/物理准确性及长尾来源未完全查明。三场原始短时大位移仅支持所查案例，不据此认定全数据坏。[三场原链](reports/2026-10-27-three-case-chain-review.md)、[原CSV核验](reports/2026-10-27-raw-case-chain-review.md)。
- 132在SUMMARY完成1000次GPU更新后被oomd终止，完整评价及另三臂未完成；用户要求保留现场，管理员日志暂缺，不自动重启/重训。[失败记录](reports/2026-10-29-ordered-input-continuation-failure.md)。原107/113及143/144/146已结案，不续跑旧额度。
- 换loss、局部修复或软件通过不证明机制，不因单次失败否定整类方法，不把真实K2接口能力说成K≥3/异构泛化或在线端到端。
