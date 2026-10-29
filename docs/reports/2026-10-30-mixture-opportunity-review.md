# 134：训练侧混合空间与高损害误选的记录及深挖

2026-10-30。**同一合法状态下，混合确有有限、条件性的空间；同时，当前选源的少数高损害错误集中于小分数差和部分高残差/混组条件。** 这是T侧reference诊断，尚不是任何新方法的可部署成绩。保留一般K的共享贡献方向，不把99.32%的历史source角色优势写为固定优先级方法。

## Record：实际执行

- package_type：EXPERIMENT_RECORD；package_version：research-pipeline-contract v1.0。
- experiment_id：LEVEL1-MIXTURE-OPPORTUNITY-20261030-134-attempt-1；source_idea_id：IDEA-MIXTURE-OPPORTUNITY-20261030；plan_version：1.0。
- source_matrix_id：MC-MULTISOURCE-RELIABILITY-20261030@1.2；selected_C_id：MR-C3接MR-C1；来源F231/M102(AI)/E129，局部1.2事前15格。
- research_question/hypothesis：检验原T合法状态的参考最优凸混合相对逐时刻最佳单源的空间；用原OOF分数量化高损害误选条件，区分排序、权重尺度、最终风险与数据角色混杂，再决定下一方法。
- execution_status：COMPLETED/exit0，一次attempt-1；独立审计COMPLETED/exit0，非额外训练或第二次科学尝试。
- 事前[Idea](../proposals/mixture_opportunity_20261030.md)、[规格](../specs/mixture-opportunity.md)、[L2S](../../HandOver/L2S-20261030-134.md)；代码冻结`0c23bd5`，`6eb7c57`只记实际启动。原argv/env全部保存[科学launcher请求](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_mixture_opportunity/134/launcher-request.json)及[审计launcher请求](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_mixture_opportunity/134/audit-launcher-request.json)。
- 新run `/home/wangguangjie/djs/association-code/runs/LEVEL1-MIXTURE-OPPORTUNITY-20261030-134-attempt-1`；独立checkout `association-code-run134`。两服务`mtad-da36326c911f.service`/`mtad-b2411a581464.service`均Linger=yes、Restart=no，事后inactive/dead、success、ExecMainStatus=0，[终态原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_mixture_opportunity/134/postrun-service-state.json)。科学launcher北京时间2026-10-30 02:06:40→02:08:14；审计02:08:48→02:09:55。

### 计划与实际、信息边界及成本

仅133已完成T training/OOF和127的T_core motion/evaluation，原manifest前32场/12组件。0新拟合/teacher前向/神经更新/ZIP/CSV/V数组/test；没有根据V挑模型、尺度、阈值或温度。reference只用于显式oracle、误差与诊断分层，任何权重都不发布为合法预测。新run未改132/133。

| 项 | 事前设计 | 实际 |
|---|---|---|
| 主人口 | 原133首anchor/首目标代表/ref位置有效/双源query | 全22627点，无删样；actualK2，不宣真实K>=3 |
| 恢复 | alpha源质量均分给本源支持原节点、保持133球面恢复 | 原节点平均latitude/sinlon/coslon充分统计；与端点和OOF真实hard逐点绑定通过 |
| 搜索 | 4097等距比例、含两单源端点；不优化/重试 | 每点全网格，候选准确总数为22627×4097＝92702819；输出grid内最优及连续gap |
| 对照 | 参考最佳单源、固定网格；3原OOF头 | 完整LINEAR/QUADRATIC/SHUFFLED；三头均有0分数平局 |
| 分析 | 12组件/去最大/2000次seed20bootstrap；预定分层 | 全部保存，全32公共逐场和所有条件层未省略 |
| 预算 | CPU4/RSS6GiB/AS12GiB，1800s advisory | CPU0–3/4线程、GPU0；程序73.617s/launcher93.953s；RSS486559744B≈.453GiB，未超建议 |
| 核验成本 | 独立数组/几何/指标 | 65.304s/launcher66.669s，RSS119250944B≈.111GiB；87453数值、223203 exact、61479 null检查通过 |

上表搜索候选数按乘法精确记录；不把候选数当实验重复。程序、初始化、传输/开发、旧teacher/原缓存及独立审计成本分别说明，上述不是整链延迟或新训练预算。发布前review修正了误差方向/排除scope漏项、log反解的分层浮点边界、选中中间权重不等于严格混合收益的平局语义；未运行134前完成并更新Idea，无事后改配方。软件fixture曾因审计scope未同步失败，修正后全链通过；不记为真实尝试。

### 全人口与排除项

全部911预测组、860目标：825有首anchor目标、35目标无anchor、86重复组、0未知anchor、147混身份组。全组58304个query中，无支持17771、单源16930、双源23603；这是全预测组scope，不能与主参考人口混用。

全部目标reference有效点38224＝首代表无支持1487＋单源13521＋双源22627＋无anchor目标的589点；首代表另15165个query无有效位置reference，重复组query5504。没有因为不可改/无ref而删去计数；主诊断空间仅22627双源点，未报告完整系统新的融合MAE。

### 主结果与数值/样本不确定性

| T主诊断量 | 结果 |
|---|---:|
| 逐query参考最佳单源MAE | 269.835m |
| 4097比例内参考最佳混合MAE | 230.235m |
| 可行平均增益 | 39.600m（约14.68%，仅此条件人口） |
| 增益中位数 | .346m |
| >1m增益 | 10962/22627，48.45% |
| >10m增益 | 8988/22627，39.72%；收益和886326.196m |
| >100m增益 | 3226/22627，14.26%；收益和624482.265m |
| 端点也达网格最小值 | 10806/22627 |
| 严格正增益>1e−6m的内部解 | 11821/22627 |
| 所选alpha在(.01,.99) | 10074/22627；仅所选权重描述 |

主增益的12组件bootstrap95%百分位区间[38.736840,44.493980]m，固定2000次seed20、按组件重抽且按点加权。这里只反映T诊断异质性，不证明独立测试显著性或可学习性。包含端点使oracle收益按构造非负，重要的是收益大小/范围，不是“全组件为正”这一形式本身。

连续误差界全22627点有定义：均值.351531m、中位.322734m、P99 .617567m、最大1.505538m；22609点（99.92%）界≤1m。故相同恢复空间下连续最优的平均收益介于[39.600354,39.951885]m；这是数值界，与上面的组件区间不同。没有精确连续全局最优主张。真实grid平局0、未定义方向0、未定义gap0；没有用退化回退解释收益。

完整原件：[summary](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_mixture_opportunity/134/summary.json)、[32逐场gzip](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_mixture_opportunity/134/scene-summary.json.gz)。主要oracle、端点/连续界、OOF真实hard/选择代价、完整scope/组件/bootstrap、预定条件层及方向cos经独立numpy复算；不调用生产core。独立最大oracle误差差约3.90e−9m。[审计](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_mixture_opportunity/134/independent-audit.json)未认证原native缓存/物理reference、原133训练和上游模型，也不独立认证浮点exact网格平局数量/排序；所选点误差/位置仍独立验证。

## Squeeze：从空间与失败条件决定研究目标

- package_type：SQUEEZE_REPORT；package_version：research-pipeline-contract v1.0；squeeze_id/package_id：SQ-MIXTURE-OPPORTUNITY-20261030-134。
- input_experiment_record_ids：上述134；evidence_status：partial（本轮执行与限定核验verified，物理机制/可学习性/真实多传感器泛化未知）。此处只分析现有原件，不增加试验或训练。

### 1. 有混合空间，但不支持处处平均或固定混合幅度

>10m的39.72%点贡献总可行收益的98.92%；>100m的14.26%贡献69.69%，而中位数只有.346m。最佳单源本身已是使用reference的逐时刻动态上界，比“全轨固定一个源”更宽松；能超过它表明真实的几何可行空间，但oracle比例用了reference，不能保证合法特征能学到这些细小权重或在V实现39.6m。

纯组19108点的平均空间38.631m，混组3519点44.863m；互补空间并非只有错误组造成。但物理原因可能包括随机残差方向抵消、系统偏差/参考构造及原状态误差，不能据有空间就认定相关Cov模型成立。误差方向cos是reference处球面切平面方向，反向/同向不直接等价传感器独立/相关。

| T组件 | query数 | 可行平均增益m | Q选择正损害和m |
|---|---:|---:|---:|
| component-13bf8d8c05af98b86d432a08 | 1461 | 38.809 | 201258.509 |
| component-1b730762e8dcc866d831d9a3 | 250 | 49.150 | 0.000 |
| component-23608a7a22c5322f7254ffc1 | 14493 | 38.925 | 382634.911 |
| component-2631a2fab1075a4e376f5869 | 90 | 42.392 | 8949.967 |
| component-4eb65272866a82b14a0055d6 | 124 | 40.656 | 13520.287 |
| component-5ee24adf22c874beaf8d076d | 122 | 89.244 | 268.643 |
| component-75391b641c3cd7287e382451 | 121 | 40.599 | 9263.013 |
| component-805a0d28ca2328b9fac2c2f2 | 222 | 27.859 | 0.000 |
| component-9ea441151d6468bd21826aea | 195 | 35.022 | 5093.983 |
| component-c7082a697d934f0004db48b9 | 4825 | 39.822 | 414090.050 |
| component-d784c25f6ac4208b0a38e796 | 477 | 53.033 | 34369.081 |
| component-e062a18f3a426eba715c16d2 | 247 | 31.611 | 0.000 |

去最大组件后8134点仍有40.805m平均空间，避免只看占14493点的最大组件。该空间尚不决定网络结构，更不允许把133/127冠军相加。

### 2. 多数选对仍会被少量高代价错误拖累

| 原OOF头 | 实际hard MAE m | 选源准确率 | 误选query数 | 实际excess>1000m点 | 全体top1%损害份额 |
|---|---:|---:|---:|---:|---:|
| LINEAR | 345.172 | 96.924% | 696 | 616 | 50.89% |
| QUADRATIC | 317.099 | 97.339% | 602 | 471 | 58.91% |
| SHUFFLED | 1994.303 | 35.882% | 14508 | 14267 | 3.07% |

Q的602个误选仅2.66%，但471个query比参考最佳单源多损失>1000m。Q实际hard的平均excess47.264m，总正损害1069448.444m；全体最坏227点（ceil1%N）占58.91%损害。微小负excess总和约3.2e−7m属于浮点误差，不能称已发现混合平局收益。个别0误选组件的positive_harm只有约6.6e−9m，top1%比例可因极小分母显示100%，是数值比例假象，不解释为该组件的真实高损害集中；全体58.91%的分子/分母为真实大误差。

| Q分数margin（log误差得分差） | query数 | 选对率 | 全部Q正损害份额 | >1000m点 |
|---|---:|---:|---:|---:|
| 0..0.05 | 752 | 63.165% | 49.69% | 236 |
| 0.05..0.1 | 1108 | 90.162% | 18.41% | 93 |
| 0.1..0.25 | 8143 | 98.219% | 21.26% | 96 |
| 0.25..0.5 | 8198 | 99.390% | 6.98% | 30 |
| 0.5..1 | 4413 | 99.524% | 3.66% | 16 |
| 1..inf | 13 | 100.000% | 0.00% | 0 |

margin<.05的752点仅3.32%，贡献49.69%损害；<.1共8.22%点贡献68.10%。这使“分数不确定时的米制风险”成为具体问题；**没有验证低margin就应等权、改选某固定源或删除输出**。高margin也不是完美保护：.5–1层仍16个>1000m错误。

### 3. 条件联系及不能解释之处

- 被选源OLS RMS≥3000m有1348点（5.96%），选对率88.50%、贡献31.58%损害；RMS100–500m层选对率98.35%。RMS是条件代理，未干预，不能将其称真实传感器不确定度或根因。
- 混组3519点（15.55%）选对率92.47%、平均excess107.505m，贡献35.37%损害；纯组仍有691138.605m损害和316个>1000m点，故不能将问题完全归因分组。
- 本样本被选源最近时距全部<60s，预定后续时距层为空，不能据这次分层证明异步/长间隔是错选原因。native数量<16只有19点，虽选对率差也不能拿小层解释总体；内部离散<100m承载98.80%损害，普通源内一致性并非充分保障。
- 全部预定分层、空层、三个head完整保留，不挑有利区间。层间相关、样本不均和多重分析使这些结果仅是条件线索；部署规则仍要只在T训练并留组件验证。

### 4. 生成角色混杂仍很强，不能退回编号规则

匿名role_0在参考最佳单源中占22473/22627＝99.3194%，role_1只有154点。角色编号仅从原source等价类生成、私有映射留服务器，不作为预测输入，也不等真实传感器健康或作者生成角色已查清。它说明为何固定源历史控制很强，**不是选择它为最终方案的理由**。

共享函数虽不输入source ID，密度/残差可能仍识别历史角色。要满足用户可扩展目标，后续质量贡献必须由状态/观测证据决定，按少数质量反转条件和留出组件检查，保留来源重命名/变化K的结构测试；MTAD实际只有两个生成源，不能以这些测试认证真实K>=3、换设备或在线端到端。

### 5. 本轮决策与下一方案

当前保留的骨架是“统一状态→合法证据/组→共享集合贡献→几何恢复”，不是FixedLOW或固定两动作配方。134确认两项不同目标：约39.6m的条件混合空间，以及约47.3m的OOF选源平均可避免损害。新目标应直接检验**最终位置风险和高损害误选**，不再把逐源log误差小或softmax数值当作贡献校准。

下一最小验证先建立只由T决定的风险校准读出作为A′，保留源等权/原共享hard和原soft作A；若只是纠正得分尺度就能改善，应先拿到该增量再决定是否加入共享小网络B。若校准无法利用39.6m空间，才检验直接米制融合损失/条件贡献学习，并把少数高代价错误单独计入。T留组件统计/拟合隔离，V只用于一次固定评价，不凭这次V追温度；不把低margin硬编码成等权或固定source优先。不预定相关Cov/大网络已有效。

新模型/校准必须另有具体Idea、信息范围、对照和预算后发布独立READY；134当前固定终点结束，0新模型训练。132现场保留，输入/时序/组与输出的0/1支持域问题继续独立发展。已有相关方法是成熟起点，不能仅因已有先例否定该适配空间；贡献强度由合法可学增量、强控和泛化证据决定。

### 6. 回流契约

mode_C_input_bundle：source_ids=[本Record,SQ-MIXTURE-OPPORTUNITY-20261030-134]；复用MR-P1/MR-P3，新增F232（训练侧条件凸混合空间与OOF高损害误选集中，DE）、M103（读出尺度/风险目标与不确定性处理的候选，AI/UN）、E130（本记录/完整原件/限定独立审计）。MR-C3由空间未知变为目标K2内部有可行空间但未验证可学；MR-C1新增风险读出/角色混杂边界。P14/F232/M103/E130/C53，精读100=70+30不变；next_skill：deep-read Mode C→成熟时discover-innovation，不把路线建议当已验证方法。

本地只取完整脱敏run/summary/32行gzip、科学/审计两launcher的真实请求/终态/日志与服务快照；每点坐标/reference/来源名/身份/alpha/特征/OOF数组均留原服务器private。原件[入口](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_mixture_opportunity/134/)，未复制实时任务队列。
