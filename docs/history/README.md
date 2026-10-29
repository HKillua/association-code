# 研究历史摘要

2026-10-27整理。当前结果只维护[研究状态](../research_status.md)，执行只认[交接索引](../../HandOver/Handover.md)。已结案命令、旧READY、预算和历史“下一步”不恢复成新任务。**优先解释真实实验，不以工具、测试、文档或诊断轮次数量替代科研成果。**

## 保留的研究事实

| 阶段 | 结论与后续约束 |
|---|---|
| 133共享源误差/可变K接口 | 39解析fit，T32/V32/十臂45对一次退出0，61483数值及训练/权重/人口独立核验通过。Qsoft对源等权MAE1276.441→1128.755m，但Qhard547.721m仍不及LOW527.446m/旧Q435.061m。排序信号不等于混合增量，原V反复开发/K2限制保留；当前soft不自动扩量。[Record/Squeeze](../reports/2026-10-30-source-risk-review.md) |
| 132输入继续的真实部分训练 | 修复FD并复用2400缓存/norm，SUMMARY真实1000GPU更新及权重完好，六Adam step1000；末步评价阶段被oomd杀死，另三臂0步、无完整科学成绩。原RUNNING陈旧、单独终态合成保留，不自动第三次或重训已完成1000；系统触发/评价峰值待查。[Record/Squeeze](../reports/2026-10-29-ordered-input-continuation-failure.md) |
| 131固定收益头全V | 原32导入/只新算268，0fit或神经前向。新268同370073点hard500.498→Q484.847m，256场/34组件好、P99+.604m，Q RMSE不及置乱、对FixedMean多数场差等反例保留。全300同441144点491.889→476.826m；48汇总主位置/人口及完整import独立核验通过。[Record/Squeeze](../reports/2026-10-29-conditional-source-value-full-review.md) |
| 127原生CV/RTS | 原T32九rho选1e-8，V32六臂/15对，同71071点RTS LOW MAE527.446→380.127m、全量四误差均好，仍有4393/最大组件SSE/exact-native尾部负例；FWD当前尾部差。离线有效点、不与125联合拼分；主位置/人口独立核验通过。[记录](../reports/2026-10-29-native-state-review.md) |
| 126输入零更新技术失败 | T2100/V300缓存及norm写出后，同时打开4200memmap超过unit软1024，真实exit1/0更新，phase normalizer陈旧；非科学负结果。原件保留，不重启，独立缓存复用恢复须新合同。[记录](../reports/2026-10-29-ordered-input-failure.md) |
| 125条件贡献 | T32新head LOCO及冻结V32十一臂真实完成；二次比hard MAE447.060→435.061m，30场/8组件改善，P95略差/覆盖重复不变。主位置/人口独立核验通过、首次audit字段失败保留。是固定候选经验收益信号，非物理源健康。[Record/Squeeze](../reports/2026-10-29-conditional-source-value-review.md) |
| 129原生对象输出 | T拒配有候选富集、duplicate217→175，但少24覆盖目标、混组增22；LOW同点MAE仅−.365m、P99+306.985m，宏F1升/微F1降。当前贪心组并入不晋级，保留组级拒配问题；可变R/decoded独立核验通过。[记录](../reports/2026-10-29-native-object-output-review.md) |
| 128失败及130几何诊断 | 128遭oomd、6场部分产物和陈旧RUNNING保留；独立130导入6/补26成功。共同候选LOO AUC.448/.527未超过静态MIN .893，候选AUC不是F1，当前固定LOO结束，不否定原生时序；限定标签/人口/秩审计通过。[失败](../reports/2026-10-29-association-geometry-failure.md)/[130记录](../reports/2026-10-29-association-geometry-review.md) |
| 124软贡献与中心幅度分离 | 原32/15臂105配对/0训练，320臂导入160新算；FixedHard447.060m，FixedSoft531.012、ShrinkHard455.287、组合555.446m。组合全部32场/9组件差，收缩P99改善伴随MAE/多数场损失；固定.75/m0=4不晋级，不自动扩量/搜参，不否定整类。独立主位置核验通过，[Record/Squeeze](../reports/2026-10-29-source-mix-shrink-review.md) |
| 123源优先更广范围 | 2464臂导入/536新算、一次全V/0训练。新268 FixedLOW MAE500.498m较LOW579.614/FixedMean602.833更低，仍177场比FM差/P99差；LOW未见与去最大组件四指标输FM。全V FL491.889m，非普遍硬源优先或同行胜负。独立主位置核验通过，[Record/Squeeze](../reports/2026-10-29-source-preference-full-review.md) |
| 120共享校正全V开发扩展 | 固定旧八臂，256原场景臂导入/2144新算；新268 MAE1297.482→602.833m、全V1294.074→611.405m，原32未见29组件仍同向，强控/尾部/覆盖限制保留。FixedHuber有不足1m MAE增量而RMSE/P95/P99反向；非独立测试或同行胜负。[Record/Squeeze](../reports/2026-10-29-shared-component-full-review.md) |
| 121源优先归因 | 同原32/71071点，LOW527.446m已优于FixedMean656.041m，FixedLOW再到447.060m；主双源均值额外价值未获支持。校正对源优先仍有80.386m增量，但P99恶化且FL对FM20场差；没有全V源优先成绩。[Record/Squeeze](../reports/2026-10-29-source-preference-review.md) |
| 122一阶速度相关剩余 | 原32/1605组全保留、1024可评；JOINT对STATIC输入相对残差均值+3.335m、RMSE+.613m，去最大组件后仍差；366解析拟合、0神经训练/物理输出改变。不接当前时间补偿，不能否定独立原生时序路线。[Record/Squeeze](../reports/2026-10-29-shared-motion-time-review.md) |
| 118共享相对分量 | 原V32/WS/同71071点，固定低源校正Mean MAE1276.328→656.041m、RMSE2168.266→1864.654m，并优于本轮Anchor/NEAR；P99有代价、对称gauge差、Huber无增量。0训练，独立主要数组核验；非绝对bias识别或同行胜出。[Record/Squeeze](../reports/2026-10-29-shared-component-review.md) |
| 119固定状态输出域 | 原V32/六臂只改位置mask；WS SPAN MAE790.264m却比FULL少1676有效点，coverage98.4417→96.1203%，共同误差不变；比UNION多保1094内部缺口且条件误差略低。0训练/恢复重算，不当状态增量/去重。[Record/Squeeze](../reports/2026-10-29-output-domain-review.md) |
| 117全V query控制与模块拆分 | R/W/恢复/全组输出接口可独立替换、历史数值exact。一次V300/38组件/8臂16比较，0训练；WS NEAR MAE1294.074→964.568m、300场均好，但强Anchor MAE更低/尾部更差，WS完整覆盖比A0低、重复1068>610。普通ALL最近取得大部分收益，支持/回退额外只−5.573m MAE；新paper bias两项更近0不等精度胜出。[Record/Squeeze](../reports/2026-10-29-query-contribution-full-review.md)／[论文比较](../reports/2026-10-29-paper-number-comparison.md) |
| 116固定身份×query贡献 | 原V32/9组件/8臂零训练，WS_NEAR同71071点MAE1276.328→927.435m、32场均好，但对Anchor MAE更差/尾部更好；WS来源等权当前无增量，覆盖/重复exact不变。透明贡献规则有实际效应，非完整方法净胜或同行比较；一次固定终点与独立数组核验完成。[Record/Squeeze](../reports/2026-10-29-query-contribution-review.md) |
| 早期Level2数据与完整流程 | 原共同网格存在名义时钟与观测错位，后改真实时间、原生有效短轨；20秒是编码单位。输出尺度修正不等于恢复有效；TCM特征MSE不能换算位置米误差，loss下降不等于完整系统成立。 |
| CDF、模式诊断与物理控制 | CDF F1 .354768低于A0 .426237，净增5088错边，打乱控制未支持专属机制，退役。dropout诊断解释随机距离，但关闭后TAM仍弱；不能当唯一根因。普通OLS改善位置，自由残差常伤已有支撑，不能由最终输出反推历史梯度冲突。 |
| 044—051模块/系统 | 学习关联收益能传到下游；融合增量有限。049参考TFM同点约63.8km，对普通控制约1.8km，未形成有效同行。050扩容未完整终评，随后切换Level1；旧人口、预算和分数不转移。 |
| 052—061 Level1正式基础 | T2100/V300/封存600。普通OLS组＋Mean位置1994.202m、覆盖62.80%；056边Macro-F1 .806738，固定9353组/610重复、位置覆盖97.2109%。固定组Anchor 885.579m、Mean1224.604m、支撑规则729.981m、固定.95为721.174m；真值二选一560.251m不可部署。困难采样/关联加训未形成覆盖与重复的全面改善，不能拼冠军属性。 |
| 062—073数值与学习竞争 | 066计算完成但原数值验收REJECTED，未被后续诊断改判。门控有局部开发收益、尾部异质性明显；4000更新Transformer两seed590.744/606.624m优于同seed门控646.750/635.776m，采用作位置基础。它们是通用控制，不是领域论文复现。 |
| 074—087领域适配与内部变体 | PTFM位置适配1845.38/1609.06m仍输自身物理初值，TSA适配也失效，不能据此否定作者方法或宣称胜同行。惩罚/收缩、helper聚合/权重/删除、PTFM去辅助目标和时间对应等未形成两seed稳定优势，相关配方关闭；不追加宽度、loss、seed或步数追分。 |
| 088—095参考与预算控制 | 088 TCM2424更新覆盖155134轨一遍，是基础端点，不是80epochs复现；输入投影冻结。090 adapter A/B 574.279/580.809m均输同预算C564.834m，关闭adapter；C为fresh Adam再训练，不是原优化器连续训练。091径向/球面位置目标734.625/740.837m仍输固定.95初值。092未执行被093/094取代；095纠正特征误差宏平均解释。 |
| 096—097完整系统／预算 | 096A覆盖3.3557%、位置MAE58587.02m，290/300场只一组；同mask TFM59040.15m对anchor583.53m，关联之外的物理输出也失效。A/B共同14916点learned59898.42 vs hold59916.82m，不能据18.40m差称TCM有用。097A为第二seed预算控制583.975593m，不与不同人口混排。 |
| 098关系配方 | seed23预算/generic/relation/bias-off MAE=564.834006/575.830125/572.775328/572.775244m；seed21=583.975593/607.817721/568.684246/568.683944m。两个原晋级规则均未过，终态关bias近零不证明训练期间无作用；当前配方关闭，不扩大为关系模型类别无效。 |
| 099 DeepAF | P/V核心及坐标初始化修复保留，尚无真实MTAD成绩，与TMTAFF四属性改版有差别。自建模拟训练不能认证同行资格；旧三臂计划HOLD，旧命令不可领取。10/28初判、11/1去留是原项目节点，不因更换候选重置。 |
| 100—101 | 100三臂4000更新，预算/固定汇总/attention汇总MAE564.968/566.092/567.187m，未胜090C564.834m，不晋级。101仅旧预测补算，揭示机会区收益与支撑内尾部异质性；原因未定，不把ME小当胜论文。[保留的完整分析](../reports/2026-10-26-experiment-review.md) |
| 10/27未筛候选 | 观测约束融合在来源—矩阵交接未完成前被提前主推，已撤回优先地位；本次移出工作区，旧草稿在903f679。来源阅读继续保留，尚无完成的正式矩阵／新Idea；未启动该方案。 |
| 107／113固定新单位端点 | 107训练TCM2424/TFM1000完成，原评价旧schema预检FAILED保留；113另run零训练修复评价全V300退出0。共同422317位置点TFM MAE42787.223m、cap128 Mean1127.413m、090C439.694m；全部300场/38组件位置不利，速度有收益。当前适配不晋级强恢复baseline，非完整作者复现，不重训/加步/seed/TAM。[Record/Squeeze](../reports/2026-10-29-fixed-endpoint-review.md) |
| 114／115静态关系与位置控制 | 114同分数全V decoded宏F1 .765657→.851820，raw不变；115固定V32真正重算普通Mean，共同MAE变差而RMSE/覆盖改善，重复92→217。内部控制有价值但不能称可信同行胜负或学习融合/条件B增量；固定终点已到，不自动扩量。[关系](../reports/2026-10-29-static-partition-review.md)／[位置](../reports/2026-10-29-static-position-review.md) |

跨阶段比较必须说明人口、覆盖、训练量与选模差异；失败结论不因删除副本撤销。Level2的9001/reference复用限制观测降噪解释；21分钟窗未采用。当前Level1按场景划分但共享A身份，不能称未见身份泛化，详见[数据约定](../data_contract.md)。

## 本轮精简（2026-10-27）

当前状态只在研究状态，领取名单只在HandOver；README/资料库记忆不再复制数字或队列。已删除结案096—101旧READY、重复逐臂回传、旧夜间队列及专用测试、无引用OLS脚本/配置、100专属比较CLI/测试。未筛观测约束草案和099旧长合同移出日常入口，均可从下列Git定位恢复。论文规范全文与原始阅读保留；没有新训练或研究结论。

本轮受管理工作文件368→302、Markdown43→23，文件内容约20.1→9.7 MB（不含Git历史、环境及数据）。README/AGENTS/研究状态/交接四入口合计文本约减少62%；505项测试导入发现无错误，22项现有指标/缓存检查通过；生产模型、数据约定及102合同未改。

**删除前版本：[`903f679`](https://github.com/HKillua/association-code/tree/903f679)**。096等原件、100逐臂配置/状态/曝光、旧科学合同均可按原路径读取。本地被中断时追加的重复字段映射未发布，撤去重复副本，不删除既有实证分析。

## 原件与恢复

- **第二轮精简前：[`9eefd2c`](https://github.com/HKillua/association-code/tree/9eefd2c)**。含本轮移除的已审052—098回传、实验合同及旧变体源码。
- **第一轮精简前：[`1a25183`](https://github.com/HKillua/association-code/tree/1a25183)**。含更早完整工作副本；099导入/合成排错在[`e7e4a11`](https://github.com/HKillua/association-code/tree/e7e4a11)。
- 更早Level2/052—061在[`5776cd3`](https://github.com/HKillua/association-code/tree/5776cd3)，001—018在[`612afea`](https://github.com/HKillua/association-code/tree/612afea)。提交号仅历史定位，不作运行准入。

```sh
git show 9eefd2c:HandOver/S2L-20261025-098D.md
git show 1a25183:HandOver/S2L-20261024-091-summary.md
```

按需读单文件，不另建archive或恢复整套退役流程。[当前结果索引](../../HandOver/results/README.md)登记保留原件。完整服务器输入/权重/预测仍在`/home/wangguangjie/djs/association-code/runs/`，Git摘要不能改名后喂给模型。用户Downloads原ZIP、本地数据/权重/runs和活动服务均未删除；清理也不改写Git历史。

第二轮整理（2026-10-25）：受管理文件557→290，工作副本约51.9→5.0 MiB；Markdown 50→25、Python 249→200、测试文件98→76、配置文件64→49、工具Python 52→33。共享函数与fixture原样迁移，当前096/097/099及物理输入/四权重共17项CPU检查通过，全部测试导入发现无断链。当前096执行/队列/收尾比较代码未改；不把这些软件检查当作真实实验成绩。数据、权重、runs、依赖环境和Git历史保留。
