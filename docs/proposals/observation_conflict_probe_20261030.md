# 138候选：T侧观测证据冲突与实际损害的联合诊断

2026-10-30；计划版本1.0，**IMPLEMENTATION_SPEC：已由root及独立来源代理review，允许实现，未READY、未运行**。本文只规定一个有界只读诊断，不另造科研流程，也不生成新融合方法。唯一代码仓、数据权限、终态记录仍遵循仓内AGENTS、科研流程与skills。

## 1. Idea交接及是否值得新立

- package_type：IDEA_PACKAGE；package_version：research-pipeline-contract v1.0。
- idea_id / package_id：IDEA-OBSERVATION-CONFLICT-PROBE-20261030；候选实验138；status：diagnostic。
- source_matrix_id：MC-OBSERVATION-DRIVEN-20261030-v1.3 / 全局Mode C1.56；selected_C_id：OD-C1 / C54，当前有限支持、评分7，未晋级为强贡献方法。
- source_P_ids：[OD-P1, OD-P3]；source_F_ids：[OD-F1, OD-F2, OD-F6, OD-F10/F237]；source_M_ids：[OD-M1, OD-M2, OD-M5, OD-M7/M106(AI/UN)]；source_E_ids：[OD-E1, OD-E2, OD-E6, OD-E10/E135]。
- 直接上游：[局部Mode C v1.3](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md#observation-quality-136-feedback)，[136 Record→Squeeze](../reports/2026-10-30-observation-quality-review.md)，[134 Record→Squeeze](../reports/2026-10-30-mixture-opportunity-review.md)。先消费上述回流，不把新结果直接跳接为方法。
- 模式：Pattern 2，现象驱动。领域为统一经纬度/速度/航向下的多源轨迹融合；新增模块不输入固定源角色。
- evidence_status：partial。已有执行/主要数组证据已核；同query多证据冲突、少数角色反转中的信息价值、条件混合空间与实际损害的联合分布仍未知。

**建议立为一次窄范围缓存诊断，而不是再测一个模型。** 134已有QOOF误选、单变量分层及整体网格混合空间；136已有各proxy独立相关、完整混合效果和损害集中度。但两者尚未回答：**同一query上几项证据意见相反时，谁错、错多大、有没有可信的互补证据，以及这些联系是否仅来自原生成角色。** 138只填这个联合条件缺口，不重做134网格、136代理计算、源单变量相关扫描或全V比较。如果实现预审发现这份联合表已存在，应复用并取消138，不为了用完额度建立重复实验。

## 2. 两个最小问题、假设与可反驳预测

**Q1：LOO与nearest严格冲突时，OLS与QOOF的支持关系能否区分可避免损害；proxy的源排序与已有连续混合输出是否表现不同？**

这里有两个竞争解释：一是残差仍含部分排序信息，但固定inverse读出没有兑现；二是残差本身的排序常错，改变权重尖锐程度也未必有帮助。若LOO的源选择代价在冲突层低于nearest，而其已保存连续融合并未同样好，才获得“信息与读出需区分”的条件证据；若LOO排序和混合都差，不据此直接放大LOO贡献。QOOF为已存在的监督强证据控制；OLS为另一自残差解释，不重拟合。

**Q2：上述联系在匿名role_0严格最佳、role_1严格最佳与参考平局中是否改变；各层已存在的混合空间足不足以支持继续研究贡献？**

若共同偏好在角色反转层反复错误，或各证据冲突只对应固定角色，需优先调查可观测证据不足/角色混杂，不能把匿名role变回部署优先级。若冲突层、逐组件/去最大组件与角色反转条件仍呈一致的误差代价差，并且实际输出距已有可行混合有空间，才支持另立合法贡献方案。这里只作描述，**没有为138建立新LOCO或新规则留组件泛化验证，只有原133 Q头的既有LOCO**。角色与最优误差是reference诊断变量，无法直接作在线门控。

这两问只建立条件联系，不能单次证明偏移、时距、相关误差或分组污染为根因。还需区分可观测分类与reference知情分类；后者不构成预测能力。

## 3. 输入allowlist、来源对应与完整分母

仅原T_core前32场、12组件。读取已完成136/133/134/127的`run.json`与必要元数据核血缘；**不打开V数组或含V指标的完整summary**，不读T_test、ZIP、CSV、原生事件归档、137run或新模型。136/133全summary内有V内容，本轮不以“顺便读取”方式消费；T公共结果有独立文件，优先使用它。

| 来源 | 本轮允许读取的具体内容 | 用途 |
|---|---|---|
| 136 | `private/policy-T_core-{sid}.npz`；`private/diagnostic-T_core-{sid}.npz`；`private/states-T_core-{sid}.npz`；`training-scene-summary.json.gz` | 已保存source proxy/known/weights、完整六臂真实预测、合法source标签与人口 |
| 133 | `private/training-{sid}.npz`、`private/oof-{sid}.npz`中QUADRATIC分数、`T-fold-index.json` | T训练pack/标签对应与原12 LOCO的QOOF；不加载fullT预测头，不读模型参数或再拟合 |
| 134 | `private/scene-{sid}.npz`、`private/roles.npz`、T专用`scene-summary.json.gz`和`summary.json` | 既有endpoint误差、best_single/grid误差、gain/bound、真实Qhard、源行/角色/纯混组；summary只含T，可读作绑定 |
| 127 | `private/native-motion/T_core/scene-{sid}.npz`、分离的`native-evaluation/T_core/scene-{sid}.npz`、native-cache元数据 | 固定node/query/source/support/nearest及合法参考/身份绑定；不重算状态 |
| manifest | 明确原科学绝对路径，T名单及component映射 | 12组件和旧血缘核对，避免再次把不同checkout相对路径误作同一路径 |

所有源行通过**场景ID＋节点ID/顺序＋membership＋group_offsets＋node_to_source_row＋group内source等价类＋query索引/秒值**绑定，不能假定两个run的第0行就是同一个source，更不能把source字符串排序当质量。对齐时核133与136每个源行对应完全相同的节点成员及组；134的`source_rows[M,2]`必须指向这些确切源行。若行序不一致，只允许由完整成员集合构成唯一双射并保存映射；不是凭两行误差大小配对。任何重复/缺失/多义映射失败保全，不删不匹配query继续。

主诊断人口`D2`沿134的**原22627个**合法多源query：组first anchor的目标已知、同目标首代表组、该query位置reference有效、当前两源available。保存`scene/group/query/source_rows[2]`唯一键，核与133/136 labeled mask逐项一致、源标签总数45254、每query标签权总和1。几何诊断权为每query1，不把两source标签误作两倍样本；`D2`不与后文主分格D混用。

完整人口另有两个不同分母，必须一起保留：

- T完整首代表可评位置点37635，用136六臂完整artifact复核，无新增预测；其中0支持1487、1支持13521、多源22627。0/1层不进入源排序/混合空间，保留null与数量，不能记作正确选源。
- 所有911预测组×64query共58304组query，包含重复/未知/无reference等；与860目标、825有anchor、35未覆盖、86重复组、38224有效reference点/589缺预测完整绑定。source支持总量沿136统计0/1/多源，**不能和37635或22627混用**。

逐场重建并核134 scope：重复anchor组/query、未知anchor、首次anchor参考无效、首次anchor有效参考的0/1/2+source、目标漏anchor与其有效reference点。节点组的纯/混身份仅用作评价分层。若实际出现K>2，记录不符合本K2诊断合同并失败；不临时扩展oracle或冒称已测真实K≥3。

## 4. 三类量分开，禁止生成新预测

### 4.1 来源偏好与endpoint选择诊断

在D2每点，两源端点误差`e_s`用134保存的原节点恢复endpoint_errors_m；先与133/136源标签`errors_m`数值绑定（rtol1e−9、atol1e−5m），主几何口径始终用endpoint误差，不从source_states再混经度。端点列顺序沿`source_rows[:,0/1]`，**列0对应alpha=1、列1对应alpha=0**，不是递增alpha顺序。134输入的source moment/endpoint与原native绑定已有审计，本次不重网格。

四个偏好变量固定为：

- `P_L`：136保存的source LOO proxy较小者；任一available源known=false则该query为unknown，不把未知填0当最可信。
- `P_O`：136保存的source OLS proxy较小者；同样保留known。
- `P_N`：每源当前支持成员的最小nearest_seconds较小者；绑定136 SOURCE_NEAREST保存source_weights。这里是源内min时距，不是133特征中的source内平均时距。
- `P_Q`：133该组件留出QUADRATIC OOF得分较小者；核134保存`oof_QUADRATIC_scores`、fold heldout_component与training_components隔离。OOF只隔离原133 head，上游teacher不因138变成严格全链OOF。

代理直接读136的`ols__source_proxy_m2`/`loo__source_proxy_m2`；不能将133的mean(RMSE)特征反变换后平方来替代mean(RMSE²)。nearest同样不能从133的mean nearest特征推测source-min。136保存的source权重只用于原输出/偏好绑定，不重新计算新的贡献配方。

严格较小/精确平局按原数值定义；精确平局赋两源各1/2诊断质量，strict winner必须唯一。unknown独立标志，不伪造成平局。不新增epsilon偏好阈值、不调温度；跨文件几何一致性容差不是改变偏好或参考平局的阈值。另记近参考平局导致label与endpoint排序不一致的数值边界计数，不悄悄改主定义。

若偏好质量为`p_s`，定义`selection_cost=Σ_s p_s e_s`、`selection_regret=selection_cost−min_s e_s≥0`，以及`credit=Σ_s p_s 1[e_s=min e]`。这里是**假想选择一个端点的期望代价/参考credit**，不恢复新位置、不称hard融合预测成绩；平局时尤其不能与几何混合误差等同。未知偏好这些量为null，分母单列。

本轮不计算新LOO-hard/OLS-hard轨迹，不产出新policy供部署，不给T新预测臂起“方法名”。**0新预测精确指0新物理位置输出、0模型前向；允许仅读已存分数/代理派生偏好及期望endpoint选择代价。** 混组中的endpoint误差只相对first-anchor参考目标，属于当前组决策风险；它不等于源自身物理健康度或绝对传感器准确度。

### 4.2 已保存的真实输出风险

136六臂真实位置由完整T artifact读取，沿原first代表/valid人口独立计算径向误差；重点保留equal、LOO_F1/F100、OLS_F1/F100、SOURCE_NEAREST。两floor需先确认每点权重与输出一致，虽一致仍保留两个标签/零差对照，不据136事后挑一个winner。

134 `QUADRATIC_hard_error_m`/hard_position是**已执行且已保存**的QOOF真实hard输出，仅覆盖D2，可读取绑定，不补造其0/1/无ref完整T成绩。严格winner时将实际hard端点误差与selection_cost对照；平局下实际几何混合和期望端点代价分别展示，两者差不是实现错误的充分条件。

每个既有真实输出在D2报告：误差mean/RMSE/P95/P99；相对equal的signed差、正害/负收益、数量及总和；相对best_single的signed excess及正部。`signed excess`允许为负（有益混合），不能一律叫非负regret。全37635人口保留136旧六臂核对值，条件D2结果不冒充全人口成绩。

### 4.3 复用已有混合空间

仅使用134每个D2点已有的`best_single_error_m`、`grid_error_m`、`gain_m`、`gap_bound_m`、`min_resultant`等；核同query、网格4097和原端点。不再跑4097网格，不重新选alpha，不建立新oracle。

报告各固定层的`gain=best_single−grid`均值/中位/P95、>10m点数及增益总和；并报告已有真实输出与grid误差的**signed gap**。连续已存权重可能略优于有限网格，因此signed gap可负，不能clip成“非法”或声称4097即连续最优。有限gap_bound的计数与分位和退化无界计数继续保留，不能以数值界掩盖小收益；不据alpha给新模型生成训练目标。

## 5. 固定联合层与主要估计量

不扫全部相关性、不重跑134既有margin/RMS/native-count单变量图。本轮按可观察偏好先建立以下互斥穷尽主分区；reference角色/组纯度在后面的诊断层另外使用。

| 主格 | 固定条件 | 所回答的局部问题 |
|---|---|---|
| A | L、N、Q都strict且L=N=Q | 共识是否仍会高损害？ |
| B | L、N、Q都strict，L=N≠Q | 观测一致但Q异议是否有信息？ |
| C | L、N、Q都strict，L≠N且Q=L | LOO与nearest冲突、Q支持LOO时谁更好？ |
| D | L、N、Q都strict，L≠N且Q=N | 同样冲突、Q支持nearest时谁更好？ |
| E | L/N/Q全known但至少一个精确平局 | 不强行选边，量化平局与实际混合差异 |
| U | L为unknown | 保留缺proxy证据；D2的N按支持定义必须known，N或Q非有限属于输入错误，不当unknown掩盖 |

OLS作为独立的第二种代理，**只在C/D两个严格冲突格**进一步分为O支持L、O支持N、O平局、O未知四种；固定8个子格，空格保留。其余格报告O与各偏好的配对一致/冲突数量，不展开大立方交叉。

补充一张四偏好全部6对的pair表：strict相同/strict相反/至少一tie/任一unknown数量，各自源选择regret/credit及已有输出误差摘要；用于检查OLS并非遗漏与输入对应，不作为六个新确认性假设。

匿名role分层固定为**role_0严格reference-best / role_1严格reference-best / 两端点reference误差精确平局**，沿134保存的role mapping，不重新按本轮成绩排序命名。名称“反转层”仅指相对134已报道role_0更常最佳的事后描述，不是角色1理应差。三层分别交叉主格A–U（18格）；纯/混目标组分别交叉A–U（12格），**不把role×purity×OLS继续全交叉**，避免稀疏筛选。公开只匿名编号；source名字映射留private，role从不参与偏好、分数、权重或部署函数。

主要量均在事前固定的C∪D严格冲突人口上，保留其占D2的比例与组件覆盖：

1. 既存真实输出`mean(e_LOO_F1−e_SOURCE_NEAREST | C∪D)`，再分别C、D展示。它比较完整实际风险，不把硬选择credit代替混合成绩。
2. 三个并列诊断差`mean(R_L−R_N)`、`mean(R_O−R_N)`、`mean(R_Q−R_N)`；O未知时只在O与N皆known的明确子人口计算，另列覆盖，不对未知作0。C、D和OLS八子格保留同样定义，不从三者挑winner作单一检验。

上述量与每格正损害计数/总量、credit、既有grid增益一起回答“信息不足还是读出尚未利用”的最小问题。Q2只为这些量增加role/purity边界，不发布角色门控。全部A–U也报告已存nearest/LOO/OLS相对equal的>1000m损害数量，以及`0<损害≤1000m`数量/总害，用来判断损害广泛还是仅少数大点；1000m沿134/136已有描述阈值，不是新的政策参数。top1%集中度不再新扫，引用136已存T统计作总体背景即可。

按构造C中`R_Q=R_L`、D中`R_Q=R_N`，OLS支持同一端点时也有相应恒等关系；这些不是新发现或彼此独立证据。真正要检验的是各条件中reference风险差的方向/量级是否分离、真实连续输出是否兑现，以及关系在组件/角色边界中是否保留。

## 6. 汇总、不确定性与结果路由

完整32场、12组件、去掉原按场数定义的最大组件汇总均保留，不能只报有信号的格。点加权主量与场均值分开；各层报告query数、可用偏好数、独立component数、纯/混与角色计数，零分母输出null而不是0。

仅对§5两项主要估计量及其预定并列差做12组件seed20、2000次描述性bootstrap。条件格为空的抽样不补抽、如实计数并排除于区间数值计算；若该量实际只有一个组件，区间为null并说明不足。报告有效抽样数；不把偏好×角色几十个格都附未经校正的显著性检验。T固定且上游teacher使用全T、134/136已分析过T，故这不是独立确认性检验。

| 观察结果 | 下一条合理路线；并非本轮执行 |
|---|---|
| C/D中某观测偏好具有较低端点选择损害，但对应已存连续混合明显未兑现；角色反转/去最大仍有信号 | 回Mode C区分排序与读出，再设计单因素贡献读出/安全选择对照；不先选硬或软网络 |
| proxy在冲突格的源排序、连续融合均差，QOOF仍好 | 不为残差盲目锐化/调floor；检查可见动态/噪声建模是否提供不同信息，监督风险方案保留为可接受参照 |
| 各规则共同出错主要出现在role_1最佳，或有益关系只在多数role_0中 | 先记录角色混杂与样本不足；需要额外可辨证据/更独立数据条件，不返回固定role_0优先 |
| 各候选证据都无法在高害条件区分，而混合空间仍明显 | 优先补信息/辨识条件，不能靠更复杂门控承诺收益；允许第三项到此停止 |
| 条件grid空间小，主要差在选择损害 | 进一步研究少犯高代价选择错误；不能把不存在的混合收益包装成创新 |
| 结果几乎与134/136已有结论重复、联合层不能改变上述路由 | 明确“新增信息有限”，不追加范围/查询、继续找图；一次终点后记录收尾 |

三层质量：不主张新算法或绝对原创，新增价值是可追溯的联合失败条件；已有相关方法不否定适配空间。关联/分组、状态、输出与reference维持原条件，使对比只解释现有贡献与证据的联系；偏好来自不同信息权限（Q监督OOF，proxy非reference拟合）必须明确标识。

## 7. 可执行规格、资源、独立验收

独立runner CLI参数合同（尚不存在、不构成启动命令）：`--quality-run --source-run --mixture-run --native-run --manifest --output --cpu-threads 4 --time-policy advisory`。所有source路径明确固定：

| 参数 | 固定绝对路径 |
|---|---|
| `--quality-run` | `/home/wangguangjie/djs/association-code/runs/LEVEL1-OBSERVATION-QUALITY-20261030-136-attempt-1` |
| `--source-run` | `/home/wangguangjie/djs/association-code/runs/LEVEL1-SOURCE-RISK-20261030-133-attempt-1` |
| `--mixture-run` | `/home/wangguangjie/djs/association-code/runs/LEVEL1-MIXTURE-OPPORTUNITY-20261030-134-attempt-1` |
| `--native-run` | `/home/wangguangjie/djs/association-code/runs/LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1` |
| `--manifest` | `/home/wangguangjie/djs/association-code-run136/configs/data/mtad-level1-scene3000.json` |

`--output`必须是单独新目录，建议名`LEVEL1-OBSERVATION-CONFLICT-PROBE-20261030-138-attempt-1`，绝对位置、checkout、launcher、CPU核和代码历史待实现核验后由L2S发布。一次attempt；不允许在原run下写文件，不覆盖、不失败重试/换阈值/减样本。单纯改CLI参数名/输出布局的兼容修订需同步本文，不能改变科学估计量。

CPU≤4、RSS6GiB、AS12GiB，1800s advisory；0GPU/0神经更新/0新fit/0模型前向/0新物理位置输出/0网格搜索。工作量O(原T节点×64＋D2×固定偏好/格数)，逐场流式，只在内存累计小标量/索引数组；1800s不是截断样本门槛，超建议保留告警并只完成固定T32。成本记录输入文件/数组读取数、场数、耗时/峰值/实际退出，失败当前场部分工作未知则明说。

拟议输出：

- `run.json`：全部输入绝对路径/代码历史/真实命令、固定T名单与组件、阶段/完成数量、0新fit等声明、预算/成本/终态和失败现场。
- `alignment.json`与每场私有`aligned-{sid}.npz`：唯一join键、原源行双射、偏好winner/tie/known与诊断质量、endpoint误差、既存各真实输出误差、best/grid/gain/bound、role/purity、主格/子格mask；身份/坐标/原source名仍private。
- 公共`scene-summary.json.gz`完整32行、`summary.json`全量/12组件/去最大、主格与有限交叉、全部6对偏好表、bootstrap、完整排除/缺失人口和字段单位。所有零格、null、坏例保留，无新模型checkpoint/部署权重文件。
- `independent-audit.json`：独立核验完成/失败、覆盖和未审事项；只读独立调用另记成本，不冒充第二次科学尝试。

最小验收（实现前冻结）：

1. 用手算2源例验证strict/tie/unknown、互斥穷尽A–U、OLS8子格、role翻转映射与全部6偏好对；改变source名字/交换源行但保持成员对应时统计不变，匿名role只相应重标。
2. 平局例证明`Σp_s e_s`可以不同于实际几何融合误差；优于最佳单源的已存混合应有负signed excess，不能clip成零。unknown要保留，不提高credit。
3. 故意错query秒值、调换一个节点/source成员、重复join或让OOF训练组件含holdout时失败；不能仅凭长度22627认为已经对齐。136 artifact完整valid/目标/duplicate人口需与127、134 scope绑定。
4. 调用路径allowlist测试：生产流程不读取V/Test/native-events/models；小fixture完成全部32名单时保留V文件不可访问的条件，不使用运行后日志自己证明无泄漏。
5. 独立auditor仅NumPy/标准库，从原T保存文件重建join、四偏好、endpoint期望代价、真实artifact径向误差、固定分区/role条件、主要汇总与bootstrap；不导入生产诊断函数。核原136六臂T人口/主要量及134 oracle scalar精确来源，不再计算4097网格或CV/源代理；科学原先已核字段复用边界明写。
6. 若134或136私有必需键缺失，先保全并报告；本合同不授权补跑原方法、重造source_states/alpha、读取V或更换对照。所有源绑定与主要量通过后，才允许用联合层解释。

known_limits_and_missing_evidence：真实K2、T重复开发/上游监督、少数角色反转样本和相关query、物理reference不明、噪声/偏差可辨性未知；本次只用旧误差联合诊断，未提供任何新预测的泛化性能。

stopping_and_interpretation_rules：完成原T32、固定表/分母/独立核验即结束；失败保全不自动续跑；没有信号也完整Record→Squeeze→Mode C，不直接扩V或定模型。next_skill：record-experiment。
