# DeepAF 方法独立审计：核心实现存在，同行资格待验证

**后续证据更新：** 导入后的单次1024更新核验已完成，同次256→1024的MAE529.789→115.963m，原大幅递推漂移明显缓解；本文保留原256阶段审计，最新判断见[接收复审](2026-10-25-099-import-review.md)。真实Level1资格仍待核，099保持HOLD。

## 状态与结论

**状态：METHOD_QUALIFICATION_PENDING。** 审计起点为本地提交`fb9ef8108b070348047d1622cb1ce537f01a4203`。本报告响应用户“先确保我们目前实现了DEEPAF方法”的要求，重新核查方法身份、物理表示与验证证据，不把上一轮作者自审或23项软件检查直接视为方法验收。

目前能够确认：代码实现了DeepAF的P/V双网络核心，并将其接入fixed056离线位置融合任务。本次已修复一处初始化坐标缺陷，完成相关软件回归及两臂各256更新的独立合成诊断；两臂能学习，但位置误差仍明显累积。尚不能确认原论文完整在线DeepAF系统已复现，也不能把当前适配认证为合格的同行性能对照。**099大批正式比较保持HOLD_METHOD_QUALIFICATION；软件就绪不等于同行资格通过。** 具体任务状态由当前HandOver登记；本次没有运行Level1或服务器GPU任务。

这不要求A位置融合研究先重造原论文全部航迹管理模块。物理表示和核心行为通过必要验证后，可使用“DeepAF P/V在fixed056位置融合任务上的适配”作为同行对照；方法名称、适配范围及训练充分性仍须如实披露。

## 一、证据来源与已确认核心

原文：Cui等，*DeepAF: Transformer-Based Deep Data Association and Track Filtering Network for Multi-Target Tracking in Clutter*，Aerospace 2025，12(3)，194，[DOI](https://doi.org/10.3390/aerospace12030194)，[作者公开PDF](https://mdpi-res.com/d_attachment/aerospace/aerospace-12-00194/article_deploy/aerospace-12-00194.pdf)。依据为Eq.(9)–(14)、§3.1–3.3（PDF第6–11页）和§4.1（第12页）。

审阅实现：[模型](../../association/models/deepaf_peer.py)、[训练器](../../association/training/deepaf_peer.py)、[评价](../../association/evaluation/deepaf_peer.py)、[缓存比较](../../tools/analysis/compare_deepaf_peer.py)、[模型检查](../../tests/test_deepaf_peer_model.py)、[数据合同](../data_contract.md)及[099合同](../specs/peer-feasibility-sprint-099.md)。

| 原文关键要求 | 当前实现及可核查入口 | 审计结论 |
|---|---|---|
| 独立P/V两网，各6 encoder＋6 decoder、d32、4个head、FF64 | `DeepAFPeer`、`_DeepAFNetwork`、`MODEL_CONFIG` | 已实现，265732参数；不是普通Transformer换名 |
| 当前无序候选差分集合与过去估计历史交互 | encoder无位置编码；decoder对历史使用sinusoidal编码 | 核心形式保留 |
| self-attention采用scaled dot product | `_Attention`普通分支 | 与Eq.(11)一致 |
| cross-attention采用Eq.(12) | `mahalanobis_scores`计算`2qᵀk−qᵀq−kᵀk` | 公式保留；原文称negative Mahalanobis，但该式没有显式协方差逆，不应擅加 |
| 两网共享本时刻候选，虚拟候选取上一时刻V | `forward_batch`在更新P/V之前构造同一`candidates` | 顺序正确，没有把当前V提前用于当前P |
| P/V反馈来自自身过去估计 | `previous_p/previous_v`及两条历史队列 | 已实现；forward不接受GT、reference或identity |
| P/V平方误差监督与真实参数更新 | `deepaf_supervision`、`deepaf_objectives`及trainer replay | 双网有真实梯度、更新和检查点；具体目标与训练方式仍有适配，见下文 |

## 二、必须修正或核验的物理表示问题

### 初始OLS速度与球面局部位置原先不在同一个坐标表示中

审计起点的`deepaf_inputs`直接采用`motion['ols_velocity_mps']`作为初始速度，并据此构造虚拟前位置、P/V初始历史及carry token。可是`association/data/motion_evidence.py::build_motion_evidence`明确规定该速度采用经纬度线性平面：东西方向使用`cos(mean visible latitude)`缩放；DeepAF位置、候选差分与标签则使用以`base[0]`为原点的球面log图。

两者同为m/s，但分量定义不相同。恒纬度且原点在轨迹起点时可以近似一致，不能据任意全段回归差异就断定瞬时速度必错；变纬度或原点偏移时，则需要实际变换或误差证据，不能仅凭单位相同认定物理闭合。

独立runtime审计通过真实`build_motion_evidence`路径构造了一个**非Level1实测**反例：15小时内纬度70→75度、经度0→25度、Q64。原初始东西速度约15.4801m/s，球面log图在原点的对应导数约17.6069m/s，低约12.08%，相应虚拟前位置差约1823m。这证明了实现中的坐标近似会产生可观偏差，不代表已量化Level1全部数据中的影响。

**已完成修正：** motion evidence暴露其既有native OLS经纬角速度`ols_slope_degrees_per_second`。DeepAF把该角速度轨迹平移到实际初始化位置`base[0]`，用原点切向微分`R*rad([dlat,dlon])*[1,cos(base0_lat)]`得到速度；不对Q64重采样轨迹重新拟合，也不声称这是原anchor曲线在任意点的log Jacobian。缺失、形状错误或非有限角速度显式失败。核心P/V网络和训练目标不随此修正改变。

routing已升为`fixed056-pv-closed-loop-tbptt1-v2`，模型、trainer和四份config一致；旧v1 envelope或configuration检查点不能被v2 loader静默解释。实际混合`base0`不等于anchor初始位置、原始中间观测不落在query上等情况均有回归。

验证由独立runtime审计执行，共覆盖20个不同用例（`tests.test_deepaf_peer_model`、`tests.data.test_motion_evidence`、`tests.test_deepaf_peer_pipeline`）。首轮18项通过，2项新增测试有fixture／路径错误；修正测试后仅重跑这2项，全部通过。测试同时覆盖真实builder坐标导数、native OLS来源、输入拒绝、两网更新、8步双周期训练和三个评价端点。首轮28.856秒，定向重跑11.948秒。没有CUDA或真实Level1验证。

## 三、不能因披露了适配就自动判定有效

| 项目 | 原文或未明确部分 | 当前适配 | 对资格判断的影响 |
|---|---|---|---|
| 初始历史 | §3.3由实际初始航迹段计算P/V与历史；§3.2.1提及通常3–6个量测起始 | 只有一条有效OLS历史，其余mask；虚构`p_-1=-v_OLS×first_dt`，q0即更新 | 需要检查短历史、长dt和初始漂移；它不是已核实的作者启动实现 |
| 缺测与空候选 | 原流程允许空关联门，连续3–4次空门可终止 | `support[0]=True`且validator要求anchor每query有效 | 当前生产adapter不能表达native carry-only／空gate，只能表达有可见插值／外推anchor的固定组任务 |
| P标签 | Eq.(13)使用position difference truth或desired value，未完全说明构造方式 | `(GT_k−stopgrad(predicted_previous_P))/dt` | 是条件化纠偏目标；可由位置更新式解释，但尚无作者实现证据证明该训练解释唯一正确，不能直接称“原文标签已复现” |
| V标签 | Eq.(10)、§3.3为速度状态，原模拟可提供真实速度 | GT相邻有效位置差分／dt，首点用前差分 | CA／CT下通常为区间平均速度而非该时刻速度；长dt时差异可能变大 |
| 时间与位置域 | 原实验为1s周期、30s场景、二维Cartesian位置 | 全场Q64、真实变长dt、球面局部米制 | 需要验证尺度与递推稳定性，不能用Q数量相近替代时间协议一致 |
| 门控与航迹生命周期 | 门控选候选、起始、连续缺测终止 | fixed056组、anchor外推、helper support | 属于研究任务改变；不构成完整在线MTT复现 |
| 输出属性 | 原系统输出网络P和V | 对外speed／heading仍为普通Mean，V仅参与内部递推 | 同行结论必须限位置融合 |
| 优化方式 | 原文明确Adam和P/V MSE，但lr、batch等公开细节不足 | 两个预定lr、TBPTT1、联合目标／clip、dropout0、scale10 | 是本项目已冻结训练合同，不是已核实的作者训练配置；同曝光不保证充分训练 |

尤其需要区分“P目标存在另一种原文解释”和“已经证实当前P目标写错”。现有原文不足以唯一辨认训练标签与历史数据构造方式；不能把某个合成试验表现较好的解释反过来冒称作者实际实现。V的区间平均代理也需要单独标注，不能因为保留了MSE就忽略目标语义变化。

## 四、当前评价闭合与信息边界

评价和缓存比较已有实质性保护：所有原组、重复组先预测后做reference对应；输入、组、truth、mask、普通控制和Mean末两维逐值核对；物理误差从缓存重新计算；球面越过πR的绕行次数独立重算。最终存在绕行时，比较器标为`UNRESOLVED_PEER_GEOMETRY`，不以折回地球后的偶然低误差认证收敛。`COMPLETED`也不会自动变成peer合格或我方胜出。

但共同原始观测和共同人口不等于模型整合信息的方式完全相同。我方Transformer在整个Q64上进行非因果注意力，并使用可见轨迹统计；当前DeepAF按query递推，而候选插值与初始OLS本身又可能使用query后的可见观测。因此本轮是**离线预处理加顺序滤波的适配**，既不能称纯online，也不能把它与我方的所有输入信息使用条件写成完全等价。

该差异不自动否定同任务位置融合比较，但需保留在方法解释和成本／性能结论中。098的关系增量仍必须由其既有内部对照验证，不能把全部同行性能差归给关系先验。

## 五、23项软件检查和CV32步探针的证据强度

上一轮统一23项检查证明了网络、训练、检查点、完整评价人口和缓存比较能贯通；具体记录见[软件验收](2026-10-25-099-peer-feasibility-review.md)。本次审计没有无理由地重复运行它们。

原`test_closed_loop_learns_two_simple_cv_groups_in_32_updates`只包含两条固定CV轨迹、约2–3m/s速度、两个固定偏差的位置流，没有观测噪声、漏检或clutter，32步后仍评价同样两条训练轨迹；原OLS数值由构造速度赋值。本次已把其OLS改为真实`build_motion_evidence`路径并通过回归，但仍是同样的简单训练集拟合检查。大幅下降只证明这套闭环目标能拟合一个简单任务，不能替代独立行为证据。

原文§4.1则包含1s采样、50m位置噪声、检测率0.9、Poisson(8) clutter、100–250m/s初速度及CV／CA／CT。现有检查没有证明这些条件下的关联／滤波行为、独立场景泛化、生产训练充分性或同行竞争力。不能以更多同类单元测试替代缺失的行为证据。

## 六、目前最短的资格验证计划及其限度

在坐标修正之后，使用一个有上限的**原文风格已起始单航迹核心验证**，而非直接投入099三臂完整比较：

- 合成训练192条、独立held-out 96条，包含CV／CA／CT、50m噪声、Pd=.9及Poisson(8) clutter。held-out是合成场景，不是MTAD封存T_test。
- 六个带噪可见量测构成已起始航迹；明确这是对起始已成功的条件验证，不宣称实现完整track initiation。
- 后续允许无真实候选，仅carry token仍可递推。GT只用于生成模拟物理过程和训练／评价标签，不进入模型历史、不用于输入侧识别真实候选。
- 复用同一P/V网络核心。分别检查两种目标解释：P真差分／V时刻速度，与P纠偏增量／V位置差分。每臂固定256次更新，CPU单线程；预算和目标规则在结果之前固定。
- 报告held-out位置与速度误差、逐CV／CA／CT结果、随时间漂移及缺测／候选污染行为，并与可见全候选质心和初始OLS-CV外推比较量级。质心无候选时回退OLS-CV；没有用GT挑出正确量测作为控制。不要只报训练loss或相对随机模型改善。

**该验证足以补充关键核心行为证据，但不能单独认证作者整系统或其原论文性能。** 256步的阴性结果可能反映目标解释、优化或初始化尚未解决，不应直接否定DeepAF；阳性结果也不能证明8000步MTAD适配充分训练。两种目标解释的性能差只能帮助定位选择的后果，不能确定作者实际采用了哪一种。

模拟中原文没有充分公开的动力学离散化、门控、历史训练样本生成和优化细节须继续列明。若门控／生命周期没有实现，不能把这个probe叫原论文完整复现。对同一生成过程另抽轨迹也只是有限分布内验证，不是未见动力学或真实MTAD泛化。

### 已完成的实际验证结果

执行前合同：[deepaf-method-qualification.md](../specs/deepaf-method-qualification.md)。入口：[deepaf_native_probe.py](../../tools/analysis/deepaf_native_probe.py)。本地运行目录为`runs/LOCAL-DEEPAF-METHOD-PROBE-20261025-attempt-1`，由本次同提交的探针代码复用未改变的P/V核心；不经过生产球面bootstrap。Python／Torch／NumPy版本、命令与来源见[provenance.json](evidence/deepaf-method-probe-20261025/provenance.json)。

实际完成两臂各256次成功更新，共512次；CPU单线程，总计276.13秒。训练／held-out／采样seed分别为20261025／20261026／20261027。两臂同初始化、同采样计划；独立重新核对512条训练记录中的两臂采样相同，并从逐场误差重新计算36组端点／运动／控制指标，全部与输出一致。没有选择中途端点、追加更新、自动选标签或晋升方法。

| 模型／控制 | 全体位置MAE（m） | 全体位置RMSE（m） | 内部速度向量RMSE（m/s） |
|---|---:|---:|---:|
| 随机初始P/V | 2383.4850 | 2897.7844 | 212.6095 |
| P真位移／V瞬时真速度，256更新 | 821.2486 | 1124.8124 | 115.2507 |
| P纠偏增量／V位置差分，256更新 | 528.5300 | 780.9050 | 110.3837 |
| 初始六点OLS-CV外推 | 1331.6150 | 1990.7213 | 171.7198 |
| 全部可见候选质心 | 120.3161 | 137.2337 | 192.7132 |

逐运动位置MAE如下，每种运动32条独立held-out轨迹、每轨迹25个后续扫描：

| 运动 | P真位移／V瞬时速度 | P纠偏／V位置差分 | OLS-CV | 候选质心 |
|---|---:|---:|---:|---:|
| CV | 788.5243 | 485.0543 | 485.5564 | 118.9028 |
| CA | 1122.5569 | 729.4197 | 1677.9605 | 122.2812 |
| CT | 552.6645 | 371.1159 | 1831.3282 | 119.7643 |

来源：[summary.json](evidence/deepaf-method-probe-20261025/summary.json)、[逐场指标](evidence/deepaf-method-probe-20261025/scene-metrics.json)、[时序与条件误差](evidence/deepaf-method-probe-20261025/behavior-summary.json)、[训练曲线](evidence/deepaf-method-probe-20261025/training-curve.csv)。完整原生JSON仍保留在上述本地run，亦包含于修订交付包。

这些结果证明核心在独立带噪轨迹上有学习变化，但没有证明获得稳定的位置滤波表现。第一到第25个预测扫描的总体位置MAE，真位移标签为97.72→1508.84m，纠偏标签为92.94→996.03m；两臂三种运动的位置误差均高于质心控制。真位移臂的CV还弱于OLS-CV，纠偏臂的CV MAE相近但RMSE更差（769.04对669.27m）。V在CA／CT和总体上优于两个控制，支持学习到部分运动信息，不能据此掩盖位置漂移。

**这不是以“必须赢质心”事后新增资格门槛。** 杂波以真目标为中心在±500m产生，使质心在本生成条件下天然携带位置线索；它不代表真实广域多目标关联排名。当前比较提示尚未充分利用可见位置线索，需要解释训练与递推行为。256步不足以证明收敛，末段训练损失仍下降：真位移臂P MSE在193–224／225–256步均值为80.15／67.49，纠偏臂为8885.51／6286.53；两种P标签的loss数值不能互相直接排名。因此不能直接归因为确定性代码bug、不能否定DeepAF，也不能把当前偏弱结果计为我方胜利。

实际训练4800帧，检测4321次、漏检479次、杂波38349个；held-out 2400帧，检测2165次、漏检235次、杂波19174个。**两部分完全没有真实候选的帧数都是0**：漏检帧仍有杂波，不能称本次性能探针验证了carry-only。单独4项软件检查覆盖仿真单位、GT／身份／运动标签不进入forward、人工全空候选时的有限递推和previous-V反馈、两臂短训练及禁止覆盖输出；4项通过，3.866秒。人工空候选的有限数值检查不是已训练模型的空候选精度验证。

另外，两臂同时改变P与V标签，不能单独把性能差归因于P解释；两臂V指标均按瞬时真速度评价，纠偏臂V却用区间速度训练。该Cartesian／1s／六点已正确起始探针没有测试生产球面、长dt、q0和单条初始历史；其有限值也不能代替生产几何检查。没有导出训练权重，保留了完整随机种子、实际采样和误差证据以便复核；没有宣称已复现原文表格。

## 七、恢复正式比较的条件

恢复099正式比较前至少需要：物理表示修正通过；核心与fixed-group adapter的输入／输出职责明确；有限独立合成检查能回答递推、缺测和目标语义问题；真实Level1资格检查未发现单位、q0、长dt、mask或明显未学习问题。方法质量应由这些证据决定，不由它是否输给我们决定。

通过后可写“DeepAF P/V核心已实现并完成行为验证，fixed056位置融合适配具备正式比较条件”。在完整原协议与训练证据缺失时，仍不能写“已复现作者完整DeepAF及原文成绩”。若未通过，则保留资格待定并明确阻塞项，不让代码数量、测试数量或正常退出替代研究结论。
