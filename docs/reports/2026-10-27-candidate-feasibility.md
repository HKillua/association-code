# C06–C09成立条件与比较接口核验

日期：2026-10-27。输入为资料库[Mode C 1.4](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-14)、已读原文/作者代码和主仓数据接口。遵循 `discover-innovation` 与 `transfer-ml-methods`，先核条件再设计对照。本报告是**静态可行性证据**，没有新目标域效果，不生成实验记录。当前安排只见[研究状态](../research_status.md)，执行只见[交接](../../HandOver/Handover.md)。

## 1. 四个连接不是同一层级的模型排行榜

| 连接 | 已满足或可建立的条件 | 关键缺口／反例 | 条件判断 |
|---|---|---|---|
| C06 未知时差 | 真实异步时间戳、参考时间映射、动态观测可读 | 未证实传感器存在未知clock/延迟；没有校准局部后验。空间偏差沿速度方向可与时差混淆 | 暂缓直接移植AF-EO；已知异步时刻建模仍可研究 |
| C07 相关信息 | 有原生观测和reference监督，可另建局部估计 | OLS残差不是协方差；attention不是信息矩阵；原文同步固定源数，相关机制与Q/R纠偏未单独隔离 | 先定义并建立A′的合法局部信息，再检验相关处理增量 |
| C08 条件状态过程 | 完整原生事件、真实Δt、逐属性mask及独立reference目标接口可复用 | ACSSM实际传播/方差/旁路存在差异；CRU前向滤波与离线全窗恢复信息不等；观测/状态分离、连续传播、校准是三个问题 | 保留独立路线；不能因代码可下载就选ACSSM，也不自动换成CRU |
| C09 细粒度跨来源传递 | 来源可以匿名分组，Q/K/V算子可变源数；与旧100不是同一干预 | 原T1规则网格/固定变量语义/观测锁回；无来源padding mask；缺失重建不证明数值去噪或错组鲁棒 | 可形成参数匹配的机制对照；不把变量改名为来源当原创 |

C08回答“估计对象与动力过程是什么”，C09回答“来源间如何传递模式”。不能只凭源论文分数互相排序。103仅能约束旧三场尾部归因，不能成为所有独立新路线的准入门槛；同向大错也不直接证明时差或共同偏置。

## 2. 直接证据定位与迁移代价

### C06/C07

- DE：[alignment.py](../../association/data/alignment.py) `validate_alignment`（154–160行附近）只验证固定offset一致性，明确不估唯一时钟。AI：真实timestamp不齐不等于未知clock已存在。
- DE：[motion_evidence.py](../../association/data/motion_evidence.py) 63–74行说明OLS诊断不是校准协方差。AI：不能把`fit_rmse_m`平方直接当IFNet局部后验。
- 来源：[TSP/IFNet逐式分析](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/recent_domain_sources_20261027.md)。TSP补件公式订正不认证作者程序；IFNet共享零均值随机噪声不等于持久共同位置偏差。
- 若以后选择C06：A可信时间戳动态估计→A′同一局部后验/公共时刻接口→A′+B仅加入相对时差估计。须先有未知时差证据及可辨识条件，不能把reference误差最小化所得偏移送入forward。
- 若以后选择C07：A普通恢复→A′局部估计与透明融合→A′+B仅加入相关处理。至少保留独立信息求和/保守相关控制，报告正定性、求逆、NLL/覆盖率；误差降低不能独自证明学到了真实相关性。

### C08

作者代码根目录：[ACSSM](/Users/a1/Projects/biye-ronghe/papers/recent_timeseries_20261027/acssm/code/)、[CRU](/Users/a1/Projects/biye-ronghe/papers/cross_domain_sources_20261027/cru_code/)。完整来源阅读见[时序来源§4](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/recent_timeseries_sources_20261027.md)。

| DE定位 | 实际行为 | AI/UN边界 |
|---|---|---|
| ACSSM `lib/sde.py:80–138`；`lib/data_utils.py:228,308` | 绝对时间进入逐步指数，loss另一处才取相邻差；均值有`C(E(Z))`历史旁路；标准差经线性映射 | 需要显式Δt、正确协方差变换及旁路控制；对发表成绩影响未知 |
| ACSSM `lib/losses.py:41–62`；`model/control.py:28,120,239` | 采样MSE先对各样本评分；零值当缺失的forward-fill；history/full权限不同 | 后验均值MSE、样本期望误差、概率评分必须分开；不能把有效零值当mask |
| CRU `lib/CRULayer.py:30–77`；`lib/CRUCell.py:277,437,483–500` | RKN/CRU共享外层，真实相邻Δt，显式测量方差和矩阵指数；仅保存协方差分块对角 | 可用于隔离连续传播，但不提供任意完整协方差或源间相关解法 |
| CRU `lib/CRU.py:159–162` | 返回前向滤波状态 | 不能用较少未来信息的控制证明离线全窗模型更好 |

可归因设计应收窄：共享事件编码/监督/信息集，A确定性恢复→A′离散潜状态更新→A′+B仅改真实Δt传播。A′+B−A′只检验连续传播；若另研究测量方差，必须分开控制。查询网格或时间原点变化不应凭空制造观测；这些属于实现性质检查，不是海上效果。

ACSSM仓无可确认的复用许可证；不直接拷贝其代码进入主仓。独立实现论文机制仍须明确与作者实现的差异。CRU是2022基础机制来源，不算新增近两年论文。

### C09

作者代码根目录：[T1 GitHub版](/Users/a1/Projects/biye-ronghe/papers/recent_timeseries_20261027/t1/code/)、[官方附件版](/Users/a1/Projects/biye-ronghe/papers/recent_timeseries_20261027/t1/supplement_src/)。来源正文p4–5式1/3/4、Table5及反例表见[时序来源§5](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/recent_timeseries_sources_20261027.md)。

| DE定位 | 真实机制／条件 | 迁移判断 |
|---|---|---|
| 附件`nn/modules/t1/backbone_imputation.py:90–115` | 每通道c用整窗`Q_c,K_c,V_c∈R^(M×L)`形成M×M来源attention | 一通道一张全窗权重；不是每时刻独立可靠度 |
| GitHub `models/T1.py:274–300,568–569` | 算子按运行时M reshape；完整模型有M专属embedding；attention无source mask | 可变M并非算子禁止，但必须匿名化并显式排除padding/空来源；值mask不能替代source mask |
| GitHub `models/T1.py:339–340,626–690` | 固定H/T归一、各变量独立统计、imputation不使用x_mark | 必须建立共同物理窗口与真实事件→潜表示，不能伪造规则观测；每源独立归一可能抹掉几何差异 |
| 附件`imputation/t1/core.py:32–35`；`nn/modules/loss.py:156–166` | 最终锁回已观测值；ORT/MIT重建原观测，权重随配置变化 | 不直接支持支撑内去噪。改reference监督/解除锁回是A′共同适配，不是路由机制B |

旧100：[native_query.py](../../association/models/native_query.py) 150–169行先对每源独立patch摘要、query解码，最后等权平均；冻结090C残差见[训练29–46行](../../association/training/native_query.py)。100未隔离跨来源交互或每通道路由。旧098：[query_relation.py](../../association/models/query_relation.py) 116–127行有每query的helper集合attention，但不是T1的整窗通道绑定。这些负结果保留，不推广为C09失败。

## 3. 可复用接口与不得继承的旧假设

- [source_events.py](../../association/data/source_events.py)：canonical全事件、原秒、逐属性有效mask；同源匿名分组只去重同一原始event ID，数值相等的不同原事件保留。来源名用于等价类而非模型特征。
- [dataset.py](../../association/data/dataset.py)：先建观测后读reference；[targets.py](../../association/data/targets.py)显式固定时钟映射/逐属性监督mask。监督不进入事件构造、forward或分组。
- [sequence_position.py](../../association/evaluation/sequence_position.py)：可复用输出组—目标绑定、遗漏/重复人口和完整指标；复用前需解耦090C/残差特例。
- 实验100的`NativeExperiment`和相应evaluator明确绑定frozen090C及速度/航向不变。这是历史实验定义，不是新模型必须继承的骨架。

## 4. 与TMTAFF真正同任务比较的最小工作

| 问题 | 静态已核事实 | 必要处理 |
|---|---|---|
| 四属性单位 | `data/timed.py`原值统计、`targets.py`原reference、`training/tmtaff.py`原目标未统一；`PairFusion`仅speed softplus | 新单位版本须贯通输入、normalizer、目标、单例、普通控制、checkpoint/缓存声明和报告。旧神经输出不能整体乘系数修复 |
| 分组人口 | `evaluation/tmtaff.py:evaluate_complete`已有`association_records`与固定threshold入口，核场景/节点/候选一致并跳过TAM forward | 复用056缓存接口；固定组结果只能支持恢复/融合链，不能宣称赢完整系统 |
| 普通控制 | 该evaluator的`predicted_observed_controls`是cap128选点支撑内插值，外侧缺预测；090C控制是canonical_all+OLS | 同名Anchor/Mean不同信息预算不能混表。共同比较须复用统一控制或明确另名 |
| 指标 | `paper_fusion_metrics.py`已有预测/真值分别声明速度单位；TMTAFF调用此前走native默认 | 显式接线即可，不重造公式；标准残差SD与印刷Eq29诊断保持分列 |
| 架构/训练 | `PairFusion`使用FC memory/递归与单例旁路；现配置1000更新不等于作者80epochs | 冻结并声明适配解释/实际曝光，未公开分支不猜成精确复现；不得一次改反馈/结构/loss再归因单位 |

本报告说明**为什么需要修复**，实现与验收记录放现有[TMTAFF spec](../specs/tmtaff-reference.md)。软件修复成功不等于已获得同协议训练结果，也不证明历史失败仅由单位造成。

## 5. 证据边界

SC来源作者主张、DE静态代码/原文、AI上述迁移判断、UN目标域增量分别保留。三名子代理独立检查C08、C09、C06/C07及比较接口，主代理核主仓输入/评价契约后整合；这不是三次独立实验。没有扫描新原始人口，没有打开T_test，没有作者复现实验，没有服务器实际诊断结论。
