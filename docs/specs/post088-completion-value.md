# 088后重设计：补全的下游价值，而不是单独救feature分数

2026-10-24。已合入远程`8ca1bb3`结案复核；本设计输入为`f7923eb`回传的088及本地独立复算，独立复核原件分别保存。旧`68e83e3`的092尚未发布/领取，现由本合同的093/094替代；090/091科学设计保留，更新调度。已冻结实现`042bc0628dbdac3137a15c8ca386b03f452baca8`，具体L2S已READY；发布与领取以当前交接为准，不表示任何新任务已经运行。

## 证据增量与设计选择

`SQUEEZE-088-LEVEL1-TCM-20261024`已核39,048项日志/人口/聚合语义等式；原权重、输入卷和24,599条per_fragment留服务器，本地未重新前向。证据见[复算](../../HandOver/results/LEVEL1-TMTAFF-TCM-20261024-088/post088-redesign-review.json)。Mode C增量：

| 证据/现象 | 问题 | 支持与未知 | 新设计连接 |
|---|---|---|---|
| E088-1：155134片段一次有序覆盖、2424更新、V300/24599完整终评（DE） | P1：基础端点是否有效？ | 可接纳执行端点；一遍不等于充分训练，固定T初末缺失（UN） | 保留088原件，不能以其一遍结果宣称原论文能力上限 |
| E088-2：native一步feature MSE .0299894 vs hold .0175013，正反向均差（DE） | P2：学习补全是否有增量？ | 不能仅归咎多步rollout；不能由feature差直接推断最终任务差（UN） | 093/094同下游、同初始化与实际采样，只改变completion provider |
| E088-3：自然query inside构造相同；outside .0242917 vs .0200876，before较好而after较差（DE） | P3：误差取舍怎样传到下游？ | 分层不是正向独有故障证明；不能后验选before-model/after-hold为已验证候选 | 双臂在before/after全部支撑外统一切换provider，完整人口评价 |
| E088-4：回传逐场仅人口，完整per_fragment仍存在；终评7036s占68.52%（DE） | P4：异质性与误差集中度未知 | 本地不能重聚场景/组件性能，亦不必重复两小时前向 | 095零前向原件聚合与GPU并行，不作前置门槛 |

本批不再追加TCM训练臂：当前未发现确定实现bug，排序/预算/参数化/任务分布尚未分离，优先测其实际下游作用。TCM训练充分性保持开放，不能把这一决定写成充分收敛或永久退役。完整基线仍是one-pass TCM条件下的有界pilot，而非作者协议复现。新增九个GPU臂（090三、091二、093二、094二）足以形成持续队列。

## 093/094共同的唯一输入干预

包类型`IDEA_PACKAGE`，ID `IDEA-088-COMPLETION-VALUE-20261024`；源Mode C为上表，证据对现象verified、对机制partial。目标是直接检验冻结088 completion在真实关联/物理恢复中的价值，不主张新的模型创新。

- `learned`：完整沿用088冻结TCM的现有native complete，inside为观测插值，outside为原自回归特征。
- `state_hold`：inside逐值沿用同一观测插值；before用首原生观测、after用末原生观测的四状态，并在**当前query tau**重新过同一冻结输入投影和时间编码。不是复制旧时刻latent、零填充、随机TCM或仅替换after。
- 全部原生输入、Q64/cap128/time20、T-only normalizer、同088投影权重、目标与完整T/V人口保持；不读真实身份修正输入，不使用V误差选择路由。
- provider必须显式写入`ModelConfig`、run与checkpoint，并在重载、训练cache、TAM评分和TFM评价中保留。默认learned保持旧数学行为。state_hold只允许native冻结TAM/TFM阶段，禁止混入TCM/full/JOINT或registered接口。
- cache绑定固定policy和冻结权重；中途换policy须拒绝旧cache。state_hold不调用TCM encoder/decoder，不改变随机流。

## 093A/B：优先TFM物理价值配对

093A=learned088，093B=state_hold。均从同有效088源转移并冻结input_embedding/TCM，独立fresh TFM与FusionLoss，同seed20、Adam .002、constant、clip1、真实正对batch64、microbatch8、track_microbatch4；TFM_FROZEN1000，其他阶段0，64000对曝光。保持原standard_smooth_l1 beta .001度、train_position_affine、shortest_arc，不同时换loss。

两臂全部初始参数与优化器起点一致，训练采样与dropout随机流一致。**初始预测无需相同**，因为输入completion是干预；各自保存0/final全V oracle-only物理端点，完整预测与普通控制。TFM自身时序融合可能把outside输入差异传到inside输出，不能给下游inside输出加精确相等门槛。

配对读取实际成功更新中的有序scene/pair indices（含重复）、标签和任务有效计数，以及初始/最终RNG；不能仅凭seed或旧汇总`sampling`计数宣称actual batch相同。旧data_contract只含名单/时间/节点/计数等语义，不能夸称完整训练浮点数组已逐值复算；日志中明确实际核验范围。源/normalizer/完整评价缓存须实际比对，禁止hash作为科学门槛。

结果报告全人口、覆盖、无输入/缺参考/无预测、全部四属性误差、共同有效点、逐场/组件与最大组件外、原输入支撑内/外。方法之间覆盖不同时先报告覆盖，再在交集比较，不能以各自条件MAE直接排名。初始→final和A−B均报告；量纲保持物理单位。不执行TAM、不选择V最优checkpoint、不把oracle分数称为预测分组系统成绩。

## 094A/B：TAM实际管线配对

094A=learned088，094B=state_hold。同源、同fresh TAM及其他初始参数、同seed20/Adam .002/clip1，TAM_FROZEN4000，正负32/32整批64，其余阶段0；实际256000对曝光。TAM实际整批64，不能将TFM的microbatch8误报为TAM实际batch。

两臂沿同一T-only `train_negative_median_v1`校准策略，各自独立计算margin，校准pair人口必须逐条配对并保持训练RNG；TAM表示输入不同，margin可以不同。**这是completion＋同校准规则的管线策略比较**，不能把全部差异唯一归给表示；不另扫margin、不使用V校准、不假装有效目标系数完全相同。返回实际margin、正负距离分布、负项活跃率和梯度。

仅固定final4000的完整V所有候选分数，按原规则各自选Macro-F1阈值；同V开发选择偏差明确披露。配对表同时报告阈值、边/分组指标、覆盖/重复及逐场，不把raw距离或不同阈值的FP变化当纯权重效果。全候选节点、pair顺序、标签与人口必须逐值相同。未训练TFM不进入物理系统评分。

## 科研判断与继续分支

本批预定主要比较为093 final位置MAE及RMSE/P99/覆盖、094 final关联Macro-F1及mPCA/mPFA/分组覆盖/重复；四属性和分层完整报告。这里先判断TCM的作用方向与代价，不临时选择最好属性给其背书，不将单seed的小差异称为显著提升。paired执行验收不设先赢物理控制门槛，正确执行的负结果照常保留。

| 成对结果 | 下一步可支持的决定 |
|---|---|
| learned在TFM和TAM均更好，且覆盖/主要尾部无实质代价 | feature MSE不宜单独裁决补全；可进入匹配确认及已训练模块组合系统评价 |
| hold在两项均占优 | 当前一遍TCM未产生有效下游增量；后续选择有固定T探针的训练充分性或hold-residual参数化对照，不能宣称所有TCM无用 |
| 关联与融合方向相反 | 补全价值依任务而异，不能把模块最佳值拼成同一系统；明确选定接口后才做完整组合 |
| 两者都差于任务相容普通控制，或差异小/混合 | 下游训练量、目标、表征或协议仍开放；禁止宣布完成强baseline或通过换弱对手抬高主方法 |

最终完整learned TAM＋learned/hold TFM组合还需同一冻结阈值/数据契约与实际预测分组评价接口，本次不冒充已完成。TCM/下游预算或seed扩展须另写结果驱动的具体合同。

## 095及资源调度

095只读088服务器完整`completion-evaluation.json`和实际run/summary/history/coverage及固定manifest；拒绝`not_a_replay_input`的抽取聚合。零checkpoint/原卷/模型/前向/训练。重聚24599片段到300场/38组件/最大组件外，核回聚等式，报告native/query model−hold、before/after、距离/长度、片段与场景胜负、集中度、参考覆盖。探索性分层不用于选部署规则，不计算缺乏训练seed不确定性的伪泛化置信结论。

全局最多四个已领取单GPU，CPU4/臂。首轮090A、090B、093A、093B；接续091A、091B、090C、094A、094B，FIFO按实际资源，不挤占他人。095 CPU4与GPU并行、I/O错峰，不作为任何GPU前置。093/094各8h、090/091各4h、095 1h均为advisory，仅提醒不截断。每项一次，失败保全，无自动重试/缩batch/加步。

新预算共09012000＋0918000＋0932000＋0948000＝30000次更新，不同模型/批大小不当等算力。088现已完成，不再占名额；已执行的088不改代码、不重跑。T_test封存，066保持REJECTED。

## 实现验收映射

1. 旧learned模型/阶段默认行为保持；新hold在before/after/inside及时间单位、缺值mask边界上与088诊断定义一致。
2. policy保存/重载不丢失，registered/TCM/full/JOINT拒绝不支持的hold，cache拒绝policy变化。
3. 同实际TCM源、冻结逐值、fresh下游初参/随机流、成功batch顺序和标签配对；干预造成初始预测差异允许。
4. 真tiny Level1 TCM→两TFM→两TAM→终点cache配对全链；保留无输入、缺参考、完整候选和T_test拒读。
5. CPU比较不前向、拒绝不完整或混policy/来源；095完整per_fragment回聚与摘录拒绝。
6. 各具体L2S、冻结SHA与实际测试/独立review记录完成后才发布READY。旧092明确SUPERSEDED，不能从旧包领取。


## 实际验收记录（2026-10-24）

本轮主代理在当前实施代码实际执行：

```text
python -m unittest tests.test_completion_policy tests.models.test_timed_models \
  tests.training.test_completion_cache tests.integration.test_tcm_fit_cli \
  tests.test_completion_value_fit tests.test_level1_tmtaff_downstream \
  tests.test_tcm_completion_audit -q
40 tests / 12.290s / OK
```

这40项含新policy6、真实四臂wrapper及缓存配对3、095完整JSON3，以及旧native/cache/TCM-TAM-TFM与旧sourcewrapper回归；不把上一版090/091的43项当作本轮重跑。090/091数学/采样实现本轮未改，沿用其原已登记的针对性验证。

实际tiny路径是TCM来源→learned/hold各TFM→learned/hold各TAM→保存/重载→完整V两端点/候选→CPU配对，保留空场/缺参考/T_test拒读。两臂全部初始参数、fresh Adam、实际有序样本与任务计数、初末全RNG一致，源投影/TCM逐值冻结。learned启用/关闭audit的末态参数与RNG逐值相同。控制保持计数却改变实际pair顺序、或保持NPZ形状却改变真实节点的反例被拒绝。

两名独立审查者分别只读核core/wrapper/交接及缓存比较。review发现并修复预期不同的target policy元数据误判；真实空场检查修正intervals序列化shape；TAM原阈值绑定selected/history/summary，TFM实际初末snapshot逐值绑定初始/最终参数。修后独立只读跑通两phase V2/C2原件，内存篡改TAM summary阈值明确拒绝；无剩余阻断。独立review另亲跑4个核心边界测试通过。

095主代理另用真实088已回传run/summary/history/coverage通过正式配置预检，再确认其聚合摘录被明确拒绝为per_fragment输入；未运行正式24599片段聚合。工具只按场保留完整物理数组，但精确尾部误差向量与checkpoint仍随人口占内存；没有实测正式Level1峰值RSS，CPU/I/O需现场错峰。

新GPU效果、完整095聚合和服务器领取均未知。代码验证、freeze、发布、领取、执行验收及科研选择是不同状态；原实验目录不改写、不自动重跑。旧092的退役只影响未发布计划，不改已完成088。


### 冻结与访问状态

实现冻结`042bc0628dbdac3137a15c8ca386b03f452baca8`；九段L2S执行/联表Bash块、224处相关相对链接、新JSON及diff whitespace检查通过。090/091/093/094/095具体交接已READY，旧092 SUPERSEDED。最后同步远程`8ca1bb350ddb8352bbdcf1736219293c73eb75b4`没有新实验，只补086/088独立结案审查；其原件与本次`post088-redesign-review.json`均保留。

发布尝试实际失败：git HTTPS没有可用用户名凭据；GitHub插件create_tree返回403 `Resource not accessible by integration`。这不是自动审批系统驳回，也不需要重复批准科研预算。本地提交已完成，远程尚未取得本批代码；将全部相对该远程base的提交、更新文件、协议和审查打包为可快进git bundle。不声称服务器已运行或卡已占用。需有写权限的开发环境合入/推送后，按当前HandOver领取；不在活动实验checkout直接pull，不强推，不覆盖源运行。
