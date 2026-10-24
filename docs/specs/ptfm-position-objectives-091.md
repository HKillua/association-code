# 091：PTFM位置目标的两步桥接

包类型：`IDEA_PACKAGE`；ID `IDEA-091-POSITION-OBJECTIVE-BRIDGE`；来源矩阵 `MODE-C-086087-CLOSE-20261024` 的 C2/P2/F2/E2。2026-10-24，用户要求继续有依据的并行探索；本项是新预算，不恢复086尝试。证据对已发生的误差取舍成立，位置目标是否为根因仍未知；本项是适配研究，不宣称原作者复现或新颖机制。

## 问题、干预与控制

086B取消辅助监督后，MAE比086A低347.101m，但RMSE高490.821m，且位置MAE仍从自身fixed.95初值721.1745升至1494.0488m。辅助任务不是充分解释。下一步分别检验平方尾部权重和位置几何度量，避免一次换成Haversine却把所有差异归因几何。

令 `r=sqrt((dlat/s_lat)^2+(wrap(dlon)/s_lon)^2)`，`d=Haversine_m/1000`；尺度仍是原T-only输入normalizer。

| 臂 | 位置数据目标（每个有监督样本） | 对照问题 |
|---|---|---|
| 已完成086B | `mean(r²)/3` | 原标准化经纬平方参考 |
| 091A radial-linear | `alpha_r*mean(r)/3` | 同一规范度量，改变误差幂次/尾部权重 |
| 091B geodesic-linear | `alpha_d*mean(d)/3` | 同为线性距离，改变规范径向度量为物理距离 |

两臂均position-only，加原有效任务常数ln3及原LSTM权重L2=1e-4；同原076B/086B全部初值、T输入normalizer、实际scene/anchor/member/mode/任务mask、seed20261023、fresh4000、batch8、Adam.001、clip1。辅助头无数据监督但保留零梯度图与Adam状态；不把辅助输出列为合格完整四属性系统。旧adaptive/086默认loss与梯度保持逐值兼容。

## T-only固定数值校准

从原086B和076B逐条相同的4000成功batch重建全部32000个T曝光。只使用原物理fixed.95位置初值及T真值：`alpha_f = E_sample[mean_valid(r0²)] / E_sample[mean_valid(f0)]`。零监督样本贡献零并留在32000分母。两臂都计算全部三种均值，输入、源batch和常数可直接配对。任何有效值非有限、均值非正或来源不符，保全失败，不另选校准样本。

`loss-calibration.json`独立记录`split=T_core, truth_used=true`、原输入/曝光/三个均值/alpha及定义；原input normalizer仍`truth_used=false`且不改。校准不前向模型、不读取V，不消费训练RNG。校准匹配的是**初始平均数据项**，不是梯度、裁剪、Adam或L2相对作用等价；这是两个预定归一化目标策略对照，不能声称纯几何或纯梯度因果。真实alpha、位置数据贡献、梯度/clip和L2全程回传，不按V调整。

## 固定T探针

固定原086B前96个成功更新×8＝768个曝光（每个训练模式256个）；小型软件fixture使用其全部较短日志。原batch与输入重建核对后，保留重复和零监督样本；用共同step0、旧086B-final、新臂final各一次CPU无梯度前向，均报告规范平方r²、规范线性r、Haversine km、物理MAE/RMSE、L2及三模式/原支撑区域统计。复制模型到CPU以统一算术，不移动活跃训练参数或消费原RNG。完整探针计划留服务器。

这是训练曝光探针，不是独立验证或完整T端点评估；不选择checkpoint、不调alpha/超参，不增加更新。旧086B的r²与物理误差在同一探针上可能同向或反向，帮助解释目标取舍；任何T/V差异仍不唯一识别泛化根因。

## 完整V与出口

固定056、原V300/38组件/9353组/610重复/442263有效位置点，CPU初末完整评价；全部输出组先预测再对齐真值。保存自身初末、076B、086B、强072C的完整缓存比较，四种MAE权重、RMSE/P95/P99、H/I、原支撑层/组件/最大组件外；两新臂成功后一次CPU只读配对表。

对自身fixed.95及强072C分别使用既有3% MAE、RMSE/P99不增、另外三种MAE及最大组件外不退化、过半场景/组件改善的预声明screen。赢086B不足以晋级。A改善B不进一步改善支持幂次线索；只有B改善支持该规范线性框架内的度量适配线索；两臂均仍差于初值则本目标桥接未解决问题。各一次，失败不自动重试；不扫alpha/学习率/步数，不因A先回成绩取消B。单seed与反复使用V、共享身份限制继续披露。

每臂单GPU/CPU4，4000更新及完整初末V，4h仅提醒；校准与T探针额外耗时未知。088已完成，本轮只读复用既有来源；CPU/内存/IO现场错峰，全局最多四个已领取单卡。T_test封存，066保持REJECTED。具体冻结与执行以当前L2S091为准，DRAFT不是可领取任务。

## 验收映射

| 行为 | 必要软件证据 |
|---|---|
| 旧loss与梯度不变 | 原adaptive/fixed-equal/position-only在相同张量/不同mask下逐值对照 |
| 新目标正确、零误差有限梯度 | 手算规范径向、跨经度/零距离、无监督目标 |
| 校准仅T且原分母 | 原batch重建、手算alpha、零监督保留、源batch/配置篡改拒绝 |
| 探针固定且不选ckpt | 首96原更新/三个冻结端点/原RNG不变 |
| 源与评价人口完整 | 真实tiny训练→保存/加载→固定T探针→CPU V→只读配对比较；错误来源/复用输出拒绝 |

## 实现与本地实测

新目标通过独立`position_objective`和`position_loss_normalization`字段及新scope启用；旧配置默认分支保持原行为。`models/ptfm.py`只扩展目标入口，`training/ptfm.py`扩展原件配对/校准/探针与checkpoint记录；独立`evaluation/ptfm_position_objectives.py`负责T校准与固定探针。两个新配置、独立runner/comparator与[L2S091](../../HandOver/L2S-20261024-091.md)形成完整执行入口；不改sequence/090/092模块。

使用`/workspace/scratch/b913d6c71f1a/experiment-dev-venv/bin/python`：

| 验证 | 实际结果 |
|---|---|
| `-m unittest tests.test_ptfm_position_objectives -v` | 3项PASS，6.285s；真实tiny074C→076B→086B→091A/B训练、checkpoint保存/加载、三个固定T端点、完整V缓存、两臂零前向联表；异常源/alpha/探针/终点拒绝 |
| `-m unittest tests.test_ptfm tests.test_ptfm_residual_fit tests.test_ptfm_objective_controls -v` | 14项PASS，10.239s；旧直接/残差/086完整链、mask/梯度/失败状态和完整人口回归 |
| 对`git show 442d2ab:association/models/ptfm.py`加载的原模块逐值对照 | direct/residual×adaptive/fixed-equal/position-only×all/partial/none，共18组loss、diagnostics、L2与所有参数梯度完全相等 |
| 独立只读复核 | 090 owner未发现阻塞；另以原objectives作12组all/partial/no-position/none的loss/diagnostics/所有参数梯度逐值对照，PASS |
| 入口及交接 | 两个CLI `--help`通过；L2S091两段bash语法和两条相对文档链接通过；修改文件`git diff --check`通过 |

真实tiny模型训练、Adam、保存加载、探针前向、全部缓存生成及比较没有mock；测试仅替换外部archive/生产规模合同与小型数据载入。源码检查和测试均不代表正式Level1/GPU实验已运行。没有扫描、额外优化或服务器访问；完整正式数值和资源耗时仍未知。最终冻结发布由主代理完成。

## 本地合入前独立对抗性复核（2026-10-24）

在本地 `.venv/bin/python` 重新执行 `tests.test_ptfm_position_objectives`、`tests.test_ptfm_objective_controls`、`tests.test_ptfm_residual`、`tests.test_ptfm`，20项全部通过（8.761秒）。另外以合包前 `8ca1bb3` 原PTFM模块独立比较 direct/residual × 三种旧objective × all/partial/none mask，18种情况的loss、diagnostics、L2及全参数梯度逐值相等；极点、对跖点及日期变更线的新目标值和梯度均有限。本次未发现需要改变091数学或训练行为的阻断缺陷。

原086B日志只保存任务mask计数，不能据其声称逐query mask已与历史数组逐值核对；现有重建约束是相同规范输入协议、实际样本/成员和任务计数。固定T探针的96更新是重复训练曝光，不是代表全部T的随机独立抽样；与V的差异也不能单独识别过拟合。α只匹配初始样本均值，不能消除梯度、裁剪和L2相对作用变化；即使091改善，也只支持预声明目标策略的适配线索，不自动证明幂次或几何是唯一根因。
