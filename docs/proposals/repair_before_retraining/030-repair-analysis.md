# 030数据深挖：先修什么，为什么

2026-10-19；回应用户“分析数据啊，看看怎么改”。本轮只读原量测、020/021日志和029/030原件，无新神经训练、无真实权重前向、无T_test；固定4个V场景、86片段、5504查询，4008点可评。以下是开发诊断与待验证方案，不是新方法或测试成绩。完整数值存入[030-audit.json](030-audit.json)的`deeper_analysis`，没有新建服务器READY。

## 1. 主要发现：目前把三类不同的问题混在同一个预测头里

### 已有量测信息没有得到充分保留

固定控制：速度线性内插，航向最短角弧内插，支撑外保持各自端点值。不调参数、不使用关联标签预测。与teacher在相同有效点比较：

| 指标 | 有效点 | teacher | 原量测控制 |
| --- | ---: | ---: | ---: |
| 全部速度MAE（native单位） | 4008 | 23.115 | 18.500 |
| 全部航向圆周MAE（度） | 4008 | 68.167 | 48.652 |
| 支撑内速度MAE | 1608 | 25.392 | 8.240 |
| 支撑内航向圆周MAE | 1608 | 65.688 | 20.469 |
| 9002支撑内速度MAE | 690 | 25.638 | 19.203 |
| 9002支撑内航向圆周MAE | 690 | 67.879 | 47.702 |

9001/reference复用会放大支撑内优势，但9002也有改善，不能把全部收益解释成复制reference。前端速度保持反而不如teacher（28.218 vs20.784），所以不能全场无条件改成端点保持。证据支持**使用观测派生底座并学习修正**，不支持把原量测视为永远正确。

### 9002的大部分支撑内位置误差是场景共同偏移

四场景各自的共同平移均值解释95.58%–98.22%的纠偏原始能量，合并四场只用一个均值则为64.43%。偏移随场景改变，不能学习一个固定source偏置后宣称解决。

进一步做了**真实身份配对oracle，仅用于诊断**：在固定原共同query网格下，每次留出一艘船，仅用同场其余3或4个身份的9001/9002原量测重叠点估计平移，再应用于留出船。估计不读取reference，不读取被留出身份的坐标；按`(其他身份, query)`合并重复证据。原690个9002支撑内有效点、39片段、17身份全部保留。

- 量测底座MAE **4488.713米→722.788米，下降83.90%**。
- 688/690点、39/39片段、17/17被留出身份均改善。
- 这是整场其他船观测和**真值身份**辅助的离线诊断，不是无标签算法、独立测试或单轨补全成果；与030冻结FP32底座4489.317米有约0.6米数值差异，oracle内部比较均用同一float64底座。

这建立了一个具体机制：其他目标可提供同场景的跨源校准信息。近似模型`z_s(t)=x(t)+b_scene,s+noise`下，只有一条偏移轨迹时，真实位置与未知偏移缺少独立参照；直接要求self头输出无偏reference，很容易把校准困难误判成时间建模失败。该模型是有数据支持的解释，不代表全MTAD都只含平移误差。

### 长外推需要运动趋势，但端点局部差分不稳

两个公式在看结果前固定，均不使用速度属性、reference或身份配对预测；支撑内保持同一030底座，支撑外按球面匀角速度延伸。一个使用端点相邻两点，另一个使用该片段首尾两点。

| 位置MAE（米） | 全部4008点 | 支撑外2400点 |
| --- | ---: | ---: |
| 原端点保持底座 | 8530.00 | 12953.95 |
| teacher | 8559.44 | 12922.49 |
| 端点相邻两点运动 | 20079.13 | 32241.00 |
| 片段首尾平均运动 | 5556.20 | 7987.69 |

首尾平均运动对teacher总体改善35.09%，但对原底座逐场变化为：1671变差8.22%，2056改善58.63%，290改善0.38%，113改善28.05%。收益高度不均，不能把总体数值当作稳定优势；也不能选择V最优公式直接替换正式方法。它说明当前输入包含尚未利用的运动趋势，同时短时位置差分会放大噪声。所有查询和缺监督点仍保留，没有缩窗或删除长距离样本。

## 2. 速度输出有一个可直接写清的尺度问题

当前输入速度为z_log1p，输出却为`v=softplus(53.31045+r)`，而损失在log-speed空间。要表示030这些实际目标，网络原始输出r需跨越约−53.52至1059.97，中位−32.27；它从0初始化。

单因素候选改法：`v=v0*exp(sigma_log*r)`，复用已冻结T_core的`v0=53.31045`、`sigma_log=1.14794`。r=0的初值不变，同一批目标所需r变为−3.919至2.647，中位−0.810。保留原速度loss、数据和其余输出即可检验该参数化。不能重解释旧checkpoint，也不能把数学上尺度更合理说成已提升分数；指数输出须保留有限值检查，不按V范围截断。

相同物理预测点的raw-output导数RMS约从0.0236变1.136，但这不是网络参数梯度，更不等于Adam训练快48倍。位置loss占比小、head范数小、全部1000步clip均不足以证明一个共同根因。当前没有理由一起提高位置权重、去正则或放开裁剪。

## 3. Baseline必须单独处理

按020原seed/config重建始终冻结的input embedding，在同4V复现029一步hold feature MSE至约1.9e−12、query hold至约1.7e−11；没有冒称取得训练权重。

原生相邻四属性zscore变化平方量中，航向占93.08%，纬经合计0.0318%。只改变航向的embedding敏感性也远大于位置。因此029整体feature MSE低25.38%不能说明位置补全有效。这是该4V输入/控制的几何性质，不是实际TCM训练梯度的分解。

TCM约400步后loss进入波动平台；1000步均未裁剪，不能照搬候选的解释。原生一步teacher与最多61步query递推存在条件差异，尚未证明是根因。未发现next-time、逆序或mask错误；原文激活描述本身有差异，不能随意换GELU后宣称修复。

baseline的优先训练对照是**原一步目标 vs加入短程free-running目标**，固定embedding、normalizer、样本及更新预算；使用原生已观测未来点监督，分别看一步和多步误差。该改法属于时间适配研究，不得冒称论文精确实现。正式baseline仍须完成明确的分阶段训练与完整下游评价，不能用眼下TCM/TAM中途端点宣布超过TMTAFF。

## 4. 证据连接与推荐修改顺序

复用原矩阵`MC-MTAD-XVIEW-20261018-v1`，本次为其实验回流增量。DE表示直接数据/代码证据，AI为推断，UN为待验证；不宣称新颖性。

| 问题P与现象F | 机制M及状态 | 对应修改/否证 |
| --- | --- | --- |
| P1 已知速度航向被重建成近常数；F1 原量测控制更好（DE） | 缺乏显式观测底座、速度输出坐标尺度不适合原log目标（代码DE，历史原因AI） | 先验证log-speed单因素，再加观测底座；如果仅输出方差变大而误差不降，则不通过 |
| P2 单轨无法可靠还原9002绝对位置；F2 留身份校准oracle大幅改善（DE） | 同场跨目标共享偏移可估（DE）；未知配对下可恢复程度UN | 配准与时间补全分开；必须用预测对应替换真值对应验证，否则不能计作方法 |
| P3 外推残差近常量；F3 稳定运动趋势有价值但两点差分劣化（DE） | 需要可控的时变运动估计，局部噪声可能放大（AI） | 单独比较可靠运动底座/残差，逐场逐距离报告；失败场景不得删除 |
| P4 baseline补全验收与位置目标脱节；F4 特征变化主要由航向驱动（DE） | 总latent MSE不能代表物理恢复（DE）；递推反馈影响UN | 独立一步/短rollout对照，最终回到关联和融合，不搬候选物理loss |

**推荐先落实C1：保留可见信息，修正输出参数化。** 最小可归因A/B只改变速度chart，使用同一T_core样本流、初始化、原teacher目标、更新数和评价人口；不同时撤掉cross/BCE或换运动底座。若有收益，再在同一log chart下比较有无观测速度/航向底座。角度残差可写为`wrap(theta_base + delta_theta)`；速度可写为`(v_base+eps)*exp(sigma_log*r)`，eps固定沿用1e−6，零残差时与原量测差这一固定epsilon，输出保持正值且保留溢出检查。这是后续独立因子，不能与第一个A/B混为同一改动。

**研究主线的下一连接是C2：场景校准与动态补全分离。** 候选结构为`量测底座 + 场景/来源校准项 + 随query变化的动态残差`。校准只能来自实际可见两源轨迹及预测对应，先检验普通几何/运动对应能收回多少oracle收益，再决定是否需要学习型关联。9001可作为当前数据的坐标锚，但必须披露这一数据条件，不能把它当作任意传感器都无偏。观测底座不等于在观测点强制残差为0；后者会锁死9002偏差。

本轮不把这几个变化同时打包重训。具体训练样本、累计更新和完整命令仍需在实施时冻结；这份分析不构成服务器执行授权。现有029/030无需重复导出同一结果，020/022—025不自动恢复。

## 5. 复算与审查

从仓库根目录运行以下三个独立、零训练的CPU分析；可追加ZIP路径参数。它们只读固定四个V场景，输出到`/tmp`，不改正式模型：

```bash
.venv/bin/python docs/proposals/repair_before_retraining/030-native-attributes.py
.venv/bin/python docs/proposals/repair_before_retraining/030-motion-controls.py
.venv/bin/python docs/proposals/repair_before_retraining/030-baseline-feature.py
```

属性控制在支撑内与现有prepare_timed_scene独立核对最大差2.84e−14。校准oracle已独立重读原量测并重算全部17个留出身份，offset和690点误差一致；运动控制以独立三维球面旋转核对前后延伸、日期线和高纬，最大差5.7e−9米。复核计算不等于增加了独立实验样本；所有结论仍限于四个已暴露开发场景。

## 6. 已实现的第一项与待确认A/B（2026-10-20）

**DRAFT：代码已准备，未启动真实训练，031/032尚不是READY。** 两组配置分别是[031原公式](../../../configs/cooperative-speed-a-031.json)与[032新公式](../../../configs/cooperative-speed-b-032.json)，唯一模型差异为`speed_output_policy`。新旧输出对应`physical_softplus`与`scaled_log`；缺省保持旧公式。新公式的`speed_reference`参与实际计算并存入权重，阻止跨公式静默加载。相同seed不增加随机初始化，零残差物理速度及非速度输出不变；指数非有限值直接报错，不按V范围裁剪。

复用`cooperative_fit.py fit --phase teacher --save-predictions`。teacher仍是原self＋正对cross＋BCE及原学习损失权重；只在其结束后停止后续训练，不能称为纯self训练。保留原`endpoint_fraction=.25`人工端点裁剪，V评价无人工遮挡。utility/adapt更新必须显式为0。固定最后一步，不选V最优checkpoint；不做关联评分/阈值选择、utility或融合。

| 冻结项 | 两组共同约定 |
| --- | --- |
| 人口与输入 | 原T_core512/V64、全部有效短轨、cap128/Q64、20秒时间单位、原offset/gap600/逐属性mask；T_test封存 |
| 训练 | 从相同seed20初始化，各1000次teacher更新、batch64、Adam lr1e-4、clip1、FP32；原模型规模与原采样流 |
| 新增总量 | 2次单卡作业、合计2000次更新；每组一次，失败计入次数，无自动重试/resume |
| 调度 | 最多两个单卡作业，可并行；沿用服务器环境与GPU现场授权，本地不监控服务器 |
| 时间 | 每组4小时仅作提醒，历史021 teacher约2.58小时；两层advisory，超时不自动杀进程或加步数 |
| 评价 | 每组固定teacher端点，一次完整V64独立片段补全，所有无reference片段仍保留；不只重评原四场 |

预期文件为`teacher.pt`、`history.json`、`run.json`、`execution.json`、`completion-evaluation.json`、`completion-predictions/index.json`和逐场NPZ。后者保存已有forward的预测、位置底座、reference、逐属性mask、时间/区域及source/identity，不追加推理。

主要比较速度MAE，辅助比较原`log(v+1e-6)`速度MSE和预测方差；位置米误差、圆周航向误差、覆盖按全体/支撑内/支撑前/支撑后/观测点/距离桶检查，逐场与来源从NPZ复算。先核两组人口、mask及原始目标一致，再报告效应与分布，不靠调阈值或删场景取得收益。仅方差增大而误差未降不支持假设；负结果仍是有效完成。本轮单seed开发对照不证明泛化、统计显著或超过TMTAFF。baseline多步训练、观测属性底座和场景校准尚未实现，不包含在该预算中。

通过的实现检查：相同seed初值与非速度输出、公式倍率/梯度、checkpoint加载隔离；带dropout的原shared与新teacher端点模型/优化器/loss状态逐位一致；真实native ZIP小夹具的完整CLI输出、无输入但有reference的场景保留、无下游阶段。它们均是CPU合成检查，不是MTAD效果。

服务器执行命令草案如下。确认后分别发布独立L2S及输出目录；当前不得从本段领取。从服务器主仓开始，每次仅选择表中的一组参数；提交时需记录实际命令、GPU、配置、退出码及偏差。冻结checkout不能在运行时pull。

| 参数 | A / 031 | B / 032 |
| --- | --- | --- |
| `task_id` | `031` | `032` |
| `task_config` | `cooperative-speed-a-031.json` | `cooperative-speed-b-032.json` |

```bash
set -euo pipefail
# 以下为A；B只替换上表两项，不循环自动提交。
task_id=031
task_config=cooperative-speed-a-031.json
task_repo=$(pwd -P)
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code="$task_repo/../association-code-run$task_id"
task_out="$task_repo/runs/COOPERATIVE-SPEED-20261020-$task_id-attempt-1"
task_record="$task_out-launch"
task_result="$task_repo/HandOver/results/COOPERATIVE-SPEED-20261020-$task_id"
test ! -e "$task_code"
test ! -e "$task_out"
test ! -e "$task_record"
test ! -e "$task_result"
mkdir "$task_result"
git worktree add --detach "$task_code" HEAD
cp "$task_code/configs/$task_config" "$task_result/actual_config.json"
"$task_python" "$task_code/tools/setup/gpu_allocation.py" --device auto \
  --output "$task_result/gpu.json" > "$task_result/gpu_uuid.txt"
IFS= read -r task_gpu < "$task_result/gpu_uuid.txt"
[[ "$task_gpu" =~ ^GPU-[a-fA-F0-9-]+$ ]]
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_record" --time-policy advisory \
  --max-seconds 14400 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES="$task_gpu" --env OMP_NUM_THREADS=4 \
  --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "$task_python" "$task_code/tools/experiments/cooperative_fit.py" fit \
  --archive /home/wangguangjie/djs/data/mtad/数据集level2.zip \
  --manifest "$task_repo/runs/PREP-NATIVE-20261017-007-attempt-1/manifest.json" \
  --alignment "$task_code/configs/tmtaff-time-reference-alignment.json" \
  --config "$task_code/configs/$task_config" --output "$task_out" \
  --device cuda:0 --cpu-threads 4 --phase teacher --save-predictions --time-policy advisory
```

训练完成需`run.phase=teacher/status=COMPLETED`、1000更新及可读teacher权重；阶段只含准备/teacher/补全评价，`completed_routes=[]`，`identity_selection_performed=false`。全V人口与原021数据契约一致、NPZ逐场齐全、解码数等于片段数。服务器回传实际命令、退出码、人口/结果JSON和history，较大NPZ可压缩回传；权重留服务器并登记路径。SUBMITTED不能代替COMPLETED，失败保留原件并停止，不自行改参数求通过。
