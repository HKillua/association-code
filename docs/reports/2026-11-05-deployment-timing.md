# 部署耗时：同机常驻模型的有界计时

用户已澄清要求完整应用端到端与1Hz在线；该口径的后续实测见[完整链路报告](2026-11-05-pipeline-online-timing.md)。本页只保留早先单组后端计时，不能代替整场/在线部署结论。

2026-11-05，用户询问部署后的大概耗时；按[事前范围](../../HandOver/L2S-20261105-timing.md)完成一次工程profiling。[原代码/全部样本/真实launcher](/Users/a1/Projects/biye-ronghe/docs/evidence/20261105_deployment_timing/originals/)、[清单](/Users/a1/Projects/biye-ronghe/docs/evidence/20261105_deployment_timing/manifest.json)。这次不产生新的精度结论或论文发现，不增加Mode C的文献/现象计数。

## 测量单位与实际结果

同一台服务器Xeon E5-2620 v4/4线程、RTX3090 GPU0（共享，启动前利用率0%），B1、固定原V场692的前两个有双源共同支持的WS组，分别含3/4个片段；每组输出64个query。模型常驻/eval/inference_mode，2次预热＋5次计时，GPU每次前后同步。PTFM/DeepAF使用已完成152训练的末轮权重，我们使用178 full风险头；从缓存motion经特征/输入构造到CPU物理输出，排除reference评价。

下表范围是两个组各5次调用的**中位数之间范围**，不是全部调用最小最大、95%CI或大规模均匀样本。

|当前项目适配后端|CPU：缓存motion→输出|RTX3090：缓存motion→输出|任务范围|
|---|---:|---:|---|
|PTFM双分支，196519参数|5.931–10.776ms|10.835–13.289ms|网络处理52个共同支持点，另12点由Mean补齐|
|DeepAF P/V，265732参数|981.856–989.439ms|1833.294–1899.266ms|64个query依次闭环输出|
|我们的178/O，1505参数＋91二次系数|32.760–51.422ms|35.699–43.228ms|64个query的特征/评分/单O恢复；共享WS组仅用于同输入计时|

PTFM两组均分成2个共同支持段（34＋18或31＋21），任务核与DeepAF/178范围不同。不能据此当作者完整系统或公平精度排序，也不能把此处178后端称HARD/STRUCT完整系统。

## 瓶颈与部署含义

- 我们的特征构造CPU中位数约30.525–38.499ms；同GPU测试阶段另计的CPU特征构造33.604–36.469ms。GPU常驻评分从特征到CPU分数约.801–1.209ms。当前主要开销在小数组物理特征的Python/NumPy循环；1505参数不能代表整链成本。
- PTFM预构输入的核心约CPU3.547–3.716ms/GPU7.456–8.871ms。DeepAF全64query核心约CPU1012.346–1070.864ms/GPU1837.327–1851.355ms。小batch、顺序多次调用及当前实现开销可以使GPU更慢；本次未做算子profile，不能把差异唯一归因为某一项，也不能外推所有GPU部署。
- 原`predict_residual`每次重建net。本次对常驻net只改变生命周期，并在CPU/GPU各组核分数与原API exact相同，单O读出与原五臂读出O exact相同。原API另测仍保留；CPU短调用有明显调度抖动，不能据单项中位数相减声称精确节省量。尚未提交生产优化。
- 若只做粗容量估算，50组按当前串行逐组调用约为PTFM .30–.54s、DeepAF49–49.5s、我们1.64–2.57s的CPU后端工作量，**这是两个小组计时的线性外推**，不是50组实测；组规模、支持段、并行/批量和调度都会改变它。
- 缺少的完整系统计时包括原生motion构造、关联特征/基础score、CTX/HARD/STRUCT前向、构组、输入读取及外部I/O。TMTAFF也没有完成同口径计时，因此没有编造其部署数字。旧147/178/196总实验耗时包含多策略/评价/审计/写盘，不能代替上述部署延迟。
- 当前统一输入使用整段历史canonical_all/OLS，按64query离线窗口计时；计算耗时与等待观测的算法延迟不同。这批数字不能认证严格因果在线/单事件更新延迟。

## 执行、边界与全部计时样本摘要

一次attempt1、无失败/重试/训练/拟合/阈值或封存测试。新输出`/home/wangguangjie/djs/association-code/runs/DEPLOYMENT-TIMING-20261105-attempt-1`，launcher同前缀launch-attempt-1；临时脚本只读冻结run196代码，未改活动模型或旧run。CPU16–19×4、GPU0 UUID GPU-c5a00c37-f181-90b3-f220-be6e7457f473兼容共卡；未干预其他进程。unit mtad-58b32b103f4a.service，Linger=yes/Restart=no，COMPLETED/真实exit0/inactive/dead；北京14:07:02→14:10:30，总207.891880s。计时阶段131.195163s，余为导入/读输入/权重等准备及落盘；单次调用计时排除这些成本。14公共原件已回传并核验，私有motion/权重原位保留。

完整5次原始样本见timing.json。以下只将每项样本中位数与样本范围展开，未选择最好一次替代结果。

|方法|设备|case|计时范围|中位数ms|样本最小ms|样本最大ms|
|---|---|---:|---|---:|---:|---:|
| ptfm | cpu | 0 | resident_core | 3.716 | 3.654 | 3.891 |
| ptfm | cpu | 0 | cached_motion_to_output | 5.931 | 5.779 | 6.111 |
| ptfm | cpu | 1 | resident_core | 3.547 | 3.514 | 3.622 |
| ptfm | cpu | 1 | cached_motion_to_output | 10.776 | 10.658 | 17.142 |
| deepaf | cpu | 0 | resident_core | 1070.864 | 960.483 | 1163.408 |
| deepaf | cpu | 0 | cached_motion_to_output | 981.856 | 960.824 | 1070.722 |
| deepaf | cpu | 1 | resident_core | 1012.346 | 979.294 | 1062.016 |
| deepaf | cpu | 1 | cached_motion_to_output | 989.439 | 961.523 | 1083.743 |
| our178_O | cpu | 0 | resident_score_from_features | 10.359 | 8.396 | 13.060 |
| our178_O | cpu | 0 | cached_motion_to_output | 32.760 | 31.449 | 55.930 |
| our178_O | cpu | 0 | feature_construction | 30.525 | 30.286 | 31.073 |
| our178_O | cpu | 0 | original_reconstruct_score_api | 2.156 | 2.088 | 15.227 |
| our178_O | cpu | 1 | resident_score_from_features | 9.602 | 6.926 | 11.768 |
| our178_O | cpu | 1 | cached_motion_to_output | 51.422 | 48.525 | 61.000 |
| our178_O | cpu | 1 | feature_construction | 38.499 | 36.539 | 39.748 |
| our178_O | cpu | 1 | original_reconstruct_score_api | 10.032 | 9.210 | 12.656 |
| ptfm | cuda:0 | 0 | resident_core | 8.871 | 8.068 | 9.484 |
| ptfm | cuda:0 | 0 | cached_motion_to_output | 10.835 | 9.166 | 16.364 |
| ptfm | cuda:0 | 1 | resident_core | 7.456 | 6.923 | 7.559 |
| ptfm | cuda:0 | 1 | cached_motion_to_output | 13.289 | 10.539 | 18.114 |
| deepaf | cuda:0 | 0 | resident_core | 1837.327 | 1803.255 | 1911.356 |
| deepaf | cuda:0 | 0 | cached_motion_to_output | 1899.266 | 1875.961 | 1913.585 |
| deepaf | cuda:0 | 1 | resident_core | 1851.355 | 1824.962 | 1882.312 |
| deepaf | cuda:0 | 1 | cached_motion_to_output | 1833.294 | 1797.698 | 1850.359 |
| our178_O | cuda:0 | 0 | resident_score_from_features | 1.209 | 1.180 | 1.254 |
| our178_O | cuda:0 | 0 | cached_motion_to_output | 35.699 | 35.530 | 38.126 |
| our178_O | cuda:0 | 0 | feature_construction | 33.604 | 32.798 | 36.385 |
| our178_O | cuda:0 | 0 | original_reconstruct_score_api | 1.833 | 1.763 | 1.848 |
| our178_O | cuda:0 | 1 | resident_score_from_features | 0.801 | 0.780 | 0.878 |
| our178_O | cuda:0 | 1 | cached_motion_to_output | 43.228 | 39.124 | 45.315 |
| our178_O | cuda:0 | 1 | feature_construction | 36.469 | 34.190 | 38.014 |
| our178_O | cuda:0 | 1 | original_reconstruct_score_api | 1.705 | 1.665 | 1.785 |
