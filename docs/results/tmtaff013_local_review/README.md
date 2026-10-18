# TMTAFF-TIME-COST-013：本地原件核查

2026-10-18。只读核查返回原件、实际执行版本的配置和代码；未打开数据、运行模型、修改历史回执或增加预算。可复算摘要见 [review.json](review.json)。

**结论：013 是已完成 8 次更新、两次 V 关联验证后，在最终完整评价中被外层超时中断的 FAILED 测量。** 原件明确记录终评已进入并运行 **79.3846 秒**；不能说“没有启动最终完整评价”。但终评内部先对全部场景补全/关联评分，再做物理融合，现有日志没有子阶段进度，**无法确定中断前是否已经调用最终物理融合解码器**。没有最终融合结果，也不能把训练时已有 TFM 前向与最终融合评价混为一谈。

原件入口：[run.json](../../../HandOver/results/TMTAFF-TIME-COST-20261018-013/files/run.json)、[history.json](../../../HandOver/results/TMTAFF-TIME-COST-20261018-013/files/history.json)、[coverage.json](../../../HandOver/results/TMTAFF-TIME-COST-20261018-013/files/coverage.json)、[执行命令](../../../HandOver/results/TMTAFF-TIME-COST-20261018-013/command.txt)、[服务器回执](../../../HandOver/S2L-20261018-013.md)。

## 1. 已核实的运行和指标

- 外层退出码 **124**；起止 `05:31:48Z` → `06:31:27Z`，两者相差 **3579 秒**，与 `elapsed_seconds.txt` 一致。3570 秒是 TERM 时限，另有最多 30 秒清理；3579 秒仍在该 3600 秒包围内，不能仅因超过 3570 就判越界。
- `run.json` 为 `FAILED`，失败 `InterruptedError: SIGTERM: preparation/fit interrupted, not completed`；`cursor.completed_updates=8`，四阶段各 2，未记录 resume。命令使用原 007 manifest、既定 config/alignment、新 attempt-1 输出和 cuda:0；没有重新运行 prepare CLI 的证据。运行内部装配/参考对齐属于本轮既定数据准备。
- 实际版本记录为 `f25958a7c9786bdebc794a0b91d0e0e616a86b34`，是领取回执提交；S2L 中 `42bd391` 是领取前 pull 的叙述，二者不宜混写为实际执行版本。这里只记录版本，不设 SHA 准入。
- 原件确认 `contrastive`、seed20、FP32 距离、有效 batch64、pair/track microbatch1、验证 tracks1/pairs1024、训练前负例中位数校准；不能与 010/011 CE 的分数直接作同协议排名。

| 阶段 | 完成更新 | 两步秒数 | loss | 说明 |
| --- | --- | --- | --- | --- |
| TCM | 1、2 | 3.3055 / 1.4099 | 1.02812 → 0.97082 | 128 条轨迹呈现，127 条去重轨迹 |
| TAM_FROZEN | 1、2 | 11.6379 / 24.3898 | 23.27068 → 35.39971 | 128 对呈现，110 对去重；99 是轨迹数 |
| TFM_FROZEN | 1、2 | 146.4385 / 149.7244 | 84.61626 → 79.33288 | 128 个正对呈现，96 对去重；67 是轨迹数 |
| JOINT | 1、2 | 207.9527 / 205.7615 | 120.95902 → 106.14118 | 128 对呈现，128 对去重；130 是轨迹数 |

TFM 第一步记录 `gradient_norm=76648.34375`。代码 [fusion_training.py](../../../association/fusion_training.py) 返回 `clip_grad_norm_` 的结果，这是**裁剪前**范数；配置 clip=1，不能把 76648 当作最终更新使用的梯度范数或直接断言参数爆炸。两步没有训练充分性证据。

| 完成的验证 | 日志 macro-F1 | 阈值 | 秒数 | 是否替换选中关联快照 |
| --- | --- | --- | --- | --- |
| TAM_FROZEN @2 | **0.37077052529632837** | −62.44205093383789 | 552.9479 | 是 |
| JOINT @2 | 0.3628350200715334 | −98.66802215576172 | 559.9393 | 否 |

两次选择均记录 64 个 eligible scene；第二次 macro-F1 低 **0.79355 个百分点**。阈值候选数 28989/28982 是不同分数产生的阈值候选数，**不是候选边数，不能拿它判丢边**。回传包无完整逐边分数、summary 或 checkpoint，故这里是核验日志所载指标，未独立重算 macro，也无法得出 micro-F1、precision、recall。不能给未返回指标补值。

选中状态还需注意：代码在 TFM_FROZEN 完成时保留此前关联选择信息、更新冻结关联模型及新 TFM 的状态快照。因此末端应理解为“较早选中的关联状态＋完成 TFM 阶段的融合器”，而不是把没有 fusion 指标的 TAM 选择称为完整任务最优。

## 2. margin 可直接从 4096 条记录复算

已逐条读取 `margin_calibration.samples`，重新计数 `(scene_id,pair_index)` 和负例距离：

| 项目 | 本地重算 |
| --- | --- |
| 批次 / 配对呈现 | 64 / 4096 |
| 正例 / 负例呈现 | 2048 / 2048 |
| 去重对 / 重复呈现 / 场景 | 3927 / 169 / 116 |
| 负例距离中位数 | **72.64158630371094** |
| 负 hinge 严格激活 / 中位数相等 | 1024 / 0 |

这些量与记录一致。实际 `effective_margin` 为上述中位数，配置构造默认 margin1 没有取代它。校准耗时 **1498.0977 秒**。缓存终态 hits205/misses4020/evictions0，占用 131,727,360 B，说明当前容量没有发生淘汰；不能用本轮证据声称“加大缓存就能解决校准成本”。

## 3. 统一分母的成本账

下表全部以外层 **3579 秒**为分母，避免把不同计时口径混成百分比。

| 项目 | 秒 | 外层占比 |
| --- | ---: | ---: |
| margin 校准 | 1498.0977 | 41.858% |
| 八次训练更新之和 | 750.6202 | 20.973% |
| 两次完整 V 关联验证之和 | 1112.8872 | 31.095% |
| 已进入但失败的最终完整评价 | 79.3846 | 2.218% |
| run 内未分项时间，含准备及其他开销 | 126.7207 | 3.541% |
| 外层与 run 计时之差 | 11.2895 | 0.315% |

`run.elapsed_seconds=3567.7105`；上述六项相加为 3579 秒。校准＋两次验证共 **2610.9849 秒，即 72.953%**。S2L 的“74%”“合计98.1%”没有统一分母；不能照抄。回执称准备耗时98.96秒，但 preparation.json 未返回，所以该数只标为**服务器报告**，不从126.7207秒剩余项中反推出精确准备时长。外层与 run 差额还可包含退出处理、记录写盘等，现有计时不足以单独归因。

终评 79.3846 秒的直接证据是 `final_evaluation_seconds`。执行代码 [complete_training.py](../../../association/complete_training.py) 在恢复选中状态及调用 evaluator 前开始计时，并在异常的 `finally` 中写入耗时。`evaluate_complete` 先完成全部场景编码/评分，随后才遍历分组融合；它没有返回局部进度。此前完整关联验证约553/560秒，使“终评很可能仍在前置评分”成为合理推测，**但不是物理融合尚未开始的实证**；失败计时也未等待所有 outstanding CUDA 工作同步。

## 4. 回执中需要更正或保留未知的内容

| 回执说法 | 本地原件核查 |
| --- | --- |
| 逐场景 offset 全为0 | 实际执行版本 alignment 与当前配置内容相同：576场景中 **503 个为0、73 个非0，最大2453秒**；全部 gap600。例：1030→273秒、2505→26秒。实际 preparation 没有返回，不能断言服务器真的误用了全0；应先视为回执概括错误。 |
| `1,097,789,52 B` | run 原件是 **1,097,789,952 B = 1.0223966 GiB**；reserved **1,468,006,400 B = 1.3671875 GiB**。只是本进程 allocator 峰值，不是独占容量测量。 |
| run/history/coverage 大小131818/428/8773 B | 本地返回文件实际为 **542088 / 10165 / 35174 B**，均仍小于2 MiB。 |
| TAM coverage 99“对”；TFM67、JOINT130未明单位 | 原 coverage 中这些是轨迹呈现/去重轨迹数；成对呈现每阶段都是128。具体去重对为110/96/128。 |
| 数据准备全表已逐项吻合 | run 的训练 sampler 记录支持512场景、18,249,384对、110,733正对；64验证scene可由选择日志确认。完整输入/目标表、reference读数和准备98.96秒仍须 preparation 原件，当前未独立验完整表。 |
| 服务器 preparation/checkpoint 大小、末端GPU释放、pgrep无残留 | 对应原件或快照未返回，保留为回执叙述，不宣称本地验证。选卡快照确为RTX3090 GPU0，且当时该卡compute查询为空；不证明全程独占。 |

## 5. 同协议效率方向，仅建议、未实施

1. **优先排查重复冻结前向。** TFM_FROZEN 中关联模型为 eval 且冻结，当前每个 pair microbatch 重算 TCM；同一步只有19/48个去重端点，却有64对。可研究按相同输入缓存冻结补全、保持既定样本/对顺序及损失分母的实现优化，先做输出/梯度等价验证；不能把它直接推广到有 dropout、可训练 TCM 的 JOINT。
2. **复用同一选中状态下已算的验证补全/分数。** 最终评价重新遍历全部V；同一关联状态若保留 CPU 编码/分数，可能避免重复计算。须保证当前选中状态与缓存对应，保持全候选、原阈值选择和全目标融合指标；这不是删最终评价。
3. **先测构造/调用开销，再讨论预算。** 校准缓存无淘汰，不能仅从峰值显存约1 GiB推定任意批量可行。减少64批校准、减少验证、改变microbatch/采样或延长训练都属于另一个具体方案，本轮没有批准或实施；不把服务器建议“必须先压缩校准/验证”当唯一解。

本轮成本测量已经终态，不能自动重试或续跑。现有证据支持“原计划在这次预算内未完整结束”，不支持“完整流程从未开始”“新候选优于参考”或任何收敛结论。

## 6. 最小复算命令

在仓库根目录使用普通 Python 读取返回 JSON 即可，不需要 PyTorch、ZIP 或 GPU：

```bash
python3 - <<'PY'
import json, statistics
from pathlib import Path
p = Path('HandOver/results/TMTAFF-TIME-COST-20261018-013')
r = json.loads((p/'files/run.json').read_text())
h = json.loads((p/'files/history.json').read_text())
a = json.loads(Path('configs/tmtaff-time-reference-alignment.json').read_text())
negative = [d for b in r['margin_calibration']['samples']
            for d, y in zip(b['distances'], b['labels']) if y == 0]
print('margin', statistics.median(negative))
print('steps', len(h), sum(x['seconds'] for x in h))
print('validation', [(x['stage'], x.get('selection')) for x in h if 'selection' in x])
print('final', r['final_evaluation_status'], r['final_evaluation_seconds'])
print('nonzero offsets', sum(v['time_offset_seconds'] != 0 for v in a.values()))
print('file bytes', {n: (p/'files'/n).stat().st_size for n in ('run.json','history.json','coverage.json')})
PY
```

本地以上复算退出0；[review.json](review.json)保存完整精度的算术、证据缺项和文件大小，未写入任何新实验结果。
