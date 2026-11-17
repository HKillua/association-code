# 152A/B 和 TMTAFF 的离线基线证据待办

2026-11-17。只核本地已有合同、运行记录、报告和冻结实现。152A/B 的科学运行已结束，完整账本和独立复算待完成。TMTAFF 的训练侧断点待证。本次未运行数据诊断、模型前向或训练。 [离线科研流程](../research_workflow.md#2-必须遵守的研究顺序)

## 152A/B · 用原件收尾

**已有证据。** 原 151 缓存、114 WS 和原 147 Qhard 绑定同一 V300／38 组件。公开初核显示，152A（PTFM）150 轮、427,800 次更新、6,918,900 组呈现，152B（DeepAF）20 轮、80,800 次更新、922,520 组呈现。两臂均覆盖 2,100 个训练场的 46,126 个可训练组，采用 seed 20 和固定末轮，科学程序报告退出 0。同 WS 分组的全域条件融合中，原 147、152A PTFM＋Mean 扩域、152B 的位置 MAE 为 590.760／738.682／1,361.695 米。152A 的 RMSE、P99、速航误差低于 147。152A 核心域与 147 的 265,135 个共同位置点已有保存配对，位置 MAE 561.948／315.829 米，RMSE 1,199.516／2,434.336 米。152B 发布的速航沿用 Mean。 [原合同](../../HandOver/L2S-20261030-152.md) · [启动件与边界](../../HandOver/S2L-20261030-152-start.md) · [公开初核](2026-11-10-offline-peer-route-review.md)

| 待核原件 | 用保存结果完成的工作 |
|---|---|
| 两个原 `-launch-attempt-1`、科学 `run.json` | 绑定实际命令、冻结 checkout、设备／CPU、服务与程序真实退出状态、时间和 advisory 记录。初核科学总时间为 368,244.844／348,728.843 秒；再从原 `epoch-summary.json` 核实训练时间，从可定位的阶段时间记录核准备、末轮探针及 V 评价。缺独立计时的阶段保留未知，GPU allocator 峰与 RSS 分开。 |
| 原 `training.jsonl`、`epoch-summary.json`、固定 T 探针与 `private/last-epoch.pt` | 核每轮实际更新、每场组索引出现次数、监督分母、有限损失／梯度、末端及归一化绑定，再对账总额度。152B 第三列是位置差分 V 的监督计数，第二列重复位置计数。冻结实现每轮覆盖写同一 `last-epoch.pt`；启动回执中的“每epoch checkpoint”不能作为有整套历史权重的证据。可复用每轮日志和 u0／末轮探针判断记录到的训练过程。 |
| 151 `private/index.json`、114 WS、147 及两臂保存的 V300 逐场分组和数组 | 复核真实场序、组成员与首节点、Q64、目标及参考掩码、重复和漏目标。原 `raw-V_select-*.npz`、`states-V_select-*.npz`、`scenes/*.json.gz` 可对账 300 场与同一原 147 的预测。逐属性核单位、有符号 ME、MAE、残差 SD、覆盖及位置 RMSE/P95/P99。把 PTFM 核心域、Mean 扩域和两法共同支持单列，同时核 DeepAF 速航逐值沿用的控制。 |
| 两臂 `scene-summary.json.gz`、`summary.json` 及逐场误差 | 从保存数组独立复算全域、共同域、按场／38 组件、去最大、相对原 147 新增超过 1 公里增害及尾部取舍。完整厚尾、缺测和重复保留。V300 有历史开发暴露和共享身份，组件区间只作描述。原件缺失则登记无法核验的项目。 |

原件逐项接齐后补实际 Record → Squeeze → Mode C，使用原定型号和单次训练路径。权重、私有数组及身份留原服务器目录，回传实际命令、退出和脱敏证据。以上可以利用原件完成，无需重训或新推理。独立真实数据复算另冻结相应实验范围。 [152 原设计](../proposals/fusion_peer_full_training_20261030.md) · [冻结执行代码](https://github.com/HKillua/association-code/blob/bda61e56ab1211d5deb0750c8acc7d88f41c7447/tools/experiments/fusion_peer_full_fit.py) · [已核机读原件](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261110_route_audit/152-aggregate-observation.json>)

## TMTAFF · 断点证据与最小补证

- **已定位到 V 的判边前。** 094A/B 受控配对各 4,000 步／256,000 对呈现，训练采样与初始状态绑定。两臂的 T 负距离 margin 分别为 74.5034／83.4843，和最终 V 选的负欧氏分数阈值是两种量。V300 跨源 ROC AUC .49584／.51063、AP .00749／.00798，正例先验 .00750；到 90% 召回仍有 1,991,537／1,983,437 条假边。C147 重算 4,439,092 个 094A 分数逐值一致，原门接受 99.973%，290 场只产一组，位置覆盖 3.356%。重选一个全局阈值无法恢复该冻结端点的近随机排序。C147 是退化对照，论文 TMTAFF 胜负仍待可用整链同场证据。 [094 配对原回执](https://github.com/HKillua/association-code/blob/903f6799a3fd040ff26198f6870ea0d307d8c520/HandOver/S2L-20261024-094-summary.md) · [V300 分数分离](2026-11-12-frozen-association-pr-review.md) · [C147 整链](2026-11-14-offline-tmtaff-own-chain-c147.md)
- **训练侧先读保存账本。** 094 冻结实现会在 `history.json` 的 `tam_train_batch` 保存更新前训练模式的正／负距离均值、两类损失和落入 margin 的负边比例。先对照 `run.json`／校准报告、`history.json` 和原初始与末端 checkpoint 确认可取记录。已知两臂监督曝光一致，训练输入数组和查询掩码未独立逐值重放。批均值无法给出端点在固定配对上的正负距离分位数、排序或同源／跨源 PR。若旧账本不足，独立 Idea／READY 冻结训练侧身份隔离或明确已见的 fit 子集，保持输入、候选、端点、标签权限及推理模式，先看分位数、每场及同源／跨源 AUC／PR，再对表示或尺度做单变量验证。已见 fit 的诊断只解释训练侧表现，V300 和封存测试不用于选择新阈值。 [094 原始训练记录接口](https://github.com/HKillua/association-code/blob/e718fca51be65992efd94e3f85a1d482637f2f93/association/training/tmtaff.py#L264-L285) · [既有分量诊断与反例](../proposals/tmtaff_component_qualification_20261111.md)
- **窗、配准和 TFM 分开核。** 作者 §§2.1–2.2 描述先时空配准、再补到共同时间范围，报告长度 64、两源周期 10／20 秒。原 Level、场名单、时窗秒数、查询构造、校准参考源及算法仍缺。Cui 的 0.5° 格属于 MTAD 生成规则。项目用原生首末生成全场 Q64，观测按 CAP128 保首尾，20 秒为编码单位。与 C147 同组同输入的 Mean 位置 MAE 低于 TFM 15.654 公里。历史新单位 113 的四场固定 WS／CAP128 重融也从 Mean 的 1.101 升到 41.052 公里。两条证据支持另查原组位置读出，配准或损失各自的贡献仍未知。后续若实测窗口或配准，先固定一种变量、实际改动的输入及候选标签核法，保留同组同输入 Mean 和源时钟对照。作者表 3 的 mESD 程序仍缺，使用项目残差 SD 时注明定义。 [作者协议原件](</Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c147_tmtaff_author_protocol_20261114.md>) · [本地输入规格](../specs/tmtaff-reference.md) · [C129 同组对照](2026-11-13-tmtaff-same-group-support-review.md)
