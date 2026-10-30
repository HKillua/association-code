# 148：PTFM/DeepAF真实训练接通与TMTAFF保存端点核查 — Record→Squeeze

2026-10-30。两路真实GPU训练均完成各8次更新；全T2100训练与V300同行比较尚未完成。TMTAFF原107没有重训。本报告不拿训练侧探针认证公平胜负，也不把loss下降叫机制。

## Record：事前合同与真实操作

EXPERIMENT_RECORD `LEVEL1-PTFM-T-CONNECTION-20261030-148A-attempt-1` / `LEVEL1-DEEPAF-T-CONNECTION-20261030-148B-attempt-1`；source_idea_id `IDEA-FUSION-PEER-T-CONNECTION-20261030`、plan1.0，[Idea/spec](../proposals/fusion_peer_training_connection_20261030.md)、[L2S148](../../HandOver/L2S-20261030-148.md)。消费Mode C1.61和原PTFM/DeepAF来源分析/方法边界，不新增精读论文。

原127 canonical_all首T8缓存/node绑定、Q64；同347个可训练真组池，均恰两源有共同支持，量测速kn转m/s一次，参考保持m/s。fresh seed20、Adam .001/clip1，固定8×4，独立新权重，无旧072/074C/099 sampler/checkpoint；同一固定计划32组呈现、32 unique、实际来自5/8场，不把32当遍历T8或独立实验重复。0V/test/raw CSV/teacher/选模；T真身份只作训练分组/监督，forward无truth/身份字段。

- PTFM：两独立源状态支路，保原特征门控/LSTM并行、任务权重loss；去旧额外context_gate。训练每组最长连续共同支持段，同源多片段支持内均值、局部归一化、208→128、Eq21解释、航向/输出映射均为明确适配。T探针段外使用公共Mean回退；这不是原生全域能力。两分支换序报告输出差异，不要求原有不同参数分支天然置换等价。
- DeepAF：独立P/V各6encoder/6decoder、d32、4头、原距离cross-attention；5步自身历史和自身V虚拟候选，去fixed095初始化，改可见Anchor-OLS起点/切向速度。仍离线全窗OLS、实际dt、TBPTT1、P desired和位置差分V；对外速度航向Mean，不宣作者在线生命周期复现或四属性学习。

| 实际成本 | PTFM148A | DeepAF148B |
|---|---:|---:|
| code / 冻结目录 | 65f6efd / run148A | 65f6efd / run148B |
| 北京时间起止 | 18:44:50–18:45:12 | 18:44:50–18:45:55 |
| 参数数 | 196519 | 265732 |
| 程序 / launcher秒 | 22.267 / 35.439 | 64.943 / 73.946 |
| RSS峰B | 1340268544 | 1243529216 |
| 进程CUDA allocated峰B | 30358528 | 83190272 |
| 单卡/CPU | GPU0，CPU4–7×4 | GPU1，CPU8–11×4 |
| 成功更新/呈现 | 8 / 32 | 8 / 32 |
| 分支实际变化张量数 | 两支各4 | P/V各262 |

GPU0/1按实际request UUID兼容共卡，未触碰他人进程；allocator峰不是整卡总显存。unit分别`mtad-40153272581d.service`/`mtad-3a2b933d2043.service`，均同一launcher COMPLETED/returncode0、真实inactive/dead/ExecMainStatus0、Restart=no/Linger=yes，无超1800s advisory、无失败重试。未记录完整CPU accounting/硬件功耗，不从墙时推算；程序不含Python import前时间，launcher包含，不称在线延迟。

原件：[148A](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_full_benchmark_peer_connection/148A/)、[148B](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_full_benchmark_peer_connection/148B/)含run/summary/training、命令/退出/服务及实际曝光。normalizer均值/scale、样本计划/身份/坐标/权重只留服务器，导出明确标脱敏；没有删除原始内容。计划主要额度一致，无自动续训；4组探针自身179有效位置点，训练组非独立确认集。

## Squeeze：训练能更新，不等训练充分或强baseline

SQUEEZE_REPORT `SQ-FUSION-PEER-T-CONNECTION-20261030-148`，输入两份Record，证据为真实端点/训练日志，未另做独立重训/梯度复算。

| 固定第一batch训练探针 | u0 MAE / RMSE m | u8 MAE / RMSE m | u8 P95 / P99 m |
|---|---:|---:|---:|
| 同点Mean | 1485.411 / 1598.912 | 同左 | 2265.331 / 2375.839 |
| PTFM核心＋段外Mean | 8359.579 / 12884.965 | 7684.540 / 10814.214 | 25028.259 / 28437.713 |
| DeepAF原P/V任务适配 | 64997.511 / 78931.678 | 1912.208 / 2205.181 | 3463.338 / 3632.255 |

PTFM两分支均更新，第一/末步loss4.374760→3.640209；DeepAF P/V均更新，4169.030777→1.731147，但不同batch变化不能当完整学习曲线的同点因果。固定探针DeepAF大错明显收缩，是正面训练通路证据；两者u8仍差Mean，8步不支持否定方法或认为已收敛。PTFM段内129点的分支交换位移MAE790.585→907.667m，表明其分支参数不对称，不能冒称集合不变。选定共同局部原点只固定对照坐标，不使网络数学上对称。

已回答：真实T数据、新单位、合法输入与两个独立分支的参数更新可以接通，旧训练锁与fixed095不再是运行前置。未回答：全T2100的曝光/收敛、完整WS预测组/单源/错组/漏目标、V300覆盖与尾部、按原域共同支持的比较、输入点数匹配和多seed稳定性。后续必须另定具体全量合同，用统一入口；不拿本表和147 V表混排名，也不靠无效适配证明胜原方法。

## TMTAFF：只读原107训练痕迹，仍未锁定失效根因

来源原107/tfm COMPLETED1000端点、checkpoint/summary/history；`tmtaff_saved_training_trace.py`在独立d0b25a5 checkout CPU2/taskset14–15有界调用exit0、17.834880s、RSS1156984832B。0模型构造/forward/更新、新V数组/raw reference/test；原107/113不动。脱敏原件：[saved trace](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_full_benchmark_peer_connection/tmtaff-saved-trace.json)。此项是既有实验只读再分析，不是新训练或方法成绩。

DE：checkpoint历史、normalizer、position_mean/std逐项一致，物理速度buffer1，未见这些保存字段的串用；64000次对呈现来自802个T场，29002个unique场内pair，不是T2100一遍。末步原生loss系数lat/lon/vel/cou为1.060129/.882952/.138427/.506399；它们权衡不同原生损失，不是传感器可信度。

| 原T日志窗口 | 总loss均值 | lat分量 | lon分量 | speed分量 | course分量 |
|---|---:|---:|---:|---:|---:|
| 1–100 | 17.493081 | .407598 | .557709 | 29.569927 | .942832 |
| 401–500 | 9.459810 | .324859 | .400084 | 20.989558 | .990518 |
| 901–1000 | 7.764735 | .288297 | .337250 | 24.862266 | .970718 |

T训练日志的位置信号也并未接近零；但batch随窗口变化、分量是原生SmoothL1/其它loss，不能把该表当固定T物理MAE或据其判定仅过拟合。全球T归一化输出每1单位对应纬向45.342511km、经向赤道尺度58.240868km，说明数值误差放大敏感性；不是normalizer错误或失效根因的证明。当前未读T物理输出，不能分离T拟合、TCM特征、递归/输出参数化与域差；原113四万米级失败不只由预测错组解释（原oracle也差），保留所有竞争解释。

偏差/缺口：148B原summary的scope复用了含PTFM段外Mean说明的通用字符串；实际DeepAF代码输出完整Q64位置，仅速度/航向Mean，不使用PTFM段外位置回退。此处订正，原summary不覆盖；主仓后继文字已分方法标注，未重跑或改变数值。本轮确实使用真实GPU，但只覆盖小额T8及明确适配，尚无可信同行排名。TMTAFF再分析没有新的干预，不以“尺度很大”直接决定换模块。进入全量训练前，复用本报告固定曝光/物理恢复/原适用域/输出权限的合同，不能加8步后自动无限追数值。

回流：F247、E147（148A）、E148（148B）、E149（原107保存痕迹再分析），M02/现有协议边界定向更新，不增新机制或论文。与147 F246/E146合并到Mode C1.62；下一独立Idea须基于新增证据，自己的方法与同行适配分别核，不拼模块冠军。
