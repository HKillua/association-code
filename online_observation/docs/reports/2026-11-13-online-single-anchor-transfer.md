# 单锚跨目标迁移与正确接回

## Record

`EXPERIMENT_RECORD` v1.0，`REC-ONLINE-SINGLE-ANCHOR-20261113`，completed。`IDEA-ONLINE-SINGLE-ANCHOR-20261113` plan v1.0、C-ON42。检验单个外部历史锚能否降低其他目标的真边误差，以及在同一候选图和时刻保护正确接回。原始对照和报告权限沿用 C41。 [冻结计划](../proposals/online_single_anchor_transfer_20261113.md) · [纯传感器来源](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_single_anchor_transfer_sources_20261113.md)

冻结提交 `a10a71c`。原28场、28共享 A 组件、24场稀疏。只重读 C41 的482个固定窗、1,477个已冻结报文读出，量测不重放。候选和发布保持，全部场的新读出写盘后才读 A-B。Apple M2、Python 3.12.14、CPU线程上限2。北京时间15:58:25–15:58:29，一次真实退出0，墙时3.63秒、程序2.469秒、最大 RSS 236,584,960 B。零训练、选型、新场、V、位置参考及封存测试。122项在线测试通过，独核1,169个有锚读出和3,507项分差恒等式。 [日志](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/single_anchor_transfer_attempt_1.log) · [逐场读出](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/single_anchor_transfer_attempt_1/summary.json) · [核验](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/single_anchor_transfer_attempt_1/verification.json)

```sh
.venv/bin/python -m online_observation.single_anchor_transfer_gap \
  --archive PART1 PART2 PART3 --manifest configs/data/mtad-level1-scene3000.json \
  --calibration-run PRIVATE_C41_OUTPUT --output-dir NEW_PRIVATE_DIRECTORY
.venv/bin/python -m online_observation.single_anchor_transfer_squeeze \
  --summary NEW_PRIVATE_DIRECTORY/summary.json --output NEW_PRIVATE_DIRECTORY/aggregate.json
```

## Squeeze

`SQUEEZE_REPORT` v1.0，`SQ-ONLINE-SINGLE-ANCHOR-20261113`。20个换源错接窗（2组件、40焦点边），对照76个同形正确接回（8组件），其中22个来自目标同组件。主读出为原接回前最后已到的涉事报文。锚先按原候选资格锁定，读出时严格早于决策、年龄≤60秒，同源方向和配准阶段，与两候选的时距均相差≤10秒。最新有锚的18个错接和58个正确接回窗，事后均核实锚正确且与涉事身份不重合。 [聚合原件](/Users/a1/Projects/biye-ronghe/online_observation/docs/evidence/20261113_online_mtad/single_anchor_transfer_attempt_1/aggregate.json)

### 残差迁移与排序反转

18个有锚错接窗的真边RMSE 1,618→410 m，两锚等权 1,539→369 m，按锚×目标身份等权 1,583→438 m。18窗来自两个锚对，每个组件一对，关联4个目标身份。第一个锚在16个错接窗及16个同锚正确接回上分别为1,640→421 m及1,732→473 m；第二个锚在2个错接及4个同锚正确窗上分别为1,432→308 m及1,679→526 m。按锚去重仍观察到真实跨目标误差降低，锚数量只有2。

58个有锚正确接回窗里，38个真边残差缩小，其中16个归一化假边代价降得更多。直接用单锚的33次误拦由23次原已错和10次新翻错组成，同时解除14次原误拦。10次新错中8次真边残差仍改善。同样10窗改用共用方差后仍偏向假边，其中8窗相对共用方差原实报也是新翻错，另2窗原已错。共用方差时，修正只通过锚向量在两候选分离轴上的投影改变相对代价。21次通过一致性规则的误拦里，5次相对原实报为新误拦，其中3次真边改善、假边降得更多。

### 使用、弃权与真配对保护

| 接回前最新证据 | 错接命中（20组） | 正确接回误拦（76组） | 同组件误拦（22组） |
|---|---:|---:|---:|
| 原实报 | 7 | 47 | 11 |
| 同刻屏蔽 | 8 | 33 | 13 |
| 无锚的相同方向一致性 | 4 | 30 | 11 |
| 有近期锚就修正 | 13 | 33 | 10 |
| 锚后同刻屏蔽 | 11 | 23 | 7 |
| 锚后实报、屏蔽及共用方差都支持才干预 | 9 | 21 | 6 |
| 继续保留 I22 | 0 | 0 | 0 |

两臂有相同近期锚支持时，主规则较无锚一致性从4／18、25／58变为9／18、21／58。按锚对等权的平衡正确率为60.6%，高于无锚同规则45.4%，低于同刻屏蔽61.8%。两个目标组件分别命中8／16、1／4，误拦5／18、1／4，净 +3、0；相对无锚同规则第二组件没有新增命中。严格最新的9组命中中位提前0.611秒。用曾经警报口径，命中11／20、误拦27／76，命中提前中位2.664秒。接回当批主规则7／20、20／76，提前量为0。原28个稀疏错接总组中，主规则命中12／28、误拦31／142，不换源组为3／8、10／66。

本轮固定60秒锚龄，因此正确接回对照的直接单锚误拦33／76。C41事后最近单锚用600秒、误拦38／76。这里同锚支持、原实报、屏蔽和主规则采用相同到达时刻。固定规则仍未达到全人口命中减误拦为正的条件。新留场保持隔离，稀疏长错链、原正确配对留存与首报尚无同延迟发布结果。 [Mode C v47](/Users/a1/Projects/biye-ronghe/online_observation/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v47-单锚跨目标与排序回流)
