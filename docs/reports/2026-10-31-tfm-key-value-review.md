# 164：冻结TFM的K/V尺度分离 — Record→Squeeze

2026-10-31。冻结107末步、原161 T2四正对，四臂一次exit0；单改K能明显放松首步attention集中，但位置误差未恢复。没有重训107，不把干预响应叫唯一训练根因或可信同行输赢。

## Record

EXPERIMENT_RECORD `LEVEL1-TFM-KEY-VALUE-20261031-164-attempt-1`，source_idea_id `IDEA-TFM-KEY-VALUE-SEPARATION-20261031`、plan1.0 / research-pipeline-contract v1.0，本聊天计算定位。来源Mode C1.70/M02/F258/E161；[事前Idea/spec](../proposals/rare_reversal_separation_20261031.md)、[L2S](../../HandOver/L2S-20261031-164.md)。假设AI：161同时改变K/V，首步集中和输出响应可能经不同路径；本轮只作操作分离，不选倍率/加步。

冻结3d5191a/new checkout164/run/launcher attempt1；107 `tfm/checkpoint.pt`、127 native与161 layer-run原T2/4positivepairs/6节点出现、Q64共256query/204有效位置，0训练/V/test。K/V各1或.01，四臂NORMAL/KEY_ONLY/VALUE_ONLY/BOTH，same权重/mask/自回归/projection；两个相等倍率用同tensor以保持packed projection路径，独立倍率经cross-MHA hooks改各输入。全部6层中间量、first-step attention重放、同解码索引跨4输入统计保留；不同场的同index不冒充相同物理时间。

北京时间02:08:52.893–02:09:29.463，程序36.570074s/launcher40.208406s，RSS1109004288B/CUDA49682432B；GPU0 UUIDc5a00c37-f181-90b3-f220-be6e7457f473兼容共卡、CPU16–19×4/RSS8GiB/1800s advisory，原unit `mtad-04efa3f015a3.service`成功inactive/dead/ExecMainStatus0/Restart=no/Linger=yes。完整实际命令/env、输入/输出、退出/服务与每层公共摘要见[164原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_rare_reversal_separation/164/)，原坐标、完整层数组/权重留server。

| T2四正对，204同点/m | MAE | RMSE | P95 | P99 |
|---|---|---|---|---|
| NORMAL | 58291.802 | 62211.113 | 92115.956 | 94290.793 |
| KEY_ONLY | 58373.377 | 62287.362 | 92196.576 | 94371.394 |
| VALUE_ONLY | 58344.443 | 62259.595 | 92165.325 | 94340.122 |
| BOTH | 58551.924 | 62452.326 | 92365.598 | 94540.298 |

NORMAL及BOTH对161相应正常/倍率.01在两场位置最大差均0m；这是冻结原件绑定，不是对新164所有层forward的独立重放审计。软件非退化BOS fixture核首层V-only不改attention/K-only有变化、both等价161及hook清除；原fixture BOS=0导致无法检验K变化，修正后提交前通过，不是科学失败或成绩。

## Squeeze：放松attention集中不等于恢复输入依赖和精度

SQUEEZE_REPORT `SQ-TFM-KEY-VALUE-20261031-164`。冻结系统操作证据partial；无学习曲线/梯度新证据。第一场first-step attention entropy第一层/末层：NORMAL .097500/2.286e−24；KEY_ONLY4.145706/3.877269；VALUE_ONLY .097500/7.383e−19；BOTH4.145706/3.915770。第二场对应NORMAL .097495/2.193e−24、KEY4.145687/3.876698、VALUE .097495/7.103e−19、BOTH4.145687/3.915261，方向一致。

同解码index跨四输入末层最大range：NORMAL1.430511e−6、KEY1.230836e−5、VALUE8.642673e−6、BOTH4.409254e−5。K-only放松首层/末层分布、V-only首层分布保持却输出改变，支持在当前冻结网络中区分logit与value路径；四臂仍约58km MAE，所有改动都比正常更差。不能把跨输入差增大或entropy接近均匀当精度恢复。

首层query由同BOS固定，K-only的首层比较较直接；后续层/自回归步的query、归一化与反馈改变，末层entropy变化不能都归为只改K的直接因果。不同输入key/value幅度、原107训练曝光/梯度、层内归一化及损失监督仍是竞争解释；单倍率冻结干预既不能认证训练病因，也不能说明重训此倍率有效。样本仅T2四正对、重复原107/161，不作新域/多seed或全V300推断，独立完整层重放未做。

本配方不接部署/不扩大、不重训107/追加TAM/步数；闭合“仅放松首步集中即可救精度”的当前简单假设。后继若研究训练，需新证据区分真实梯度/监督传播与K/V/层归一化效应，不能直接沿同倍率继续加训练。原失败适配保留，PTFM/DeepAF公平比较不受改动。

```text
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-TFM-KEY-VALUE-20261031-164
input_experiment_record_ids: [LEVEL1-TFM-KEY-VALUE-20261031-164-attempt-1]
claim_scope: frozen107_T2_four_pairs_KV_2x2_no_train_V_or_test
primary_result_with_uncertainty: K_only_attention_relaxed_without_precision_recovery_T2_descriptive
process_dynamics: 36.570s_zero_updates
mechanism_and_intermediate_findings: partial_logit_vs_value_operational_separation_first_layer_fixed_BOS
errors_heterogeneity_and_boundary_conditions: both_scenes_same_direction_204_common_positions_deeper_feedback_confounding
robustness_sensitivity_and_alternative_explanations: NORMAL_BOTH_match161_exact_software_fixture_not_scientific_replication
new_P: none
new_F: F261
new_M: none_M02_partial_update
evidence_index: full_layer_summary_cross_input_response_position_run_launcher_originals
data_quality_and_provenance_gaps: independent_layer_replay_and_training_gradient_missing_same_T2
open_questions: training_supervision_gradient_normalization_vs_scale
next_routes: deep_read_Mode_C1.71
next_skill: deep-read Mode C
```
