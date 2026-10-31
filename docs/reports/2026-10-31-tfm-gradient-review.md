# 167：冻结TFM原损失梯度路径 — Record→Squeeze

2026-10-31。原107末步T2四正对，eval及train-seed20各一次forward/total与position梯度，0optimizer/参数更新/V/test，TFM/损失权重前后exact、TCM冻结。梯度不是全断，前部输入路径远小于输出投影；这是当前冻结点的操作线索，不能认证原训练唯一成因或直接据此加步。

## Record

EXPERIMENT_RECORD `LEVEL1-TFM-GRADIENT-FLOW-20261031-167-attempt-1`，source_idea_id `IDEA-TFM-FROZEN-GRADIENT-FLOW-20261031`、plan1.0 / research-pipeline-contract v1.0，本聊天定位；Mode C1.72/M02/F261/E164。问题：K-only放松attention但精度没恢复，原loss对输入/中间层/输出的梯度是否全断或尺度不同？[事前Idea/spec](../proposals/object_conditions_context_20261031.md)、[L2S167](../../HandOver/L2S-20261031-167.md)。

冻结d379bba/new checkout167、science/launcher attempt1，等待165原科学服务成功终态后FIFO启动，最多四单卡。原107末步TCM/TFM/normalizer/FusionLoss/log_variance和gradient_clip=1，127原native T2/four cross-source positivepairs、Q64/204有效位置。TCM eval completion后detach，TFM分别eval和train(new dropout seed20)，原物理mps target/valid及四属性objective，没有teacher forcing/新目标/归一化或训练step。T身份仅用于固定原正对与监督，不用于部署分组。

分别求原total和加权position部分的每参数及left/right输入、memory、states梯度；记录finite/None/zero/L2/max、原方差权/损失/有效mask与理论clip比例。所有fusion/objective参数/缓冲前后逐tensorexact未更新；没有创建优化器；逐tensor前后比较覆盖TFM和损失模块，TCM仅冻结eval/no_grad、未另作逐tensor比较。新dropout RNG不等原107训练随机批次回放，当前grad也不等历史1000步grad。完整原数组沿164 run登记的161 normal预测绑定，两场最大位置差0m，与164正常指标一致；不是虚构164未存的物理数组或完整独立层审计。

北京时间11:07:22.320–11:08:09.062，程序46.741543s/launcher53.870208s，RSS1296355328B/CUDA455421952B；GPU0兼容共卡/CPU16–19×4/RSS8GiB/1800s advisory。原unit `mtad-66fcf24e8cf2.service`见[167真实launch原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_object_conditions_context/167/)，成功inactive/dead/success/ExecMainStatus0/Restart=no/Linger=yes。完整实际argv、所有参数/层梯度/损失/模式及退出/服务已回传；私有位置/权重留server。

| T2四正对，同204点/m | MAE | RMSE | P95 | P99 |
|---|---|---|---|---|
| eval | 58291.801701 | 62211.112858 | 92115.955687 | 94290.793134 |
| train | 57863.282343 | 61809.263588 | 91330.184360 | 94165.667952 |

## Squeeze：有梯度与可有效学习是不同命题

SQUEEZE_REPORT `SQ-TFM-GRADIENT-FLOW-20261031-167`，来源上列Record；partial、同冻结T2开发定位。四次forward全部finite、梯度未报非有限，权重未变。原方差四维[.471640646,.566281855,3.612000704,.987363100]；scene0 total eval/train4.427426/4.432129，scene3为4.512688/4.518466。分量及各自单位不相同，不能只比loss数字断定任务权重足够。

下表投影/各层/input列为原加权position项梯度L2，末两列为原total参数梯度和其clip=1的理论全局缩放；两者分列避免混淆。

| position梯度L2（量纲/参数数不同） | 输出投影 | decoder6 | decoder1 | FC2 | 左encoder | memory | 左输入 | total参数梯度L2 | 原clip=1比例 |
|---|---|---|---|---|---|---|---|---|---|
| 0 eval | 6.144176e+00 | 5.255852e-01 | 2.078023e-06 | 1.537667e-03 | 7.384020e-06 | 1.663503e-06 | 1.144889e-08 | 6.216692e+00 | 1.608572e-01 |
| 0 train | 5.621587e+00 | 5.140541e-01 | 1.066977e-06 | 1.569026e-03 | 9.200919e-06 | 1.798806e-06 | 1.585027e-08 | 5.702242e+00 | 1.753696e-01 |
| 3 eval | 6.030628e+00 | 4.275160e-01 | 4.349486e-07 | 2.342800e-03 | 5.063699e-06 | 2.532656e-06 | 1.286110e-08 | 6.179917e+00 | 1.618144e-01 |
| 3 train | 5.015869e+00 | 3.928694e-01 | 1.759169e-06 | 2.124809e-03 | 4.906435e-06 | 2.427932e-06 | 1.253726e-08 | 5.184371e+00 | 1.928874e-01 |

所有input.left梯度约1.1e−8–1.6e−8、memory约1.7e−6–2.5e−6，而输出投影position梯度约5–6；前四decoder/encoders也很小，eval与新train模式方向相似。**不支持“整个输入梯度严格为零”**，也不能以非零叫原训练充分可学。不同模块参数数量/激活量纲/尺度不一致，L2不可直接做可训练性排名；原全局clip会共同缩放，不单独证明梯度消失根因。attention饱和、层内归一化、权重尺度、监督几何和历史优化仍可混杂。

新dropout模式位置MAE57.863km，依然严重失效；一次随机mask较eval低约429m不是可部署改进、训练配方证据或正常模型选取。正常eval与164/161完全绑定，排除本次autograd开启导致该指标变化的解释；没有做独立重放全部梯度/历史更新，软件tiny-network有限梯度/不更新测试不能替代真实原因认证。

无科学失败/重试/加步/重训107。一次状态读取早于167入口创建run.json，随后核原同一launcher/服务终态，属于读取时序、不当科学失败或重新提交。下一轮若研究训练，先针对原输入路径/损失几何与梯度传递形成有对照的具体干预，保持完整物理与强控，不凭entropy或梯度数量直接选方案。PTFM/DeepAF活动合同不改，不拿失效TFM判赢原论文。

```text
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-TFM-GRADIENT-FLOW-20261031-167
input_experiment_record_ids: [LEVEL1-TFM-GRADIENT-FLOW-20261031-167-attempt-1]
claim_scope: frozen107_T2_original_loss_gradients_new_train_dropout_no_updates
primary_result_with_uncertainty: input_gradients_nonzero_but_tiny_vs_projection_at_current_point_no_precision_recovery
process_dynamics: 4forwards_total_position_gradients_46.742s_zero_optimizer_TFM_objective_weights_exact_TCM_eval
mechanism_and_intermediate_findings: M02_partial_gradient_scale_and_layer_path_not_unique_training_cause
errors_heterogeneity_and_boundary_conditions: eval_train_same_order_two_scenes_only_no_historical_gradient_replay
robustness_sensitivity_and_alternative_explanations: eval_matches161_164_exact_scales_dimensions_and_norms_confounding
new_P: none
new_F: F264
new_M: none
evidence_index: full_167_parameter_gradient_summary_native_normal_binding_and_actual_launcher
data_quality_and_provenance_gaps: no_independent_gradient_replay_or_training_intervention_no_V_peer_validity
open_questions: supervised_geometry_and_gradient_transmission_vs_scaling_or_normalization
next_routes: deep_read_Mode_C1.73
next_skill: deep-read Mode C
```
