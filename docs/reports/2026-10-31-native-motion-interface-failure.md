# 184/185：生产分组类型接口失败，非方法阴性

2026-10-31，REC-ENDPOINT-KINEMATICS-184、REC-ONE-SIDED-PERSISTENCE-185 → SQ-NATIVE-LIST-INTERFACE-FAILURE，research-pipeline-contract v1.0。[原Idea/plan1.0](../proposals/native_motion_interventions_20261031.md)，MC1.93/C64/C65；两科学各一次FAILED，evidence_status=verified_execution_failure，0完整T场/0V/0神经更新，未得到科学分数。原312b123 run184/185与launcher/失败状态保持，不自动重投或覆盖。

## Record：实际执行与偏差

按[L2S184](../../HandOver/L2S-20261031-184.md)/[185](../../HandOver/L2S-20261031-185.md)启动，CPU16–19/20–23×4、无GPU，3600s advisory/RSS8GiB，Linger=yes/Restart=no。两服务`mtad-1dd613fccd75.service`/`mtad-2495a10c7722.service`真实failed/failed、Resultexit-code/ExecMainStatus1。

184 payload10:47:58.081869Z–10:48:06.212000Z，launcher8.262476s；程序.282694s/RSS401588224B。185 payload10:47:58.480031Z–10:48:07.563516Z，launcher9.215925s；程序1.824369s/RSS401424384B。总成本17.478402s，0超时/重试/数据排除，退出前没有完整第一场输出或summary。

两原入口`native_motion_pilot.py`经`groups_from_membership`得到list[list[int]]；新`ni.bridge`/`ni.persistence`却把各组当NumPy数组进行布尔索引，均抛`TypeError: only integer scalar arrays can be converted to a scalar index`。184在第一组构建stream时失败；185在首场全部节点工作拟合后、第一组读出选择时失败。185 run.json尚未更新native_likelihood_fits而显示0；按已抵达失败行与首场157节点的有效报告数，可确认前置**314个native OU工作fit已完成**，未保存模型/逐fit日志，不能当已审模型或写零实际拟合。上述CPU耗时包含它们。

15项发布前检查虽通过，但测试给新函数的是list[np.ndarray]，遗漏生产list[list[int]]，不是MTAD数据坏或公式被否证。两[失败原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_native_motion)分别12文件8345/8101B，加真实终态清单/原stderr与request完整回传、逐文件大小核齐；private未产生完整新预测，原数据/权重/183/152均不动。

## Squeeze：可支持的定位与修复边界

DE：错误来自组成员容器在接口间的不一致，与实际语法/栈位置一致；没有走到误差评价，不支持C64/C65科学假设成立或失败。最小修复为两个新函数每组入口`np.asarray(nodes,int)`，不改物理公式/OU网格/源选择/数据/预算；新增同一list和数组输入逐数组相等检查。软件修复也不等实际场景资格已过。

该发现先回Mode C1.94/F290/E193，保留184/185失败与额外成本；再独立[Idea186](../proposals/native_motion_interface_qualification_20261031.md)用原全部T32生产入口/独立重建、不读参考/不评价效果建立条件。186成功并Record/Squeeze/Mode C之后，才另发新编号科学合同；不是在旧目录自动重试，亦不因接口修复就宣效果。183的独立特征路线不受此错误影响，继续原额度。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-NATIVE-LIST-INTERFACE-FAILURE
input_experiment_record_ids: [REC-ENDPOINT-KINEMATICS-184, REC-ONE-SIDED-PERSISTENCE-185]
execution_status: failed_before_first_complete_scene
evidence_status: verified_execution_failure
claim_scope: production_list_group_interface_only_no_scientific_effect_result
primary_result_with_uncertainty: none_no_complete_T_or_V_output
process_dynamics: science8.262476_plus9.215925_seconds_185_314_native_working_fits_reached_not_checkpointed
new_F: F290
new_E: E193
new_M: none
next_skill: deep-read_Mode_C_before_explicit_repair_qualification
```
