# 130：128失败保全后的分块后续与六场原件导入

2026-10-29，执行前Idea/实现规格1.0。**须先由主仓完成128失败Record→Squeeze→Mode C，再独立发布本项；不重启128，不是原attempt自动重试。** 科学问题/候选/公式/分母完全承接[128合同](association_geometry_signal_20261029.md)及OA-C6r；这里只改变内存安排和已完成证据复用，不能写成新研究方法或独立第二次复现。实际执行只认HandOver。

## 1. 真实失败与证据边界

128服务`mtad-6b9ec1dd0a50.service`的真实journal显示systemd-oomd杀死该unit的3个进程；服务Result=signal、ExecMainStatus=9。原run/launcher残留RUNNING，无真实payload退出文件或summary，不能据陈旧状态继续等待或补造正常退出。原文件不改，明确失败事实见原run旁`terminal-synthesis.json`、`posthoc-service-journal.txt`及主仓独立失败报告。

原顺序前6场692/700/864/1187/1293/1582已各有完整features/evaluation/公共row；第7场1830无完整row。1830有551节点、151525个pair，完整pair-distance输出约77.6MB；原完整广播还有多份临时角度/坐标/三角函数数组。**这只支持减小暂存内存的修复动机，不证明oomd触发的唯一分配或真实峰值。** 原最后记录的约2GB是最后采样的进程高水位，真实终态/服务MemoryPeak未取到就写未知；系统压力、并发与整个服务unit资源不可由该采样排除。[128实际Record/Squeeze](../reports/2026-10-29-association-geometry-failure.md)保存具体范围和未知。

## 2. 固定输入与最小实现改动

- 原056/system、114 WS、manifest及128原run.scene_ids的32场顺序完全固定。前6来自原run.scene_ids[:6]，不能用后验按文件名排序的partial索引重排。原partial第7文件即使存在也不导入、不覆盖；第7起按原32顺序重新构造剩余26场。
- 新130只在pair距离构建处按**4096个pair一块**调用128同一个`distance`，保留全部P×64距离/support。128构造器增加可选builder参数，默认仍原完整广播；130显式用新纯helper。候选、score、LOO、shared/full统计、common-defined人口、数值负控、全部scorer及缺失处理均不变。
- **原L1圆均值fallback、原未wrap的圆均值表示、单贡献exact复制全部保持。** 不改成hypot，不引入新wrap，不改物理公式/尺度/容差。两成员atol1e−6m/rtol1e−10，欧氏重构atol1e−4m²/rtol1e−10原样。每新场保存/evaluate后释放完整pair tensor再处理下场，避免前后两场大数组同时驻留；特征列/评估数组照常保留用于全32聚合。
- 只读原128 `terminal-synthesis.json`的effective_status=FAILED/cause=systemd_oomd、同一unit/systemd信号9/终态、completed_scenes=6及preserved_run_status绑定；不能将仍活动或没有明确故障证据的运行当可导入失败。无修改oomd配置/服务策略/他人进程的动作。

## 3. 原六场导入、绑定与新26场

前6场`private/features-*.npz`、`private/evaluation-*.npz`和`scenes/scene-*.json.gz`复制到新130目录，**逐文件byte-exact核对，不重算其特征/候选求解/距离/负控**。输入node/source、原pair/scores/threshold、joint support、原WS组数/候选人口与标签绑定完整核对；从已冻结feature列和原身份重建的pure/known/scorer数组须与已保存evaluation全部exact，公共人口/固定AUROC也须exact。只是验证输入/标签和已存结果，不计一次新预测或独立复验。

剩余26场使用相同128构造器及唯一分块builder：每场先完整feature落盘，再打开identities及pair labels，reference数组不读。保留每场所有singleton/无support/unknown/单类/null；不删大场/候选/难例、不降低Q或精度。公共row字段和旧6场原件一致；复用来源另存`scene-provenance.json`及run/summary的6导入/26新算计数，不往原六场row塞新字段冒充原件。

全32/9组件、原6/新26、去场景数最大组件分别报告相同128候选AUROC/共同定义人口/几何负控/缺失和真实成本。原六场不是独立重复。仍不改变partition/物理输出，不把候选AUC叫最终F1/融合增量、概率似然或机制证据。原因/适配效果继续AI/UN，根因与可信同行资格均未证明。

## 4. 资源、停止与验收

新130 attempt-1一次，CPU≤4/RSS4GiB/AS8GiB/**900s advisory**，原32固定终点；0模型/前向/训练/GPU/ZIP/reference数组/T_test，无新score/阈值/normalizer。分块降低pair暂存上界，不保证宿主压力下绝不再OOM。若再次真实故障，原件/已成场景保全，结束本项，不自动再分块改配方/重试；最终进程RSS与服务MemoryPeak按实际可得范围分开记，不把最后sample叫真实峰值。

| 行为 | 实现与必要验收 |
|---|---|
| 数学不改 | 同fixture默认广播与4096分块所有private数组/audit exact；特殊L1边界和经度180原表示检查 |
| 前6不重算 | fixture六场完整原件byte exact，constructor只调用后续场次数；不读取原第7残留partial为结果 |
| 原失败不覆盖 | 原128 run及partial文件bytes保持；新目录exist_ok=false |
| 标签/故障绑定 | 原evaluation损坏或故障unit仍active拒绝，部分已导入结果保全为130 FAILED |

真实新计算完成后Record→Squeeze→Mode C，保留128失败和130新结果的不同ID/成本；无自动后继。完整坐标/身份/geometry大数组留服务器，回传必要脱敏原件/全部公共逐场/命令/退出及资源证据。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-ASSOCIATION-GEOMETRY-CONTINUATION-20261029
plan_version: '1.0'
source_matrix_id: MC-ASSOCIATION-OUTPUT-ACTION-20261029-REVIEW-R3，承接SQ-ASSOCIATION-GEOMETRY-FAILURE-128-20261029的主仓ModeC回流后发布
selected_C_id: OA-C6r
source_P_ids: [OA-P7]
source_F_ids: [OA-F15, OA-F17, OA-F18, OA-F22]
source_M_ids: [OA-M7]
source_E_ids: [OA-E15, OA-E17, OA-E18, OA-E22, EXP-128-20261029]
research_question: 承接128相同问题，用有界pair暂存完成同32场而不重复已完成六场
hypothesis_and_mechanism: 仅内存安排修复，科学机制仍沿128待验，不把分块叫新方法
testable_predictions: 默认/分块fixture数值exact、6导入26新算、相同完整候选/共同定义人口和负控
design_and_controls: 128十scorer/候选/公式/容差全部冻结，完整静态pair与LOO控制保留
sample_materials_or_experimental_units: 原32/9组件，6场复用不是独立重复
primary_outcome_and_estimand: 原128覆盖/null、LOO median AUC及数值恒等负控
secondary_outcomes: 全scorer/完整分层/原6新26/场景组件/真实资源成本
analysis_plan_and_uncertainty: 单次确定性开发诊断，无候选IID显著性，原失败真实峰值未知
resources_time_cost_and_risk: 单次CPU4/RSS4GiB/AS8GiB/900s advisory/原32终点，仍可能受宿主压力影响
ethics_safety_privacy_or_compliance: 不改128/oomd/他人进程；private留server；test封存
stopping_and_interpretation_rules: 130错误保全即结束，无自动重试改配方扩量
status: diagnostic
evidence_status: partial
missing_evidence: [completed_full32_target_signal, true128_peak_memory, matched_learned_heads, physical_mechanism]
next_skill: record-experiment_after_actual_execution
```
