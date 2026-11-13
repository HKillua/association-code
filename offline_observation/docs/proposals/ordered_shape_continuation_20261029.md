# 132：复用126缓存，修复采样输入的文件句柄生命周期

2026-10-29，执行前Idea/规格1.0，实现已核验，实际执行只看主交接。消费[126失败Record→Squeeze](../reports/2026-10-29-ordered-input-failure.md)及Mode C1.48的F227/M98/E125、IT-C4-resource；原科学问题/四臂数值合同仍为[126v1.1](ordered_shape_signal_20261029.md)。主交接发布后才可一次执行，不重启或修改126。

## 事实、根因和研究边界

126真实FAILED/exit1，2949.129498s，0 optimizer update、0完成臂，T2100/V300缓存和四臂T-only normalizers已完成；没有shared plan、初值、history或模型。真实stderr指向`build_plan→sampler_input:215–216→np.load(mmap_mode='r')→Errno24`。旧列表推导一次同时持有2100×2=4200个memmap；构造失败发生在后续finally前。`phase=normalizer`是未及时更新的阶段字段，不说明归一化泄漏：原`fit_normalizers`已逐场try/finally关闭。

这是执行资源/代码失败，**不是有序输入无效的科学负结果**；F227/M98/E125不增加科学精读或效果证据。132只修正采样输入的存储传输，不改变模型、原始信息、训练或评价。无需提高全局ulimit、修改他人进程、重新CSV/ZIP准备或改实验配方。

## 最小修复与不变项

新`ordered_shape_continuation.py`仅将采样器输入改为逐场`np.load(...,allow_pickle=False)`普通ndarray；每次调用返回即关闭文件，不再同时挂4200个mmap。原`BalancedScenePairSampler`、seed20、T scene/pair顺序、正负128/128、批256、1000实际计划、四臂共用计划的生成顺序完全不变。采样器构造后释放普通pairs/labels输入，只留下其原有class pools和label副本。

额外RAM诚实计入：T pairs文件约442.390MB、labels约27.901MB；采样器另存原label copy及class pools，峰值比映射方式高但仍按原RSS8GiB止线检查。无需重复将大features驻留内存。低FD软件反例必须同时显示旧列表映射失败、新普通数组完成；小数据原/新计划和完整曝光统计逐元素相同。

四臂SUMMARY/ORDERED/DISTRIBUTION/SHUFFLED、30→64→64→1/6209参数、同seed20全新共享初值、输入映射/46列cache、FP64 base＋FP32 residual、Adam .001/clip1/BCE、各1000成功更新共4000、final-only一次全V推理、完整阈值开发/空层/≥/并列较高约定、信息权限均沿126v1.1，不恢复已有训练（原本0步）。六配对、all/same/cross、joint/no-joint、正式ITSA/MSTSA八指标、V300/38组件和去最大全部保留；不从本轮V挑频率/seed/步数或超参。

## 缓存与normalizer的可核完整性

- 输入只接受已失败的原126v1.1：status FAILED/exit1、原四臂和1000×4预算、0完成/0更新、原Errno24、无plan/initial/arm目录；source056与manifest内容同源。原失败目录、日志、norm、cache均只读，完整保留失败归属。
- 对T2100/V300的2400场逐一核原五文件集合（12000文件），四`.npy`头的dtype/shape/C顺序、payload准确字节长度；metadata严格为node_ids/sources/identities/intervals/joint_support_queries，核完整节点、唯一node ID、有限正span、Q64联合support计数。**不额外扫描约5.9GB的46列features，不重新计算DCT/22维/OLS或CSV；不以头检查冒称重新证明特征值的生成正确性。** 原T normalizer已在126完整扫描T后落盘；后续训练/评分仍有原finite检查，失配直接失败，无原CSV重建回退。
- T pairs/labels在普通载入交给原采样器前核全部i<j顺序、shape/dtype及identity-label exact；源排序不改。V同样核原pairs/labels，并用056少量必要字段node/source/identity/query/support重绑metadata，不读取target/target_valid/native_states或reference成员。
- 126 metadata没有直接保存query/support。非空场由原生intervals恢复原`linspace(min start,max end,64)`及区间support，与056 exact，并核联合support计数；空场只继承056的合法Q64，明确没有独立缓存query原件。nearest/exact/原物理states并未保存于这个cache，不伪造独立重核；其原126 canonical_all绑定来自已完成准备的冻结代码/原件。这些限制写入data-audit。
- 原`normalizers.json`直接读取后按相同值保存，核四臂schema、T pair总数、30维输入、原22字段顺序、FP32 cache/FP64 accumulation、四臂共同前22列、SUMMARY后8维均值0/尺度1；不重新拟合或重扫T大特征。新run的`private/cache`仅符号链接原126 cache，原文件不复制、不更写。

## 一次终点、代价和错误规则

新132独立run/Idea、plan1.0，另记录`scientific_idea_id=IDEA-ORDERED-SHAPE-SIGNAL-20261029`与`scientific_plan_version=1.1`。复用原`train_arm`，所以final checkpoint内部仍标原科学Idea126；这是明确的模型合同来源，不伪称132新模型结构。原126的0步失败和132的实际成功更新分开。

一张同FIFO兼容GPU/显式UUID（进程cuda:0）、CPU≤4、RSS8GiB，7200s advisory与原固定终点不变；CUDA不设8GiB硬AS。记录132所有缓存inventory/采样输入/计划、训练/评价/写盘成本，另列126原2949.129498s及前后累计墙时，不能把原49分钟准备成本隐藏成0。没有原ZIP/CSV或参考读取；T_test封存。private权重/坐标/身份/预测/cache留服务器，仅必要脱敏记录回传。

输入结构失配、Errno、非有限、OOM或代码失败立即保全新run，不自动重试/加步/换参数/回原CSV；正常终点后独立保存scores/训练记录审计，再Record→Squeeze→Mode C，才讨论输入问题。原科学解释出口完整沿126v1.1；此技术修复不提供任何预期正结果承诺。

## 实现、验收和严格审计适配

```text
python tools/analysis/ordered_shape_continuation.py \
  --failed-run <原126-attempt-1> \
  --manifest configs/data/mtad-level1-scene3000.json \
  --source-system <原056>/system --output <全新132-attempt-1> \
  --device cuda:0 --cpu-threads 4 --time-policy advisory
```

软件验收：原/新相同seed的真实小计划和曝光字典exact；RLIMIT_NOFILE=32下旧映射Errno24、新普通数组成功；缓存多字节/错误joint或norm/非0步旧run必须拒绝。它们只是软件证据，不是MTAD效果。没有本地模拟训练认证或额外预算。

126独立auditor只需**精确132分支**：run/summary接受132 Idea与plan1.0并强制scientific Idea126/1.1、failed_run为原0步FAILED126、cache软链指向原cache、normalizers与原件exact；其余四臂/4000更新/6209/所有指标和checkpoint保持原检查。final.pt仍严格Idea126，绝不能放宽成任意ID/version。旧126失败本身仍不能走COMPLETED审计，不能覆盖其失败文件。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-ORDERED-SHAPE-CONTINUATION-20261029
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.48
selected_C_id: IT-C4-resource
source_P_ids: [IT-P1-refined]
source_F_ids: [F227]
source_M_ids: [M98]
source_E_ids: [E125]
research_question: 有界文件句柄传输能否完成原有序形状四臂训练合同，不重新准备数据
hypothesis_and_mechanism: 普通ndarray载入消除4200个同时持有的映射句柄；属于技术修复，不预判输入科学增量
testable_predictions: 低FD旧失败新完成、原计划数值exact，真实132达到原固定终点且不改变人口
design_and_controls: 原126v1.1四臂/1000成功步每臂/6209/同实际批次；只替换sampling输入传输
sample_materials_or_experimental_units: 原只读T2100/V300五文件cache与完整T-only四臂normalizers
primary_outcome_and_estimand: 原科学三项ORDERED相对SUMMARY/DISTRIBUTION/SHUFFLED场景macro-F1差，技术完成与效果分开
secondary_outcomes: 全原raw/论文资格八指标/配对/组件/覆盖/训练量及全链成本
analysis_plan_and_uncertainty: 沿126v1.1，单seed重复开发V/共享身份；技术失败不算科学负结果
resources_time_cost_and_risk: 单GPU/CPU4/RSS8GiB/7200s advisory/4000更新；额外ordinary sampling RAM诚实记录
ethics_safety_privacy_or_compliance: 原126只读、不重CSV/reference/test；private留server
stopping_and_interpretation_rules: 固定原终点或真实故障保全，不自动后继/重试/重准备
status: implementation_verified_execution_only_by_HandOver
evidence_status: partial
missing_evidence: [real132_completion, actual_learned_increment, independent_generalization, credible_peer]
next_skill: record-experiment_after_actual_execution
```
