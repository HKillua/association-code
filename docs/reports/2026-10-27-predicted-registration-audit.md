# 固定预测对应下的相对配准条件：实际记录与深挖

2026-10-27。**三场的相对源差结构不依赖先用真实身份筛对；预测组留出和query前后半检查均保留明显结构。但这没有证明绝对位置恢复改善，也没有选定最终方法。** 3035有错配组和恶化组，1043留出只有两组，均保留在结果中。

上游为事先写入的[Idea 1.0](../proposals/predicted_registration_audit_20261027.md)（C10，Mode C 1.6a），输入为103已返回的三份公开`points-*.json.gz`。完整数值、每组结果、运行时间、版本和命令保存在[原数值JSON](2026-10-27-predicted-registration-audit.json)；不覆盖103，不重新打开原始ZIP、NPZ或权重。

## 1. EXPERIMENT_RECORD：实际执行

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-PREDICTED-REGISTRATION-20261027
source_idea_id: IDEA-PREDICTED-REGISTRATION-20261027
plan_version: 1.0
execution_status: completed-exploratory
research_question_and_hypothesis: predicted-input correspondences may preserve relative source displacement across groups and query periods
planned_vs_actual_design_and_controls: fixed node-component split; group-balanced coordinate median; zero and group-mean controls; postfit oracle sensitivity
actual_configuration: three returned 103 public tables; Q64; early 0-31; late 32-63; no training
samples_materials_or_data_provenance: 103 canonical_all input-derived differences in three postselected V scenes
all_runs_or_repetitions: one successful numerical execution; no numerical retries
evidence_locations: tools/analysis/audit_predicted_registration.py; paired JSON in this directory
known_limits_and_missing_evidence: development scenes; dependent groups/queries; offline interpolation; absolute gauge and motion-direction diversity unknown
next_routes: squeeze-experiment then Mode C
next_skill: squeeze-experiment
```

真实命令，cwd为主仓：

```bash
.venv/bin/python tools/analysis/audit_predicted_registration.py \
  --source HandOver/results/LEVEL1-THREE-SCENE-DIAG-20261027-103 \
  --output docs/reports/2026-10-27-predicted-registration-audit.json
```

执行UTC 09:09:59.547888–09:09:59.857948，脚本计时0.309716s，退出0；macOS 15.3.2 arm64、Python 3.12.14、NumPy 2.5.4，峰值RSS 148275200 bytes（141.406MiB）。0 optimizer update、0模型前向、0 ZIP读取、未打开T_test；一个本地CPU进程，不把墙时称为GPU成本或单核利用率。

根代理另从原public `query_pairs`独立重建三场中心和全时留出组等权残差，没有复用本脚本函数；中心在1e−10容差内一致，留出残差分别629.06280117431/244.72350408425459/621.3166850694843m，与JSON相同。这是同一派生结果的核验，不是第二实验或独立重复。

**全部软件操作与偏差：** 最初用系统`python -m unittest ...`因缺NumPy而未导入脚本；没有读取研究数据。改用主仓既有`.venv/bin/python`，6项检查通过，随后上面唯一一次真实分析成功。成功后代码审查增加RUNNING/FAILED记录与禁止覆盖检查，并补第7项软件测试；没有改变提取/拟合/统计、重算或覆盖既有数值。最后7项必要测试通过（0.027s），包括真值/标签字段扰动主结果不变、共享node组件隔离、组等权、缺早期数据不回退、保留错配、方向错误/重复行停止、失败记录及禁止覆盖。现有数值JSON由加失败记录前的同数值算法生成，故没有后来新增的`interpreter`字段；解释器由真实命令及版本记录定位。合成测试只验证软件性质，不作为MTAD效果证据。

## 2. 信息权限、人口和估计对象

103的`query_pairs.delta_NE_m`直接由`local_delta(motion[b], motion[a], lat0)`产生；`lat0`是可见canonical量测纬度均值。它不是从两份真值误差中相减重建，坐标尺度也不依赖reference。源方向固定9002−9001、轴序[N,E]、单位m。

脚本先投影到固定组、node/source、query索引/时刻、输入支撑及直接差向量，拟合函数看不到oracle身份或reference误差。额外oracle-only pairs通过`fixed_group_index is None`排除，不能混入预测对应。对所有预测跨源组先构node连接组件，再按组件最小group index的排序rank%3==0留出。三场无组间node复用，组件均为单组；组内可能有多对node。随后才限制为双方有输入时间支撑。单源组、无共同支撑的pair-query仍登记，原任务人口不改变。没有依据误差或身份去重、挑健康组。

| scene | 全预测组 | 跨源组／有共同支撑组 | 无跨源组 | 所有跨源pair-query | 双方支撑／未共同支撑 | 拟合／留出组件 |
|---|---:|---:|---:|---:|---:|---:|
| 3035 | 76 | 47／47 | 29 | 7744 | 2463／5281 | 31／16 |
| 2819 | 17 | 11／11 | 6 | 1472 | 426／1046 | 7／4 |
| 1043 | 7 | 6／6 | 1 | 704 | 268／436 | 4／2 |

冻结拟合后读取的重复组标记显示，三场共有10/4/1个重复预测组，其中跨源重复组1/0/0个；全部保留，没有因重复目标可能相关而删除。相同真实目标仍可能出现在多个不同node/组，场内源偏差也共享，**组件隔离不等于独立样本或独立测试集**。

对每个场景，设每组全部纳入pair-query差为\(d_{gi}\)，先取\(\bar d_g=n_g^{-1}\sum_i d_{gi}\)，主中心为拟合组\(\bar d_g\)的逐坐标中位数\(b\)。组等权均值中心只作预定敏感性对照。主残差为\(G^{-1}\sum_g n_g^{-1}\sum_i\|d_{gi}-b\|_2\)，零中心为基线；不是组均值向量的范数，也不是按全部query直接加权。JSON另保留逐点及组级mean/median/P95/min/max和每组计数。

## 3. 主要结果：全时拟合组→全时留出组

| scene | 冻结中心[N,E] m | 拟合组／pair-query | 留出组／pair-query | 留出零中心残差→冻结中心残差 m | 组等权mean中心敏感性 m |
|---|---|---:|---:|---:|---:|
| 3035 | [−2384.086, −2259.449] | 31／1619 | 16／844 | 3179.501→629.063 | 635.119 |
| 2819 | [−2857.611, −2684.556] | 7／257 | 4／169 | 3937.242→244.724 | 247.322 |
| 1043 | [−1870.385, −1735.956] | 4／157 | 2／111 | 2903.963→621.317 | 620.122 |

最后一列是另一预定估计器在相同留出数据的残差，未据结果切换主估计。它与主估计方向一致，不能声称“稳健中位数机制已胜出”。拟合组零中心→主中心残差分别3301.224→400.742、3962.273→302.678、2565.018→255.250m。

| scene | 留出组残差均值的范围 m | 留出逐点残差median／P95／max m | 组均残差恶化数 |
|---|---|---|---:|
| 3035 | 219.279–3154.045 | 300.276／3712.714／5187.729 | 2／16 |
| 2819 | 232.478–261.234 | 231.404／475.910／841.773 | 0／4 |
| 1043 | 300.293–942.340 | 288.590／1514.517／16743.555 | 0／2 |

3035组0/3的组均残差分别增加435.456/368.484m；没有从主表删除。1043虽组均都下降，仍有16.744km单点残差。报告这些范围和尾部，不能用数千pair-query算独立样本显著性；只有三场、每场留出16/4/2组，未给总体发生率、置信区间或泛化结论。

## 4. 固定前半query拟合→后半检查

另一个中心只在拟合组件的query 0–31上估计，32–63不参与该中心。其与全时中心的距离分别8.599/17.026/18.740m。所有时间分层如下；各格为“组数／pair-query数；零中心→早期冻结中心残差m”。

| scene | 拟合组早半 | 拟合组晚半 | 留出组早半 | 留出组晚半 |
|---|---|---|---|---|
| 3035 | 29／845；3269.520→281.959 | 28／774；3275.558→458.683 | 15／419；3145.209→477.836 | 16／425；3186.705→765.065 |
| 2819 | 7／150；3914.377→247.443 | 5／107；4021.290→343.960 | 4／104；3926.730→245.410 | 3／65；3956.431→246.738 |
| 1043 | 4／87；2561.882→252.733 | 3／70；2568.625→240.507 | 2／59；2930.046→664.706 | 2／52；2869.123→562.487 |

早半无共同支撑的组仍在固定分组名单中，不能换组补齐。早期组均值mean中心下，晚半留出残差分别761.914/246.756/563.381m，与主估计同方向；全部其他敏感性格保存在JSON，未选最好一格。

3035晚半留出组0/3分别增加1112.595/1878.432m，其余下降；后两场该分层无组均恶化。3035晚半逐点P95为4262.117m，1043为1479.065m。**这是离线输入插值后的query差跨时段检查，不是严格在线的早时训练—晚时预测。** canonical插值可使用完整可见窗的相邻原事件，chart也基于全场可见输入。它没有验证因果时序权限、真实航向多样性或新场景迁移。

## 5. 估计冻结后的oracle敏感性

| scene | 纳入不同身份pair-query／全部 | 涉及不同身份组 | 同拟合集oracle中心与主中心差m | 留出同身份组残差m | 留出不同身份组残差m |
|---|---:|---:|---:|---:|---:|
| 3035 | 497／2463（20.179%） | 6 | 0.290 | 269.714（14组／628对时刻） | 1791.041（4组／216对时刻） |
| 2819 | 0／426 | 0 | 0 | 244.724（4／169） | 不可估：无此类 |
| 1043 | 0／268 | 0 | 0 | 621.317（2／111） | 不可估：无此类 |

3035是100对有共同支撑的不同node pair中的22对当前标签不一致，不能将497个pair-query称为497个独立错配。其拟合集中有2个含错配组、281个错配pair-query；留出有4个含错配组。一个预测组可以同时包含同身份与异身份pair，所以14+4不应等于16。所有6个错配组与45个有同身份pair组亦有重叠。

oracle中心只删除同一预先拟合集内的异身份pair后重新计算，拟合组数仍31/7/4。它不引入原预测组外的oracle对应。0.290/0/0m的差值支持这三个固定样本中主中心不是依赖真值清洗才出现；后两场没有错配，不能用零差值声称对错配鲁棒。当前身份标签正确与否也不由本实验验证。

## 6. SQUEEZE_REPORT与可回流证据

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-PREDICTED-REGISTRATION-20261027
input_experiment_record_ids: [EXP-PREDICTED-REGISTRATION-20261027]
claim_scope: input-only relative displacement stability in three returned, postselected V scenes
data_quality_and_provenance_gaps: no absolute raw coordinates; original identity correctness not verified; offline derived queries
primary_result_with_uncertainty: all three scene heldout means decrease; 3035 contains worsening groups; 1043 has only two heldout groups
process_dynamics: fixed early-query estimator remains similar; not a causal forecast evaluation
mechanism_and_intermediate_findings: relative structure persists without oracle selection; no proof of bias cause or absolute recovery
errors_heterogeneity_and_boundary_conditions: mismatches increase residuals; unsupported and single-source populations not covered; extreme residuals remain
robustness_sensitivity_and_alternative_explanations: group-mean center and coordinate-median center agree in direction; shared input interpolation and source geometry remain alternatives
new_P: none; existing P02/P05 refined
new_F: candidate predicted-correspondence relative structure with group/time limits
new_M: none verified; M07/M05 conditions refined
evidence_index: paired JSON scenes.*.source_pairs[0]; sections 2-5
open_questions: absolute gauge; motion diversity; full-V prevalence; real reference-chain cause; out-of-scene generalization
next_routes: Mode C increment then candidate comparison, no automatic training
next_skill: deep-read Mode C
```

| 类型 | 本次证据/推断 | 能支持的边界 |
|---|---|---|
| DE（候选新F） | 不读oracle标签的预测组中心，在三场组留出及前后半query检查均降低平均相对差残差；各结果/恶化组见§3–5 | “只能靠oracle对应才有结构”在这三个样本下不成立；不能上升为全V/新场景规律 |
| DE（异质性） | 3035含497个错配pair-query且有恶化组；1043余留极大单点 | 不能把单一常量平移当作所有组/时刻的完整解释或直接修复 |
| AI（M07细化） | 来源间共享几何偏差、时间采样/插值、类似运动或生成机制均可能产生结构 | 需要区分相对结构与真实传感器偏置；未识别唯一根因 |
| UN | 绝对共同零点、真实运动方向多样性、全V效果、可靠锚点、在线条件 | 相对残差下降不推出位置MAE下降，更不解释全部100km级误差 |
| E | 本记录、原数值JSON、只读脚本，来自同一103输入，非独立重复 | 可作为C10部署条件的新证据，但不增加论文数或独立实验样本量 |

候选去留建议：C10获得有限的输入可部署性证据，应与C08/C09在同一候选表重新比较；本记录不把配准升格为最终方法，也不自动发布训练。若后续进入恢复实验，必须先决定可解释的绝对锚点/对称性处理、统一完整人口和普通控制，并把相对一致性与正式球面位置收益分别评价。三案例原链104仍是另一项未决证据问题，不能用本次相对结果替它结案。
