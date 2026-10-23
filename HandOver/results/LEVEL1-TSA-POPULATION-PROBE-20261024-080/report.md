# 080：TSA缓存人口对照

仅使用074D既有V评分与079既有T评分；零GPU、训练、前向或压缩卷读取。
固定原阈值，不重新关联/分组。T/V为相同抽样规则，不是逐样本配对，也非独立身份泛化。
固定先验0.5重加权AP只调整同一分数总体的类别权重，不能当部署AP或校准。

完整数字、逐场/组件与079平均margin分解见summary.json；抽样原对应见selected-views.json.gz。
判读须区分：同一V的平衡/完整人口差异、类先验的代数影响，以及同规则T/V仍有的组成/共享身份/设备差异。
079裁窗使用正例上下文，不能把裁窗分数直接当可部署新模型。

本项不会自动触发TSA重训；若无清楚的单因素假设，暂停该配方，继续078方法主线。

| Population | n+/n- | AUC | AP | AP at prior .5 | FPR at original threshold |
|---|---:|---:|---:|---:|---:|
| train_balanced_full / ITSA / pooled | 428/428 | 0.882424 | 0.861533 | 0.861533 | 0.196262 |
| train_balanced_full / MSTSA / pooled | 470/470 | 0.914015 | 0.888949 | 0.888949 | 0.165957 |
| validation_balanced_full / ITSA / pooled | 401/401 | 0.892836 | 0.859697 | 0.859697 | 0.147132 |
| validation_balanced_full / MSTSA / pooled | 435/435 | 0.934517 | 0.919567 | 0.919567 | 0.124138 |
| validation_full_candidates / pooled / all | 2624/583231 | 0.89106 | 0.0382286 | 0.862864 | 0.146839 |
| validation_full_candidates / pooled / itsa_all | 877/295529 | 0.872317 | 0.0150142 | 0.825666 | 0.188212 |
| validation_full_candidates / pooled / itsa_eligible | 875/27659 | 0.889959 | 0.212695 | 0.865851 | 0.176651 |
| validation_full_candidates / pooled / itsa_ineligible | 2/267870 | 0.914095 | 6.3999e-05 | 0.888968 | 0.189405 |
| validation_full_candidates / pooled / mstsa_all | 1747/287702 | 0.916863 | 0.0960094 | 0.902681 | 0.104341 |
| validation_full_candidates / pooled / mstsa_eligible | 1432/216663 | 0.93128 | 0.106546 | 0.913596 | 0.124124 |
| validation_full_candidates / pooled / mstsa_ineligible | 315/71039 | 0.867362 | 0.059222 | 0.848873 | 0.044004 |
