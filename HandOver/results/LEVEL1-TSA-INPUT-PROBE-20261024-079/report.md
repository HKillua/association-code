# 079 frozen T_core input qualification

One fixed final checkpoint; no fit, regrouping, threshold selection or V/test observations.

These are balanced T-only diagnostic populations. AP is not comparable to075 full-candidate AP.

| Head | Domain | Input | Positive / negative | AUC | AP |
|---|---|---|---:|---:|---:|
| ITSA | same_scene | full | 428 / 428 | 0.8824242291903223 | 0.861533224554736 |
| ITSA | same_scene | train_crop | 428 / 428 | 0.9113623460564242 | 0.8893295612555152 |
| ITSA | cross_scene | full | 428 / 428 | 0.9522501965237139 | 0.9346797211314133 |
| ITSA | cross_scene | train_crop | 428 / 428 | 0.9768320377325531 | 0.9710365522715866 |
| MSTSA | same_scene | full | 470 / 470 | 0.9140153915799004 | 0.8889493257187335 |
| MSTSA | same_scene | train_crop | 470 / 470 | 0.976283386147578 | 0.9675235819322171 |
| MSTSA | cross_scene | full | 470 / 470 | 0.9885061113626075 | 0.9873386446214228 |
| MSTSA | cross_scene | train_crop | 470 / 470 | 0.9977908555907651 | 0.9971838249480696 |

Training subset with balanced controlled negatives, not validation/full-candidate AP or independent generalization. Crops use positive context; cross-scene clocks are not aligned. No causal conclusion about original InfoNCE gradient difficulty.

Interpretation: crop/full differences are paired representation effects. Same/cross-scene differences may include geography, timing and candidate composition; they do not establish a shortcut or a causal training-negative failure. Empty or too-small eligibility subsets are non-identifiable, not successful. No follow-on training is automatically authorized by this report.
