# TMTAFF-style association reference: conformance and explicit choices

2026-10-17 update: the historical epoch trainer below remains available unchanged. The opt-in `staged_updates_v2` trainer implements update-bounded TCM pretraining, frozen embedding+TCM TAM training, and a short joint stage, with class-balanced within-scene pairs. Frozen completion caches contain only TCM outputs, never learned TAM outputs. These are explicit new training choices described in [the v2 contract](specs/staged-training-v2.md), not newly discovered author-code details. FIRST-FIT-010 has completed on native data, and its full validation scores have been independently recomputed; COMPARE-011 has returned the TSA result while CDF1/A_full remain in progress. Dataset usage and EVIDENCE-009 have now been reviewed; configs/first-fit-v2.json defines the first bounded training-validation fit, dispatched only through HandOver.

This code is an independently implemented TCM + TAM **association reference**. It does not implement TFM, reproduce the authors' joint three-module training, or establish performance on the authors' unpublished preprocessing/split. The source examined is *Transformer-based Multi-target Track Association and Fusion Framework*, DOI [10.3390/app16147251](https://doi.org/10.3390/app16147251), especially PDF pp. 6–12 and 15–16. Source paper labels, official MTAD V4 labels, and this pilot's A–B identity protocol must not be conflated.

## Implemented correspondences

| Paper operation | Implementation |
| --- | --- |
| Z-score continuous attributes, positional vector before dimension expansion, Eq. 1–3 | The trainer fits feature normalization on training data only. `embed_inputs` adds sinusoidal positions in the four-dimensional attribute space, then a bias-free linear map and ReLU. No source, batch, identity, or reference state enters the model. |
| TCM Transformer encoder and decoder, six blocks, model width 128 and eight heads | Separate six-layer encoder/decoder with those production defaults; Transformer feed-forward width 256 (paper p. 9 says twice model width), GELU and post-norm. |
| Predict current embedded point from earlier observed points, Eq. 13 | Shifted next-embedding MSE, forward and reversed observed sequences. Both encoder attention and decoder self/cross attention are causal during teacher forcing. Targets are detached embeddings of the fragment's own observed measurements. |
| Complete before/after observed support and concatenate original embedded observations, Eq. 12 | Reverse embedded observed history for earlier endpoints; autoregress with learned decoder for later endpoints. Observed embeddings are preserved exactly. No arbitrary internal-gap replacement or GT/reference-assisted completion. |
| Shared Siamese Transformer TAM, separate from TCM | One additional Transformer encoder processes every completed track. Weights are shared across all tracks/pairs, not between TCM and TAM. |
| Full temporal feature comparison, Eq. 15–16 | `encode_tracks` returns `[N, length*d_model]`, **not a mean-pooled vector**. CE uses a bias-free linear map of absolute differences. Contrastive raw score is negative Euclidean distance over the entire temporal representation. |
| Contrastive and cross-entropy objectives | Both implemented. Sigmoid is fused into numerically stable BCE-with-logits. Contrastive loss is `mean(y*d^p + (1-y)*max(0,m-d)^2)/2`; `p=1` follows the printed Eq. 15, and margin defaults to an explicitly chosen 1.0. |

## Decisions not established by the paper

- The paper does not provide a complete decoder input, beginning token, attention cross-mask, or rollout recipe. Here, observed embeddings form decoder history; the last output predicts the next embedding, which is appended autoregressively. During inference the encoder sees the complete observed fragment, which is historical to every predicted endpoint. Forward/reverse sequences are not re-embedded after reversal. No teacher-forced future targets enter inference.
- The paper mentions zero-padding if a decoder fails to generate enough data. This implementation rolls out every required endpoint up to the configured length; it does not silently zero-pad a shorter prediction. Nonfinite predictions fail the caller's checks.
- Observed support must be contiguous on the data module's registered scene grid. Data registration/interpolation and short-support policy are recorded separately. A single observed grid point is retained and can condition completion; it contributes no next-point TCM training target. A track with no observed point or an internal missing mask is explicitly rejected.
- The paper's architecture sections state ReLU for input embedding and GELU for feed-forward blocks, while its training section refers generally to LeakyReLU. We use the specific architecture definitions. Transformer residual/attention dropout and final CE dropout are 0.1; exact author dropout placement is unavailable.
- Eq. 15 appears to use an unsquared positive Euclidean distance and a squared negative hinge, despite common contrastive implementations squaring both. The default follows the printed expression. `contrastive_positive_power=2` is available only as a separately recorded sensitivity; neither the margin nor this choice may be tuned on test data.
- The embedding layer is a learned module, but TCM-only pretraining should freeze it: detached targets prevent target-side gradients but do not alone prevent collapse of a jointly moving target space. Association-stage freezing/fine-tuning is an explicit trainer choice and is stored with the run. This is an implementation choice, not a claimed author detail.
- The pilot trainer's default association phase jointly updates input embedding, TCM and TAM with `association_loss + 0.1*completion_loss`. This prevents silently substituting an entirely fixed random input embedding, while remaining a separately specified **two-module** objective: the authors' joint TFM objective is absent. Frozen-TCM association is available only as an explicit configuration. Autoregressive gradients may increase memory/time considerably; throughput and complete-stage budget gates apply before interpreting results. The trainer's upper bound of 80 TCM epochs plus 80 association epochs resolves the earlier ambiguous stage allocation; it is not the paper's unqualified 80-epoch recipe, and the combined per-fit time cap still applies across both stages and caching.
- The paper's random truncation augmentation and joint TCM/TAM/TFM optimization are not implemented by this model module. Any trainer augmentation must be separately specified. Since TFM and three-module joint training are absent, any improvement is over this documented association reference, not a demonstrated win over the complete published system.
- Support groups are processed in batches; each endpoint rollout repeatedly calls the decoder and may be expensive for six-layer length-64 models. Benchmark complete TCM + TAM stages and caching before a server training budget is authorized. A smaller CPU test configuration validates software behavior only.

## Public API

`ModelConfig` is a serializable frozen dataclass. Save its values, branch, training stage configuration, normalization and state dict with every checkpoint. `AssociationModel(config, branch)` exposes:

- `embed_inputs(values, observed) -> [N,L,d_model]`
- `completion_loss(values, observed) -> scalar`
- `complete(values, observed) -> [N,L,d_model]`
- `encode_tracks(values, observed) -> [N,L*d_model]`, with `embedding_dim=L*d_model`
- `score_pairs(embeddings, pairs) -> [E]`
- `association_loss(scores, labels) -> scalar`
- `completion_parameters(include_embedding=False)` yields encoder/decoder parameters for completion pretraining; `input_embedding`, `tcm_encoder`, `tcm_decoder`, and `tam_encoder` are explicit submodules.
- `set_completion_trainable(bool, include_embedding=True)` controls gradient eligibility and frozen-TCM evaluation mode across subsequent `model.train()` calls. Checkpoints must reapply the stored stage choice; parameter `requires_grad` flags are not encoded in a bare state dict.

`values` is floating `[N,L,4]`, `observed` is boolean `[N,L]`, and `pairs` is integer `[E,2]`. Input normalization is external and must be training-fitted. Missing input positions may contain nonfinite sentinels, since they are explicitly excluded; observed positions must be finite. All unordered scene pairs, including same-source positives, are eligible under the pilot identity protocol. The model itself receives no labels except through its loss functions.

## Verification boundary

`tests/test_models.py` checks actual encoder/decoder gradients, shared TAM gradients for both objectives, causal teacher-forcing (including cross attention), exact observation preservation, bidirectional endpoints, missing-value exclusion, pair symmetry, short-support finite behavior and checkpoint roundtrip. Passing these tests does not establish scientific convergence, speed feasibility, author-code equivalence, or performance superiority.
