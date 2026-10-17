# CDF1 validation comparison

`cdf1_validation.py` reuses a **completed A0/CE/seed20 reference** and its
`T_head`/`V_select` score caches. It loads no dataset archive and never opens
`T_test`. Use the reference's original configuration: the loader checks completion,
actual configuration, role/branch/seed, model contents, and cache scenes/components,
candidate pairs and labels. Historical hashes may be absent or stale; they do not
gate reuse.

010 is complete. The CDF1 and TSA implementations are integrated into local main;
The formal [011 READY handover](../../HandOver/L2S-20261017-011.md) authorizes execution. The
following is an interface example:

```bash
python tools/experiments/cdf1_validation.py \
  --reference runs/FIRST-FIT-20261017-010-attempt-1 \
  --manifest runs/PREP-NATIVE-20261017-007-attempt-1/manifest.json \
  --config configs/first-fit-v2.json \
  --device cuda:0 --cpu-threads 4 \
  --output runs/CDF1-VALIDATION-unique-attempt-1
```

The new output contains `heads/` (existing training artifacts), `validation.json`,
`validation.md`, and full `V_select/<arm>/` score caches for the seven fitted or
rule-based head arms. A0 uses the original 010 cache. It trains all five learned arms using the unchanged head
schedule: 40 epochs for each delta `[0.5, 1, 2]`, at most 3 hours inside
`train_heads`; the handover must bound total time including artifact loading and
reporting. Existing output directories are refused; failures retain their files.

The table includes A0, calibrated A_prime, CDF1 B, EndpointMarginal, PairMLP,
ScenePool, FixedRule and B_shuffle. Each uses the same validation candidates,
with its own validation-selected threshold. It also reports descriptive B-minus-
control differences, corrected and newly wrong edges and their net change, plus
up to five actual corrected and five newly wrong A0-to-B examples in cache order.
These complete caches support later common-subset comparisons without training again.
A_full and independent literature baselines such as
TSA-cTFER are separate parallel jobs and are not included here. Results from
this selected validation set are exploratory, not final test performance or
proof of innovation. No model/data/metric code is changed by this entry point.

# TSA-cTFER offline adaptation

`tsactfer_fit.py` trains one independent dual-head contrastive encoder on T_core
and selects on the complete V_select. It does not open T_test or run the online
two-stage assignment. See [source correspondence and fixed adaptation rules](../../docs/tsactfer_adaptation.md)
and `configs/tsactfer-first-fit.json`. Execution requires the formal independent
L2S handover; this README is not a READY task. The authorized additional budget
is CDF1 3 hours + A_full 3 hours + TSA 6 hours, with at most two concurrent
single-GPU jobs, one attempt each and no retries. T_test remains sealed.
