# CDF1 validation comparison

`cdf1_validation.py` reuses a **completed A0/CE/seed20 reference** and its
`T_head`/`V_select` score caches. It loads no dataset archive and never opens
`T_test`. Use the reference's original configuration: the loader verifies its
entire hash, code fingerprint, split, checkpoint and cache bindings.

After the current 010 task has terminated, a separate authorized handover can run:

```bash
python tools/experiments/cdf1_validation.py \
  --reference runs/FIRST-FIT-20261017-010-attempt-1 \
  --manifest runs/PREP-NATIVE-20261017-007-attempt-1/manifest.json \
  --config configs/first-fit-v2.json \
  --device cuda:0 --cpu-threads 4 \
  --output runs/CDF1-VALIDATION-unique-attempt-1
```

The new output contains `heads/` (existing training artifacts), `validation.json`
and `validation.md`. It trains all five learned arms using the unchanged head
schedule: 40 epochs for each delta `[0.5, 1, 2]`, at most 3 hours inside
`train_heads`; the handover must bound total time including artifact loading and
reporting. Existing output directories are refused; failures retain their files.

The table includes A0, calibrated A_prime, CDF1 B, EndpointMarginal, PairMLP,
ScenePool, FixedRule and B_shuffle. Each uses the same validation candidates,
with its own validation-selected threshold. It also reports descriptive B-minus-
control differences. A_full and independent literature baselines such as
TSA-cTFER are separate parallel jobs and are not included here. Results from
this selected validation set are exploratory, not final test performance or
proof of innovation. No model/data/metric code is changed by this entry point.
