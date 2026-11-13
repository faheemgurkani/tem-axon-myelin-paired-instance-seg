# Baseline: AxonPath zero-shot

AxonPath (paucabar/axonpath @ 6d633bf; bundled `em` weights) is a three-head U-Net: semantics + fibre distance + axon distance, paired by a ≥ 90% overlap rule.

`run.py [--split val|all]`, local, on the M4's MPS GPU, ~3.4 s per image:
1. Downscale ×0.269 (2.152 → 8 nm/px, the model's training scale) and normalise percentiles 1–99%.
2. `run_inference(min_diameter=20)`, then nearest-neighbour upsample.
3. Our axon = AxonPath's **inner cylinder** (axon + inner tongue, matching our GT's convention). If a fibre has none, use its axon instance; fibres with neither are dropped (15%). Myelin = fibre − axon.
4. Output: `data/predictions/axonpath/*.tif`; log `results/day3/axonpath_runlog.csv`.

Results (dev-val): F1@0.5 0.784, Dice 0.71 / 0.62, g-MAE 0.060. These are *off-the-shelf* numbers at another lab's scale and convention, not AxonPath's capability.
