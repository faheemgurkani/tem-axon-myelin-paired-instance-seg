# Baseline: Cellpose-SAM zero-shot (+ B1 pixels = "cellpose_b1")

The planned fine-tuned Cellpose (B3, on filled fibers) did not fit the $10 GPU budget (D-26). What was run instead:

1. `zeroshot.py [--limit N]`: `cpsam_v2` zero-shot, local MPS, at ×1/6 (12.9 nm/px), so fibers fall inside Cellpose-SAM's 7.5–120 px training range.
   - Output: `data/predictions/cellpose_zs_instances/*.tif` (int32 regions).
   - Cellpose tiles the whole image into cell-like regions, including non-myelinated ones.
2. `scripts/run/day4_postprocess.py` splits each region with **B1's** semantics:
   - axon = the largest B1-axon component, holes filled;
   - ring = B1 myelin minus the axon;
   - regions without either are dropped.
   - Output: `data/predictions/cellpose_b1/`.

Its pixels equal B1's, so B1 vs Cellpose+B1 isolates **grouping**: bottom-up watershed vs top-down regions.
- F1: 0.885 vs 0.904.
- g-MAE: 0.0114 vs 0.0160 (Cellpose+B1 is better on g).
