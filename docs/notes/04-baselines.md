# 04 — Baselines

> **Updated plan (2026-11-11):** see the report's "Five baselines, one shared pairing step" section. It adds **B0**, a reproduction of ASTIH's TEM1 numbers
> (nnU-Net Dice 0.974/0.924, fiber F1@0.5 0.791; Cellpose 0.882) to validate our evaluator, plus a B1∪B3 consensus run and an **oracle-pairing** diagnostic.
> Cellpose and μSAM must be trained on **filled fibers**, never on rings. Use nnU-Net ResEnc-**L**, and run the optional B5 as affinities + mutex watershed.

**Rule:** every baseline gets the **same pairing post-processing** and is scored with **our replica of the official
metric** ([05](05-evaluation-framework.md)) on the **same image-level split**. Time-box this phase.

| # | Baseline | Output type | How it becomes paired instances | Code | Status |
|---|---|---|---|---|---|
| B1 | **nnU-Net v2 (ResEnc-M/L), 2D**, labels {bg, axon, myelin, inter-fiber boundary}; artifact label → ignore | Semantic | Marker-controlled watershed: seeds = axon components; flood myelin within a distance cap; drop rings without a seed and axons without a ring | MIC-DKFZ/nnUNet | **The bar to beat** |
| B2 | **AxonDeepSeg generalist + ASTIH TEM models**, zero-shot and fine-tuned | Semantic | Same pairing as B1 | axondeepseg/axondeepseg, axondeepseg/ASTIH | Tests domain shift |
| B3 | **Cellpose-SAM (cpsam_v2) / cpdino**, fine-tuned on *fiber* (axon ∪ myelin) instances | Instance (outer) | Axon = nnU-Net axon head ∩ each fiber | MouseLand/cellpose | Instance-native |
| B4 | **μSAM EM generalist**, automatic instance segmentation, fine-tuned | Instance | As B3 | computational-cell-analytics/micro-sam | Foundation model |
| B5 | *(optional)* **StarDist dual-radius** or **HoVer-Net-style distance heads** | Explicit geometry | Nesting by construction | stardist/stardist | Bridge to our method |

## Challenge-mode recipe (if we ever submit)

- B1 with a 5-fold ensemble + TTA (flips/rot90) or a boundary-loss variant, whichever is better on HD95.
- Rule-based post-processing:
  - fill axon holes
  - enforce one ring per axon
  - remove small or unmyelinated candidates (ring-coverage ratio, as in gACSON's 70% rule)
  - handle border-cut fibers the way the GT does
- Write `(2k, 2k+1)` IDs to **uint16** TIFF/PNG.
- Because g-MAE only counts matched fibers, tune the detection threshold jointly over all three ranks, not on Dice alone.

## Per-baseline notes (fill in as we run them)

- B1:
- B2:
- B3:
- B4:
- B5:
