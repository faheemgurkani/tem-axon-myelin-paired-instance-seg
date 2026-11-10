# 05 — Evaluation Framework

Build this **before any training**. The concrete default and switch for every ambiguity are in the report's evaluator table and are mirrored in `configs/default.yaml`
(IoU > 0.5 matching; MONAI-style max of directed P95 for HD95; one-empty HD95 capped at the 10,936 nm diagonal; g-MAE always reported with fiber recall).
The g-ratio sensitivity is Δg/g = p/r per 1-px interface shift: +0.015 for a 0.2 µm axon, +0.003 for a 1 µm axon. It lives in `src/axon_myelin_seg/evaluation/`.

## Tier 1: official metric replica (`official.py`, `ranking.py`)

Implement exactly as in [01](01-challenge-spec.md):
- Mask GT label `1` everywhere.
- Validate pairing: every even `X` has odd `X+1` and vice versa.
- Dice for axon (even ≥ 2), myelin (odd ≥ 3) and their mean.
- HD95 in nm (× 2.152) for the axon and myelin boundaries.
- g-ratio MAE over fibers matched by IoU on `X ∪ X+1`, with equivalent-circle diameters from `regionprops`.
- Rank aggregation over (DSC_Mean ↑, HD95_Mean ↓, g-MAE ↓), with ties broken by HD95.

Each ambiguity (per-image vs pooled Dice; global vs per-instance HD95; IoU threshold; empty-mask HD95; border fibers)
is a **config switch**. Report the main numbers under one fixed setting, plus a sensitivity table across settings.
*Metrics Reloaded* pitfalls P3.1/P3.2 are exactly this problem. If the organizers release reference code, mirror it.

## Tier 2: instance diagnostics the official metrics hide (`diagnostics.py`)

- F1 at IoU 0.5 / 0.75; PQ = SQ × RQ; AJI (fiber level).
- Merge and split counts.
- **Pairing-violation rate:**
  - axon without a ring
  - ring without an axon, or with more than one axon
  - axon pixels touching background
  - myelin component with β1 ≠ 1
- **Recall-adjusted g-MAE:** g-MAE alongside fiber recall, because the official g-MAE ignores missed fibers.
- Unmyelinated-axon false positives, probed with ASTIH TEM2's unmyelinated labels.

## Tier 3: morphometry (`morphometry.py`)

- Per-fiber g-ratio, axon diameter, myelin thickness and fiber diameter.
- **Bland–Altman** plots (bias, limits of agreement) against GT.
- Errors stratified by **axon-diameter bins**, since small axons dominate g-ratio sensitivity.
- Distribution-level agreement: the g vs diameter scatter that neuroscientists actually use.

## Protocol

- **Split by image, and by animal or block if metadata allows** (`data/splits.py`). Never split by tile. Use grouped k-fold, and
  bootstrap CIs over images or animals.
- **Score full 4096×3008 images after stitching**, not tiles. Seam and border errors concentrate there.
- **External test sets** for generalization claims: ASTIH TEM1/TEM2 and AimSeg GT ([07](07-datasets.md)), after checking for overlap.
- Freeze the split file (`configs/splits/*.json`) and commit it. All baselines use it.
