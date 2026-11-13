# 05 — Results, days 1–5 (13 Nov 2026)

**Setup for every table:**
- Unless stated, results are on the **53 dev-val images** (1,619 GT fibers), scored as in [04](04-methods.md).
- CIs are 95% cluster-bootstrap over overlap groups. "Δ" means a paired difference.
- Raw outputs are in `results/day1` … `results/day5`, with metrics only (committed).
- Predictions are in `data/predictions/` (local).

## 1. Day 1: data plumbing

| Check | Expected | Measured |
|---|---|---|
| Images that load and validate | 266 | **266** |
| Orphan IDs → ignore | 47 | **47** |
| Topology-defect fibers → ignore | ~296 | **296** (116 images; 1.3% of all pixels) |
| Dev split | ~213/53, no group crossing | **213 / 53**; val median density 0.605 vs train 0.601 fibers/µm² |
| cv3 folds | — | 177/89, 177/89, 178/88 |

Files: `results/day1/` (`images.csv`, `cleaning_summary.json`).

## 2. Day 2: evaluator checks and the human ceiling

**Evaluator tests (all pass):**
- GT vs GT gives perfect scores.
- A 3-px shift gives HD95 = 3 px.
- One toy case per taxonomy class is classified as built.
- **Sensitivity:** eroding every axon by 1 px gives g-MAE between 0.002 and 0.01, the same size as the human noise floor.

**Registration bug in the EDA, and the leakage it hid:**
- The EDA's overlap script ran ECC in the wrong direction (it estimated B→A and used the result as A→B).
- So only 5 pairs looked well aligned, and the EDA wrongly blamed "misregistration".
- With the fix, 18 of 22 known pairs align (NCC ≥ 0.8).
- Re-scanning all 710 ORB candidates found **8 new overlapping pairs**:
  - 4 are confirmed by annotation agreement (axon Dice ≥ 0.98);
  - 4 are confirmed by eye.
- **5 of them crossed the v1 dev split.** Groups v2 (22 groups, 52 images) were built and the splits regenerated **before any training**. 39 of the 53 val images changed.
- v1 is kept in `results/day1/*_v1_superseded.json`.

**Two mistakes in the first scoring run (fixed):**
- It scored all 710 candidate pairs, not just real overlaps, giving spurious alignments.
- It ignored only A's label 1. A fiber cut by B's border then counted as a human "miss".

**Human-vs-human ceiling** (28 pairs at NCC ≥ 0.8; pixel metrics from the 17 pairs with ≥ 3 fibers). Files: `results/day2/ceilings_pooled.json`, `ceilings_pairs.csv`.

| Metric | Ceiling |
|---|---|
| Fiber F1@0.5 = F1@0.75 (TP 118, FP 7, FN 11) | **0.929** |
| Merges / splits between annotators | **0 / 0** |
| Axon / myelin Dice (median) | 0.987 / **0.951** |
| HD95 axon / myelin (median) | 9 nm / **644 nm** |
| g-MAE (118 fibers) | **0.0067** |
| Pairing correct / violation rate | 1.00 / 0.15 |

**What it means:**
- **Human disagreement is all selection** (which fibers get labelled). It is never separation.
- Myelin HD95 of 644 nm *between humans* shows that HD95 is driven by extra or missing rings, not outline placement.
- Caveat: only 129 fibers, in a different region than dev-val. It is an indication, not a paired comparison.

## 3. Day 3: pairing rule, AxonPath, timing

**R1, the pairing rule on perfect pixels** (all 266 images, 7,968 fibers; `results/day3/r1_*`):

| Metric | Value |
|---|---|
| F1@0.5 / F1@0.75 | **0.99994** / 0.99994 (TP 7,967, FN 1) |
| Merges / splits | 0 / 0 |
| g-MAE | **0.0017** (a quarter of the human floor) |

- The watershed line between touching rings lands close enough to the annotator's line every time. **The rule is not an error source.**
- So any separation error in a model must come from its *pixels*.

**GT violation baseline** (8,015 GT objects; checked on the raw GT):

| Violation | Count |
|---|---|
| Axon without ring | 4 |
| Ring without axon | 43 |
| Axon touching background | 181 |
| Ring Euler ≠ 0 | 334 |
| **Any** | **489 (6.1%)** |

Compare model violation rates with this, not with 0.

**AxonPath zero-shot** (dev-val):
- F1@0.5 **0.784** [0.746, 0.824]; Dice 0.71 / 0.62; g-MAE 0.060, biased +0.030.
- 42 merges / 19 splits; separation share **9%**.
- 767 predictions on label 1.
- Speed: 3.4 s per image on MPS.
- **Reading:**
  - Detection is fine (recall 0.88), but pixels are poor because of the scale and domain shift.
  - Its axon definition differs too (inner cylinder vs ours), even after mapping.
  - It is reported as "off-the-shelf", **not** as AxonPath's capability.
- **v1 bug:** using inner cylinders only dropped 25% of fibres. v2 falls back to AxonPath's axon instance, and 15% are still dropped.

**Timing run** (5-epoch trainer, L4, `2d_bigpatch`):

| | Measured |
|---|---|
| Epoch 0 (includes compile) | 398 s |
| Steady epoch | **142.5 s** |
| GPU utilisation | 97% (compute-bound, so more CPU cores wouldn't help) |
| GPU memory | 9.2 of 22 GB |

250 epochs would cost ≈ $12, which doesn't fit. **D-10 was revised to 150 epochs**, so the learning-rate schedule still anneals fully.

## 4. Day 4: B1, scores and oracles

**B1 training** (`data/nnunet/logs/modal_b1_launch.log`):

| Item | Value |
|---|---|
| Epochs | **150/150**. Early stopping never fired (longest stale stretch under the 0.001 rule: 19 epochs) |
| Best EMA pseudo-Dice | 0.874 at epoch 134 (final LR 1e-4); used for validation |
| Wall time / preemptions | 6.34 h / 0 |
| GPU utilisation | mean 94%, memory 9.2 GB |
| Cost | **$7.69** |
| nnU-Net's own validation Dice | axon 0.882, myelin 0.861 |

**Evaluator cross-check:**
- Our axon Dice for B1 (0.8822) equals nnU-Net's own (0.8823).
- Myelin is 0.8593 vs 0.8607, because nnU-Net also ignores the cleaned orphan and defect fibers.

**Scores** (files: `results/day4/summary_*.json`, `scores_*.csv`):

| Metric | Ceiling | AxonPath | Cellpose+B1 | **B1** |
|---|---|---|---|---|
| **F1@0.5** [CI] | 0.929 | 0.784 [0.746, 0.824] | 0.885 | **0.904 [0.881, 0.928]** |
| F1@0.75 | 0.929 | 0.732 | 0.869 | **0.881** |
| Recall / precision | — | 0.88 / 0.71 | **0.97** / 0.81 | 0.96 / **0.85** |
| TP / FP / FN | 118/7/11 | 1,429/597/190 | 1,569/359/50 | 1,559/271/60 |
| Predictions on label 1 (ignored) | — | 767 | 773 | 752 |
| Dice axon / myelin (mean) | 0.987 / 0.951 | 0.71 / 0.62 | 0.879 / 0.857 | **0.882 / 0.859** |
| HD95 myelin (nm) | 644 | 977 | 552 | 555 |
| **g-MAE** | 0.0067 | 0.060 | **0.0114** | 0.0160 |
| Merges / splits | 0 / 0 | 42 / 19 | 1 / 3 | 4 / 2 |
| Pairing correct | 1.00 | 0.80 | 0.88 | 0.87 |
| Violation rate | 0.15 | 0.39 | 0.50 | 0.37 |

**Paired differences** (`results/day5/paired_differences.json`):

| | ΔF1@0.5 | ΔF1@0.75 | Δg-MAE |
|---|---|---|---|
| B1 − AxonPath | +0.120 [0.096, 0.142] | +0.149 | −0.044 [−0.056, −0.033] |
| B1 − Cellpose+B1 | **+0.019** [0.012, 0.028] | +0.012 [0.003, 0.021] | **+0.0046** [0.0031, 0.0061] (B1 worse) |

**Reading:**
- B1 is within ~0.025 F1 of the human ceiling.
- **The same pixels grouped top-down (Cellpose+B1) give 29% lower g-MAE**, but 0.019 lower F1.
- **Why B1's Dice is below the plan's ≥ 0.95 / ≥ 0.88:** image-level Dice counts every predicted unlabelled fiber as FP pixels. Without background FPs (O-sel-broad) it is 0.948 / 0.908. So the patch size is not the cause, and D-24's fallback is not triggered.

**B1 error taxonomy** (331 errors = 271 FP + 60 FN):

| Class | n | Share |
|---|---|---|
| pred other_bg | 200 | 60% |
| pred unlabelled_real | 61 | 18% |
| pred low_iou | 10 | 3% |
| **pred gap / merge / split_piece** | **0 / 0 / 0** | 0% |
| GT missed | 46 | 14% |
| GT low_iou | 12 | 4% |
| **GT merged / split** | **2 / 0** | 0.6% |

**Oracles** (F1@0.5 gain [paired CI]; files `results/day4/oracles_*.json`):

| Baseline | O-sel | O-sel-broad | **O-sep** | O-bnd (F1@0.75) | Separation share |
|---|---|---|---|---|---|
| AxonPath | +0.019 | +0.136 | +0.015 [0.008, 0.022] | +0.055 | 9.0% |
| Cellpose+B1 | +0.018 | +0.091 | +0.000 [−0.001, 0.002] | +0.016 | 1.0% |
| **B1** | **+0.016** [0.009, 0.024] | **+0.074** [0.051, 0.095] | **−0.001** [−0.006, +0.003] | +0.023 [0.015, 0.034] | **0.6%** |

**g-MAE under the oracles** (B1):

| Condition | g-MAE |
|---|---|
| Base | 0.0160 |
| **O-sep** | **0.0093**; Δ −0.0067 [−0.0083, −0.0050], i.e. **−42%** |
| O-bnd | 0 (by construction) |

- O-sep gives 0.0093 for Cellpose+B1 too, as it must: same pixels, GT grouping.
- So, for B1's pixels:

| Source | g-MAE |
|---|---|
| Floor with perfect grouping | 0.0093 |
| Added by the watershed | +0.0067 |
| Added by Cellpose regions | +0.0021 |
| Human floor | 0.0067 |

**Reading:**
1. Separation costs B1 nothing in detection. The 896×1152 patch plus the distance watershed already separate touching fibers.
2. Selection dominates. O-sel-broad would put B1 at 0.978, *above* the human ceiling. That is a bound, since it trusts the single annotator about every skipped fiber.
3. Boundary placement decides the IoU-0.75 matches and the g-ratio.
4. **Grouping does matter for g-ratio** (−42% with perfect grouping). This was not anticipated.

## 5. Visual check: what B1's background "FPs" are

Gallery: 16 random `other_bg` fibers, `data/processed/day4/b1_other_bg_gallery.png` (local), made with `scripts/run/day4_fp_gallery.py --seed 0`.

| Judgement | n |
|---|---|
| Real myelinated fiber left unlabelled, mostly **thin myelin** | ~7 |
| Ragged or abnormal profile the protocol excludes; a judgement call | ~7 |
| True FP (fragment, sliver) | ~2 |

AxonPath's gallery showed the same split (~7 real / 3 FP / 2 ambiguous of 12).

**Consequences:**
- About half of the "FPs" are defensible fibers. Most of the rest are abnormal fibers that the protocol skips **on purpose**, which is a *learnable* rule B1 hasn't fully learned.
- The ring-candidate detector misses thin-myelin fibers, so strict O-sel understates selection.

## 6. Ring topology and g-ratio error (`results/day5/fiber_analysis_b1.json`)

Over B1's 1,559 matched fibers:

| | Value |
|---|---|
| Fibers whose predicted ring is broken or whose axon touches background | **28.1%** (ring Euler ≠ 0: 26.2%; axon touches bg: 11.1%) |
| g-MAE: defective / clean fibers | **0.032 / 0.0098** |
| Share of total g error on defective fibers | **56%** |
| Defect share by diameter quartile (smallest → largest) | 16% → 22% → 30% → **45%** |
| g-MAE by diameter quartile | 0.017 → 0.016 → 0.016 → 0.014 (flat) |
| Median IoU of matched fibers | 0.963 |

**Reading:**
- Broken predicted rings, mostly on large fibers, carry most of the g error.
- B1's clean fibers are near the human g floor.
- g error is **not** a small-fiber problem here.
- **This is correlation, not cause:** broken rings may sit on intrinsically hard fibers.

The per-fiber table `results/day5/fibers_b1.csv` is git-ignored, because it is a per-fiber derivative of the GT.

## 7. Day 5: the H1 decision (T-14, D-27)

| H1 condition | Needed | B1 | Met? |
|---|---|---|---|
| O-sep gain | ≥ +0.03 | −0.0014 [−0.0058, +0.0030] | No |
| O-sep gain ≥ O-sel gain | — | −0.001 vs +0.016 / +0.074 | No |
| Separation share | ≥ 30% | 0.6% | No |

**H1 = NO-GO**, unambiguously. AxonPath and Cellpose+B1 also fail it. Per D-15 the rule is applied as written.

The g-ratio finding (§4, §6) is recorded as **H5**. It is not used to overturn the decision, because it was not pre-registered.
