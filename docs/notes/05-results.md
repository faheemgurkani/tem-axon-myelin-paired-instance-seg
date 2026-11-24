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

**Human-vs-human ceiling** *(to be called "repeat-annotation agreement" in papers until two annotators are confirmed, D-34)* (28 pairs at NCC ≥ 0.8; pixel metrics from the 17 pairs with ≥ 3 fibers). Files: `results/day2/ceilings_pooled.json`, `ceilings_pairs.csv`.

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

## 8. Phase 1 (15–16 Nov): C1 analyses, H5c development, provisional H7

All on the 53 dev-val images unless stated; CPU on the M4. Scripts: `scripts/run/phase1_*.py`; numbers: `results/phase1/`.

**Error breakdown for every model** (from `results/day4/summary_*.json`). Selection dominates for all three, not just nnU-Net:

| Model | F1@0.5 | Errors | Strict selection | **Broad selection** | Separation | Boundary | GT missed |
|---|---|---|---|---|---|---|---|
| B1 | 0.904 | 331 | 18.4% | **78.9%** | 0.6% | 6.6% | 13.9% |
| Cellpose+B1 | 0.885 | 409 | 17.4% | **80.7%** | 1.0% | 9.8% | 8.6% |
| AxonPath | 0.784 | 787 | 11.2% | **68.2%** | 9.0% | 10.5% | 12.2% |

**T-16′ C1 analyses** (`phase1_c1_analyses.py`, `c1_analyses.json`). Validated: B1 F1 0.904, g-MAE 0.0160 and myelin HD95 555 nm reproduce day 4 exactly.

| | B1 | Cellpose+B1 | AxonPath |
|---|---|---|---|
| g-MAE: equivalent circle (primary) [95% CI] | 0.0160 [0.0139, 0.0182] | **0.0114** [0.0101, 0.0128] | 0.060 |
| g-MAE: minor axis / Feret / perimeter | 0.021 / 0.027 / 0.051 | 0.014 / 0.015 / 0.075 | 0.067 / 0.065 / 0.090 |
| g bias: equivalent circle / perimeter | −0.0002 / −0.043 | +0.003 / −0.074 | +0.030 / +0.019 |
| g-quality, s = 0.10 (s = 0.05) | 0.766 (0.661) | **0.785** (0.696) | 0.396 (0.197) |
| Bland–Altman g: bias, LoA | −0.0002, ±0.057 | +0.003, ±0.035 | +0.030, ±0.16 |
| Bland–Altman D_F / D_A bias (nm) | −2.5 / −2.0 | −7.8 / −2.8 | +5.9 / +25.8 |
| g-MAE by fiber-diameter quartile (small → large) | 0.017 / 0.016 / 0.016 / 0.015 | 0.012 / 0.011 / 0.011 / 0.011 | 0.057 / 0.065 / 0.062 / 0.055 |
| Oblique GT fibers (eccentricity > 0.9) | 1.7%; g-MAE without them 0.0159 | 0.0114 | 0.059 |
| Myelin HD (nm): official / mean-P95 / HD100 | 555 / 327 / 1643 | 552 / 312 / 1659 | 977 / 561 / 1992 |

**Reading:**
- **Perimeter-based g is unsafe** for comparing methods: large negative bias from rough outlines, and it flips the B1 vs Cellpose+B1 order. Minor axis and Feret keep the order.
- **Rank stability:** the order is stable on 9 of 11 metrics. Perimeter g and HD100 swap B1 and Cellpose+B1; AxonPath is always last.
- **Gaming curve** (`c1_gaming_curves.csv`):
  - dropping B1's *smallest* fibers barely helps (g-MAE 0.0160 → 0.0142 at half recall), so g-MAE isn't gameable by size here;
  - dropping the **worst 5%** gives 0.0116 (−27%, Cellpose level) for −5 points of recall, and the worst 20% gives 0.0073;
  - so recall must always sit next to g-MAE (D-08). g-quality combines both.
- g error is flat across fiber size (binned by fiber diameter, as D-32 requires). Oblique fibers are too rare to matter.

**H5c development, DEV ONLY** (`phase1_h5c_dev.py`, `h5c_dev_summary.json`). Target: ≥ 50% of O-sep's g gain (0.0160 → 0.0093), with ΔF1 and Δrecall ≥ −0.005.

| Variant | F1@0.5 | Recall | g-MAE | Share of O-sep gain |
|---|---|---|---|---|
| V0, current D-11 (−EDT of foreground) | 0.9040 | 0.9629 | 0.0160 | 0% |
| **V1, nearest axon (EDT from axons)** | **0.9048** | **0.9660** | **0.0124** | **54%** |
| V2, axon-radius-weighted (axomyelin prior) | 0.9035 | 0.9660 | 0.0134 | 39% |
| V3, V2 + ring closing | 0.9035 | 0.9660 | 0.0134 | 39% |

- **V1 frozen** as `pairing.semantic_to_paired_instances_h5c` at commit **7c673ca**.
- The 54% is in-sample (dev chose it) and is expected to shrink. Confirmation is H5c on the 213 non-dev T-21 images.

**18-fiber addendum** (`expert_review_addendum.py`): reproduced the repeat-annotation disagreements exactly (11 A-only + 7 B-only = 18). The blind packet is in `data/processed/expert_review_addendum/` (local only).

**T-23 (H7), PROVISIONAL** (`phase1_h7_protocol_bias.py`, `h7_provisional.json`):
- **Outer-ring tracer validation on 7,967 labelled fibers:**
  - v1: bias −0.051, LoA ±0.079;
  - v2 (dark band opened with a 5-px disk): bias −0.025, LoA ±0.088, MAE 0.037;
  - **fails the pre-set gate** (|bias| ≤ 0.02, LoA ≤ 0.08), so **H7a and H7c are not reportable from automatic tracing**. They need expert or hand tracing. Iteration was stopped after one principled revision, to avoid tuning to the gate.
- **Detector recall on labelled fibers: 65%** (the ring detector finds about 2 in 3 real fibers as dark-ringed holes).
- **H7b, provisional: the protocol under-counts myelinated fibers by 6.5% [5.5%, 7.5%]**, which would pass H7b (CI lower bound ≥ 5%). It assumes:
  - the 83% real share of candidates (Claude's single rating);
  - the 65% recall measured on normal fibers.
  - **Confirmatory only with the D-33 expert ratings.**
- Mean-g shift with the (biased) traced g: about −0.005, tiny, as the arithmetic caveat predicted.

### 8.1 T-25 image fingerprint (16 Nov; analysis-only proxy, D-37)
Script `scripts/run/phase1_t25_fingerprint.py`; outputs `results/phase1/t25_*`.
- **Features** (labelled fibers, physical units, same pipeline on both datasets): fiber density, myelin volume fraction, median and p90 g, share with g > 0.85, median fiber diameter. **The unlabelled-candidate rate is deliberately left out:** it is H7's outcome, so clustering on it would make the H7 stratification circular.
- **Ours: two weak clusters, not distinct conditions.**
  - Silhouette 0.31 at k = 2, falling for k = 3–5. Overlap-group bootstrap ARI: median 0.96, 5th percentile 0.62.
  - Cluster A (116 images): dense, thin g (density 81 /100 µm², median g 0.730, 0.8% with g > 0.85, fibers 724 nm).
  - Cluster B (150 images): sparse, higher g, larger fibers (49 /100 µm², 0.776, 6.2%, 819 nm).
  - B also has higher contrast and noise, so the split is partly acquisition, not only tissue. Two overlap groups straddle the clusters.
- **TEM1 sanity check:** the same features per mouse with k = 3 reproduce the Phase 0b proxy exactly (ARI 1.0). This only shows the pipeline is consistent, because the features overlap. The test mouse 26 stays in the dense, control-like cluster.
- **H7 by cluster (provisional, same assumptions as §8 H7b):**

  | Cluster | Candidates per 100 µm² background | Candidates per labelled fiber | H7b count bias |
  |---|---|---|---|
  | A, dense | 4.3 [3.3, 5.4] | 0.033 | **4.0% [3.0, 5.0]** |
  | B, sparse / high g | 5.5 [4.6, 6.6] | 0.079 | **9.2% [7.7, 10.9]** |

  - Per background area the skip rate is similar (A/B rate ratio 0.78 [0.58, 1.03]), matching the area-controlled result in the AG2 check (§6 of that note).
  - **But the count bias is more than twice as large in the sparse, high-g images**, because a similar number of skipped fibers is set against fewer labelled ones. That is the stratum where a fiber count matters most for disease.
  - Pooled 6.5% hides this: H7b would pass in B and fail in A (CI lower bound 3.0% < 5%).
  - **Wording:** still not "the protocol masks disease". Cluster B is a proxy, partly acquisition, and the real share is a single non-expert rating. Confirmatory after D-33, stratified by these clusters (D-37).

## 9. Phase 2 (16 Nov): evaluator validation on TEM1 (T-17) and H6
Script `scripts/run/phase2_t17_h6.py`; numbers `results/phase2/`. TEM1 test mouse `sub-nyuMouse26` (8 images, 503 fibers), ASTIH's released TEM1 model.

| | Dice axon | Dice myelin | F1@0.5 | Precision | Recall |
|---|---|---|---|---|---|
| ASTIH published | 0.974 | 0.924 | 0.791 | 0.747 | — |
| Ours, ASTIH scheme as their script runs (row seeds) | 0.9744 | 0.9243 | **0.792** | **0.748** | 0.851 |
| Ours, ASTIH scheme with point seeds | 0.9744 | 0.9243 | 0.905 | 0.844 | 0.996 |
| Ours, paired-fiber evaluator (challenge convention) | | | **0.979** | 0.963 | 0.996 |

- **The evaluator is validated (D-38).** The published F1 is reproduced to 0.001, but only through a seed-placement artefact in ASTIH's evaluation script: (1, H, W) masks make each centroid seed a whole image row.
- The remaining gap to 0.979 is mostly 1-px axon specks, which ASTIH counts as instances and a paired-fiber evaluator doesn't.
- **H6a holds at the pre-registered setting (fragile):** strict selection share 0 of 21 errors, broad 7 of 21 = 33%. At the ds 3 sensitivity, strict = 9.5%, which fails.
- **H6b holds:** the top FP class is split_piece (11 of 19), mostly speck debris.
- Compared with our data (B1: 331 errors, broad 78.9%), the TEM1 model makes very few errors on near-complete labels, and almost none of them are selection.
- **TEM2 (21 ROIs, 10 mice, 292 fibers; released 5-fold model):** Dice 0.9717 / 0.9294 (published 0.971 / 0.929); F1 as-run 0.947 (published 0.949), point seeds 0.960, our paired evaluator 0.976. **H6c holds:** strict 0/14, broad 1/14 = 7% (border fibers scored), stable across detector settings; border-ignored 0/3 and 1/3.
- **B1 zero-shot on the outside sets** (T-22; 2.152 nm rescaled; D-11 pairing): F1 0.930 (TEM1), 0.931 (TEM2 full), 0.946 (TEM2 test), 0.947 (AxonPath).
  - **Broad selection share 3.6% / 10.7% / 9.7% / 3.4%, against 78.9% on our dev images:** same model, different labels.
  - Thin-fiber (g > 0.85) recall gap: −0.186 (TEM1), −0.254 (TEM2 full, 35 thin), +0.064 (TEM2 test, 9 thin), −0.177 (AxonPath).
  - On the 8 held-out TEM1 mice: −0.196 [−0.274, −0.155]. H8 waits for the reference model.

### 9.x H8 decided: the thin-fiber gap is a label effect, not an intrinsic difficulty (18 Nov)

- **Reference model:** Dataset502 (TEM1, near-complete labels, 12 training mice), fold 0 = the 8 held-out mice (64 images), `nnUNetTrainer_150epochs_ckpt10` (fixed 150 epochs, D-39), same ResEnc-M 2d_bigpatch recipe as B1. Validation Dice: axon 0.962, myelin 0.892.
- **Same 3,029 GT fibers (454 thin, g > 0.85), same D-11 pairing, same evaluator:**

| Model | Recall thin | Recall normal | Gap |
|---|---|---|---|
| B1 (trained on our selective labels, zero-shot) | 0.751 | 0.947 | −0.196 |
| TEM1 reference (complete labels) | 0.969 | 0.987 | −0.018 [−0.029, −0.006] |

- **gap_B1 − gap_ref = −0.178, mouse-bootstrap 95% CI [−0.258, −0.138] (8 mice). H8 holds** (rule: ≤ −0.10 and CI upper < 0).
- The reference still misses thin fibers slightly more (−0.018), so thin fibers are a little harder, but about 90% of B1's gap is not explained by difficulty.
- **Confound to state in the paper:** B1 is out of domain on TEM1 (different lab and scanner); the reference is in domain. Domain shift could hurt thin fibers more than normal ones. The gap-of-gaps controls for overall accuracy (B1 normal recall 0.947 is still high), not for an interaction of shift with thinness. T-24 (C3) is the cleaner test: same data, labels made selective vs complete.
- Reference taxonomy on the held-out mice: F1 0.971, 180 errors, strict 0.6%, broad 38% (small error count; most are other_bg).
- Files: `results/phase2/h8_ref.json`, `h8_compare.json`, `h8_images_ref.csv`, `tem1ref_logs/`.

## 10. Phase 3a (19 Nov): H5 confirmation and D-03 3-fold C1 numbers

Script `scripts/run/phase3a_h5.py`; outputs `results/phase3a/` (`h5_summary.json`, `c1_cv3_summary.json`, per-image and per-fiber CSVs). Input: T-21 out-of-fold B1 semantics (each image predicted by the cv3 fold model that did not train on it). Base = D-11 pairing; H5c = frozen rule (7c673ca). CIs: paired cluster bootstrap over overlap groups, 1,000 resamples.

### 10.1 H5 on the 213 non-dev images (189 overlap groups; 6,142 fibers matched in base)

| | Base | H5c (frozen) | O-sep |
|---|---|---|---|
| g-MAE (fiber-weighted) | 0.0161 | 0.0137 | 0.0076 |
| F1@0.5 | 0.8929 | 0.8935 | 0.8958 |
| Recall | 0.9674 | 0.9696 | 0.9737 |
| Fibers with \|g error\| > 0.03 | 13.3% | 9.6% | 2.9% |

| Hypothesis | Rule | Result | Verdict |
|---|---|---|---|
| **H5a** | O-sep cuts g-MAE ≥ 25% relative, CI excludes 0 | **−52.7%**; Δ −0.0085 [−0.0096, −0.0073] (dev was −42%) | **Holds** |
| **H5b** | Defective ≤ 35% of fibers, ≥ 40% of \|g error\|, ≥ 50% of O-sep's reduction | 25.4% of fibers · 49.0% of error · 61.5% of reduction | **Holds** |
| **H5c** | Captures ≥ 50% of O-sep's gain, Δg-MAE CI excludes 0, ΔF1 and Δrecall lower CI ≥ −0.005 | **27.7%** captured; Δg-MAE −0.0023 [−0.0030, −0.0017]; ΔF1 +0.0007 [−0.0007, +0.0024]; Δrecall +0.0022 [+0.0008, +0.0039] | **Fails** (capture only; all guards pass) |

- **Gate 3a outcome (§7.2 table): H5a holds, H5c fails → "C2 for morphometry becomes a candidate (new budget decision)."** This is the owner's call (budget, D-36).
- **Grouping matters more for g than for detection:** separation is 3.5% of detection errors (C1 below) but grouping explains 53% of g-ratio error. The paper may say **"ring topology"** (H5b holds), not only "grouping".
- **H5c's dev estimate shrank as expected:** 54% in-sample → 28% out of sample. The frozen rule still gives a real, free improvement (g-MAE −15%, fibers with \|g err\| > 0.03 from 13.3% to 9.6%) with no detection cost, but it does not reach the pre-registered bar.
- H5b sensitivity: every base-matched fiber is also matched under O-sep; 40 fibers are matched only under O-sep (\|g error\| sum 1.84), outside the rule.
- Dev images (53) were not used for H5.

### 10.2 D-03 3-fold C1 numbers (all 266 out-of-fold images)

| Scope | F1@0.5 | Errors | Strict selection | Broad selection | Separation | O-sel broad ΔF1 | O-sep ΔF1 |
|---|---|---|---|---|---|---|---|
| All 266 | 0.892 | 1,866 | 16.0% | **79.5%** | 3.8% | +0.084 [0.074, 0.094] | +0.003 [0.002, 0.005] |
| Non-dev 213 | 0.893 | 1,474 | 15.5% | 80.3% | 3.5% | +0.084 [0.073, 0.095] | +0.003 [0.001, 0.005] |
| Dev 53 (cv3 model) | 0.888 | 392 | 17.6% | 76.8% | 4.8% | +0.083 [0.060, 0.106] | +0.005 [0.001, 0.009] |

- **C1's dev headline replicates on 3-fold:** broad selection 79.5% (dev, B1: 78.9%); separation stays small (3.8% vs dev's 0.6% on B1, a little higher with the cv3 models); O-sel broad gains +0.084 F1 against +0.003 for O-sep.

### 10.3 Exploratory: H5 measures on AxonPath CC TEM (context only, not part of the rule)

B1 zero-shot (out of domain), 20 images, 1,836 matched fibers, hand-drawn pairs, image bootstrap (`phase3a_h5_axonpath.py`, `h5_axonpath_summary.json`).

| | Base | H5c | O-sep |
|---|---|---|---|
| g-MAE | 0.0293 | 0.0283 | 0.0221 |

- O-sep cuts g-MAE by **24.6% [21.1, 28.9]**: grouping still matters out of domain, but about half as much as on our data (52.7%), right at the 25% bar.
- H5c captures **13% [−4, 29]** of that gain: the frozen rule barely helps out of domain.
- Defects are less concentrated: defective fibers are 35.6% of matched fibers and carry 39.3% of \|g error\| and 40.1% of O-sep's reduction (all below the H5b bars). Out of domain, g error is spread across boundary placement as well as topology, which fits B1's higher base g-MAE there (0.029 vs 0.016).


## 11. Phase 3b (24 Nov): T-24 TEM1 controlled omission pilot and H9

Benchmark (D-35/D-42/D-43): ASTIH TEM1, 12 training mice with simulated two-part omission (27% of training fibers removed: 1,081 from the protocol stratum g > 0.85 / g < 0.65 / ecc > 0.9 with p = 0.9, 230 at random), 8 held-out mice with full labels. Four arms with the same nnU-Net ResEnc-M `2d_bigpatch` recipe for 80 epochs: R1 naive, R2 hard ignore on 1,169 mined candidates, R3 the same ignore budget at random positions, R4 selection-gated soft weights (w = 1 − t·(1 − a)). Scored with `scripts/run/t24_score.py`, which was written before training (pairing D-11, IoU > 0.5, mouse bootstrap with 1,000 resamples).

| Arm | Normal recall | Violation (stratum found) | Hard-normal recall | Protocol F1 | g-MAE |
|---|---|---|---|---|---|
| R1 naive | 0.961 | 0.667 | 0.952 | 0.841 | 0.028 |
| R2 hard ignore | 0.987 | 0.855 | 0.977 | 0.832 | 0.023 |
| R3 random ignore | 0.971 | 0.707 | 0.950 | 0.841 | 0.029 |
| R4 selection-gated soft | 0.973 | 0.744 | 0.964 | 0.839 | 0.026 |
| Reference (complete labels) | 0.995 | 0.958 | 0.988 | 0.819 | 0.021 |

**H9 fails.**
- R4 − R2 = −0.013, CI [−0.020, −0.005].
- R4 − R3 = +0.003, CI [−0.002, +0.007].
- Violation R4 − R1 = +0.076, above the +0.05 guard.
- Hard-normal recall R4 − R2: CI [−0.026, +0.003].

Reading: recall and violations move together along one axis, which is how much candidate background is removed from the background loss. Hard ignore recovers recall but also re-learns the excluded fibers. The selection model (â AUC 0.74) is too weak to separate the two. Caveats: one seed per arm; R2 was trained on an A10G, the others on an L4.

### 11.1 Why R4 lost: audit (24 Nov; exploratory, H9 stands)

- **Code and data:** no bug. The loss equals nnU-Net's at weight 1, and zero-weight pixels get no gradient. The weights survive augmentation, and the preprocessed weight channel equals the d map. Simulated labels are exact. Candidate footprints cover deleted fibers well: mean pixel coverage is 0.92 for random deletions and 0.75 for stratum deletions.
- **What R4 did:** its pixel-weighted mean weight was 0.73 on missed normals and 0.76 on stratum fibers, so it was a near-uniform down-weight. Frontier excess (recall above the R1–R2 line at the arm's own violation): R4 +0.002 [−0.003, +0.009], R3 +0.005 [−0.004, +0.015].
- **Causes:**
  1. The selection model never used the 3,278 labelled-fiber instances as "normal" examples (PU, D-44).
  2. It was linear, so it could not express the two-sided band (D-45).
  3. H9's recall-only comparison against hard ignore was nearly unwinnable.
  4. There was one run per arm.
- **Selection model after the fixes**, scored on the non-expert candidates (decisions used only expert-subset information):

  | Variant | â AUC | Weight, missed normals | Weight, stratum |
  |---|---|---|---|
  | Round 1 (deployed) | 0.73 | 0.73 | 0.76 |
  | PU boosted trees, band terms, uncalibrated | 0.87 | 0.32 | 0.75 |
  | PU boosted trees, band terms, calibrated (R4b) | 0.87 | 0.68 | 0.89 |

  The R4h gate (top 161 candidates by q) ignores 49% of missed-normal pixels and 6% of stratum pixels, with a precision of 0.66.
- **Violation floor:** stratum fibers near a cut are almost always found. The share found ranges from 0.80 within 0.02 of a cut to 0.51 deeper than 0.05 for R1, and from 0.99 to 0.90 for the reference. R1 still finds 83% of oblique stratum fibers. A segmenter cannot learn a global-threshold protocol, so the endpoint is a trade-off.
- **Feature ceiling:** B1's features track the truth closely: correlation with true g 0.92, with true eccentricity 0.93. With the true g and eccentricity, â AUC would reach 0.95. 38% of real unlabelled fibers lie within 0.03 of a g cut-off.

### 11.2 Round 2 and H10 (pre-registered 24 Nov; 07-plan §7.7)

**Result: H10a, H10b and H10c all fail** (24 Nov; `results/phase3b/t24_scores.json`). Same 64 test images (8 mice), 2,179 normal and 850 stratum test fibers. All four runs completed 80 epochs on A10G without restart; tranche D cost $13.88.

| Arm | What it is | Normal recall | Violation | Hard-normal recall | g-MAE | Frontier excess E [95% CI] |
|---|---|---|---|---|---|---|
| R1 | naive (round 1) | 0.961 | 0.667 | 0.952 | 0.028 | 0 (line end) |
| R1b | R1 replicate | 0.963 | 0.698 | 0.950 | 0.028 | −0.002 [−0.005, +0.001] |
| R2 | ignore all candidates (round 1) | 0.987 | 0.855 | 0.977 | 0.023 | 0 (line end) |
| R3 | random ignore (round 1) | 0.971 | 0.707 | 0.950 | 0.029 | +0.005 [−0.004, +0.015] |
| R4 | round-1 soft weights | 0.973 | 0.744 | 0.964 | 0.026 | +0.002 [−0.003, +0.009] |
| **R6** | **oracle: ignore exactly the missed normals** | 0.965 | 0.684 | 0.953 | 0.025 | **+0.002 [−0.005, +0.010]** |
| R4b | PU-corrected soft weights | 0.969 | 0.716 | 0.957 | 0.026 | +0.001 [−0.007, +0.009] |
| R4h | PU-corrected hard gate (top 161) | 0.962 | 0.686 | 0.950 | 0.027 | −0.001 [−0.012, +0.011] |
| ref | complete labels | 0.995 | 0.958 | 0.988 | 0.021 | — |

- Noise N = |E(R1b)| = 0.002. E − E(R3): R4b [−0.012, +0.003], R4h [−0.015, +0.003]. With R1b as the line's low end every E rises by ≈ 0.002; no verdict changes.
- **H10a fails:** even the oracle does not beat the R1–R2 line. Pre-registered consequence: C3 as a methods paper is dropped; T-24 is a benchmark result for C1 (07-plan D-48).
- **The pipeline was checked before reading this:** R6's training labels carry exactly the 23.2M oracle ignore pixels, and they survive preprocessing (label 3: 23.9M in Dataset507 vs 0.73M in 503).
- **What it means (exploratory):** ignoring exactly the randomly missed normal fibers recovers only +0.004 recall (≈ noise), while ignoring every candidate (R2) recovers +0.026. The recall R1 loses is therefore caused mainly by the *protocol-excluded* (stratum) fibers taught as background, not by the 7% random misses, to which nnU-Net is nearly robust. A selection model that separates missed normals from exclusions has nothing to recover here. This revises our reading since D-35 and supports H8's hidden-stratification finding in C1.
- **Noise:** R1 vs R1b differ by 0.031 in violation, so single-run violation differences of ≈ 0.03 are noise.
- **Limits:** one seed per arm; 7% omission and the stratum rule are our simulation choices; a higher random-omission rate was not tested.

## 12. Reviewer-lens checks for the C1 paper (25 Nov; T-28, D-49)

**Scripts:** `scripts/run/paper_cis.py`, `paper_taxonomy_extras.py`, `paper_h4_tem1.py`, `paper_fig_qualitative.py`.
- **Outputs:** `results/paper/{cis,taxonomy_sensitivity,protocol_realism,h4_tem1}.json` and `h4_tem1_images.csv`.
- **Compute:** CPU only, plus local MPS for the TEM1 predictions; $0.

### 12.1 Confidence intervals (G4)

**B1's broad selection share** (cluster bootstrap, 1,000 resamples):

| Data | Broad selection share [95% CI] | Clusters |
|---|---|---|
| Ours, 3-fold | 79.5% [76.2, 82.8] | 236 groups |
| TEM1 | 3.6% [2.4, 4.7] | 20 mice |
| TEM2 | 10.7% [7.4, 15.7] | 10 images |
| TEM2 test | 9.7% [0, 22.9] | 21 images |
| AxonPath | 3.4% [0.9, 6.9] | 20 images |

- **Strict selection share** on ours: 16.0% [13.9, 18.3].
- **Separation share** on ours: 3.8% [2.8, 4.7].
- **Separation share is larger on the outside sets:** TEM1 10.5%, TEM2 23.6%. With few selection errors, the other causes make up a larger share of a much smaller total.

**Benchmark, normal recall / violation per arm** (mouse bootstrap):

| Arm | Normal recall [95% CI] | Violation [95% CI] |
|---|---|---|
| R1 | 0.961 [0.947, 0.975] | 0.667 [0.614, 0.720] |
| R2 | 0.987 [0.980, 0.993] | 0.855 [0.811, 0.885] |
| R6 | 0.965 [0.951, 0.978] | 0.684 [0.629, 0.737] |

- **R6 − R1 recall:** +0.005 [−0.002, +0.010]. The paper text said +0.004; it is now +0.005, rounded from 0.0046.
- **R1b − R1 violation:** +0.031 [0.016, 0.046]. The replicate differs "significantly" by the mouse bootstrap, so **the bootstrap over test mice does not capture training noise.** Single-run gaps of ≈ 0.03 are not evidence.

**Visual check:** 7 of 16 *other background* fibers are real. Wilson 95% CI: 23–67%.

### 12.2 Taxonomy sensitivity (G5; 266 out-of-fold images, 1,866 errors)

| Setting | Strict selection | Broad selection | Separation |
|---|---|---|---|
| Ring-detector ds ∈ {3, 4, 5} × min hole diameter ∈ {200, 300, 400} nm | 13.1–17.1% | 79.5% | 3.8% |
| Default (ds 4, 300 nm) | 16.0% | 79.5% | 3.8% |
| IoU 0.75 (2,230 errors) | 13.4% | 66.5% | 4.8% |

- The broad share and the separation share do not depend on the detector.
- **The headline is robust.**

### 12.3 Protocol realism (G3)

Share of fibers in the simulated T-24 stratum (g > 0.85 or g < 0.65 or ecc > 0.9) on our data:

| Fibers | Count | In stratum | g > 0.85 | ecc > 0.9 |
|---|---|---|---|---|
| Labelled GT | 7,968 | 9.4% | 3.3% | 1.6% |
| B1 matched | 7,699 | 11.6% | — | — |
| B1 *unlabelled real* | 298 | 25.5% | 11.1% | 7.7% |
| B1 *other background* | 1,186 | 47.6% | 24.5% | 21.2% |

- The real protocol leaves out the kind of fibers the simulation deletes: unlabelled predictions are enriched 2.7–5×.
- The *other background* class includes true false positives with irregular shapes, so its figure is an upper bound.

### 12.4 H4 (registered 25 Nov, `07-plan` §7.8, `dae7a63`): **fails**

64 held-out TEM1 images, 8 mice:

| Model | F1@0.5 | g-MAE | GQ₀.₁ | Broad selection | Separation |
|---|---|---|---|---|---|
| B1 | 0.945 | 0.0296 | 0.680 | 3.1% | 12.4% |
| Cellpose+B1 | 0.942 | **0.0252** | **0.713** | 1.2% | 3.2% |
| AxonPath | **0.961** | 0.0405 | 0.599 | 24.9% | 28.3% |
| Reference (not in the rule) | 0.971 | 0.0267 | 0.725 | — | — |

**F1 order reverses: AxonPath comes first**, after ranking last on our dev split (0.784).
- F1 B1 − Cellpose+B1: +0.003 [0.000, 0.007].
- F1 Cellpose+B1 − AxonPath: −0.019 [−0.027, −0.013].

**The g-MAE order reproduces, with every CI excluding 0:**
- B1 − Cellpose+B1: +0.0044 [0.0034, 0.0051].
- AxonPath − Cellpose+B1: +0.0153 [0.0091, 0.0213].

**Exploratory reading:**
- AxonPath's training labels exclude only degenerating fibers. On our selective labels its extra detections count as false positives; on TEM1's complete labels they count as hits.
- So **F1 rankings partly measure agreement with the training protocol.** That supports the paper's thesis.
- GQ₀.₁ ranks Cellpose+B1 > B1 > AxonPath on both datasets.
- **Caveat:** AxonPath's training-data overlap with TEM1 is not documented.

### 12.5 Qualitative figure (G7)

- Only TEM1 crops are shown (CC-BY): 99 held-out thin fibers (g > 0.85) are missed by B1 and found by the reference.
- The figure shows one crop each from mice 34, 26 and 31.
- No challenge images are used.
