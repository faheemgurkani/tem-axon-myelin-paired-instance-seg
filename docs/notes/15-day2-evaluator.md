# 15 — Day 2: the evaluator and annotation ceilings (T-04 to T-07), executed 13 Nov 2026

Contribution C1 rests on this code. Every later number goes through it, so its rules are written out in full here.

| Module | Role |
|---|---|
| `src/axon_myelin_seg/evaluation/official.py` | T-04: official-metric replica |
| `evaluation/diagnostics.py` | T-05: F1, PQ, merge/split, pairing correctness, violations |
| `evaluation/taxonomy.py` | T-06: why each fiber error happened |
| `evaluation/runner.py` | All of the above in one call (`score_image`) |
| `evaluation/stats.py` | Cluster bootstrap over overlap groups (D-22) |
| `scripts/run/day2_ceilings.py` | T-07 |

Tests are in `tests/test_evaluator.py`; the toy cases are described below.

## Background for a reader new to the problem

- Each image is a cross-section through brain white matter. Nerve fibers are cut across, so each looks like a light disc (the **axon**) inside a dark ring (the **myelin** sheath). One axon plus its ring is a **fiber**.
- The **g-ratio** is axon diameter ÷ fiber diameter, typically about 0.6–0.8. It is the biological quantity scientists want per fiber, because myelin thins in disease.
- The challenge mask stores axon *X* (even) and its ring *X+1*. A prediction is only useful for g-ratio if each axon is paired with **its own** ring. That is why we measure *pairing*, not just pixels.
- **Label 1** marks regions the annotators chose not to score: fibers cut by the image border, and some large or oblique fibers. Anything predicted there is neither right nor wrong.

## T-04 Official replica — `official.py`

Decisions (all in the module docstring and the plan's D-05–D-08):

| Point | Rule | Why |
|---|---|---|
| Label 1 | Removed from *both* maps before every metric (D-05) | The spec says label 1 is ignored |
| What is a fiber | Even X ≥ 2 **with** X+1 present. Unpaired IDs count for Dice/HD95 but cannot match | Orphan IDs exist in the GT (47) |
| Matching | One-to-one at fiber IoU > 0.5 on X ∪ X+1. Above 0.5 a match is unique, so greedy and Hungarian agree | The spec says "matched by IoU" without giving the threshold or algorithm |
| g | √(A_axon / (A_axon + A_myelin)), the equivalent-circle diameter ratio (D-06) | Spec |
| HD95 | Per image and per compartment: the max of the two directed 95th percentiles of boundary distances × 2.152 nm. One side empty → image diagonal 10,936 nm; both empty → NaN (D-07) | The spec doesn't say which HD95 variant; this is the common one |
| g-MAE | Reported with recall and the matched count (D-08) | g-MAE ignores missed fibers, so it can be gamed |

**Fiber-index trick:** for any label L ≥ 2, `L // 2` is the same for axon X and myelin X+1. So `labels // 2` is a fiber-instance map, and a single `np.unique` over (GT fiber, predicted fiber) pairs gives every intersection at once. That is fast enough to score a 4096×3008 image in a few seconds on the M4.

**Tests:**
- GT vs GT: Dice 1, HD95 0, g-MAE 0, recall and precision 1, F1 1, PQ 1.
- HD95 of a 3-px shifted square is exactly 3.
- **Sensitivity check** (plan expectation "1 px changes g by ~0.005"): on sample_0001, moving every axon-boundary pixel into its own myelin (erosion by 1 px) gives a g-MAE in (0.002, 0.01) with a negative bias. **Matches the expectation.** A single pixel of boundary placement is therefore the same size as the annotation noise floor (0.006–0.007).

## T-05 Diagnostics — `diagnostics.py`

- **F1@0.5, F1@0.75, PQ = SQ × RQ.** Predicted fibers with ≥ 50% of their area on GT label 1 are *ignored*: not TP, not FP.
- **Merge:** a predicted fiber covers ≥ 20% of each of ≥ 2 GT fibers. **Split:** ≥ 2 predicted fibers each have ≥ 20% of their *own* area inside one GT fiber. Both are counted over all fibers, matched or not.
- **Pairing correct** (per matched fiber), all three of:
  - the predicted axon is one 8-connected component;
  - it doesn't touch background or another fiber (enclosed by its own ring);
  - it has IoU ≥ 0.5 with the GT axon.
- **Violations** (per predicted object): any of axon without ring, ring without axon, axon touching background, or ring Euler ≠ 0.

**Bug found and fixed (day 3, before any model result):**
- The violation checks first ran on the prediction *after* GT label-1 pixels were set to 0. So any predicted axon next to an ignored region "touched background".
- Violations describe the prediction itself, so they now run on the **raw** prediction.
- Also, `axon_touches_other` no longer counts label 1 as "another fiber".
- The T-07 ceiling numbers in `results/day2/` were computed *before* this fix. Only their `violation_rate` and `pairing_correct_rate` are affected; they are re-run in §T-07b.

## T-06 Taxonomy — `taxonomy.py`

Every unmatched predicted fiber gets exactly one class, first matching rule wins:

1. **gap.** The predicted axon is ≥ 50% on GT background, and the 5-px band around it is ≥ 75% GT myelin from ≥ 2 different GT fibers. This is the light triangle between three touching sheaths. It looks like an axon in a ring, but the "ring" belongs to the neighbours.
2. **unlabelled_real.** The predicted axon is ≥ 50% on GT background and overlaps a *ring candidate* by ≥ 50% of the smaller area. A ring candidate is an unlabelled dark closed band around a hole, d ≥ 300 nm (same detector as `scripts/eda/unl_loc.py`). These are real fibers the annotators skipped.
3. **merge**: as in T-05.
4. **split_piece**: ≥ 20% of its area lies inside a GT fiber that a *different* prediction matched.
5. **low_iou**: ≥ 20% of its area lies inside a GT fiber that nothing matched.
6. **other_bg**: < 20% of its area on any GT fiber. Not a gap, no ring candidate.
7. **other.**

Unmatched GT fibers are classified as:
- **missed:** predicted fibers cover < 10% of it;
- **merged;**
- **split;**
- **low_iou.**

**Error families, used by the oracles and H1:**

| Family | Classes |
|---|---|
| Separation | pred gap, merge, split_piece; GT merged, split |
| Selection | pred unlabelled_real (broad version: + other_bg) |
| Boundary/delineation | pred low_iou; GT low_iou |
| Detection | GT missed |

**Easy to miss:**
- The **ring-candidate detector uses the GT** to pick its intensity threshold: half-way between the GT myelin median and the GT background median.
  - This is legitimate for an *evaluator*. It is never used for a prediction.
  - Its recall is unknown: the EDA's manual review found ~1 real unlabelled fiber per image, and the detector catches only part of them. That is why "other_bg" exists and why O-sel is also reported in a broad version.
- **Precedence matters.** A gap prediction usually overlaps neighbours' myelin by ≥ 20%. Checked after merge, it would be called a merge. Gap is therefore checked first.

**Tests** (toy maps in `tests/test_evaluator.py`) build one example of each class and check it is labelled as constructed:
- merge of two touching GT fibers;
- a GT fiber cut into two halves (the larger half matches, the other is a split_piece);
- a gap between three touching discs;
- an unlabelled fiber over a candidate hole;
- a fiber on background with no candidate;
- a prediction inside label 1, which is ignored, not FP.

## T-07 Annotation ceilings — `scripts/run/day2_ceilings.py`

**Idea:** some training tiles overlap, so the same tissue was annotated twice. Scoring one annotation against the other, after registration, gives the score of a second human. A model can't be expected to beat that.

### A registration bug in the EDA

`scripts/eda/overlap_agree.py` (EDA, 12 Nov) behaved like this:
- It **initialised ECC with the A→B transform** but called `cv2.findTransformECC(template=B, input=A, …)`. That call estimates the opposite direction (B→A).
- It then **used the result as A→B**.
- Every pair it "refined" was therefore registered in the wrong direction. That is why the EDA saw NCC 0.2–0.6 on refined pairs and kept only 5 unrefined pairs with NCC > 0.8.
- The EDA concluded the other pairs were "misregistered" (doc 11 called the 8% disagreement an upper estimate). In fact they were mis-*computed*.

**Fix:**
- Template = A, input = B, so the warp maps A → B, matching the initialisation. Then warp B into A with `WARP_INVERSE_MAP`.
- Keep ORB or ECC, whichever has the higher NCC.
- **Result:** ECC now improves NCC on every pair, e.g. 0018/0050 goes from 0.37 to 0.875. **18 of the 22 EDA pairs reach NCC ≥ 0.8** (was 5).
- The 4 below 0.8 (0118/0178, 0042/0163, 0016/0248, 0189/0247) have 23–267 ORB inliers and are surely the same tissue. Their NCC stays at 0.67–0.73, probably because contrast differs between acquisitions. They stay in the groups but are not used for ceilings.

### New overlaps → the v1 split leaked (fixed)

**A first-run mistake, worth recording:** the first run scored *all 710* ORB candidate pairs from `data/processed/eda/overlap.csv`, not just the 22 with ≥ 20 inliers. ECC then produced spurious alignments on unrelated tiles, and the summary was invalid (mean axon Dice 0.57). The 710-pair scan was kept as `results/day2/registration_scan_all710.csv`, and it turned out to be useful: 8 pairs below the inlier cut reached NCC ≥ 0.82 after correct ECC.

**How the 8 were checked:**
- 4 are confirmed by independent annotation agreement (axon Dice ≥ 0.98): 0123/0181, 0018/0238, 0148/0218, and 0014/0110 (the last was already linked through 0256).
- 4 have almost no labelled fibers in the overlap, so they were checked by eye (`data/processed/day2/new_overlap_check.png`). All are unmistakably identical tissue.

Five crossed the v1 dev split, so groups and splits were regenerated before any training (doc [14](14-day1-data-plumbing.md), "Revision").

### Second fairness bug in the first scoring run

**Bug:** the first run ignored only *A's* label 1. But a fiber cut by **B's** tile border is label 1 in B and a whole fiber in A, so it scored as a human "miss". That is a tiling artifact, not annotator disagreement. Mean axon Dice came out at ~0.6.

**Fix:** pixels that are label 1 in *either* annotation are ignored, as in the EDA.

### Results (corrected) — `results/day2/ceilings_pairs.csv`, `ceilings_pooled.json`

28 pairs reach NCC ≥ 0.8. 17 of them have ≥ 3 GT fibers in the overlap, which is enough for pixel metrics.

| Metric | Human vs human (ceiling) | Plan expectation |
|---|---|---|
| Axon Dice, median over 17 pairs (IQR) | **0.987** (0.983–0.992) | 0.98–0.99 ✓ |
| Myelin Dice, median (IQR) | **0.951** (0.921–0.957) | 0.90–0.97 ✓ |
| HD95 axon, median | 8.6 nm (= 4 px) | — |
| HD95 myelin, median | 644 nm | — |
| g-MAE, fiber-weighted (118 matched fibers) | **0.0067**, bias −0.0005 | ≈ 0.006 ✓ |
| **Fiber F1@0.5, pooled** (TP 118, FP 7, FN 11) | **0.929** | — |
| Fiber F1@0.75, pooled | **0.929**, identical | — |
| Merges / splits between annotators | **0 / 0** | — |
| Pairing correct, matched fibers | 0.983 (before the T-05 fix) | — |

**The headline finding for H1:**
- **When two humans disagree about a fiber, it is always selection** (labelled vs not labelled). It is never separation (no merges, no splits). Every fiber both labelled is matched at IoU > 0.75.
- So the human ceiling is ~0.93 F1 because of *which* fibers get labelled, not *how* they are grouped. A model trained on these labels inherits ~7% selection noise that no architecture can remove.
- That pushes toward the plan's NO-GO expectation. The decisive number is still the B1 decomposition (day 4).

**Easy to miss:**
- **HD95 myelin is large (644 nm) even between humans.** One unmatched or extra ring in an image puts that ring's whole boundary at ~0.5–1 µm from anything. HD95 is therefore dominated by *selection* disagreements, not boundary placement. Read HD95 differences between methods with this in mind.
- The ceiling pixel metrics come from small overlaps (8–45% of a tile), with only 129 GT fibers pooled. CIs are wide, and day 4 adds bootstrap intervals.
- The ceiling uses a *different region* (the overlaps) than the dev-val evaluation. It is an indication, not a paired comparison.

### T-07b

The ceilings are re-run after the T-05 violation fix (only `violation_rate` and `pairing_correct_rate` change). The result is recorded in doc 16.

## Status

- [x] T-04
- [x] T-05 (fixed)
- [x] T-06
- [x] T-07 (re-run of violations pending)
