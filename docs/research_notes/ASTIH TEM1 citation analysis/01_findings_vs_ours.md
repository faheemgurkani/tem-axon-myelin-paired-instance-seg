# What the TEM1 users found, and how it compares with ours

## Per-paper findings on TEM1
1. **Jelescu 2016 (biology, the source).**
   - Cuprizone demyelination is **patchy (~5 um patches)**; smaller axons demyelinate first (myelinated axon count -35% at 6 wk); near-complete remyelination after diet stop.
   - **Annotation:** only "**full myelin rings**" segmented by ImageJ intensity threshold + connected components; unmyelinated axons not segmented (counted manually); g is an **aggregate per-image g** (not per fiber).
   - Fixation shrinkage confounds counts (+35% total axon count in cuprizone, called artifactual).
   - 3 mice per group x timepoint, 8 images per mouse -> **subject IDs exist** (20 subjects in ASTIH).
2. **Zaimi 2018 (AxonDeepSeg TEM model).**
   - Trained on 17 mice, tested on 3 mice (resampled to 10 nm/px, 512 px patches, small U-Net).
   - Mice test: **axon Dice 0.949, myelin Dice 0.855**, pixel accuracy 0.945; detection by **axon centroid** (no IoU): sensitivity 0.960, precision 0.965. Macaque CC (out-of-domain TEM): 0.907 / 0.752.
   - Errors named qualitatively: ambiguous/atypical myelin, inhomogeneous thickness, atypical axon intensity.
3. **Collin 2024 (generalist).**
   - nnU-Net, resolution-ignorant, 384x640 patches. Aggregating modalities (incl. TEM1) is >= dedicated models in-domain (paired t-test p = 0.03) and better out-of-domain: TEM2 dedicated 0.640/0.604 -> generalist 0.697/0.706.
   - Admits unquantified inter-/intra-rater variability in the masks ("not deemed problematic").
4. **Boschet 2024 (pseudo-labelling by modality translation).**
   - TEM1 = labelled source. TEM1-trained nnU-Net applied directly to macaque TEM: **axon 0.895, myelin 0.820** Dice; translation adds nothing for TEM->TEM; it helps only for SEM/BF.
   - Labels TEM1 "healthy" (it includes cuprizone mice) - a metadata error.
5. **ASTIH 2026 (benchmark).**
   - nnU-Net (1000 epochs, 5-fold, 92M params): **Dice axon 0.974, myelin 0.924**.
   - Instances: AxonDeepSeg watershed (axon-centroid seeds, -EDT of axon mask, constrained to axon+myelin) on **both** prediction and GT (GT instances are watershed-derived; 2-px gaps make them exact).
   - Detection: **F1@0.5 = 0.791 (P 0.747, R 0.851)**; F1@0.3 = 0.887. Cellpose (fiber-only, no myelin): F1@0.5 0.882 (P 0.990, R 0.797).
   - Matching via stardist.matching; no g-ratio, no pairing, no error decomposition, no human ceiling.

## Matches with our work
| Our finding / choice | TEM1 literature | Match? |
|---|---|---|
| Shared seeded -EDT watershed pairing (D-11); R1: lossless on GT pixels | ASTIH uses the same ADS watershed; GT instances made by it | **Same method.** R1 on TEM1 is trivially lossless (GT built by it) - so TEM1 cannot test pairing (already noted for T-17) |
| Selective annotation: abnormal / broken rings left as background | Jelescu annotated only "full myelin rings" by threshold; ADS/ASTIH refined those masks | **Likely the same bias** (incomplete/demyelinating rings under-labelled). Not stated as such by any of them |
| Unmyelinated axons = background | Not labelled in TEM1 ("resolvable but unlabeled") | **Same** |
| Pixel Dice and fiber F1 disagree | ASTIH: Dice 0.974 but F1@0.5 0.791 (precision 0.747) | **Same phenomenon, opposite direction.** Ours: Dice 0.882/0.861 but F1 0.904. Our Dice is pulled down by predictions on unlabelled real fibers (selection); TEM1 is denser (49 vs 29 fibers/img) and its low IoU-0.5 precision is undiagnosed |
| nnU-Net is the strong baseline | Collin 2024, Boschet 2024, ASTIH all use nnU-Net | **Same** |
| Instance-native model is conservative | ASTIH Cellpose: P 0.990, R 0.797 | **Consistent** with our Cellpose+B1 being more conservative (F1 0.885) |
| Error decomposition, pairing metric, human ceiling, per-fiber g error | None of the 5 TEM1 papers | **Gap holds on TEM1** -> C1 is new there too |
| No animal IDs (our limitation) | TEM1 has 20 subject IDs | TEM1 allows **subject-level** splits/bootstrap - stronger stats for T-17 |
| g-ratio | Jelescu: aggregate per-image g only; nobody evaluates per-fiber g error on TEM1 | Our g-MAE / H5 analysis would be **first** on TEM1 |

## Differences to keep in mind for T-17
- Acquisition: Philips CM12 + Gatan, 2.36 nm/px, ~6x9 um, ~3762x2286 px (ours: AMT NS12, 2.152 nm/px, 4096x3008).
- Labels: semantic (axon/myelin PNG) with 2-px gaps; ours are paired uint16 instances with label 1 = ignore. No "ignore" class in TEM1, so predictions on unlabelled rings count as FP there.
- Pathology: control + cuprizone (patchy de/remyelination) - closer to our data's pathology than TEM2.
- Benchmark numbers to reproduce with our evaluator (already in 07-plan T-17): Dice 0.974/0.924, F1@0.5 0.791; set the IoU threshold explicitly (ASTIH CLI default 0.3).

## Concrete hypotheses for T-17 (from this comparison)
1. ASTIH's F1@0.5 precision of 0.747 is dominated by **selection** (predictions on unannotated incomplete/demyelinating rings) or by **low_iou** on thin myelin - run our taxonomy on TEM1 to tell which. If selection: our main finding replicates externally.
2. Separation share will again be small on TEM1 (same watershed, disjoint axons).
3. Subject-level bootstrap on TEM1 gives honest CIs that our data cannot.
