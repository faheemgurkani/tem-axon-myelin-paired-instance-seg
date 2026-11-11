# 11 — Deep EDA: label quality, image forensics and false-positive patterns (12 Nov 2026)

This follows [10](10-eda.md). The trigger was another team reporting "false-positive patterns" in the data. The goal is to find everything in the
266 training images and masks that will affect training and testing, using checks that microscopy and morphometry researchers run routinely.

- **Where things are:**
  - All scripts are in `scripts/eda/`.
  - Tables (CSV) and figures are in `data/processed/eda/` (git-ignored, because they contain image crops).
  - Overlap groups for splitting are in `configs/splits/overlap_groups.json`.
- **Runtime** on 8 cores: about 3 min for overlap detection and 5 min for deep quality.

## A. Technique log: what was done, why experts do it, how

| # | Technique | Why a domain expert runs it | Implementation (script) |
|---|---|---|---|
| T1 | **Field-of-view (FOV) overlap detection** | TEM images are often tiles from the same grid square or mosaic. Overlapping tiles leak between train and validation and inflate scores. It is the most common hidden leak in microscopy datasets. | ORB features (1500 per image, CLAHE first) at 1/4 resolution. All 35,245 pairs, including mirrored copies. Ratio test, then RANSAC similarity transform. A pair counts as real overlap if it has ≥20 inliers and scale is 0.95–1.05 (`overlap.py`). |
| T2 | **Re-annotation agreement from overlaps** | Inter- and intra-rater variability sets the *ceiling* on any metric. Normally it needs a second annotator; overlapping tiles give it for free, because the same tissue was annotated twice. | Refine the alignment with ECC (affine, 1/2 resolution). Warp B's image and mask into A's frame and erode the valid area by 20 px. Then compute pixel Dice, fiber IoU, Δg, and how B labelled each of A's fibers (`overlap_agree.py`, `disagree_viz.py`). |
| T3 | **Edge-alignment profile of label boundaries** | Checks whether annotators trace the membrane, its inner side or its outer side, and whether that is consistent. A systematic offset biases g and HD95; variable offsets set an HD95 noise floor. | Gaussian gradient magnitude (σ=1.5). Signed distance to each fiber's outer contour and to the axon contour, binned −6…+6 px. Mean gradient per bin; the argmax is the offset (`deep_quality.py`). |
| T4 | **Myelin thickness profile around each ring** | Real sheaths have near-uniform thickness. Large variation flags outfoldings, merged fibers, open rings or annotation slips. | Euclidean distance from every outer-contour pixel to the axon: median, 2nd/98th percentile and CV (`deep_quality.py`). |
| T5 | **Intensity sanity of labels** | Compact myelin is electron-dense (dark) and axoplasm is light. A myelin label on bright pixels, or an axon label on dark pixels, means a label error or pathology. | Per fiber: fraction of myelin pixels brighter than the image's background median, and fraction of axon pixels darker than the image's myelin median. |
| T6 | **g-ratio definition sensitivity** | Morphometry papers disagree on g: equivalent circle, minor axis (robust to oblique cuts), Feret, or thickness-based. If the evaluator's definition differs from ours, that can outweigh any method difference. | Per fiber: g from equivalent-circle area, minor axes, max Feret, and median thickness. |
| T7 | **Oblique-section check** | Fibers cut off-perpendicular look elliptical, and equivalent-circle diameters then over-estimate size. Experts filter them by eccentricity or use the minor axis. | Eccentricity and major/minor axes of axon and fiber. Compared between labelled, unlabelled and label-1 objects. |
| T8 | **TEM image forensics** | Detects acquisition and processing differences that cause domain shift (summarised in section D). | `deep_quality.py` |
| T9 | **Lamellar-periodicity calibration attempt** | Compact CNS myelin repeats every ~10–12 nm. If resolved, an FFT peak lets you check the pixel size independently of metadata. | Radially averaged power spectrum of 64×64 patches deep inside myelin (3 images). |
| T10 | **False-positive candidate mining** | Finds structures a model will predict that the GT calls background. Experts then sort them by eye into a taxonomy. | Dark-band closed-hole detector at 1/4 resolution; keep candidates that do not overlap labels (`unl_loc.py`). Random and large-size galleries were inspected by eye (`gallery.py`). |
| T11 | **Outlier-fiber galleries** | Extreme values of T3–T6 are usually either label errors or pathology. Looking at them is what separates the two. | The 4 most extreme fibers per criterion, viewed with label contours. |

T8 (TEM image forensics) checks:
- noise σ (wavelet MAD)
- sharpness (Laplacian variance, high-frequency power share)
- illumination non-uniformity (quadratic surface fit)
- knife chatter and scan banding (FFT spike score, row/column profiles)
- JPEG 8×8 blockiness
- histogram comb (contrast stretching)
- stain-precipitate specks
- saturated or black corners

## B. Findings: false-positive patterns (what the other team probably saw)

The training GT contains things a good model *should* segment but that are scored as background.

1. **Real myelinated fibers left unlabelled (annotation false negatives).**
   - T10 found 305 unlabelled ring structures larger than 450 nm (348 above 400 nm) in 152 of 266 images, about 1.1 per image.
   - In a random gallery of 32, about **28 are genuine myelinated fibers**: compact normal ones, oblique ones, and fibers with vacuoles or periaxonal debris.
   - Their shape matches labelled fibers (median eccentricity 0.734 in both), so they are **random misses, not a rule**.
   - Lower bound: about 4% of fibers. The overlap check (T2) agrees: of 150 fibers annotated in one tile, **9 are background and 3 are label 1 in the overlapping tile**, i.e. about 8% disagreement on *which* fibers are annotated.
2. **Interstitial "pseudo-fibers".**
   - Among small unlabelled candidates, about 40% are the light triangular gaps enclosed by three or more touching sheaths.
   - To a model, such a gap looks like an axon inside a ring. This is a structural FP trap that only arises because 71% of fibers touch a neighbour.
   - It is exactly where by-construction pairing should help: the "ring" around the gap belongs to other fibers.
3. **Myelin splits, vacuoles and organelles** inside or between sheaths: about 25–35% of small candidates.
4. **Label 1 is not only border fibers.**
   - There are 448 interior label-1 objects. They are *larger* (median equivalent diameter 854 nm vs 755 nm for annotated fibers) and *more elongated* (19% with eccentricity > 0.9 vs 6% for annotated fibers).
   - So annotators deliberately ignored large, oblique or ambiguous fibers. Predictions there are free; predictions on finding 1 are penalised.
5. **Acquisition artifacts that look like structure:**
   - **Black corner shadows** in 10 images (0003, 0010, 0045, 0056, 0092, 0123, 0125, 0234, 0244, 0264). These also drove the only thumbnail correlations found in [10](10-eda.md).
   - **A horizontal seam or fold band** near the bottom of 0144.
   - **Strong illumination gradients:** relative range up to 64% (0234), and over 44% in the top 5% of images.

## C. Findings: label quality and conventions

- **Re-annotation ceiling (T2).** Only pairs with a well-aligned overlap (image NCC > 0.8) were used: 0055/0212, 0024/0141, 0091/0201, 0054/0152, 0236/0257. On these:
  - axon Dice 0.98–0.99
  - **myelin Dice 0.90–0.97**
  - fiber IoU 0.95–0.99
  - per-fiber |Δg| = **0.004–0.010**

  Pairs with poorer alignment give lower numbers because residual misregistration from TEM distortion and drift creeps in. Treat these as an *upper bound on achievable myelin Dice (~0.93–0.96)* and a
  *floor on g-MAE (~0.006)*. Whether the tiles were annotated independently is unknown; if one annotator drew both, inter-rater noise is larger.
- **Boundary convention (T3).**
  - Averaged over images, the strongest intensity edge lies ~2–3 px *outside* the outer fiber contour and ~3–4 px *inside* the axon contour.
  - The per-image argmax varies: 1–6 px outer, −6 to 0 px inner.
  - Pixel-level zooms show the contours on the edge transition, which is itself 5–8 px wide and noisy. The edge peak may also be a neighbouring membrane (axolemma, outer tongue).
  - Interpretation: **labels follow a consistent but smooth convention, with ±1–2 px placement noise between images.** Two consequences:
    - Do **not** snap predictions to image edges (active contours, edge refinement). That would move them away from the convention.
    - A few px of HD95 error is unavoidable.
- **Thickness uniformity (T4).**
  - Median CV is 0.13.
  - 1% of fibers reach thickness ≈ 0 somewhere: open rings or an axon touching the outer edge, mostly at tile borders.
  - CV is above 0.35 for the top 1%: outfoldings and merged sheaths.
- **Intensity sanity (T5).**
  - Myelin labels on bright pixels: median 2.8%, 99th percentile 23.5%. The top cases are sheaths with splitting or vacuoles where the label spans the bright gaps.
  - Axon labels on dark pixels: 99th percentile 19.5%, max 64%. These are degenerating dark axoplasm or debris.
  - No label swaps (myelin inside axon) were found.
- **Pathology is present and labelled inconsistently (T11).** The dataset has the hallmarks of a de/remyelination model:
  - g up to 0.94: large axons with very thin sheaths, typical of remyelination.
  - dark and degenerating axons
  - vacuolated or split myelin
  - periaxonal debris
  - redundant myelin and outfoldings

  Some of these fibers are labelled and some are left as background (B1).
- **g definition matters more than annotation noise (T6).** Mean g is 0.750 (equivalent circle), 0.725 (minor axis), 0.790 (Feret) and 0.763 (thickness).
  That spread of 0.065 is about **10× the re-annotation g noise**. The evaluator must use exactly the spec's equivalent-circle definition.
- **Oblique cuts (T7).** 6% of annotated axons have eccentricity > 0.9 (major/minor ratio ≥ 2.3). Equivalent-circle diameters over-estimate them, but the spec scores g this way, so we follow it.

## D. Findings: image forensics (T8, T9)

- **Raw 8-bit data, no compression traces.**
  - JPEG 8×8 blockiness is 1.000 ± 0.001.
  - No empty histogram bins inside the 0.5–99.5% range, so no contrast stretching.
  - Odd/even grey-level balance is 1.00.
- **Noise varies 3×.**
  - Wavelet σ ranges 7.5–22 grey levels (median 16).
  - It is a continuum, not two clusters. The 41 lowest-noise images (σ < 12) are darker (mean 103 vs 124), smoother-looking (possibly denoised or longer exposure) and denser in fibers.
  - The Laplacian "sharpness" correlates with noise at r = 0.99, so it measures noise, not focus. **No defocused images were found.**
- **Illumination non-uniformity.** The median relative range is 20%, the 95th percentile 44% and the max 64%. Both vignetting (bright centre) and linear gradients occur.
- **Periodic artifacts.** No knife chatter or scan banding. The FFT spike score is flat (2.68 ± 0.04) except 0234 (3.31), which is explained by its black corners.
- **Stain-precipitate specks:** about 620 small very dark specks per image, present almost everywhere. They are a normal part of the appearance and a minor FP risk for tiny "axons".
- **Pixel-size check from myelin lamellae (T9).** This was not possible: no peak at the ~10–12 nm lamellar period (≈5 px) at this magnification and noise level. **We take 2.152 nm/px as given.**
- **Fiber density** ranges 0.11–1.47 fibers/µm² (13×), so annotated content and class balance vary strongly between images.

## E. Findings: leakage

- **22 overlapping image pairs, forming 18 groups that cover 40 images** (group sizes 4, 3, 3, then 2s).
  - Transforms are pure translations: scale 0.96–1.02, rotation < 3°. These are neighbouring tiles of the same section.
  - Overlap ranges 8–76% of the image area.
  - Examples: 0055/0212 (45%), 0118/0178 (73%), 0042/0163 (76%), 0002–0045–0147, 0008–0034–0060–0264.
- **No exact or mirrored duplicates.**
- **The test set (134 images) is very likely tiled from the same sections,** so some test tiles probably overlap training tiles. Do not exploit this. Report it if confirmed when the test images are released.

## F. What this means for training and testing

1. **Splits must be group-aware.** Use `configs/splits/overlap_groups.json` as a GroupKFold constraint. Overlapping tiles must never sit on both sides of a split, or validation scores will be inflated.
2. **Background is not clean negatives.** About 4–8% of real fibers are unlabelled. Options:
   - down-weight the background loss inside confident ring-like structures (a partial-label / positive-unlabelled view)
   - mine these regions and mark them ignore for training only
   - at minimum, do **not** treat FP counts on them as model errors in our own diagnostics

   The official test GT has the same misses, so expect a few % of unavoidable Dice and fiber-F1 penalty.
3. **Add an FP taxonomy to `diagnostics.py`.** Classify each unmatched predicted fiber as:
   - (a) on an unlabelled real-fiber candidate
   - (b) on an interstitial gap
   - (c) on label 1, which is free
   - (d) other

   This is the selection-vs-pairing split needed for the go/no-go experiment ([09](09-research-plan.md)).
4. **The interstitial-gap trap is evidence for our method.** A semantic + watershed pipeline can create a "fiber" from neighbours' myelin. A nested-by-construction decoder only creates a fiber when the ring belongs to it. Measure the FP rate on (b) specifically.
5. **Mind the metric ceilings.** Myelin Dice is capped at about 0.93–0.96 and g-MAE floors at about 0.006 by annotation noise. Method differences smaller than this are noise. Report bootstrap CIs.
6. **Augmentation and normalisation:**
   - per-image intensity normalisation
   - strong brightness, contrast and gamma augmentation
   - low-frequency illumination-gradient augmentation
   - Gaussian/Poisson noise across σ 7–22
   - random black-corner masks

   No defocus augmentation is needed.
7. **Boundary convention:** learn it from the labels and don't add edge-snapping post-processing. The g-ratio must use the equivalent-circle definition exactly.
8. **Patch sampling:** density varies 13×, so sample patches by fiber density, or oversample dense images, to keep touching-fiber cases frequent.
9. **Label cleaning (training only):**
   - drop the 47 orphan IDs, or treat them as ignore
   - treat the 296 fibers with topology defects as ignore or repair them
   - keep pathological fibers as labelled, since the test GT includes them

## G. Log

- 2026-11-12: T1–T11 run on all 266 images. Overlap groups were saved for splits. The figures were inspected by eye:
  - random and large unlabelled-candidate galleries
  - an outlier-fiber gallery
  - an image-outlier panel
  - pixel-level edge zooms
  - an overlap-disagreement sheet. 2 of its 9 crops look affected by residual misregistration or the frame edge, so the ~8% disagreement figure is an upper estimate.
