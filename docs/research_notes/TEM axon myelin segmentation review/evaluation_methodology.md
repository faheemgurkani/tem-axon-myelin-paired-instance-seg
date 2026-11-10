# Evaluation methodology for paired axon/myelin instance segmentation (as of 10 Nov 2026)

Scope: metric definitions, reference implementations, pitfalls, statistics and ranking for the SIPAIM 2026 TEM
axon/myelin challenge (spec: `docs/notes/01-challenge-spec.md`; draft: `docs/notes/05-evaluation-framework.md`).
Reference source code was read directly from GitHub with `gh api` on 2026-11-10 (stardist `stardist/matching.py`,
google-deepmind `surface_distance/metrics.py`, MONAI `monai/metrics/hausdorff_distance.py` + `utils.py`,
vqdang `hover_net/metrics/stats_utils.py`, BrainLesion `panoptica/metrics/*.py`, `panoptica/instance_matcher.py`).

## Metrics Reloaded and Reinke et al.: recommendations for small-object instance segmentation, boundary metrics, aggregation and empty cases

### Takeaway
The citations are confirmed: Metrics Reloaded is *Nat Methods* 21(2):195–212 and Reinke et al. is *Nat Methods* 21(2):182–194, both 2024. In Reinke's taxonomy, **P3.1 = "Inadequate metric implementation"** and **P3.2 = "Inadequate metric aggregation"**, so our draft's citation of "P3.1/P3.2" for the ambiguity problem is correct. For instance segmentation, Metrics Reloaded recommends explicit detection metrics (F_β, or PQ as an alternative), per-instance segmentation metrics, and a boundary metric, with NSD as the default boundary metric when the reference is noisy. Metrics should be computed per image and then aggregated, respecting the hierarchical data structure. It does not mention HD95 among its defaults.

### Cited Findings
- Metrics Reloaded: Maier-Hein et al., *Nature Methods* 21(2):195–212 (Feb 2024), DOI 10.1038/s41592-023-02151-z, author manuscript PMC11182665. It covers image classification, object detection, semantic segmentation and instance segmentation, and it was built through a multistage Delphi process around a "problem fingerprint" — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/); [Nature PDF](https://www.nature.com/articles/s41592-023-02151-z.pdf)
- Reinke et al., "Understanding metric-related pitfalls in image analysis validation", *Nature Methods* 21(2):182–194 (2024), DOI 10.1038/s41592-023-02150-0 — [Edinburgh Research Explorer](https://www.research.ed.ac.uk/en/publications/understanding-metric-related-pitfalls-in-image-analysis-validatio-2/); [PMC11181963](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181963/)
- Reinke taxonomy (Fig. 2): P1 = inadequate choice of problem category. P2 = poor metric selection (P2.1 domain interest; P2.2 properties of target structures; P2.3 properties of the data set; P2.4 properties of the algorithm output). P3 = poor metric application: **P3.1 inadequate metric implementation; P3.2 inadequate metric aggregation**; P3.3 inadequate ranking scheme; P3.4 inadequate metric reporting; P3.5 inadequate interpretation of metric values — [PMC11181963](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181963/)
- Reinke P3.1 says that "the exact boundary extraction method is not standardized", which matters most for contour-sensitive metrics such as the HD. P2.3 says HD95 "was designed to deal with spatial outliers" — [PMC11181963](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181963/)
- Reinke P3.2 warns against simply aggregating metric values "over the entire data set and/or all classes" and against ignoring the hierarchical data structure. It notes that TorchMetrics computes DSC by default "as a global average over all pixels in the data set" — [PMC11181963](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181963/)
- Reinke P2.2: for small structures, "single-pixel differences may hugely impact the metric scores". P2.3 and P2.4 cover empty references and empty predictions, which cause division-by-zero cases — [PMC11181963](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181963/)
- Metrics Reloaded, instance segmentation: "we recommend selection of a set of detection metrics to explicitly measure detection performance". PQ is recommended "as an alternative to the F_β Score". Segmentation metrics are applied "per-instance" — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)
- Metrics Reloaded, boundary metrics: "Our general recommendation is therefore to complement an overlap-based metric with a boundary-based metric". When annotation imprecision should be tolerated (FP2.5.7), "our default recommendation is the Normalized Surface Distance (NSD)". It also recommends "outlier-robust metrics, such as NSD" (FP4.3.2) — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)
- Metrics Reloaded, aggregation: "metric values should first be computed per image and then be aggregated over the set of images". Extended Data Tab. 1 says to "Respect the hierarchical data structure when aggregating metrics" (citing Liang & Zeger, i.e. GEE) and to report "confidence intervals for all metrics" for small test sets — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)
- Metrics Reloaded, matching: if overlapping predictions are impossible and there are no confidence scores (FP5.4), it recommends "an assignment strategy based on IoU > 0.5". The localization criterion and assignment decision guides are Suppl. Notes 2.7.7 and 2.7.8. For small structures it recommends considering the problem as object detection, and for size variability it recommends size stratification (Suppl. Note 2.4) — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)
- Metrics Reloaded tells readers to use the reference implementations in the MONAI MetricsReloaded repository (Extended Data Tab. 1). It defers the empty-reference (FP4.6) and empty-prediction (FP5.2) rules to Suppl. Note 2.4 — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)
- Metrics Reloaded on ranking: "Avoid combining closely related metrics", and "Provide information beyond plain tables that make possible uncertainties in rankings explicit" — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)

### Inferences
- **Our task matches the FP5.4 case exactly.** A uint16 label map cannot contain overlapping predictions, and the challenge has no confidence scores. Metrics Reloaded therefore points to IoU > 0.5 assignment. Use it as the primary matching rule for g-MAE and for the diagnostics.
- The official metric set has an overlap metric (Dice) and a boundary metric (HD95) but **no detection metric**. Metrics Reloaded would add F1@0.5 or PQ. This supports Tier 2 of our framework, and it means fiber-level recall must sit next to g-MAE.
- Mixing HD95_Mean with Dice in a mean-rank scheme combines a metric that is correlated with Dice (both are driven by the same boundary errors). That conflicts with the advice to "avoid combining closely related metrics". Note this when discussing the official ranking.
- Report NSD (surface Dice at tolerance τ) as an extra boundary metric. A sensible τ is the inter-annotator boundary disagreement. If that cannot be measured, use 2–3 px ≈ 4–6 nm. This tolerance is our assumption and is not in any source.

### Gaps
- Suppl. Note 2.4 (empty-reference and empty-prediction rules, aggregation for object detection) and Suppl. Note 2.7.7/2.7.8 (decision guides DG8.x) were not readable in the PMC HTML. The fetched text was also inconsistent on the threshold for small objects: one passage says "lower" (FP3.1) and another "higher" (FP3.5). Check against the Supplementary PDF.
- I did not verify which Reinke pitfalls, if any, discuss PQ or AJI explicitly. The fetched text did not mention them.

## Instance metrics, matching strategies and reference implementations (including HD95 differences)

### Takeaway
With an IoU > 0.5 threshold, matching is provably unique, so greedy max-IoU and Hungarian give the same matches. The algorithm only matters below 0.5. **"HD95" is not one metric.** DeepMind and MONAI take the **max of the two directed 95th percentiles**, while panoptica (medpy-style) takes the **95th percentile of the pooled distance set**. Boundary extraction and empty-mask conventions also differ between libraries. The evaluator must fix all three and document them.

### Cited Findings
- **StarDist `matching()`** (default `thresh=0.5`, `criterion='iou'`). It relabels sequentially, builds the overlap matrix, and solves `linear_sum_assignment` with cost `-(scores >= thr) - scores/(2*n_matched)`. This maximises the number of above-threshold matches and uses IoU as a tie-breaker. A pair is a TP if its score ≥ thr. It returns precision, recall, F1, accuracy, `mean_matched_score` (= SQ), `mean_true_score`, and `panoptic_quality = sum_matched_IoU / (tp + fp/2 + fn/2)`. `matching_dataset` can pool counts or average `by_image` — [stardist/matching.py](https://github.com/stardist/stardist/blob/main/stardist/matching.py)
- **HoVer-Net `stats_utils.py`**: `get_fast_aji` pairs each GT with its argmax-IoU prediction (IoU > 0). `get_fast_aji_plus` uses Hungarian (`linear_sum_assignment(-pairwise_iou)`) for unique pairs. `get_fast_pq(match_iou=0.5)` thresholds directly when `match_iou >= 0.5` ("all IoU(p,g) > 0.5 pairing is proven to be unique") and uses Munkres otherwise. DQ = tp/(tp + 0.5fp + 0.5fn) — [hover_net/metrics/stats_utils.py](https://github.com/vqdang/hover_net/blob/master/metrics/stats_utils.py)
- HoVer-Net paper: AJI suffers from "over-penalisation of the overlapping regions", which "is likely to occur when the model completely fails to detect the neighbouring instance". PQ = DQ × SQ, where DQ is the detection F1 and SQ is the mean IoU of matched pairs. Uniqueness of IoU > 0.5 pairs is attributed to Kirillov et al. — [arXiv 1812.06499](https://arxiv.org/html/1812.06499v5)
- Foucart, Debeir & Decaestecker (Sci Rep 2023), "Panoptic quality should be avoided…": IoU, used both as the matching rule and inside SQ, "is not appropriate for segmentation of cell nuclei, due to the small size of these objects". A single entangled PQ score is also "poorly interpretable" — [PMC10224986](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10224986/); [Sci Rep](https://link.springer.com/article/10.1038/s41598-023-35605-7)
- **panoptica** (BrainLesion) separates instance approximation, matching and evaluation. `NaiveThresholdMatching(matching_threshold=0.5, allow_many_to_one=False)` is its default style of matcher. HD95 is `np.percentile(np.hstack((d_pred→ref, d_ref→pred)), 95)`, a **pooled** percentile. The border is `mask XOR binary_erosion(mask, connectivity)`, i.e. inner-boundary pixels, with voxel spacing supported — [panoptica repo](https://github.com/BrainLesion/panoptica) ([hausdorff_distance.py](https://github.com/BrainLesion/panoptica/blob/main/panoptica/metrics/hausdorff_distance.py), [assd.py](https://github.com/BrainLesion/panoptica/blob/main/panoptica/metrics/assd.py), [instance_matcher.py](https://github.com/BrainLesion/panoptica/blob/main/panoptica/instance_matcher.py))
- **MONAI `compute_hausdorff_distance(percentile=95, directed=False, spacing=…)`**: edges are `binary_erosion(seg) ^ seg` (inner boundary), cropped to the foreground by default. It computes the directed distance sets in both directions, takes `torch.quantile(d, 0.95)` of each, and returns their **max**. If exactly one mask is empty it returns **inf** (the current code explicitly avoids NaN from quantile interpolation). If both are empty it returns **NaN**, which `do_metric_reduction` drops from the mean. Without `spacing`, unit spacing is used — [MONAI hausdorff_distance.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/hausdorff_distance.py); [MONAI utils.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/utils.py)
- **DeepMind `surface-distance`**: borders come from a neighbour-code correlation, i.e. mask voxels whose neighbourhood is not full. Each surface element carries an **area weight** (surfel area). `compute_robust_hausdorff(sd, percent)` finds the weighted percentile in each direction with `searchsorted` on cumulative surfel areas and returns `max(perc_gt→pred, perc_pred→gt)`. An empty side gives `np.inf`. `compute_dice_coefficient` returns NaN when both masks are empty. It also provides `compute_surface_dice_at_tolerance` (NSD) — [surface_distance/metrics.py](https://github.com/google-deepmind/surface-distance/blob/master/surface_distance/metrics.py)

### Inferences
- **Concrete HD95 definition for the official replica (recommended default):** for each image and compartment c ∈ {axon (all even ≥ 2), myelin (all odd ≥ 3)}, take the binary compartment masks after removing GT==1 pixels from both GT and prediction. Extract inner boundaries with `mask ^ binary_erosion(mask, 4-connectivity)`. Compute EDT distances with `sampling=(2.152, 2.152)` nm. Set HD95 = max(P95(d_pred→gt), P95(d_gt→pred)), with linear-interpolation percentile, i.e. MONAI semantics. Expose switches for `pooled_percentile` (panoptica/medpy), `surfel_weighted` (DeepMind) and `boundary=inner|outer|both`. The spec's phrase "symmetric 95th percentile of boundary-to-boundary distances" fits both the max-of-directed and the pooled readings, so the sensitivity table must show both.
- **Empty compartment policy:** GT empty and prediction empty → NaN, excluded. Exactly one empty → a capped worst value, not inf, so that means stay finite. The cap is the image diagonal: sqrt(4096² + 3008²) = 5081.9 px × 2.152 = **10,936 nm**. Report the count of capped cases. At ~hundreds of fibers per image, empty compartments should be rare, but a blank prediction or an artifact-only image will trigger this.
- **Artifact masking for HD95:** decide whether label-1 pixels are removed before boundary extraction, which creates artificial boundaries at artifact edges, or whether distances are computed on unmasked masks and only boundary points inside the artifact are dropped. Default: set prediction pixels to background inside GT==1 and drop boundary points lying in or adjacent to GT==1. Keep this as a switch.
- **Fiber matching for g-MAE:** "maximum IoU" on X ∪ X+1 with no threshold. Default: IoU > 0.5, which is unique, so greedy = Hungarian and matches StarDist/HoVer-Net PQ. Switches: `iou_thresh ∈ {0 (Hungarian, any overlap), 0.1, 0.3, 0.5, 0.75}` and `algo ∈ {greedy_max_iou, hungarian}`. Below 0.5, greedy and Hungarian can differ, so report both.
- **Dice:** "across matched instances and global foreground masks" most plausibly means the compartment-level Dice on the global masks. Default: per-image global compartment Dice, then the mean over images (Metrics Reloaded). Switches: `pooled` (sum TP/FP/FN over the test set, as TorchMetrics does by default, cf. Reinke P3.2) and `matched_instance_mean` (per-fiber axon/myelin Dice averaged over matched pairs, unmatched GT = 0 or dropped).
- **AJI for paired fibers:** compute it at fiber level (X ∪ X+1). Prefer AJI+ (Hungarian, unique) over AJI (argmax, over-penalises merges/misses). PQ/SQ/RQ from StarDist semantics at IoU 0.5. Because CNS fibers here are large in pixels (a 0.2 µm axon is ~93 px across, with ~6,800 px of axon area), Foucart's small-object IoU argument is weaker here than for nuclei. It still applies to the thin myelin ring of small fibers, where a ring of ~12–20 px thickness loses a large share of its IoU to 1–2 px shifts.
- Do not use COCO-style mAP as a headline. There are no confidence scores in a label map, and Reinke documents that "mAP" has been redefined inconsistently in cell challenges.

### Gaps
- `seg-metrics` (Jingnan Jia) was not inspected, and its HD95 convention is unverified. Check it before citing it.
- MONAI's `SurfaceDiceMetric` source was not read. Its NSD class-threshold API should be checked with Context7 or the source before use.

## Morphometric agreement: g-ratio definitions, validation statistics, and boundary-shift sensitivity

### Takeaway
Automated myelin morphometry is usually validated against manual measurement with Bland–Altman (bias, 95% LoA) and ICC. The challenge uses the area-based equivalent-circle g, √(A_axon/A_fiber). **My sensitivity calculation shows that a 1-px (2.15 nm) shift of the axon/myelin interface changes g by +0.015 (≈ 2.2%) for a 0.2 µm axon, and by only +0.003 (0.4%) for a 1 µm axon.** 2-px shifts double this. So g-MAE is dominated by small fibers and by systematic interface bias, and it should be stratified by axon diameter.

### Cited Findings
- The challenge defines D = 2√(A/π) for the axon and for the fiber (X ∪ X+1) via `skimage.measure.regionprops`, with g = D_axon/D_fiber. This is algebraically equal to √(A_axon/A_fiber) — `docs/notes/01-challenge-spec.md` (from the challenge site source, [tem-challenge](https://alexandrojim.github.io/tem-challenge/))
- AxonDeepSeg light-microscopy validation (Sci Rep 2022), as summarised by the search: Bland–Altman of myelin thickness showed the automated method overestimating by 0.06 ± 0.15 µm, with 95% LoA −0.1 to 0.17 µm. Manual straight-line measurements gave g-ratios that "departed substantially from ground truth" — [PMC8993871](https://pmc.ncbi.nlm.nih.gov/articles/PMC8993871) (attribution from the search summary; verify the exact figures)
- An automated optical-microscopy pipeline (Frontiers in Neuroanatomy 2023) reported ICC 0.87 for g-ratio and 0.99 for axon diameter against manual measurement — [Front. Neuroanat. 2023](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2023.1260186/full)
- The literature does not use a single standard protocol. Bland–Altman, ICC and significance tests against manual labels all appear — [search synthesis of the above](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2023.1260186/full)

**Sensitivity calculation (my own; script in the session scratchpad, reproduced here).** Annulus model: axon radius r, fiber radius R = r/g₀, pixel size p = 2.152 nm. To first order, g = r/R gives Δg ≈ g(Δr/r − ΔR/R).
- Interface shift only (axon/myelin boundary moves outward by k px, outer boundary fixed): Δg = k·p/R = g₀·k·p/r. So **Δg/g = k·p/r, independent of g₀**.
- Outer boundary only (fiber grows by k px): Δg ≈ −g₀·k·p/R.
- Uniform dilation of both boundaries by k px: Δg ≈ k·p·(1 − g₀)/R. This partially cancels.

Exact results (Δg; positive = g overestimated):

| D_axon (µm) | g₀ | interface +1 px | interface +2 px | outer +1 px | outer +2 px | both +1 px | both +2 px | Δg/g (interface +1 px) | myelin thickness |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.6 | +0.0129 | +0.0258 | −0.0076 | −0.0151 | +0.0051 | +0.0101 | 2.15% | 67 nm ≈ 31 px |
| 0.2 | 0.7 | +0.0151 | +0.0301 | −0.0104 | −0.0205 | +0.0045 | +0.0088 | 2.15% | 43 nm ≈ 20 px |
| 0.2 | 0.8 | +0.0172 | +0.0344 | −0.0135 | −0.0266 | +0.0034 | +0.0067 | 2.15% | 25 nm ≈ 12 px |
| 0.3 | 0.7 | +0.0100 | +0.0201 | −0.0070 | −0.0138 | +0.0030 | +0.0059 | 1.43% | 64 nm ≈ 30 px |
| 0.5 | 0.7 | +0.0060 | +0.0121 | −0.0042 | −0.0083 | +0.0018 | +0.0036 | 0.86% | 107 nm ≈ 50 px |
| 0.8 | 0.7 | +0.0038 | +0.0075 | −0.0026 | −0.0052 | +0.0011 | +0.0022 | 0.54% | 171 nm ≈ 80 px |
| 1.0 | 0.7 | +0.0030 | +0.0060 | −0.0021 | −0.0042 | +0.0009 | +0.0018 | 0.43% | 214 nm ≈ 100 px |
| 1.5 | 0.7 | +0.0020 | +0.0040 | −0.0014 | −0.0028 | +0.0006 | +0.0012 | 0.29% | 321 nm ≈ 149 px |

**Shape-definition check (my own calculation).** Model the axon as an ellipse with semi-axes a, b (aspect ratio AR = a/b) and a uniform myelin thickness t = 0.3a. Then g_area = √(A_ax/A_fib) = 0.769 / 0.752 / 0.728 / 0.693 for AR = 1.0 / 1.2 / 1.5 / 2.0, while g_minFeret (minor axis) = 0.769 / 0.735 / 0.690 / 0.625. **Different definitions diverge by 0.02–0.07 for oblique or oval profiles.** This is larger than the 1-px boundary effect, so only compare against literature values computed with the same definition.

### Inferences
- **Recommended morphometry validation (Tier 3):** per matched fiber, report the bias and 95% LoA of Bland–Altman on g, D_axon and myelin thickness, using (D_fiber − D_axon)/2 as thickness. Add ICC(2,1) (two-way random, absolute agreement, single measure) and Lin's CCC. Report R² only as a supplement, because it measures correlation, not agreement. Stratify by D_axon bins (< 0.3, 0.3–0.5, 0.5–0.8, > 0.8 µm). Also compare the **distribution-level** outputs that biologists use: the per-image mean g, and the slope/intercept of g vs D_axon regression. A model with low per-fiber MAE can still bias the slope if its errors depend on size.
- A systematic 1-px inward/outward bias at the axon/myelin interface creates g errors (≈ 0.015 at 0.2 µm) of the same order as biologically meaningful group differences. This is an inference; I did not find a cited effect size for the comparison. Therefore **report the signed mean error (bias) of g as well as the MAE**, because MAE hides the direction.
- Myelin thickness of small fibers is only ~12–30 px. A 2-px error is ~7–17% of the ring thickness, so myelin Dice and IoU for small fibers are fragile. This is another reason to stratify by size.
- Border-cut fibers: an equivalent-circle diameter of a truncated profile is meaningless. Default for g-MAE: **exclude GT fibers whose fiber mask touches the image border** (or any GT==1 region), and exclude the predictions matched to them. Keep `include_border_fibers=True` as a switch to mirror the likely official behaviour. This follows the CellSeg precedent (next sections).
- The recall-adjusted g-MAE should be reported as the pair (g-MAE on matched fibers, fiber recall@0.5). Another option is g-MAE with missed GT fibers assigned a penalty, e.g. |g_GT − g_pop_mean|. That penalty is our construction, not from a source, so label it as such.

### Gaps
- I did not retrieve a primary source stating which g definition dominates EM neuroscience papers: the area-based / equivalent circle (as in AxonDeepSeg), the perimeter-based, or the manual shortest-diameter. Also not retrieved: Chomiak & Hu 2009 (PLoS ONE) on optimal g. Verify these before writing claims about convention.
- No cited source was found for typical corpus callosum g-ratio group differences in mouse models (e.g. cuprizone or shiverer). The 0.2–1.5 µm diameter range and g ≈ 0.6–0.8 in the table are user-provided assumptions.
- No source was found that validates TEM g-ratio with Lin's CCC specifically. The ICC and Bland–Altman evidence above is from light microscopy and optical pipelines.

## Challenge ranking methodology: BIAS, rank-then-aggregate vs aggregate-then-rank, stability, challengeR

### Takeaway
Rankings are not robust to the test set, the ranking scheme or the annotators (Maier-Hein 2018). Best practice is to report ranking uncertainty through bootstrap resampling of test cases, Kendall's τ, and significance maps (challengeR). NeurIPS CellSeg is a worked example: rank-then-aggregate per case, plus 1000 bootstrap samples, alternative schemes, and one-sided Wilcoxon tests.

### Cited Findings
- Maier-Hein et al., "Why rankings of biomedical image analysis competitions should be interpreted with care", *Nat Commun* 9:5217 (2018), DOI 10.1038/s41467-018-07619-7. Key quote: "the rank of an algorithm is generally not robust to a number of variables such as the test data used for validation, the ranking scheme applied and the observers that make the reference annotations". A correction was later published (Nat Commun 10:588) — [arXiv 1806.02051](https://arxiv.org/abs/1806.02051); [Warwick WRAP](https://wrap.warwick.ac.uk/111662); [White Rose correction record](https://eprints.whiterose.ac.uk/id/eprint/143349/)
- Teaching slides on that paper describe its bootstrap test: resample 1000× and check whether the original winner is still the winner — [Heidelberg slides](https://hci.iwr.uni-heidelberg.de/system/files/private/downloads/1241112364/lasse-becker-czarnetzki_pitfalls-of-competition.pdf)
- challengeR: Wiesenfarth et al., "Methods and open-source toolkit for analyzing and visualizing challenge results", *Sci Rep* 11:2369 (2021), DOI 10.1038/s41598-021-82017-6. It provides `aggregateThenRank`, `rankThenAggregate` (mean/median, tie handling) and `testThenRank` (significance level), and a `bootstrap` with n cases drawn with replacement (the README uses 1000). Visualisations include blob plots, violin plots, significance maps and line plots across ranking methods — [DOAJ record](https://doaj.org/article/18233fd8c57d48c38215d19972211289); [challengeR README](https://rdrr.io/github/wiesenfa/challengeR/f/README.md); [bootstrap.ranked.list](https://rdrr.io/github/wiesenfa/challengeR/man/bootstrap.ranked.list.html)
- NeurIPS 2022 CellSeg ranking: compute metrics per case and team, rank the teams on each case, then average and normalise the ranks (rank-then-aggregate). Robustness was checked with five alternative schemes, 1000 bootstrap samples with Kendall's τ, and one-sided Wilcoxon signed-rank tests (p < 0.05) — [Ma et al., arXiv 2308.05864](https://arxiv.org/html/2308.05864v2)
- A 2024 challenge report (PSFHS) used challengeR with five ranking methods and 1000 bootstrap samples — [arXiv 2409.10980](https://arxiv.org/pdf/2409.10980)
- Metrics Reloaded: "Provide information beyond plain tables that make possible uncertainties in rankings explicit" — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)

### Inferences
- **The official scheme is aggregate-then-rank per metric, followed by the mean of ranks across metrics.** Each team gets one dataset-level DSC_Mean, HD95_Mean and g-MAE. These are ranked and the ranks averaged, with ties broken by HD95. Implement `ranking.py` with `scheme ∈ {aggregate_then_rank (official), rank_then_aggregate (per image, CellSeg-style)}`, rank ties as `min` or `average`, and the HD95 tiebreak.
- For our paper, compare methods with (i) a 1000× bootstrap over test images (or over animals, if known) that recomputes the full ranking and reports P(rank = 1) and Kendall's τ against the original, and (ii) a challengeR-style significance map (pairwise one-sided Wilcoxon on per-image scores, Holm-corrected). The bootstrap can run in Python; calling challengeR via R is optional.
- With a mean-rank over three metrics, a method can win by being slightly better on two correlated metrics (Dice and HD95). Report the per-metric ranks next to the aggregate.

### Gaps
- The BIAS reporting guideline (Maier-Hein et al., "BIAS: Transparent reporting of biomedical image analysis challenges", *Medical Image Analysis* 66:101796, 2020) was not fetched in this session. Its citation details come from memory and must be verified before use.
- The quantitative results of Maier-Hein 2018 were not retrieved, so they should not be quoted until checked in the full text. From memory, these include the number of challenges analysed and how often the winner changed under bootstrap or a different scheme.

## Data splitting, leakage, confidence intervals and statistical tests

### Takeaway
Leakage from splitting below the subject level is documented: up to 41% score inflation when tiles from the same subject appear in both train and validation. Split by animal or block when metadata exists, otherwise by whole image, and never by tile. Use grouped k-fold, a cluster (group) bootstrap for CIs, and paired Wilcoxon signed-rank tests on per-image scores with Holm correction.

### Cited Findings
- Bussola et al., "AI slipping on tiles: data leakage in digital pathology" (ICPR 2021 workshops; arXiv 1909.06539): leakage usually comes from experiments "not accounting for the subjects in their data partitioning schemes". Tiling amplifies it. Scores were inflated **up to 41%** when tiles from the same subject were in training and validation. The remedy is patient-wise splitting — [arXiv 1909.06539](https://arxiv.org/pdf/1909.06539); [Springer](https://link.springer.com/chapter/10.1007/978-3-030-68763-2_13)
- Metrics Reloaded: respect the hierarchical data structure in aggregation (Liang & Zeger), and report confidence intervals for small test sets — [PMC11182665](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)
- Reinke P3.2: ignoring the hierarchical data structure is a named aggregation pitfall — [PMC11181963](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181963/)
- CellSeg used one-sided Wilcoxon signed-rank tests (p < 0.05) to compare teams — [arXiv 2308.05864](https://arxiv.org/html/2308.05864v2)

### Inferences
- **Hierarchy here:** animal → tissue block/grid → image → fiber. Fibers in one image are not independent, and neither are images from one animal. So: (a) fiber-level statistics (g-MAE, Bland–Altman) should get CIs from a **cluster bootstrap over images** (resample images and keep all their fibers), or over animals if IDs are known; (b) the unit of method comparison is the per-image score, or the per-animal mean if animals are known.
- **Splits:** `sklearn.model_selection.GroupKFold` / `StratifiedGroupKFold` with `groups = animal_id or block_id or image_id`, five folds, with the split JSON frozen. Infer groups from filename patterns and TIFF metadata such as acquisition timestamps; adjacent-in-time images are likely the same grid. If groups are unknown, cluster images by acquisition time or near-duplicate embedding and treat each cluster as a group. This is a conservative heuristic of ours.
- **Tests:** for methods A vs B, a paired two-sided Wilcoxon signed-rank on per-image metric values. For k methods, a Friedman test followed by pairwise Wilcoxon with **Holm** correction. Report the effect size (median paired difference with a bootstrap CI), not only p-values. Wilcoxon assumes independent pairs, so run it on per-animal means as well once animal IDs are known.
- With ~53 held-out images (20% of 266), CIs for Dice will be narrow but CIs for ranks will be wide. That makes bootstrap rank stability essential before claiming "better than baseline".

### Gaps
- I found no microscopy-specific (EM) study quantifying image-level vs animal-level leakage. The evidence above is from histopathology. Whether the challenge's 266 images share animals or grids is unknown; there is no metadata on the site.

## Large-image evaluation: tiling, stitching and border objects

### Takeaway
Evaluate on full stitched images. CellSeg **removes cells touching the image boundary before computing metrics**, and the winner used Gaussian-weighted sliding windows to avoid double-counting at patch seams. For TEM fibers, keep border fibers in Dice and HD95 (pixels are pixels), but by default exclude border-cut fibers from g-ratio and from instance matching, with a switch.

### Cited Findings
- CellSeg: "we remove these cells when computing the metrics", because boundary cells are usually incomplete and of limited downstream value. The test set included two whole-slide images. The paper does not describe how WSIs were scored (tiled or whole) — [arXiv 2308.05864](https://arxiv.org/html/2308.05864v2)
- MEDIAR inference: sliding window with 0.6 overlap and a Gaussian importance map. Patch predictions are merged so seam cells are not double-counted — [MEDIAR arXiv 2212.03465](https://arxiv.org/pdf/2212.03465); [arXiv 2308.05864](https://arxiv.org/html/2308.05864v2)
- CoNIC (Graham et al., *MedIA* 2024; arXiv 2303.06274) applied the top models to 1,658 WSIs (~700 M nuclei per model) for downstream analysis — [arXiv 2303.06274](https://arxiv.org/abs/2303.06274v2)

### Inferences
- **Stitching protocol:** predict tiles with ≥ 25% overlap and blend semantic/flow outputs with Gaussian weights. Run the instance/pairing step **once on the stitched full-resolution map**, not per tile, so instance IDs never need reconciling across seams. If per-tile instance outputs are unavoidable, resolve them with a "center-in-core" rule: keep an instance from the tile whose non-overlap core contains its centroid.
- Add a **seam diagnostic**: the rate of fiber splits and merges in a band of ±64 px around tile seams vs elsewhere.
- `border_policy` switch: `include` (likely official), `exclude_gt_touching_border` (CellSeg-style: remove the GT fibers and the predictions matched to them, and drop unmatched predictions touching the border), and `exclude_for_morphometry_only` (our default).
- **Scale bar:** the burned-in scale bar region should be checked in GT. If it is not label 1, mask it with a fixed ROI in a sensitivity run.

### Gaps
- The CoNIC, Lizard, MoNuSeg and PanNuke border conventions were not verified in this session. PanNuke and CoNIC use fixed 256×256 patches with metrics computed per patch (from memory, unverified). Whether they drop border nuclei is unconfirmed.

## What challenge winners do (2022–2026)

### Takeaway
Verified winning recipes combine strong pretraining and data-centric augmentation, a small ensemble, flip TTA, sliding-window inference with Gaussian blending, and simple morphological post-processing. For this challenge's metrics, the post-processing that enforces valid axon-in-myelin pairing and removes small or invalid fibers probably matters most for Dice, HD95 and g-MAE alike.

### Cited Findings
- MEDIAR (NeurIPS 2022 CellSeg winner, T1-osilab): SegFormer encoder + MA-Net decoder, Cellpose-style gradient flows, cell-wise intensity perturbation and boundary-exclusion augmentation, oversampling of minor modalities, two-phase pretraining on external data, then fine-tuning. Final = **ensemble of two models**, one from each pretraining phase. **TTA** with horizontal/vertical flips (summed). Sliding window with 0.6 overlap and a Gaussian map. Post-processing removes small cells and fills holes — [MEDIAR arXiv 2212.03465](https://arxiv.org/pdf/2212.03465); [MEDIAR GitHub](https://github.com/lee-gihun/mediar); [CellSeg arXiv 2308.05864](https://arxiv.org/html/2308.05864v2)
- CellSeg #2 (sribdmed): unsupervised grouping of images into four clusters, a classifier routing each image to a group-specific ConvNeXt model, watershed + NMS. #3 (cells): uncertainty-aware contour proposal network with ResNeXt-101 and Listen2Student pseudo-labelling — [arXiv 2308.05864](https://arxiv.org/html/2308.05864v2)
- CellSeg added a runtime metric to discourage large ensembles, citing the DSB winner's 32-model ensemble — [arXiv 2308.05864](https://arxiv.org/html/2308.05864v2)
- MEDIAR's validation-phase F1 was 0.9067 — [arXiv 2212.03465v1](https://arxiv.org/abs/2212.03465v1)

### Inferences
- The SIPAIM ranking has no runtime metric, so a 5-fold ensemble with 8× dihedral TTA (rotations and flips are valid for TEM) is free performance. Ensemble the **semantic probabilities and flows** before the pairing step, not the instance maps.
- The metric-aware post-processing implied by the analysis above:
  1. Fill axon holes, and require each myelin ring to enclose exactly one axon. This fixes pairing violations, which hurt matching and therefore g-MAE coverage.
  2. Drop tiny fibers below a size threshold tuned on validation. Such fibers barely affect Dice but make many FP or poor matches.
  3. Correct any systematic interface bias by morphological calibration, since g is very sensitive to 1-px interface shifts (table above).
  4. Do not suppress uncertain fibers just to improve g-MAE. That gaming is possible, but it costs Dice and recall, and we report recall anyway.
- nnU-Net is the standard strong semantic baseline in MICCAI challenges. Its default 5-fold ensemble + mirroring TTA mirrors the MEDIAR recipe.

### Gaps
- The CoNIC 2022 winners' methods (ensembling, TTA) were not retrievable from the abstract pages in this session.
- No verified summary of 2023–2026 MICCAI segmentation-challenge winning recipes was retrieved. Claims such as "most winners use nnU-Net" are widely repeated, but this session did not source them, so treat them as unverified.
