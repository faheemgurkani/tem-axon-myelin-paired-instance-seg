# Prior Literature Review (verbatim, as pasted 10 Nov 2026)

> This is the original brief from the first deep-research pass, preserved unedited. A few words were lost when it was
> pasted; they are marked `[…]`. The edited and corrected working version is
> [`../notes/03-literature-review.md`](../notes/03-literature-review.md), and the search log is
> [`prior-review-search-log-2026-11-10.md`](prior-review-search-log-2026-11-10.md). Facts verified from the challenge site
> ([`../notes/01-challenge-spec.md`](../notes/01-challenge-spec.md)) supersede this text, in particular the dataset is 400 = 266 + 134,
> masks are uint16, and there is no "~450 unlabeled" pool.

---

# Literature Review and Method-Design Brief: SIPAIM 2026 TEM Axon & Myelin Paired-Instance Segmentation Challenge (status as of 10 Nov 2026)

The strongest starting point for this challenge is a well-tuned nnU-Net (the AxonDeepSeg/ASTIH lineage) that predicts axon, myelin and boundary/distance channels, followed by an explicit, topology-enforcing pairing step that assigns exactly one myelin ring to each axon. No published method yet directly optimizes paired, nested axon–myelin instances for per-fiber g-ratio error. That gap is both your competitive edge and your paper.

## TL;DR

- **Baselines to run first:** (1) nnU-Net v2 semantic (3-class) plus classic watershed pairing, which is the de facto standard; (2) the AxonDeepSeg generalist/ASTIH TEM models; (3) Cellpose-SAM and μSAM (micro-sam) EM generalist for fiber-level instances; (4) a distance/boundary multi-head U-Net (HoVer-Net/StarDist-style) adapted to rings. Architecture novelty is the wrong lever. nnU-Net Revisited (Isensee et al., MICCAI 2024) found that "many of these recent claims fail to hold up when scrutinized for common validation shortcomings, such as the use of inadequate baselines, insufficient datasets, and neglected computational resources". Its recipe for state of the art is CNN U-Nets (ResNet/ConvNeXt variants) inside the nnU-Net framework, scaled to modern hardware.
- **Where SOTA falls short:** axon/myelin tools are almost all *semantic* segmenters with ad-hoc watershed post-processing. Myelin is consistently harder than axon because it has two interfaces. Abutting sheaths are separated by heuristics. Cross-domain generalization is poor: the best open multi-domain generalist reached only axon Dice 0.697 and myelin Dice 0.706 on out-of-distribution macaque TEM at 0.009 µm/px (TEM2; Collin et al., arXiv 2409.11552, Table 2). Per-fiber g-ratio is the weakest-reproduced morphometric (R² = 0.805 vs >0.96 for diameters in a 2026 nnU-Net TEM study).
- **Publishable direction:** a "nested-instance" formulation in which a fiber is a[…] object with an inner contour (axon) and outer contour (myelin) predicted jointly. Examples are a two-level distance/flow field, or star-convex/polar inner-outer radii, with containment-topology losses and boundary (HD-surrogate) losses. Add EM self-supervised pretraining for staining robustness and report g-ratio agreement (Bland–Altman) with subject-level splits. Target MICCAI/ISBI first, then Medical Image Analysis or IEEE TMI.

## Key Findings

### 1. Challenge facts and red flags (verify before building)

- The site specifies TEM at 100 kV, 3000×, 2.15 nm/px, with "Axon & Myelin (paired instances)" and label encoding 0 = background, 1 = artifacts (ignored), even X ≥ 2 = axon, X+1 = paired myelin. The submission is due **13 Nov 2026 (23:59:59 CET)** and requires predictions, a 2–4 page SIPAIM-format paper and replicable code. Registration closed 9 Nov 2026, and the test set was slated for release on 9 Nov.
- **Inconsistency:** the pre-submission checklist says predictions are "Single-channel PNG masks… with label values {0, 1, 2}", which contradicts the paired-ID scheme. Ask the organizers which format the evaluator expects. If IDs exceed 255, you will also need 16-bit PNG.
- **Leaderboard is almost certainly placeholder content:** the page shows "MyeloVision Net 0.934", "AxonFormer 0.921" and "TEM-SwinUNet 0.908" mean Dice while also saying submissions are "under independent validation" and winners "will be announced at SIPAIM 2026". Do not treat these as real reference scores.
- **Possible data overlap (unverified):** the public ASTIH "TEM1" set (DANDI 001436) contains 158 TEM images of mouse corpus callosum (splenium) from 20 control and cuprizone mice at 2.36 nm/px, CC-BY-4.0. The challenge says ~157 annotated images at 2.15 nm/px. The near-identical count may be coincidental, but if the challenge data derive from a related source, using TEM1 for pretraining could leak test data. Check acquisition metadata before using it for anything other than external validation.

### 2. Domain-specific […] (axo[…]n/myelin in EM)

| Method / paper | Venue, year (quality) | Approach | Instance and pairing handling | Reported results | Code / data |
|---|---|---|---|---|---|
| AxonDeepSeg (Zaimi et al.) | Scientific Reports 2018 (peer-reviewed, field reference) | U-Net-style CNN, 3-class semantic (axon/myelin/background), SEM and TEM models | None natively; morphometrics from post-processing | "In both models and all test samples, computed myelin Dice was lower than axon Dice", attributed to myelin having two ambiguous interfaces | github.com/axondeepseg/axondeepseg (MIT, actively maintained, updated 2026) |
| Multi-domain aggregation / generalist model (Collin, Boschet, Boudreau, Cohen-Adad) | arXiv 2409.11552, 2024 (preprint, NeuroPoly) | nnU-Net trained on aggregated TEM/SEM/BF/CARS data, resolution-ignorant, 5-fold, patch 384×640 | Semantic only | Generalist beats dedicated models (p = 0.03077). OoD Dice: SEM2 axon 0.834 / myelin 0.783; TEM2 (macaque) axon 0.697 / myelin 0.706. Authors judged SAM "not ideal" because it needs per-object prompts | model_seg_generalist release in AxonDeepSeg |
| ASTIH (Collin, Boudreau, Cohen-Adad) | MIDL 2026, PMLR 315 (peer-reviewed) | Benchmark collection plus baseline models | "standardized boundary delineation between adjacent fibers, enabling… semantic and instance segmentation" | 5 datasets, >69,000 manually segmented fibers, TEM/SEM/BF, mouse/rat/rabbit, 0.2–0.002 µm/px. Baseline metric values not verified here | github.com/axondeepseg/ASTIH |
| AimSeg (Carrillo-Barberà et al.) | PLoS Computational Biology 2023 (peer-reviewed) | ilastik pixel/object classifiers plus Fiji post-processing; segments axon, **inner tongue** and compact myelin | Explicit hierarchical instances (fibre ⊃ inner region ⊃ axon) | Evaluated by precision, recall, F1 and Jaccard; mouse corpus callosum TEM, remyelination | Zenodo 8351731 (CC-BY-4.0): 5 validation images (4 mice) + 3 control |
| MyelTracer (Kaiser et al.) | eNeuro 2021 (peer-reviewed) | Semi-automated GUI g-ratio tool | Assumes circu[…]; requires manual separation of adjacent sheaths | Semi-automated; used as a measurement reference | github.com/HarrisonAllen/MyelTracer |
| gACSON / DeepACSON (Behanova, Abdollahzadeh et al.) | Comput. Methods Programs Biomed. 2022; Nature Communications 2021 (peer-reviewed) | 3D-EM: semantic myelin, then instance axons via H-maxima of distance transform → marker watershed → bounded volume growing | Myelin partitioned by watershed seeded with axon instances; axon counted as myelinated if >70% of an enclosing hollow cylinder is myelin | g-ratio, diameter and eccentricity per cross-section | Public software |
| EM-SAM / AxonCallosumEM (Cheng et al.) | arXiv 2307.02464, 2023 (preprint); applied in Frontiers in Neuroinformatics 2025 | SAM ViT-B image encoder fine-tuned on mouse corpus callosum EM (Rett model), 4 nm/px, 1024×1024 patches | Semantic; instances derived afterwards (Shank3B study reports axon and myelin instance labels) | Dataset covers 1.83×5.76 mm, >600,000 patches; EM-SAM-Base reported 0.984 (metric definition not verified) | Dataset described in paper |
| nnU-Net optic-nerve TEM pipeline | Ophthalmology Science 2026 (peer-reviewed clinical journal) | nnU-Net with 1 encoder + 2 decoders (inner axon mask, outer fiber mask), morphological cleanup | Two nested masks, i.e. inner and outer contours predicted separately | Morphometric agreement with 3 experts: R² > 0.96 for d, D and area; 0.915 myelin thickness; **0.805 g-ratio** | Not verified |
| MicroStructFormer + AxonDeepSeg | bioRxiv 2026 (preprint) | Multi-class semantic (axon, myelin, mitochondria, periaxonal space, abnormal) | **Myelin-guided watershed** (dilated myelin as barrier, distance-transform-maxima seeds), then geodesic assignment of myelin to nearest axon | Morphometric study, gene-edited mice | Not verified |
| SnapG | bioRxiv 2026 (preprint) | Threshold/contour-based end-to-end g-ratio tool | Per-axon IDs; notes the periaxonal space complicates g-ratio | Validated on 201 manually measured sciatic nerve axons | Python tool |
| Le Couedic et al. | IPTA 2020 (conference, mid-tier) | DL for "challenging myelin sheaths" in elderly mouse corpus callosum TEM | Notes individual sheath separation is not achieved | Debris and aging artifacts are major error sources | — |
| Deep active learning (Lubrano di Scandalea, Perone, Boudreau, Cohen-Adad; NeuroPoly, arXiv 1907.05143) | arXiv 2019 | U-Net + MC-dropout uncertainty sampling, on "two realistic small datasets of 14 and 24 images" (SBEM and TEM) | Semantic | "Reached a maximum Dice value after adding 3 uncertainty-selected samples… versus 15 randomly-selected samples" | github.com/neuropoly/deep-active-learning |

**What this tells you:** every mainstream axon/myelin tool is (a) semantic first and (b) pairs axon and myelin by watershed or geodesic heuristics. Two exceptions point toward the right formulation: AimSeg's hierarchical instances and the optic-nerve study's inner/outer dual decoder. Nobody trains end-to-end on a paired-instance objective, and nobody optimizes per-fiber g-ratio error directly.

### 3. Surveys (domain and technique)

| Survey | Venue, year | Use for you |
|---|---|---|
| "Wrapped up: advancements in volume EM and application in myelin research" | De Gruyter (Methods in Microscopy), 2024 | Domain survey; tabulates MyelTracer, AimSeg, ilastik and gACSON with stated limitations (circularity assumption, manual separation of adjacent sheaths, under-estimated axon area) |
| Nunes et al., "A survey on cell nuclei instance segmentation and classification: leveraging context and attention" | Medical Image Analysis 99 (2025) 103360 | Instance-segmentation taxonomy (HoVer-Net, StarDist, Mask R-CNN, CIA-Net). Context/attention helped HoVer-Net but not Mask R-CNN |
| Chen et al., "A Systematic Review on Cell Nucleus Instance Segmentation" | IET Image Processing 2025 (mid-tier) | Distance-regression vs polygon (StarDist) families. Reports HoVer-Net AJI 0.775 / PQ 0.764 and StarDist Dice 0.809 / PQ 0.784 on its benchmark compilation |
| Zhang & Jiao, "Towards SAM for Medical Image Segmentation: A Survey" (SAM4MIS, arXiv 2305.03678, with maintained GitHub list incl. SAM 2/3) | Survey + living repository | Foundation-model adaptation landscape (adapters, MedSAM, SAM 2) |
| "A Review of Deep Learning Approaches Based on SAM for Medical Image Segmentation" | Bioengineering (MDPI), Nov 2025 (lower-tier venue) | Covers Jan 2023–Oct 2025 SAM variants: adapters, volumetric, uncertainty |
| Ma et al., "How Distance Transform Maps Boost Segmentation CNNs" | MIDL 2020 | Empirical comparison of boundary loss, HD loss and SDF regression. Note: distance-map regression as an auxiliary task sometimes *degrades* results |
| "Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook" | arXiv 2312.05391 | Boundary-level loss taxonomy |
| Maier-Hein et al., "Metrics Reloaded" and Reinke et al., "Understanding metric-related pitfalls" | Nature Methods 21 (2024) 195–212 and 182–194 | The reference for metric choice and aggregation pitfalls |
| Isensee et al., "nnU-Net Revisited: A Call for Rigorous Validation" | MICCAI 2024 | Baseline-rigor argument; ResEnc nnU-Net presets |

Coverage gap: I found no dedicated survey of *domain generalization/stain robustness for EM* or of *topology-aware segmentation* in a top venue during this search. Use the TopoMortar dataset paper (arXiv 2503.03365) and the related-work sections of clDice, Betti-matching and Skeleton Recall as de-facto reviews.

### 4. Transferable general techniques and their fit to paired nested instances

| Technique | Venue | Key evidence | Fit for this task |
|---|---|---|---|
| nnU-Net (+ ResEnc presets) | Nature Methods 2021; MICCAI 2024 | Self-configuring; most newer methods fail to beat it | **Must-have backbone/baseline.** Train it to output axon, myelin and an inter-fiber boundary class (ASTIH-style delineation) |
| Cellpose → Cellpose3 → Cellpose-SAM; v4.2 (June 2026) adds DINOv3-based "cpdino" models and cpsam_v2 | Nature Methods 2021/2025; bioRxiv 2025 | Flow-field instances; SAM backbone; robust to blur, noise and downsampling | Excellent for **fiber-level** (outer contour) instances of touching rings. Needs a second step for the inner contour |
| StarDist | MICCAI 2018 | Star-convex polygons | Fibers are roughly star-convex. A **dual-radius** variant (inner + outer radius per ray) gives nesting by construction |
| HoVer-Net / CellViT | MedIA 2019 / MedIA 2024 | Horizontal/vertical distance maps separate touching nuclei | Distance-map idea transfers directly to separating abutting myelin |
| μSAM (Segment Anything for Microscopy) | Nature Methods 22 (2025) 579–591 | Generalist LM and EM models plus automatic instance segmentation; trained on >17,000 images with >2 million annotated structures | Strong baseline and annotation accelerator; EM generalist was trained on organelles/neurites, not myelin rings |
| MedSAM / SAM 2 adapters | Nat. Commun. 2024; ICLR 2025 | Prompt-based | Prompting thousands of fibers per image is impractical (NeuroPoly's own critique); use as an encoder only |
| EM self-supervised pretraining: CEM500K (eLife 2021), RETINA (PLoS Comput Biol 2025), GAN pretraining (CVPRW 2024) | Peer-reviewed | Pretraining on unlabeled EM improves transfer; SEM-pretrained features helped TEM tasks | Use for staining/preparation robustness and label efficiency, especially since 608 images exist but only ~157 are labeled |
| clDice | CVPR 2021 | Topology guarantee for tubular structures | Myelin is an annulus, not a tube; use a skeleton-based variant on the *myelin ring* to penalize breaks |
| Skeleton Recall Loss | ECCV 2024 | Uses less GPU memory than clDice (5.6 vs 16.2 GiB) but more nnU-Net training time (7.1 vs 3.1 h) (TopoMortar, Table 1); best overall in its own benchmarks | Memory-cheap ring-continuity penalty. TopoMortar (Valverde et al., arXiv 2503.03365) found "clDice generally achieved the most topologically accurate segmentations, Skeleton Recall loss performed best particularly with noisy labels" |
| Betti matching (Stucki et al.) | ICML 2023; efficient 3D version 2024 | Persistent-homology […] correspondence | Ideal *metric*/loss for "one hole per ring" (β1 = 1 per myelin component) but costly at 4096×3008; apply on fiber crops |
| Topological interaction module (Gupta et al.) | ECCV 2022 (oral) | Enforces **containment and exclusion** via impermissible adjacent-label combinations; convolutional and efficient | **Most directly relevant**: axon must be contained by myelin; axon must never touch background; code in TopoXLab/TopoInteraction |
| Boundary loss (Kervadec et al.) | MIDL 2019 / MedIA 2021 | On ISLES, GDL Dice rose from 0.511 to 0.644 with 2D boundary loss and to 0.659 in 3D ("about 13% improvement in DSC over using Generalized Dice loss alone"); HD95 fell from 5.320 to 4.795 mm (2D) and 2.725 mm (3D) (arXiv 1812.07032v4) | Targets HD95 |
| HD loss (Karimi & Salcudean) | IEEE TMI 2020 | Differentiable HD surrogates via distance transforms | Targets HD95 directly; watch compute cost |

### 5. Public datasets for pretraining and external validation

| Dataset | Modality, species, region | Resolution | Size / annotation | Access |
|---|---|---|---|---|
| ASTIH TEM1 (DANDI 001436) | TEM, mouse corpus callosum splenium, control + cuprizone | 2.36 nm/px | 158 images, 20 mice; axon/myelin/axonmyelin masks with inter-fiber delineation | CC-BY-4.0 (**check overlap with challenge**) |
| ASTIH TEM2 (DANDI 001350) | TEM, mouse corpus callosum, Flox/SRF-cKO, high-pressure frozen | not verified | ~10 annotated images (myelinated + **unmyelinated axons**, some oligodendrocyte nuclei/processes) | CC-BY-4.0; ideal for unmyelinated-axon hard negatives |
| ASTIH SEM1 | SEM, rat cervical spinal cord | ~0.1 µm/px | 10 samples, 8 rats | CC-BY-4.0 |
| ASTIH BF1/BF2 | Bright-field, rat/rabbit peripheral nerve | 0.1–0.2 µm/px | Details not verified | See ASTIH |
| White Matter Microscopy Database (OSF yp4qg) | Multiple TEM/SEM/BF incl. macaque TEM | various | Source of the generalist model's data | OSF |
| AimSeg GT (Zenodo 8351731) | TEM, mouse corpus callosum, remyelination | downsampled ×4 | 8 images, hiera[…] instances incl. inner tongue | CC-BY-4.0 |
| AxonCallosumEM | Serial SEM, mouse corpus callosum (Mecp2) | 4 nm/px | >600,000 1024² patches, semantic GT | Per paper |
| AxonEM | 3D EM, mouse/human cortex | ~8 nm | 3D axon instances (not myelin pairs) | MICCAI 2021 |
| CEM500K | Unlabeled cellular EM (TEM + SEM) | various | ~500k images for self-supervised pretraining | eLife 2021 |

## Details

### Recommended evaluation framework

1. **Replicate the official metrics exactly:** Dice (axon, myelin, mean), HD95 in nm on axon and myelin boundaries, and per-fiber g-ratio MAE with IoU-based matching (g = equivalent axon diameter / equivalent fiber diameter). Then add a rank-aggregation script that mimics the challenge, including the HD95 tiebreak. Implementation details (per-image vs pooled Dice, HD95 when an instance is missing, the IoU threshold for matching, handling of label 1 and of border-cut fibers) can move scores substantially. Metrics Reloaded flags exactly these "inadequate implementation/aggregation" pitfalls (P3.1/P3.2), so mirror the organizers' reference code if released.
2. **Add instance diagnostics that the official metrics hide:** F1 at IoU 0.5/0.75, PQ (= SQ × RQ), and AJI for fibers. Count merge/split errors and the **pairing-violation rate**: axons without a ring, rings without an axon or with more than one axon, axon pixels touching background, and myelin components with β1 ≠ 1. Report unmyelinated-axon false positives separately, using TEM2's unmyelinated labels as a probe.
3. **Morphometry:** per-fiber g-ratio, axon diameter and myelin thickness, with Bland–Altman plots and bias/limits of agreement. Stratify errors by axon-diameter bins, because small axons dominate g-ratio sensitivity. Also report distribution-level agreement (g-ratio vs diameter scatter), which is what neuroscientists actually use.
4. **Splitting:** split by animal/subject (and by preparation batch if known), never by tile. The Reinke pitfalls paper stresses that non-independent samples must be accounted for in aggregation; use grouped k-fold, report per-animal variance, and use bootstrap CIs over subjects. Keep a held-out external set (ASTIH TEM1/TEM2, AimSeg GT) for generalization claims.
5. **Large-image handling:** evaluate on full 4096×3008 images after stitching, not on tiles. Tile-seam and border-cut fibers are where instance errors concentrate.

### Gaps a new method can target

- **Nesting/topology:** no axon/myelin method enforces "exactly one myelin ring per axon"; pairing is post-hoc. ECCV 2022 containment/exclusion constraints and Betti matching exist but have never been applied to axon–myelin units.
- **Touching fibers:** separation relies on watershed with dilated myelin barriers and distance-transform seeds (MicroStructFormer, gACSON), or on manual separation (MyelTracer). ASTIH now supplies delineated boundaries to learn from.
- **Inner boundary precision:** myelin Dice < axon Dice consistently since 2018. g-ratio R² is only 0.805 even when diameters reach >0.96. AimSeg shows that the inner tongue and periaxonal space bias "axon" definitions, so align with the organizers' annotation convention.
- **Generalization:** the best open generalist scored only axon Dice 0.697 / myelin Dice 0.706 on out-of-distribution macaque TEM (TEM2, Collin et al., arXiv 2409.11552, Table 2), even though it was trained on TEM1, SEM1, CARS1 and BF1–3 and "consistently detects more small axons"; the organizers name staining/preparation variation as a core gap.
- **Label efficiency:** only ~157 of 608 images are labeled. In prior axon/myelin work (Lubrano di Scandalea et al., arXiv 1907.05143), uncertainty-based active learning reached max Dice with 3 uncertainty-selected samples vs 15 random ones, on datasets of 14 and 24 images.

## Recommendations

**(a) Ranked baselines (run in this order)**

1. **nnU-Net v2 (ResEnc-M/L), 2D, labels {bg, axon, myelin, inter-fiber boundary}, plus marker-controlled watershed pairing** (seeds = axon components; flood myelin within a distance cap; drop myelin rings without seeds; drop axons without rings). Code: github.com/MIC-DKFZ/nnUNet. This is the bar you must beat.
2. **AxonDeepSeg generalist + ASTIH TEM baselines**, zero-shot and fine-tuned (github.com/axondeepseg/axondeepseg, /ASTIH). This tests domain shift directly.
3. **Cellpose-SAM (cpsam_v2) or cpdino, fine-tuned on fiber (axon+myelin) instances**, with axons obtained by a nnU-Net semantic head intersected per fiber (github.com/MouseLand/cellpose).
4. **μSAM EM generalist, automatic instance segmentation, fine-tuned** (computational-cell-analytics/micro-sam).
5. **Optional: StarDist dual-radius or HoVer-Net-style distance heads** as the "explicit-geometry" baseline.

**(b) For the challenge (deadline 13 Nov 2026, so days remain)**

- Ship baseline 1, using either 5-fold ensemble + TTA (flips/rot90) or the boundary-loss variant, depending on which validates better on HD95.
- Add rule-based post-processing: fill axon holes; enforce one ring per axon; remove small or unmyelinated candidates (ring-coverage ratio, akin to gACSON's 70% rule); exclude border-cut fibers if the GT does.
- Write ID pairs (2k, 2k+1) and confirm the PNG bit depth.

**(c) Publishable methodology directions (after the challenge)**

1. **Nested-instance representation (core contribution):** each fiber is one object with two coupled fields, e.g. a Cellpose-style flow to the fiber center plus a signed distance to the inner and outer contours, or polar inner/outer radii. Pairing and containment then hold *by construction*, and g-ratio becomes a differentiable function of the predicted radii/areas. This allows a **g-ratio consistency loss** alongside a boundary/HD surrogate.
2. **Containment-topology regularization:** a TopoInteraction-style loss (axon ⊂ myelin, axon ∩ background-adjacency = ∅, myelin–myelin exclusion between instances) plus Betti/skeleton-recall on rings. Evaluate with a new pairing-violation metric.
3. **Staining-robust pretraining:** MAE/DINO-style self-supervision on all 608 challenge images + CEM500K + unlabeled ASTIH, then fine-tuning. Claim robustness via leave-dataset-out evaluation (challenge → TEM1/TEM2/AimSeg).
4. **Label efficiency:** semi-supervised mean-teacher or pseudo-labels on the ~450 unlabeled images, and uncertainty-driven active learning.

**Target venues:** MICCAI 2027 main or ISBI 2027 for the method; MIDL for a benchmark/metric paper (pairing-aware evaluation); Medical Image Analysis or IEEE TMI for the journal extension with multi-dataset generalization; Scientific Data or NeuroImage if you release annotations.

## Caveats

- The challenge site's internal content (data.md, submission.md) did not load. Dataset counts (608/157 vs 400 = 266/134) and the precise matching rule are unverified, and the leaderboard appears to be placeholder content.
- I could not read the ASTIH PDF, so its baseline metrics, split policy and per-dataset fiber counts are not reported. The ASTIH README's "four datasets / 60,000 fibers" conflicts with the paper abstract's "five / 69,000"; I followed the paper.
- Several sources are preprints (EM-SAM, Cellpose-SAM, SnapG, MicroStructFormer, multi-domain aggregation); treat their numbers as provisional. The EM-SAM "0.984" score's metric is unclear.
- Venue details for MyelTracer (eNeuro 2021), StarDist (MICCAI 2018), HoVer-Net (MedIA 2019), CellViT (MedIA 2024), clDice (CVPR 2021) and MedSAM (Nat. Commun. 2024) come from background knowledge rather than pages fetched in this review.
