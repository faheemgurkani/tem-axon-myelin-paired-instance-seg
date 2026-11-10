# 03 — Literature Review and Method-Design Brief (as of 10 Nov 2026)

Origin: a deep-research pass (235 sources) run before this repo existed and preserved here. Several sources are
preprints, and some venue details come from background knowledge. Re-verify any number before citing it in a paper.

> **Superseded in part by the 2026-11-11 deep-research report** ([`../reports/TEM axon myelin segmentation review.md`](../reports/TEM%20axon%20myelin%20segmentation%20review.md)).
> Its audit table marks each claim below CONFIRMED / CORRECTED / UNVERIFIED. Key corrections: there is no published "dual-radius StarDist";
> SAM4MIS = CIBM 2024 (arXiv 2401.03495); DeepACSON is in *Communications Biology* 2021; gACSON's 70% is only an example; EM-SAM's 0.984 is semantic mIoU on
> serial **SEM**; the TopoMortar numbers are in Table 6 (BMVC 2025 oral); the optic-nerve code repo returns 404; the generalist's "TEM2" is macaque, not ASTIH TEM2.
>
> **Corrections from the 2026-11-10 site crawl (see [01](01-challenge-spec.md)):**
> - Dataset is **400 images = 266 annotated train + 134 unannotated test**, 4096×3008. The review's "~157 labeled /
>   608 total / ~450 unlabeled" figures come from an earlier site version. **There is no extra unlabeled pool** besides
>   the 134 test images.
> - The mask format is **resolved**: `uint16` paired IDs (`.tif` or uint16 `.png`). The `{0,1,2}` checklist is stale.
> - The **matching rule** is max-IoU on the full fiber mask (no threshold stated). g-MAE is computed over matched fibers only.
> - The TEM1-overlap concern is weaker (158 vs 266 images; 2.36 vs 2.15 nm/px) but **still unverified**. Both are mouse
>   corpus callosum. Check provenance before pretraining on ASTIH TEM1.

---

## TL;DR

- **Start from** a well-tuned nnU-Net (the AxonDeepSeg/ASTIH lineage) predicting axon, myelin and boundary/distance channels,
  followed by an explicit topology-enforcing **pairing step** (exactly one myelin ring per axon).
- **The gap:** no published method directly optimizes paired, nested axon–myelin instances for per-fiber g-ratio error.
  That is both the competitive edge and the paper.
- **Architecture novelty is the wrong lever.** *nnU-Net Revisited* (Isensee et al., MICCAI 2024): "many of these recent
  claims fail to hold up when scrutinized for common validation shortcomings, such as the use of inadequate baselines,
  insufficient datasets, and neglected computational resources." CNN U-Nets (ResEnc/ConvNeXt) inside nnU-Net remain SOTA.
- **Where SOTA falls short:**
  - tools are *semantic* plus ad-hoc watershed
  - myelin is harder than axon (two interfaces)
  - abutting sheaths are separated by heuristics
  - poor cross-domain generalization: the best open generalist reaches axon Dice 0.697 / myelin 0.706 on OOD macaque TEM (TEM2, Collin et al., arXiv 2409.11552, Table 2)
  - per-fiber g-ratio is the weakest-reproduced morphometric: R² = 0.805 vs > 0.96 for diameters (2026 nnU-Net optic-nerve TEM study)
- **Publishable direction:** a *nested-instance* formulation in which a fiber is one object with inner (axon) and outer
  (myelin) contours predicted jointly, plus containment-topology and boundary losses, EM self-supervised pretraining
  for staining robustness, and g-ratio Bland–Altman evaluation with subject-level splits.

## 1. Domain-specific SOTA (axon/myelin in EM)

| Method / paper | Venue, year | Approach | Instance / pairing | Reported results | Code / data |
|---|---|---|---|---|---|
| **AxonDeepSeg** (Zaimi et al.) | Sci. Reports 2018 | U-Net CNN, 3-class semantic, SEM and TEM models | None natively; morphometrics from post-processing | "In both models and all test samples, computed myelin Dice was lower than axon Dice" (two ambiguous myelin interfaces) | github.com/axondeepseg/axondeepseg (MIT, maintained) |
| **Multi-domain generalist** (Collin, Boschet, Boudreau, Cohen-Adad) | arXiv 2409.11552, 2024 | nnU-Net on aggregated TEM/SEM/BF/CARS, resolution-ignorant, 5-fold, patch 384×640 | Semantic only | Generalist > dedicated (p = 0.03077). OoD: SEM2 axon 0.834 / myelin 0.783; TEM2 (macaque) axon 0.697 / myelin 0.706. SAM judged "not ideal" (per-object prompts) | `model_seg_generalist` in AxonDeepSeg |
| **ASTIH** (Collin, Boudreau, Cohen-Adad) | MIDL 2026, PMLR 315 | Benchmark collection + baselines | "standardized boundary delineation between adjacent fibers, enabling… semantic and instance segmentation" | 5 datasets, > 69,000 fibers, TEM/SEM/BF, mouse/rat/rabbit, 0.2–0.002 µm/px. Baseline numbers **not verified** | github.com/axondeepseg/ASTIH |
| **AimSeg** (Carrillo-Barberà et al.) | PLoS Comput. Biol. 2023 | ilastik pixel/object classifiers + Fiji; axon, **inner tongue**, compact myelin | Explicit hierarchy (fibre ⊃ inner region ⊃ axon) | Precision/recall/F1/Jaccard; mouse CC TEM, remyelination | Zenodo 8351731 (CC-BY-4.0) |
| **MyelTracer** (Kaiser et al.) | eNeuro 2021 | Semi-automated GUI g-ratio tool | Assumes circular axons; manual separation of adjacent sheaths | Measurement reference | github.com/HarrisonAllen/MyelTracer |
| **gACSON / DeepACSON** (Behanova, Abdollahzadeh et al.) | CMPB 2022; Nat. Commun. 2021 | 3D-EM: semantic myelin → instance axons via H-maxima of DT → marker watershed → bounded growing | Myelin partitioned by watershed seeded with axons; myelinated if > 70% of an enclosing hollow cylinder is myelin | g-ratio, diameter, eccentricity | Public |
| **EM-SAM / AxonCallosumEM** (Cheng et al.) | arXiv 2307.02464 (2023); Front. Neuroinform. 2025 | SAM ViT-B encoder fine-tuned, mouse CC EM (Rett model), 4 nm/px, 1024² patches | Semantic; instances afterwards | > 600k patches; "0.984" (metric unclear) | Per paper |
| **nnU-Net optic-nerve TEM** | Ophthalmology Science 2026 | nnU-Net, 1 encoder + **2 decoders** (inner axon mask, outer fiber mask) + morphological cleanup | **Nested masks: inner/outer contours** | vs 3 experts: R² > 0.96 for d, D, area; 0.915 myelin thickness; **0.805 g-ratio** | Not verified |
| **MicroStructFormer + AxonDeepSeg** | bioRxiv 2026 | Multi-class semantic (axon, myelin, mitochondria, periaxonal space, abnormal) | **Myelin-guided watershed** (dilated myelin as barrier; DT-maxima seeds) + geodesic myelin→axon assignment | Morphometric study, gene-edited mice | Not verified |
| **SnapG** | bioRxiv 2026 | Threshold/contour g-ratio tool | Per-axon IDs; periaxonal space complicates g | 201 manually measured sciatic axons | Python |
| **Le Couedic et al.** | IPTA 2020 | DL for challenging sheaths, elderly mouse CC TEM | Sheath separation not achieved | Debris and aging artifacts are major error sources | — |
| **Deep active learning** (Lubrano di Scandalea et al.) | arXiv 1907.05143 | U-Net + MC-dropout uncertainty sampling; SBEM/TEM sets of 14 and 24 images | Semantic | Max Dice after 3 uncertainty-selected vs 15 random samples | github.com/neuropoly/deep-active-learning |

**Takeaway:** every mainstream tool is (a) semantic first and (b) pairs axon and myelin with watershed or geodesic
heuristics. The exceptions pointing the right way are AimSeg's hierarchy and the optic-nerve dual decoder. Nobody trains
end-to-end on paired instances, and nobody optimizes per-fiber g-ratio error directly.

## 2. Surveys

| Survey | Venue | Use |
|---|---|---|
| "Wrapped up: advancements in volume EM and application in myelin research" | De Gruyter (Methods in Microscopy) 2024 | Domain survey; tool limitations (circularity, manual sheath separation, under-estimated axon area) |
| Nunes et al., survey on cell-nuclei instance segmentation (context & attention) | Medical Image Analysis 99 (2025) 103360 | Instance taxonomy (HoVer-Net, StarDist, Mask R-CNN, CIA-Net) |
| Chen et al., Systematic Review on Cell Nucleus Instance Segmentation | IET Image Processing 2025 | Distance-regression vs polygon families; HoVer-Net AJI 0.775/PQ 0.764; StarDist Dice 0.809/PQ 0.784 |
| Zhang & Jiao, "Towards SAM for Medical Image Segmentation" (SAM4MIS) | arXiv 2305.03678 + living GitHub list | Foundation-model adaptation (adapters, MedSAM, SAM 2/3) |
| Review of SAM-based DL for medical segmentation | Bioengineering (MDPI) Nov 2025 | Jan 2023–Oct 2025 SAM variants |
| Ma et al., "How Distance Transform Maps Boost Segmentation CNNs" | MIDL 2020 | Boundary/HD loss/SDF comparison. DT regression as an auxiliary task *sometimes degrades* results |
| "Loss Functions in the Era of Semantic Segmentation" | arXiv 2312.05391 | Boundary-loss taxonomy |
| Maier-Hein et al., *Metrics Reloaded*; Reinke et al., metric pitfalls | Nature Methods 21 (2024) | Metric choice and aggregation pitfalls (P3.1/P3.2) |
| Isensee et al., *nnU-Net Revisited* | MICCAI 2024 | Baseline-rigour argument; ResEnc presets |

**Coverage gap:** no top-venue survey on EM domain generalization / stain robustness or on topology-aware segmentation.
Use TopoMortar (arXiv 2503.03365) and the related-work sections of clDice, Betti matching and Skeleton Recall instead.

## 3. Transferable techniques

| Technique | Venue | Evidence | Fit |
|---|---|---|---|
| **nnU-Net (+ResEnc)** | Nat. Methods 2021; MICCAI 2024 | Self-configuring; hard to beat | **Must-have backbone.** Output axon, myelin and an inter-fiber boundary class |
| **Cellpose → Cellpose3 → Cellpose-SAM**; v4.2 (Jun 2026) adds DINOv3 "cpdino" and cpsam_v2 | Nat. Methods 2021/2025; bioRxiv 2025 | Flow-field instances; robust to blur/noise | Fiber-level (outer contour) instances of touching rings; needs an inner-contour step |
| **StarDist** | MICCAI 2018 | Star-convex polygons | **Dual-radius** (inner + outer per ray) gives nesting by construction |
| **HoVer-Net / CellViT** | MedIA 2019 / 2024 | H/V distance maps split touching nuclei | Transfers to separating abutting myelin |
| **μSAM** | Nat. Methods 22 (2025) 579–591 | LM/EM generalists + automatic instance segmentation; > 17k images, > 2M objects | Baseline and annotation accelerator; EM model trained on organelles/neurites, not rings |
| MedSAM / SAM 2 adapters | Nat. Commun. 2024; ICLR 2025 | Prompt-based | Thousands of prompts per image is impractical. Use as an encoder only |
| **EM SSL pretraining**: CEM500K (eLife 2021), RETINA (PLoS CB 2025), GAN pretraining (CVPRW 2024) | Peer-reviewed | Unlabeled-EM pretraining improves transfer; SEM features help TEM | Staining robustness and label efficiency |
| **clDice** | CVPR 2021 | Tubular topology | Myelin is an annulus; use a skeleton variant on the ring to penalize breaks |
| **Skeleton Recall** | ECCV 2024 | 5.6 vs 16.2 GiB (clDice), but 7.1 vs 3.1 h nnU-Net training (TopoMortar Table 1) | Cheap ring-continuity penalty; best with noisy labels (TopoMortar) |
| **Betti matching** (Stucki et al.) | ICML 2023; 3D 2024 | Persistent-homology matching with spatial correspondence | Metric/loss for "one hole per ring". Costly at full resolution; apply on fiber crops |
| **Topological interaction** (Gupta et al.) | ECCV 2022 oral | Enforces **containment and exclusion** via impermissible adjacent-label combinations; convolutional | **Most directly relevant:** axon ⊂ myelin; axon never touches background. Code: TopoXLab/TopoInteraction |
| **Boundary loss** (Kervadec et al.) | MIDL 2019 / MedIA 2021 | ISLES: GDL Dice 0.511 → 0.644 (2D) / 0.659 (3D); HD95 5.320 → 4.795 mm (2D) | Targets HD95 |
| **HD loss** (Karimi & Salcudean) | IEEE TMI 2020 | Differentiable HD via DT | Targets HD95; watch compute |

## 4. Gaps to target

- **Nesting/topology:** no axon/myelin method enforces "exactly one ring per axon"; pairing is post-hoc.
  Containment constraints (ECCV 2022) and Betti matching have never been applied to axon–myelin units.
- **Touching fibers:** separated by watershed heuristics (MicroStructFormer, gACSON) or by hand (MyelTracer). ASTIH now
  provides delineated boundaries to learn from.
- **Inner boundary precision:** myelin Dice < axon Dice since 2018; g-ratio R² 0.805 even when diameters are > 0.96.
  The inner tongue and periaxonal space bias the "axon" definition (AimSeg), so align with the organizers' annotation convention.
- **Generalization:** the generalist reaches 0.697/0.706 on OOD macaque TEM, despite training on TEM1, SEM1, CARS1 and BF1–3.
- **Label efficiency:** in prior work, active learning reached max Dice with 3 uncertainty-selected vs 15 random samples.

## 5. Recommendations (from the review)

**Baselines (in order):** see [04](04-baselines.md).
**Evaluation:** see [05](05-evaluation-framework.md).
**Research directions and venues:** see [06](06-research-directions.md).
**Datasets:** see [07](07-datasets.md).

## Caveats carried over

- The ASTIH PDF was not read, so its baseline metrics, splits and per-dataset fiber counts are unknown. Its README says "four datasets /
  60,000 fibers" while the paper says "five / 69,000". We follow the paper.
- Preprints: EM-SAM, Cellpose-SAM, SnapG, MicroStructFormer and the multi-domain generalist. EM-SAM's "0.984" metric is unclear.
- Venue details for MyelTracer, StarDist, HoVer-Net, CellViT, clDice and MedSAM come from background knowledge, not fetched pages.
