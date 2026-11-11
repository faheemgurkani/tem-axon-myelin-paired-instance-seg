# Nested / hierarchical / part-aware instance segmentation outside the axon–myelin domain: novelty threat check (status 12 Nov 2026)

Scope: is "instance-first, part-second, nested-by-construction paired decoding for annular objects" new, and is a "dual-radius" polar head (K outer radii, r_in = sigmoid·r_out) new? PDFs saved to `docs/references/papers/`: `alblas2022_carotid_polar_2112.01137.pdf`, `degeus2021_part_aware_panoptic_2106.06351.pdf`, `gillert2023_inbd_2212.03022.pdf`.

**Headline correction to the prior review.** The prior review called a dual-radius star-convex head with containment by construction "proposed here and not yet published" ([instance_segmentation_techniques.md, Q4/Q6](../TEM%20axon%20myelin%20segmentation%20review/instance_segmentation_techniques.md)). That is **too strong**. The core idea already exists as a *single-object* polar regressor in vascular MRI:
- Alblas et al. (SPIE MI 2022) predict lumen radius plus a non-negative wall thickness per angle, so r_outer ≥ r_lumen by construction.
- Chen et al. (IEEE Access 2020) also nest two polar contours.
- Bi-Polar Mask (ICIP 2022) predicts two polar contours from one shared center *per instance*, for cell + nucleus.

What has not been found is the multi-instance, dense (StarDist/PolarMask-style, per-pixel candidate + NMS) version, in which every candidate carries both contours with a guaranteed-contained inner contour, applied to annular objects.

---

## Q1. Cell + nucleus joint instance segmentation: does anything guarantee containment by construction?

### Takeaway
No cell + nucleus method found guarantees nucleus ⊂ cell by construction at the instance level. HydraStarDist/HSD-WBR uses a soft penalty. Mesmer makes nuclear and whole-cell masks independently and pairs them after the fact by maximum overlap. Bi-Polar Mask pairs the two contours by construction through a shared center, but containment is not shown to be guaranteed (full text not read). Pairing by construction exists (Bi-Polar Mask). Containment by construction for dense multi-instance cell/nucleus does not, as far as found.

### Cited Findings
- **HydraStarDist / HSD-WBR** (De, Urbanski, Yakimovich; arXiv 2504.12078v1, Apr 2025; preprint, no venue named).
  - Architecture: a StarDist backbone with a shared encoder and two decoder branches, one per object class, each predicting object probability and K = 32 rays. Both models "integrate the knowledge of spatial correlation implicitly by using a joint encoder."
  - Containment: HSD-WBR adds the Within-Boundary Regularisation penalty Λ = (1 + ε − Σ(ŷ²⊥·ŷ¹)/Σ y²⊥)⁻¹, where ŷ²⊥ is the inverted predicted outer mask. Quote: "Here, we penalise the prediction of inner nested object instances" outside the outer object. Total loss L′ = ΣL₁ + ΣL₂ + λ₃Λ. So containment is **penalised, not guaranteed**.
  - Pairing: each branch runs its own StarDist NMS. No explicit inner–outer pairing step is described.
  - Results (JTPR inner/outer):
    - HeLaCytoNuc: HSD-WBR 0.898/0.855, StarDist 0.817/0.771, Cellpose 0.820/0.773.
    - VACVPlaque: HSD-WBR 0.784/0.849 vs StarDist 0.833/0.841, so plain StarDist is better on inner objects there.
  - Code: "will be released … under MIT" with the camera-ready version. No URL found.
  - Source: [arXiv 2504.12078 (HTML)](https://arxiv.org/html/2504.12078)
- **Bi-Polar Mask** (Zhao & Zhou, IEEE ICIP 2022, pp. 3421–3425, DOI 10.1109/ICIP46576.2022.9897272; peer-reviewed). It "utilizes two polars to represent the contours of cell and nucleus" and designs "a shared center at the nuclei, which allows for straightforward binding the pair of contours". The setting is cervical cytology with overlapping cells. **Closeness: high for the dual-contour-per-instance idea.** Containment is not shown to be guaranteed: the abstract gives no inner = fraction × outer parametrisation (full text not accessed). — [IEEE Xplore](https://ieeexplore.ieee.org/document/9897272/); [ICIP 2022 video page](https://rc.signalprocessingsociety.org/conferences/icip-2022/spsicip22vid0681); [researchr bibtex](https://researchr.org/publication/ZhaoZ22-35/bibtex)
- **Mesmer** (Greenwald et al., *Nature Biotechnology* 2022).
  - Architecture: ResNet50-FPN with "four prediction heads (two for nuclear segmentation and two for whole-cell segmentation)". Each pair is a pixel-wise interior/boundary/background transform plus an inner-distance transform, followed by marker-based watershed.
  - Pairing: the nuclear and whole-cell masks are produced independently. Downstream, each cell is matched to its nucleus by "maximum overlap", and nuclei larger than their cell are split. So there is **no containment by construction**.
  - Source: [PMC9010346](https://pmc.ncbi.nlm.nih.gov/articles/PMC9010346/)
- **DoNet** (Jiang et al., CVPR 2023): models the nucleus-in-cytoplasm relation through cell-attention-guided proposals to handle overlapping cytology. The relation is encouraged by design, not guaranteed (search summary; mechanism not verified from full text). — [CVF open access](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_DoNet_Deep_De-Overlapping_Network_for_Cytology_Instance_Segmentation_CVPR_2023_paper.pdf)
- **InstanSeg**: the authors extended it to "simultaneous segmentation of nuclei and cells" in multiplexed fluorescence via a channel-invariant network. The nucleus–cell pairing mechanism was not found (bioRxiv 10.1101/2024.09.04.611150 returned 403). — [arXiv 2408.15954](https://arxiv.org/abs/2408.15954); [Edinburgh thesis record](https://era.ed.ac.uk/bitstreams/400f1e39-5a4f-4392-a3f6-e92cb924a63b/download)

### Inferences
- The closest cell + nucleus prior art for our decoder:
  - Bi-Polar Mask: one center, two polar contours per instance, so pairing by construction.
  - HSD-WBR: nested star-convex instances, but containment by penalty.
- Our claim must be narrowed to **containment guaranteed by parametrisation** (r_in = σ·r_out from one shared center/candidate), which neither paper claims. Bi-Polar Mask has to be cited and differentiated: shared center at the nucleus in overlapping cytology, versus a shared center inside the axon with a non-overlapping ring.
- Annular objects add something new compared with cell/nucleus: the "outer minus inner" region *is* the second instance (myelin), and the two masks partition the fiber. Cell/nucleus methods treat the nucleus as a sub-object, not as a hole that defines the ring.

### Gaps
- Bi-Polar Mask full text, numbers and code: not accessed (IEEE paywall). Whether its nucleus rays are constrained to lie inside the cell rays is **unverified**. This is the single most important item to check before submission.
- Cellpose multi-channel, CellSAM and DeepCell-v2 nucleus–cell pairing mechanisms were not re-verified here. The prior review states Cellpose cyto + nuclei is a two-model pattern without containment.
- No quantitative containment-violation rates were found for any cell + nucleus method.

---

## Q2. Polar / star-convex multi-contour ("dual-radius", ring, annulus, concentric) representations

### Takeaway
**Dual-radius polar regression with containment by construction already exists for single, pre-centered objects in vessel-wall MRI.** Alblas et al. 2022 predict r_lumen and a thickness per angle, with r_outer = r_lumen + thickness. Chen et al. (IEEE Access 2020) nest two polar contours. Tree-ring INBD (CVPR 2023) iterates polar boundaries outward from the previous ring. No dense multi-instance StarDist/PolarMask variant with per-ray inner + outer radii and guaranteed containment was found.

### Cited Findings
- **Alblas, Brune, Wolterink, "Deep learning-based carotid artery vessel wall segmentation in black-blood MRI using anatomical priors", SPIE Medical Imaging 2022** (arXiv 2112.01137; peer-reviewed proceedings).
  - Framing: "we cast vessel wall segmentation as a multi-task regression problem in a polar coordinate system".
  - Mechanism, quoted from Section 3.2: "We describe both contours in polar coordinates in terms of local radii at fixed angles [φᵢ]… and use a multitask CNN to simultaneously estimate radii values for both contours at each angle… To obtain non-intersecting contours, we define r_outer wall as the sum of r_lumen and the local vessel wall thickness, and let the CNN predict the latter, so that r_lumen ≤ r_outer wall."
  - The abstract claims an "inductive bias that guarantees ring-shaped vessel walls".
  - Setup: N = 31 angles, rays of length 127. A centerline from Dijkstra on a learned cost map supplies **one center per artery per slice**. This is not a dense detector.
  - Source: [arXiv 2112.01137](https://arxiv.org/abs/2112.01137); local PDF.
- **Thibeau-Sutre, Alblas et al. (arXiv 2308.09538, 2023)**: an uncertainty-based QA follow-up on the same nested-wall model. It reports that the polar approach degrades when "the center found during the first step" is not enclosed in the lumen. — [arXiv 2308.09538](https://arxiv.org/abs/2308.09538)
- **Chen L. et al., "Automated Artery Localization and Vessel Wall Segmentation Using Tracklet Refinement and Polar Conversion", IEEE Access 8:217603–217614 (2020)**.
  - Alblas et al. write: "similar to the work by Chen et al., we nest two circles without intersection to obtain a ring-shaped segmentation mask".
  - The arXiv version (1909.02087) extracts lumen and outer-wall contours from the max/min radial gradient of a polar probability map plus a snake. Nesting comes from post-processing, not from parametrisation (search summary).
  - Source: [arXiv 1909.02087](https://arxiv.org/abs/1909.02087)
- **Zhou, Xiao, Li, Fan, Ruan, "Intracranial vessel wall segmentation … tiered loss incorporating class inclusion", *Medical Physics* 49(11) 2022**.
  - A single-channel, level-set-like output y is thresholded at η1 < η2, giving lumen ⊂ whole vessel **by construction at the semantic level**. Quote: "the set of lumen pixels should be a proper subset of the whole vessel pixel set."
  - Polar per-direction regression of lumen and whole-vessel boundaries is used as a benchmark, not as the proposed method.
  - Source: [PMC9742123](https://pmc.ncbi.nlm.nih.gov/articles/PMC9742123)
- **INBD: Gillert et al., "Iterative Next Boundary Detection for Instance Segmentation of Tree Rings in Microscopy Images of Shrub Cross Sections", CVPR 2023** (code: github.com/alexander-g/INBD).
  - Explicitly treats donut-shaped "hollow" instances: "in the hollow (h) mode, objects are defined as single calendar years and are donut-shaped (with a hole)". It notes a baseline "is used because it cannot model hollow objects".
  - Mechanism: "We sample on a polar grid starting from the previous detected ring boundary", with ρ from ρ₀ (the previous boundary) to ρ_N. "Polar coordinates impose a prior, ensuring a coherent (quasi-)convex shape".
  - So each new ring lies radially outside the previous one by construction. There is one concentric object family per image, not many independent fibers.
  - Source: [arXiv 2212.03022](https://arxiv.org/abs/2212.03022); local PDF.
- **IVUS lumen / media-adventitia**: pixel-mask methods (IVUS-Net, IVUS-U-Net++). A MedIA 2023 paper approximates lumen and media by ellipses. None of these was found using per-A-line radii with a positivity constraint. — [arXiv 2102.10480](https://arxiv.org/abs/2102.10480); [IVUS-Net arXiv 1806.03583](https://arxiv.org/abs/1806.03583)
- **Polar instance-segmentation family** (PolarMask/PolarMask++, ESE-Seg, SplineDist, MultiStar, CircleSnake, Eigencontours, 2026 polygon DETR 2603.09245): all are single-contour per object. None of them represents holes. — [PolarMask++ 2105.02184](https://arxiv.org/abs/2105.02184); [MultiStar 2011.13228](https://arxiv.org/abs/2011.13228); [ESE-Seg 1908.04067](https://arxiv.org/abs/1908.04067); [Polygon DETR 2603.09245](https://arxiv.org/html/2603.09245v1)
- A targeted 2025–2026 search for per-ray inner + outer radii for rings, annuli or myelin found no matching paper beyond HSD (arXiv 2504.12078). — search, 12 Nov 2026

### Inferences
- "Dual-radius polar head" is **not novel as a representation**. Alblas 2022's r_outer = r_lumen + softplus/ReLU-type thickness is mathematically equivalent in purpose to our r_in = σ·r_out: both guarantee nesting along each ray.
  - Ours parametrises inner as a fraction of outer. That makes the per-ray ratio a direct g-ratio and is scale-invariant. Theirs parametrises outer as inner plus thickness.
  - These are design variants, not a conceptual gap.
- What remains defensibly novel (pending the Bi-Polar Mask full text):
  - (a) Embedding the nested dual-radius parametrisation in a **dense, multi-instance, bottom-up detector** (StarDist-style per-pixel candidates + NMS on the outer polygon). The pair then shares one NMS decision, which yields paired instance IDs for many touching annular objects per image. Alblas uses one externally supplied center per vessel, INBD one concentric family per image, and HSD two independent heads with a penalty.
  - (b) Using the fraction parametrisation for an interpretable per-ray g-ratio.
  - (c) Application to TEM fibers.
- The paper must cite Alblas 2022 (and Chen 2020) as "nested polar contours by construction for single vessels", and present ours as its multi-instance generalisation. Claiming the representation itself as new would be refuted by a reviewer from medical imaging.

### Gaps
- Alblas' exact activation for thickness non-negativity is not stated in the extracted text. The guarantee rests on "let the CNN predict" thickness, so non-negativity is implied by the claim r_lumen ≤ r_outer.
- Optic disc + cup joint polar (M-Net) and cardiac endo/epicardium polar regressors were not checked in this pass. Myocardium polar regressors are another likely "outer = inner + thickness" prior art.
- Materials-science rings (fibers in composites, bubbles, wires) were not searched.

---

## Q3. Part-aware panoptic and hierarchical / nested instance segmentation; containment losses

### Takeaway
Part-aware panoptic segmentation gives "parts only inside their parent instance" **by construction through a top-down merge**: the instance is decided first and parts are assigned within it. Our "instance-first, part-second" decoder follows this established pattern, so it is not novel in itself. Containment-style losses (TopoInteraction, WBR) and hierarchical-label activations exist, but they are soft or semantic-only.

### Cited Findings
- **Part-aware Panoptic Segmentation (de Geus et al., CVPR 2021)**.
  - The task definition requires that a part-level class "must be assigned to – and only to – all pixels that are assigned" a scene-level thing class with parts.
  - Baseline merge: "a straightforward top-down, rule-based merging approach … for each segment in the panoptic segmentation prediction that does require an additional part label, we identify and extract the part predictions for the pixels corresponding to this segment", prioritising "panoptic segmentation over part segmentation".
  - So part ⊂ instance holds **by construction (post-hoc merge)**.
  - Source: [arXiv 2106.06351](https://arxiv.org/abs/2106.06351); local PDF.
- **Panoptic-PartFormer** (ECCV 2022) and **PanopticPartFormer++**: things, stuff and parts are all object queries trained as one mask-classification problem. The ++ version adds "part-whole cross-attention" via masked cross attention. **JPPF** (Muralidhara et al.) is a "parameter-free joint fusion module" for consistent panoptic-part output. — [arXiv 2301.00954](https://arxiv.org/abs/2301.00954); [arXiv 2311.18618](https://arxiv.org/abs/2311.18618); [DFKI JPPF](https://av.dfki.de/?p=14090)
- **TopoInteraction** (Gupta, Hu, Chen et al., ECCV 2022 oral, LNCS doi 10.1007/978-3-031-19818-2_40, pp. 701–718).
  - A convolution-based module encoding "containment and exclusion" between *semantic* classes, trained with an extra loss that penalises violations. That makes it soft.
  - Code: TopoXLab/TopoInteraction. **Venue verified**, confirming the prior review.
  - Source: [arXiv 2207.09654](https://arxiv.org/abs/2207.09654); [mlanthology](https://mlanthology.org/eccv/2022/gupta2022eccv-learning)
- **Multi-level activation** (arXiv 1804.01910; ECCV-W 2018): semantic nesting by construction. Verified in the prior review and not re-fetched. — [arXiv 1804.01910](https://arxiv.org/abs/1804.01910)

### Inferences
- "Instance-first, part-second" is directly prior art from PPS. Containment of the part in its parent instance is by construction there too.
- Applying it to two *nested* parts (core + ring), where the inner part's complement inside the parent is the second part, is an application of PPS rather than a new principle.
- The novelty is the coupling with a nested *geometric* parametrisation and the paired-ID output format.

### Gaps
- Hierarchical "instance-of-instance" methods in biology (organelle-in-cell graphs, mitochondria–cristae) were not searched in this pass.

---

## Q4. Annular / hollow-object instance segmentation in biomedical imaging

### Takeaway
Hollow-object instance segmentation is rare, and the only explicit "donut instance" deep method found is INBD (tree rings, CVPR 2023). Vessel-wall rings are handled as single-object nested polar regression. Glands, hair follicles, plant cell walls, nerve fascicles with perineurium and muscle fibers were not found to have nested-by-construction instance decoders, though this pass did not exhaustively search them.

### Cited Findings
- INBD: hollow tree-ring instances, polar iterative boundary detection, nesting by construction. — [arXiv 2212.03022](https://arxiv.org/abs/2212.03022)
- Alblas 2022 and Chen 2020: ring-shaped vessel wall from two nested polar contours, one vessel per center. — [arXiv 2112.01137](https://arxiv.org/abs/2112.01137)
- A 2025 narrative review of nuclei and gland segmentation mentions a dual-stream gland + lumen model with rotation-equivariant convolutions. That model is not polygonal and not nested by construction (search summary only). — [arXiv 2208.12460](https://arxiv.org/abs/2208.12460)

### Inferences
- The axon–myelin unit (one hole per ring, many densely packed rings, each pair requiring paired IDs) sits between INBD (many nested rings, one center) and HSD/Bi-Polar Mask (many instances, two contours, no guaranteed nesting). No method found covers that cell of the design space.

### Gaps
- Glands, nerve fascicles/perineurium, plant cell walls and muscle fibers were not specifically searched (tool budget).

---

## Q5. Bottom line, and what the paper must cite

### Takeaway
- **Dual-radius polar head:** NOT novel as a representation (Alblas et al. SPIE 2022: r_outer = r_lumen + predicted thickness, "guarantees ring-shaped vessel walls"). Novel only as a dense multi-instance StarDist/PolarMask-style head with fraction parametrisation (r_in = σ·r_out) and a shared NMS, applied to many touching annular objects. That is a moderate, defensible delta.
- **Instance-first, part-second nested decoding:** NOT novel as a principle (PPS top-down merge, CVPR 2021; Panoptic-PartFormer).
- **Combination:** the claim "nested-by-construction paired instance decoding for densely packed annular objects" appears unclaimed in the literature found. It still has to be phrased as a combination or generalisation, and Bi-Polar Mask must be checked.

### Closeness ranking (closest first)

| Work | Venue / status | Representation | Containment | Code | Closeness |
|---|---|---|---|---|---|
| Alblas et al. 2022 | SPIE MI 2022, peer-reviewed | Per-angle r_lumen + thickness (N = 31), single pre-centered vessel | **By construction** | Not found | **High** (representation) |
| Bi-Polar Mask (Zhao & Zhou) | ICIP 2022, peer-reviewed | Two polar contours, shared nucleus center, multi-instance | Pairing by construction; containment unverified | Not found | **High** (multi-instance dual polar) |
| HydraStarDist / HSD-WBR | arXiv 2025 preprint | Two StarDist heads, joint encoder | Penalty (WBR) | Promised, MIT; no URL | **High** (nested star-convex instances) |
| INBD | CVPR 2023 | Iterative polar boundaries from previous ring; hollow instances | By construction (outward sampling) | github.com/alexander-g/INBD | Medium |
| Chen et al. | IEEE Access 2020 | Polar probability map + gradient + snake, nested circles | Post-processing | Not found | Medium |
| Part-aware Panoptic Seg. | CVPR 2021 | Top-down merge of parts inside instances | By construction (merge) | Yes (panoptic_parts) | Medium (decoder principle) |
| Panoptic-PartFormer(++) / JPPF | ECCV 2022 / journal; WACV-class | Query-based parts + wholes / fusion | Soft / fusion | Yes | Low–medium |
| Zhou et al. tiered loss | Med Phys 2022 | Single-channel level-set thresholds | Semantic by construction | Not checked | Low–medium |
| Mesmer | Nat Biotech 2022 | Independent nuclear + cell heads, watershed, max-overlap match | None (post-hoc) | DeepCell | Low |
| TopoInteraction | ECCV 2022 | Containment/exclusion loss on semantic labels | Penalty | TopoXLab/TopoInteraction (MIT) | Low |
| Multi-level activation | ECCV-W 2018 | Nested-class activation | Semantic by construction | — | Low |

### Inferences
- Must-cite closest prior art: Alblas et al. 2022 (plus Chen 2020), Bi-Polar Mask (ICIP 2022), HydraStarDist/HSD-WBR (arXiv 2504.12078), INBD (CVPR 2023), Part-aware Panoptic Segmentation (CVPR 2021), StarDist/PolarMask, TopoInteraction (ECCV 2022), and Mesmer as the post-hoc pairing contrast.
- A suggested novelty sentence: "Unlike single-vessel polar regressors that nest an outer contour on a pre-located center [Alblas 2022], and multi-instance two-contour models that pair cell and nucleus without guaranteeing containment [Bi-Polar Mask; HSD-WBR], we predict, densely and for every candidate, an outer star-convex contour and an inner contour as a per-ray fraction of it, so that ring/core pairing and containment hold by construction for all touching annular instances in a single shot."

### Gaps
- The Bi-Polar Mask full text is the biggest residual risk. If its nucleus rays are parametrised relative to the cell rays, the multi-instance claim shrinks to "applied to annular objects with partition semantics".
- Polar myocardium (endo/epi) and optic disc/cup regressors were not checked. They may also use "outer = inner + thickness".
