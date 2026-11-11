# Bi-Polar Mask, nested-polar follow-ups, InstanSeg pairing and ratio losses: novelty-threat check (12 Nov 2026)

Scope: three narrow threats to the planned method:
- a dense multi-instance polar head with outer radius r_out per ray and inner radius r_in = σ·r_out;
- a shared NMS, giving paired inner/outer IDs with containment guaranteed;
- a per-fiber differentiable g-ratio loss |sqrt(A_in_soft / A_out_soft) − g_gt|.

This note extends `nested_instance_other_domains.md` and does not repeat Alblas 2022, HSD-WBR, INBD or Mesmer.

PDFs saved to `docs/references/papers/`:
- `goperma2026_npsnet_2604.09062.pdf`
- `meng2022_miccai_vcdr_shape_aware.pdf`
- `vanherten2024_world_of_forms_2409.11837.pdf`
- `chen2024_access_overlapping_cervical_survey.pdf`

The Bi-Polar Mask PDF could **not** be obtained.

**Headline:** a new and closer threat to the parametrisation was found. **NPS-Net** (arXiv 2604.09062, Apr 2026) uses literally r_cup(θ) = α(θ)·r_disc(θ) with α ∈ [0,1], and claims nesting "by construction". It is single-object only (one disc/cup per pre-centered crop) and is a preprint. The fraction parametrisation r_in = σ·r_out can therefore no longer be claimed as new in any form. The defensible delta narrows to: dense multi-instance + shared NMS + annular partition semantics + per-instance g-ratio loss.

---

## Q1. Bi-Polar Mask (Zhao & Zhou, ICIP 2022): are nucleus rays constrained inside cell rays? Is NMS shared?

### Takeaway
**Unverified.** The full text could not be accessed (IEEE paywall, no institutional access in the user's browser, no open-access copy anywhere). Everything verifiable says the two polar contours share one center at the nucleus, which pairs them by construction. Nothing public says the nucleus rays are parametrised relative to the cell rays (ratio or offset). Because it predicts one center and two contours per candidate, a single detection/NMS step is implied but not confirmed.

### Cited Findings
- Bibliographic record: Wanbi Zhao, Xu Zhou, "Bi-Polar Mask for Joint Cell and Nuclei Instance Segmentation", ICIP 2022 (Bordeaux, 16–19 Oct 2022), pp. 3421–3425, DOI 10.1109/ICIP46576.2022.9897272. — [IEEE Xplore](https://ieeexplore.ieee.org/document/9897272); [researchr](https://researchr.org/publication/ZhaoZ22-35/bibtex)
- Method sentences (abstract, verbatim):
  - "we propose Bi-Polar Mask that utilizes two polars to represent the contours of cell and nucleus."
  - "We also design a shared center at the nucleus, which allows for straightforward binding of the pair of contours."
  - "The polar representation in nature overcomes the challenges of object overlapping and cell folding."
  - Dataset: "A large dataset with thousands of cell and nucleus annotations was constructed for evaluation."
  - Source: [IEEE Xplore abstract](https://ieeexplore.ieee.org/document/9897272)
- Paper structure, from the Xplore outline: 1 Introduction, 2 Bi-Polar Mask, 3 Experiments, 4 Conclusion. A 5-page ICIP paper.
- The visible intro motivates the work with an "objective measure (e.g., nuclei to cytoplasm ratio [1])". The N:C ratio is the stated downstream use, which is the analogue of our g-ratio. The rest of the paper is behind "Sign in to Continue Reading". — [IEEE Xplore](https://ieeexplore.ieee.org/document/9897272)
- Access status:
  - Semantic Scholar: openAccessPdf "CLOSED".
  - OpenAlex: `is_oa: False`, `any_repository_has_fulltext: False`.
  - Google Scholar: lists only the ieee.org version ("All 2 versions").
  - The IEEE stamp/PDF endpoint redirected back to the abstract page, so no subscription.
  - Sources: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ICIP46576.2022.9897272); [OpenAlex](https://api.openalex.org/works/doi:10.1109/ICIP46576.2022.9897272)
- Code: none found (no GitHub hit, and none named in the abstract or video page). — [ICIP 2022 video page](https://rc.signalprocessingsociety.org/conferences/icip-2022/spsicip22vid0681)
- Citation footprint is tiny. 2–3 citers:
  - Xplore shows "2 Cites in Papers", Google Scholar "Cited by 3".
  - Semantic Scholar lists: a YOLOv8 underwater segmentation paper (J. Phys. Conf. Ser. 2023); Tong et al., "Cell image instance segmentation based on PolarMask using weak labels" (CMPB 2023, doi 10.1016/j.cmpb.2023.107426); and Chen et al.'s cervical overlapping-cell survey (IEEE Access 2024).
  - The survey cites Bi-Polar Mask only as reference [154], inside an image-rendered summary table. It gives no textual description of the parametrisation (checked by full-text grep).
  - Sources: [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ICIP46576.2022.9897272); [Chen et al. 2024 survey](https://doi.org/10.1109/ACCESS.2024.3445371)

### Inferences
- With a shared center at the nucleus, the plain PolarMask design would regress two independent sets of ray lengths from that center. Whether the authors constrained r_nuc ≤ r_cell cannot be confirmed either way. In overlapping cytology the nucleus is always inside its cell, so a constraint would be natural, but the abstract only claims "binding", not containment.
- **Safe paper wording:** "Bi-Polar Mask [Zhao & Zhou 2022] pairs cell and nucleus polar contours through a shared center". Do not claim that it lacks containment unless the full text is read.
- Bi-Polar Mask's center is at the inner object (nucleus). Ours is a candidate inside the inner object (axon) too, so that is not a differentiator.
- Our differentiators:
  - inner = σ·outer as an explicit per-ray ratio, with containment guaranteed;
  - the ring (outer \ inner) is itself the second instance class (partition semantics);
  - non-overlapping, densely touching annuli versus overlapping cytology;
  - a ratio loss.

### Gaps
- **Full text unread.** Unknown: the ray parametrisation (independent, offset or ratio), the NMS details, the containment loss, the dataset size, the numbers, and the baselines (presumably PolarMask/Mask R-CNN).
- Recommended action: request the PDF via inter-library loan or from the authors (Xu Zhou; affiliation not confirmed). Alternatively, ask Sir Faizan for IEEE access.

---

## Q2. Follow-ups and nested-polar / outer = inner + thickness / ratio regressors (2022 – Nov 2026)

### Takeaway
- No dense multi-instance "concentric StarDist" beyond HSD-WBR (penalty-based, already logged) was found, and no Bi-Polar Mask follow-up uses dual polar contours.
- Single-object nested-polar regressors with containment by construction are now well populated across three domains:
  - vessel wall: Alblas 2022, already logged;
  - cardiac LV: van Herten et al. 2024/25 (r_epi = r_endo + thickness); Tan et al. MedIA 2017 (endo/epi radial distances, constraint unverified);
  - optic disc/cup: **NPS-Net 2026, with the exact ratio form r_cup = α·r_disc, α ∈ [0,1]**.

### Cited Findings
- **NPS-Net** (Goperma, Basnet, Zhao, Kyoto Univ.; arXiv 2604.09062v1, 10 Apr 2026; preprint, no venue).
  - Representation: "predicts two normalised radial boundary functions" in a "disc-centered polar domain".
  - Shape-prior branch: "r_c^s(θ) = α^s(θ) · r_d^s(θ), where α^s(θ) … ∈ [0, 1] is the predicted angular cup-to-disc ratio. Since α^s ∈ [0, 1], nesting (where the cup is contained within the disc) is preserved in the prior."
  - Dense branch: cup occupancy P_c = P_d · Q, with Q = σ(·): "Since both P_d and Q lie in [0,1], the product satisfies P_c ≤ P_d everywhere for any input". The disc itself uses monotone occupancy via a cumulative sum of non-negative decrements.
  - Single object: "We assume that the input is a region-of-interest crop … centered on the optic nerve head. Define the polar anchor as (cx, cy) = (W/2, H/2)". One disc/cup per crop. No detection and no NMS.
  - Losses: Dice+BCE (Cartesian and polar), a rim-thickness loss SmoothL1(r_d − r_c, r_d* − r_c*), and shape-prior losses. **No vCDR/ratio loss.** vCDR is evaluation only.
  - Results: RIM-ONE zero-shot cup Dice 0.8064 vs. BEAL 0.6782; vCDR MAE 0.0684; 0 % nesting violations from ablation M3 onward.
  - Code: github.com/Rimsa78/NPS-Net.
  - Source: [arXiv 2604.09062](https://arxiv.org/html/2604.09062); local PDF.
- **World of Forms** (van Herten, …, Wolterink, Išgum; arXiv 2409.11837, v2 21 Feb 2025; "Submitted to Medical Image Analysis", acceptance not confirmed).
  - Ray-cast template meshing in CCTA: "two radii for each ray-cast: one for the endocardial (r_in) and epicardial (r_out) boundary … we frame this problem by predicting the distance to the endocardial boundary and the myocardial thickness (r_myo), where r̂_myo = r̂_out − r̂_in. The myocardium thickness should always be larger than 0 to obtain a topologically correct myocardium segmentation."
  - One LV per scan, template-based, not multi-instance.
  - Source: [arXiv 2409.11837](https://arxiv.org/abs/2409.11837); local PDF.
- **Tan et al., "Convolutional neural network regression for short-axis left ventricle segmentation in cardiac cine MR sequences", Medical Image Analysis 2017** (doi 10.1016/j.media.2017.04.002).
  - Per the search summary, the LV segmentation is "parameterized in terms of the radial distances between the LV centerpoint and the endo- and epicardial contours in polar space".
  - Whether epi ≥ endo is enforced by construction was **not verified** (full text not fetched).
  - Single LV per slice.
  - Source: [OpenAlex record / DOI](https://doi.org/10.1016/j.media.2017.04.002)
- **M-Net** (Fu et al., IEEE TMI 2018; arXiv 1801.00926): a polar transform plus multi-label pixel segmentation of OD/OC. It notes "the OC is restricted to the region inside OD" as a prior motivating the polar transform. It is pixel labelling, not radius regression, with no containment guarantee. — [arXiv 1801.00926](https://arxiv.org/abs/1801.00926)
- A 2024–2026 search for joint cell + nucleus polar instance methods found no new dual-polar work beyond Bi-Polar Mask and HSD-WBR. — [search 12 Nov 2026; HSD arXiv 2504.12078](https://arxiv.org/abs/2504.12078)

### Inferences
- Prior art for nested polar parametrisation (all single object):
  - **ratio form:** NPS-Net 2026;
  - **thickness form:** Alblas 2022, van Herten 2024/25;
  - **multi-instance with shared center (pairing; containment unverified):** Bi-Polar Mask 2022;
  - **multi-instance with penalty:** HSD-WBR 2025.
- The remaining gap is the combination: **dense per-pixel candidates + ratio-parametrised inner contour + one shared NMS on the outer polygon, many touching annuli per image, and inner/ring partition giving paired IDs.**
- NPS-Net must be cited as the closest prior art for "r_in = σ·r_out". It is a preprint, so it does not strictly count as prior publication for some venues, but reviewers may know it. Note that NPS-Net also calls α "angular cup-to-disc ratio", which is the direct analogue of our "per-ray g-ratio" interpretation. That interpretability claim is therefore also not new.

### Gaps
- No multi-instance optic-cup or myocardium polar method was found. Those domains have one object per image, so this is expected.
- The Tan 2017 constraint form is unverified.
- Materials science (fibers in composites, hollow fibers, bubbles) was not searched.

---

## Q3. InstanSeg nucleus–cell pairing; N:C-ratio (or analogous ratio) losses

### Takeaway
**InstanSeg pairs nucleus and cell post-hoc, not by construction.** Its output channels are split into two independent halves, each decoded separately into labels. The function `resolve_cell_and_nucleus_boundaries` then:
- matches cells to nuclei by intersection-over-nucleus-area > 0.5;
- keeps the largest nucleus per cell;
- relabels the matched cell to the nucleus ID;
- overwrites cell pixels with nucleus pixels, which **forces containment after the fact**.

No published per-instance N:C-ratio training loss was found. The closest ratio-supervision precedent is Meng et al. (MICCAI 2022): a **differentiable vCDR estimation layer** from soft area and perimeter of the OD/OC probability maps, used as weak supervision. It is a single object per image, and the ratio is computed from areas, as ours is.

### Cited Findings
- **InstanSeg code** (github.com/instanseg/instanseg, HEAD 414fbc9, 29 Sep 2026):
  - In `instanseg/utils/loss/instanseg_loss.py`, with `cells_and_nuclei`: "iterations = 2; dim_out = int(self.dim_out / 2)", and the prediction is sliced `prediction[0:dim_out]` (nuclei) and `prediction[dim_out:]` (cells). Each half is clustered independently.
  - Then, `if lab.shape[1] == 2 and resolve_cell_and_nucleus: lab = resolve_cell_and_nucleus_boundaries(lab)`.
  - Its docstring, verbatim: "It will first match the labels of the largest nucleus and its cell. It will then erase from the cell masks all the nuclei pixels. This resolves nuclei 'just' overlapping adjacent cell. It will then recover the nuclei pixels that were erased by adding them back to the cell masks."
  - Matching uses `get_intersection_over_nucleus_area` with `iou > 0.5`, and `allow_unnucleated_cells=True` by default.
  - Source: [InstanSeg repo, biological_utils.py](https://github.com/instanseg/instanseg/blob/main/instanseg/utils/biological_utils.py); [instanseg_loss.py](https://github.com/instanseg/instanseg/blob/main/instanseg/utils/loss/instanseg_loss.py)
- InstanSeg multiplexed preprint: the method "allows for the simultaneous prediction of nucleus and cell labels even when paired labels are not present in the ground truth" (search summary). — [bioRxiv 10.1101/2024.09.04.611150](https://www.biorxiv.org/content/10.1101/2024.09.04.611150.full.pdf)
- **Meng et al., "Shape-Aware Weakly/Semi-Supervised Optic Disc and Cup Segmentation with Regional/Marginal Consistency", MICCAI 2022** (Univ. Liverpool).
  - Ratio as a training signal: "a differentiable vCDR estimating module is proposed for the end-to-end training … the supervision on vCDR can serve as a weakly-supervision for OD & OC region and boundary segmentation".
  - Computation: area A_o ≈ Σ PM (soft probability sum) and perimeter P_o ≈ Σ of the detected soft boundary. The long semi-axis a comes from Euler's ellipse formulas, then "θ(vCDR) = (a_OC + e⁻⁶)/(a_OD + e⁻⁶)".
  - Code: github.com/smallmax00/Shape-Aware-Weakly-Semi-Supervised-Optic-Disc-and-Cup-Segmentation.
  - Single OD/OC per image, semantic (not instance) level.
  - Source: [Liverpool repository PDF](https://livrepository.liverpool.ac.uk/3169382/); local PDF.
- A patent (US 12,086,982) predicts the CDR with the same network and cross-checks it against the CDR computed from the segmentation (search summary only). — [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12086982)
- N:C ratio:
  - The N:C literature found treats N:C as a post-hoc metric computed from predicted masks. For example, a bladder-cytology UNet poster compares predicted vs. GT per-cell N/C but does not train on it.
  - No paper using a per-instance N:C loss was found.
  - Sources: [EDIT AI 2025 poster](https://jlevy44.github.io/EDIT_AI_Virtual_Conference_25/assets/posters/bladder.pdf); [Wikipedia N:C definition](https://en.wikipedia.org/wiki/Nuclear%E2%80%93cytoplasmic_ratio)
- g-ratio: searches found no differentiable g-ratio training loss for axon/myelin. AimSeg, Rajani 2020 and the measurement-oriented DL workflow compute g-ratio post hoc. — [AimSeg PMC10691719](https://pmc.ncbi.nlm.nih.gov/articles/PMC10691719/); [Rajani IPTA 2020](https://discovery.ucl.ac.uk/10121668/7/Rajani_1IPTA_2020_Myelin_Hal.pdf)

### Inferences
- InstanSeg is a clean contrast case: two independent instance decoders plus heuristic post-hoc matching and forced containment. Our method makes pairing and containment part of the representation. This parallels Mesmer (max-overlap matching).
- "Differentiable clinical ratio from soft areas as a loss" has precedent (Meng 2022 vCDR, semantic and single-object). Our g-ratio loss is thus a **per-instance, multi-instance version of an existing idea**, not a new loss concept.
- With the polar parametrisation, the soft areas can be computed analytically from the rays: A ≈ ½Σ r² Δθ. Then sqrt(A_in/A_out) is a smooth function of σ. This is a neat property worth stating, but it is a modest contribution.
- Cite Meng 2022 for the ratio-loss idea, and NPS-Net's rim-thickness loss as a related polar-thickness supervision.

### Gaps
- InstanSeg bioRxiv full text was not read. The pairing claim rests on the current repo code, which could differ from the paper version.
- A per-instance N:C or CDR loss in multi-instance settings might exist in non-indexed venues. None was found in 3 searches.
- No check was made for whether Bi-Polar Mask uses an N:C loss: its intro mentions N:C as the motivation, but the full text is unread.
