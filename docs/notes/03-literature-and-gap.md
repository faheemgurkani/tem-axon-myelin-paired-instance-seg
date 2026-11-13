# 03 — Prior work and the research gap

This is the short version. Full evidence and links are in the two deep-research reports:
- `docs/reports/TEM axon myelin segmentation review.md` (report A, 11 Nov): SOTA, datasets, baselines, evaluator defaults;
- `docs/reports/Paired axon myelin gap validation.md` (report B, 12 Nov): a stress test of our gap claims.

Their source notes are in `docs/research_notes/`. Downloaded papers are in `docs/references/papers/` (local).

## 1. How the field segments axons and myelin today

| Tool | How fibers are formed | Pairing |
|---|---|---|
| AxonDeepSeg (Sci Rep 2018) and its generalist; ASTIH baselines (MIDL 2026) | U-Net / nnU-Net semantic masks → watershed seeded at axon centroids | Rule |
| AimSeg (PLoS CB 2023) | ilastik + Fiji; fibre ⊃ inner region ⊃ axon hierarchy | Rule |
| gACSON / DeepACSON (3D EM) | Semantic myelin → axon instances by watershed | Rule |
| MicroStructFormer (bioRxiv 2026), AxonID (2026), Won et al. (Sci Rep 2026) | Semantic → watershed / nearest-sheath / centroid matching | Rule |
| **AxonPath** (code Sep–Oct 2026, preprint pending) | **One U-Net, three heads**: semantics + fibre distance + axon distance | Rule: an inner cylinder joins the fibre containing ≥ 90% of it |
| Optic-nerve dual decoder (Ophthalmology Science 2026) | One encoder, two decoders: inner (axon) and outer (fiber) masks | Not stated. g-ratio R² 0.805 while diameters exceed 0.96 |
| MyeliMetric (ASN Neuro 2025) | No segmentation; statistics only | — |

**Common pattern:** pixel losses, then a post-hoc pairing rule, then g-ratio computed afterwards.
- No axon/myelin paper trains on g-ratio error, within our search.
- Nobody reports whether each axon was paired with its own ring.

**Generalisation is weak:** AxonDeepSeg's multi-domain generalist reaches only axon Dice 0.697 / myelin 0.706 on out-of-domain macaque TEM. Cross-lab robustness is one of the organisers' two stated gaps.

**Strong-baseline lesson** (*nnU-Net Revisited*, MICCAI 2024): a well-configured nnU-Net beats most "new architecture" claims. Architecture novelty is the wrong lever.

## 2. Building blocks that already exist elsewhere (prior art we must cite, not claim)

| Idea | Prior art |
|---|---|
| Nested polar contours, containment by construction, *one object* per crop | Alblas et al. 2022 (vessel wall); **NPS-Net** 2026 (r_cup = α·r_disc, the closest to a "per-ray g-ratio") |
| Two contours per instance, *many* instances | Bi-Polar Mask (ICIP 2022; paywalled, still unread); HydraStarDist / HSD-WBR 2025 (containment encouraged, not guaranteed) |
| Instance first, parts second | Part-aware panoptic segmentation (CVPR 2021) |
| Cell + nucleus pairing | InstanSeg, Mesmer: also post-hoc rules |
| Ratio-of-soft-areas loss | Meng et al. MICCAI 2022 (cup-to-disc), Kervadec et al. 2021. Gains are modest and confounded |
| Containment / topology losses | TopoInteraction (ECCV 2022), clDice, Betti matching, Skeleton Recall |
| Other techniques considered, deferred (D-23) | Boundary / HD losses (Kervadec 2019, Karimi 2020), SSL pretraining on unlabelled EM (CEM500K), μSAM, Cellpose-SAM fine-tuning, StarDist / HoVer-Net distance heads |

## 3. The gap as validated (12 Nov, report B)

| Claim | Verdict |
|---|---|
| ASTIH's Dice 0.97/0.92 vs fiber F1 0.79 shows pairing is unsolved | **Reword.** F1 measures fiber *detection*; it never checks the axon↔ring assignment. The same pipeline gets 0.949 on TEM2 |
| Every tool pairs with post-hoc rules | **Holds** |
| No model predicts axon and ring jointly | **False** (AxonPath, optic-nerve decoder) |
| No one trains on g-ratio error | Holds in this domain; the idea exists elsewhere (Meng 2022) |
| No dual-radius StarDist exists | Literally true but misleading (NPS-Net, Bi-Polar Mask) |

**What remains novel** (a combination, not a single piece):
- a dense, single-shot decoder for many touching ring-shaped objects, where each fiber's axon and ring are a paired, nested partition **by construction**;
- a per-instance g-ratio loss under full supervision;
- a **pairing-correctness metric** that exposes errors fiber F1 hides.

**Do not claim:** first ratio loss, first nested polar representation, first joint axon–myelin network.

## 4. How our own results (days 1–5) change the gap

See [05](05-results.md) for the numbers.

1. **The "pairing is the bottleneck" motivation fails on this dataset.**
   - With an nnU-Net whose patch fits several fibers, plus a distance-based watershed, separation is 0.6% of fiber errors.
   - The pairing rule is lossless on perfect pixels.
   - The detection argument for a nested decoder (C2) is therefore gone here (H1 = NO-GO).
   - It may still hold on harder data (TEM1 cuprizone, AxonPath's other nerves). That is H4, untested.
2. **Selection is the real detection problem.**
   - It is the only thing human annotators disagree on, and 78% of B1's errors.
   - Annotation protocols that skip abnormal fibers are not modelled by any tool we found.
   - This supports **C1**, an error-decomposition evaluation that no benchmark reports, and **C3**, learning from incompletely annotated fibers.
3. **A new, narrower morphometry gap (H5, post hoc).**
   - Grouping and ring topology barely affect detection, but they account for ~40% of B1's g-ratio error.
   - An instance-first grouping of the *same* pixels already lowers g-MAE.
   - If confirmed under a pre-written rule, the nested-by-construction idea comes back, motivated by **morphometry**, not detection. That fits the field's known weakness: g-ratio is the worst-reproduced measurement (optic-nerve R² 0.805).
