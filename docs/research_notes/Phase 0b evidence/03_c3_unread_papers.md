# The three C3 papers left unread on 14 Nov (D1), read 15 Nov

## Pototzky, Kirschner, Sultan, Schmidt-Thieme. "Training Object Detectors if Only Large Objects are Labeled." BMVC 2021
**Local copy:** `docs/references/papers/pototzky2021_bmvc_only_large_objects_labeled.pdf` (+ .txt). Read: abstract, intro, related work, method summary, Table 1, conclusion.

- **Setting:** labels kept only for the largest X% of objects (CityPersons, EuroCityPersons, COCO). This is **size-dependent omission, simulated by the authors.** It isn't deliberate exclusion by a human protocol, but it is the only non-random omission study we have found.
- **Method:** PCIS = pseudo-labels + output consistency across scales + an anchor-scale-dependent ignore strategy.
- **Key numbers** (CityPersons miss rate with the largest 50% labelled; lower is better):

  | Method | Miss rate |
  |---|---|
  | Naive supervised | 37.2 |
  | Data distillation | 31.5 |
  | **Oracle ignore** (anchors overlapping *every* missing object ignored) | **21.7** |
  | Unbiased teacher | 16.3 |
  | PCIS | 13.8 |
  | Lower bound (100% labels + ignore) | 11.7 |

- **Lessons for C3:**
  1. A perfect ignore of the missing objects cuts the error a lot (37 → 22), which supports D-18's hard ignore.
  2. **Pseudo-labelling beats even oracle ignoring.** So "ignore" is not the ceiling of what C3 can do. Pseudo-labelling candidate fibers as positives is a stronger baseline, but it conflicts with the protocol for *deliberately* excluded abnormal fibers (see the caveat below).
  3. Their related work says earlier pseudo-label attempts for missing annotations "slightly degraded performance" without the scale trick. Results depend on the setting.

## Vădineanu, Pelt, Dzyubachyk, Batenburg. "Reducing Manual Annotation Costs for Cell Segmentation by Upgrading Low-Quality Annotations." J Imaging 10(7):172, 2024
Read in full (HTML, MDPI) 15 Nov. No PDF kept.

- An "upgrade network" learns to map perturbed labels to clean ones, trained on a few clean images. Perturbations: omission, inclusion, and boundary bias.
- **Omission is random:** a random subset of cell instances is deleted (omission rates of 30–70%). It is not tied to cell type or appearance.
- **Data:** synthetic HL60/granulocyte, EPFL hippocampus FIB-SEM (mitochondria), Lizard (colon histology).
- **Result:** heavily perturbed labels go from Dice 0.3 to 0.9.
- **Use for us:** it confirms the 14 Nov verdict. Correction methods are tested on random omission only. It is a possible "label repair" baseline family, but it needs a perturbation model of *our* omission, which is selective. It does not threaten C3's novelty.

## Wang, Ye, Liao, Zhang, Zhang, Wang G. "ReCo-I2P: An Incomplete Supervised Lymph Node Segmentation Framework Based on Orthogonal Partial-instance Annotation." MICCAI 2025, LNCS 15972:507–517
Read: abstract, the 3 reviews, the rebuttal and the meta-review (MICCAI open-access page). Full PDF download failed twice (server timeout).

- **Partial-instance annotation (PIA):** only some lymph-node instances are labelled, and only on two orthogonal slices. Co-teaching plus prototype feature enhancement. Evaluated on LNQ2023 and CT Lymph Node.
- **How they made the partial labels:** instances kept **by size** ("larger instances more likely to be retained"). Again simulated and size-based, not protocol-driven.
- **Reviewer point:** most baselines came from noisy-label learning, not incomplete-label learning. They cite **TIA** (Ju et al., MICCAI 2024, target-level incomplete annotation) as a related setting that needs background slices.
- **Use for us:** not a competitor to C3. Cite it under partial-instance annotation. The method is for 3D CT and sparse slices.

## Caveat surfaced by this reading (affects H3 and the C3 design)

Our unlabelled fibers are of **two kinds** ([02](../../notes/02-challenge-and-data.md) §3):
1. **Random misses** of normal fibers (~4–8%). These should be positives.
2. **Deliberate exclusions** of abnormal fibers (~25% of ring-like structures). Under the challenge protocol these are *negatives*.

Ignoring or pseudo-labelling both kinds alike would teach the model to predict abnormal fibers, which **lowers official precision**. H3's test set (overlap-disagreement fibers: labelled in one tile, background in the other) contains only kind 1 by construction, because if the fiber was labelled in the other tile, it is protocol-eligible. So H3 measures recovery of random misses. **The pilot must also report false positives on protocol-excluded fibers:** the abnormal ring-like structures left as background (0), split into the taxonomy's `unlabelled_real` and `other_bg` classes. Interior label-1 fibers are ignored by the metrics, so they can't show this. That tells us whether the method unlearns the protocol. Proposed as an amendment to T-15′: an added secondary readout, without changing H3.

## Update, 15 Nov (D-34)
- **C3 also has an outside test:** on TEM1, TEM2 and AxonPath, thin-myelin fibers are labelled, so C3 vs B1 recall on them is a known-positive readout (H8, `07-plan.md` §7.5). It covers thin myelin only; the outside sets also exclude degenerating fibers.
- **Evaluation rule:** Parikh et al. 2026 (arXiv) show an "observed-label illusion": a correction looks like failure on the same biased labels. C3 is therefore judged on known positives, and official precision is reported separately.
- **Pseudo-label arm:** optional, with Bukas-style harmonisation. Hard ignore stays primary.

