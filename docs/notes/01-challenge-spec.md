# 01 — Challenge Specification (verified from the website source)

**Source:** https://alexandrojim.github.io/tem-challenge/, crawled on 2026-11-10 from its GitHub Pages source repo
`alexandrojim/tem-challenge` at commit `ec9b8e3` (last commit 2026-11-02, "update researchers").
Raw markdown snapshot: [`../references/challenge-site-snapshot/`](../references/challenge-site-snapshot/).

> The site renders `content/*.md` client-side, so plain web fetchers see only "Loading…". The earlier
> literature review ([03](03-literature-review.md)) could not read `data.md` and `submission.md`. This file
> **supersedes** that review's challenge facts wherever the two disagree.

## Identity

| Item | Value |
|---|---|
| Name | TEM Axon & Myelin Segmentation Challenge ("Ultrastructural Segmentation of Axons and Myelin in Mouse Corpus Callosum TEM") |
| Host | ICM.Quant platform, Paris Brain Institute (Institut du Cerveau, ICM), with SIPAIM 2026 |
| Contact | gabriel.jimenez@icm-institute.org |
| Data donors / PIs | B. Nait-Oumesmar, B. Hassan, D. Racoceanu (ICM / Sorbonne); T. Karam, L.-M. Ghayad (PhD); K. Ourahmoun |
| Annotators | S. Rajabzadeh, K. Ourahmoun (ICM.Quant); D. Jimenez (PUCP, DSP Lab) |
| Prizes | 250 / 150 / 100 EUR. Winners must register for SIPAIM (main conf. 18–20 Nov 2026 or workshop 16 Nov 2026) |
| Sponsor | MathWorks (temporary MATLAB licences for registrants) |

## Task

**Paired instance segmentation** of myelinated fibers. For each fiber, segment:
- the **axon** (inner, light axoplasm) as instance `X` (even, ≥ 2)
- its **myelin sheath** (dark concentric ring) as instance `X+1` (the next odd number)

The output must make morphometry possible per fiber: axon diameter, myelin thickness, fiber diameter and g-ratio.
The organizers name two gaps: (1) poor generalization across tissue preparation, staining and imaging, and
(2) failure to keep the axon-in-myelin nesting.

## Dataset (`data.md`)

| Item | Value |
|---|---|
| Modality | TEM, imaging mode, AMT NS12 camera |
| Tissue | Mouse (*Mus musculus*) corpus callosum (white matter), ICM |
| Count | **400 images = 266 train (annotated) + 134 test (unannotated release)** |
| Size | **4096 × 3008 px**, single-channel grayscale |
| Acquisition | 100 kV, 3000× direct magnification, 480 ms × 4 averaged frames, binning 1, gain 1, gamma 1.0, no sharpening |
| Pixel size | **0.002152 µm/px (≈ 2.15 nm/px)**; the 1 µm scale bar is ≈ 465 px. Note that a **scale bar is burned into the images** |
| Download | ICM OneDrive (~4–5 GB train), linked from the site's Dataset section |
| Intended use | Supervised, semi-supervised or self-supervised methods are all explicitly allowed |

### Label encoding (`uint16`)

| Value | Meaning |
|---|---|
| `0` | Background / extra-axonal space (includes **unmyelinated** tissue) |
| `1` | Artifacts. **Ignored** in training loss and evaluation |
| `X` even ≥ 2 | Axon instance |
| `X+1` odd ≥ 3 | Myelin sheath paired with axon `X` |

Unmyelinated axons count as background. They are hard negatives.

## Evaluation (`submission.md`)

- **Artifact masking:** GT label `1` pixels are excluded from all metrics.
- **Permutation invariance:** predicted IDs need not match GT IDs. Predicted pairs `(X, X+1)` are matched to GT
  pairs "via maximum IoU", **provided each predicted axon is consecutively paired with its myelin**.
- **Dice:** per compartment (axon = all even labels, myelin = all odd labels ≥ 3), plus their macro-mean,
  "across matched instances and global foreground masks", reported across all test micrographs.
- **HD95** on axon and myelin boundaries, symmetric 95th percentile of boundary-to-boundary distances,
  × 2.152 nm/px, so it is reported **in nm**.
- **g-ratio MAE:** match each predicted fiber `(X ∪ X+1)` to a GT fiber by IoU on the full fiber mask. Then
  `D_axon = 2·sqrt(A_axon/π)` and `D_fiber = 2·sqrt(A_fiber/π)` (equivalent-circle diameters via
  `skimage.measure.regionprops`), `g = D_axon / D_fiber`, and the MAE is taken over **matched** fibers only.
- **Ranking:** aggregate rank across **DSC_Mean, HD95_Mean and g-ratio MAE**. Ties go to the lower HD95_Mean.
- **Replicability check:** the top teams' code and weights must reproduce their submitted predictions.

### Ambiguities in the metric spec (each changes the scores)

1. Is Dice computed per image and then averaged, or pooled over all pixels? Does "across matched instances" mean unmatched predictions are dropped?
2. Is HD95 computed on the global compartment mask per image, or per matched instance? What happens when a compartment is empty?
3. No IoU threshold is given for matching (≥ 0.5? any overlap? Hungarian?).
4. Since g-MAE uses only *matched* fibers, **missing fibers carry no g-ratio penalty**. A conservative,
   high-precision predictor is rewarded on g-MAE but penalized on Dice. Exploit this, and also report it as a flaw of the metric.
5. Are fibers cut by the image border included?
6. Does the scale-bar region carry label `1`?

## Submission

```text
TeamOrLastName_Institution_SIPAIM2026.zip
├── method_description.pdf          # 2–4 pages, SIPAIM format
├── code/
│   ├── README.md
│   ├── requirements.txt (or environment.yml)
│   ├── train.py
│   ├── predict.py
│   └── checkpoints/ (or download script/link)
└── predictions/
    └── <test filename>.tif|.png    # uint16, same names as the test images
```

## Dates (as stated, all CET)

| Milestone | Date |
|---|---|
| Launch and train data | October 2026 (site went live around 17 Oct) |
| Registration closes | 9 Nov 2026 23:59 (**passed, and we did not register**) |
| Test images released | 9 Nov 2026 (`IS_TEST_DATA_RELEASED` was still `false` at crawl time) |
| Final submission | 13 Nov 2026 23:59:59 (`config.js` says `2026-11-14T23:59:59`) |
| Results | Live at SIPAIM 2026, 16–20 Nov |

**Our stance:** we are not bound by this timeline. We are using the challenge as a well-defined problem
statement and benchmark protocol for research. Exceptional access is "contact the organizers".

## Inconsistencies on the live site

| Topic | Authoritative (content/*.md) | Conflicting (stale UI / README) |
|---|---|---|
| Mask format | `uint16` paired IDs; `.tif` or uint16 `.png` | Checklist: "PNG masks with label values {0, 1, 2}" |
| Dataset size | 400 = 266 + 134 | Earlier versions (seen in the prior review): ~608 total / ~157 annotated |
| Acquisition | 100 kV, 2.15 nm/px | Repo README: 80–120 kV, 3.8–5.5 nm/px, "Class 1 Axon, Class 2 Myelin" |
| Archive name | `TeamOrLastName_Institution_SIPAIM2026.zip` | Checklist: `submission_[TeamOrAuthor].zip` |
| Paper location | `method_description.pdf` at the archive root | Checklist: `paper/` |
| Deadline | 13 Nov 23:59:59 CET | `config.js`: 14 Nov 23:59:59 |
| Leaderboard | "TBA at SIPAIM" | Hard-coded podium in `index.html` ("MyeloVision Net 0.934", "AxonFormer 0.921", "TEM-SwinUNet 0.908"): **confirmed placeholder** |

The commit history ("instance seg challenge" on 18 Oct, "fix dataset size" ×2 on 22 Oct) shows the challenge was
first a 3-class *semantic* task and was rewritten as *paired instance*. The `{0,1,2}` checklist is left over from that.

## Practical implications

- **No public test GT.** The official test score cannot be reproduced. All our numbers come from an internal
  held-out split of the 266 training images, plus external datasets.
- **No subject metadata** is described. Check the filenames and TIFF metadata for animal or block IDs before designing splits.
- Each image is 4096×3008 at 2.15 nm/px ≈ 8.8 × 6.5 µm, so a few hundred fibers per image. Instance IDs can exceed 255, which is why the format is uint16.
