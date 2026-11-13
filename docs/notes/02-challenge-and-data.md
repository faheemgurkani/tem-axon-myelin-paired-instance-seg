# 02 — The challenge and the data

**Sources:**
- Challenge site https://alexandrojim.github.io/tem-challenge/, crawled 10 Nov 2026 from repo `alexandrojim/tem-challenge` @ `ec9b8e3`. A raw snapshot is in `docs/references/challenge-site-snapshot/`. The site renders client-side, so plain fetchers see only "Loading…".
- EDA scripts in `scripts/eda/`. Their tables and figures are in `data/processed/eda/` (git-ignored, because they contain image crops).
- The data licence is unstated, so **no images, masks or crops are ever committed**.

## 1. The challenge (verified)

**Identity and status:**
- TEM Axon & Myelin Segmentation Challenge: ICM.Quant (Paris Brain Institute) with SIPAIM 2026. Contact gabriel.jimenez@icm-institute.org.
- Registration closed on 9 Nov and **we did not register**. We use the challenge as a problem statement and benchmark protocol only, and its deadlines don't bind us.
- **There is no public test GT**, so every number we report comes from held-out training images.

**Data:**
- Mouse corpus callosum TEM, AMT NS12 camera, 100 kV, 3000×.
- **266 annotated train + 134 unannotated test** images, 4096×3008 (W×H), 8-bit grayscale, **2.152 nm/px**, so one image covers ≈ 8.8 × 6.5 µm.
- The data was received on 12 Nov in `data/raw/train-dataset/{train_samples,train_instance_labels}/sample_0001..0266.tif`.
- Supervised, semi- and self-supervised methods are all allowed.

**Label encoding (`uint16`):**

| Value | Meaning |
|---|---|
| 0 | Background, including **unmyelinated** axons (hard negatives) |
| 1 | "Artifact": **ignored** in loss and metrics. In practice it is mostly fibers; see §3 |
| X even ≥ 2 | An axon |
| X+1 odd | The myelin ring of axon X |

**Metrics:**
- GT label 1 is masked out everywhere.
- **Dice** for axon (all even labels) and myelin (all odd labels ≥ 3).
- **HD95** of axon and myelin boundaries, in nm.
- **g-ratio MAE:** each predicted fiber (X ∪ X+1) is matched to a GT fiber "by max IoU", then g = D_axon / D_fiber with equivalent-circle diameters D = 2√(A/π). The MAE is taken over **matched fibers only**.
- **Ranking:** aggregate rank over mean Dice, mean HD95 and g-MAE, with ties broken by HD95. Top teams must hand over reproducible code.

**What the spec leaves open** (our choices are in [04](04-methods.md) §2):
- Dice per image or pooled?
- HD95 global or per instance? What if a compartment is empty?
- Which IoU threshold, and which matching algorithm?
- Are border fibers included?

g-MAE ignoring misses is a **flaw**: a conservative model can game it.

**Site inconsistencies** (the content pages win over stale UI):
- A `{0,1,2}` PNG checklist left over from the challenge's earlier semantic version.
- A README with 80–120 kV and 3.8–5.5 nm/px.
- A placeholder leaderboard podium.
- A deadline that differs between the text and `config.js`.

The submission format (zip with `method_description.pdf`, `code/{train.py, predict.py, …}`, `predictions/*.tif`) is why the repo keeps `train.py` and `predict.py`.

## 2. What the training set contains (EDA, 12 Nov)

**Files:**
- No TIFF metadata: no pixel size, no animal ID, no acquisition fields.
- The file order is shuffled, and there are no duplicates.
- **So splits can only be made by image** (plus overlap groups, §4).

| Quantity | Value |
|---|---|
| Annotated fibers | **7,968** (per image: median 29, range 5–63) |
| Pixel share: axon / myelin / label 1 / background | 17.4% / 11.8% / 14.5% / 56.4% |
| Fiber IDs | Not consecutive in 264/266 images; the largest ID is 189 |
| Orphan IDs | 47 (43 rings without an axon, 4 axons without a ring) |
| Topology-defect fibers | 296 (3.7%, in 116 images): ring in several pieces, not a single annulus, or axon not enclosed |
| Axon enclosure | 99.5% fully inside their own ring; **0** touch another fiber |
| Fibers touching a neighbour | **71%**; 14% share more than a quarter of their outline |

**Morphometry** (median [5–95%]):

| Quantity | Value |
|---|---|
| Axon diameter | 562 nm [334–1035] |
| Fiber diameter | 755 nm [481–1279] (= 351 px) |
| Myelin thickness | 94 nm [62–144] |
| g-ratio | **0.750** [0.651–0.841] |

- g rises with axon size (r = 0.65; 0.64 for axons under 300 nm, 0.83 above 1 µm). This is the textbook pattern, so the labels are biologically plausible.
- **g sensitivity:** 1 px of axon-boundary shift moves g by ≈ 0.006 here, because fibers are large.
  - Only 2.4% of axons are under 300 nm, where it would be ≈ 0.015.
  - g-ratio differences between methods will therefore be small and need CIs.

**Image appearance:**
- Mean intensity varies 79–167 between images.
- Noise σ ranges 7.5–22, a continuum.
- Illumination non-uniformity reaches 64%.
- 10 images have black corner shadows.
- Fiber density varies 13× (0.11–1.47 fibers/µm²).
- Raw 8-bit, with no compression traces and no defocused images.

**Pathology is present:** very thin myelin (g up to 0.94, i.e. remyelination), dark degenerating axons, split or vacuolated myelin, and outfoldings. This is consistent with a demyelination/remyelination model.

## 3. Annotation conventions (settled by inspection)

1. **Label 1 is mostly fibers.**
   - There are about 11 label-1 regions per image, fiber-sized; 85% touch the image border. **Border-cut fibers are label 1.**
   - There are also about 1.7 *interior* label-1 fibers per image (448 in total). They are larger (median 854 vs 755 nm) and more oblique than annotated fibers: deliberately excluded.
2. **Selective annotation.** About 25% of ring-like structures are left as **background (0)**. They are not ignored. They are mostly abnormal fibers (decompacted, split, redundant myelin), and annotated fibers are mostly compact, regular ones.
3. **Random misses.** About 1.1 clearly normal fibers per image are unlabelled, ≈ 4–8% of real fibers. Their shapes match labelled fibers, so they are misses, not a rule.
4. **Interstitial gaps:** the light triangle between ≥ 3 touching rings looks like an axon in a ring. It is not a fiber (~40% of small unlabelled candidates).
5. **The inner tongue / periaxonal space counts as axon.** The axon label runs to the inner edge of the dark compact myelin.
6. **Boundary convention:** contours follow a consistent, smooth convention with ±1–2 px placement noise between images. The strongest image edge lies 2–4 px away. **Never snap predictions to image edges.**
7. **g definition matters more than noise.** Mean g is 0.750 (equivalent circle), 0.725 (minor axis), 0.790 (Feret), 0.763 (thickness). That spread of 0.065 is 10× annotation noise, so use exactly the equivalent-circle form.
8. 6% of axons are cut obliquely (eccentricity > 0.9). Equivalent-circle diameters over-estimate them, but the spec scores them this way.

**Consequences:**
- Background is not clean negatives.
- Predictions on label 1 must be free.
- A model gets punished for segmenting abnormal fibers that the protocol skips.

## 4. Overlapping tiles (leakage, and the human ceiling)

- Some training tiles show the same tissue: neighbouring tiles of one section, related by translation, overlapping 8–76% of the area.
- **v2 (final): 22 groups covering 52 images** (`configs/splits/overlap_groups.json`).
  - The EDA found 18 groups / 40 images (kept as `overlap_groups_v1_eda.json`).
  - Day 2 fixed an EDA registration bug and found 8 more pairs ([05](05-results.md) §2).
- The search only covered 710 candidate pairs from an ORB feature scan, so unknown overlaps may remain.
- The 134 test tiles very likely overlap training tiles. **Do not exploit this.**
- Overlaps were annotated twice, which gives the **human-vs-human ceiling** ([05](05-results.md) §2).

## 5. External datasets (for later validation, not used yet)

| Dataset | Content | Pixel size | Use |
|---|---|---|---|
| **ASTIH TEM1** (DANDI 001436, CC-BY) | Mouse CC, control + cuprizone, 158 images, 7,711 fibers; semantic masks; test subject `sub-nyuMouse26` | 2.36 nm | **External replication** (T-17): has a published reference (nnU-Net Dice 0.974/0.924, fiber F1@0.5 0.791) |
| ASTIH TEM2 (DANDI 001350) | Mouse CC, unmyelinated axons labelled | 4.93 nm | Hard negatives |
| **AxonPath data** (Zenodo 23084690, CC-BY) | 20 mouse CC TEM (2,074 fibers) + other nerves; **fiber instance IDs** + axon / inner tongue / myelin | ~2.10 nm | Closest to the challenge; conversion deferred |
| AimSeg GT (Zenodo 8351731) | 8 mouse CC TEM, remyelination | 8.42 nm | Superseded by AxonPath |
| ASTIH SEM1 / BF1 / BF2, WMMD, CEM500K, … | Other modalities, or unlabelled | various | Pretraining only |

**Overlap with the challenge data:** unlikely. TEM1 is from another lab (NYU, Philips CM12) with a different pixel size and image size. This is not confirmed by the organisers.

**Conversion notes:**
- ASTIH instances come from connected components (fibers are separated by 1–2 px gaps).
- For AxonPath, compare its *inner cylinder* with our axon, because the tongue counts as axon here.

## 6. EDA techniques used, and why experts run them (`scripts/eda/`)

| Check | Why | Script |
|---|---|---|
| Field-of-view overlap (ORB features + RANSAC, then ECC) | Neighbouring tiles of one section are the most common hidden leak in microscopy | `overlap.py`, `overlap_agree.py` |
| Re-annotation agreement on overlaps | Gives the human ceiling without a second annotator | `overlap_agree.py`, `scripts/run/day2_ceilings.py` |
| Edge-alignment profile of label contours | Shows which edge annotators trace. A systematic offset biases g and HD95 | `deep_quality.py` |
| Myelin thickness profile around each ring | Flags outfoldings, merged sheaths, open rings | `deep_quality.py` |
| Intensity sanity of labels | Myelin is dark and axoplasm light; violations mean label errors or pathology | `deep_quality.py` |
| g-definition sensitivity | Different g formulas can outweigh method differences | `deep_quality.py` |
| Oblique-section check (eccentricity) | Oblique cuts inflate equivalent-circle diameters | `deep_quality.py` |
| Image forensics: noise, blur, illumination, banding, JPEG traces, black corners | Finds acquisition differences that cause domain shift | `deep_quality.py` |
| Pixel-size check from myelin lamellae (FFT) | Independent scale check. **Not resolvable** at this magnification, so 2.152 nm/px is taken as given | `deep_quality.py` |
| Unlabelled ring-candidate mining + galleries | Finds structures a model will predict but the GT calls background | `unl_loc.py`, `unlabeled_rings.py`, `gallery.py` |
| Outlier-fiber galleries | Extreme values are either label errors or pathology; looking separates the two | `gallery.py` |

