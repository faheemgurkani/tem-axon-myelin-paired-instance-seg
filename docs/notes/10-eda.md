# 10 — Exploratory Data Analysis of the challenge training set (12 Nov 2026)

Scripts: `scripts/eda/fiber_stats.py` (per-image and per-fiber tables), `artifact_stats.py` (label-1 components) and
`unlabeled_rings.py` (unannotated ring-like structures). Each takes `<data/raw/train-dataset> <out>` and runs in about 2 minutes
on 8 cores. Only aggregate statistics are recorded here. The data licence is unstated, so no images or masks go into the repo.

## 1. What is in the archive

- `train-dataset/train_samples/sample_0001..0266.tif`: 8-bit grayscale, 4096×3008, uncompressed, with **no TIFF metadata**. There is no pixel-size tag, no
  animal or acquisition fields, and no scale bar was seen.
- `train-dataset/train_instance_labels/*.tif`: uint16, deflate-compressed, the same size, and paired by filename.
- The test images are not in this archive.
- **No near-duplicates.** The maximum correlation between thumbnails is 0.26.
- **The file order is shuffled.** Intensity, fiber count and artifact share show no trend with the index (|r| ≤ 0.02).
  The filename carries no animal information, so splits can only be made **by image** (see the risks in 08).

## 2. Label statistics

| Quantity | Value |
|---|---|
| Annotated fibers (axon+myelin pairs) | **7,968** (per image: median 29, range 5–63) |
| Pixel share: axon / myelin / artifact (1) / background | 17.4% / 11.8% / 14.5% / 56.4% |
| Fiber IDs | **Not consecutive** in 264/266 images (max ID ≈ 3× fiber count; about 14 unused IDs per image) |
| Orphan labels | 43 myelin IDs without an axon and 4 axons without myelin, across 30 images |
| Topology defects | 296 fibers (3.7%, in 116 images): myelin split into several components (163), non-annular myelin (Euler ≠ 0), or an axon not enclosed |
| Axon containment | Axon pixels touch only their own myelin: 99.5% fully enclosed. Contact with another fiber is **0** |
| Gap between axon and myelin | Essentially none (median 0 px). The axon label runs up to the inner edge of compact myelin |
| Fibers touching another fiber | **71%**. The median shared perimeter is 9%; 14% of fibers share more than 25% of their outline; mean of 1.25 touching neighbours |

## 3. Morphometry (2.152 nm/px, equivalent-circle diameters)

| Quantity | Median [5–95%] |
|---|---|
| Axon diameter | 562 nm [334–1035] (min 40 nm) |
| Fiber diameter | 755 nm [481–1279] |
| Myelin thickness | 94 nm [62–144] |
| g-ratio | **0.750** [0.651–0.841]. Per-image mean ranges 0.68–0.82 (SD 0.029) |

g-ratio by axon diameter: ≤300 nm: 0.643 (n=190), 300–500: 0.711, 500–700: 0.754, 700–1000: 0.789, >1000: 0.829.
The correlation between g and diameter is r = 0.65. This is the classic pattern for mouse corpus callosum, so the GT g values are biologically plausible.

**g sensitivity in this data.** The median axon radius is 131 px, so a 1-px shift of the inner boundary moves g by ≈ 0.0057 (95th percentile 0.009).
Only 2.4% of axons are under 300 nm. The worst case in our report (+0.015 for a 0.2 µm axon) therefore applies to very few fibers.

## 4. Annotation conventions (settled by inspection)

1. **Label 1 ("artifact") mostly marks fibers that are not annotated, not debris.** There are 3,039 label-1 components, about 11 per image, with a median area of 89k px, which is the size of a fiber.
   85% of them touch the image border and are filled solid. **Border-cut fibers are labelled 1**, which is why only 3.3% of annotated fibers touch the border.
   There are also about 1.7 *interior* label-1 blobs per image, which are whole fibers excluded from annotation, often abnormal ones. The unused IDs are probably these fibers.
2. **Selective annotation.** In some images, clearly myelinated fibers stay **background (0)**: fibers with decompacted, split or redundant myelin, and outfoldings.
   They are neither labelled nor marked as ignored. A crude ring detector found 2,062 unlabelled ring-like candidates against 5,758 matched to GT fibers and 555 on label 1,
   so about **25% of ring-like structures** sit in background (91 images over 30%, 12 over 50%, e.g. samples 0263, 0010, 0251).
   This includes some false candidates, such as unmyelinated axons and organelles, but visual checks confirm many real, abnormal fibers.
   The annotated fibers are mostly **compact, regular** ones.
3. **Inner tongue / periaxonal space:** these are included in the axon label. The axon boundary sits on the inner edge of the dark compact myelin band, with no unlabelled gap.
4. **Unmyelinated axons** are background, as the spec says.

## 5. Image appearance

- Mean intensity ranges 79–167 between images.
- Myelin/background contrast, (bg−myelin)/bg, ranges 0.10–0.50 (median 0.36).
- Saturation is negligible.
- There is strong brightness and contrast heterogeneity, which calls for per-image normalisation plus intensity and contrast augmentation.

## 6. What this means for our research (see report `docs/reports/Paired axon myelin gap validation.md`)

- **The GT pairing itself is clean.** It is 1:1, axons are always enclosed, and no axon touches another fiber. A nested-by-construction decoder therefore matches the
  annotation exactly, and the 3.7% of fibers with topology defects are label noise to clean or ignore when training. **Pairing difficulty comes from separation:** 71% of fibers touch a
  neighbour, so shared sheath boundaries are where the watershed and our decoder will be tested.
- **A new, possibly dominant error source: which fibers count.** Abnormal myelinated fibers are background, not ignored. A model that segments them correctly
  loses Dice and HD95 for it, and an extra predicted fiber that matches no GT fiber does not affect g-MAE at all. Fiber *selection* (compact vs abnormal) is
  a hidden classification task that neither ASTIH nor our earlier plan models. Consequences:
  - The evaluator should report FP fibers that fall on unlabelled ring structures separately from true spurious detections.
  - The oracle-pairing go/no-go experiment must separate *selection* errors from *pairing* errors. Otherwise "pairing is the bottleneck" can't be tested.
  - A per-fiber "annotatable" score (keep/discard head) may be worth more than any pairing improvement. This is also a possible contribution.
  - Self-training on the unlabelled test images will spread these conventions, so pseudo-labels must use the same filter.
- **The g-ratio loss is less urgent than we thought.** Fibers here are large (median axon 562 nm). A 1-px error costs about 0.006 in g, so g-MAE differences
  between methods will be small (about 0.01) and need bootstrap CIs. H3 (gains for small fibers) can only be tested on about 190 fibers under 300 nm. Keep the loss,
  but re-scope H3 to overall g bias, or test it on AxonPath and ASTIH, which have more small axons.
- **Evaluator settings now known:**
  - Border fibers are already label 1 in GT, so mask label 1 in predictions too and drop the `border_fibers` switch's effect.
  - IDs are not consecutive, so never assume consecutive IDs.
  - Orphan IDs exist in GT, so pair by `X/X+1` and skip orphans rather than failing.
- **The inner-tongue convention** (tongue counted as axon) is decided. For AxonPath, compare against `InnerCylinder`, not `Axon`.
- **No animal metadata** exists, so generalisation claims stay image-level unless the organisers provide it.
