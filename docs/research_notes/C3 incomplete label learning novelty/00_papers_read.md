# C3 novelty search: papers read (14 Nov 2026)

**Question (TODO Q5 item 1):** is C3 new? C3 trains with incomplete fiber labels by ignoring confident unlabelled ring candidates (D-18, T-15′).
- If it is not new, how was it done before?
- Does any method handle objects that annotators leave out **on purpose** (selection that depends on the object), rather than objects missed at random?

**Search:** about 12 web searches covering PU learning, sparsely annotated detection, partial or ignore losses, incomplete instance annotation, annotation-error studies, MNAR selection, and EM sparse labels. The key papers were read in full text (HTML or PDF).

## 1. Methods that use the same mechanism as C3 (ignoring or down-weighting unlabelled regions)

### nnU-Net ignore label: Gotkowski et al., arXiv 2403.12834 (2024; v2 Aug 2025)
- Partial loss: Dice and CE are computed only on annotated pixels.
- The documentation (`external/nnUNet/documentation/ignore_label.md`) lists "coarsely masking out faulty segmentations in the reference segmentations" as a typical use case.
- **The D-18 mechanism is a built-in nnU-Net feature.** C3 adds no new loss.

### Engelson, Ehrhardt et al., "LNQ Challenge 2023: Learning Mediastinal Lymph Node Segmentation with a Probabilistic Lymph Node Atlas", MELBA 2(2024), arXiv 2406.03984 (full HTML read)
- **Selective, non-random omission.** The training CTs mark only "clinically relevant" nodes: 1.41 per patient, against 8.61 in the test set. The marked nodes are larger (16.5 mm vs 8.2 mm), and experts "are more likely to annotate strongly enlarged lymph nodes".
- **Mechanism.** The paper argues "false positive predictions can actually be true positives, which are missing in the ground truth". The pixel weight is:
  - 1 on labelled pixels;
  - 1 − p where there is no label and atlas p ≤ 0.25;
  - 0.75 where there is no label and p > 0.25.

  The weight multiplies only the pixel-wise Dice loss.
- **Effect (5 runs, significant):**

  | | Before | After |
  |---|---|---|
  | Dice | 0.527 | 0.551 |
  | Node recall | 0.445 | 0.476 |
  | Precision | 0.806 | 0.797 |

  Larger gains came from oversampling the fully annotated cases, Tversky loss and augmentation.
- **The closest prior work to C3:** a prior marks where unlabelled objects are likely, and the loss is softened there. The prior here is an anatomical atlas; ours would be ring evidence plus an out-of-fold model.

### Sparsely annotated object detection (SAOD), natural images
Read from search snippets and abstracts only:
- Yang, Liang & Carin, "Object Detection as a Positive-Unlabeled Problem", BMVC 2020.
- SparseDet (arXiv 2201.04620), which splits proposals into labelled, unlabelled and background.
- Calibrated Teacher (arXiv 2303.07582).
- BMVC 2021, "labelling only large objects", with a scale-dependent ignore strategy. This is the only SAOD setting found where missingness depends on the object (its size).
- SOIS (arXiv 2208.09023), which uses foreground-map consistency so that unannotated instances are not pushed to background.

The standard SAOD benchmarks thin COCO or VOC labels **at random**.

## 2. PU learning for cells

### Zhao, Pang, Liu, Liu, Ye, "Positive-unlabeled learning for binary and multi-class cell detection in histopathology images with incomplete annotations", MELBA 2023 (MICCAI 2021 extension; arXiv 2302.08050, HTML read)
- **How labels are made incomplete:** "we randomly deleted the annotations … until only one annotated cell per patch was kept". This is **uniform random deletion** on all four datasets.
- **Method:** Faster R-CNN with a non-negative PU loss.
  - The class prior π is assumed known; in practice it is tuned on validation recall, because precision is unreliable when labels are incomplete.
- **F1 against the baseline that treats unlabelled cells as negatives:**

  | Dataset | Baseline | PU |
  |---|---|---|
  | MITOS | 0.470 | 0.507 (full labels: 0.523) |
  | CRC, 1 label per patch | 0.102 | 0.522 |
  | TUPAC | 0.623 | 0.667 |

- **One selection test:** keeping only the easiest cell per patch (the one pathologists agreed on most) still helped.
- **No modelling of selection bias:** the class-prior sensitivity analysis covers random deletion only.

## 3. Studies that measure omission

### Vădineanu, Pelt, Dzyubachyk, Batenburg, "An Analysis of the Impact of Annotation Errors on the Accuracy of Deep Learning for Cell Segmentation", MIDL 2022, PMLR 172:1251–1267 (PDF read)
- **Omission is simulated as "a random subset"** of the cells, at rates of 10–70%.
- With SegNet, Dice drops about 10% at 30% omission. UNet and MSD are robust to moderate omission; MSD sometimes collapses above 30%.
- A 2024 *J Imaging* follow-up by the same authors proposes a network that "upgrades" low-quality annotations. The DOAJ page returned 403, so it was not read.

### Komura et al., "Restaining-based annotation for cancer histology segmentation to overcome annotation-related limitations among pathologists" (SegPath), *Patterns* 4(2):100688, 2023 (PMC9982301; partly read verbatim)
- **Direct evidence of non-random omission.**
  - Cells that pathologists missed (but immunofluorescence caught) differ in morphology:
    - plasma cells without the typical cartwheel nucleus;
    - lymphocytes with paler nuclei.
  - The authors call this "an inherent morphological bias in the annotations of pathologists".
- **Remedy:** a new labelling procedure only. There is no learning-side correction; reweighting is left as future work.
- This is the same phenomenon as our "abnormal fibers = background".

### Bukas et al., MICCAI 2026 workshop (HAIC), already in the 2025–26 report
Model-assisted labels cause batch effects in TEM. This is a risk if C3's ignore masks come from a model.

## 4. Formal treatment of selection that depends on the label (MNAR)

### Mai, Huang, Du, Wu, "A Robust Classifier Under Missing-Not-At-Random Sample Selection Bias", arXiv 2305.15641 (2023; venue not stated; HTML read)
- Labels are missing with P(s|x,y) ≠ P(s|x).
- **Method (BiasCorr):**
  - a probit selection model (Greene);
  - pseudolabels for the unlabelled samples;
  - one shared "soft selection value".
- **Data:** tabular only (Adult, German credit, Drug). No images and no segmentation.
- **Search result:** no paper found that applies an MNAR selection model to segmentation or instance masks.

## 5. EM and other leads (not read in depth)
- ε-Seg (arXiv 2510.18637): very sparse labels in EM.
- Pagano et al., *Front Bioinform* 2023: FIB-SEM with sparse labels and Cross Pseudo Supervision.
- MICCAI 2025: ReCo-I2P (partial-instance lymph nodes) and RefineSeg. The 2024 MICCAI target-level incomplete annotation paper was also seen.

None of these is about deliberately omitted EM structures.
