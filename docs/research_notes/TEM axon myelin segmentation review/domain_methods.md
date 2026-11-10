# Domain methods: automated axon/myelin segmentation and g-ratio morphometry in EM (status 10–11 Nov 2026)

Scope: verify the prior review (`docs/references/prior-literature-review-2026-11-10.md`) against primary sources, fill gaps, and find newer work. Local copies of the primary sources: PDFs and `.txt` extractions in `docs/references/papers/` (ASTIH `astih_collin26a.*`, generalist `generalist_2409.11552.*`, EM-SAM `arxiv_2307.02464.*`, active learning `arxiv_1907.05143.*`, Le Couedic `lecouedic_ipta2020.*`). Shallow clones are in `external/ASTIH` and `external/axondeepseg`, at AxonDeepSeg HEAD 05a8dd1, 30 Oct 2026, version.txt 5.5.0.

Labels: **[PR]** = peer-reviewed, **[PREPRINT]** = not peer-reviewed, **[ABSTRACT]** = meeting abstract.

---

## Q1. ASTIH (Collin, Boudreau, Cohen-Adad; MIDL 2026, PMLR 315:3608–3627) [PR]

### Takeaway
ASTIH has five datasets and 69,000+ fibers. Its baselines are per-dataset nnU-Net v2.2 semantic models (axon and myelin channels), converted to fiber instances with AxonDeepSeg's centroid-seeded watershed, plus a Cellpose (cyto3 init) fiber-only instance model. Evaluation is Dice and mIoU per class, plus fiber detection P/R/F1 from `stardist.matching` at IoU 0.3 and 0.5. **No HD95 and no g-ratio error metric are reported.** On TEM1, the dataset most similar to the challenge, nnU-Net reaches axon Dice 0.974 and myelin Dice 0.924, but fiber F1 is only 0.791 at IoU 0.5. Cellpose reaches 0.882.

### Cited Findings
- **Venue and status:** Full Paper, MIDL 2026, PMLR 315:3608–3627, CC-BY 4.0. The PDF is linked from the PMLR landing page to raw.githubusercontent.com/mlresearch/v315 (`collin26a.pdf`). — [PMLR page](https://proceedings.mlr.press/v315/collin26a.html); [PDF](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Size (abstract):** "five meticulously curated datasets… over 69,000 manually segmented axon fibers", spanning TEM, SEM and BF; mouse, rat and rabbit; brain, spinal cord and PNS; 0.2–0.002 µm/px. — [PDF, Abstract](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Table 1** (resolution in µm/px; subjects; number of axons; DANDI ID):

  | Dataset | Region, species | Resolution | Subjects | Axons | DANDI |
  |---|---|---|---|---|---|
  | TEM1 | Brain, mouse | 0.00236 | 20 | 7,711 | 001436 |
  | TEM2 | Brain, mouse | 0.00493 | 10 | 26,629 | 001350 |
  | SEM1 | Spinal cord, rat | 0.1 | 10 | 7,948 | 001442 |
  | BF1 | PNS, rat | 0.1 | 8 | 17,992 | 001440 |
  | BF2 | PNS, rabbit | 0.211 | 14 | 9,558 | 001630 |

  — [PDF Table 1](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Table 2** (images / GT masks / average axons per image / average image size / FG:BG ratio):

  | Dataset | Images | GT masks | Avg axons/img | Avg image size | FG:BG |
  |---|---|---|---|---|---|
  | TEM1 | 158 | 158 | 49 | 3762×2286 | 0.98 |
  | TEM2 | 86 | 10 | 437 (myelinated only; 2,782 unmyelinated/img) | 4135×4016 | 4.66 |
  | SEM1 | 10 | 10 | 795 | 1225×1064 | 1.84 |
  | BF1 | 8 | 8 | 2,249 | 5946×5805 | 0.062 |
  | BF2 | 14 | 8 | 1,195 | 1440×1024 | 0.85 |

  Appendix A.2 states TEM2 has "over 26,000 fibers, including 4372 myelinated axons", so Table 1's 26,629 counts unmyelinated axons as well. — [PDF Table 2, App. A.2](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Boundary delineation:**
  - Two-stage annotation. Stage 1 was semantic correction, in which "the myelin sheaths of adjacent, densely packed axons were often merged". The legacy ADS models were trained on that version.
  - Stage 2 was manual separation of adjacent fibers, then "an automated validation script verified topological constraints… ensuring every myelin region contained exactly one axon region", with violators manually corrected (Sec. 2.1.2, Fig. 3).
  - Gap width is 2 px in TEM1 and TEM2 (App. A.1, A.2) and 1 px in SEM1, BF1 and BF2 (App. A.3–A.5).
  - Masks are 8-bit semantic PNGs: `seg-axon-manual`, `seg-myelin-manual`, plus `seg-uaxon/nuclei/process-manual` for TEM2. **There are no instance-ID maps.** Instances are recovered from the delineated semantic masks.
  - — [PDF Sec. 2.1.2, App. A](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Split policy:** "for TEM1, SEM1, BF1 and BF2, data partitioning is performed at the subject level… For TEM2… the testing set includes images taken from every individual mouse" (Sec. 2.1.3). — [PDF](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
  - Concrete test subjects in code: TEM1 `sub-nyuMouse26` (one mouse), SEM1 `sub-rat6`, BF1 `sub-uoftRat02` and `sub-uoftRat07`, BF2 `sub-22G132040x3`. TEM2 uses an external test zip, `data_axondeepseg_srf_testing r20250513-neurips2025`. — [ASTIH scripts/dataset_list.json](https://github.com/axondeepseg/ASTIH/blob/main/scripts/dataset_list.json)
- **Baseline models:**
  - nnU-Net v2.2, "separate output channels for axon and myelin classes", 1000 epochs, 5-fold CV. TEM1, SEM1 and BF2 models have 92M params; TEM2 and BF1 have 126M. Training took 24–48 h on one A6000.
  - Cellpose fine-tuned from cyto3 "on the combined axon and myelin masks to segment individual fibers": 500 epochs (300 for TEM1), 6.6M params, "does not support myelin morphometry" (Sec. 2.2.1).
  - Model URLs in code: TEM1 = `default-TEM-model r20240403 model_seg_mouse_axon-myelin_tem_light`; TEM2 = `model_seg_unmyelinated_stanford_tem_best` (5-class).
  - — [PDF Sec. 2.2.1](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf); [dataset_list.json](https://github.com/axondeepseg/ASTIH/blob/main/scripts/dataset_list.json)
- **Semantic-to-instance step** (Sec. 2.2.2, Fig. 4): "We compute connected components from the axon mask to identify centroids, which serve as seed points. The exact Euclidian distance transform is applied to the axon mask… marker-controlled watershed… on the negative distance map, constrained by the composite axon-myelin mask." The authors concede that it "may struggle in densely packed or pathological tissues" (Conclusion). — [PDF](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Instance evaluation:**
  - The same watershed is applied to both prediction and GT. Thanks to the delineation, this "yields a perfect instance segmentation for the ground truth".
  - Matching uses `stardist.matching`, because MONAI `compute_panoptic_quality` is "computationally intractable" for images with more than 1,000 axons.
  - Thresholds are IoU 0.3 and 0.5 (Sec. 2.2.3). Instances are **fibers** (axon ∪ myelin), not separate axon and myelin objects.
  - In code: Dice/mIoU come from MONAI `DiceMetric`/`MeanIoU` per image, then mean ± std over test images. Detection P/R/F1 are averaged per image and TP/FP/FN are summed per dataset (`aggregate_det_metrics`). The default `--matching_thresh` is 0.5.
  - — [PDF Sec. 2.2.3](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf); [scripts/evaluate_models.py](https://github.com/axondeepseg/ASTIH/blob/main/scripts/evaluate_models.py)
- **Table 3** (nnU-Net, Dice and mIoU mean, std in parentheses):

  | Dataset | Axon Dice | Axon mIoU | Myelin Dice | Myelin mIoU |
  |---|---|---|---|---|
  | TEM1 | 0.974 (0.00542) | 0.950 | 0.924 (0.00692) | 0.859 |
  | TEM2 | 0.971 (0.0217) | 0.945 | 0.929 (0.0254) | 0.869 |
  | SEM1 | 0.925 | 0.861 | 0.880 | 0.786 |
  | BF1 | 0.856 | 0.750 | 0.829 | 0.708 |
  | BF2 | 0.774 | 0.632 | 0.598 | 0.426 |

  — [PDF Table 3](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Table 4** (fiber detection P / R / F1, nnU-Net + watershed vs Cellpose):

  | Dataset | IoU 0.3 nnU-Net | IoU 0.3 Cellpose | IoU 0.5 nnU-Net | IoU 0.5 Cellpose |
  |---|---|---|---|---|
  | TEM1 | 0.837 / 0.955 / 0.887 | 0.990 / 0.797 / 0.882 | 0.747 / 0.851 / **0.791** | 0.990 / 0.797 / **0.882** |
  | TEM2 | 0.958 / 0.980 / 0.965 | 0.996 / 0.830 / 0.902 | 0.945 / 0.958 / 0.949 | 0.996 / 0.829 / 0.902 |
  | SEM1 | — | — | 0.701 / 0.712 / 0.707 | 0.968 / 0.791 / 0.871 |
  | BF1 | — | — | 0.525 / 0.567 / 0.545 | 0.877 / 0.904 / 0.889 |
  | BF2 | — | — | 0.252 / 0.313 / 0.279 | 0.815 / 0.761 / 0.787 |

  Text: Cellpose shows "exceptional precision (> 0.97 in EM)", while "nnU-Net achieves higher recall on the high-resolution TEM datasets… though its watershed-based separation is less robust". — [PDF Table 4, Sec. 3](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Appendix B (IoU-threshold sensitivity, one BF1 image with more than 2,000 axons):** nnU-Net F1 "decreases linearly across the entire range". Cellpose F1 is "stable between 0.3 and 0.7, before decreasing sharply". — [PDF App. B](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **TEM1 provenance** (App. A.1): from Jelescu et al. 2016 (NYU), control and cuprizone mouse corpus callosum, Philips CM12 TEM, about 6×9 µm² per micrograph. The original masks came from an ImageJ macro and were corrected for ADS 2018 and again for ASTIH. — [PDF App. A.1](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **TEM2 provenance** (App. A.2): Iram et al. 2024 (PNAS), SRF-cKO, HPF-FS, JEOL JEM-1400 120 kV. — [PDF App. A.2](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **Code licence:** MIT. Datasets carry their own licences on DANDI. The README still says "four histology datasets… over 60,000", so it is stale relative to the paper. The citation block gives PMLR 315 pp. 3608–3627. — [ASTIH README](https://github.com/axondeepseg/ASTIH)

### Inferences
- ASTIH's instance protocol scores fibers only. It does not score axon and myelin as separate paired instances, and it reports no per-fiber g-ratio error. The SIPAIM metric set (HD95 in nm, g-ratio MAE over IoU-matched fibers) therefore has no published ASTIH reference numbers.
- The TEM1 gap between pixel Dice (0.974/0.924) and fiber F1 at IoU 0.5 (0.791) is direct evidence that the watershed pairing step, not semantic quality, is the bottleneck on mouse corpus callosum TEM.
- On TEM, Cellpose's precision–recall trade-off (precision 0.99, recall about 0.80) mirrors nnU-Net's (precision 0.75, recall 0.85). An ensemble or hybrid is a natural candidate.
- The challenge overlap concern for TEM1 is weakened but not resolved:
  - TEM1 images average 3762×2286 at 2.36 nm/px; the challenge has 4096×3008 at 2.15 nm/px.
  - TEM1 comes from a Philips CM12 at NYU, while the challenge is organized by ICM Paris.
  - The image sizes and pixel sizes differ, so the two sets are unlikely to be the same acquisitions. This is an inference, not verified.

### Gaps
- Per-image fiber counts per split, and how TEM2 test images are chosen per mouse, are not given beyond the external test zip.
- ASTIH does not report HD95, PQ or g-ratio agreement.

---

## Q2. Multi-domain generalist (Collin, Boschet, Boudreau, Cohen-Adad; arXiv 2409.11552 v1, 17 Sep 2024) [PREPRINT]

### Takeaway
The Table 2 numbers in the prior review are exact. Its dataset naming is ambiguous, though: the generalist paper's "TEM2" is **macaque at 0.009 µm/px**, not ASTIH's TEM2 (mouse SRF-cKO at 0.00493 µm/px). No peer-reviewed venue was found.

### Cited Findings
- **Table 2 (OoD Dice):**
  - Generalist: SEM2 axon 0.834 / myelin 0.783; TEM2 axon 0.697 / myelin 0.706.
  - Dedicated: SEM2 0.824 / 0.774; TEM2 0.640 / 0.604.
  - "For SEM2 and TEM2, we respectively used the SEM and TEM dedicated models."
  - "the generalist model consistently detects more small axons, possibly due to its multi-resolution training set" (Sec. 3.2).
  - — [arXiv 2409.11552 p.8](https://arxiv.org/abs/2409.11552)
- **Statistics:** "The Dice scores of the generalist model are significantly greater than the ones of dedicated models (p=0.03077, N=8)", from a paired Student's t-test that only uses dedicated models on their own training datasets (Sec. 3.1). — [arXiv 2409.11552](https://arxiv.org/abs/2409.11552)
- **Table 1 datasets** (pixel size in µm/px):

  | Dataset | Species, modality | Pixel size | Status |
  |---|---|---|---|
  | TEM1 | Mouse TEM | 0.00236 | 1,360 Mpx |
  | TEM2 | Macaque TEM | 0.009 | Partially annotated |
  | SEM1 | Rat SEM | 0.1 | |
  | SEM2 | Human SEM | 0.13 | |
  | SEM3 | Dog SEM | 0.26 | |
  | CARS1 | Rat CARS | 0.225 | |
  | BF1–BF4 | Rat, rabbit, human, cat BF | | |

  Training used TEM1, SEM1, CARS1, BF1, BF2 and BF3. OoD evaluation used TEM2, SEM2, SEM3 and BF4; SEM3 and BF4 were unlabelled and judged qualitatively only. — [arXiv 2409.11552 Table 1, Sec. 2](https://arxiv.org/abs/2409.11552)
- **Training:** nnU-Net, "5-fold cross-validation scheme for 1000 epochs. The generalist model is trained with a batch size of 13 and a patch size of 384x640"; best-validation checkpoint; one 48 GB A6000. It is "resolution-ignorant": images are not resampled to a common pixel size. Reported Dice values are 5-fold ensembles. — [arXiv 2409.11552 Sec. 2](https://arxiv.org/abs/2409.11552)
- **SAM:** "SAM is not ideal for axon and myelin segmentation because it heavily relies on prompts, which need to be specified for every element to segment… automating the pipeline would require to generate accurate prompts which shifts the task from segmentation to object detection". Transformer methods were also tried: "preliminary results were not convincing". — [arXiv 2409.11552 Sec. 1–2](https://arxiv.org/abs/2409.11552)
- **Weights:** `model_seg_generalist` releases r20240224 (ensemble) and r20240416 (single fold). — [ADS model_cards.yaml](https://github.com/axondeepseg/axondeepseg/blob/master/AxonDeepSeg/model_cards.yaml)
- **Venue:** No workshop or journal venue was found. Aggregators list it as arXiv only. — [search result listing](https://www.catalyzex.com/paper/multi-domain-data-aggregation-for-axon-and)

### Inferences
- The generalist's OoD macaque TEM Dice (0.70) is far below in-domain ASTIH TEM Dice (0.97 / 0.92). Resolution and species shift dominate, so for the challenge, in-domain fine-tuning matters more than generalist zero-shot use.

### Gaps
- Whether a later peer-reviewed version exists is unverified.

---

## Q3. AxonDeepSeg (ADS): versions, models, morphometrics, touching-fiber logic

### Takeaway
ADS has been at v5.5.0 (15 Apr 2026) since April. Master (30 Oct 2026) adds a multi-resolution TEM model (Aug 2026) plus morphometric filtering and axon counting (Sep 2026). ADS outputs semantic axon/myelin masks only. Fiber instances come from a centroid-seeded watershed on the axon distance transform, masked to axon ∪ myelin. Its g-ratio is exactly d_axon,eq / d_fiber,eq, the same definition the challenge uses.

### Cited Findings
- **Releases:**
  - v5.5.0, 2026-04-15: diameter approximation overlay, napari pin.
  - v5.4.0, 2026-02-13: "Reset instance seg and database state…", `download_model` refactor.
  - v5.3.0, 2025-10-16: QA report.
  - v5.2.0, 2025-09-05: "Save raw instance segmentation v2" (PR #926).
  - v5.1.0, 2025-06-27: morphometric aggregation; "Use skimage.measure.label with 1-connectivity".
  - — [GitHub releases API](https://github.com/axondeepseg/axondeepseg/releases)
- **Post-v5.5.0 commits:**
  - 2026-07-30: torch/nnunetv2 update, Apple MPS.
  - 2026-08-12: CoreML speedups.
  - 2026-08-31: "Add multires TEM model (#1001)".
  - 2026-09-18: "Morphometrics filtering + Axon counting (#1005)".
  - 2026-10-30: `--allow-large-images` fixes, script renames.
  - — [commits](https://github.com/axondeepseg/axondeepseg/commits/master)
- **Model registry** (`model_cards.yaml` at HEAD, 6 entries):
  - `generalist` (`model_seg_generalist`, 2 classes, ensemble and single-fold): the default for `download_model`.
  - `dedicated-BF`, `dedicated-SEM`, `dedicated-CARS`.
  - `multires-TEM` (`model_seg_tem-multires`): "TEM axon/myelin segmentation, multi-resolution training (TEM1+TEM2), nnUNet 2.8.1", 0.00236–0.016 µm/px, single fold. Release v1.0 is dated 2026-08-15 and is "fold_all".
  - `unmyelinated-TEM` (`model_seg_unmyelinated_stanford`, 5 classes: myelinated and unmyelinated axon, myelin, OL nuclei, OL processes, at 0.00493 µm/px).
  - There is **no instance or "axonmyelin-index" model**. Instance output is the watershed product.
  - — [model_cards.yaml](https://github.com/axondeepseg/axondeepseg/blob/master/AxonDeepSeg/model_cards.yaml); [model_seg_tem-multires releases](https://github.com/axondeepseg/model_seg_tem-multires/releases)
- **Licence:** MIT ("Copyright (c) 2017 NeuroPoly"). — [LICENSE](https://github.com/axondeepseg/axondeepseg/blob/master/LICENSE)
- **Touching-fiber logic** (`get_watershed_segmentation` in `AxonDeepSeg/morphometrics/compute_morphometrics.py`):
  - Seeds are axon-component centroids, cast to int.
  - `distance = ndi.distance_transform_edt(im_axon)` is computed on the axon mask only, "because we know that each axon object is already isolated".
  - The function returns `watershed(-distance, im_centroid, mask=im_axon + im_myelin)`.
  - Holes in axon ∪ myelin are filled only when it is called from the napari plugin.
  - — [compute_morphometrics.py](https://github.com/axondeepseg/axondeepseg/blob/master/AxonDeepSeg/morphometrics/compute_morphometrics.py)
- **Per-fiber morphometrics** (`get_axon_morphometrics`):
  - Axons are labelled with connectivity 2 when myelin is given.
  - The axonmyelin label is read at the **axon centroid pixel**. If that pixel has none, a warning is logged ("Myelin object not found for axon centroid").
  - `myelin_thickness = (eqdiam(axonmyelin) - eqdiam(axon))/2` and `gratio = (axon_diam/2)/(axon_diam/2 + myelin_thickness)`, which equals eqdiam(axon) / eqdiam(fiber). There is an optional ellipse mode using the minor axis.
  - Each fiber gets an `image_border_touching` flag from its bounding box.
  - — [compute_morphometrics.py](https://github.com/axondeepseg/axondeepseg/blob/master/AxonDeepSeg/morphometrics/compute_morphometrics.py)
- **Original 2018 paper (Zaimi et al., Sci Rep 8:3816)** [PR]:
  - U-Net-like, 3-class softmax, 512×512 patches. SEM model 1,953,219 params; TEM model 1,552,387.
  - Table 3 test Dice (axon / myelin): SEM rat1 0.9089 / 0.8193; SEM rat2 0.9244 / 0.8389; SEM human 0.8089 / 0.7629; TEM mice 0.9493 / 0.8552; TEM macaque 0.9069 / 0.7519.
  - Quote: "In both models and all test samples, computed myelin Dice was lower than axon Dice", attributed to myelin having "two interfaces".
  - The 2018 paper's g-ratio was an **aggregate** estimate, sqrt(1/(1+MVF/AVF)), not per-fiber. The paper does not describe the watershed association, which lives in the code.
  - — [PMC5830647](https://pmc.ncbi.nlm.nih.gov/articles/PMC5830647/)

### Inferences
- Failure modes of the ADS pairing, inferred from the code:
  - Integer centroids of non-convex or crescent axons can fall outside the axon, which produces a missing or mis-assigned fiber.
  - Myelin between two fibers is split by watershed flooding order over a flat basin, because the axon EDT is 0 outside axons. The split is not by a true myelin–myelin boundary.
  - Fragmented axons produce several seeds, which split one ring.
  - Myelin with no axon seed is dropped.
- These are exactly the SIPAIM pairing failure modes. The ASTIH-style 2 px inter-fiber gap is what makes this watershed exact on GT.
- "Myelin Dice < axon Dice" is CONFIRMED for ADS 2018, and still holds in ASTIH Table 3 on four of five datasets (TEM1 0.974 vs 0.924). The exception is TEM2 myelin mIoU, though its Dice is still lower.

### Gaps
- Release notes for the multires-TEM model give no accuracy numbers.

---

## Q4. Optic-nerve TEM nnU-Net (Ma, Hao et al., Ophthalmology Science 6(5):101141, 2026) [PR]

### Takeaway
This is a single-encoder, dual-decoder nnU-Net (inner-axon decoder and whole-fiber decoder). Its pixel-level F1 is modest (0.771 inner / 0.697 outer). Morphometric R² is above 0.96 for diameters and area, 0.915 for myelin thickness, and **0.805 for g-ratio**. The prior review's numbers are confirmed. The code URL in the paper returns 404 via the GitHub API.

### Cited Findings
- **Authors and venue:** Rui Ma, Zixuan Hao (equal), …, Mei-Ling Shyu, Richard K. Lee. Ophthalmol Sci 2026 Mar 3; 6(5):101141, doi:10.1016/j.xops.2026.101141, PMC13091109. — [PMC13091109](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091109/)
- **Architecture:**
  - nnU-Net ("20-layer fully convolutional model"), one encoder and **two decoders**.
  - Decoder 1 handles inner axons with Dice + BCE. Decoder 2 handles the whole myelinated fiber with a boundary-weighted Dice.
  - Adam, LR 0.001 with cosine schedule, batch 16, 5000 epochs, early stopping on validation F1.
  - The paper is internally inconsistent: the training section describes a combined weighted BCE + Dice loss.
  - — [PMC13091109](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091109/)
- **Post-processing:** soft threshold, connected components with an area threshold, dilation/erosion, and pruning of partial or border axons. **No explicit pairing or watershed is described.** — [PMC13091109](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091109/)
- **Data:**
  - 368 TEM images from 23 mice (C57BL/6J, DBA/1J). Conditions: naïve, PBS, MTSAAV/hND4 gene therapy, silicone-oil ocular hypertension.
  - JEOL JEM-1400 at ×3000, pixel size not reported.
  - 50 / 25 / 25% split, 92 test images. The split unit (image or animal) is not stated.
  - — [PMC13091109](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091109/)
- **Metrics:** pixel-level precision, recall and F1. Table 1 test values: inner P 0.763 / R 0.779 / F1 0.771; outer P 0.690 / R 0.704 / F1 0.697. The Results text misreports F1 as 0.763 and 0.691, which are the precision values. — [PMC13091109](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091109/)
- **Morphometric agreement (Fig. 4):** "R2 > 0.96 for d, D, and myelinated axon area"; myelin thickness R² 0.915; g-ratio R² 0.805. No Bland–Altman. U-Net, Swin-UNet, DeepLab-V3 and ConvNeXt-UNet were compared, but only in the supplementary Table S1. AxonDeepSeg is not compared. — [PMC13091109](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091109/)
- **Code:** the paper states it is at github.com/umddm/tem-axon-quantification. `gh api repos/umddm/tem-axon-quantification` returned **404 Not Found** on 11 Nov 2026. No licence is stated. The article is CC BY-NC-ND 4.0. — [PMC13091109](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091109/); GitHub API check (this session)
- **Likely precursor** [ABSTRACT]: ARVO 2025, "Deep Learning–Based Transmission Electron Microscopy Analysis of Myelination in the Glaucomatous Murine Optic Nerve Head".
  - Residual U-Net; SO-induced IOP mice, n = 3 per group; ×3000.
  - Axon-fiber P / R / F1 0.905 / 0.927 / 0.916; myelinated-axon 0.874 / 0.891 / 0.882.
  - "correctly identifying over 98% of myelinated axons"; g-ratio "average error rate of just 5.8%".
  - Authorship was not visible on the page.
  - — [IOVS abstract](https://iovs.arvojournals.org/article.aspx?articleid=2803330)

### Inferences
- This is the only published method that predicts nested inner and outer masks with separate heads. Its g-ratio R² of 0.805 against at least 0.96 for diameters shows that a small relative error in both contours compounds in the ratio. This supports the idea of a direct g-ratio loss or consistency term.

### Gaps
- Tables S1–S5 (per-architecture comparison) were not read. The code is unavailable (404). It is unclear whether the split was by animal.

---

## Q5. MicroStructFormer (Deng, Miao, Bagheri, Peterson, Khadra; bioRxiv 10.64898/2026.05.26.727755, v1 27 May 2026, v2 10 Jul 2026) [PREPRINT]

### Takeaway
MicroStructFormer is Mask2Former (Swin-L, initialised from ADE20k) fine-tuned for organelles, periaxonal space (PAS) and abnormal myelin. ADS supplies the axon and myelin masks. Pairing is a heuristic two-step: myelin-guided marker watershed for axons, then geodesic outward propagation to assign myelin, PAS and abnormalities. The prior review's description is CONFIRMED.

### Cited Findings
- **Architecture:** "MicroStructFormer was initialized from… facebook/mask2former-swin-large-ade-semantic", domain pre-trained on public EM datasets, then fine-tuned on their spinal cord EM. Input is resized to 384×384; AdamW, LR 4e-4, batch 4, H100. Validation uses (weighted) mIoU only. — [bioRxiv v2 full text](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v2.full)
- **Two parallel branches:** MSF gives mitochondria, PAS, abnormality and unmyelinated classes. "AxonDeepSeg (ADS) that produced axon and myelin masks". The outputs are merged into one semantic map. — [bioRxiv v2](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v2.full)
- **Pairing:**
  - (1) "MYELIN-GUIDED WATERSHED: …we slightly dilated the myelin mask to create a barrier between neighboring fibers, and used seeds derived from distance-transform maxima within each axon core to drive a marker-controlled watershed".
  - (2) "GEODESIC ASSIGNMENT…: seeds along the boundary of each axon core… propagated… outward over the geodesic distance field… periaxonal space, fiber abnormalities, and any contacting or overlapping myelin… assigned to the nearest axon", with "a mild radius-dependent scaling". Fragmented axon cores are merged first.
  - (3) Each mitochondrion goes to the axon with maximal boundary contact.
  - — [bioRxiv v2 Methods](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v2.full)
- **g-ratio:** circle-equivalent diameters from areas. The "exclusive fiber" (axon plus compact myelin) is used for primary analyses; the "inclusive" variant adds PAS and abnormal regions. Fibers touching the border and those under 0.5 µm are excluded in some analyses. — [bioRxiv v2](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v2.full)
- **Data:** cervical spinal cord EM of Mbp enhancer-edited lines (WT, M3KO, M3(225)KO, M5KO, M3M5KO, M1EM3M5KO, M3KOKI) at P30, P90 and P400, in the CST, GC and VM regions. No quantitative instance or pairing accuracy is reported; validation is distributional (violin plots against GT, Fig. S2). — [bioRxiv v2](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v2.full)
- **Code and data:** "Annotated EM datasets, trained model weights, analysis code… publicly available through… https://dnq2023.github.io/MSF_WEBSITE/". The licence was not checked. — [bioRxiv v2](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v2.full)
- **New lead:** the paper cites "Mysh", described as having "improved per-fiber segmentation accuracy through cascaded architectures" (ref. 37). — [bioRxiv v2 Intro](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v2.full)

### Gaps
- No segmentation or pairing accuracy numbers.
- "Mysh" could not be identified after two searches (see Q11).

---

## Q6. SnapG, AimSeg, MyelTracer, gACSON/DeepACSON, Le Couedic, deep active learning

### Takeaway
All of these are semantic or threshold-first. Instances come from contour nesting (MyelTracer, SnapG), seeded expansion (AimSeg), or watershed (gACSON). Only AimSeg represents a real hierarchy (fibre ⊃ inner region ⊃ axon / inner tongue). Prior review corrections: SnapG is classical, not DL. DeepACSON is in *Communications Biology*, not *Nature Communications*. The gACSON "70%" figure is an example threshold, not a fixed rule.

### Cited Findings
- **SnapG** [PREPRINT] (Bekku, Cai, Xin, Sall, Kestel, Liang, Salzer; bioRxiv 10.64898/2026.01.14.699516, 15 Jan 2026):
  - Fully classical NumPy/OpenCV: average filter, global threshold, contour filtering (edge-touching, area, convexness, circularity).
  - Contours are "assumed to represent axon inner edges". Myelin thickness is a user-chosen percentile (60% PNS; about 30% for dense CNS) of distances from the inner contour to the nearest outer edge. Radius = perimeter/2π; g = r/(r+t).
  - Validation, sciatic nerve: 201 axons, manual 0.669 ± 0.004 vs SnapG 0.675 ± 0.003 (p = 0.25).
  - Validation, optic nerve: manual 0.748 vs SnapG 0.757 (p = 0.132), with 24.61% of axons needing manual tracing because of the periaxonal space.
  - Validation, corpus callosum: batch vs individual mode.
  - Code: github.com/KK201431873/SnapG.
  - Dense CNS fibers are not separated explicitly; the authors instead lower the percentile.
  - — [bioRxiv full text](https://www.biorxiv.org/content/10.64898/2026.01.14.699516v1.full)
- **AimSeg** [PR] (Carrillo-Barberà et al., PLoS Comput Biol 19(11):e1010845, 2023):
  - ilastik autocontext pixel classifier (compact myelin, axoplasm, membrane, mitochondria) plus an object classifier (large axon, small axon, inner tongue), run in Fiji.
  - Stage 1: inner region from the inner myelin border. Stage 2: inner-region seeds expand into myelin to form fibre ROIs. Stage 3: axon / inner tongue.
  - Nested RoiManager sets linked by axon ID.
  - Metrics: object P/R/F1 **averaged over IoU 0.5–0.9 in steps of 0.05**, plus Jaccard.
  - Mean F1, automated → supervised: inner region 0.83 → 0.87; fibre 0.85 → 0.88; axon 0.71 → 0.83.
  - g-ratio CCC: myelin 0.83 → 0.84; axon 0.75 → 0.87.
  - Data: mouse corpus callosum TEM, 5 validation images from 4 mice plus 3 control images. Originals 4096×4096 at 2.1046 nm/px, downsampled ×4 to 8.4182 nm/px.
  - Code: github.com/paucabar/AimSeg. Data: Zenodo 10.5281/zenodo.8351731.
  - — [PMC10691719](https://pmc.ncbi.nlm.nih.gov/articles/PMC10691719/)
- **MyelTracer** [PR] (Kaiser et al., eNeuro 8(4), 2021, doi:10.1523/ENEURO.0558-20.2021):
  - OpenCV bilateral filter, user threshold, `findContours`; nested features grouped with `pointPolygonTest`.
  - The user must "use the cut tool to separate adjacent myelin sheaths".
  - Diameters "of perfect circles from area measurements".
  - Time saving 40–60%. Corpus callosum mean g-ratio 0.8072 vs manual 0.8066 (p = 0.9045).
  - Code: github.com/HarrisonAllen/MyelTracer.
  - — [eNeuro](https://www.eneuro.org/content/early/2021/06/23/ENEURO.0558-20.2021)
- **gACSON** [PR] (Behanova et al., Comput. Methods Programs Biomed. 220:106802, 2022, per the ASTIH reference list):
  - Axon instances come from Bounded Volume Growing. Seeds can be distance-transform maxima, H-maxima (H = 2) watershed centroids, or manual.
  - Myelin is assigned by a watershed on the distance transform of the complement of axon space, which "divides myelin content equally between two adjacent axons".
  - Myelinated test: "e.g., when more than 70% of the cylinder volume is myelin", so 70% is an example threshold.
  - g = d_e/(d_e + d_m), with d_m the median thickness along the myelin skeleton.
  - Rat SBEM; 368 axons; no segmentation accuracy numbers.
  - Code: github.com/AndreaBehan/g-ACSON (MIT).
  - — [arXiv 2112.06476v2](https://arxiv.org/html/2112.06476v2); [ASTIH refs](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- **DeepACSON** [PR] (Abdollahzadeh et al., *Communications Biology* 4:179, 2021):
  - SBEM; two FCNs (DCNN-mAx for myelin, myelinated axons and mitochondria; DCNN-cN for nuclei).
  - Instances via connected components plus **cylindrical shape decomposition** for under-segmentation, and an SVM to remove false positives.
  - **No myelin-to-axon assignment and no g-ratio.**
  - Semantic F1: myelin 0.886; myelinated axons 0.861.
  - Code: github.com/aAbdz/DeepACSON.
  - — [PMC7876004](https://pmc.ncbi.nlm.nih.gov/articles/PMC7876004/)
- **Le Couedic et al.** [PR, conference] (IPTA 2020, Paris; HAL hal-02929744):
  - U-Net with ImageNet SE-ResNet50 encoder plus TTA.
  - 23 images of elderly mouse corpus callosum TEM (CADASIL-group context); the comparison table shows 21 train / 2 test.
  - Pixel Dice: axon 0.955, myelin 0.917, background 0.867.
  - Quote: "the distinction between each individual myelin sheath is not made."
  - The abstract's "average precision of 92%" is pixel precision.
  - No code.
  - — [HAL PDF](https://hal.science/hal-02929744/document)
- **Deep active learning** [PREPRINT] (Lubrano di Scandalea, Perone, Boudreau, Cohen-Adad; arXiv 1907.05143):
  - U-Net with MC-dropout.
  - "Two realistic small datasets of 14 and 24 images", SBEM and TEM.
  - Quote: "maximum Dice value after adding 3 uncertainty-selected samples… versus 15 randomly-selected samples".
  - Code: github.com/neuropoly/deep-active-learning.
  - — [arXiv 1907.05143](https://arxiv.org/abs/1907.05143)

### Gaps
- Licences for the SnapG, AimSeg and MyelTracer code repos were not checked.

---

## Q7. EM-SAM / AxonCallosumEM (arXiv 2307.02464) and the Frontiers in Neuroinformatics 2025 Shank3B study

### Takeaway
"0.984" is **mIoU** (semantic, test split = splenium) for EM-SAM-Base at 1024² input. The 2025 Shank3B follow-up produced separate axon and myelin instance labels via connected components with dilation and erosion. It describes no pairing. It computes per-fiber g-ratio from areas and validates on n = 2 mice. Note the imaging is **serial SEM** (multiSEM), not TEM.

### Cited Findings
- **AxonCallosumEM** [PREPRINT] (Cheng, Zhao, Wang, Zhang; arXiv 2307.02464, Jul 2023):
  - Rett (RTT) mouse model; "over 600,000 patches… 1024 × 1024 with a pixel size of 4 nm"; semantic GT for myelinated axons and myelin sheaths.
  - Split by anatomy: train genu, validation body, test splenium.
  - Table 1 mIoU: UNet 0.919, scratch ViT-B 0.947, MAE-B 0.966, BEiT-B 0.962, **EM-SAM-B 0.984**.
  - SAM ViT-B encoder fine-tuned without prompt encoder or neck, with a conv decoder, BCE loss, 200k iterations.
  - g-ratio is aggregate, estimated as 1/√(1+MVF/AVF) at 16 nm/px.
  - — [arXiv 2307.02464 Table 1, Sec. 2.3, 3–4](https://arxiv.org/abs/2307.02464)
- **Shank3B follow-up** [PR] (Zhao, Cheng, … Ruobing Zhang; Front. Neuroinform. 19, 2025, doi:10.3389/fninf.2025.1563799):
  - Zeiss MultiSEM 505 at 4 nm, downsampled to 8 nm.
  - EM-SAM with a U-shaped decoder: 150k iterations, 30 h inference, then about **4 months of manual refinement**.
  - "Semantic masks were converted to instances using connected-component labeling, dilation, and erosion"; axon and myelin instance labels were produced separately, with no pairing described.
  - Per-fiber area-based g-ratio: WT 0.699 ± 0.025, Shank3B 0.738 ± 0.023.
  - Myelinated axons: 458,671 (WT) vs 625,754 (Shank3B).
  - n = 1 mouse per group.
  - Dice and mIoU against U-Net and UNETR are only in a supplementary table.
  - Data available "without undue reservation"; no code statement.
  - — [Frontiers full text](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2025.1563799/full)
- **NeuroPoly SAM attempt:** github.com/axondeepseg/sam_myelin_seg_TEM (MIT, Jun–Nov 2023), a SAM experiment on their TEM dataset (20 subjects, 1,360 Mpx). The Results section is empty ("[...]"), consistent with the generalist paper's "not convincing". — [GitHub](https://github.com/axondeepseg/sam_myelin_seg_TEM)

---

## Q8. Other leads (AxonDeep, Moiseev, Daeschler, BasNet, dense 3D WM, meta-learning, Janjic)

### Cited Findings
- **AxonDeep** [PR] (Deng et al., TVST 10(14):22, 2021; bioRxiv 2021.05.21.445196):
  - **Light microscopy** of PPD-stained semithin optic nerve, not TEM.
  - ResNeXt-50 FCN with two outputs (axon mask and inter-axon border), watershed at inference, GAN-style semi-supervision.
  - Test Dice 0.81 ± 0.04, count Pearson R 0.97, vs expert-2 Dice 0.79.
  - Axon only; no myelin pairing.
  - — [PMC8709929](https://pmc.ncbi.nlm.nih.gov/articles/PMC8709929/)
- **Moiseev, Hu, Li** [PR] (J Peripher Nerv Syst 24(1):87–93, 2019):
  - CNN segmentation of mouse nerve cross-sections. Axon diameter and myelin thickness deviate "within 9.5%" and manual refinement is still needed.
  - The AAN abstract reports 280 training images and Dice plateauing at 0.91.
  - The full text was not read.
  - — [Houston Methodist record](https://scholars.houstonmethodist.org/en/publications/morphometric-analysis-of-peripheral-myelinated-nerve-fibers-throu/); [AAN abstract](https://www.aan.com/MSA/Public/Events/AbstractDetails/40350)
- **Daeschler et al.** [PR] (Sci Rep 12:5975, 2022):
  - ADS v3.2.0 U-Net on **bright-field** rat nerve at 0.1 µm/px; this is ASTIH BF1.
  - Fiber detection sensitivity 0.99, precision 0.97, F 0.98.
  - Myelin-thickness Bland–Altman bias +0.06 ± 0.15 µm.
  - g-ratio agreement described only as "acceptable", with no number.
  - 5,894 fibers in 6 min 11 s vs about 786 min manually.
  - The fetched summary also lists overlap "axons 0.93 ± 0.03, myelin 0.99 ± 0.01". Treat the 0.99 with caution; it was not checked against the table.
  - — [PMC8993871](https://pmc.ncbi.nlm.nih.gov/articles/PMC8993871/)
- **"A semi-automated approach to dense segmentation of 3D white matter EM"** [PREPRINT] (Kleinnijenhuis, Johnson, Mollink et al.; bioRxiv 10.1101/2020.03.19.979393): SBF-SEM of the mouse corpus callosum genu, dense annotation released as a resource; g-ratio "highly variable between as well as within axons". Only snippets were read. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.03.19.979393.full.pdf)
- **Meta-learning** [PR, workshop] (Nguyen, …, Bunyak; IEEE AIPR 2022, doi:10.1109/AIPR57179.2022.10092238): U-Net with meta-learning for hypoglossal nerve EM across magnifications; 5–7% improvement on unseen 250× and 2500× images. — [researchr](https://researchr.org/publication/NguyenLSLNB22); [PMC10197949](https://pmc.ncbi.nlm.nih.gov/articles/PMC10197949)
- **Janjic et al.** [PR] (J Neurosci Methods 2019, PMC6751333): "measurement-oriented" DNN workflow on human white matter EM; per-fiber measurements; ICC between DNN and annotation "mostly >0.9". Only snippets were read. — [PMC6751333](https://ncbi.nlm.nih.gov/pmc/articles/PMC6751333)
- **BasNet:** not found after three searches. No Springer paper applying BASNet to axon or myelin was located. UNVERIFIED and possibly a mis-remembered lead.

---

## Q9. 2025–2026 work, SIPAIM participants and other challenges

### Takeaway
No 2025–2026 MICCAI, ISBI, MIDL, MedIA or TMI paper on paired or nested myelinated-axon instance segmentation or g-ratio-optimised training was found. The 2026 work found is:
- ASTIH (MIDL);
- the optic-nerve dual-decoder nnU-Net (Ophthalmol Sci);
- MicroStructFormer and SnapG (bioRxiv);
- ADS multires-TEM (software);
- MyeliMetric (a post-segmentation g-ratio toolbox, ASN Neuro, Dec 2025, not read).

No SIPAIM 2026 participant papers are public. The challenge page still shows the leaderboard flagged as under validation.

### Cited Findings
- The SIPAIM page shows: "MyeloVision Net" (ETH Zürich) 0.934; "AxonFormer" (Sorbonne/ICM) 0.921; "TEM-SwinUNet" (JHU) 0.908 mean Dice. It also says "Submissions are currently under independent validation by the ICM.Quant team" and "Winners will be announced at SIPAIM 2026". The organizer is the ICM.Quant Platform, Paris Brain Institute. — [challenge site](https://alexandrojim.github.io/tem-challenge/)
- Web search found no SIPAIM 2026 TEM challenge papers or other axon/myelin challenges. — [search, 11 Nov 2026](https://alexandrojim.github.io/tem-challenge/)
- **Fuster-Barceló & Uhlmann** (arXiv 2602.08505, Feb 2026) [PREPRINT]: DINOv2, DINOv3 and OpenCLIP frozen or with LoRA for EM **mitochondria** (Lucchi++, VNC). Multi-dataset training "causes severe degradation". Not axon/myelin, but relevant to foundation-model transfer in EM. — [arXiv 2602.08505](https://arxiv.org/abs/2602.08505)
- **MyeliMetric:** Python toolbox for standardized g-ratio analysis of already-segmented EM (bioRxiv Jul 2025; ASN Neuro Dec 2025), PMC12818378. Not read in detail. — [PMC12818378](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12818378/)

### Inferences
- The leaderboard entries remain unverifiable and should be treated as placeholders (consistent with the prior review).
- "Nested and paired instance learned end-to-end with a per-fiber g-ratio objective" still appears unpublished as of Nov 2026, within the limits of these searches.

### Gaps
- MICCAI 2026 proceedings and ISBI 2026 were not searched systematically; only general web searches were run.
- "Mysh" was not found.

---

## Q10. Prior-review claim audit

| Claim in prior review | Status | Evidence |
|---|---|---|
| ASTIH: MIDL 2026, PMLR 315, 5 datasets, >69,000 fibers, TEM/SEM/BF, mouse/rat/rabbit, 0.2–0.002 µm/px, "standardized boundary delineation" | **CONFIRMED** (pp. 3608–3627) | [PDF](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf) |
| "Baseline metric values not verified" | **FILLED** | Table 3: TEM1 Dice 0.974 / 0.924. Table 4: TEM1 F1 at IoU 0.5 = 0.791 (nnU-Net), 0.882 (Cellpose) |
| README 4 / 60,000 vs paper 5 / 69,000 | **CONFIRMED**; the README is stale | [README](https://github.com/axondeepseg/ASTIH) |
| ASTIH TEM2 resolution "not verified", "~10 annotated images" | **CORRECTED**: 0.00493 µm/px; 86 images of which 10 have GT; 10 subjects; 26,629 annotated axons incl. unmyelinated (4,372 myelinated); 5 classes | ASTIH Tables 1–2, App. A.2 |
| TEM1: 158 images, 20 mice, 2.36 nm/px, control + cuprizone, CC-BY-4.0 | **CONFIRMED** (licence per DANDI, not re-checked). Also: 7,711 axons, average 3762×2286 px, single test mouse `sub-nyuMouse26` | ASTIH Tables 1–2; dataset_list.json |
| ASTIH SEM1 "10 samples, 8 rats" | **CORRECTED**: Table 1 lists 10 subjects for SEM1 | ASTIH Table 1 |
| Generalist Table 2 numbers (0.697 / 0.706; 0.834 / 0.783), p = 0.03077, patch 384×640, 5-fold, resolution-ignorant, SAM "not ideal" | **CONFIRMED** (N = 8; trained on TEM1, SEM1, CARS1, BF1–3) | arXiv 2409.11552 |
| "TEM2 (macaque) … 0.009 µm/px" | **CONFIRMED**, but clarify that this is the *generalist paper's* TEM2, not ASTIH TEM2 | arXiv 2409.11552 Table 1 |
| ADS (Sci Rep 2018): myelin Dice < axon Dice, "two interfaces" | **CONFIRMED** (exact quote) | PMC5830647 |
| ADS: "None natively; morphometrics from post-processing" | **CONFIRMED and refined**: centroid-seeded watershed on the axon EDT, masked to axon ∪ myelin. v5.5.0 is current; multires-TEM model added Aug 2026 | ADS code |
| Optic-nerve study: 1 encoder + 2 decoders; R² > 0.96 (d, D, area), 0.915 thickness, 0.805 g-ratio | **CONFIRMED**. Add: 368 images, 23 mice, pixel-level F1 0.771 / 0.697. **Code "not verified" → CORRECTED to "stated but repository 404"** | PMC13091109 |
| MicroStructFormer: myelin-guided watershed plus geodesic assignment; classes | **CONFIRMED**. Add: Mask2Former Swin-L; ADS supplies axon/myelin; Mbp enhancer mice, spinal cord; code and weights via the project website | bioRxiv v2 |
| SnapG: "Threshold/contour-based end-to-end g-ratio tool", 201 sciatic axons, periaxonal space complicates g-ratio | **CONFIRMED** (classical, not DL); code on GitHub | bioRxiv |
| AimSeg: hierarchy, P/R/F1, Jaccard, Zenodo 8351731, 5 validation images (4 mice) + 3 control | **CONFIRMED**. Add: F1 averaged over IoU 0.5–0.9; numbers above. "Downsampled ×4" CONFIRMED (2.1 → 8.4 nm/px) | PMC10691719 |
| MyelTracer: assumes circularity; manual separation of adjacent sheaths | **CONFIRMED** | eNeuro |
| gACSON "axon counted as myelinated if >70% of an enclosing hollow cylinder is myelin" | **CORRECTED**: 70% is given as an *example* threshold ("e.g."). Myelin is split by watershed on the distance from axon space | arXiv 2112.06476 |
| "gACSON / DeepACSON… Nature Communications 2021" | **CORRECTED**: DeepACSON is *Communications Biology* 4:179 (2021). gACSON is CMPB 220:106802 (2022). DeepACSON does no myelin-to-axon assignment and no g-ratio | PMC7876004 |
| EM-SAM 0.984 "metric definition not verified" | **RESOLVED**: mIoU (semantic), test = splenium | arXiv 2307.02464 Table 1 |
| AxonCallosumEM: 4 nm/px, 1024² patches, >600,000 patches; "Serial SEM" | **CONFIRMED**. Note the prior table row said "fine-tuned on… mouse corpus callosum EM"; the modality is serial SEM, not TEM | arXiv 2307.02464; Frontiers 2025 |
| Shank3B study "reports axon and myelin instance labels" | **CONFIRMED with caveat**: connected components with dilation and erosion, separate labels, no pairing described; n = 2 mice; about 4 months of manual refinement | Frontiers 2025 |
| Le Couedic (IPTA 2020): "individual sheath separation is not achieved"; debris and aging artifacts | **CONFIRMED** (quote). Add: Dice axon 0.955 / myelin 0.917 on 2 test images; SE-ResNet50 U-Net | HAL PDF |
| Deep active learning: 14 and 24 images, 3 vs 15 samples | **CONFIRMED** | arXiv 1907.05143 |
| AimSeg venue PLoS Comput Biol 2023 | **CONFIRMED** (19(11):e1010845) | PMC10691719 |
| MyelTracer eNeuro 2021 | **CONFIRMED** (8(4), ENEURO.0558-20.2021) | eNeuro |
| Leaderboard is placeholder | **CONFIRMED as still unvalidated** ("under independent validation") | challenge site |
| ASTIH BF1/BF2 "details not verified" | **FILLED**: BF1 rat PNS, 0.1 µm/px, 8 subjects, 17,992 axons. BF2 rabbit tibial nerve, 0.211 µm/px, 14 subjects (8 GT images), 9,558 axons | ASTIH Tables 1–2 |
| "Per-fiber g-ratio is the weakest-reproduced morphometric (R² 0.805 vs >0.96)" | **CONFIRMED** for that one study only; AimSeg shows a similar pattern (g-ratio CCC 0.75–0.87 vs fibre-area CCC 0.9987) | PMC13091109; PMC10691719 |
| BasNet (Springer) lead | **UNVERIFIED**: not found | — |

---

## Q11. Consolidated method catalogue (for the report writer)

| Method | Venue / status | Architecture | Instances / pairing | Data | Key results (metric) | Code / licence |
|---|---|---|---|---|---|---|
| ASTIH baselines | MIDL 2026 [PR] | nnU-Net v2.2 (92–126M params); Cellpose cyto3 fine-tuned | Semantic + ADS centroid watershed; Cellpose fiber-only | 5 datasets, 62 subjects | TEM1: Dice 0.974 / 0.924; fiber F1 0.791 (IoU 0.5), Cellpose 0.882 | MIT code; data on DANDI |
| ADS generalist | arXiv 2024 [PREPRINT] | nnU-Net, patch 384×640 | Semantic + watershed | 6 training sets | OoD macaque TEM Dice 0.697 / 0.706 | MIT |
| ADS multires-TEM | Software, Aug 2026 | nnU-Net 2.8.1, fold_all | Semantic + watershed | TEM1 + TEM2 | none published | MIT |
| ADS 2018 | Sci Rep [PR] | U-Net, ~1.5–2M params | Semantic | Mouse / macaque TEM, rat SEM | TEM mouse Dice 0.949 / 0.855 | MIT |
| Optic-nerve dual-decoder | Ophthalmol Sci 2026 [PR] | nnU-Net, 1 encoder / 2 decoders | Nested inner / outer masks, CC post-processing | 368 TEM, 23 mice | Pixel F1 0.771 / 0.697; g-ratio R² 0.805 | Repo 404 |
| MicroStructFormer + ADS | bioRxiv 2026 [PREPRINT] | Mask2Former Swin-L | Myelin-barrier watershed + geodesic assignment | Mbp spinal cord EM | mIoU only (no number extracted) | Project website |
| AimSeg | PLoS CB 2023 [PR] | ilastik RF + Fiji | Hierarchical ROIs (seed expansion) | 8 mouse CC TEM | F1 (IoU 0.5–0.9 averaged) fibre 0.85–0.88 | GitHub; Zenodo |
| EM-SAM | arXiv 2023 [PREPRINT] / Front. Neuroinform. 2025 [PR] | SAM ViT-B encoder + conv / U decoder | Semantic → CC instances | Serial SEM, 4 nm | mIoU 0.984 | Not released |
| gACSON | CMPB 2022 [PR] | Classical / BVG, 3D | Axon BVG; myelin by watershed on distance | Rat SBEM | No accuracy numbers | MIT |
| DeepACSON | Commun. Biol. 2021 [PR] | FCN + CSD | 3D axon instances, no myelin pairing | Rat SBEM | Axon F1 0.918 | GitHub |
| SnapG | bioRxiv 2026 [PREPRINT] | Threshold / contours | Inner contour + percentile thickness | Mouse sciatic / ON / CC TEM | Mean g within ±0.01 of manual | GitHub |
| MyelTracer | eNeuro 2021 [PR] | Threshold / contours GUI | Nested contours, manual cuts | Mouse CC / ON / sciatic | 40–60% time saved | GitHub |
| Le Couedic | IPTA 2020 [PR] | SE-ResNet50 U-Net + TTA | Semantic only | 23 aged mouse CC TEM | Dice 0.955 / 0.917 | — |
| AxonDeep | TVST 2021 [PR] | ResNeXt-50 FCN + border channel + GAN semi-supervision | Axon instances via watershed (no myelin) | LM optic nerve | Dice 0.81 | — |
| Daeschler | Sci Rep 2022 [PR] | ADS U-Net | ADS watershed | BF rat nerve (BF1) | Fiber F 0.98 | ADS (MIT) |
