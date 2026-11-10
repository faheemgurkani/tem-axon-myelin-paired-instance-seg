# Instance-Segmentation and Foundation-Model Techniques for Nested, Paired Axon/Myelin Instances (status 10–11 Nov 2026)

Scope: techniques catalogue for the "instance + foundation-model" family, applied to the SIPAIM 2026 TEM task (each fiber = axon ID X, even, + myelin ID X+1; hundreds of touching fibers per 4096x3008 image; metrics Dice, HD95, per-fiber g-ratio). Prior-review claims (docs/references/prior-literature-review-2026-11-10.md, sections 3–4) are tagged **CONFIRMED / CORRECTED / UNVERIFIED**. Repo facts were checked with `gh api` on 10 Nov 2026, and shallow clones of cellpose, micro-sam and nnUNet are in `external/`. Preprints are labelled.

## Q1. nnU-Net v2, nnU-Net Revisited (ResEnc), 2D suitability, and 2025–2026 successors

### Takeaway
nnU-Net v2 with the ResEnc presets is still the best-validated semantic backbone. "Revisited" (MICCAI 2024) benchmarks only 3D datasets, so its 2D microscopy claims are inferred. nnU-Net is semantic, so instances must come from an extra boundary or region target plus post-processing. Its "region-based training" can encode the nesting fiber ⊃ axon at the semantic level.

### Cited Findings
- **CONFIRMED.** Title, venue and the validation-shortcomings sentence, verbatim from the abstract: "we demonstrate that many of these recent claims fail to hold up when scrutinized for common validation shortcomings, such as the use of inadequate baselines, insufficient datasets, and neglected computational resources." It ends: the recipe for SOTA is "1) employing CNN-based U-Net models, including ResNet and ConvNeXt variants, 2) using the nnU-Net framework, and 3) scaling models to modern hardware resources", and the results "indicate an ongoing innovation bias towards novel architectures". arXiv comment: "Accepted at MICCAI 2024" — [arXiv 2404.09556 (API record)](https://arxiv.org/abs/2404.09556)
- **ResEnc presets (CONFIRMED).** M has a GPU budget similar to the default ("9-11GB VRAM", ~12 h on A100), L needs "24GB VRAM" (~35 h) and XL needs "40GB VRAM" (~66 h). "We recommend nnU-Net ResEnc L as the new default nnU-Net configuration". Usage: `nnUNetv2_plan_and_preprocess -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)` — [nnUNet/documentation/resenc_presets.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)
- Benchmark (Dice, 5-fold CV, all trained from scratch). nnU-Net (org.) scores BTCV 83.08 / ACDC 91.54 / LiTS 80.09 / BraTS 91.24 / KiTS 86.04 / AMOS 88.64. ResEnc L scores 83.35 / 91.69 / 81.60 / 91.13 / 88.17 / 89.41. MedNeXt L k3 scores 84.70 / 92.65 / 82.14 / 91.35 / 88.25 / 89.62. STU-Net L scores 83.36 / 91.31 / 80.31 / 91.26 / 85.84 / 89.34. SwinUNETR scores 78.89 / 91.29 / 76.50 / 90.68 / 81.27 / 83.81. U-Mamba Bot scores 83.51 / 91.79 / 80.40 / 91.26 / 86.22 / 89.13 — [resenc_presets.md table](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)
- All six "Revisited" datasets are 3D CT/MRI. The paper title says "3D Medical Image Segmentation", and no 2D or microscopy dataset is benchmarked — [resenc_presets.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)
- 2D use: nnU-Net v2 has a native `2d` configuration (`nnUNetv2_train DATASET 2d FOLD`) and sliding-window patch inference, so arbitrarily large images are fine. A 2026 changelog entry says "2D TIFF predictions use PackBits; `.tiff` may select NaturalImage2DIO" — [how_to_use_nnunet.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md), [changelog.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/changelog.md)
- Region-based training "can be used to represent **hierarchical classes**, for example when organs + substructures are to be segmented". Overlapping regions are learned from integer label maps. This is the semantic mechanism for learning "fiber = axon ∪ myelin" and "axon" as nested regions — [region_based_training.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md)
- **Maintenance.** The repo is Apache-2.0 with 9,032 stars and was last pushed 2026-11-10. The last GitHub *release* is still v2.4.1 (2024-04-21, "Residual Encoder UNets released!"). Development continues on master: the changelog covers safetensors checkpoints, DDP fixes and new resampling defaults — `gh api repos/MIC-DKFZ/nnUNet` and [/releases](https://github.com/MIC-DKFZ/nnUNet/releases)
- **Successors.** *Primus / PrimusV2* is now documented inside nnU-Net. It is a "Transformer-centric" 3D architecture with 8×8×8 tokens and 3D RoPE. "PrimusV2 ... reaches parity with state-of-the-art convolutional architectures such as ResEnc-L and MedNeXt". It is 3D-only — [nnUNet/documentation/primus.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md)
- *nnInteractive* (arXiv 2503.08373, preprint) describes itself as "the first comprehensive 3D interactive open-set segmentation method". It takes points, scribbles, boxes and lasso prompts, was trained on 120+ volumetric datasets including 3D microscopy, and integrates with Napari and MITK. Repo: Apache-2.0, pushed 2026-10-27 — [arXiv 2503.08373](https://arxiv.org/abs/2503.08373), [GitHub](https://github.com/MIC-DKFZ/nnInteractive)
- *MedNeXt* (MICCAI 2023) is a ConvNeXt 3D encoder-decoder that grows kernels by upsampling small-kernel networks — [arXiv 2303.09975](https://arxiv.org/abs/2303.09975). *STU-Net* is built on nnU-Net and scales from 14M to 1.4B parameters, pretrained on TotalSegmentator — [arXiv 2304.06716](https://arxiv.org/abs/2304.06716)
- The repo docs also include `finetuning_from_nnssl_checkpoints.md`, i.e. fine-tuning from nnU-Net self-supervised (nnssl) checkpoints — [nnUNet documentation folder](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation)

### Inferences
- For this task, use nnU-Net v2 `2d` + ResEnc-L as the semantic baseline. Train it with three or four labels (background, myelin, axon, optional inter-fiber boundary) and region-based targets {fiber = axon ∪ myelin, axon}. Instances then come from watershed/connected components on the fiber region, seeded by axon components. One axon seed per fiber gives the pairing (axon X, myelin X+1) almost for free.
- Primus, nnInteractive, MedNeXt and STU-Net are 3D-oriented and add little for a 2D TEM challenge. MedNeXt-2D or ResEnc-XL can be ablations if GPU allows.

### Gaps
- No peer-reviewed 2D microscopy benchmark of ResEnc presets was found. Primus has no arXiv ID or venue in the doc I read (UNVERIFIED venue).

## Q2. Cellpose family (2021 → Cellpose 2/3 → Cellpose-SAM → 4.x), fine-tuning and object-count limits

### Takeaway
**CONFIRMED.** Cellpose v4.2 (release v4.2.1.1, 14 Jun 2026) ships `cpsam_v2` and DINOv3-based `cpdino` / `cpdino-vitb`. Cellpose-SAM is still a bioRxiv preprint with no journal version recorded. Cellpose predicts one instance per object via flows. It cannot represent a hole (axon inside a myelin ring) as a separate nested instance. Its natural use here is fiber-level (outer contour) instances.

### Cited Findings
- Citations listed by the repo: Cellpose — *Nature Methods* 18(1):100–106 (2021). Cellpose 2.0 (human-in-the-loop) — *Nature Methods* (2022). Cellpose3 (image restoration) — *Nature Methods* (2025), s41592-025-02595-5. Cellpose-SAM — *bioRxiv* (2025) — [cellpose README](https://github.com/MouseLand/cellpose)
- **Cellpose-SAM is not in a journal: CONFIRMED (prior "bioRxiv 2025").** The bioRxiv API shows v1 posted 2025-05-01 and returns "no articles found for published version of 10.1101/2025.04.28.651001" (queried 10 Nov 2026) — [bioRxiv API details](https://api.biorxiv.org/details/biorxiv/10.1101/2025.04.28.651001), [published-version lookup](https://api.biorxiv.org/pubs/biorxiv/10.1101/2025.04.28.651001)
- Cellpose-SAM abstract: it "adapted the pretrained transformer backbone of a foundation model (SAM) to the Cellpose framework ... substantially outperforms inter-human agreement and approaches the human-consensus bound". It is made robust "to channel shuffling, cell size, shot noise, downsampling, isotropic and anisotropic blur". **CONFIRMED: the prior claim "robust to blur, noise, downsampling"** — [bioRxiv 2025.04.28.651001](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1)
- **v4.2 models: CONFIRMED, with one addition.** The docs say: "`cpsam_v2`: ... CellposeSAM model released in June 2026 using the SAM-ViTL backbone, it includes a fix in the training for low contrast regions". "`cpdino`: ... CellposeDINO model released in June 2026 using the DINOv3-ViTL backbone". "`cpdino-vitb`: ... DINOv3-ViTB backbone (smaller model)". "`cpsam`: original CellposeSAM ... April 2025". DINO models use 384×384 tiles by default, changeable via `bsize`. SAM models use fixed 256×256 tiles ("position embeddings do not support different image sizes") — [cellpose docs/models.rst](https://github.com/MouseLand/cellpose/blob/main/docs/models.rst)
- Release v4.2.1.1 (2026-06-14), "new Cellpose4 models!", includes PR #1454 "adding dino models and gpu-accelerated augmentations". Earlier 2026 releases were v4.0.9 (Mar 4), v4.1.0 (Mar 25) and v4.1.1 (Mar 31). The repo is BSD-3-Clause with 2,433 stars and was last pushed 2026-06-14 — [cellpose releases](https://github.com/MouseLand/cellpose/releases)
- Size prior: the models "were trained on images with a range of diameters from 7.5 to 120 pixels, with a mean size of 30 pixels". Larger objects need the `diameter` parameter (downsampling relative to 30 px) and possibly a larger `niter` — [models.rst](https://github.com/MouseLand/cellpose/blob/main/docs/models.rst)
- Fine-tuning: the documented recipe is `python -m cellpose --train --dir ... --test_dir ... --learning_rate 0.00001 --weight_decay 0.1 --n_epochs 100 --train_batch_size 1`. `--min_train_masks` filters out images with too few masks. Fine-tuned models expect "a similar ROI diameter distribution as the training data" — [docs/train.rst](https://github.com/MouseLand/cellpose/blob/main/docs/train.rst), [models.rst](https://github.com/MouseLand/cellpose/blob/main/docs/models.rst)
- **Max object count.** There is no hard limit in code. Masks are uint16 and switch to uint32 when there are more than 65,535 objects (`"more than 65535 masks in image, masks returned as np.uint32"`, `dtype = "uint16" if n_seeds < 2**16 else "uint32"`). The eval defaults are `min_size=15` px, `max_size_fraction=0.4` (masks larger than 40% of the image are removed), `flow_threshold=0.4` and `tile_overlap=0.1` — [cellpose/dynamics.py and models.py (cloned 10 Nov 2026)](https://github.com/MouseLand/cellpose/blob/main/cellpose/dynamics.py)
- Independent benchmark: Archit & Pape, MIDL 2026 (PMLR v315), mean segmentation accuracy (mSA = per-image mean over IoU thresholds 0.50:0.05:0.95 of TP/(TP+FP+FN)). CellPoseSAM scores label-free cells 0.544, fluorescence cells 0.363, fluorescence nuclei 0.483, histopathology nuclei 0.418. CellPose 3 scores 0.424 / 0.218 / 0.438 / 0.155. The benchmark has no EM data — [arXiv 2603.17845 / PMLR v315](https://arxiv.org/html/2603.17845)

### Inferences
- A Cellpose flow field converges to one center per object. A ring and its enclosed disk share a center, so a single Cellpose model cannot output axon and myelin as two instances. Use it either (a) on fiber = axon ∪ myelin masks, splitting the inner contour afterwards with a semantic axon/myelin map, or (b) as two models (fiber and axon) matched by containment. The Cellpose cyto + nuclei two-model pattern works the same way and does not enforce containment.
- At the challenge pixel size of 2.152 nm/px (repo spec), a 0.5–3 µm fiber is roughly 230–1,400 px across. That is far above Cellpose's 7.5–120 px training range, so heavy downsampling via `diameter` (e.g. ×4–×10) is required. That hurts thin-myelin HD95. `cpdino` with a larger `bsize` is the better option for large objects.

### Gaps
- No Cellpose-SAM or cpdino numbers on EM or myelin were found. No paper or preprint describing `cpdino` / `cpsam_v2` was found; only docs and release notes exist (UNVERIFIED performance).

## Q3. μSAM / micro-sam: EM models, AIS decoder, fine-tuning, 2026 updates

### Takeaway
**CONFIRMED:** *Nature Methods* 22 (2025) 579–591. AIS (decoder predicting foreground plus center/boundary distances, then watershed) is the practical automatic mode. A newer APG mode (MIDL 2026) is competitive with Cellpose-SAM. As of v1.8.14 (Sep 2026) the shipped models are still SAM-v1 ViT-based, with no SAM2/SAM3 generalist. The EM generalist targets *roundish organelles*, not rings.

### Cited Findings
- The paper appeared in *Nature Methods* vol. 22, pp. 579–591 (online Feb 2025, issue Mar 2025, erratum Jun 2025) — [MDC edoc record](https://edoc.mdc-berlin.de/25428), [Nature Methods s41592-024-02580-4](https://www.nature.com/articles/s41592-024-02580-4)
- **">17,000 images with >2 million annotated structures": CONFIRMED, but the source is the University of Göttingen press release, not the paper abstract** — [Uni Göttingen press release](https://uni-goettingen.de/en/3240.html?id=7727)
- Modes, quoted from the docs: AMG uses "a grid of points ... merged via non-maximum suppression". AIS "uses an additional segmentation decoder ... predicts foreground probabilities as well as the normalized distances to cell centroids and boundaries. These predictions are used as input to a watershed". APG "derives point prompts from the segmentation decoder (see AIS) and merges the resulting masks via NMS". "APG yields the best overall instance segmentation results ... and is competitive with CellPose-SAM" — [micro-sam doc/apg.md](https://github.com/computational-cell-analytics/micro-sam/blob/master/doc/apg.md)
- Current EM models are `vit_l/b/t_em_organelles`, "for mitochodria and nuclei in electron microscopy". Guidance: use them "for segmenting mitochondria, nuclei or other roundish organelles". An older v2 model, `vit_b_em_boundaries`, was "for segmenting compartments delineated by boundaries such as cells or neurites in EM". Other families: LM (`vit_*_lm`), MedicoSAM (`vit_b_medical_imaging`) and PathoSAM (`vit_*_histopathology`) — [doc/finetuned_models.md](https://github.com/computational-cell-analytics/micro-sam/blob/master/doc/finetuned_models.md)
- **Prior claim "EM generalist trained on organelles/neurites, not myelin": CONFIRMED with nuance.** The current EM generalist covers organelles. Neurites/boundaries were covered only by a superseded model.
- Releases: v1.5.0 (2025-05-28; QLoRA export, smaller-GPU training configs). v1.6.0 (2025-06-14): "improved versions of the light microscopy and electron microscopy models". v1.7.0 (2025-12-23): automatic prompt generator, training from in-memory data. v1.8.0 (2026-06-04): pip installation via a C++ `bioimage-cpp` backend. Latest is v1.8.14 (2026-09-06). Repo: MIT, 745 stars, pushed 2026-11-10 — [micro-sam releases](https://github.com/computational-cell-analytics/micro-sam/releases)
- No SAM2/SAM3 generalist model is shipped. Only `sam_3d_wrapper` / `simple_sam_3d_wrapper` modules exist (grep of a clone, 10 Nov 2026) — [micro-sam repo](https://github.com/computational-cell-analytics/micro-sam)
- "Revisiting foundation models for cell instance segmentation" (Archit & Pape, MIDL 2026, PMLR v315), mSA. μSAM-AIS scores 0.480 / 0.347 / 0.513 / 0.390 (label-free cell / fluorescence cell / fluorescence nucleus / histopathology nucleus, the last via PathoSAM). APG scores 0.541 / 0.344 / 0.518 / 0.398. CellSAM scores 0.380 / 0.272 / 0.386 / 0.277. SAM3 scores 0.269 / 0.143 / 0.255 / 0.331. SAM-AMG scores 0.106 / 0.091 / 0.283 / 0.200. SAM2 was excluded as inferior to SAM and "produced no objects for several datasets". SAM3 is "not yet competitive with domain-specific foundation models" and depends heavily on the text prompt. No EM data — [arXiv 2603.17845](https://arxiv.org/html/2603.17845), [PMLR v315](https://proceedings.mlr.press/v315/archit26a.html)

### Inferences
- AIS's three-head output (foreground + center distance + boundary distance → watershed) is structurally the same as an nnU-Net/HoVer-style distance-map pipeline. μSAM AIS fine-tuned on fiber masks is a reasonable "foundation-encoder" baseline. Like Cellpose, it yields one instance per fiber, and pairing has to be added.
- Fine-tuning μSAM (LoRA/QLoRA supported) on the challenge's 266 annotated training images (per docs/notes/01-challenge-spec.md; the prior review said ~157) is cheap. The SAM image encoder at 1024×1024 input forces tiling of 4096×3008 images.

### Gaps
- No μSAM result on myelin or TEM nerve was found. No µSAM SAM2/SAM3-based generalist has been announced as of v1.8.14.

## Q4. StarDist and polar/star-convex extensions, including multi-contour / dual-radius / ring objects

### Takeaway
**CORRECTED.** I found no published "dual-radius" (inner + outer radius per ray) StarDist that gives nesting by construction. The prior review presented this as an existing variant; it is a design idea. The closest existing work is **HydraStarDist / HSD-WBR** (arXiv 2504.12078, preprint, 2025). It predicts two nested star-convex object sets in one shot, with containment encouraged by a penalty rather than guaranteed.

### Cited Findings
- StarDist is maintained: BSD-3-Clause, 1,293 stars, latest release 0.9.2 (2025-12-15), last push 2026-02-14 — [stardist releases](https://github.com/stardist/stardist/releases)
- HydraStarDist (De, Urbanski, Yakimovich; arXiv 2504.12078, 16 Apr 2025, not peer-reviewed as far as found): "HSD implicitly incorporates spatial correlation priors based on object interaction through a joint encoder. HSD-WBR further enforces the prior in a regularisation layer with the penalty we proposed named Within Boundary Regularisation Penalty (WBR). Both architectures achieve nested instance segmentation in a single shot". Evaluated with IoU_R and AP on nuclei-in-cells (HeLaCytoNuc) and plaques-in-wells (VACVPlaque) — [arXiv 2504.12078](https://arxiv.org/abs/2504.12078)
- MultiStar (ISBI 2021) extends StarDist to *overlapping* objects by predicting overlap pixels to improve proposal sampling and NMS — [arXiv 2011.13228](https://arxiv.org/abs/2011.13228)
- PolarMask (CVPR 2020) formulates instances as "instance center classification and dense distance regression in a polar coordinate" and reports 32.9% mask mAP on COCO — [arXiv 1909.13226](https://arxiv.org/abs/1909.13226)
- A search for ring/annulus StarDist variants returned only a non-DL patent fitting concentric circles by inner and outer radii from a shared center (US 11295467) — [search summary; USPTO 11295467](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11295467)

### Inferences
- A **dual-radius star-convex head** is an open, novel opportunity. For each center pixel, predict K outer radii r_out(θ) and K inner radii r_in(θ) = σ(·)·r_out(θ). The sigmoid-fraction parametrisation guarantees axon ⊂ fiber by construction, and the axon/myelin pair shares one center and one NMS decision, so the pairing (X, X+1) is automatic. The inner/outer ratio per ray is directly a radial g-ratio profile.
- Risk: myelin outlines of obliquely sectioned or crenated axons are not always star-convex from a single center, so polygon quantisation at K=32–64 rays will limit HD95 on large fibers. A refinement step is needed, e.g. pixel-level snapping to a semantic map.
- SplineDist (ISBI 2021) was not verified in this pass (see Gaps). It would smooth such contours.

### Gaps
- SplineDist venue and numbers were not verified (no tool call spent). HydraStarDist numeric results and code availability were not extracted. No ring-specific StarDist publication was found despite targeted searching.

## Q5. HoVer-Net, CellViT/CellViT++, InstanSeg, CellSAM, MEDIAR and the NeurIPS 2022 cell segmentation challenge

### Takeaway
These are all "one label per object" methods for nuclei or cells. None represents nested pairs natively. They supply separation mechanisms for touching objects: HV distance maps (HoVer-Net, CellViT), seeded embedding clustering (InstanSeg), detector-prompted SAM (CellSAM), and flow/gradient fields (MEDIAR). Venue claims for HoVer-Net and CellViT are CONFIRMED. CellViT++ is CORRECTED to preprint; the prior review did not explicitly cite a venue for it.

### Cited Findings
- HoVer-Net (MedIA 2019) regresses horizontal and vertical distance maps, then applies a marker-based watershed. Its original paper reports CoNSeP AJI 0.571 / PQ 0.547 and Kumar AJI 0.618 / PQ 0.597 — [arXiv 1812.06499](https://arxiv.org/pdf/1812.06499), [Nunes et al. survey text](https://arxiv.org/html/2407.18673v1). The CPM-17 figure of AJI 0.705 / PQ 0.697 comes from a search summary only, so treat it as unverified.
- **CellViT: CONFIRMED as *Medical Image Analysis* vol. 94 (2024), doi 10.1016/j.media.2024.103143.** Abstract: "a mean panoptic quality of 0.50 and an F1-detection score of 0.83" on PanNuke. It uses a ViT/SAM encoder. Licence: "Apache 2.0 with Commons Clause" (not fully OSI-open) — [CellViT GitHub README](https://github.com/TIO-IKIM/CellViT), [arXiv 2306.15350](https://arxiv.org/abs/2306.15350)
- CellViT++ is arXiv 2501.05269 (Jan 2025), a preprint, and its README still cites the 2024 MedIA CellViT paper. It uses a foundation-model encoder plus a lightweight retrainable classifier and claims reduced carbon footprint and zero-shot segmentation. Some checkpoints cannot be shared "due to their license". Repo last pushed 2025-07-23, so it is effectively dormant in 2026 — [arXiv 2501.05269](https://arxiv.org/html/2501.05269v1), [GitHub](https://github.com/TIO-IKIM/CellViT-plus-plus)
- InstanSeg (arXiv 2408.15954, preprint; no journal version found) is "embedding-based", uses a modified U-Net and clusters pixel embeddings around optimally selected seeds. It claims accuracy gains on six public datasets "while reducing the processing time by at least 60%". A bioRxiv companion handles multiplexed nuclei + cells. Repo: Apache-2.0, release v0.1.2 2026-09-29, active — [arXiv 2408.15954](https://arxiv.org/abs/2408.15954), [GitHub](https://github.com/instanseg/instanseg)
- CellSAM (Van Valen lab) trains a detector, "CellFinder", whose boxes prompt SAM. The Caltech record dates the published version to Dec 2025 in *Nature Methods*, but the repo README still cites bioRxiv 10.1101/2023.11.17.567630. Repo: Apache-2.0, pushed 2026-10-28 — [Caltech authors record](https://authors.library.caltech.edu/records/v8v89-3xq78), [GitHub](https://github.com/vanvalenlab/cellSAM)
- NeurIPS 2022 challenge: published as "The Multi-modality Cell Segmentation Challenge: Towards Universal Solutions", *Nature Methods* (2024). It has >1,500 labelled images from >50 experiments, and the top teams used Transformer-based algorithms. MEDIAR (top team) reports "0.9067 F1-score at the validation phase". Its final test F1 was not found. MEDIAR repo: MIT, last push 2024-04-01 (unmaintained) — [arXiv 2308.05864](https://arxiv.org/abs/2308.05864), [arXiv 2212.03465](https://arxiv.org/abs/2212.03465), [GitHub MEDIAR](https://github.com/Lee-Gihun/MEDIAR)
- A bioRxiv 2026 comparison (Cellpose cyto3, Cellpose-SAM, μSAM, CellSAM) on live-cell and spatial-omics images found Cellpose-SAM strong on phase contrast and SAM-based models strong on fluorescence — [Duke CompHealth listing](https://comphealth.duke.edu/publications/foundation-cell-segmentation-models-performance-on-live-microscopy-and-spatial-omics-data/)

### Inferences
- HV maps transfer to separating abutting myelin sheaths: compute them on fiber masks, not myelin rings. A ring's centroid lies in its hole, so per-ring HV maps are ill-posed. The same caution applies to centroid-distance heads (μSAM AIS).
- An InstanSeg-style embedding head is the most flexible for nesting. One could learn a shared fiber embedding plus a binary "inner/outer" channel, so both parts of a fiber cluster to the same seed and are then split by the semantic channel. That gives pairing by construction.

### Gaps
- MEDIAR final test-phase F1 and the winning test numbers were not extracted. CellSAM journal volume/pages were not confirmed (the README still cites bioRxiv). The PanNuke bPQ/mPQ table for CellViT-SAM-H was not extracted.

## Q6. Nested / hierarchical / part-aware instance segmentation and containment-by-construction

### Takeaway
Few methods guarantee containment between paired instances by construction. Those that do are: (i) hierarchical/region label parametrisations, such as multi-level activation and nnU-Net regions, which guarantee *semantic* nesting; (ii) shared-seed decoding, where inner and outer parts inherit one instance ID; (iii) a dual-radius star-convex parametrisation, which is proposed here and not yet published. Joint cell + nucleus systems mostly rely on separate heads plus matching, or on soft penalties.

### Cited Findings
- Multi-level activation (Piraud et al., ECCV 2018 BioImage Computing workshop) replaces softmax with "a multi-level activation layer and three compatible losses" for hierarchically nested classes (nuclei in cells, DSB 2018). It improved Dice significantly (p < 0.007) and "can be generalized to a higher number of classes, with or without further relations of containment" — [arXiv 1804.01910](https://arxiv.org/abs/1804.01910)
- Part-aware Panoptic Segmentation (de Geus et al., CVPR 2021) unifies scene parsing and part parsing and introduces PartPQ. Baselines merge panoptic and part predictions, so parts inherit their parent instance ID by construction of the merge. Code: tue-mps/panoptic_parts — [arXiv 2106.06351](https://arxiv.org/abs/2106.06351)
- HydraStarDist / HSD-WBR: nested two-class instances in a single shot, with containment via a joint encoder plus the WBR *penalty* (soft) — [arXiv 2504.12078](https://arxiv.org/abs/2504.12078)
- Topological interaction module (Gupta et al., ECCV 2022): enforces containment and exclusion between semantic labels via impermissible adjacent-label combinations. Repo TopoXLab/TopoInteraction: MIT, last push 2024-02-19 — [GitHub](https://github.com/TopoXLab/TopoInteraction) (venue as stated in the prior review; not re-verified here)
- Mesmer / DeepCell (Greenwald et al., *Nature Biotechnology* 40:555–565, 2022) predicts whole-cell and nuclear segmentations from TissueNet (≈1.3 M whole-cell + 1.2 M nuclear annotations). It reports "human-level performance" — [PubMed 34795433](https://pubmed.ncbi.nlm.nih.gov/34795433/), [PMC9010346](https://pmc.ncbi.nlm.nih.gov/articles/PMC9010346)
- InstanSeg's companion preprint does simultaneous nuclei + cells in multiplexed images — [instanseg README](https://github.com/instanseg/instanseg)
- AxonDeepSeg (the domain standard) is three-class semantic (background / myelin / axon). Axon labels "were obtained by filling the region enclosed by the inner border of the myelin sheaths", i.e. nesting is imposed through label construction, not instance prediction — [AxonDeepSeg, PMC5830647](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5830647/)
- A search for 2025 nested axon–myelin *instance* methods found only semantic + geometric post-processing pipelines (AxonDeepSeg 2018; Janjic et al. 2019; ISMRM 2020 3D U-Net) — [AxonDeepSeg arXiv 1711.01004](https://arxiv.org/pdf/1711.01004)

### Inferences
- Classification of containment guarantees:
  - **By construction:** shared-seed or shared-ID decoding, where fiber instances are found first and each fiber's pixels are labelled axon or myelin, giving axon X = 2k and myelin X+1 = 2k+1. Also dual-radius polar heads with r_in = σ·r_out, and region/multi-level activations (semantic level only).
  - **Soft:** WBR penalty, topological-interaction loss, Betti losses.
  - **Post-hoc:** two-model matching, e.g. Cellpose cyto + nuclei or Mesmer whole-cell + nuclear assignment.
- For the challenge's even/odd ID scheme, "instance-first on fibers, then part labelling inside each instance" is the simplest guaranteed-valid output. It is the part-aware panoptic recipe applied to fibers.

### Gaps
- Mesmer's exact mechanism for keeping nuclear and whole-cell masks consistent (shared watershed markers vs independent) was not verified from full text. No quantitative cell + nucleus containment-violation rates were found in any method.

## Q7. SAM / foundation-model surveys, SAM2/SAM3 for microscopy, and verification of the Nunes and Chen surveys

### Takeaway
**CORRECTED.** The SAM4MIS GitHub list's own canonical survey is Zhang, Shen & Jiao, *Computers in Biology and Medicine* 171:108238 (2024), arXiv 2401.03495, not arXiv 2305.03678. 2305.03678 is a separate, earlier Zhang & Jiao survey ("Towards SAM for Medical Image Segmentation: A Survey"). The list is active and tracks SAM3, including a MICCAI 2026 one-shot EM SAM3 paper. The MDPI review and the Nunes survey are CONFIRMED. The Chen IET numbers are UNVERIFIED.

### Cited Findings
- SAM4MIS README: it cites "Segment Anything Model for Medical Image Segmentation: Current Applications and Future Directions", *Computers in Biology and Medicine* 171:108238 (2024), plus a new survey, "Unleashing the Potential of Generalist Segmentation Foundation Models for Biomedical Image and Video Analysis", *iRadiology* (2026), doi 10.1002/ird3.70061. It mentions SAM, SAM2 and SAM3 (arXiv 2511.16719). Repo: 1,128 stars, no licence, pushed 2026-11-05 — [SAM4MIS GitHub](https://github.com/YichiZhang98/SAM4MIS)
- arXiv 2305.03678 is "Towards Segment Anything Model (SAM) for Medical Image Segmentation: A Survey", a distinct paper — [ar5iv 2305.03678](https://ar5iv.labs.arxiv.org/html/2305.03678)
- SAM4MIS's EM entries include POT-SAM3: "Prompt-Only Tuning for One-Shot Electron Microscopy Segmentation", MICCAI 2026, code AI4MyBUPT/POT-SAM3. Also SAM4EM (arXiv 2504.21544), SAMDA (few-shot EM domain adaptation, arXiv 2403.07951) and SAM-I-Am (atomic-scale EM, arXiv 2404.06638). Other entries: Dual-Adaptive SAM3 (MICCAI 2026) and SAM3 lesion segmentation (VCIBA 2026) — [SAM4MIS GitHub](https://github.com/YichiZhang98/SAM4MIS)
- **MDPI review: CONFIRMED.** Koishiyeva et al., "A Review of Deep Learning Approaches Based on SAM for Medical Image Segmentation", *Bioengineering* 12(12):1312, published 29 Nov 2025, window "January 2023 to October 2025". 214 records were identified, 74 included. It is organised around rapid adaptation, multimodal fusion, volumetric extensions and uncertainty quantification. It covers SAM2 variants but not Meta's SAM3. On microscopy: transferring SAM "reveals fundamental limitations of universal a priori segmentations", and μSAM "exposed a fundamental contradiction between universality and specialisation" — [PMC12729286](https://pmc.ncbi.nlm.nih.gov/articles/PMC12729286/)
- SAM3 on microscopy (Archit & Pape, MIDL 2026): SAM3 beats SAM-AMG but trails μSAM/APG/Cellpose-SAM on most modalities, did not recognise the prompt "nucleus", and was trained on only two relevant microscopy datasets — [arXiv 2603.17845](https://arxiv.org/html/2603.17845)
- **Nunes et al.: CONFIRMED** as MedIA (doi 10.1016/j.media.2024.103360; online Oct 2024, volume 99 / 2025), arXiv 2407.18673. **Claim "context/attention helped HoVer-Net but not Mask R-CNN": CONFIRMED.** Overall mPQ+ across five sources: HoVer-Net 0.3259 → SE-HoVer-Net 0.3442 → SRM-HoVer-Net 0.3552. Mask R-CNN-R50-FPN 0.4066 vs best SE variant 0.4106, with other variants falling to 0.3349, 0.3265 and 0.2534. The authors call the findings inconclusive: "more tailored solutions are required" — [arXiv 2407.18673](https://arxiv.org/html/2407.18673v1), [X-MOL record](https://x-mol.com/paper/1858773133630468096)
- **Chen et al.: venue CONFIRMED, numbers UNVERIFIED.** "A Systematic Review on Cell Nucleus Instance Segmentation", *IET Image Processing* (2025), doi 10.1049/ipr2.70129, by Yulin Chen, Qian Huang, Meng Geng, Zhijian Wang et al., dated July 2025. The full text returned HTTP 403, so the HoVer-Net AJI 0.775 / PQ 0.764 and StarDist Dice 0.809 / PQ 0.784 figures could not be checked. They do not match HoVer-Net's original CoNSeP (0.571/0.547) or Kumar (0.618/0.597) results, so they must come from a different dataset or compilation — [Bohrium listing](https://www.bohrium.com/paper-details/a-systematic-review-on-cell-nucleus-instance-segmentation/1149159643008729091-2717), [HoVer-Net arXiv](https://arxiv.org/pdf/1812.06499)

### Inferences
- Cite CIBM 2024 (2401.03495) as "the SAM4MIS survey". Cite 2305.03678 separately if needed, and add the iRadiology 2026 survey as the most recent.
- Do not quote the Chen numbers in the paper without the table's dataset context.

### Gaps
- POT-SAM3 numbers were not read. Coverage of the iRadiology 2026 survey was not read. No microscopy-specific foundation-model survey in a top venue (2025–2026) was located beyond the MIDL 2026 benchmark.

## Q8. EM-specific instance methods: SAM4EM, affinities + mutex watershed, LSDs

### Takeaway
Connectomics methods (affinities + LSDs + agglomeration/mutex watershed) are mature and partition-based. Every pixel gets exactly one segment. They separate abutting objects via boundary affinities and long-range repulsive edges, which suits densely packed fibers. They are not nested by themselves: a myelin ring and its axon would need to be one segment, or two segments with a separate pairing step. SAM4EM is a small-footprint workshop paper.

### Cited Findings
- LSDs (Sheridan et al., *Nature Methods* 20:295–303, 2023; online Dec 2022) add an auxiliary task predicting local shape descriptors (diameter, elongation, direction) to direct-neighbour affinities. This consistently improves affinity-based methods and approaches flood-filling networks at "about two orders of magnitude" greater efficiency. Fragments come from seeded watershed plus hierarchical agglomeration — [PubMed 36585455](https://pubmed.ncbi.nlm.nih.gov/36585455/). Repo funkelab/lsd: MIT, last release v0.1.3 (2022), pushed 2025-10-10 — [GitHub](https://github.com/funkelab/lsd)
- Mutex Watershed (Wolf et al., IEEE TPAMI 43(10):3724–3738, 2021) is a greedy signed-graph partitioning with attractive and repulsive cues, "without the need for explicit seeds or a tunable threshold". It solves a multicut-related objective globally and was best on ISBI 2012 EM at publication — [arXiv 1904.12654](https://arxiv.org/pdf/1904.12654), [Heidelberg record](https://heibib.ub.uni-heidelberg.de/search/Record/1775118959). Repo hci-unihd/mutex-watershed: no licence, last push 2021 — [GitHub](https://github.com/hci-unihd/mutex-watershed)
- SAM4EM (Shah et al., arXiv 2504.21544; CVPR 2025 workshops) is a prompt-free SAM adapter with two-stage mask decoding, LoRA and 3D memory attention for neuroscience EM (mitochondria, glia, synapses). Code: Uzshah/SAM4EM, 5 stars, no licence, last push 2025-06-09 — [arXiv 2504.21544](https://www.arxiv.org/abs/2504.21544), [CVPR 2025 virtual](https://cvpr.thecvf.com/virtual/2025/35474)
- POT-SAM3, one-shot EM segmentation with SAM3 (MICCAI 2026) — [SAM4MIS list](https://github.com/YichiZhang98/SAM4MIS)

### Inferences
- A practical transfer is to predict short-range affinities on *fiber* boundaries (the myelin outer border between touching fibers) plus long-range repulsive edges. Partition with mutex watershed (implemented in `affogato` / `elf`, used by μSAM's group) to get fibers, then label axon/myelin inside each fiber from the semantic map. This is partition-based, so fibers cannot overlap, and pairing is by construction.
- LSD-style auxiliary descriptors (local diameter/orientation) could double as a g-ratio-aware auxiliary target: local fiber diameter plus local myelin thickness.

### Gaps
- No affinity/LSD application to myelinated-axon TEM was found. SAM4EM's numeric results were not extracted.

## Q9. Technique catalogue and ranked recommendation (objective)

### Takeaway
Run baselines in this order: (1) nnU-Net v2 2D ResEnc-L, semantic + region-based, with watershed fiber instancing; (2) Cellpose-SAM / cpdino fine-tuned on fiber masks, with axon split; (3) μSAM AIS/APG fine-tuned on fibers. Then build a nested method on a **shared-instance, part-labelled** decoder. Options for that decoder: a fiber flow/distance/embedding head, an inner/outer part head, a dual-radius polar head that guarantees axon ⊂ fiber, and topology/containment losses.

### Cited Findings
Catalogue (every fact is sourced in Q1–Q8 above):

| Technique | Venue/year (status) | Core representation | Touching objects | Nested pairs? | Reported numbers (metric) | Code / licence / 2026 status | Fit |
|---|---|---|---|---|---|---|---|
| nnU-Net v2 + ResEnc | Nat Methods 2021; MICCAI 2024 | Semantic softmax; region-based (hierarchical) targets | Only via an added boundary class + post-proc | Semantic nesting via regions; instances post-hoc | ResEnc-L Dice: KiTS 88.17, AMOS 89.41 (3D) | Apache-2.0; master active 2026-11; last release v2.4.1 (2024) | **Baseline #1** |
| Cellpose / Cellpose3 / Cellpose-SAM / v4.2 (cpsam_v2, cpdino) | Nat Methods 2021, 2022, 2025; Cellpose-SAM bioRxiv 2025 (unpublished) | Flow fields + cell probability | Strong (flows) | No (one center per object) | CellPoseSAM mSA 0.544 label-free (MIDL 2026) | BSD-3; v4.2.1.1 Jun 2026 | **Baseline #2** on fibers |
| μSAM (AIS/APG) | Nat Methods 2025; APG MIDL 2026 | SAM encoder + fg/center/boundary distances → watershed; APG prompts + NMS | Good | No | APG mSA 0.541 label-free; AIS 0.480 | MIT; v1.8.14 Sep 2026 | **Baseline #3** |
| StarDist | MICCAI 2018 | Star-convex polygons (K rays) + NMS | Good for convex blobs | No; HydraStarDist (preprint) is soft-nested | n/a here | BSD-3; 0.9.2 Dec 2025 | Idea source (dual-radius) |
| HydraStarDist / HSD-WBR | arXiv 2025 (preprint) | Two star-convex heads, joint encoder, WBR penalty | As StarDist | Yes, soft | IoU_R, AP (values not extracted) | Unknown | Closest prior art |
| PolarMask | CVPR 2020 | Center + polar distances | Moderate | No | 32.9 mask mAP COCO | GitHub (not checked) | Idea source |
| HoVer-Net | MedIA 2019 | HV distance maps + marker watershed | Strong | No | CoNSeP AJI 0.571 / PQ 0.547 | (not checked) | Idea: HV maps on fibers |
| CellViT / CellViT++ | MedIA 94 (2024) / arXiv 2025 | ViT/SAM encoder + HoVer-style maps | Strong | No | mPQ 0.50, F1-det 0.83 (PanNuke) | Apache-2.0 + Commons Clause; dormant since Jul 2025 | Low (H&E-specific) |
| InstanSeg | arXiv 2024 (preprint) | Pixel embeddings clustered at seeds | Strong | Not natively; nuclei + cells variant exists | ≥60% faster (claim) | Apache-2.0; v0.1.2 Sep 2026 | Idea: shared-seed embeddings |
| CellSAM | Nat Methods 2025 (per Caltech) | Detector boxes → SAM prompts | Detector-dependent | No | mSA 0.380 label-free | Apache-2.0; active Oct 2026 | Low (prompt per object × hundreds) |
| MEDIAR | NeurIPS 2022 challenge | SegFormer-style + flow/gradient heads | Strong | No | 0.9067 F1 (validation) | MIT; unmaintained since 2024 | Low |
| Mesmer | Nat Biotech 2022 | Whole-cell + nuclear heads (DeepCell) | Strong | Paired cell/nucleus, mechanism unverified | "human-level" | DeepCell | Idea: paired heads |
| Part-aware panoptic | CVPR 2021 | Panoptic + part merge (PartPQ) | Via panoptic | **Yes, parts inherit instance ID** | PartPQ (Cityscapes/VOC) | tue-mps/panoptic_parts | **Output-format template** |
| Multi-level activation | ECCVW 2018 | Single-channel nested-class activation | n/a (semantic) | Semantic containment by construction | Dice gain, p < 0.007 | — | Idea |
| Topological interaction | ECCV 2022 | Containment/exclusion penalty on label adjacency | n/a | Soft containment | (prior review) | MIT; dormant since 2024 | Loss add-on |
| Affinities + LSD + Mutex WS | Nat Methods 2023; TPAMI 2021 | Pixel affinities (+ long-range repulsive) → graph partition | **Very strong (dense EM)** | Partition only; pair post-hoc | Best ISBI 2012 at publication | MIT (lsd), no-licence (MWS) | Strong alt for fiber partition |
| SAM4EM / POT-SAM3 | CVPRW 2025 / MICCAI 2026 | SAM(3) adapters, prompt-free / one-shot | Moderate | No | Not extracted | Tiny repo, no licence | Low; watch |
| SAM3 (zero-shot) | arXiv 2511.16719 | Concept/text prompts | Weak on dense microscopy | No | mSA 0.269 label-free | Meta | Not a baseline |

### Inferences
**Ranked baselines** (my judgement from the evidence above):
1. **nnU-Net v2 2D ResEnc-L**, labels {bg, myelin, axon} + region {fiber, axon}, with distance-transform watershed seeded by axon components. It is the rigour anchor demanded by "Revisited", cheap, and gives valid pairs by seeding.
2. **Cellpose-SAM `cpsam_v2` and `cpdino`**, fine-tuned on fiber masks, with the axon assigned per fiber from the nnU-Net semantic map. They are the strongest touching-object separators in 2025–2026 benchmarks, and `cpdino` allows larger tiles.
3. **μSAM AIS/APG**, fine-tuned on fibers (LoRA). It is the foundation-model EM baseline and shows whether SAM pretraining helps on TEM.
4. Optional: **affinities + mutex watershed** on fiber boundaries. This is the connectomics baseline best suited to dense packing.

**Ideas to adopt in a nested-instance method**, in order of expected payoff:
- (a) **Instance-first, part-second decoding:** fibers as instances, each fiber's pixels classified inner/outer. Pairing (X even, X+1) and containment are guaranteed by construction, following part-aware panoptic.
- (b) **Dual-radius star-convex head**, r_in = σ·r_out, as a novel contribution. It gives containment by construction and a direct per-ray g-ratio. Combine it with pixel-level refinement to protect HD95.
- (c) **Fiber-level separation signal:** Cellpose flows, HoVer HV maps or μSAM center/boundary distances, computed on the *filled fiber*, never on the ring.
- (d) **Soft topology/containment losses:** topological interaction (axon never touches background), and skeleton/Betti-style losses for one hole per myelin ring.
- (e) **Foundation encoders** (SAM-ViT-L via Cellpose-SAM/μSAM, DINOv3 via cpdino) for label efficiency with only 266 annotated images.

### Gaps
- None of the listed instance methods has published TEM-myelin numbers. All fit judgements are inferences.
- Prior-review claims not re-verified in this pass: MedSAM (Nat Commun 2024), SAM 2 (ICLR 2025), TopoInteraction venue (ECCV 2022), SplineDist (ISBI 2021). These are UNVERIFIED here, not contradicted.
