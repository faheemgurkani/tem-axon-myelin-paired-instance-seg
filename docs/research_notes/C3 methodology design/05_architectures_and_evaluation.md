# Architectures that can host per-instance handling of missing labels, and evaluation designs for missing-label claims

Written 15 Nov 2026. Scope: which model families make it natural to ignore, down-weight or propensity-weight a *whole* unlabelled candidate fiber, or to add a per-instance "abnormal" or "labelling-propensity" output; how to do it in nnU-Net; the risks of foundation-model pseudo-labels; and evaluation designs reviewers accept.

**What was read.**
- Local code: `external/nnUNet` (loss, deep supervision, ignore-label docs), `external/cellpose/cellpose/train.py`, `external/micro-sam/micro_sam/instance_segmentation.py` and `object_classification.py`, `external/axonpath/axonpath/training/config.py`, and the local Bukas 2026 text.
- Full methodology: Sketchpose (arXiv HTML), Soft Sampling (PDF text), InstanSeg (arXiv HTML), and the DETR loss section (ar5iv).
- Abstract or landing page only: Penquitt 2026, Wolny CVPR 2022, Mask2Former, Lesion-Harvester, CellSAM and Cellpose-SAM status.
- Search snippets only: everything else, flagged where it appears.
- Not reachable: the Nature and PMC pages for μSAM (a login redirect, and a reCAPTCHA that was not touched). μSAM's mechanism below comes from its code.

---

## Q1. Instance-native architectures: where a per-instance loss term, objectness or propensity output would live

### Takeaway
Query and proposal models (DETR, Mask2Former, Mask R-CNN, CellSAM) are the only families with a built-in **per-instance score** and a per-instance loss, so they are the natural place for an extra "abnormal" or "propensity" output.
- A missing label hurts them through one mechanism: unmatched predictions and RoIs are pushed to "no object" or background.
- The literature fix is to down-weight or ignore those terms by overlap or by confidence (Soft Sampling).
- Pixel-field models (Cellpose, StarDist, InstanSeg, HoVer-Net/CellViT, μSAM AIS, AxonPath, nnU-Net) have no instance score. For them, "per-instance" handling means painting a candidate's footprint into a per-pixel ignore or weight mask, which is easy.

### Cited Findings
**Query-based models (DETR → Mask2Former)**
- DETR supervises each unmatched prediction with the no-object class: "Prediction with no match should yield a 'no object' (∅) class prediction".
  - It "down-weight[s] the log-probability term when c_i = ∅ by a factor 10 to account for class imbalance".
  - [DETR, Carion et al., ECCV 2020, arXiv 2005.12872](https://ar5iv.labs.arxiv.org/html/2005.12872)
- Mask2Former is "Masked-attention Mask Transformer for Universal Image Segmentation", CVPR 2022 ([arXiv 2112.01527](https://arxiv.org/abs/2112.01527)).
  - Only the abstract page was read, so its own no-object weight was **not verified here**. The DETR-style matching is inherited.

**Proposal-based detectors (Mask R-CNN family): Soft Sampling, the reference recipe for missing annotations**
- Source: Wu, Bodla, Singh, Najibi, Chellappa, Davis, "Soft Sampling for Robust Object Detection" ([arXiv 1806.06986v2](https://arxiv.org/pdf/1806.06986)). The PDF uses the BMVC template; the venue is not printed on the paper (commonly cited as BMVC 2019, unverified).
  - Unlabelled instances "will be treated as background, which will generate an incorrect training signal".
  - After dropping 30% of PASCAL VOC annotations per class, Faster R-CNN mAP "only drops by 5%".
  - Soft Sampling "re-weights the gradients of RoIs as a function of overlap with positive instances", using a Gompertz curve, so uncertain background RoIs get less weight than hard negatives.
  - The paper also discusses re-weighting by "the confidence of a detector trained on partial annotations" and warns it "could generate false positives and consequently ignore regions where there are no objects".
  - Missingness was simulated by **random** per-class drops.
- The Background Recalibration Loss (arXiv 2002.05274) and "Object Detection as a Positive-Unlabeled Problem" (arXiv 2002.04672) treat missing boxes as PU detection. Only titles and snippets were read.
  - The PU paper removes 10% of annotations in training "but use[s] them to calculate detection rates" ([search snippet, arXiv 2002.04672](https://arxiv.org/html/2002.04672v1)).
- A nuclei detector (NuHTC, MedIA 2025) ignores nuclei that have boxes but no masks in the mask head. This is per-instance ignoring inside a cascade ([ScienceDirect, snippet only](https://www.sciencedirect.com/science/article/pii/S1361841525001422)).

**Detector-prompted foundation model (CellSAM)**
- CellSAM (Nature Methods, Dec 2025; DOI 10.1038/s41592-025-02879-w): a detector, CellFinder (Anchor DETR on SAM's ViT), produces boxes that prompt SAM's mask decoder.
  - It was trained in two stages: detection first, then fine-tuning with ground-truth boxes and masks.
  - Sources: [arXiv 2311.11004](https://arxiv.org/pdf/2311.11004v1.pdf); [PMC record](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12695629/). Snippets only.
  - Its per-box detection score is a ready objectness output.

**Embedding-based models (InstanSeg, Wolny CVPR 2022)**
- InstanSeg (Goldsborough … Bankhead; arXiv 2408.15954, **preprint**, no venue on the arXiv page) has three heads:
  - a seed map regressed (L1) to distance-to-boundary;
  - positional and conditional embeddings, plus an MLP that turns each seed into a per-pixel instance probability.
  - Its instance loss is a **sum over the K ground-truth instances** (Lovász hinge, K ≤ 50), and seeds are local maxima of the seed map.
  - The paper does not discuss unlabelled regions or sparse labels ([arXiv HTML](https://arxiv.org/html/2408.15954v1)).
- Wolny, Yu, Pape, Kreshuk, "Sparse Object-level Supervision for Instance Segmentation with Pixel Embeddings", CVPR 2022 ([arXiv 2103.14572](https://arxiv.org/abs/2103.14572); abstract only).
  - The instance loss is applied directly to extracted instances.
  - "For the case of positive-unlabeled supervision", a self-supervised consistency loss covers the unlabelled parts.
  - It was tested on 2D and 3D microscopy, CVPPP and Cityscapes.

**Flow and distance models (Cellpose, Omnipose/Sketchpose, StarDist)**
- Cellpose's training loss is per-pixel and has no mask argument (`external/cellpose/cellpose/train.py::_loss_fn_seg`):
  - MSE on flows (×5) / 2;
  - BCE on cell probability `(lbl[:, -3] > 0.5)`;
  - both with `reduction="mean"` over all pixels.
- Sketchpose (Cazorla, Munier, Morin, Weiss; **MELBA 2025**, DOI 10.59275/j.melba.2025-f7b3; [arXiv 2508.17798](https://arxiv.org/html/2508.17798)) changes Omnipose to learn from partial annotations.
  - The distance and flow losses are summed only over a "valid distance set" 𝒟 of annotated pixels whose nearest annotated boundary is provably the true boundary, normalised by |𝒟|.
  - An inequality term handles the rest.
  - Partial annotation was simulated **randomly** (thresholded Gaussian noise keeping 10–100% of pixels).
  - MicroBeSeg F1 falls from 0.81 (100%) to 0.60 (10%).
  - Training takes about 5 h for MicroBeSeg and 30 h for PanNuke on one GPU.
  - It is limited to 2D, binary foreground.
- StarDist's per-pixel object probability and per-polygon NMS score were **not re-read this session**. The local PDF is `docs/references/papers/schmidt2018_stardist_1806.03535.pdf`.

**Microscopy foundation model with decoder-based AIS (μSAM)**
- μSAM (Archit … Pape, Nature Methods 2025, DOI 10.1038/s41592-024-02580-4; [news summary](https://uni-goettingen.de/en/3240.html?id=7727)) was retrained on more than 17,000 microscopy images.
  - Its automatic instance segmentation (AIS) adds a UNETR decoder that predicts three maps: foreground, centre distances and boundary distances.
  - A seeded watershed then builds the instances (`external/micro-sam/micro_sam/instance_segmentation.py`, `InstanceSegmentationWithDecoder`, lines ~953–1160).
  - μSAM also ships `object_classification.py`, which pools SAM image embeddings over each segmented object to build per-object features for a classifier. This is a ready per-instance "abnormal" classifier on top of any segmentation.

**Our own baselines**
- AxonPath trains three loss terms: CE on semantic (bg/myelin/axon), MSE on the fibre distance map, and MSE on the axon distance map.
  - Default weights are (1, 1, 1) (`external/axonpath/axonpath/training/config.py:34`).
  - There is no instance score.

### Inferences
**Where each option could host C3**

| Family | Per-instance score available? | How to ignore or down-weight a whole unlabelled candidate | Where an "abnormal" or "propensity" output fits | Fit for our dense rings |
|---|---|---|---|---|
| Mask2Former / DETR | Yes (class logit per query) | Exclude the candidate's footprint from the no-object term: give queries whose masks overlap a candidate weight w ∈ [0, 1], or match them to the candidate as "ignore" | An extra class (normal/abnormal) or an extra sigmoid head on the query embedding | ~30 objects/image is easy; rings are plain masks. Data-hungry from scratch; needs COCO/SA-1B pretraining |
| Mask R-CNN / CellSAM-style detector + mask | Yes (RPN objectness + RoI class) | Soft Sampling: weight background RoIs by overlap with candidates instead of with GT | Extra RoI class or head | Boxes of touching rings overlap heavily, so box NMS can suppress neighbours. CellSAM's detector is pretrained |
| InstanSeg / Wolny embeddings | Per-seed, implicitly | The instance loss already sums over GT instances only. Only the seed/background term must be masked on candidate footprints | Per-seed MLP head | Ring topology suits embeddings (no star-convexity). InstanSeg is a preprint and nuclei-tuned |
| Cellpose / Cellpose-SAM | No (per-pixel cellprob) | Add a per-pixel mask to both flow-MSE and cellprob-BCE (a code change; there is no built-in option) | Extra per-pixel class channel (Cellpose already has `_loss_fn_class`) | Flow-tracking around an annulus is ill-posed; rings need the axon and fiber as separate objects (`baselines/cellpose_sam` already zero-shot) |
| StarDist | Per-polygon NMS score, but derived from per-pixel probability | Mask the per-pixel probability/distance loss | — | A myelin ring is not star-convex. Only the fiber (outer contour) and axon (inner) discs are, so it would need two StarDist outputs |
| μSAM AIS / nnU-Net / AxonPath (semantic or distance + watershed) | No | Paint the candidate footprint as ignore (nnU-Net: built in) | Pixel class "abnormal fiber", or **post-hoc per-object classifier** (μSAM `object_classification.py` pattern, or our own features) | Already our pipeline (B1 + `pairing.py`) |

**What this means for C3**
- A **learned labelling-propensity head** is natural only in query models, where it is one extra logit per query.
  - In B1-style models the cheapest equivalent is a **second-stage per-instance classifier**: features of each decoded fiber, or pooled network features, predict "would an annotator label this?".
  - This needs no segmentation retraining and fits the budget.
- The main cost of moving to Mask2Former or Mask R-CNN is losing B1's 0.904 F1 as the starting point.
  - The C3 effect would then be confounded with an architecture change, unless the same architecture is trained with and without C3. That doubles runs.

### Gaps
- No 2025–26 paper found that uses a query model's objectness to model **labelling propensity**, rather than true objectness, in segmentation.
- Mask2Former's exact no-object weight, CellViT/HoVer-Net/NuLite head details and StarDist's loss were not re-read this session.
  - HoVer-Net and CellViT are widely known to have a per-pixel type branch that could carry an "abnormal" class, but this was not verified here.
- InstanSeg's peer-reviewed venue was not found; treat it as a preprint.
- Wolny CVPR 2022 was read at abstract level only, so it is unknown whether it used EM data and how unlabelled background is treated.
- ImPartial (nadeemlab; the GitHub README says MICCAI 2026; [arXiv 2605.24128](https://arxiv.org/pdf/2605.24128)) was seen only as a title.

---

## Q2. Semantic-plus-postprocessing: per-instance ignore in nnU-Net, deep supervision, distance heads, and sparse-label extensions

### Takeaway
Instance-level hard ignore in nnU-Net is **zero-code**: paint each candidate fiber's footprint (disc plus ring) with the ignore label.
- The CE and Dice losses mask it, deep-supervision targets are downsampled with nearest neighbour, and validation pseudo-Dice ignores it.
- **Soft down-weighting (LNQ-style w ≈ 0.75) or propensity weighting is not built in.** It needs a custom trainer that passes a per-pixel weight map. That is a small change, because the losses already take a `loss_mask`.

### Cited Findings
**What the ignore label does**
- nnU-Net's ignore label marks regions to ignore "using partial losses, i.e. losses that are only computed on annotated pixels".
  - At inference nnU-Net "will always predict dense segmentations".
  - "Metric computation in validation is of course only done on annotated pixels."
  - Listed use cases include "Coarsly masking out faulty segmentations in the reference segmentations".
  - Source: `external/nnUNet/documentation/ignore_label.md`, lines 1–22; cite Gotkowski et al., "Embarrassingly Simple Scribble Supervision for 3D Medical Segmentation", [arXiv 2403.12834](https://arxiv.org/abs/2403.12834). The venue is not stated in the nnU-Net docs.
- The ignore label "MUST be the highest integer label value" and be named `ignore`.
  - It is compatible with region-based training, where the ignore label is not declared in `regions_class_order` (same file, lines 69–95).
- The docs recommend evaluating sparse-annotation strategies "(through simulations for example)" on **densely annotated images**, not with nnU-Net's own cross-validation (same file, lines 100–104).

**How the losses implement it**
- In `DC_and_CE_loss`, `mask = target != ignore_label`. Dice gets `loss_mask=mask`, CE gets `ignore_index`, and CE is skipped when a patch has no annotated pixel (`external/nnUNet/nnunetv2/training/loss/compound_losses.py:37–53`).
- In region-based `DC_and_BCE_loss`, the ignore channel is the last one-hot channel, and the mask is its inverse. BCE uses `reduction='none'`, then a masked mean (lines 60–100).
- Deep supervision: `DeepSupervisionWrapper` applies the same loss at each scale with fixed weights.
  - Targets are downsampled per scale by `DownsampleSegForDSTransform` (`nnUNetTrainer.py:35, 945`).
  - nnU-Net's own copy of that transform defaults to `order=0` (nearest), so ignore pixels survive at coarse scales as nearest-neighbour footprints (`.../custom_transforms/deep_supervision_donwsampling.py`).
  - The trainer imports the batchgeneratorsv2 version, whose default was not opened.

**Prior use of this exact mechanism**
- Our C3 novelty report found that LNQ (Engelson et al., MELBA 2024) down-weighted the Dice loss where an atlas predicted a node but no label existed: Dice +0.024, recall +0.03 (`docs/reports/C3 incomplete label learning novelty.md`; source note in `docs/research_notes/C3 incomplete label learning novelty/`).
- ReCo-I2P (MICCAI 2025): incomplete, partial-instance supervision for lymph nodes ([MICCAI page](https://papers.miccai.org/miccai-2025/0755-Paper3517.html); title only).

### Inferences
**C3 variants in nnU-Net**
1. **Hard ignore:** a label-conversion change only. Map candidate fibers to the ignore label (in nnU-Net format it must be the highest integer, so {0 bg, 1 axon, 2 myelin, 3 ignore}), with B1's plans and trainer.
2. **Soft or propensity weighting:** subclass the trainer.
   - Carry a float weight map as an extra target channel, and multiply it into the CE (reduction none) and the Dice `loss_mask`.
   - Downsample it for deep supervision with nearest or average pooling.
   - About 50 lines.
3. **Random-ignore control:** the same pixel budget, from random fiber-shaped masks.

**Deep supervision**
- At coarse scales the ignore footprint loses accuracy at its edges. A **1–2 px dilation of the candidate footprint** keeps the boundary between an ignored fiber and its labelled neighbour from leaking into the coarse losses.
- Also check that dilation never covers a neighbouring labelled fiber's ring. Touching fibers make this a real risk: dilate only into background.

**Checkpoint selection**
- nnU-Net's validation pseudo-Dice skips ignore pixels, so the "best" checkpoint is not biased against recovering ignored fibers.
- On *unmodified* validation folds, however, unlabelled real fibers still count as false positives. Use the final checkpoint, or our own known-positive validation.

**Distance and SDT heads**
- An ignore mask applies to regression heads the same way: use the same mask. nnU-Net has no SDT head built in; AxonPath does.
- If an SDT head is added for H5 (ring topology), mask its MSE with the same per-instance weight map.

### Gaps
- No 2024–26 nnU-Net extension specifically for **object-level partial labels or PU segmentation** was found. Search coverage was partial.
- Gotkowski 2024's peer-reviewed venue was not confirmed.
- The default interpolation order in batchgeneratorsv2's `DownsampleSegForDSTransform` was not inspected.

---

## Q3. Foundation-model pseudo-labelling for missing instances, and its risk

### Takeaway
SAM-family proposals can find unlabelled fibers, but every source warns that pseudo-labels miss objects or bias downstream measurements.
- Bukas et al. show that **mixing model-made and human labels creates batch effects in morphology features**, which matters directly for g-ratio.
- C3 should therefore use foundation models only to **propose or filter ignore candidates**, never to add positive labels to training or test sets, and should score against human labels only.

### Cited Findings
**Bukas et al. 2026**
- Bukas, Engler, Pelin, … Piraud, "AI-assisted labeling and its pitfalls: A case study in Electron Microscopy segmentation" (MICCAI 2026 workshop, local `docs/references/papers/bukas2026.txt`).
  - "We show how batch effects can be derived by a mix of human and [model labels]".
  - The batch effect is "most evident in the eccentricity and solidity" features. In a harmonized relabelling "batch effects disappear".
  - A model fine-tuned on mixed labels "becomes confused when tracing the border due to labeling noise introduced by the mixed labels and results in lower precision".
  - The mean IoU between mixed and harmonized masks was 0.901, "indicating that SAM2 primarily […]" (text truncated in grep).

**Other evidence on pseudo-label quality**
- SAM-based pseudo-labels for unsupervised nuclei segmentation "can miss nuclei"; adding a U-Net pseudo-label reduces misses but adds over-segmentation ([Nakai et al., ICCV 2025 Workshop CVAMD](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Nakai_Unsupervised_Nuclei_Segmentation_by_Improving_Pseudo_Labels_from_Segment_Anything_ICCVW_2025_paper.pdf); search summary only).
- Lesion-Harvester (Cai et al., IEEE TMI 40(1), 2021) shows the classic mining loop: a high-sensitivity proposal generator, a selective classifier, and a hard-negative suppression loss, iterated.
  - It found 9,805 extra DeepLesion lesions at 90% precision.
  - Detectors trained on harvested labels gained 7–10% AP.
  - [arXiv 2001.07776](https://arxiv.org/pdf/2001.07776); abstract and snippets only.
- Soft Sampling warns that confidence-based re-weighting from a partially trained detector "could generate false positives and consequently ignore regions where there are no objects" ([arXiv 1806.06986](https://arxiv.org/pdf/1806.06986)).
- **Available tools:**
  - μSAM AIS (decoder + watershed) and AMG (`AutomaticMaskGenerator`) are both in `external/micro-sam/micro_sam/instance_segmentation.py`.
  - CellSAM provides detector-prompted automatic segmentation ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12695629/)).
  - Cellpose-SAM is a **bioRxiv preprint** (2025, DOI 10.1101/2025.04.28.651001); no journal version was found ([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.28.651001.full.pdf)).

### Inferences
**Recommended candidate pipeline for C3**
- Use ring evidence (B1's out-of-fold prediction, plus μSAM or Cellpose-SAM zero-shot agreement) only to choose **where to ignore**.
- An ignored pixel injects no model-made shape into the targets, which removes Bukas's mixing mechanism for training.
- **Pseudo-positive variants** (adding unlabelled fibers as labels, Lesion-Harvester style) are a separate ablation. Their g-ratio outputs must be checked for a batch effect: compare per-fiber eccentricity, solidity and g between human-labelled and pseudo-labelled fibers, as Bukas did.
- Agreement between two **independent** proposal sources (B1 and μSAM) is the cheapest guard against a single model's systematic misses. It is no guard against appearance-dependent misses that both share.

### Gaps
- No paper found that measures pseudo-label bias specifically on **g-ratio or ring thickness**.
- No μSAM or CellSAM result on myelinated-axon TEM was found.
- μSAM's paper text was not reachable (Nature login redirect, PMC reCAPTCHA; the CAPTCHA was not touched), so its fine-tuning cost and EM numbers are not given here.

---

## Q4. Evaluation designs for missing-label methods

### Takeaway
Reviewers accept three designs, and C3 can use all three:
1. **Simulated omission on fully labelled data, scored on the complete labels.** It is almost always random or size-based; no top-venue segmentation paper found simulates *appearance-dependent* omission.
2. **A completely re-annotated test subset** (Lesion-Harvester, Penquitt 2026).
3. **Known-positive recall**: deleted labels kept for scoring (the PU detection paper; nnU-Net docs).

Recent work shows that scoring against incomplete labels penalises correct detections (the "observed-label illusion").

### Cited Findings
**Simulated omission (designs used so far)**
- Random drop of 30% of annotations per class (PASCAL VOC), scored on the complete labels ([Soft Sampling, arXiv 1806.06986](https://arxiv.org/pdf/1806.06986)).
- Random 10% removal, with removed objects kept "to calculate detection rates" ([PU detection, arXiv 2002.04672](https://arxiv.org/html/2002.04672v1); snippet).
- Random spatial pixel masks (10/25/50%), on MicroBeSeg and PanNuke ([Sketchpose, MELBA 2025](https://arxiv.org/html/2508.17798)).
- Size-based: "Training Object Detectors if Only Large Objects are Labeled", BMVC 2021 ([BMVA PDF](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0979.pdf); local `pototzky2021_bmvc_only_large_objects_labeled.txt`).
- Random deletion for PU cell detection (Zhao et al., MELBA 2023), and random omission (Vădineanu et al., MIDL 2022). Both come from our earlier C3 novelty report, `docs/reports/C3 incomplete label learning novelty.md`.

**Evidence that real omission is not random**
- Penquitt, Klees, van Betteray, Jashnieh, Stehr, Rottmann, Schmarje, "Object Detection Benchmarks are Incomplete" ([arXiv 2609.21822](https://arxiv.org/abs/2609.21822), Sep 2026, **preprint**; abstract only).
  - Re-annotation added up to about +60% objects (KITTI) and +40% (COCO), "mostly small, occluded, or densely packed".
  - "Missing annotations are the main source of label errors across all datasets."
  - Methods tuned on synthetic noise "struggle" on real label errors. Rankings stay largely stable.
- Pathologists miss atypical-looking cells: an "inherent morphological bias" (Komura et al., *Patterns* 2023; from our C3 novelty report).

**Completely annotated test subsets**
- Lesion-Harvester "completely annotated and publicly released 1,915 of the DeepLesion subvolumes" because the original test data were incompletely annotated, which skews evaluation ([arXiv 2001.07776](https://arxiv.org/pdf/2001.07776); IEEE TMI 2021).
- nnU-Net recommends comparing sparse-annotation strategies on densely annotated images (`external/nnUNet/documentation/ignore_label.md:100–104`).

**Scoring on biased labels**
- Parikh, Frank, Das, Feragen, "Towards Fairness under Label Bias in Image Segmentation" ([arXiv 2605.06891](https://arxiv.org/abs/2605.06891); **preprint**; the comments field names NeurIPS 2026, but no acceptance record was found).
  - A biased label set misrepresents a model's behaviour; correcting the bias looks like failure on the biased labels.
  - Already recorded in `docs/research_notes/Overlooked gaps validation/00_sources.md` and plan D-34.

**Metrics**
- Instance F1@IoU and PQ ([Kirillov et al., CVPR 2019, arXiv 1801.00868](https://arxiv.org/abs/1801.00868)) are the standard instance metrics.
- Metrics Reloaded (Maier-Hein et al., Nature Methods 2024, [arXiv 2206.01653](https://arxiv.org/abs/2206.01653)) is the accepted guide for choosing them. The local PDF is `maierhein2024_metrics_reloaded_2206.01653.pdf`; it was not re-read this session.

### Inferences
**C3 evaluation package (all on full 4096×3008 images, cluster-bootstrap over overlap groups, already in `evaluation/stats.py`)**
- **E1, known-positive recall:** overlap-disagreement fibers (H3), plus thin fibers on outside data (H8). This is the primary endpoint.
- **E2, controlled simulation on our own labels:** take fully labelled fibers and delete them under three schemes, matched for the number of deleted fibers:
  - (a) random;
  - (b) size-dependent: P(drop) rising with small diameter;
  - (c) **appearance-dependent**: P(drop) a logistic function of myelin thickness or g-ratio and of ring contrast, **fitted to the overlap-disagreement fibers**, so the simulated mechanism matches the observed one.

  Score recall on the deleted fibers and g-MAE on all fibers. Then compare C3 against B1 and against the random-ignore control under each scheme.
  - This would be, as far as this search found, the first segmentation paper with a fitted, appearance-dependent omission simulation.
  - It also answers Penquitt's point that synthetic noise does not transfer.
- **E3, official metrics on the original labels**, reported but not used as the decision rule (observed-label illusion). Also report the change in "FPs on unlabelled-real fibers" from the taxonomy.
- **E4, a small completely re-annotated subset** (for example 10–20 test images re-checked by the T-20 expert), to give a Lesion-Harvester-style unbiased estimate.
- **Report** F1@0.5/0.75, PQ, recall by subtype (normal / thin-myelin / abnormal) and g-MAE with fiber recall beside it, using paired bootstrap differences against B1 and against the random-ignore control.

### Gaps
- No MICCAI/MedIA/TMI/CVPR 2025–26 segmentation paper found that simulates **appearance-dependent** omission. The search was not exhaustive (one extended query and two related ones).
- The protocols of the Parikh experiments and of Penquitt's re-annotation were not read beyond the abstract and summaries.
- No literature consensus on how many completely re-annotated images suffice.

---

## Q5. Compute and data efficiency (≈ 6–10 runs of a few GPU-hours on an L4, 266 images of 4096×3008)

### Takeaway
Only nnU-Net-based C3 variants fit comfortably: B1 cost $7.69 for 150 epochs, and hard ignore and soft weighting reuse its plans.
- A post-hoc per-instance classifier for propensity or "abnormal" is nearly free.
- Mask2Former, Mask R-CNN or CellSAM fine-tuning, or Sketchpose-style training, would each need new pipelines and probably several runs to tune. That does not fit ≤ $60 next to the C3 ablation grid.

### Cited Findings
- B1 (nnU-Net ResEnc-M, 2d_bigpatch, 150 epochs) cost $7.69 on a Modal L4 and gives F1@0.5 0.904 (`CLAUDE.md` status, `docs/notes/05-results.md`).
- Sketchpose reports about 5 h of training for MicroBeSeg and about 30 h for PanNuke on one GPU, and calls this a limit on interactivity ([arXiv 2508.17798](https://arxiv.org/html/2508.17798)).
- InstanSeg trains for 500 epochs × 1000 batches of 3 crops at 256×256 and reports no GPU-hour totals. Baselines are described as needing "multiple days of training" ([arXiv 2408.15954](https://arxiv.org/html/2408.15954v1)).
- CellSAM fine-tuning freezes the ViT and the SAM mask decoder in stage 2 ([search summary of Nature Methods 2025 / arXiv 2311.11004](https://arxiv.org/pdf/2311.11004v1.pdf)). No cost figures were found.

### Inferences
**Suggested run budget (about $5–8 per nnU-Net run)**
1. B1 (exists);
2. hard-ignore C3;
3. soft-weight C3 (w = 0.75);
4. random-ignore control;
5. PU or propensity-weighted variant;
6–8. the three E2 simulation schemes (each a B1-vs-C3 pair, or only C3 if B1-on-simulated is trained once per scheme; this needs care, since each scheme changes the training labels, so each needs its own B1).

This already reaches about 8–10 runs, so the simulation (E2) may need to use **fewer epochs or one fold**, or be restricted to schemes (a) and (c).

**Other options**
- A propensity or "abnormal" head as a **post-hoc per-object model** (gradient boosting on fiber features, or a small MLP on pooled ResEnc features) runs on CPU locally, in line with the Modal-is-GPU-only rule.
- If a query model is wanted for the paper's novelty, the cheapest path is **zero-shot CellSAM or μSAM proposals as the candidate source**, not as the trained model.

### Gaps
- No measured L4 cost exists for Mask2Former, Mask R-CNN, CellSAM or μSAM fine-tuning on 4096×3008 TEM. Any figure would be a guess.
- No source gives the data requirement for Mask2Former from SA-1B or COCO pretraining on about 266 images of about 30 objects each.
