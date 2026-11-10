# Topology-aware, boundary/Hausdorff losses and EM pretraining for nested axon (disk) + myelin (annulus) segmentation (as of 10-11 Nov 2026)

Scope note: this was a time-boxed pass of about 15 tool calls. I verified the primary sources (paper text or official abstract) for TopoInteraction, TopoMortar, boundary loss, the HD loss abstract, Ma et al. (MIDL 2020), CEM500K, the GAN-SSL EM paper and 1907.05143. Everything else comes from search snippets of official pages and is flagged as such. Local copies are in `docs/references/papers/` (`arxiv_2207.09654.pdf/.txt` for TopoInteraction). The TopoMortar PDF in that folder is truncated or corrupt, so I read the arXiv HTML v2 instead.

Prior-review audit (section 4 of `prior-literature-review-2026-11-10.md`):

| Prior-review claim | Status |
|---|---|
| TopoInteraction is ECCV 2022 oral, enforces containment and exclusion through forbidden adjacent label pairs, is convolutional and efficient, and its code is in TopoXLab/TopoInteraction | **CONFIRMED** |
| Skeleton Recall is ECCV 2024 | **CONFIRMED** |
| "5.6 vs 16.2 GiB, 7.1 vs 3.1 h (TopoMortar, Table 1)" | **Numbers CONFIRMED, table CORRECTED.** They are in **Table 6 (Appendix O)** of v2, not Table 1. TopoMortar is now **BMVC 2025 (Oral)**, not just an arXiv preprint |
| Quoted TopoMortar conclusion | **CORRECTED.** It is a paraphrase. The exact text is "clDice generally achieved the most topologically accurate segmentations while Skeleton Recall performed best on noisy labels." |
| Boundary loss ISLES numbers 0.511→0.644/0.659 and HD95 5.320→4.795/2.725 mm, "about 13%" | **CONFIRMED** (arXiv 1812.07032v4) |
| HD loss (Karimi & Salcudean, TMI 2020), differentiable HD surrogates | **Mechanism CONFIRMED** (arXiv 1904.10030). The TMI venue was **not re-verified** this session |
| clDice CVPR 2021; Betti matching ICML 2023 | Betti matching ICML 2023 **CONFIRMED** (PMLR v202 stucki23a). clDice CVPR 2021 **UNVERIFIED this session**; the only support is TopoMortar citing it |
| "No top-venue topology-aware segmentation survey found" | **Still UNVERIFIED.** I also found none. The efficient Betti-matching paper cites a "recent survey and position paper [8]" that I could not identify |
| CEM500K (eLife 2021), RETINA (PLoS Comput Biol 2025), GAN SSL (CVPRW 2024) | CEM500K **CONFIRMED**. RETINA in PLoS CB **CONFIRMED** through PMC12143494. GAN SSL arXiv **CONFIRMED** (2402.18286), but the **CVPRW venue is UNVERIFIED** |

---

## Q1. TopoInteraction: how containment and exclusion are encoded, cost and results; follow-ups on multi-class, containment and nested priors

### Takeaway
TopoInteraction maps both containment and exclusion onto one rule: label pair (A, C) must never sit in adjacent pixels. A 3x3 (or 3x3x3) convolution finds the "critical pixels" that break the rule, and a pixel loss is re-applied only on those pixels. It adds about 0.8 s per epoch on IVUS and nothing at inference. It fits "axon ⊂ myelin" and "axon never touches background" exactly. It does **not** count holes (β1) and does **not** encode instance-level exclusion between different fibers, because it works on semantic labels only. I found no 2023–2026 top-venue follow-up dedicated to nested or annular multi-class constraints.

### Cited Findings
- **Encoding.** "Exclusion: forbid A=α and C=γ in adjacent pixels. Containment: if β contains α, set A=α and C = the union of all other labels except α and β. Then β containing α is equivalent to A=α not touching C." For axon/myelin this gives A = axon and C = background (plus any other class), which is exactly "axon never adjacent to background". [Gupta et al., ECCV 2022, arXiv 2207.09654 §3.1](https://arxiv.org/abs/2207.09654)
- **Generalized constraints.** *d-containment* requires the surrounding class to be at least d pixels thick. *d-exclusion* requires a gap of at least d pixels. Both are implemented with an all-ones kernel of size k=2d+1. A minimum myelin thickness or a minimum inter-fiber gap could be encoded the same way. [arXiv 2207.09654 §3](https://arxiv.org/abs/2207.09654)
- **Loss.** L_ti = L_pixel(f⊙V, g⊙V), where V is the binary critical-pixel map computed from the current prediction. L_total = L_ce + λ_dice·L_dice + λ_ti·L_ti. Experiments used a 3x3 all-ones kernel (8-connectivity) in 2D and 3x3x3 (26-connectivity) in 3D, on one V100 32 GB. [arXiv 2207.09654 Eq. 3–4, §4](https://arxiv.org/abs/2207.09654)
- **Cost.**
  - Complexity is O(N² log N) through FFT convolution, independent of k. The naive loop is O(N²k²) and CPU-only.
  - Memory is O(N²) for V.
  - On IVUS (batch 5, 384×384, d=1), computing L_ti took 69.4 s per epoch with the naive method, 5.9 s with the "alternate naive" method and **0.8 s** with the convolution method.
  - Inference time is unchanged.
  
  [arXiv 2207.09654 Supplement](https://arxiv.org/abs/2207.09654)
- **Containment results (Table 1, mean ± sd).**
  - **IVUS lumen** (contained by media): nnU-Net Dice 0.893 → **0.949**, HD 3.464 → **2.046**, % violations 2.708 → **0.157**.
  - **IVUS media**: Dice 0.856 → **0.910**, HD 5.646 → **3.873**.
  - **Aorta lumen** (in wall): Dice 0.906 → **0.922**, HD 36.37 → **25.96**, violations 5.424 → **3.526**.
  - **Aorta wall**: Dice 0.741 → **0.758**, HD 42.49 → **31.14**.
  - Baselines also included TopoCRF, MIDL and NonAdj.
  
  [arXiv 2207.09654 Table 1](https://arxiv.org/abs/2207.09654)
- **Exclusion results (Table 2, Multi-Atlas spleen).** Dice 0.950 → 0.960, HD 6.084 → 5.340, violations 0.819 → 0.464. The authors note the gains are mainly in HD/ASSD when nnU-Net already separates the classes. [arXiv 2207.09654 Table 2](https://arxiv.org/abs/2207.09654)
- **Ablation (Table 3, IVUS).** The choice of L_pixel (CE, MSE, Dice) changes results. For example, Dice-based L_pixel gave lumen Dice 0.937 and HD 2.385. [arXiv 2207.09654 Table 3](https://arxiv.org/abs/2207.09654)
- **Follow-up 1.** "Preserving Cardiac Integrity: A Topology-Infused Approach to Whole Heart Segmentation" (arXiv 2410.10551, **preprint**) combines a topology-preserving conv module with a topology-constrained loss. It applies an exclusion constraint (for example Myo vs LA on WHS++). [arXiv 2410.10551](https://arxiv.org/html/2410.10551v3)
- **Follow-up 2.** "Multi-level Activation for Segmentation of Hierarchically-Nested Classes" proposes a multi-level activation layer that can encode any tree of containment relations between classes. This maps directly to an axon ⊂ fiber structure. The venue is a Springer LNCS chapter; I did not confirm which ECCV-2018 workshop. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-11024-6_24)
- **Persistent-homology multi-class prior.** Clough et al. propose a PH topological loss for multi-class cardiac MRI segmentation that covers class containment and adjacency, using prior Betti numbers. [arXiv 2008.09585](https://arxiv.org/pdf/2008.09585); [arXiv 2107.12689](https://arxiv.org/pdf/2107.12689)
- **Topology-Aware Uncertainty** (Gupta, Zhang, Hu, Prasanna, Chen; **NeurIPS 2023** main track; arXiv 2306.05671). It gives structure-wise rather than pixel-wise uncertainty for **curvilinear** structures using discrete Morse theory and "perturb-and-walk" sampling. Code: github.com/Saumya-Gupta-26/struct-uncertainty. It is not a containment prior, but it could rank uncertain fibers for active learning. [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract.html)

### Inferences
- For this task, the following constraints are directly encodable with TopoInteraction on 3 classes {BG, myelin, axon}:
  - Containment "myelin contains axon": A = axon, C = BG.
  - d-containment for minimum myelin thickness.
- **Inter-fiber exclusion is not encodable semantically.** Axon_i touching myelin_j is legal at the semantic level. Two options:
  - Add an explicit inter-fiber boundary class (ASTIH-style) and impose exclusion between "boundary" and "axon".
  - Apply TI per instance crop.
- TI has no notion of β1. A myelin ring with a gap still satisfies "axon not touching BG" only if the gap is filled by myelin. A broken ring where axon leaks to BG **is** penalized, so TI does indirectly punish open rings at the axon side. Pair it with a hole-count or Betti term on fiber crops if β1=1 per ring matters.
- Cost is negligible at 4096×3008 (it is just a convolution). This makes it the cheapest topology prior in this catalogue.

### Gaps
- I did not check the TopoXLab repo code itself (the clone timed out), so I could not confirm the license or the API signature beyond the search snippet (`TI_Loss` takes inclusion/exclusion pairs).
- I found no 2023–2026 top-venue paper on "nested/annular" topology priors specifically. "TopoSeg" (ICCV 2023) and "Topologically Faithful Image Segmentation via Induced Matching" details beyond the venue were not fetched.

---

## Q2. clDice, Skeleton Recall, Betti matching, efficient Betti matching, TopoMortar, cbDice, TRACE and surveys

### Takeaway
On TopoMortar (BMVC 2025 Oral, 10 seeds, significance tests), **clDice was the only topology loss that improved topology in most settings**, and Skeleton Recall won with noisy labels. More importantly, simple augmentation plus self-distillation lifted plain CE+Dice above most topology losses. The cost differences are large: Warping 152 h, cbDice 58 h and TopoLoss 50 h, versus CEDice 2.3 h.

### Cited Findings
- **TopoMortar Table 6 (Appendix O).** nnU-Net, 12,000 iterations, batch 10, V100 32 GB.

  | Loss | GPU memory (GiB) | Training time (h) |
  |---|---|---|
  | CEDice | 5.6 | 2.3 |
  | RWLoss | 5.6 | 14.5 |
  | TopoLoss | 5.6 | 49.6 |
  | TOPO | 5.9 | 28.8 |
  | **clDice** | **16.2** | **3.1** |
  | Warping | 5.6 | 152.1 |
  | **SkelRecall** | **5.6** | **7.1** |
  | cbDice | 7.8 | 58.3 |

  [TopoMortar arXiv 2503.03365v2](https://arxiv.org/html/2503.03365v2)
- **TopoMortar exact conclusions.**
  - "clDice generally achieved the most topologically accurate segmentations while Skeleton Recall performed best on noisy labels. Additionally, data augmentation and self-distillation helped in improving topology accuracy, even when employed jointly with topology loss functions."
  - "no topology loss function was the best across all settings."
  - Data augmentation and self-distillation "can elevate Cross entropy Dice loss to surpass most topology loss functions."
  
  [arXiv 2503.03365v2](https://arxiv.org/html/2503.03365v2)
- **TopoMortar mean ID Dice** under dataset challenges: 0.90 with a small training set, 0.86 with pseudo-labels and 0.68 with noisy labels. RandHue colour augmentation improved OOD Dice and significantly decreased Betti errors on both ID and OOD sets. [arXiv 2503.03365v2](https://arxiv.org/html/2503.03365v2)
- **TopoMortar on DRIVE vs FIVES.** Six losses beat CEDice significantly on DRIVE, but only three did on FIVES, the larger dataset. Skeleton Recall ranked 5th on DRIVE and 2nd on FIVES. [arXiv 2503.03365v2](https://arxiv.org/html/2503.03365v2)
- **Skeleton Recall** (Kirchhoff, Rokuss et al., ECCV 2024, arXiv 2404.03010).
  - Recall loss on a precomputed CPU "tubed skeleton" of the GT.
  - Claims SOTA on 5 datasets and ">90%" lower compute overhead than prior topology losses.
  - Feasible for multi-class problems where clDice runs out of memory.
  - Code: MIC-DKFZ/Skeleton-Recall.
  
  [ECCV 2024 poster](https://eccv2024.ecva.net/virtual/2024/poster/1449); [arXiv 2404.03010](https://www.arxiv.org/pdf/2404.03010v1)
- **Betti matching** (Stucki et al., **ICML 2023**, PMLR v202). Induced matching of persistence barcodes defines a loss on matched features. **Efficient Betti Matching** (arXiv 2407.04683) is a C++ implementation with Python bindings that is much faster than Cubical Ripser. Code: github.com/nstucki/Betti-Matching-3D. [PMLR](https://proceedings.mlr.press/v202/stucki23a/stucki23a.pdf); [arXiv 2407.04683](https://arxiv.org/html/2407.04683)
- **Spatial-aware persistent feature matching** (Wen et al., **ICCV 2025 Workshop BIC**, arXiv 2412.02076). Reports performance comparable to Betti matching at much lower compute. [CVF](https://openaccess.thecvf.com/content/ICCV2025W/BIC/papers/Wen_Topology-Preserving_Image_Segmentation_with_Spatial-Aware_Persistent_Feature_Matching_ICCVW_2025_paper.pdf)

### Inferences
- For annular myelin, clDice and Skeleton Recall are designed for tubular centrelines. A myelin ring's skeleton is a closed loop, so either loss penalizes ring breaks. Skeleton Recall is the memory-cheap choice when there are thousands of rings per 4096×3008 image.
- TopoMortar implies the first step should be strong staining/contrast augmentation plus self-distillation on CE+Dice. Add topology losses only if Betti errors remain.

### Gaps
- **cbDice**: the venue (I believe MICCAI 2024) and its numbers were not verified this session.
- **TRACE**: I found no source this session.
- **Surveys**: no 2025–2026 top-venue survey of topology-aware segmentation was confirmed. One is referenced as "[8]" in arXiv 2407.04683 but was not identified.
- Exact Skeleton Recall per-dataset numbers were not retrieved.

---

## Q3. Boundary loss, HD loss, distance-map auxiliaries, Generalized Surface Loss and the loss survey

### Takeaway
Boundary loss gains are real but dataset- and difficulty-dependent: they are large when the regional loss struggles (ISLES) and small otherwise. Ma et al. explicitly report that DT-based methods do not all help on every dataset and are sensitive to implementation details.

### Cited Findings
- **Boundary loss** (Kervadec et al., MIDL 2019 best-paper runner-up, PMLR 102:285–296; MedIA 67, Jan 2021). Table, ISLES / WMH, DSC and HD95 in mm:

  | Loss | ISLES DSC | ISLES HD95 | WMH DSC | WMH HD95 |
  |---|---|---|---|---|
  | GDL | 0.511 | 5.320 | 0.768 | 3.634 |
  | + L_B (2D) | **0.644** | **4.795** | 0.793 | 2.039 |
  | + L_B (3D) | **0.659** | **2.725** | 0.818 | 1.702 |

  - Quote: "adding the boundary loss yielded about 13% improvement in DSC over using Generalized Dice loss alone, and about 3% improvement over using UNet cross-entropy or focal loss alone… The more difficult the tasks… the larger the gain."
  - Using L_B alone is "NA", meaning it fails to train on its own.
  
  [arXiv 1812.07032v4](https://arxiv.org/abs/1812.07032)
- **HD loss** (Karimi & Salcudean, arXiv 1904.10030; published in IEEE TMI 2020, venue not re-verified this session).
  - Three HD estimators: distance transform of the boundary, morphological erosion, and circular-kernel convolutions.
  - "approximately 18–45% reduction in HD without degrading … Dice" on prostate, liver and pancreas in US, MR and CT.
  
  [arXiv 1904.10030](https://arxiv.org/abs/1904.10030)
- **Ma et al., "How Distance Transform Maps Boost Segmentation CNNs"** (MIDL 2020, PMLR 121:479–492).
  - The five methods all improved V-Net, "However, the implementation details have a noticeable impact on the performance, and **not all the methods hold the benefits on different datasets**."
  - Recommendation: multi-head/reconstruction branches for organs, boundary/HD losses for tumours.
  - Code: JunMa11/SegWithDistMap.
  
  This answers the "can DT regression degrade results" question as **dataset-dependent; not universally beneficial**. [PMLR](https://proceedings.mlr.press/v121/ma20b.html); [GitHub](https://www.github.com/JunMa11/SegWithDistMap)
- **Loss survey** (arXiv 2312.05391, **preprint**). Reviews 25 segmentation losses with a taxonomy and runs unbiased evaluations on medical and natural datasets. [arXiv 2312.05391](https://arxiv.org/abs/2312.05391)
- **Distance-map penalty term** (Caliva et al., MIDL 2019 extended abstract, arXiv 1908.03679). A GT distance-map weighting gave better boundary shape preservation on OAI bone MRI. [arXiv 1908.03679](https://arxiv.org/abs/1908.03679)

### Inferences
- The challenge's HD95 is in nm on 2.15 nm/px images, so a 1-pixel boundary shift is about 2 nm. Myelin is thin, and inner-contour errors directly affect g-ratio. Boundary or HD losses on **both** the inner (axon) and outer (fiber) contours are justified, with HD95 measured per contour.
- Start the boundary-loss weight small and ramp it up, as Kervadec does, because the loss does not train well on its own.

### Gaps
- **Generalized Surface Loss** (Celaya et al.): the venue and numbers were not retrieved.
- I found no paper that directly optimizes a **morphometric** (g-ratio, area ratio, diameter) as a differentiable loss for ring objects. Within this budget it is an apparent open gap and a plausible novelty angle, but it is not proven absent.

---

## Q4. EM SSL and foundation models: what was pretrained, gains, weights; relevance to TEM myelin

### Takeaway
Several EM-specific pretrained encoders exist with code or weights: CEM500K (MoCoV2 ResNet-50), RETINA, EM-DINO/OmniEM, EMCF and a GAN-SSL model. None that I found reports results on TEM myelin rings or on cross-staining-protocol transfer, so the benefit for this task must be tested empirically.

### Cited Findings
- **CEM500K** (Conrad & Narayan, **eLife 2021**).
  - Data: 25 GB, 0.5×10⁶ unique 2D cellular EM images curated from ~600 3D and 10,000 2D images across >100 unrelated projects.
  - Models pretrained on it "learn features that are biologically relevant and resilient to meaningful image augmentations".
  - Reports SOTA on 6 public and 1 new benchmark segmentation tasks.
  - Dataset, pretrained models and pipeline released (EMPIAR-10592, git.io/JLLTz).
  
  [eLife 65894 / PMC8032397](https://elifesciences.org/articles/65894)
- **RETINA** (PLoS Comput Biol 2025, PMC12143494; code BaderLab/RETINA).
  - Reconstruction-based SSL on a TransUNet pretrained on CEM500K.
  - Outperformed existing models on 5 public EM datasets and converged faster than a CEM500K-pretrained UNet-ResNet50.
  - Combining ImageNet and CEM500K pretraining did not beat CEM500K alone, except a slight mitochondria gain.
  - Limited to 2D.
  
  [PMC12143494](https://pmc.ncbi.nlm.nih.gov/articles/PMC12143494/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.08.632030v1.full)
- **GAN SSL for EM** (Kazimi, Ruzaeva, Sandfeld; arXiv 2402.18286v2, Jul 2024; CVPRW venue not verified).
  - Pretraining helps segmentation, denoising, background removal and super-resolution.
  - "Fine-tuned models of lower complexity consistently outperform more complex models with random weight initialization", especially when labels are limited.
  - The data is materials-science EM (cond-mat.mtrl-sci).
  
  [arXiv 2402.18286](https://arxiv.org/abs/2402.18286)
- **EM-DINO / OmniEM** (bioRxiv 2025.04.13.648639, **preprint**). DINOv2-style self-distillation on the curated "EM-5M" dataset. Code for both the EM-DINO foundation model and the fine-tuned OmniEM was released. [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.13.648639v4.full)
- **EMCF ecosystem** (bioRxiv 2025.12.09.693109, **preprint**). EMCellFound is an MAE-pretrained generalist encoder on "EMCFD". EMCellFiner is a HAT-based super-resolution model. Weight availability was not confirmed. [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.09.693109v1.full)
- **SEM mixture-of-experts foundation model** (arXiv 2604.05960, Apr 2026, **preprint**). About 1.7B parameters with MAE pretraining. It covers SEM only, and the weight release was unconfirmed. [arXiv 2604.05960](https://arxiv.org/html/2604.05960v1)

### Inferences
- CEM500K-pretrained ResNet-50 (via MoCoV2) and EM-DINO are the two most practical drop-in encoders. A cheap ablation is ImageNet vs CEM500K vs EM-DINO init on nnU-Net/U-Net.
- The 134 unlabeled challenge test images, plus ASTIH TEM1/TEM2, could be used for in-domain continued SSL (MAE/DINO) before fine-tuning.

### Gaps
- Exact Dice gains for CEM500K and RETINA per benchmark were not extracted.
- The arXiv paper "Self-Supervised Learning in EM: Towards a Foundation Model" was not located this session.
- No DINOv3-on-EM paper was verified.
- I found no evidence that any of these models was evaluated on myelin or on cross-protocol TEM.

---

## Q5. Domain generalization and stain/contrast robustness for EM and histology

### Takeaway
EM-specific domain adaptation is mostly unsupervised or weakly supervised work on organelle and mitochondria datasets. Contrast/stain shift in nerve EM is handled mainly by histogram/CLAHE alignment. I found no EM-specific DG survey. TopoMortar shows that simple colour augmentation significantly reduces OOD topology errors.

### Cited Findings
- **EM UDA.**
  - Adversarial-prediction-guided multi-task adaptation for EM: [arXiv 2004.02134](https://arxiv.org/abs/2004.02134v1).
  - Weakly-supervised DA with sparse points: [arXiv 2210.13109](https://arxiv.org/abs/2210.13109v3).
  - Prompt-DAS: [arXiv 2509.18973](https://arxiv.org/abs/2509.18973v1).
  - BatchNorm-based fast UDA for FIB-SEM mitochondria: [HAL 04017099](https://hal.univ-brest.fr/ENGEES/hal-04017099v1).
- **Peripheral-nerve EM (J. Imaging 2026).** A model trained on high-contrast uranyl-stained images often fails on uranyl-free images. The paper uses CLAHE/histogram contrast alignment. [Borisovs & Cavaletti 2026](https://www.boa.unimib.it/retrieve/67fc9b78-86d3-4623-9223-36cb4b7617f3/Borisovs-Cavaletti-2026-J.%20Imaging-VoR.pdf)
- **RandHue on TopoMortar.** Colour augmentation (HSV hue, saturation, value) applied with p=0.5 significantly lowered Betti errors in ID and OOD. [arXiv 2503.03365v2](https://arxiv.org/html/2503.03365v2)

### Inferences
- For grayscale TEM, the analogue of RandHue is aggressive intensity augmentation: gamma, contrast, CLAHE, histogram matching to random references, and simulated stain or thickness variation. Combine it with BN-statistics test-time adaptation as a low-cost baseline.

### Gaps
- I found no published survey of DG for EM. The histopathology test-time adaptation paper ("FUSION") was seen only in snippets.
- I found no TTA study on EM stain shift.

---

## Q6. Semi-supervised and active learning with few labels for microscopy and axon/myelin

### Takeaway
The only axon/myelin-specific active-learning paper verified is arXiv 1907.05143. It used MC-dropout uncertainty sampling and reached maximum Dice after adding 3 selected images. TopoMortar's self-distillation (iterative soft pseudo-labels) is a simple semi-supervised lever that also improved topology.

### Cited Findings
- **Deep Active Learning for Axon-Myelin Segmentation on Histology Data** (arXiv 1907.05143, 2019).
  - U-Net with MC-dropout overall uncertainty selects samples to annotate.
  - Two small datasets: 14 SBEM images and 24 TEM images.
  - "Reached a maximum Dice value after adding 3 uncertainty-selected samples to the initial training set" for myelin segmentation.
  
  [arXiv 1907.05143](https://arxiv.org/abs/1907.05143)
- **TopoMortar self-distillation.** Three rounds of 4,000 iterations, retraining on the model's own soft labels each time. It improved Dice and topology with pseudo-labels and noisy labels. [arXiv 2503.03365v2](https://arxiv.org/html/2503.03365v2)
- **Topology-Aware Uncertainty** (NeurIPS 2023) gives structure-level uncertainty that could rank fibers rather than pixels for annotation. [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract.html)

### Inferences
- With 266 labeled and 134 unlabeled test images, the following are cheap and evidence-backed:
  - Mean-teacher or self-distillation pseudo-labelling on the unlabeled images (transductive, if the rules allow).
  - Optionally adding ASTIH TEM1.

### Gaps
- No mean-teacher or pseudo-label study on TEM myelin specifically was verified this session.
