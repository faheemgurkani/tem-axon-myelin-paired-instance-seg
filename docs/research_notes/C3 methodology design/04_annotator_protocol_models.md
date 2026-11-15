# Annotator and protocol models as an architecture for C3 (15 Nov 2026)

Scope: methods that explicitly model the annotator or the annotation protocol in segmentation, judged by one question: can they represent **object omission** (an annotator or protocol skipping whole instances, depending on appearance), and how do they keep the latent "true" segmentation identifiable? Zepf et al. (ICLR 2023), Parikh et al. (arXiv 2605.06891, 2026) and Bukas (MICCAI-W 2026) are already covered in `docs/reports/Overlooked gaps validation.md`. They are only extended here, not re-described. The PU and SAOD papers in `docs/research_notes/C3 incomplete label learning novelty/00_papers_read.md` (LNQ atlas weighting, Zhao PU cells, Vădineanu omission study, SAOD) are not repeated.

**How each source was read:**

| Source | How read |
|---|---|
| Zhang et al. 2020 | full text (ar5iv HTML, first 100k of 101.5k characters) |
| Pionono | full methods (arXiv HTML) |
| MultiTalent | full methods (arXiv HTML v2) |
| Tanno CVPR 2019 | search result and abstract only; the formulas are known through Zhang 2020's restatement |
| CCEM/GeoCrowdNet | first 100k of 158k characters (arXiv HTML) |
| Shi et al. MedIA 2021 | search results and abstract only |
| Zepf 2023 | abstract only |
| 2025–26 multi-rater papers | search results only (titles and snippets) |
| Nguyen et al. NeurIPS 2024 | search snippet only; OpenReview showed a bot-check page, which was not solved |

---

## 1. Annotator confusion-matrix models: formulation, identifiability, and can they represent appearance-dependent "foreground labelled as background"?

### Takeaway
Pixel-wise, image-dependent confusion-matrix (CM) models (Zhang et al., NeurIPS 2020) can *formally* represent "this annotator labels true foreground as background here". They cannot represent **whole-object omission** coherently, because they assume pixels are independent given the image. Their identifiability rests on (a) several annotators and (b) a trace regulariser plus diagonal dominance. With one protocol and one-sided omission, that regulariser points the **wrong way**: it rewards explaining all unlabelled tissue as "true fiber, omitted". No CM-family paper found models object omission.

### Cited Findings
**Tanno et al., CVPR 2019 (the origin, classification).**
- Each annotator is a confusion matrix, "jointly estimated along with the classifier predictions". A regularisation term "encourages convergence to the true annotator confusion matrix" — [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Tanno_Learning_From_Noisy_Labels_by_Regularized_Estimation_of_Annotator_Confusion_CVPR_2019_paper.html).
- CVPR 2019, pp. 11244–11253 — [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tanno_Learning_From_Noisy_Labels_by_Regularized_Estimation_of_Annotator_Confusion_CVPR_2019_paper.pdf).
- Read only as an abstract. The trace form below is how Zhang 2020 restates it.

**Zhang, Tanno, Xu, Jin, Jacob, Ciccarelli, Barkhof, Alexander, "Disentangling Human Error from Ground Truth in Segmentation of Medical Images", NeurIPS 2020** — [NeurIPS poster](https://neurips.cc/virtual/2020/poster/18835); [proceedings PDF](https://proceedings.neurips.cc//paper/2020/file/b5d17ed2b502da15aa727af0d51508d6-Paper.pdf); [arXiv 2007.15963 / ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- **Assumptions:** "(1) annotators are statistically independent, (2) annotations over different pixels are independent given the input image"; and "a single, unknown, true segmentation map" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- **Observation model:** p(ỹ_wh^(r) | x) = Σ_{y_wh} p(ỹ_wh^(r) | y_wh, x) · p(y_wh | x). The per-pixel CM is a^(r)(x,w,h)_ij := p(ỹ^(r)_wh = i | y_wh = j, x). It is "pixel-wise … on a per image basis" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- **Loss (Eq. 4):** L_total(θ,φ) = Σ_n Σ_r 1(r∈S(x_n)) · [CE(Â^(r)_φ(x_n)·p̂_θ(x_n), ỹ_n^(r)) + λ·tr(Â^(r)_φ(x_n))]. "Minimising the trace encourages the estimated annotators to be maximally unreliable" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- **Identifiability:**
  - Without the regulariser, "there are many combinations of pairs Â and segmentation model p̂" that fit the data.
  - Theorem 1 needs the true *average* CM to be diagonally dominant (a*_kk > a*_kj) and the estimate to be diagonally dominant. Then the true CMs are the argmin of the trace.
  - The authors call this "weaker than" Tanno 2019's population result.
  - Diagonal dominance is encouraged in practice by initialising the CMs to the identity — [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- **Architecture:** a 2D U-Net. The segmentation and annotator networks "share the same parameters apart from the last layer in the decoder". The annotator head outputs Â^(r)(x) ∈ [0,1]^{W×H×L×L} per annotator. A rank-1 low-rank variant is optional — [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- **Missing annotator–image pairs:** handled only by the indicator 1(r∈S(x)), which requires |S(x)| ≥ 1. The one-label-per-image setting was tested — [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- **Results (Dice %, STAPLE → Global CM → pixel-wise CM):**

  | Dataset | STAPLE | Global CM | Pixel-wise CM |
  |---|---|---|---|
  | MSLSC, dense | 55.05 | 61.58 | 67.55 |
  | MSLSC, one label | 35.74 | 40.32 | 56.43 |
  | BraTS, dense | 46.73 | 47.33 | 53.47 |
  | LIDC, one label | 57.32 | 63.25 | 68.12 |

  MNIST, MSLSC and BraTS use **simulated** annotators, and the LIDC "truth" is a STAPLE aggregate — [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- **Code:** [moucheng2017/Learn_Noisy_Labels_Medical_Images](https://github.com/moucheng2017/Learn_Noisy_Labels_Medical_Images) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963).
- NeurIPS reviewers questioned whether some ambiguity is genuine rather than error — [search summary of NeurIPS review page](https://proceedings.neurips.cc/paper/2020/file/b5d17ed2b502da15aa727af0d51508d6-Review.html) (the page itself was not read).

**Pionono: Schmidt, Morales-Álvarez, Molina, "Probabilistic Modeling of Inter- and Intra-observer Variability in Medical Image Segmentation", ICCV 2023** (not MICCAI 2023 as the brief said) — [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Schmidt_Probabilistic_Modeling_of_Inter-_and_Intra-observer_Variability_in_Medical_Image_ICCV_2023_paper.html); [arXiv 2307.11397 HTML](https://arxiv.org/html/2307.11397v1).
- **Model:** the features v = f_ω(x) (ResNet34 U-Net, L=16 channels) are concatenated with a broadcast latent z ∈ ℝ^8. A head of three 1×1 convolutions then predicts s̃ = f_θ(v, z̃).
- **Latent rater code:**
  - Prior p(z) = N(0, σ_prior I), with σ²_prior = 2.
  - Per-rater posterior q(z|r) = N(μ^r, Σ^r), with trainable μ^r and a Cholesky-parameterised Σ^r.
  - An extra "gold rater" r = M+1 is trained on gold segmentations — [arXiv HTML](https://arxiv.org/html/2307.11397v1).
- **Loss:** "ℒ_ELBO = 𝔼_q log p(S^r|X,r,ω,θ) − λ KL(q(Z|r)|p(Z))", with λ = 0.0005. The likelihood term is a generalised Dice loss. The rater parameters use a higher learning rate (0.02) — [arXiv HTML](https://arxiv.org/html/2307.11397v1).
- **Inference:** sample z ~ q(z|M+1) for the consensus (the mean is the segmentation, the variance the uncertainty), or z ~ q(z|r) to imitate rater r — [arXiv HTML](https://arxiv.org/html/2307.11397v1).
- **Results:** Gleason 2019, 6 pathologists, gold = STAPLE, Cohen's κ:

  | Method | κ |
  |---|---|
  | Pionono | 0.758 ± 0.011 |
  | STAPLE | 0.75 |
  | Prob. U-Net | 0.741 |
  | Global CM (Tanno) | 0.721 |
  | Pixel CM (Zhang 2020) | 0.692 |

  Source: [arXiv HTML](https://arxiv.org/html/2307.11397v1). Code: [arneschmidt/pionono_segmentation](https://github.com/arneschmidt/pionono_segmentation).
- **Identifiability:** there is no explicit identifiability argument. The "truth" is whatever the gold-rater code reproduces, so gold labels are required — [arXiv HTML](https://arxiv.org/html/2307.11397v1).

**Crowdsourcing identifiability theory (classification).**
- **CCEM / GeoCrowdNet** (Ibrahim, Nguyen, Fu; arXiv 2306.03288, preprint as read; the page gives no venue) — [arXiv HTML](https://arxiv.org/html/2306.03288).
  - Model: p_n^(m) = A_m f(x_n), with cross-entropy over the observed annotator–item pairs. A_m is **instance-invariant**.
  - Identifiability needs one of two conditions:
    - near-class-specialist annotators plus near-anchor items (Thm 1);
    - a "sufficiently scattered condition" on the stacked network outputs or the stacked CMs (Thm 2–3).
  - Regularisers: −λ log det(F Fᵀ) or −λ log det(Wᵀ W).
  - Results: MNIST, M=15 machine annotators, 80.97% vs CrowdLayer 65.72%; LabelMe 85.85% vs 81.41% — [arXiv HTML](https://arxiv.org/html/2306.03288).
- **Nguyen, Ibrahim, Fu, "Noisy Label Learning with Instance-Dependent Outliers: Identifiability via Crowd Wisdom", NeurIPS 2024** — [NeurIPS poster](https://neurips.cc/virtual/2024/poster/95831); [paper PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/b05e6a11df6eacb600074a42bb28ae52-Paper-Conference.pdf).
  - A search summary says it finds that "using labels produced by only one annotator is fundamentally insufficient" to detect instance-dependent outliers or identify the ground-truth classifier, and that it restores identifiability with multiple annotators and a designed loss.
  - Search snippet only; the OpenReview page showed a bot-check, which was not solved.
- A 2024 signal-processing review notes that CM methods traditionally need the annotator response matrix to be fully observed — [arXiv 2407.06902](https://arxiv.org/html/2407.06902v2) (snippet only).

**Positive-unlabelled (PU) learning with a "selected at random" labelling mechanism (the formal twin of C3's truth × selection).**
- Bekker & Davis (PMLR v94, 2018) replace "selected completely at random" with **SAR**: labelled positives are selected with a propensity e(x) that depends on attributes. "The propensity score is the probability that a positive example is selected to be labeled." Known propensities allow unbiased training, and an EM method learns under SAR — [PMLR v94](https://proceedings.mlr.press/v94/bekker18a.html); [arXiv 1809.03207](https://arxiv.org/pdf/1809.03207v4.pdf).
- A 2024 statistics preprint calls the "identifiability of the distributions of unlabeled positive and unlabeled negative data" the foremost challenge in SAR-type PU — [arXiv 2407.09735](https://arxiv.org/html/2407.09735v1) (snippet only).

### Inferences
- **The C3 model is the one-sided special case of the Zhang CM.**
  - For binary fiber/background, set a_{fg→fg}(x) = s(x), a_{fg→bg}(x) = 1 − s(x), a_{bg→bg} = 1, a_{bg→fg} ≈ 0. Then p_obs(fg|x) = p_true(fg|x) · s(x). This is exactly "observed = truth × protocol selection".
  - It is also the SAR-PU model with propensity e(x) = s(x).
  - So C3's latent-truth idea should be framed as applying a known family (Zhang 2020 / SAR-PU), not as a new likelihood.
- **The Zhang trace regulariser is harmful in this setting.**
  - Here tr(A) = 1 + s(x), so minimising it drives s → 0.
  - On unlabelled pixels the CE term is satisfied equally by (t=0, s=anything) and by (t=1, s=0). The trace term then prefers t = 1, s = 0: "all unlabelled tissue is an omitted fiber".
  - Zhang's identifiability rests on *several* annotators disagreeing. With a single protocol, nothing pulls the other way.
  - C3 therefore needs either the opposite prior, or anchors (see §4).
    - The opposite prior is a sparsity or "omission is rare" prior on 1 − s, or a ring-evidence prior on t.
- **Pixel independence cannot express whole-object omission.**
  - The Zhang CM can lower s(x) on a fiber's pixels, but nothing ties all pixels of one fiber, axon *and* myelin ring together, to a single omit or keep decision. A half-omitted fiber has positive likelihood.
  - The selection variable should live at the **instance level**: one Bernoulli s_k per fiber k, as in detection-style PU or a Dawid–Skene model over objects.
- **The appearance dependence is real in the Zhang CM** (Â depends on x), so "abnormal-looking fibers are skipped" is representable. But it is confounded with p_true(fg|x) at the same pixels: both are functions of the same features.
  - Without anchors, the network can lower t instead of s, and the likelihood is identical.
  - This is the instance-dependent non-identifiability that Nguyen et al. 2024 describe for a single annotator. That statement is from a snippet; verify it before citing.
- **Pionono offers a cleaner protocol-as-rater architecture than CMs.** Each protocol is a latent code, with a "gold" code trained on gold labels. On real data it beat both CM variants (κ 0.758 vs 0.692–0.721). It needs a gold subset; ours would be the expert normal/abnormal subset. It still has no object-level omission mechanism.

### Gaps
- Tanno 2019's own theorem (the population-level trace result) was not read in full. The formulas come from Zhang 2020's restatement.
- Nguyen et al. NeurIPS 2024 was not read: OpenReview showed a bot-check. Its "single annotator is insufficient" claim is from a search snippet only.
- No paper was found in the CM family, or in its 2023–26 successors, that models whole-instance omission, or that uses a one-sided (omission-only) CM in segmentation.
- No source was found for a proof that SAR propensity is identifiable from one labelling. Kato, Teshima & Honda (PU with selection bias, ICLR 2019, from memory) was not checked.

---

## 2. Label-style / protocol conditioning and multi-dataset learning with inconsistent label definitions: can one protocol omit abnormal objects while another includes them?

### Takeaway
Two mechanisms transfer directly:
- **Shi et al.'s marginal loss** treats a dataset's "background" as the union of true background and every class that dataset does not label. That is exactly our situation, with abnormal fibers merged into background.
- **MultiTalent's per-dataset sigmoid heads, with the loss only on annotated heads**, give each protocol its own output channel, so overlapping, nested definitions can coexist.

Style conditioning (Zepf, Parikh, Pionono codes, prompts) gives the user-chosen output protocol. But it learns *dataset* style as well as *protocol* unless domain is separated from protocol.

### Cited Findings
**Shi, Xiao, Chen, Zhou, "Marginal loss and exclusion loss for partially supervised multi-organ segmentation", MedIA 70:101979 (2021)** — [arXiv 2007.03868](https://arxiv.org/pdf/2007.03868); [researchr](https://researchr.org/publication/ShiX0Z21/bibtex). Read as abstract and search results only.
- **Marginal loss:** the background label in a partially labelled image is a merged label covering "all unlabelled organs plus true background", so its probability is a **marginal**, p(bg_obs) = p(bg) + Σ_{c unlabelled} p(c). This plugs into CE or Dice — [arXiv 2007.03868](https://arxiv.org/pdf/2007.03868).
- **Exclusion loss:** uses the fact that organs don't overlap to penalise disagreement between labelled organs and the predicted unlabelled organs. Per the ablation summary, it works best as an auxiliary term — [search summary of arXiv 2007.03868](https://arxiv.org/pdf/2007.03868).
- Evaluated on 5 datasets (liver, spleen, kidneys, pancreas); the abstract says it beats the state of the art at no extra cost — [arXiv](https://arxiv.org/pdf/2007.03868).

**MultiTalent: Ulrich, Isensee, Wald, Zenk, Baumgartner, Maier-Hein, MICCAI 2023** — [arXiv 2303.14444v2 HTML](https://arxiv.org/html/2303.14444v2).
- **Conflicting definitions as separate classes:** "C^(k) ∩ C^(j) = ∅, ∀k≠j". The same organ from different datasets is a different class, because, for example, dataset 11's heart and aorta overlap dataset 10's heart.
- **Output:** one sigmoid head per class on a shared backbone (ŷ_c = f(x, Θ, Θ_c)), "applying a Sigmoid activation function instead of the commonly used Softmax".
- **Loss:** L = Σ_c [ 1_c^(k)·(1/I)Σ_z BCE(ŷ_{z,c}, y_{z,c}) − 2Σ_z 1_c^(k) ŷ y / (Σ_z 1_c^(k) ŷ + Σ_z 1_c^(k) y) ], summed rather than averaged over classes. Classes not annotated in dataset k are left out, not treated as background.
- **Sampling:** ∝ 1/√n per source.
- **Results:** 13 CT datasets and 47 classes. +1.26 Dice (ResEnc U-Net) and +1.05 (U-Net) over dataset-specific models; BTCV 88.82 (single model). Trained in the nnU-Net framework.
- Code: MIC-DKFZ/MultiTalent — [search summary](https://arxiv.org/pdf/2303.14444v1).

**Universal models and protocol conflicts.**
- The CLIP-Driven Universal Model (Liu et al., ICCV 2023; arXiv 2301.00785) flags inconsistent protocols. For example, the AMOS aorta is the whole aorta, but in AbdomenCT-1K "a part of the upper regions annotation is missing". These were reconciled by **manual relabelling** — [arXiv 2301.00785](https://arxiv.org/pdf/2301.00785) (snippet).
- Medical SAM3 (arXiv 2601.10880, preprint) maps each dataset's labels to a canonical vocabulary — [arXiv](https://arxiv.org/html/2601.10880v1) (snippet).
- "One Model to Rule them All" (arXiv 2312.17183) manually assigns unambiguous terms and revises contradictory annotations — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.17183) (snippet).
- In short, universal models mostly *remove* protocol differences by curation rather than modelling them.

**Style conditioning (extending the known Zepf and Parikh work).**
- Zepf et al.: "an updated modelling objective conditioning on labeling style for aleatoric uncertainty estimation", applied to two existing uncertainty architectures. Two datasets with different label styles were curated — [arXiv 2303.15850](https://arxiv.org/abs/2303.15850) (abstract only).
- Parikh et al. use FiLM subgroup conditioning to output a reference style; their bias is boundary-only and simulated (see the Overlooked gaps report).

**Domain versus annotator confound, 2026.**
- "Harmonized Feature Conditioning and Frequency-Prompt Personalization for Multi-Rater Medical Segmentation" (arXiv 2605.08210, preprint) separates **scanner-induced artifacts from annotator variability**. It injects rater-specific information into a latent code "without altering the shared anatomical backbone" — [arXiv](https://arxiv.org/html/2605.08210) (snippet only).

### Inferences
- **Marginal loss as the C3 "protocol" likelihood.**
  - Give the network an explicit output class **abnormal fiber** (axon_abn / myelin_abn), plus normal axon and myelin.
  - On our labels, where background = {true bg ∪ abnormal fibers ∪ random misses}, train with p(bg_obs) = p(bg) + p(abn_axon) + p(abn_myelin).
  - On the expert subset, abnormal fibers are labelled explicitly (ordinary CE).
  - Exclusion loss: pixels labelled as a normal fiber cannot be abnormal.
  - This is Shi et al. applied verbatim with "abnormal" as the "unlabelled organ". The expert subset is what makes the abnormal class learnable, as Shi needs every organ labelled in *some* dataset.
  - Random misses are not covered by this. They need the selection term in §1 or the nnU-Net ignore mechanism.
  - Unlike the Zhang CM, this is softmax-coherent and has no trace degeneracy, because abnormal fibers are a supervised class elsewhere.
- **MultiTalent heads as protocol outputs.**
  - Sigmoid heads, one per protocol: `fiber@ours`, `fiber@TEM1`, `fiber@TEM2`, `fiber@AxonPath`, each trained only on its own dataset. Each head then *is* the protocol's selection.
  - Nesting fits the sigmoid (non-exclusive) form: fiber@ours ⊂ fiber@TEM1 for thin myelin, while both exclude degenerating fibers.
  - Add a soft constraint p(fiber@ours) ≤ p(fiber@complete) (a hinge on the difference), which is the exclusion loss in reverse.
  - nnU-Net already supports region-based sigmoid training, so this is a small trainer change. That is inferred from MultiTalent being built on nnU-Net; MultiTalent's own code was not checked.
- **Protocol identity is confounded with lab, microscope, species and preparation.** A protocol token or head learns domain as well as selection.
  - The only fully clean protocol contrast available to us is **within our own images**: our labels vs the expert's complete re-annotation of the same tiles.
  - The external datasets give protocol contrast only where domain shift is controlled. Harmonizer-style separation (arXiv 2605.08210) is the nearest recipe, but it was read as a snippet only.
- **A user-chosen protocol at inference is cheap once the model has a protocol head or code.** But "include abnormal fibers" is only as good as the abnormal supervision: no external set labels degenerating or broken-ring fibers (Overlooked gaps report §2).

### Gaps
- Shi et al.'s exact loss equations and numbers were not read; only the abstract and summaries.
- MultiTalent does not say how dataset-specific outputs are chosen at inference (the page does not say); its code was not read.
- No paper found applies marginal or exclusion losses to *instance* omission, as opposed to a whole unlabelled class.
- Zepf's architecture and results were not re-read (abstract only); the Overlooked gaps report has the earlier summary.
- DoDNet (Zhang et al., CVPR 2021, task-encoded dynamic heads) was not fetched in this pass. MultiTalent reports DoDNet (pretrained) at 86.44 on BTCV versus its own 88.82 — [arXiv](https://arxiv.org/html/2303.14444v2).

---

## 3. 2025–2026 work on annotator modelling, label bias and protocol-aware segmentation

### Takeaway
The 2025–26 multi-rater segmentation line (DiffOSeg MICCAI 2025, P-Diverse WACV 2026, ProSona, Harmonizer, JAPC MICCAI 2026) has moved from confusion matrices to **prompt or latent-code personalisation** of boundary style. None found models instance omission. Omission appears in 2025–26 work only in **benchmarks and evaluation of incomplete detection labels**, not in learning methods.

### Cited Findings
All of the following were read as search snippets only, unless stated otherwise.
- **DiffOSeg (MICCAI 2025):** two stages, "probabilistic aggregation" of multi-rater agreement, then "adaptive preference-prompting" for per-expert output. LIDC-IDRI and NPC-170 — [MICCAI 2025 PDF](https://papers.miccai.org/miccai-2025/paper/5223_paper.pdf).
- **P-Diverse (WACV 2026):** personalised plus diversified multi-rater segmentation from **one annotation per scan**; stage II synthesises multi-rater annotations — [CVF](https://openaccess.thecvf.com/content/WACV2026/papers/Almorsy_An_Efficient_Multi-Rater_Setup_Towards_Personalized_and_Diversified_Medical_Image_WACV_2026_paper.pdf).
- **ProSona (arXiv 2511.08046, preprint):** natural-language prompts control segmentation style in a shared latent space, with no annotator-specific branches — [arXiv](https://arxiv.org/html/2511.08046).
- **Harmonizer / frequency-prompt personalisation (arXiv 2605.08210, preprint):** separates scanner artifacts from annotator variability — [arXiv](https://arxiv.org/html/2605.08210).
- **arXiv 2512.00748 (preprint):** argues that personalisation methods "hard-code expert annotators", so they only replicate individuals rather than also generating diverse outputs — [arXiv](https://arxiv.org/pdf/2512.00748).
- **JAPC, "Attention-Based Prototype Calibration for Multi-Rater Few-Shot Medical Image Segmentation" (MICCAI 2026 per its repo):** rater-specific and consensus prototypes — [GitHub](https://github.com/truong2710-cyber/JAPC).
- **SimLabel (arXiv 2504.09525, preprint):** multi-annotator learning "with missing annotations". From the title, "missing" means annotator–item pairs that were not labelled, not omitted objects — [arXiv](https://arxiv.org/pdf/2504.09525) (title only).
- **Omission in benchmarks and evaluation:**
  - An anonymous NeurIPS 2025 submission, "Noisy Annotations in Segmentation", builds a stress-testing engine that perturbs masks *and omits instances* at three severity levels. It reports up to 35% mask-mAP drops (Mask R-CNN, Mask2Former, YOLACT, SAM). Unreviewed — [OpenReview PDF](https://openreview.net/pdf/c051fed24b2e704f9f33363b20ce12bb4c624b73.pdf).
  - "Object Detection Benchmarks are Incomplete" (arXiv 2609.21822, preprint): missing objects are the main label error, and validation mostly adds boxes — [arXiv](https://arxiv.org/html/2609.21822v1).
  - Noise-aware evaluation (WACV 2025) gives upper and lower bounds on a metric under missing objects — [CVF](https://openaccess.thecvf.com/content/WACV2025/papers/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.pdf).
- **Crowdsourced detection (ECCV 2024, just outside the window):** Bayesian Detector Combination infers annotator label quality "automatically" without prior skill levels — [arXiv 2407.07958](https://arxiv.org/abs/2407.07958) (abstract only). Code: github.com/zhiqin1998/bdc. Whether it models per-annotator *missed boxes* was not confirmed.
- The closest found description of a **per-annotator false-negative (omission) probability over latent true objects** is a US patent. It describes skill profiles with "a false-negative probability and a localization variance" and latent assignments between annotator and true boxes — [USPTO 11449788](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11449788) (snippet; a patent, not peer-reviewed).
- **Random ignoring of objects:** Fobi et al. (2020, Columbia) learn from "misaligned and partial labels", randomly picking objects to ignore so that unlabelled objects can still be segmented — [PDF](https://qsel.columbia.edu/assets/uploads/blog/2020/publications/learning-to-segment-from-misaligned-and-partial-labels.pdf) (snippet).

### Inferences
- **The lack of omission-aware annotator models in 2025–26 is itself a finding.** Object-level, appearance-dependent omission ("protocol selection") sits between three literatures that don't talk to each other:
  - the multi-rater line, which covers boundary style;
  - the PU/SAOD line, which covers *random* omission and is covered in the earlier novelty notes;
  - the dataset-curation and benchmark line, which *measures* omission but doesn't learn it.
- The C3 contribution can therefore be stated as: an instance-level selection model with anchors, using known components (Zhang CM / SAR-PU likelihood, Shi marginal loss, MultiTalent heads), applied to object omission.
- P-Diverse's "one annotation per scan" setting is closest to ours (one protocol per image). Its stage II synthesis is boundary-oriented according to the snippet; this is unverified.

### Gaps
- None of the 2025–26 multi-rater papers was read beyond snippets. Their losses and identifiability arguments are unknown here.
- No MedIA or TMI 2025–26 annotator-model paper surfaced in two searches. A venue-restricted search was not run.
- Fairness-under-label-bias work beyond Parikh 2026 was not found in this pass.

---

## 4. Which methods can exploit our side information (overlapping tiles, the expert normal/abnormal subset, near-complete external sets as a second protocol)?

### Takeaway
Each side-information source plays a known identifiability role:
- **Overlapping tiles** act as *repeated labels of the same object*, which separates random misses (stochastic) from protocol exclusion (deterministic).
- **The expert subset** gives *gold anchors* (Pionono's gold rater; Shi's "labelled somewhere" requirement).
- **External near-complete sets** act as a *second annotator* (the multi-annotator requirement of Zhang, CCEM and Nguyen). But this role is confounded with domain.

### Cited Findings
- Multi-annotator CM identifiability assumes several annotators with diagonally dominant CMs; the Zhang loss sums over r ∈ S(x) — [ar5iv 2007.15963](https://ar5iv.labs.arxiv.org/html/2007.15963).
- CCEM identifiability needs "near-class-specialist" annotators or sufficiently scattered CMs — [arXiv 2306.03288](https://arxiv.org/html/2306.03288).
- A single annotator is insufficient for instance-dependent outliers — [NeurIPS 2024 (snippet)](https://neurips.cc/virtual/2024/poster/95831).
- Pionono trains a separate "gold rater" code (r = M+1) on gold labels and uses it for consensus — [arXiv 2307.11397](https://arxiv.org/html/2307.11397v1).
- The marginal loss requires each merged class to be labelled in some dataset — [arXiv 2007.03868](https://arxiv.org/pdf/2007.03868) (inferred from the method summary).
- SAR propensities give unbiased training when known, and can be estimated when the selection mechanism is understood — [PMLR v94](https://proceedings.mlr.press/v94/bekker18a.html).
- Local facts from `docs/reports/Overlooked gaps validation.md`:
  - the overlapping tiles hold only **129 fibers**, and it is unknown whether both tiles had the same annotator;
  - TEM1 and AimSeg also **exclude degenerating or broken rings**;
  - the external sets label thin-myelin fibers our protocol skips.

### Inferences
- **Overlapping tiles split random misses from protocol exclusion (the Dawid–Skene logic of repeated labels).**
  - A fiber labelled in one tile and unlabelled in the other is (almost surely) a **random miss**, so s_k < 1 for reasons other than appearance.
  - A fiber unlabelled in *both* tiles, if it looks normal, is either a consistent exclusion or two misses.
  - Under independence, P(missed twice) ≈ (1 − r)², where r is the random-miss recall. That gives a direct estimate of r and so of the random-omission component.
  - Use this to set or fix the random part of the selection model: s_k = r · π(abnormal_k), so that π carries the protocol.
  - Caveat: with 129 fibers and possibly the same annotator, the independence assumption is weak.
- **The expert subset gives anchors for both t and π.**
  - Expert "normal, unlabelled" fibers are t = 1 with a random miss.
  - Expert "abnormal" fibers are t = 1 with protocol exclusion.
  - Expert "not a fiber" items are t = 0.
  - This breaks the t-vs-s confound in §1 directly. It is the C3 equivalent of Pionono's gold rater and of CCEM's anchor points.
  - It also gives the known-positive evaluation set already planned for C3.
- **External near-complete sets act as a second annotator with s ≈ 1 for normal and thin-myelin fibers**, but s ≈ 0 for degenerating fibers, like ours.
  - They pin down t for thin-myelin appearance only.
  - The protocol and domain confound means they should feed a *separate* head (MultiTalent style) or a separate code (Pionono style), not a shared one.
  - Never use them as evidence that our protocol excludes thin myelin; that comes from the expert subset.
- **Best fit overall:** a hybrid.
  1. A semantic network for latent truth, with an explicit abnormal class (Shi marginal loss on our labels; the expert subset gives the abnormal class).
  2. An instance-level selection term for random misses, calibrated from the overlapping tiles.
  3. Per-protocol sigmoid heads for the external sets.

  The Zhang trace term should *not* be used.

### Gaps
- No published method was found that uses overlapping or duplicated image regions as repeated annotations to estimate per-object omission. This agrees with the earlier report's "no source found" on overlap agreement.
- Whether the overlapping tiles had independent annotators is unknown and must be confirmed with the organisers. Without that, the (1 − r)² estimate is biased toward fewer misses.

---

## 5. Per-candidate summary: loss, identifiability, gains, code, and an adaptation sketch for nnU-Net semantic output + watershed pairing (or an instance head)

### Takeaway
Ranked by fit for C3:
1. Shi marginal/exclusion loss with an abnormal class;
2. MultiTalent per-protocol sigmoid heads;
3. an instance-level SAR-PU selection term (a one-sided Zhang CM at the object level);
4. Pionono protocol codes;
5. the pixel-wise Zhang CM with trace, which should be avoided as is.

None needs a new loss to be claimed. The novelty is the object-level selection, the anchors and the known-positive evaluation.

### Cited Findings (summary table; every entry is sourced in §§1–3)

| Candidate | Loss / architecture | Identifiability | Reported gain | Code | Read |
|---|---|---|---|---|---|
| Zhang 2020 pixel CM ([ar5iv](https://ar5iv.labs.arxiv.org/html/2007.15963)) | CE(Â^(r)(x)·p̂(x), ỹ^(r)) + λ tr(Â^(r)); shared U-Net, last-layer CM head | Several annotators; mean CM diagonally dominant; trace minimum; identity initialisation | MSLSC one label: 56.43 vs 40.32 Dice (global CM); simulated annotators | [GitHub](https://github.com/moucheng2017/Learn_Noisy_Labels_Medical_Images) | full |
| Pionono ([arXiv](https://arxiv.org/html/2307.11397v1)) | ELBO: log-lik (gen. Dice) − λ KL(N(μ_r,Σ_r) ‖ N(0,σI)); z concatenated with features → 1×1 head | Gold-rater code trained on gold labels; no theory | Gleason κ 0.758 vs 0.692 (pixel CM) | [GitHub](https://github.com/arneschmidt/pionono_segmentation) | methods |
| CCEM/GeoCrowdNet ([arXiv](https://arxiv.org/html/2306.03288)) | CE(A_m f(x)) − λ log det(FFᵀ) or (WᵀW); instance-invariant A | Class specialists / anchors, or SSC | MNIST 80.97 vs 65.72 (CrowdLayer) | not checked | partial |
| MultiTalent ([arXiv](https://arxiv.org/html/2303.14444v2)) | Per-class sigmoid heads; BCE + Dice only on the dataset's own classes | Each class supervised in its dataset; no theory | +1.05–1.26 Dice vs dataset-specific; BTCV 88.82 | MIC-DKFZ/MultiTalent | methods |
| Shi marginal + exclusion ([arXiv](https://arxiv.org/pdf/2007.03868)) | p(bg_obs) = p(bg) + Σ unlabelled p(c); exclusion penalty | Every merged class labelled in some dataset | State of the art on a 5-dataset union (abstract) | not checked | abstract |
| SAR-PU ([PMLR](https://proceedings.mlr.press/v94/bekker18a.html)) | Labelled positives selected with propensity e(x); EM | Needs known or understood selection; identifiability is the main challenge ([arXiv 2407.09735](https://arxiv.org/html/2407.09735v1)) | — | not checked | abstract |

### Inferences (adaptation sketches; these are designs, not results)
**A. Marginal loss + abnormal class (nnU-Net semantic, then existing pairing).**
- Output classes: {bg, artifact(ignore), axon_n, myelin_n, axon_abn, myelin_abn}, softmax.
- Loss on our standard labels:
  - On labelled fiber pixels, CE on axon_n or myelin_n.
  - On background pixels, −log[p(bg) + p(axon_abn) + p(myelin_abn) + ρ·(p(axon_n) + p(myelin_n))].
  - ρ ∈ [0,1] softens random misses: ρ = 0 is pure Shi; ρ = 1 ignores fiber mass in background.
- On the expert subset: plain CE with the abnormal classes.
- Exclusion: penalise p(abn) on labelled normal-fiber pixels.
- Inference: the official map takes {axon_n, myelin_n}; the "complete" protocol takes the union. Run pairing on whichever map is chosen.
- The cost is a custom nnU-Net trainer; the class and label plumbing is the same as the region or ignore-label path.

**B. Per-protocol heads (MultiTalent).**
- Sigmoid regions: fiber@ours, fiber@TEM1, fiber@TEM2 (and axon/myelin per protocol), with the loss on a protocol's head only for that protocol's images.
- Add the hinge relu(p_ours − p_TEM1) on thin-myelin-like regions (an inferred design).
- The output protocol is chosen by the head.
- Main risk: domain confound (§2).

**C. Instance-level selection (one-sided CM / SAR-PU).**
- After pairing on the truth map, pool features over each candidate fiber k to get t_k (truth) and π_k (protocol keep).
- Object likelihood:
  - labelled fibers: log(t_k · r · π_k);
  - unlabelled candidates: log(1 − t_k · r · π_k).
- r (random recall) is fixed from the overlapping tiles.
- t and π are both anchored on the expert subset.
- Add a sparsity prior on (1 − π), never a trace-style prior.
- This drops into an instance head (e.g. a Mask R-CNN- or StarDist-style head) more naturally than into nnU-Net.
- With nnU-Net, π_k can be a small classifier on pooled decoder features, trained after the semantic net (a two-stage approach).

**D. Pionono codes.**
- One latent code per protocol {ours, TEM1, TEM2, expert-complete}, concatenated before nnU-Net's final 1×1 convolution.
- The expert-complete code acts as the gold rater.
- Simple, but it learns style at pixel level; there is no object coherence.

**E. Zhang pixel CM.**
- Only as a baseline ablation, to show that the trace regulariser fails under one-sided omission. That failure is the analytic prediction in §1 and should be tested on the TEM1 controlled benchmark before it is claimed.

### Gaps
- No reported numbers exist for any of these methods under **object omission**; every gain above is for boundary or class noise, or for missing classes.
- The code for Shi, CCEM and SAR-PU was not checked.
- How much a custom nnU-Net trainer for options A and B would cost was not assessed here. It needs local inspection of `external/nnUNet` (region-based training, ignore label).
