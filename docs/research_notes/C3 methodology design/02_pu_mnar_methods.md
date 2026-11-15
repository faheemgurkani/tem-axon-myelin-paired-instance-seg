# PU learning and MNAR / selection-bias label models as a training objective for C3 (15 Nov 2026)

Scope: methods-level map of positive-unlabelled (PU) and missing-not-at-random (MNAR) label models, with emphasis on
feature-dependent labelling (SAR, not SCAR), and on dense-prediction use. This extends
`docs/research_notes/C3 incomplete label learning novelty/00_papers_read.md` (nnU-Net ignore label, LNQ 2023, Zhao et al.
PU cell detection, SAOD, Komura, Vădineanu, BiasCorr). Those papers are not repeated here.

Reading depth (honest accounting, about 20 tool calls):
- **Full text or methods read:** Bekker & Davis SAR-EM (2018 workshop version, arXiv 1808.08755); Coudray et al. JMLR 2023
  (PDF text, sections 3–4); PUe (Wang et al., NeurIPS 2023; arXiv HTML v1 of July 2026 plus NeurIPS PDF title page);
  Hu et al. ICLR 2022 (CADR, HTML).
- **Abstract or snippets only:** Kato et al. ICLR 2019; Gerych et al. AAAI 2022 (AAAI site blocked by bot protection;
  details from abstract and from PUe's related-work paragraph); Gong et al. TPAMI 2021; Furmańczyk et al.; Teisseyre et
  al. SCAR test; Sportisse et al. ICML 2023; Dist-PU; PULSNAR; 2025 survey and 2025–26 items.
- **Not read this run:** HolisticPU, Robust-PU, PULNS, TIcE, KM, nnPU original. They are listed as origins only.

Notation used throughout: y = true class (1 = real fiber), s = labelled indicator, η(x) = P(y=1|x), e(x) = P(s=1|y=1,x)
(propensity, defined on positives only), η̃(x) = P(s=1|x), π = class prior.

## 1. Core theory and estimators: SCAR vs SAR, propensity-weighted risk, instance-dependent PU

### Takeaway
Under SAR, an unbiased risk exists as soon as e(x) is **known on the labelled instances** (Bekker et al. 2020; analysed by
Coudray et al., JMLR 2023). Everything hard is in **identifying e(x)**. Every method buys identifiability with one of four
extra assumptions: (i) e depends only on a known subset of features; (ii) a parametric form or a link to η (e = k·η, or e
increasing in η, or "invariance of order"); (iii) local certainty (no class overlap somewhere); or (iv) side information
that gives e directly. Our setting violates (ii) for abnormal fibers, so C3 needs (iv).

### Cited Findings

**Definitions and the basic identity**
- SCAR: P(s=1|x,y=1) = c, a constant. SAR: selection probability for positives is determined by the attributes;
  e(x) = Pr(s=1|y=1,x); negatives are never selected — [Bekker & Davis, arXiv 1808.08755 (Workshop on Learning with Imbalanced Domains, 2018)](https://arxiv.org/html/1808.08755).
  The journal-strength version is Bekker, Robberechts & Davis, "Beyond the Selected Completely at Random Assumption for
  Learning from PU Data", ECML-PKDD 2019 — [mlanthology](https://mlanthology.org/ecmlpkdd/2019/bekker2019ecmlpkdd-beyond) (that version not read).
- Labelled-vs-unlabelled ("non-traditional") classifier targets η̃(x) = e(x)·η(x), not η(x) — [Coudray et al., JMLR 24 (2023), Eq. 4](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf).
- When the naive classifier is already Bayes-optimal: Cannings et al. (2020) show g̃* = g* if e(x) ≥ 1/(2η(x)) for all x
  with η(x) ≥ 1/2 (cited in [Coudray et al., JMLR 2023, Eq. 5](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf)).
- Origin of SCAR-PU: Elkan & Noto, KDD 2008 (label frequency c; P(y=1|x) = P(s=1|x)/c) — [PDF](https://cseweb.ucsd.edu/~elkan/posonly.pdf).

**Risk estimators (losses)**
- PN risk: R_PN(g) = π R_P(g,+1) + (1−π) R_N(g,−1). Unbiased PU (uPU, du Plessis et al. 2014/2015):
  R_uPU(g) = π R_P(g,+1) + R_U(g,−1) − π R_P(g,−1) — as restated in [PUe, NeurIPS 2023 / arXiv 2607.13428](https://arxiv.org/html/2607.13428v1).
  nnPU (Kiryo et al., NeurIPS 2017, [arXiv 1703.00593](https://arxiv.org/abs/1703.00593); not re-read) clips the negative-risk
  term at zero: π R_P(g,+1) + max(0, R_U(g,−1) − π R_P(g,−1)). Both assume SCAR: [PUe](https://arxiv.org/html/2607.13428v1)
  states that uPU, nnPU, Self-PU, ImbPU, Dist-PU and PUSB lose unbiasedness/consistency when labelled data are not uniformly sampled.
- **SAR propensity-weighted risk** (Bekker et al. 2020), 0-1 form: r_SAR(g,(X,S)) = (1{S=1}/e(X))·(1{g(X)≠1} − 1{g(X)≠0}) + 1{g(X)≠0};
  empirical risk = mean over all n samples; it is unbiased for the true risk R(g); under SCAR it reduces to the
  du Plessis estimator with e(x)=e_m. **Only e on the labelled instances is needed** — [Coudray et al., JMLR 2023, §3.3, Eq. 10](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf).
  In words: each labelled positive counts as a positive with weight 1/e and as a negative with weight −(1/e − 1); every
  example (labelled or not) also contributes a negative-class term.
- **Risk bound:** with e(x) ≥ e_m > 0, excess risk scales like 1/(n·e_m) (fast regime) or sqrt(V/(n·e_m)); a matching
  minimax lower bound holds under SCAR. Low propensity anywhere costs samples like losing labels; "n·e_m" plays the role
  of an effective sample size — [Coudray et al., JMLR 2023, Thm 1–2](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf).
  The authors note that with an estimated propensity "the theoretical analysis would have to account for the difference
  between the estimated propensity and the true one" (not done) — [same](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf).
- **PUe (normalised IPW):** ω_i = (1/e(x_i)) / Σ_j (1/e(x_j)) over labelled positives; R^e_P(g,±1) = Σ_i ω_i L(g(x_i),±1);
  R_PUe(g) = π R^e_P(g,+1) + R_U(g,−1) − π R^e_P(g,−1); the same plug-in works for nnPU, PUbN, Dist-PU. Under SAR,
  Σ_j 1/e(x_j^P) = N_P (all positives), which motivates normalisation — [Wang, Chen, Guo, Wang, NeurIPS 2023 (arXiv 2607.13428 posted Jul 2026)](https://arxiv.org/html/2607.13428v1);
  venue confirmed from the [NeurIPS 2023 proceedings PDF](https://papers.neurips.cc/paper_files/paper/2023/file/3efb4bdc6bfe13e1ff95b4407c37961d-Paper-Conference.pdf).

**How e(x) is parameterised and learned, and what identifies it**
- **SAR-EM (Bekker & Davis):** latent y; classifier θ and propensity model φ (logistic regression on a *propensity
  attribute subset* x_e) learned jointly. Init with a SCAR model; E-step computes P(y=1|unlabelled x) after multiplying
  propensities by a decay d = 0.9 (to push toward local certainty); M-step refits classifier with weighted unlabelled
  examples and refits propensity on x_e with posterior weights. The authors call SAR learning with unknown propensity
  "ill-defined" because low η and low e explain an unlabelled example equally well; remedies are (1) e depends only on a
  known x_e, (2) local certainty within each propensity stratum, (3) smoothness. Pure EM can collapse to e ≡ 1 with the
  classifier predicting the label — [arXiv 1808.08755](https://arxiv.org/html/1808.08755).
  Reported: on 20 Newsgroups SAR simulations SAR-EM with unknown propensity was "nearly as good" as with true propensity;
  multi-SCAR F1 ≈ 0.25–0.43 vs SAR-EM ≈ 0.6–0.8; SCAR estimators over-estimate label frequency when SCAR fails — [same](https://arxiv.org/html/1808.08755).
- **Kato, Teshima & Honda, ICLR 2019 ("PUSB"):** assumes "invariance of order" — the labelled-positive score preserves the
  order of the class posterior; learns an order-preserving score and then a threshold, i.e. identifies the classifier only
  up to a monotone transform — [ICLR 2019 (mlanthology)](https://mlanthology.org/iclr/2019/kato2019iclr-learning) (abstract-level).
  Hammoudeh & Lowd report brittleness of PUSB on a bias condition it targets — [search snippet of arXiv 2002.10261](https://arxiv.org/pdf/2002.10261) (not read).
- **He et al. 2018:** e(x) increasing in η(x) — as summarised in [Coudray et al. §3.3](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf).
- **Gerych et al., AAAI 2022:** first inference strategy with *provable* identifiability. Two regimes: **Local Certainty**
  (no functional form for e, but no positive/negative overlap: p(x_P|y=0)=0) and **Probabilistic Gap** (e = k·P(y=1|x),
  i.e. positives that resemble negatives are less likely to be labelled; allows class overlap) —
  [AAAI 36(6) abstract](https://ojs.aaai.org/index.php/AAAI/article/view/20624); formal statement as restated in
  [PUe NeurIPS 2023, related work](https://papers.neurips.cc/paper_files/paper/2023/file/3efb4bdc6bfe13e1ff95b4407c37961d-Paper-Conference.pdf).
  PUe calls these assumptions "too strict" — [PUe](https://arxiv.org/html/2607.13428v1). (Full text blocked by bot protection; theorem not read.)
- **Gong et al., TPAMI 2021 ("Instance-dependent PU learning with labeling bias estimation"):** graphical model with latent
  y; labelling probability depends on x; fitted by EM with Adam — [Semantic Scholar](https://www.semanticscholar.org/paper/Instance-Dependent-Positive-and-Unlabeled-Learning-Gong-Wang/c2de0bc437a93f16ec4529d73a76b1693dc717ba), DOI 10.1109/TPAMI.2021.3061456 (abstract only).
  Coudray et al. classify Bekker 2020 and Gong 2021 as "parametric model on the propensity" approaches — [JMLR 2023 §3.3](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf).
- **Double logistic model (Furmańczyk, Mielniczuk, Rejchel, Teisseyre, arXiv 2209.07787, preprint):** η and e both logistic
  with different parameters, estimated jointly; a related "TM" joint posterior/propensity method appeared at ECAI 2023 —
  [arXiv 2209.07787](https://arxiv.org/abs/2209.07787) and search snippet (not read).
- **PUe propensity estimator (deep):** min over e of π₁/n_P Σ L(e(x^P),+1) + (1−π₁)/n_U Σ L(e(x^U),−1) + α_e |Σ e(x) − n_P|,
  π₁ = n_P/(n_P+n_U) — i.e. a labelled-vs-unlabelled classifier with a sum constraint. No identifiability theorem; the
  regulariser only discourages degenerate solutions (labelled → 1, unlabelled → 0). π is assumed known — [PUe](https://arxiv.org/html/2607.13428v1).
  PUe attributes underestimation of propensity (co-directional bias, degenerate estimates) to SAR-EM — [same](https://arxiv.org/html/2607.13428v1).
- **PULSNAR (Kumar & Lambert):** for "selected not at random" positives, clusters labelled positives into subtypes and
  estimates the class proportion per subtype with a SCAR estimator (PULSCAR) against all unlabelled data —
  [PMC11622864](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11622864/), [arXiv 2303.08269](https://arxiv.org/pdf/2303.08269) (abstract-level; journal venue not verified).
- **Testing SCAR:** Teisseyre, Furmańczyk & Mielniczuk propose a test that generates artificial SCAR labels to approximate the
  null distribution; recommended as a pre-processing step — [arXiv 2404.00145, preprint](https://arxiv.org/abs/2404.00145) (abstract only).
- **Survey:** Gong et al., "A recent survey on instance-dependent positive and unlabeled learning", *Fundamental Research* 2025 —
  [PubMed 40242552](https://pubmed.ncbi.nlm.nih.gov/40242552/) (page blocked; not read).

**Reported gains and failure modes of SAR corrections**
- PUe accuracy gains (Table 1): MNIST nnPU 87.81 → 92.45, Dist-PU 89.79 → 92.57; CIFAR-10 nnPU 82.47 → 83.33, Dist-PU 84.45 → 86.97;
  Alzheimer's (real bias, unknown propensity, ResNet-50) nnPU 69.19 → 69.38, Dist-PU 69.74 → 70.51 — [PUe](https://arxiv.org/html/2607.13428v1).
  Failure modes reported: propensities near 0 inflate bias; non-monotone sensitivity to α_e; gains shrink under extreme
  imbalance; no gain under uniform labelling — [same](https://arxiv.org/html/2607.13428v1).
- SAR-EM needs calibrated classifiers (logistic regression chosen for calibration; "more powerful classifiers may need
  calibration"); complex propensity functions left to future work — [arXiv 1808.08755](https://arxiv.org/html/1808.08755).

**SCAR-era building blocks (origins only; not read this run)**
- Class-prior estimation: TIcE (Bekker & Davis, AAAI 2018) and KM (Ramaswamy et al., ICML 2016) are the standard SCAR
  estimators referenced in [Coudray et al.](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf) (cites Ramaswamy 2016, Bekker & Davis 2018).
- Dist-PU (Zhao et al., CVPR 2022, pp. 14461–14470): aligns the expected predicted label distribution on unlabelled data
  with the known prior, plus entropy minimisation and Mixup; motivated by negative-prediction preference of cost-sensitive
  PU — [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Dist-PU_Positive-Unlabeled_Learning_From_a_Label_Distribution_Perspective_CVPR_2022_paper.html) (abstract-level).
- Robust-PU (Zhu et al. 2023): reweights samples by training progress and hardness; HolisticPU (Wang et al. 2023): treats
  prediction scores as a temporal point process — one-line descriptions only, from a 2025 benchmark paper
  [arXiv 2509.24228, preprint](https://arxiv.org/pdf/2509.24228). Venues not verified.

### Inferences
- **Our labelling mechanism has two components with very different identifiability.** (1) Random misses of normal fibers
  (≈4–8%): e ≈ 0.92–0.96, roughly constant → close to SCAR within the "normal" stratum. (2) Deliberate skips of abnormal
  fibers: e ≈ 0. Under the SAR risk, any region with e → 0 gives weights 1/e → ∞ and Coudray's bound (∝ 1/e_m) blows up;
  such fibers are **not recoverable from labels alone** (positivity violation), not merely biased.
- **Kato's invariance of order and Gerych's Probabilistic Gap are both violated** by component (2): a degenerating or
  redundant-myelin fiber is visually unambiguous as a fiber (high η) but has e ≈ 0. Probabilistic Gap (e = k·η) fits only
  the "thin, ambiguous myelin" subset. So C3 should not adopt an estimator whose identifiability rests on e being a
  function of η.
- **Cannings' condition explains why a naive model can still look good:** for normal fibers with η ≈ 1 and e ≈ 0.94,
  e ≥ 1/(2η) holds, so the decision for clearly normal fibers is unaffected by the 4–8% random misses; harm concentrates
  where η is moderate (thin, oblique, low-contrast fibers), and in calibration of the foreground probability that later
  drives pairing. This is consistent with the novelty report's "modest gains" expectation.
- **The protocol output is η̃ = e·η by definition.** A factorised model (head A: η = "is a real fiber"; head B: e = "would
  the annotator label it") trained on s with BCE on the product is the SAR likelihood that SAR-EM/Gong/double-logistic
  maximise. Head A gives morphometry on all fibers; the product gives the challenge (protocol-following) output. This
  needs side information to separate the two heads (Section 4); without it, the product is fine but the split is not identifiable.
- **Use the normalised (PUe) weights, not raw 1/e,** if IPW is used on labelled fiber pixels: raw 1/e near 0.9–1 is harmless,
  but any small ê (abnormal-looking but labelled fibers) would dominate the loss.

### Gaps
- Gerych AAAI 2022 theorem and estimator details (site blocked); Gong TPAMI 2021 loss; Kato ICLR 2019 full derivation.
- PULNS (IJCAI 2021), HolisticPU, Robust-PU, TIcE, KM not read; their venues and losses are not verified here.
- No 2025–26 top-venue (NeurIPS/ICML/ICLR) SAR-PU method was found by search; a "Confidence-based PU learning with
  instance-dependent label noise" paper is in IEEE TNNLS 2025 ([DOI](https://doi.org/10.1109/tnnls.2025.3549510); title only),
  and a 2026 covariate-shift PU preprint exists ([arXiv 2605.31187](https://arxiv.org/pdf/2605.31187); title only). A NeurIPS 2025
  poster on balancing error rates in PU was named in a search summary but not verified.

## 2. MNAR semi-supervised and missing-label models in deep learning

### Takeaway
Deep MNAR semi-supervised work (Hu et al. ICLR 2022; Sportisse et al. ICML 2023; Duan et al. ICCV 2023) models
missingness that depends on the **class** (label-informative missingness, X → Y → M), not on appearance within a class.
Its doubly robust (DR) estimator structure transfers to C3, but its propensity model does not: our missingness depends on
x *within* the positive class, which is exactly the PU-SAR setting of Section 1.

### Cited Findings
- **Hu, Niu, Miao, Hua, Zhang, "On Non-Random Missing Labels in Semi-Supervised Learning", ICLR 2022** —
  [OpenReview](https://openreview.net/forum?id=6yVvwR9H9Oj), [arXiv HTML](https://arxiv.org/html/2206.14923):
  - Causal chain X → Y → M; MNAR means P(Y|X, M=0) ≠ P(Y|X).
  - **CAP** (class-aware propensity): s(x,y) = log P(y|x;θ) / (log P(y|x;θ) − log P(y;θ)), with P̂(Y) tracked by an EMA over
    mini-batches; IPW supervised loss L_CAP = (1/N) Σ (1−m_i) L_s(x_i,y_i) / p_i.
  - **CAI** (class-aware imputation): pseudo-label threshold τ(x) = τ_o·(P̂(C_x)/max_y P̂(y))^β, lower for rare classes.
  - **CADR** = L_CAP + L_CAI + L_supp, with L_supp = (1/N) Σ (1 − m_i − (1−m_i)/p_i) L_u·1(conf>τ) − (1/N) Σ (1−m_i) L_s;
    claimed unbiased if either CAP or CAI is correct (DR property cited from Seaman & Vansteelandt, not proved there).
  - CIFAR-10 MNAR accuracy (γ = labelled imbalance 20/50/100): FixMatch 56.26/65.61/72.28 → CADR 79.63/93.79/93.97.
  - Limits: propensity depends only on class; MNAR simulated by imbalancing CIFAR; classification only; sensitive to β
    (β = 0.2 gave 65.79).
- **Sportisse, Schmutz, Humbert, Bouveyron, Mattei, "Are labels informative in semi-supervised learning? Estimating and
  leveraging the missing-data mechanism", ICML 2023 (oral), PMLR 202:32521–32539** — estimates the missing-data mechanism,
  debiases any SSL algorithm by IPW, and proposes a likelihood-ratio test of whether labels are informative; includes two
  medical datasets with pseudo-realistic missingness — [PMLR](https://proceedings.mlr.press/v202/sportisse23a.html),
  [ICML](https://icml.cc/virtual/2023/oral/25445) (abstract only).
- **Duan et al., "Towards Semi-supervised Learning with Non-random Missing Labels", ICCV 2023** — reports outperforming CADR
  under its own protocol — [CVF PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Duan_Towards_Semi-supervised_Learning_with_Non-random_Missing_Labels_ICCV_2023_paper.pdf) (title/snippet only).
- **MNAR PU in recommendation:** a two-phase propensity-based debiasing framework for PU implicit feedback addressing missing
  exposure and unmeasured confounding, *Entropy* 2025 — [PMC12839574](https://pmc.ncbi.nlm.nih.gov/articles/PMC12839574/) (snippet only).
- Statistics side: DR semi-supervised inference under MAR labelling with decaying overlap (Zhang, Chakraborty, Bradic,
  *Information and Inference* 2023) — [arXiv 2305.12789](https://arxiv.org/pdf/2305.12789) (snippet only).

### Inferences
- The **DR template** (IPW term + imputation term + correction term) maps onto C3 as: IPW on labelled fiber instances with
  ê(x); imputation = soft pseudo-labels on unlabelled ring candidates from an out-of-fold model; correction term so that
  the loss stays unbiased if either ê or the imputer is right. This is a defensible methods contribution only if paired
  with the identification strategy of Section 4, since DR still needs one of the two models to be correct.
- Hu et al.'s and Sportisse et al.'s class-only propensity is the wrong model family for us (one positive class; missingness
  varies with morphology). Cite them for the DR/IPW-in-deep-SSL lineage and the "test whether labels are informative" idea,
  not as the estimator.
- Sportisse's likelihood-ratio test and Teisseyre's SCAR test suggest a cheap diagnostic for the paper: test SCAR for the
  overlap-tile known positives (does e vary with myelin thickness, eccentricity, g-ratio?).

### Gaps
- Sportisse ICML 2023 and Duan ICCV 2023 methods sections not read; whether either allows feature-dependent propensity is unverified.
- No deep MNAR paper was found where missingness depends on appearance within a class in images (other than the PU-SAR line).

## 3. PU / MNAR / propensity methods in segmentation, detection and instance segmentation (2023–2026)

### Takeaway
No 2025–26 CVPR/ICCV/ECCV/MICCAI/MedIA/TMI paper was found that applies a SAR/propensity model to segmentation or
detection. Dense PU work is (i) per-instance nnPU for detection with random omission (Yang BMVC 2020; Zhao MELBA 2023,
already covered) or (ii) per-pixel PU for remote-sensing/hyperspectral classification, all under SCAR with a fixed or
tuned class prior. Feature-dependent labelling propensity in dense prediction appears to be open.

### Cited Findings
- **Yang, Liang & Carin, "Object Detection as a Positive-Unlabeled Problem", BMVC 2020:** cross-entropy in detectors treats
  unlabelled regions as background (PN); they replace the RPN/ROI classification loss with a PU loss (per proposal /
  anchor), outperforming PN on PASCAL VOC and MS COCO "across a range of label missingness" (random removal) —
  [arXiv 2002.04672](https://arxiv.org/html/2002.04672) (abstract-level this run).
- **Per-pixel PU in remote sensing:** class-prior-free PU with a "Taylor variational loss" that down-weights the gradient of
  unlabelled data to curb overfitting, for hyperspectral imagery — [Zhao et al., arXiv 2308.15081](https://arxiv.org/pdf/2308.15081) (abstract-level; venue not verified).
  Perantoni et al. (arXiv 2510.06769, preprint; cites SPIE Remote Sensing 2023) use nnPU's non-negative risk within a
  pixel-level classifier trained from patch-level labels — [arXiv 2510.06769](https://arxiv.org/html/2510.06769v1) (snippet only).
  "Needles in the Landscape" (arXiv 2510.16814, preprint) frames archaeological-site segmentation as PU but explicitly avoids a
  PU objective, using iterative confident-positive propagation — [arXiv 2510.16814](https://arxiv.org/html/2510.16814v1) (snippet only).
- **Small objects / cells:** "Small Object Localization with 90% Annotation Reduction by Positive-Unlabeled Learning",
  *Micromachines* 16(12):1379, 2025 (MDPI; droplet single cells from partial point labels) —
  [MDPI](https://www.mdpi.com/2072-666x/16/12/1379) (title/snippet only).
- **Nuclei with incomplete annotation (non-PU):** recursive U-Net retraining — [PMC9161776](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9161776/) (title only).
- Searches for "positive-unlabeled semantic segmentation 2025", "PU detection/instance segmentation CVPR/ICCV 2025–26" and
  "PU cell segmentation MICCAI 2025" returned no matching main-conference paper (search summaries, e.g.
  [AnnofreeOD ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_AnnofreeOD_Detecting_All_Classes_at_Low_Frame_Rates_Without_Human_ICCV_2025_paper.pdf) is annotation-free 3D detection, not PU).

### Inferences
- **Formulation choice for C3: per-instance propensity, per-pixel loss.** Labelling in our data is an object-level event
  (a fiber's axon and ring are labelled together or not at all), so e should be a function of fiber-level features
  (myelin thickness, g-ratio, eccentricity, ring completeness, contrast) and broadcast as a weight to that fiber's pixels.
  Per-pixel SCAR PU (remote sensing style) would mis-model this, since pixels of one fiber share a single s.
- For unlabelled pixels, the SAR E-step posterior that an unlabelled candidate is real is
  P(y=1|s=0,x) = η(1−e)/(1−η e) (algebra from the definitions in Section 1; SAR-EM uses this kind of posterior weighting).
  Example: η = 0.99, e = 0.94 → ≈ 0.86, so a confident normal-looking unlabelled ring should be near-positive; with e → 0
  (abnormal) the posterior falls back to η, i.e. the label carries no information.
- The absence of a dense SAR paper strengthens the novelty claim recorded in the earlier report, but a reviewer will ask why
  object detection's PU line (Yang 2020, Zhao 2023) is not enough: the answer must be "their e is constant (SCAR); ours is
  demonstrably feature-dependent (test in Section 2) and partly zero by protocol".

### Gaps
- MICCAI 2025/2026, MedIA and TMI proceedings were not browsed directly; search coverage is partial.
- Methods sections of Yang BMVC 2020, the hyperspectral PU paper and the Micromachines paper were not read; their exact
  per-anchor vs per-pixel losses and priors are unverified here.

## 4. Identifying information available in our setting, and which estimators can use it

### Takeaway
SAR theory says the unbiased risk needs e only on labelled positives, and e(x) is identified by any data where y is known
independently of s. All three of our side-information sources provide this, in different ways: overlap tiles give
**repeated labelling draws on the same fiber** (a capture–recapture design), the expert set gives **y and the
normal/abnormal split on a subset** (a "validation sample"), and external near-complete datasets give an **η model** from
which e = η̃/η follows.

### Cited Findings
- Only e on labelled instances is needed for the unbiased SAR risk; when unknown, e "could be estimated using prior
  knowledge on the labeling mechanism (when available) or by defining a parametric model" — [Coudray et al., JMLR 2023 §3.3](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf).
- η̃(x) = e(x)·η(x) — [Coudray et al., Eq. 4](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf).
- SAR-EM's identification relies on e depending on a known attribute subset x_e — [Bekker & Davis](https://arxiv.org/html/1808.08755).
- PUe assumes π known and treats real-data propensity as unknown (Alzheimer's), with small gains there — [PUe](https://arxiv.org/html/2607.13428v1).
- Sportisse et al. estimate the missing-data mechanism and test informativeness — [PMLR 202](https://proceedings.mlr.press/v202/sportisse23a.html) (abstract).

### Inferences (design proposals for C3; none is from a cited paper)
- **(a) Overlap tiles as capture–recapture.** A fiber visible in two overlapping tiles has two labelling outcomes
  (s1, s2). Conditioning on "labelled at least once" removes the unknown y: under independence and a common e(x),
  P(s1≠s2 | s1+s2 ≥ 1, x) = 2(1−e)/(2−e). A zero-truncated binomial likelihood over all known-positive overlap fibers
  fits a parametric ê(x) (logistic in fiber features) **without any class prior**. Caveats: (i) fibers never labelled in
  either tile (most abnormal ones) are invisible, so ê is identified only where e is not ≈ 0; (ii) positively correlated
  misses (same annotator, same appearance) make discordant pairs rarer and bias ê upward; (iii) border-cut fibers are
  label 1 in one tile, so restrict to fibers fully inside both tiles.
- **(b) Expert normal/abnormal set as a validation sample.** It gives y for every fiber in a few images, so
  e(x) = P(s=1|y=1,x) is fitted by ordinary logistic regression of s on x among expert-confirmed fibers. This is the
  Coudray/Bekker "propensity known" case for the training risk, and it also yields an estimate of π per image (useful for
  nnPU-style terms). It is the only source that sees the e ≈ 0 abnormal stratum, so it is required for the morphometry
  ("all real fibers") head; the protocol head does not need it.
- **(c) External near-complete datasets.** Train η on them (after the overlap check required by
  `02-challenge-and-data.md` §5), train η̃ on our labels, and take ê = η̃/η. Fragile under domain shift and unstable where
  η is small; best used as a prior or sanity check for (a)/(b), not as the estimator.
- **Recommended C3 objective (candidate):** factorised heads η (real fiber) and e (labelling propensity) with
  (1) BCE on s for the product η̃ = η·e on all training fibers/pixels; (2) a zero-truncated-binomial loss for e on overlap
  known positives; (3) supervised BCE for e and η on the expert subset. Deploy η̃ for the challenge/protocol metrics and
  η for morphometry. Baselines: nnU-Net ignore label (hard ignore), LNQ soft weight, SCAR nnPU, PUe-weighted nnPU with ê
  from (a), and a random-ignore control (already planned in the novelty report).
- **Simplest defensible variant if budget is tight:** keep nnU-Net, set per-pixel weights from ê on labelled fibers
  (PUe-normalised) and the posterior P(y=1|s=0,x) on unlabelled candidates (soft ignore), with ê from (a). This needs no
  new architecture and one GPU run.

### Gaps
- No paper found that estimates PU propensity from repeated/overlapping annotations of the same object; the capture–recapture
  idea above is our inference and needs checking against the ecology/epidemiology literature (dual-system estimation) before claiming novelty.
- Whether annotator misses in overlapping tiles are independent is unknown; the overlap fibers' annotation order/annotator
  identity is not in the data.

## 5. Practical gains and pitfalls

### Takeaway
Reported gains from SAR corrections on real (not simulated) bias are small (PUe on Alzheimer's: +0.2 to +1.8 accuracy
points), larger only on simulated bias. The main pitfalls are degenerate propensity solutions, extreme weights from small
ê, class-prior misspecification, and calibration of flexible models.

### Cited Findings
- Small real-data gains: PUe on Alzheimer's +0.19 (nnPU) to +1.84 (uPU) accuracy; larger on MNIST/CIFAR with simulated bias — [PUe](https://arxiv.org/html/2607.13428v1).
- Degeneracy: EM collapses to e ≡ 1 without decay; SAR-EM can underestimate propensities (PUe's critique) — [Bekker & Davis](https://arxiv.org/html/1808.08755), [PUe](https://arxiv.org/html/2607.13428v1).
- Small propensity: risk bound scales with 1/e_m — [Coudray et al.](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf); ê near 0 inflates bias and gains vanish under extreme imbalance — [PUe](https://arxiv.org/html/2607.13428v1).
- Calibration: flexible classifiers need calibration in SAR-EM — [Bekker & Davis](https://arxiv.org/html/1808.08755).
- Class-prior misspecification: wrongly assuming SCAR biases the label-frequency estimate upward — [Bekker & Davis](https://arxiv.org/html/1808.08755);
  Zhao et al. (MELBA 2023) tune π on validation recall because precision is unreliable under incomplete labels (already in
  `00_papers_read.md`).
- Hyperparameter sensitivity: CADR β, PUe α_e non-monotone — [Hu et al.](https://arxiv.org/html/2206.14923), [PUe](https://arxiv.org/html/2607.13428v1).
- PUSB brittleness on its own bias condition — [Hammoudeh & Lowd snippet](https://arxiv.org/pdf/2002.10261).

### Inferences
- Expect gains concentrated on overlap-disagreement fibers and on thin/ambiguous-myelin fibers, and near zero on clean
  normal fibers (Cannings condition). Report on the known-positive set, as the novelty report already requires.
- Clip ê below (e.g. ≥ 0.2) and normalise weights; fit e with a low-capacity model (logistic on 5–10 fiber features)
  rather than a CNN head, unless (b) provides enough fibers, to avoid the degenerate "labelled → 1" solution.
- Since e is ≈ 0.94 for most normal fibers, a misspecified π matters less than in SCAR PU with tiny label frequency; the
  sensitive parameter is the e of the thin/abnormal-boundary stratum.

### Gaps
- No dense-prediction study reports SAR-PU gains, so there is no direct effect-size prior for C3.
- The 2025 PU benchmark (arXiv 2509.24228, "Accessible, Realistic, and Fair Evaluation of PU Learning Algorithms") was not
  read; it may contain relevant evaluation pitfalls (e.g. validation without negatives).
