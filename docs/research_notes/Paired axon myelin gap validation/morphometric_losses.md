# Prior art for differentiable morphometric-ratio (g-ratio) losses: novelty check (as of 12 Nov 2026)

Scope note: this was a time-boxed pass of about 20 tool calls. I read the primary text (full PDF, or the full-text page) for four sources: Meng et al. MICCAI 2022, Janjic et al. 2019, R-Super (arXiv 2608.27668) and Naidoo et al. MIDL 2022. The Kervadec shape-descriptor paper was read via its ar5iv HTML. Other items come from search snippets and are flagged. Local PDFs are in `docs/references/papers/`: `meng2022_miccai_vcdr.pdf`, `arxiv_2608.27668_rsuper_volume_loss.pdf` and `naidoo2022_midl_loss_lv_volume_ef.pdf`.

Planned method under test: for each GT fiber region, compute soft areas Â_axon = Σp_axon and Â_fiber = Σp_fiber, then penalize |sqrt(Â_axon/Â_fiber) − g_gt|, weighted toward small fibers.

## Q1. Does any work put a g-ratio, myelin-thickness or axon-diameter term into a training loss?

### Takeaway
I found none. Every axon/myelin deep-learning paper I checked trains with pixel losses (Dice, CE or unspecified) and computes g-ratio afterwards. This includes AxonDeepSeg, Janjic 2019, the 2026 optic-nerve TEM nnU-Net paper, AimSeg and gACSON. A g-ratio term inside the loss appears to be absent from the axon/myelin literature, though I cannot prove it is absent everywhere.

### Cited Findings
- **Dice-loss example.** A peripheral-nerve morphometry CNN chose Dice as its loss. It reports g-ratio 0.53±0.11 versus 0.53±0.10 manual (−0.5%), but axon diameter and myelin thickness differed by about 9.5%. A g-ratio can match on average while its components are wrong. — [Morphometric Analysis of Peripheral Myelinated Nerve Fibers through Deep Learning, PMC6420354](https://pmc.ncbi.nlm.nih.gov/articles/PMC6420354) (search snippet; not re-read)
- **2026 nnU-Net on murine optic-nerve TEM.** Inner-axon F1 was 0.763 and outer-fiber F1 0.697. G-ratio R² was 0.805, which the authors attribute to boundary errors on both contours. It uses a standard loss, and g-ratio is computed post hoc. — [Deep Learning–Driven TEM Analysis of Murine Optic Nerve Myelinated Axons (ScienceDirect, 2026)](https://www.sciencedirect.com/science/article/pii/S2666914526000795); [PMC13091109](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091109/) (snippet)
- **Post-hoc tools.** AxonDeepSeg, AimSeg and gACSON all derive myelin thickness or g-ratio after segmentation. gACSON defines g = d_e/(d_e+d_m). — [AxonDeepSeg](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5830647/); [AimSeg](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10691719/); [gACSON arXiv 2112.06476](https://arxiv.org/html/2112.06476v2) (snippets)
- **Nearest axon-specific loss.** arXiv 2509.03154 (2025) uses connectivity-aware losses for axon-initial-segment light-sheet data. Its Negative Centerline Loss penalizes uncovered label centerline. This targets continuity and length, not morphometry. — [arXiv 2509.03154](https://arxiv.org/html/2509.03154) (snippet)
- **GitHub code search.** `gh search code` for "g_ratio loss" and "gratio_loss" returned no myelin-related hits, only unrelated `logratio_loss` functions. — GitHub code search, run 12 Nov 2026
- **Measurement bias in g-ratio.** A 2025 commentary argues that the positive g-ratio vs axon-diameter slope comes largely from measurement and sampling artifacts. This matters for size-weighted g-ratio targets. — [G-Ratio Commentary—Why You've Been Doing It Wrong, PMC12140491](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12140491/) (snippet)
- **Earlier pass.** The previous pass in this repo reached the same conclusion: "I found no paper that directly optimizes a morphometric (g-ratio, area ratio, diameter) as a differentiable loss for ring objects." — `docs/research_notes/TEM axon myelin segmentation review/topology_losses_pretraining.md`

### Inferences
- Within the axon/myelin domain, a per-fiber g-ratio loss looks novel. Novelty in method, as distinct from application, is weaker because of the analogues in Q2.

### Gaps
- I did not search Google Scholar full text, ProQuest theses or bioRxiv directly (search-engine coverage only). A thesis or workshop paper could exist. Recommended check before submission: search Scholar for "g-ratio" "loss function" and "myelin" "differentiable".

## Q2. Close analogues in other domains

### Takeaway
The closest prior art is **Meng et al., MICCAI 2022**, together with its **TMI 2023** extension. They build a differentiable vertical cup-to-disc ratio from **soft areas (Σ of probability maps)** plus soft perimeters, and supervise it with MSE. That is a ratio-of-soft-areas biomarker loss, per image, for one nested pair (cup ⊂ disc). This is structurally the same as axon ⊂ fiber. The second key prior art is Kervadec's line of work: Constrained-CNN losses (size = Σp), bounding-box global constraints, and the MIDL 2021 Best Paper shape descriptors, which include **ratios of descriptors between classes**. The third is R-Super (MedIA 2026), whose soft volume is masked by a region mask, Σ t·o. The same "Σp inside a GT region" construction appears in the planned loss.

### Cited Findings
**Meng et al., MICCAI 2022 (LNCS, pp. 524–534, DOI 10.1007/978-3-031-16440-8_50).** "Shape-Aware Weakly/Semi-Supervised Optic Disc and Cup Segmentation with Regional/Marginal Consistency." [PDF (Liverpool repository)](https://livrepository.liverpool.ac.uk/3169382/1/MICCAI2022_Shape-Aware%20Weakly_Semi-Supervised%20Optic%20Disc%20and%20Cup%20Segmentation%20with%20Regional_Marginal%20Consistency.pdf); [dblp](https://dblp1.uni-trier.de/rec/conf/miccai/MengCZZGHPPMZ22.html)
- **Mechanism, in the paper's words.**
  - "we approximate Ao with the summed pixel value of PM, which can be regarded as the area of oval shape in pixel level."
  - The perimeter is the sum of a width-1 soft boundary from a spatial-gradient layer.
  - The long semi-axis a comes from (A, P) under an ellipse model (Eq. 7), and θ(vCDR) = (a_OC + e−6)/(a_OD + e−6).
  - "we apply a MSE loss (LvCDR) between the prediction and ground truth to fully-supervise the vCDR estimation and weakly-supervise the OD & OC segmentation."
- **Setting.** The loss is computed per image, on one disc/cup pair. It is used **weakly**: 38,421 UK Biobank images carry only vCDR labels. The overall loss is L = L_PM + L_mSDF + L_B + λ(L_Ru + L_Bu + L_vCDR), with a ramp-up weight λ.
- **Reported effect (Table 2, UK Biobank vCDR).**
  - Without L_vCDR ("w/ Both"): MAE 0.127, Corr 0.341.
  - With L_vCDR only: MAE 0.112, Corr 0.410.
  - Full model (semi): MAE 0.102, Corr 0.453.
  - OD/OC Dice changes are small (OC 86.5 → 86.7).
  - The text says the weakly supervised loss "can bring a large improvement of 82.2% MAE." That figure apparently compares against Label-only, MAE 0.631 → 0.112, so it mixes in the extra vCDR-labelled data.
- **Journal extension.** Meng et al., IEEE TMI 42(2):416–429, 2023, "Dual Consistency Enabled Weakly and Semi-Supervised Optic Disc and Cup Segmentation With Dual Adaptive Graph Convolutional Networks." It keeps a differentiable vCDR estimating layer. — [AMiner](https://www.aminer.cn/pub/63108da290e50fcafdcd3031/dual-consistency-enabled-weakly-and-semi-supervised-optic-disc-and-cup-segmentation) (snippet; not read)

**Kervadec et al., MIDL 2021 Best Paper, PMLR 143:354–368.** "Beyond pixel-wise supervision for segmentation: A few global shape descriptors might be surprisingly good!" [ar5iv 2105.00859](https://ar5iv.labs.arxiv.org/html/2105.00859); [PMLR](https://proceedings.mlr.press/v143/kervadec21a.html); code at github.com/hkervadec/shape_descriptors
- **Descriptors.**
  - "The volume of the predicted segmentation is simply a summation of the predicted probabilities" (the zeroth moment).
  - Centroid and average distance to centroid come from soft moments.
  - Length uses a relaxed Potts term.
  - **Ratios of descriptors between two classes are bounded** a ≤ R ≤ b.
- **Penalty and supervision.** Constraints are per class and per image. The penalty is an extended log-barrier on ±10% bounds around the GT descriptor, with the pixel loss set to zero.
- **Result.** On ACDC, descriptors only gave Dice 0.768 versus CE 0.876. On Promise12 the figures were 0.651 versus 0.871. Descriptors used **alone** therefore lose 0.11–0.22 Dice.
- **Related Kervadec work (not re-read this session).**
  - Constrained-CNN losses (MedIA 2019) use a quadratic penalty on soft size Σp. [arXiv 1805.04628](https://arxiv.org/abs/1805.04628)
  - "Bounding boxes for weakly supervised segmentation: global constraints get close to full supervision" (MIDL 2020). [arXiv 2004.06816](https://arxiv.org/abs/2004.06816)

**R-Super (Bassi et al., "Report Supervision", arXiv 2608.27668, 27 Aug 2026; stated as published in Medical Image Analysis 2026).**
- **Volume Loss.** "𝑉𝑠,𝑜 = 𝑣 Σ t_xyz · o_xyz … Since T_o contains tumor probabilities, Eq. 3 provides a soft estimate of tumor volume. Importantly, this estimate is differentiable." It is matched to a volume derived from the report diameters (ball or ellipsoid).
- **Granularity.** The loss is per organ or sub-segment, summed over all tumors. A separate **Ball Loss** enforces per-tumor characteristics, and is described as "stricter than the Volume Loss."
- **Reported effect.** Up to +15% tumor-detection F1 and DSC over mask-only training on external validation. [arXiv 2608.27668](https://arxiv.org/abs/2608.27668) (local PDF read)

**Cardiac EF and volume.**
- **Neural-ODE echo registration (arXiv 2306.09687).** Adds a "clinical attribute loss": |EF from registration-propagated masks − EF from a pretrained segmenter|. EF is a ratio of soft areas/volumes. — [arXiv 2306.09687](https://arxiv.org/pdf/2306.09687) (snippet; not read)
- **Naidoo et al., MIDL 2022 short paper.** "Influence of Loss Function on Left Ventricular Volume and Ejection Fraction Estimation." It only compares standard losses and has no EF loss. See Q4 for the numbers. — [UWL repository](https://repository.uwl.ac.uk/9076/7/A.influence_of_loss_function_on_.pdf)

**CDR in chest X-ray and carotid IMT.**
- I found **no** differentiable CTR loss. CTR is computed in post-processing; for example arXiv 1807.03434 trains with CE plus an adversarial term. — [arXiv 1807.03434](https://arxiv.org/pdf/1807.03434) (snippet)
- I found **no** IMT-thickness loss. The nearest is a carotid "connection loss" for interface continuity (Rhee & Kway 2022, [arXiv 2201.12152](https://arxiv.org/pdf/2201.12152v1.pdf), snippet).

### Inferences
- The planned soft-area estimator (Σp over a region) is exactly the Kervadec, Meng and R-Super construction, so it cannot be claimed as new.
- The ratio-of-soft-areas biomarker supervised against GT is Meng 2022 (vCDR) and Kervadec 2021 (ratio constraints). The planned loss differs in three ways:
  - It is applied **per instance, to hundreds to thousands of fibers per image**, using GT instance masks as integration domains.
  - It uses the **sqrt-area g-ratio** of a nested annulus. Meng instead uses an ellipse semi-axis model.
  - It adds **size weighting**.
- R-Super's Σ(t·o) is the same "mask the soft sum by a region" trick, applied to organ masks.

### Gaps
- I did not read the Constrained-CNN or bounding-box papers this session, or the TMI 2023 extension. Their details come from memory and snippets.
- I did not find a 2023–2026 MICCAI paper with a per-instance ratio loss in nuclei or cells (for example a nucleus-to-cytoplasm ratio loss). This is worth one more targeted search, "nuclear-cytoplasmic ratio loss".

## Q3. Janjic et al. 2019, "Measurement-oriented deep-learning workflow" (PMID 31377177)

### Takeaway
"Measurement-oriented" refers to the **post-segmentation measurement and rejection stage**, not to the loss. The network is a 2019-era patch classifier (45×45 patch → softmax over 3 classes) with no measurement term in training. This paper is not a novelty threat, but it is the right citation for per-fiber g-ratio measurement with artifact rejection.

### Cited Findings
- **Citation.** Janjic P, Petrovski K, Dolgoski B, …, Kocarev L, Dwork AJ. J Neurosci Methods 2019;326:108373. doi:10.1016/j.jneumeth.2019.108373, PMC6751333. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/31377177/)
- **Abstract.** "Automated measurement of each putative fiber rejects measures encountering pre-defined artifacts and excludes fibers failing to satisfy pre-defined conditions." G-ratios differed from expert segmentation by 2.96%, 0.74% and 2.83%, and ICCs were mostly >0.9. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/31377177/)
- **Network.**
  - CNN classifying the central pixel of 45×45 patches: 3 conv layers (80/70/60 maps, 4×4), a 500-unit fully connected layer and softmax.
  - Initialized from a denoising autoencoder, trained with Adadelta.
  - The loss is not named. No measurement enters training.
- **Ray-Measurement Tool.** The paper describes itself as "a measurement-oriented approach that is insensitive to classification errors." Rays run from the ellipse-fit centroid, and "Measurements along a ray meeting one or more specific stopping conditions are disqualified." ROIs with implausible myelin/axon ratios are rejected. — [PMC6751333](https://pmc.ncbi.nlm.nih.gov/articles/PMC6751333/)

### Inferences
- Cite Janjic as prior per-fiber, measurement-focused evaluation. It shows a measurement-robust pipeline at **inference**, which is the opposite design choice to putting the measurement into the loss.

### Gaps
- I read only the first ~100k characters of the PMC page; the remaining text was not reviewed.

## Q4. Do ratio or measurement losses improve the derived measure compared with plain Dice?

### Takeaway
The evidence is thin, mostly positive, and confounded. Meng's ablation shows vCDR MAE 0.127 → 0.112 when the vCDR loss is added, but that loss also brings ~38k extra weakly labelled images. Kervadec shows that descriptor losses alone hurt Dice. Two negative or neutral pieces of evidence argue that Dice is a poor proxy for derived measures and that soft Dice biases volume. Together they motivate a measurement loss without showing it helps when full masks are already available.

### Cited Findings
- **Positive (Meng 2022, Table 2).**
  - Adding L_vCDR to the same architecture: UKBB vCDR MAE 0.127 → 0.112, Corr 0.341 → 0.410.
  - Segmentation Dice changed by ≤0.2 points.
  - The vCDR loss here also exploits vCDR-only labelled images, so its gain is not isolated from the extra data.
  - [Meng et al. 2022](https://livrepository.liverpool.ac.uk/3169382/1/MICCAI2022_Shape-Aware%20Weakly_Semi-Supervised%20Optic%20Disc%20and%20Cup%20Segmentation%20with%20Regional_Marginal%20Consistency.pdf)
- **Positive (R-Super).** Report-derived volume and ball losses raise tumor F1/DSC by up to +15% over mask-only training. This is again an extra-label (weak-supervision) setting. — [arXiv 2608.27668](https://arxiv.org/abs/2608.27668)
- **Negative or limiting (Kervadec 2021).** Shape-descriptor losses alone reach Dice 0.768 versus 0.876 for CE on ACDC. They are useful as weak supervision, not as a replacement for the pixel loss. — [ar5iv 2105.00859](https://ar5iv.labs.arxiv.org/html/2105.00859)
- **Dice decoupled from the measure (Naidoo et al., MIDL 2022), echo LV:**

  | Loss | DSC | Volume error (ml) | EF error (%) |
  |---|---|---|---|
  | BCE+Dice | 0.928 | 6.64 | 3.47 |
  | Focal Tversky | 0.934 | 3.11 | 3.48 |
  | Dice | 0.932 | 4.32 | 3.69 |

  - Quote: "excellent scores for DSC and HD do not imply the most optimal volume computation for EF."
  - Quote: "some large errors in the volume measurements may be cancelled out in the EF measurements."
  - In the EF example, a ratio, errors partly cancel.
  - [Naidoo et al. 2022](https://repository.uwl.ac.uk/9076/7/A.influence_of_loss_function_on_.pdf)
- **Soft-Dice volume bias.** Soft-Dice optimization yields a volume bias that scales with inherent uncertainty. CE yields correct volume estimates by direct summation of soft maps. — [Bertels/Tilborghs et al., MedIA, arXiv 2211.04161](https://arxiv.org/pdf/2211.04161) (snippet; not read)
- **Domain caution.** A Dice-trained nerve CNN matched mean g-ratio (−0.5%) while diameter and thickness were off by about 9.5%. — [PMC6420354](https://pmc.ncbi.nlm.nih.gov/articles/PMC6420354) (snippet)

### Inferences
- **Ratio cancellation.** g-ratio, like EF, can be right while its parts are wrong. A g-ratio-only loss could be satisfied by co-scaled axon and fiber errors. Pair it with per-instance area terms (|Â_axon − A_axon|, |Â_fiber − A_fiber|), or report diameter and thickness errors too.
- **Soft-area calibration.** Σp equals the hard area only if probabilities are calibrated or near-binary (the Bertels/Tilborghs result). The loss should be evaluated on **hard** post-argmax g-ratio, not only on soft g-ratio.
- **Required ablation.** No study isolates "measurement loss vs Dice with identical full mask supervision" for nested ratios. Running that ablation would be a contribution in itself.

### Gaps
- I found no published negative result in which a ratio loss worsened the derived measure. The absence may reflect publication bias.

## Q5. Bottom line: what is novel and what must be cited

### Takeaway
**Not novel:**
- Differentiable soft-area (Σp) biomarkers.
- Ratio-of-areas supervision of a nested structure pair, which is Meng 2022 vCDR.
- Descriptor or ratio constraint losses (Kervadec).
- Region-masked soft sums (R-Super).

**Apparently novel (no counter-example found):**
- (i) A **per-instance** ratio loss over many touching objects, with GT instance masks as integration domains.
- (ii) Applied to **g-ratio of axon/myelin** in EM, which is a new domain.
- (iii) The sqrt(area-ratio) g-ratio form with **size weighting** toward small fibers.
- (iv) Use alongside **full mask supervision**, to target the evaluation biomarker rather than to provide weak labels.

### Cited Findings
- **Must cite: Meng et al. MICCAI 2022.** Differentiable vCDR from Σp areas plus perimeters, with an MSE loss. — [PDF](https://livrepository.liverpool.ac.uk/3169382/1/MICCAI2022_Shape-Aware%20Weakly_Semi-Supervised%20Optic%20Disc%20and%20Cup%20Segmentation%20with%20Regional_Marginal%20Consistency.pdf)
- **Must cite: Meng et al. TMI 2023.** The journal version. — [AMiner](https://www.aminer.cn/pub/63108da290e50fcafdcd3031/dual-consistency-enabled-weakly-and-semi-supervised-optic-disc-and-cup-segmentation)
- **Must cite: Kervadec et al. MIDL 2021.** Soft-moment descriptors and inter-class ratios. — [PMLR 143](https://proceedings.mlr.press/v143/kervadec21a.html)
- **Must cite: Kervadec et al. MedIA 2019.** Constrained-CNN size losses. — [arXiv 1805.04628](https://arxiv.org/abs/1805.04628)
- **Should cite: R-Super (MedIA 2026).** Soft region-masked volume loss and per-tumor Ball Loss. — [arXiv 2608.27668](https://arxiv.org/abs/2608.27668)
- **Should cite: Naidoo et al. MIDL 2022.** Dice is decoupled from volume and EF. — [UWL](https://repository.uwl.ac.uk/9076/7/A.influence_of_loss_function_on_.pdf)
- **Should cite: Janjic et al. 2019.** Measurement-oriented per-fiber g-ratio, as the domain baseline. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/31377177/)

### Inferences
- Do not frame the novelty as "first differentiable biomarker loss" or "first ratio loss". Meng 2022 refutes both. Defensible framing:
  - The first **instance-level** morphometric (g-ratio) loss for paired axon/myelin segmentation.
  - Evaluated with full supervision against an explicit Dice/CE baseline.
  - Reported on hard-mask g-ratio, axon diameter and myelin thickness errors, stratified by fiber size.
- Reviewers will likely ask why not regress g-ratio directly, and whether the loss just mirrors Dice. Prepare ablations:
  - Area-only terms versus the ratio term.
  - Size weighting on versus off.
  - Soft g-ratio versus hard g-ratio.

### Gaps
- I did not check Kervadec's own follow-ups, such as Bounding-box tightness priors, or 2024–2026 "clinical-metric-aware" loss papers in nuclei or polyp segmentation for per-instance ratio terms. I recommend one more targeted Scholar search before submission: "instance-wise" "area ratio" loss, and "nuclear-cytoplasmic ratio" loss.
