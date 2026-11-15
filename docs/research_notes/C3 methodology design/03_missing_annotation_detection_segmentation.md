# Learning with missing instance annotations: methods for C3 (detection, instance segmentation, cells, lesions, remote sensing)

Research run of 15 Nov 2026. It extends `C3 incomplete label learning novelty/00_papers_read.md` and `Phase 0b evidence/03_c3_unread_papers.md`; papers fully covered there (SparseDet, Calibrated Teacher, Yang et al. PU detector, BMVC 2021 PCIS, Vădineanu 2022/2024, LNQ 2023 atlas weighting, Panteli exclusive CE, Zhao et al. PU cells) are only cross-referenced, not re-summarised.

**Reading depth key** (stated per source below): *full* = method and results read (HTML or extracted PDF text); *partial* = selected sections; *abstract* = abstract, proceedings page or search snippet only. Preprints are marked **[preprint]**.

**Search coverage:** about 20 web searches; full-text reads of 8 papers; a programmatic title scan of the complete MICCAI 2024, 2025 and 2026 open-access indexes (papers.miccai.org) for incomplete/partial/missing/sparse/omission/positive-unlabeled titles. That scan found only three instance-omission segmentation papers across the three years (TIA 2024, ReCo-I2P 2025, ImPartial 2026). Everything else that matched concerned missing modalities, noisy labels or sparse slices.

---

## Q1. Sparsely annotated object detection / instance segmentation, 2024–2026: mechanisms, how unlabelled positives are found, gains

### Takeaway
From 2024 to 2026 the sparsely annotated object detection (SAOD) field moved from fixed-threshold pseudo-labels to teacher–student designs with an explicit pseudo-label **quality gate**: strong/weak student pairs (Co-Student, ECCV 2024), learned selection policies (PECL, NeurIPS 2024), feature-similarity gates (SAMPD, AAAI 2025), easy-to-hard mining with reweighting (S²Teacher), and a known-vs-novel split (DPOD, 2026 preprint). Almost all of them still add recovered objects as **positives** rather than ignoring them. None is an instance-segmentation method. The only instance-level positive-unlabeled (PU) method with masks is still SPOCO (CVPR 2022), from microscopy.

### Cited Findings
**Detection (natural images and aerial)**
- **Co-Student** (Wu, Han, Zheng, Wang; ECCV 2024, LNCS, DOI 10.1007/978-3-031-72970-6_26). A teacher denoises pseudo-labels, and two students (one on strong, one on weak augmentation) exchange the cleaned pseudo-labels with consistency regularisation. It reports +1.0 to +3.0 AP over prior SAOD work across five sparse settings and 95.1% of fully annotated FCOS on COCO. Code: github.com/hustvl/CoStudent (MIT, cvpods, FCOS). Abstract only — [ECVA page](https://ecva.net/papers/eccv_2024/papers_ECCV/html/6407_ECCV_2024_paper.php); [code](https://github.stncp.dns-dynamic.net/hustvl/CoStudent)
- **AdaptFG**, "As Pseudo-Label Free as Possible" (Yao, Liu, Jia, Chen, Zhuo; AAAI 2025, 39(9):9418–9426). It argues that pseudo-labels inject noise when labels are extremely sparse. Instead, a CLIP-conditioned VAE feature generator with an "Adaptor" synthesises class features, plus inter-class relation reasoning. It reports outperforming prior SAOD methods on VOC and COCO. No code was found. Abstract only — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/33020)
- **PECL**, Progressive Exploration-Conformal Learning (Lu, Wang, Xu, Zheng, Cui; NeurIPS 2024). It targets aerial images, where objects are dense (68.4 per image in DOTA vs 7.7 in COCO) and confidence is uneven, so fixed-threshold filtering fails. A "conformal pseudo-label explorer" learns a selection policy with reinforcement-learning feedback from a "multi-clue selection evaluator", in a closed loop with detector updates. The authors note the RL part increases training time. Code: github.com/SAOD-research/PECL. Abstract only — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/47a287298e7887d1c25d4aabb918bd54-Abstract.html)
- **S²Teacher** (Lin et al.). Sparsely annotated *oriented* detection in remote sensing. It "progressively mines pseudo-labels for unlabeled objects, from easy to hard" and "reweights the loss for unlabeled objects". It claims near-fully-supervised DOTA performance with 10% of instances annotated. arXiv 2504.11111 (Apr 2025); the AAAI OJS lists it (article 37634), but the page was bot-blocked, so the year (probably AAAI-26) is unconfirmed. Code was announced as "will be public". Abstract only — [arXiv](https://arxiv.org/abs/2504.11111); [AAAI OJS listing](https://ojs.aaai.org/index.php/AAAI/article/view/37634)
- **SAMPD** (Lee, Shin, Park, Kim; AAAI 2025; arXiv 2501.02640). Read in full.
  - Teacher–student. A pseudo-label's feature is compared by cosine similarity to the ground-truth features. The pseudo-label is **positive if its max similarity > τ1 = 0.9, negative if < τ2 = 0.7, and uncertain in between**. A contrastive loss uses these groups.
  - Per-modality loss weights are the mean max-similarity.
  - Accepted pseudo-labels are promoted to ground truth during training, and GT patches are pasted into low-saliency regions.
  - SSD-VGG16, 80 epochs, 2 GPUs.
  - Miss rate (lower is better), KAIST, 30/50/70% labels removed:

    | Method | 30% | 50% | 70% |
    |---|---|---|---|
    | Supervised | 13.89 | 19.27 | 31.87 |
    | Co-mining | 11.66 | 17.90 | 28.80 |
    | SparseDet | 10.92 | 18.07 | 28.20 |
    | Calibrated Teacher | 10.47 | 17.67 | 25.48 |
    | **SAMPD** | **8.56** | **15.27** | **23.52** |

  - Code: github.com/VisualAIKHU/SAMPD.
  — [arXiv HTML](https://arxiv.org/html/2501.02640v2); [AAAI OJS listing](https://ojs.aaai.org/index.php/AAAI/article/view/32472)
- **DPOD**, "Towards Sparsely Annotated Open-World Object Detection" (Han, Kim, Park; arXiv 2608.12714, 13 Aug 2026) **[preprint]**. Read in full. It separates recovery of *unlabelled known-class* objects from discovery of *unknown* objects.
  - **Known-unlabelled recovery:** a teacher filters multi-view student predictions by consistency, and the survivors are **added to the sparse ground truth** and matched to max-IoU proposals.
  - **Unknown candidates:** objectness ≥ τo = 0.2 and cross-view logit cosine ≤ τs = 0.95.
  - A facility-location loss pushes the embeddings of known and unknown proposals apart.
  - Thresholds are fixed heuristics, which the authors flag as a limitation.
  - Results (K-mAP / U-Recall): Easy split Task 1, 56.48/55.34 vs CROWD 53.28/45.53. Extreme split, 45.37/45.56 vs 34.37/40.66.
  - Code: github.com/HelloHeeju/SA-OWOD.
  — [arXiv HTML](https://arxiv.org/html/2608.12714)
- **Venue corrections from DPOD's related work:**
  - SparseDet is listed as **ICCV 2023**.
  - Co-mining is listed as **AAAI 2021** (two detectors exchanging high-confidence predictions).
  - Background Recalibration Loss (BRL) is listed as **ICASSP 2020**.
  - Dense Teacher is listed as ECCV 2022.
  — [arXiv HTML](https://arxiv.org/html/2608.12714)
- **Origin of "sample out the unlabelled":** Niitani et al., CVPR 2019 (sampling techniques for large-scale detection from sparsely annotated objects). It uses a pretrained model's pseudo-labels plus structural cues to filter unreliable ones. Abstract only — [CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Niitani_Sampling_Techniques_for_Large-Scale_Object_Detection_From_Sparsely_Annotated_Objects_CVPR_2019_paper.pdf)
- **Benchmark practice:** Roboflow100-VL (NeurIPS 2025 Datasets & Benchmarks) re-annotated non-exhaustive datasets and pseudo-labelled the rest with a trained teacher. It treats missing labels as something to fix *before* evaluation. Snippet only — [arXiv](https://arxiv.org/pdf/2505.20612)

**Instance segmentation with missing instances**
- **SPOCO** (Wolny, Yu, Pape, Kreshuk; CVPR 2022). A proposal-free pixel-embedding method with differentiable instance extraction, so the loss acts on instances. It is "the first approach to train with weak positive-unlabeled supervision": only some object masks are given, the background is unlabelled, and "a novel self-supervised consistency loss" covers the unlabelled parts. Evaluated on 2D/3D microscopy, CVPPP and Cityscapes. Code: github.com/kreshuklab/spoco. Abstract only — [arXiv](https://arxiv.org/abs/2103.14572)
- **SOIS** (Xue et al., arXiv 2208.09023, 2022) **[preprint]**. Open-world instance segmentation. A foreground-pixel prediction with a cross-task consistency loss is said to help with incomplete instance annotation: AP100 +4.75 (UVO→UVO), and 30% labels beat 50% fully supervised. No code. Abstract only — [arXiv](https://arxiv.org/abs/2208.09023)
- No 2024–2026 top-venue paper was found on sparsely annotated *instance segmentation* with Mask R-CNN-style models; searches returned only detection papers and the 2021–2022 works above. — [search outcome; see Gaps]

### Inferences
- The field's mechanism menu is:
  1. **Ignore or down-weight** (Niitani sampling, BRL, the S²Teacher reweighting).
  2. **Pseudo-label as positive with a quality gate** (Calibrated Teacher, Co-Student, SAMPD, DPOD, PECL).
  3. **PU risk** (Yang 2020; Zhao 2023).
  4. **Consistency on unlabelled regions** (SPOCO, SOIS).
  5. **Avoid pseudo-labels** (AdaptFG).

  Since 2024 the top-venue winners are in family 2. That reinforces the 03_c3 lesson that ignore is a floor, not a ceiling.
- SAMPD's gate is a three-way similarity split (positive > 0.9, negative < 0.7, uncertain ignored), and it maps directly onto our two kinds of unlabelled fibers. A random-miss normal fiber looks like labelled fibers (high similarity → pseudo-positive). A protocol-excluded abnormal fiber should look *unlike* them (low similarity → stays background, or is ignored if uncertain). This is the closest published mechanism to a protocol-respecting C3. The similarity would be computed against the labelled-fiber feature distribution of the nnU-Net encoder or an instance embedding.
- DPOD's separation of "known-class unlabelled" from "unknown" is the detection analogue of separating "normal skipped fiber" from "abnormal excluded fiber". Its facility-location separation loss is one way to keep the two apart in feature space.

### Gaps
- Co-Student, AdaptFG, PECL and S²Teacher were read at abstract level only, so their exact thresholds and losses are not recorded here. S²Teacher's venue/year is unconfirmed (OJS bot-blocked).
- BRL's exact loss was not read. Only DPOD's one-line description is available.
- No SAM-assisted SAOD paper was found at a top venue. Searches for SAM plus missing annotations returned only SAMEO (CVPR 2025, amodal masks; not about missing labels) and SAM2 hole-filling.

---

## Q2. Cell/nucleus instance segmentation with incomplete labels, 2023–2026

### Takeaway
The 2025–2026 cell literature handles partial annotation almost entirely by **restricting the loss to annotated pixels**: Sketchpose (MELBA 2025) for Cellpose/Omnipose-style flows, and ImPartial (MICCAI 2026) with a Noise2Void-style self-supervised term. The partial labels are always **simulated by random sampling**. PU and pseudo-label methods for missing cells are mostly 2021–2023 (Fujii MICCAI 2021, Zhao MELBA 2023). No 2024–2026 cell paper models selective omission.

### Cited Findings
- **Sketchpose** (Cazorla, Munier, Morin, Weiss; MELBA 3, 2025, DOI 10.59275/j.melba.2025-f7b3; arXiv 2508.17798). Read in full.
  - **Losses:** Omnipose-style flow MSE and distance MSE, **summed only over a "valid distance set" 𝒟** built from strokes and known boundaries; pixels outside 𝒟 contribute nothing. A one-sided ReLU² term penalises predicted distances that exceed the GT on stroke pixels. The boundary head and the Omnipose weight image are removed.
  - **Partial annotations:** simulated by thresholding Gaussian-filtered white noise (σ = 50) to keep 10/25/50/100% of pixels at random.
  - **F1@0.5:**

    | Dataset | 100% | 50% | 25% | 10% | Comparison |
    |---|---|---|---|---|---|
    | MicrobeSeg | 0.81 | 0.69 | 0.72 | 0.60 | Cellpose 0.79 (full labels) |
    | PanNuke | 0.77 | 0.79 | — | 0.73 | LKCell 0.83 |

  - Code: napari-sketchpose (Bitbucket).
  — [arXiv HTML](https://arxiv.org/html/2508.17798)
- **ImPartial** (Shrivastava, Nadeem; MICCAI 2026, LNCS 16891:594–602). Read in full via PMC.
  - **Loss:** segmentation loss only on scribbled pixels, plus a blind-spot (Noise2Void-like) Gaussian-mixture reconstruction loss on about 20% imputed pixels.
  - **Scribbles:** "randomly sampling ground-truth annotations".
  - **TissueNet results:**

    | Method | Scribbles | mIoU | AP | F1@0.5 |
    |---|---|---|---|---|
    | ImPartial | 10% | 0.7486 | 0.7290 | 0.8261 |
    | Cellpose | 10% | 0.6952 | — | — |
    | Cellpose | 100% | 0.7568 | 0.7494 | 0.8333 |

    mAP@0.5:0.95 stays below full-label Cellpose (0.368 vs 0.446). Ablation: scribble loss alone gives mIoU 0.695; the joint loss gives 0.749.
  - Code: github.com/nadeemlab (ImPartial).
  — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13637462); [MICCAI 2026 index](https://papers.miccai.org/miccai-2026/0491-Paper3854.html)
- **RNA2seg** (Genome Biology 2025). It assigns background only where there is positive evidence of background (low RNA density). Pixels that are neither confident cell nor confident background are excluded by "partial backpropagation". Snippet only — [Genome Biology](https://link.springer.com/article/10.1186/s13059-025-03908-9)
- **Fujii et al.**, "Cell Detection from Imperfect Annotation by Pseudo Label Selection Using P-classification" (MICCAI 2021, per search). Labelled cells are positives and other detections are unlabelled; PU learning with P-classification selects reliable pseudo-labels. Abstract only — [arXiv](https://arxiv.org/pdf/2107.09289)
- **BoNuS** (arXiv 2401.07437, 2024) **[preprint]**. Partial point labels; it first detects all nuclei to create pseudo points, with curriculum learning. Snippet only — [arXiv](https://arxiv.org/html/2401.07437v1)
- **Nakai et al.** (ICCV 2025 Workshop CVAMD). Unsupervised nuclei segmentation; SAM pseudo-labels "often miss nuclei or erroneously segment non-nuclei", fixed by SAM + U-Net majority voting. Snippet only — [CVF](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Nakai_Unsupervised_Nuclei_Segmentation_by_Improving_Pseudo_Labels_from_Segment_Anything_ICCVW_2025_paper.pdf)
- **MICCAI 2025 and 2026 scan:** the only cell/nucleus titles with label-efficiency terms were ImPartial (2026), active learning with crowdsourced nucleus annotations (2025) and weakly semi-supervised cervical cell detection (2025). None addresses unannotated instances. — [MICCAI 2025 index](https://papers.miccai.org/miccai-2025/); [MICCAI 2026 index](https://papers.miccai.org/miccai-2026/)

### Inferences
- **Sketchpose is the closest drop-in for a Cellpose-style flow head:** losses masked to a "valid set". Our "valid set" would be all pixels except confident unlabelled ring candidates, which is D-18's ignore expressed for flow/distance targets. It is the natural citation if C3 also gets an instance head (Cellpose-SAM fine-tune).
- **RNA2seg's rule** ("label background only where background is positively evidenced, else no gradient") is a cell-domain precedent for a *conservative* ignore. For us, background would be supervised only where no ring-like structure exists. That is a stronger ignore than D-18 and worth an ablation.
- Every 2025–2026 cell method simulates partial labels randomly. A cell reviewer will therefore accept random-deletion experiments as standard, but they give no evidence about selective omission.

### Gaps
- No 2024–2026 top-venue paper was found that applies PU learning or teacher–student pseudo-labelling to *missing nuclei instances* (as opposed to point/scribble weak supervision).
- Cellpose 3, StarDist, InstanSeg, CellViT and HoVer-Net were not found to have published missing-instance variants.
- NuInsSeg's ambiguous-region masks and their evaluation use were not verified in this run.

---

## Q3. Medical lesion segmentation with missing lesions, and remote sensing

### Takeaway
Medical work on missing lesions in 2024–2025 is sparse and simulation-driven:
- FedIA (MICCAI 2024): random lesion removal, confidence-0.8 relabelling.
- ReCo-I2P (MICCAI 2025): size-biased retention, co-teaching pseudo-labels.
- TIA (MICCAI 2024): one lesion per slice, CAM-based.
- LNQ follow-ups (MELBA): loss masking vs pseudo-labelling.

Remote sensing has the clearest *negative* result: model uncertainty does not reliably identify missing buildings.

### Cited Findings
- **ReCo-I2P** (Wang, Ye, Liao, Zhang, Zhang, Wang; MICCAI 2025, LNCS 15968:467–477). The full PDF was obtained and read in this run; earlier only the reviews had been read.
  - **Architecture:** a VNet with a second decoder pass on prototype-enhanced features (intra-batch prototypes plus a momentum memory bank of N inter-batch prototypes).
  - **Pseudo-label:** Y(i) = hard label of whichever prediction (P1 or P2) is more confident at pixel i.
  - **Reliability:** R(i) is the cosine similarity of the pixel feature to the class-centre feature.
  - **Loss:** L_ReCo = L_wCE(P1, Y; R2) + L_wCE(P2, Y; R1). This is co-teaching, where each branch learns the pseudo-label weighted by the *other* branch's reliability. "A combination of losses" for foreground/background mining is also used.
  - **Results (LNQ2023, DSC):** lower bound 11.33 (orthogonal partial-instance annotation, oPIA) and 39.22 (partial-instance annotation, PIA); co-teaching 41.52/42.93; TriNet 44.08/43.78. The CT Lymph Node upper bound is 67.46. ReCo-I2P's own row was not extracted.
  — [MICCAI 2025 page](https://papers.miccai.org/miccai-2025/0755-Paper3517.html)
- **TIA**, Target-level Incomplete Annotation (Ju et al.; MICCAI 2024; JBHI extension). Partial read. One lesion region is annotated per slice. Pseudo-labels come from CAMs of local-view classification branches with a K-means "class distance loss" and a weighted combination M_o = y + W_f·M_f + W_s·M_s. DSC 47.8% on CD71 (Crohn's). **The supplement compares *which* lesion is annotated:**

  | Choice of lesion | DSC | mIoU |
  |---|---|---|
  | random | 15.15 | 6.76 |
  | largest | 16.14 | 7.12 |
  | smallest | 14.56 | 5.88 |
  | most salient | 16.73 | 7.87 |

  Code: github.com/HeyJGJu/CD_cTIA. — [MICCAI 2024 page](https://papers.miccai.org/miccai-2024/037-Paper1747.html); [supplement](https://papers.miccai.org/miccai-2024/supp/1747_supp.pdf)
- **FedIA** (Xiang et al.; MICCAI 2024). Treats incomplete annotation as noisy data.
  - Estimates per-client completeness from a linear trend fitted to early training IoU/loss, and weights aggregation by it.
  - Low-completeness clients relabel **false-negative lesions only**, using the global model's softmax with a **0.8 confidence threshold**.
  - Missing lesions are simulated by **random removal**. A reviewer objected that small or regionally concentrated lesions would be *systematically* missed in reality; the authors deferred this to future work.
  - Reviewers asked for **recall** in addition to Dice.
  - Code: github.com/HUSTxyy/FedIA.
  — [MICCAI 2024 page with reviews](https://papers.miccai.org/miccai-2024/321-Paper1182.html)
- **"Mask the Unknown"** (Fischer, Kiechle, Lang, Peeken, Schnabel; arXiv 2406.14365, accepted to MELBA). Compares noisy-label training, **loss masking of unlabelled data**, pseudo-labelling with TotalSegmentator, and adding public datasets on LNQ2023 (3rd place, test Dice 0.628). The abstract does not say which strategy won. It does report that training only on clinically enlarged nodes did not generalise to smaller pathological nodes, and that including all visible nodes helped. Abstract only — [arXiv](https://arxiv.org/abs/2406.14365)
- **Remote sensing.** Frontiers in Remote Sensing (2022/23) removed building pixels at 11 omission-noise levels and found "uncertainty is not sufficient for identifying noisy labels": edge uncertainty masks the missing-building signal. Snippet only — [Frontiers](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2022.1100012/pdf); [DLR e-lib](https://elib.dlr.de/193825/1/frsen-03-1100012.pdf)
- **Kang et al.**, "Deep Learning-Based Building Footprint Extraction With Missing Annotations" (IEEE GRSL, 2021 per search). A joint **logit-adjusted cross-entropy + weighted Dice** loss for missing buildings caused by land-cover change and outdated maps. Snippet only; the repository page was bot-blocked — [UJI repository](https://repositori.uji.es/xmlui/handle/10234/194927)
- **SelvaMask** (arXiv 2602.02426, Feb 2026; ISPRS Annals XI-1-2026 version listed) **[preprint/annals]**. Tropical crown delineation with exhaustive annotation of all visible crowns (>8,800 crowns). It is *not* an incomplete-label method. Inter-annotator agreement falls with crown size. Abstract only — [arXiv](https://arxiv.org/abs/2602.02426)
- **Older lesion-mining precedents** (MICCAI 2021 universal lesion detection; arXiv 2009.02577): mine unlabelled lesions with a segmentation branch, then **ignore** the mined suspicious lesions during fine-tuning; about 90% of mined instances were true lesions. Snippet only — [MICCAI 2021](https://miccai2021.org/openaccess/paperlinks/2021/09/01/029-Paper0324.html); [arXiv](https://arxiv.org/pdf/2009.02577)

### Inferences
- ReCo-I2P's reliability-weighted co-teaching is single-network and cheap: a second decoder pass, with only P1 at inference. It is the most nnU-Net-compatible "pseudo-label as positive" design in the medical literature. Because its pseudo-label is pixel-wise, it says nothing about instance or ring topology.
- FedIA's reviewer exchange is direct evidence of what MICCAI reviewers will raise: random-deletion simulation will be challenged as unrealistic, and recall must be reported.
- The remote-sensing uncertainty result warns against using a **plain uncertainty threshold** to find skipped fibers. Edge uncertainty dominates there; for us the analogue is myelin-boundary uncertainty. A ring-shape or ring-evidence prior (cf. the LNQ atlas prior) is better grounded.
- The 2021 lesion-mining "mine then ignore" design is essentially D-18 in a medical setting, so cite it as precedent.

### Gaps
- ReCo-I2P's own results row and its full loss combination were not extracted.
- The winning strategy in "Mask the Unknown" is unknown (abstract only).
- No 2025–2026 remote-sensing paper on incomplete *instance* labels (buildings or trees) at a top venue was found.

---

## Q4. Do any methods distinguish random misses from systematic or protocol omission? Any that learn what annotators skip?

### Takeaway
No vision method found *learns the annotator's omission policy*. The non-random cases in the literature are **simulated by object size**: SAMPD's small-box-biased removal, BMVC 2021 (largest X% kept), ReCo-I2P (larger nodes kept). LNQ uses clinical relevance. TIA compares which lesion is annotated. The formal tool for omission that depends on the object is PU learning under the **Selected-At-Random (SAR)** assumption with a labelling *propensity* e(x), but it has only been applied to tabular data.

### Cited Findings
- **SAMPD (AAAI 2025)** removes boxes with **higher probability for small boxes** "to mimic human miss patterns". At 30% removal, size-biased removal gives a supervised miss rate of 13.89 vs 13.04 for random, so the authors treat it as the harder setting. — [arXiv HTML](https://arxiv.org/html/2501.02640v2)
- **TIA (MICCAI 2024)**: the choice of annotated lesion changes results. DSC: salient 16.73 > largest 16.14 > random 15.15 > smallest 14.56. — [supplement](https://papers.miccai.org/miccai-2024/supp/1747_supp.pdf)
- **FedIA reviewers** state that real misses are systematic (small or regionally concentrated lesions). The authors used random removal and acknowledged this. — [MICCAI 2024 page](https://papers.miccai.org/miccai-2024/321-Paper1182.html)
- **LVIS** documents that real missing annotations cluster: "Missing annotations often occur in 'crowds' where there are a large number of instances and delineating them is difficult". 9% of positive-set image-category pairs were flagged non-exhaustive. — [LVIS, CVPR 2019, arXiv 1908.03195](https://arxiv.org/abs/1908.03195)
- **"Mask the Unknown"**: training only on clinically enlarged (selectively labelled) nodes did not generalise to smaller pathological nodes. — [arXiv](https://arxiv.org/abs/2406.14365)
- **PU theory.** Under SCAR the labelling probability of a positive is constant; under **SAR it depends on the attributes**. Propensity scores e(x) then allow unbiased learning (Bekker & Davis, "Learning from PU data under the SAR assumption"; "Beyond the Selected Completely At Random Assumption"). Learning e(x) from PU data alone is **unidentifiable without extra assumptions**. Snippet/abstract level — [arXiv 1809.03207](https://arxiv.org/pdf/1809.03207); [arXiv 1808.08755](https://arxiv.org/html/1808.08755)
- **Risk bounds** for PU under SAR when the propensity of labelled instances is known (JMLR 24, 2023). — [JMLR](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf)
- **PULSNAR** clusters selected-not-at-random positives into subtypes and estimates class proportions per subtype. — [arXiv 2303.08269](https://arxiv.org/pdf/2303.08269); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11622864/)
- **PUe** (arXiv 2607.13428, 2026) **[preprint]**: biased PU enhancement via causal inference. Title/snippet only — [arXiv](https://arxiv.org/pdf/2607.13428)
- **DPOD** separates "unlabelled known-class" from "unknown" objects with a cross-view similarity rule (τs = 0.95) and an embedding separation loss. This is the nearest detection mechanism to treating two kinds of unlabelled objects differently, but the two kinds are classes, not annotator decisions. — [arXiv HTML](https://arxiv.org/html/2608.12714)

### Inferences
- **The gap is confirmed and sharpened.** Selective omission exists in the vision literature only as a *simulation knob* (size). No method estimates a per-object labelling propensity from image appearance. A SAR-PU formulation with e(x) depending on fiber appearance would be new for segmentation; the identifiability caveat must be handled.
- **We have a rare identifying resource:**
  - overlap-disagreement fibers (labelled in one tile, unlabelled in the other) give e(x) < 1 for normal fibers;
  - the protocol rule (abnormal → background) says the abnormal class should be *negatives*, not low-propensity positives.

  That is exactly the "extra assumption" the SAR literature says is needed.
- **Practical reading:** C3 needs a three-way rule for unlabelled ring candidates:
  1. looks like labelled normal fibers → positive or ignore;
  2. looks abnormal → keep as background (the protocol);
  3. uncertain → ignore.

  SAMPD's similarity gate (0.9/0.7) is the closest published template for this rule.

### Gaps
- Not found: any paper that *learns* which objects annotators skip from data (an annotator-omission model) for detection or segmentation.
- The MICCAI 2026 GP paper "Explicit Modelling of Annotation Bias and Variability" was seen in the index but not read. It probably concerns boundary bias, but this is unverified.
- The venue of PULSNAR and of the Bekker SAR papers was not verified in this run.

---

## Q5. Evaluation protocols: simulated omission, fully annotated subsets, known-positive sets — what reviewers accept

### Takeaway
Three protocols are accepted:
1. **Simulated deletion** from a fully annotated set, scored against the full labels. This is the universal SAOD and cell standard, using random or size-biased removal at fixed rates.
2. **Federated or known-exhaustive evaluation**, as in LVIS: false positives count only where annotation is verified exhaustive, and recall counts everywhere.
3. **Relabelling or re-annotating** the evaluation set (Roboflow100-VL).

Reviewers ask for recall and challenge random-only simulation.

### Cited Findings
- **SAOD splits** (VOC/COCO), as used by DPOD and inherited from BRL/Co-mining practice:

  | Split | Labels removed | Rule |
  |---|---|---|
  | Easy | ~18.3% | one per image |
  | Hard | ~38.8% | half |
  | COCO-50miss | ~46.7% | half, no per-class guarantee |
  | Keep1 | ~50% | one kept per category |
  | Extreme | ~65.8% | one kept per image |

  The paper does not say whether removal is random. — [arXiv HTML](https://arxiv.org/html/2608.12714)
- **SAMPD:** 30/50/70% removal, both random and size-biased. — [arXiv HTML](https://arxiv.org/html/2501.02640v2)
- **Cells:** random pixel or scribble sampling (Sketchpose: 10–100% via thresholded smoothed noise; ImPartial: random scribble sampling; Vădineanu: random instance deletion). — [Sketchpose](https://arxiv.org/html/2508.17798); [ImPartial](https://pmc.ncbi.nlm.nih.gov/articles/PMC13637462)
- **LVIS federated evaluation:**
  - Positive set P_c is exhaustively annotated for c; negative set N_c contains no c.
  - "if a detector reports a detection of category c on an image i ∉ P_c ∪ N_c, the detection is not evaluated".
  - For images whose exhaustive flag e_ci is false, "we do not count false positives for category c … We do measure recall on these images".
  - The flag comes from a verification stage: ≥4 annotators must agree the labelling is exhaustive, and 2 dissenters mark it false.

  — [LVIS arXiv 1908.03195](https://arxiv.org/abs/1908.03195)
- **FedIA reviewers** requested recall and per-image Dice and questioned the realism of random removal. — [MICCAI 2024 page](https://papers.miccai.org/miccai-2024/321-Paper1182.html)
- **Roboflow100-VL (NeurIPS 2025 D&B)** re-annotated non-exhaustive datasets and told annotators to "check backgrounds for missed objects". — [arXiv](https://arxiv.org/pdf/2505.20612)
- **Pototzky BMVC 2021** reports both the oracle ignore (anchors overlapping all missing objects ignored) and the 100%-label upper bound. Already in 03_c3; listed here because the reference arms (oracle ignore, full-label bound) are part of the accepted protocol. — [local PDF `docs/references/papers/pototzky2021_bmvc_only_large_objects_labeled.pdf`]

### Inferences
- **Recommended C3 protocol, combining the accepted pieces:**
  - **(a) Simulated deletion on our labels**, scored on the deleted fibers as known positives (recall of deleted fibers):
    - random at 5/10/20/30% (the field standard);
    - *appearance-biased*, e.g. deleting thin-myelin or irregular fibers preferentially, following SAMPD's size-biased precedent.

    Both include a full-label upper bound and an oracle-ignore arm (Pototzky).
  - **(b) Known-positive real misses:** the overlap-disagreement fibers (H3).
  - **(c) An LVIS-style federated precision:** count false positives only on tiles verified exhaustive (the expert-review subset, `Phase 0b evidence/10_expert_review_protocol.md`), plus a protocol-violation false-positive rate on the abnormal ring structures left as background.
  - Report official precision separately, per the observed-label-illusion rule.
- LVIS is the strongest citation for "precision only where exhaustiveness is verified". It is a CVPR 2019 benchmark design that reviewers know, so the C3 evaluation is not an ad hoc invention.

### Gaps
- No paper found evaluates on a *real* (non-simulated) missing-instance set the way H3 does. LVIS's non-exhaustive flags are the closest, but they are used to excuse false positives, not to measure recovery.

---

## Q6. Top candidates for C3: exact rule, thresholds, architecture, compute, code; adaptation to nnU-Net semantic + watershed or an instance head; baselines a reviewer will demand

### Takeaway
In order of fit to our setting:
1. **Gated pseudo-labelling** (SAMPD-style three-way similarity gate on an out-of-fold teacher) as the main "beyond ignore" arm.
2. **Hard or soft ignore** (nnU-Net ignore label; LNQ atlas weighting; mine-then-ignore) as the primary and the floor.
3. **Reliability-weighted co-teaching** (ReCo-I2P) as a cheap single-network alternative.
4. **Masked-loss instance head** (Sketchpose/SPOCO) if C3 gets an instance branch.
5. **PU risk** (nnPU; Zhao 2023) as the theory baseline.

Reviewers will expect at least: naive, ignore (predicted and oracle), a teacher–student pseudo-label method (Calibrated Teacher or Co-Student), a PU loss, a co-teaching/noisy-label method, and a full-label upper bound under simulated deletion.

### Cited Findings (mechanism facts per candidate)

**Candidate comparison** (all rows are from sources cited above):

| Candidate | Venue | Rule | Thresholds | Arch / compute | Code |
|---|---|---|---|---|---|
| SAMPD gate | AAAI 2025 | cosine(pseudo-label feature, GT features): >τ1 positive, <τ2 negative, else uncertain; promote positives to GT | τ1 0.9, τ2 0.7, temperature 0.1 | SSD-VGG16, teacher–student, 80 epochs, 2 GPUs | VisualAIKHU/SAMPD |
| DPOD | arXiv 2026 [preprint] | multi-view consistent teacher predictions added to GT; novel if objectness ≥ τo and cross-view cosine ≤ τs | τo 0.2, τs 0.95 | CROWD-based OWOD detector | HelloHeeju/SA-OWOD |
| Co-Student | ECCV 2024 | teacher-denoised pseudo-labels exchanged between strong/weak students | not read | FCOS, three networks | hustvl/CoStudent |
| PECL | NeurIPS 2024 | RL policy selects pseudo-labels | learned | slower (RL) | SAOD-research/PECL |
| ReCo-I2P | MICCAI 2025 | Y = more-confident of two branch predictions; reliability-weighted cross-CE | none (confidence comparison) | VNet + 2nd decoder pass, P1 only at test | not recorded |
| FedIA relabel | MICCAI 2024 | flip unlabelled→lesion where softmax ≥ 0.8 | 0.8 | federated | HUSTxyy/FedIA |
| Sketchpose | MELBA 2025 | flow/distance MSE only on valid set; one-sided ReLU² | — | Omnipose-style | napari-sketchpose |
| ImPartial | MICCAI 2026 | scribble-only seg loss + blind-spot GMM reconstruction | 20% blind-spot pixels | U-Net-like | nadeemlab |
| SPOCO | CVPR 2022 | embedding loss on labelled instances + consistency on unlabelled | — | pixel embeddings | kreshuklab/spoco |
| LNQ atlas weighting | MELBA 2024 (already in 00_papers_read) | weight 1 on labels; 1−p or 0.75 on unlabelled by prior p | p 0.25 | nnU-Net-style | — |

### Inferences (adaptation sketches; not literature claims)
**A. C3-ignore (primary, D-18).**
- Out-of-fold B1 predicts ring candidates on train images.
- Unlabelled candidates are set to nnU-Net's ignore label: Gotkowski 2024, already cited. The 2021 lesion "mine-then-ignore" design is a precedent.
- Variant: an LNQ-style soft weight from a ring-evidence prior instead of a hard 0/1.
- Floor ablation: an RNA2seg-style conservative background (supervise background only where there is no ring evidence).
- Cost is the same as one B1 run plus out-of-fold inference.

**B. C3-gate (main "beyond ignore" arm).**
- For each unlabelled candidate, compute an instance embedding: mean-pooled nnU-Net encoder features over the ring, or a frozen foundation-model embedding.
- Compute cosine similarity to the bank of labelled-fiber embeddings from the same image or fold.
- Apply SAMPD's split:
  - > τ1 → write the candidate into the label map as axon X / myelin X+1 (a positive pseudo-label, paired by the existing `pairing.py`);
  - < τ2 → leave it as background, so abnormal fibers stay negatives under the protocol;
  - in between → ignore label.
- Tune τ1/τ2 on overlap-disagreement fibers (H3 positives) against the expert-flagged abnormal fibers (negatives), not on official labels.
- Cost: one extra training run. No new architecture.
- This directly addresses the two-kind problem in 03_c3 §Caveat, which no published method does.

**C. C3-coteach (cheap alternative).**
- A ReCo-I2P-style second decoder pass, or two nnU-Net folds as peers.
- Each peer learns the other's confident pseudo-label weighted by feature-centre reliability.
- This needs a custom nnU-Net trainer, so it is more engineering than B.

**D. Instance head (if adopted).**
- A Cellpose-SAM fine-tune with Sketchpose-style loss masking (flows/distance only on the valid set).
- SPOCO is the PU-embedding reference for nested rings: the axon and myelin of one fiber share an embedding cluster.

**E. PU baseline.**
- Pixel- or instance-level nnPU (Kiryo; Yang 2020; Zhao 2023) with a class prior from the 4–8% miss estimate.
- Expected weak, because SCAR is violated by protocol omission (Q4). That makes it a useful *negative* baseline to motivate the SAR or gate design.

**Reviewer-demanded baseline set** (reasoned from the FedIA, ReCo-I2P and BMVC 2021 review and baseline patterns):
1. Naive (B1).
2. Oracle ignore and full-label upper bound under simulated deletion.
3. Predicted ignore (C3-ignore).
4. Teacher–student pseudo-labels (Calibrated Teacher or Co-Student logic transplanted).
5. Co-teaching / noisy-label (the ReCo-I2P reviewers explicitly asked for incomplete-label rather than noisy-label baselines, so include both).
6. PU loss.
7. A zero-shot foundation-model pseudo-label source (Cellpose-SAM or micro-sam), given Nakai 2025's SAM-miss finding.

All scored by recall on known positives plus federated precision (Q5).

### Gaps
- Calibrated Teacher, Co-Student and PECL thresholds and losses were not re-read here (Calibrated Teacher is already known to the project).
- No measured compute numbers exist for transplanting SAOD methods to dense semantic segmentation. The cost estimates above are reasoning, not sourced.
- No published method validates a similarity gate against *protocol-excluded* objects, so whether abnormal fibers really fall below τ2 is an empirical question for the pilot.
