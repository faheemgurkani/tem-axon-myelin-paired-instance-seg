# Model why fibers go unlabelled, then ignore

**Bottom line.** No TEM or EM method, and no segmentation method in any field we searched, models *why* objects are missing from training labels. Every in-domain axon–myelin tool treats unlabelled pixels as background. The EM methods that handle incomplete labels (SPOCO, WDA-Net, SomaNet, LSD-style loss masking) all assume, or simulate, **random** omission. The best-fitting design for C3 is therefore not a new architecture. It is a **fiber-level selection model**, fitted on CPU, that splits each unlabelled ring candidate into three possibilities: a randomly missed normal fiber, a fiber the protocol deliberately excludes, or not a fiber. The posterior from that model becomes a per-fiber loss weight in the existing nnU-Net ResEnc-M (B1) recipe. This is viable as a methods contribution only in a narrow form. The loss machinery is prior art: nnU-Net's ignore label, LNQ soft weighting, and propensity-weighted PU learning under the "selected at random" (SAR) assumption. What C3 can still claim is **the first application of an instance-level SAR selection model to segmentation masks**, with the random component identified from repeat annotation in overlapping tiles. Its evidence would be a **controlled, appearance-dependent omission benchmark on ASTIH TEM1** that scores both recovery of missed fibers and fidelity to the protocol. The budget allows that benchmark only as about seven runs of roughly 80 epochs, inside the $34 of tranches B and C. Two outcomes are likely: small gains on official metrics, which theory predicts, and a decisive effect only on known-positive fibers. The project's own estimate of a methods paper (about 1 in 3) still stands.

**How to read this report.** Each source is tagged with its venue or status and how deeply it was read: *full* (methods and results read), *methods*, *abstract*, or *snippet* (search result only). **Preprint** marks work that has not been peer-reviewed. Sentences that begin "Inference:", and every design element in §3–§4, are our reasoning, not findings from a cited paper. The underlying notes are in [`docs/research_notes/C3 methodology design/`](../research_notes/C3%20methodology%20design/). This report builds on, and partly supersedes, the [C3 novelty report](C3%20incomplete%20label%20learning%20novelty.md) and the [overlooked-gaps validation](Overlooked%20gaps%20validation.md).

## 1. Thirty-odd methods handle missing objects; none conditions on why they are missing

### Inside TEM and EM: background by default, random omission otherwise

The axon–myelin literature has no training-side correction at all. AxonPath trains plain DiceCE on a three-class map, plus MSE on two signed-distance maps. It has no ignore index and no loss mask ([local code, `external/axonpath/axonpath/utils/losses.py`](../../external/axonpath/axonpath/utils/losses.py); read in full). The ASTIH and AxonDeepSeg code bases contain no partial-label logic either ([note 01](../research_notes/C3%20methodology%20design/01_em_incomplete_labels.md)).

The clearest in-domain statement of our problem comes from a 2026 optic-nerve TEM study. It trained nnU-Net on 368 images from 23 mice and reached **F1 0.771 for axons and 0.697 for fibers**. Its authors note that **"degenerated profiles were excluded from manual annotation and model training"** ([Ophthalmology Science 6(5), 2026](https://www.sciencedirect.com/science/article/pii/S2666914526000795); abstract and snippet only, full text 403). That is the same selective protocol as the SIPAIM data, carried into training without any correction. Whether those profiles were masked as ignore or left as background is unknown. Only one paper models abnormal myelin explicitly: MicroStructFormer adds a supervised "abnormality" class, assigned per fiber ([bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v1.full); **preprint**, full). It treats abnormality as a class to segment, not as a reason a label is missing.

Wider EM work gives four mechanisms, and none of them ties omission to appearance:

- **SPOCO** (Wolny et al., [CVPR 2022](https://arxiv.org/html/2103.14572v3); full) is the method-level precedent. It is a pixel-embedding instance model with differentiable instance extraction. Labelled objects get pull, push and instance-Dice losses. Unlabelled pixels, background included, get two terms: a push away from labelled clusters, and an EMA-teacher consistency loss on anchors sampled in the unlabelled region. **On MitoEM-R with 10% of instances labelled it reaches AP@0.5 0.481, against 0.560 with full labels. Without the consistency term it falls to about 0.30.** Sparsity was simulated by random instance sampling.
- **WDA-Net** ([arXiv 2404.00667](https://arxiv.org/html/2404.00667); **preprint**, methods) weights the loss 1 on labelled regions and on confident background (p < 0.1), and 0 elsewhere. It adds entropy-filtered pseudo-labels and a counting prior. **With 15% random points it reaches Dice 90.1–93.2, against 93.6–94.6 fully supervised.**
- **SomaNet** (Khateri, Sierra et al., [arXiv 2609.23019](https://arxiv.org/html/2609.23019v1), Sep 2026; **preprint**, full) comes from the gACSON/DeepACSON white-matter group. It still sets unannotated voxels to 0 in its supervised loss, and relies on EMA-teacher pseudo-labels to recover. It reports **PQ 0.74–0.82 against 0.60–0.66 for the baselines, with no ablation**. It also does not say how the partial subset was chosen.
- **Local Shape Descriptors** and a 2024 sparse-to-dense follow-up mask the loss to annotated regions ([bioRxiv LSD](https://www.biorxiv.org/content/10.1101/2021.01.18.427039v2.full); snippet. [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v1.full); **preprint**, snippet).
- **micro-sam** sidesteps the problem in prompt-only fine-tuning, because only labelled objects are ever prompted. Its instance decoder, however, "is expected" to get dense labels ([micro-sam docs](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html); read).

Inference: inside EM, C3 would be the first work to treat unlabelled fibers as anything other than background or random misses. The optic-nerve paper gives the "problem exists, nobody corrects it" citation.

### Beyond EM: the field moved to gated pseudo-labels, still under random omission

Sparsely annotated object detection has converged on teacher–student pseudo-labelling with a quality gate. Examples include Co-Student ([ECCV 2024](https://ecva.net/papers/eccv_2024/papers_ECCV/html/6407_ECCV_2024_paper.php); abstract), PECL's reinforcement-learned selection policy ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/47a287298e7887d1c25d4aabb918bd54-Abstract.html); abstract) and S²Teacher ([arXiv 2504.11111](https://arxiv.org/abs/2504.11111); AAAI listing unconfirmed, abstract).

The template closest to our needs is **SAMPD** ([AAAI 2025](https://arxiv.org/html/2501.02640v2); full). It compares each pseudo-label's feature with the ground-truth features by cosine similarity. Above **τ1 = 0.9** the pseudo-label becomes a positive, below **τ2 = 0.7** a negative, and in between it is treated as uncertain. On KAIST with 30% of labels removed this cut the miss rate from **13.89 to 8.56**. SAMPD is also one of very few papers that simulate non-random omission, by preferentially removing small boxes "to mimic human miss patterns". DPOD ([arXiv 2608.12714](https://arxiv.org/html/2608.12714), Aug 2026; **preprint**, full) goes one step further: it separates "unlabelled known-class" objects from "unknown" ones with a cross-view similarity rule (τs = 0.95). This is the closest detection analogue to separating missed normal fibers from excluded abnormal ones. AdaptFG ([AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33020); abstract) argues against pseudo-labels altogether when labels are very sparse. Soft Sampling re-weights background RoIs by their overlap with positives ([arXiv 1806.06986](https://arxiv.org/pdf/1806.06986); PDF read, BMVC unverified). Its authors warn that confidence-based re-weighting "could generate false positives and consequently ignore regions where there are no objects". For object detection as a whole, Yang et al. recast missing boxes as a PU problem ([BMVC 2020](https://arxiv.org/html/2002.04672); abstract).

Cell and nucleus work from 2025–26 restricts the loss to annotated pixels, and always simulates partiality at random:

- **Sketchpose** sums its flow and distance MSE only over a "valid set" of pixels ([MELBA 2025](https://arxiv.org/html/2508.17798); full). On MicroBeSeg, F1@0.5 falls from 0.81 with 100% of pixels labelled to 0.60 with 10%.
- **ImPartial** pairs a loss on scribbled pixels with a blind-spot reconstruction term ([MICCAI 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13637462); full). With 10% scribbles it reaches TissueNet F1 0.826, against 0.833 for full-label Cellpose.
- **RNA2seg** supervises background only where background is positively evidenced ([Genome Biology 2025](https://link.springer.com/article/10.1186/s13059-025-03908-9); snippet), a conservative-ignore precedent.

The older PU line for cells is Zhao et al. ([MELBA 2023](https://arxiv.org/abs/2302.08050); full). It runs nnPU inside Faster R-CNN under uniform random deletion: **at one label per patch, F1 rises from 0.102 to 0.522**.

Medical lesion work is the nearest to deliberate selection:

- **LNQ 2023** annotators labelled only "clinically relevant" lymph nodes: 1.41 per patient in training against 8.61 in test. The team weighted the Dice loss 1 − p or 0.75 where an atlas predicted an unlabelled node, and got **Dice +0.024 and recall +0.03** ([Engelson et al., MELBA 2024](https://arxiv.org/abs/2406.03984); full).
- **ReCo-I2P** uses reliability-weighted co-teaching between two decoder passes ([MICCAI 2025](https://papers.miccai.org/miccai-2025/0755-Paper3517.html); full PDF).
- **FedIA** relabels false negatives above softmax 0.8 ([MICCAI 2024](https://papers.miccai.org/miccai-2024/321-Paper1182.html); page and reviews read). Its reviewers objected that **real misses are systematic, not random**, and asked for recall.
- **TIA**'s supplement shows that *which* lesion is annotated changes the result: Dice 16.73 when the most salient lesion is labelled, against 14.56 for the smallest ([MICCAI 2024 supplement](https://papers.miccai.org/miccai-2024/supp/1747_supp.pdf); partial).

Remote sensing contributes a useful negative result: **model uncertainty does not reliably identify missing buildings**, because edge uncertainty swamps the signal ([Frontiers in Remote Sensing](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2022.1100012/pdf); snippet).

### PU and missing-data theory: SAR is the right family, and MNAR-SSL is the wrong one

The formal tool for "the label depends on what the object looks like" is PU learning under the **SAR assumption**. Each positive x is labelled with propensity e(x) = P(s=1 | y=1, x), and a labelled-versus-unlabelled classifier learns η̃(x) = e(x)·η(x), not η(x) ([Bekker & Davis 2018, arXiv 1808.08755](https://arxiv.org/html/1808.08755), full; [Coudray et al., JMLR 2023, Eq. 4](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf), §3–4 read). The SAR propensity-weighted risk is **unbiased as soon as e is known on the labelled instances**. Its excess risk scales like 1/(n·e_min), so any region where e approaches 0 costs as much as losing the labels ([Coudray et al., Thm 1–2](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf)).

All the difficulty is in identifying e. Bekker & Davis call SAR learning with unknown propensity "ill-defined", because a low η and a low e explain an unlabelled example equally well. They identify it by restricting e to a known subset of attributes, and fit it by EM ([arXiv 1808.08755](https://arxiv.org/html/1808.08755)). Other identification routes:

- Kato et al. assume "invariance of order" between the labelled-positive score and the true posterior ([ICLR 2019](https://mlanthology.org/iclr/2019/kato2019iclr-learning); abstract).
- Gerych et al. assume either local certainty or a "probabilistic gap", e = k·η ([AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20624); abstract).
- PUe uses normalised inverse-propensity weights ([NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/3efb4bdc6bfe13e1ff95b4407c37961d-Paper-Conference.pdf); full via [arXiv 2607.13428](https://arxiv.org/html/2607.13428v1)). On real, unsimulated bias (Alzheimer's) it gains only **+0.2 to +1.8 accuracy points**, against much larger gains on simulated bias.

Deep MNAR semi-supervised learning is a different model family:

- CADR ([Hu et al., ICLR 2022](https://arxiv.org/html/2206.14923); full) makes missingness depend on the **class**, via the chain X → Y → M. It combines a class-aware propensity, class-aware imputation and a doubly robust correction term. On MNAR CIFAR-10 it lifts FixMatch from 56.26 to 79.63.
- [Sportisse et al., ICML 2023](https://proceedings.mlr.press/v202/sportisse23a.html) (abstract) estimates the missing-data mechanism and adds a likelihood-ratio test of whether labels are informative.

Inference: our omission depends on appearance *within* the positive class. In PU terms that is SAR, not class-level MNAR. The project's plan audit already concluded this ([07-plan §6](../notes/07-plan.md)). C3 should therefore use the SAR estimator and borrow only the doubly robust *template* and the informativeness test from the MNAR-SSL line.

### Annotator and protocol models capture boundary style, not object omission

Zhang et al.'s pixel-wise confusion-matrix model fits CE(Â(x)·p̂(x), ỹ) + λ·tr(Â), and identifies the truth only with several annotators and diagonal dominance ([NeurIPS 2020](https://ar5iv.labs.arxiv.org/html/2007.15963); full). Pionono learns a latent code per rater plus a "gold rater" code trained on gold labels. It reaches Gleason κ 0.758, against 0.692 for the pixel confusion matrix ([ICCV 2023](https://arxiv.org/html/2307.11397v1); methods). Two other methods handle labels that differ between datasets:

- **MultiTalent** gives each dataset's classes their own sigmoid heads, and computes the loss only on the classes that dataset annotated ([MICCAI 2023](https://arxiv.org/html/2303.14444v2); methods).
- **Shi et al.'s marginal loss** treats a partially labelled image's "background" as background plus every class it leaves unlabelled ([MedIA 2021](https://arxiv.org/pdf/2007.03868); abstract).

The 2025–26 multi-rater line, including DiffOSeg (MICCAI 2025), P-Diverse (WACV 2026) and ProSona (**preprint**), personalises boundary style. No paper in it models instance omission ([note 04 §3](../research_notes/C3%20methodology%20design/04_annotator_protocol_models.md); snippets). Parikh et al. show that a correction "can appear unsuccessful when evaluated against the same biased annotations used for training". In their experiment the true gap fell from 5.5 to 0.5 pp while the gap measured on the biased labels rose from 11.5 to 12.0 pp ([arXiv 2605.06891](https://arxiv.org/abs/2605.06891); **preprint**, full).

Inference: the Zhang model with binary classes and one-sided noise *is* the SAR-PU model, with a(fg→fg) = e(x). Its trace regulariser then equals 1 + e(x), so minimising it drives e → 0. The model would explain all unlabelled tissue as "omitted fiber", the opposite of what C3 needs. Pixel independence also allows half-omitted fibers. The selection variable must therefore sit at the fiber level.

### Source map

| Source | Venue / status | Depth | Mechanism | Omission assumed | Role in C3 |
|---|---|---|---|---|---|
| [Optic-nerve TEM](https://www.sciencedirect.com/science/article/pii/S2666914526000795) | Ophthalmol Sci 2026 | snippet | nnU-Net, degenerated profiles excluded | selective, uncorrected | motivation |
| [MicroStructFormer](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v1.full) | bioRxiv 2026, **preprint** | full | supervised abnormality class | — | prior art for an abnormal class |
| [SPOCO](https://arxiv.org/html/2103.14572v3) | CVPR 2022 | full | PU embeddings + EMA consistency | random | EM origin; optional regulariser |
| [WDA-Net](https://arxiv.org/html/2404.00667) | **preprint** (BIBM 2022 ext.) | methods | known-positive + confident-negative mask | random | baseline (budget permitting) |
| [SomaNet](https://arxiv.org/html/2609.23019v1) | **preprint** 2026 | full | unlabelled = 0 + teacher pseudo-labels | unanalysed | cautionary precedent |
| [nnU-Net ignore / Gotkowski](https://arxiv.org/abs/2403.12834) | arXiv 2024, venue unconfirmed; [local docs](../../external/nnUNet/documentation/ignore_label.md) | full (docs + code) | partial CE/Dice | — | host mechanism; hard-ignore baseline |
| [LNQ / Engelson](https://arxiv.org/abs/2406.03984) | MELBA 2024 | full | atlas-weighted Dice | selective (clinical) | soft-weight baseline; closest prior |
| [Zhao PU cells](https://arxiv.org/abs/2302.08050) | MELBA 2023 | full | nnPU in Faster R-CNN | random | SCAR-PU baseline origin |
| [SAMPD](https://arxiv.org/html/2501.02640v2) | AAAI 2025 | full | three-way similarity gate | random + size-biased | gate template |
| [DPOD](https://arxiv.org/html/2608.12714) | **preprint** 2026 | full | known vs unknown split | random | analogue for two kinds of unlabelled |
| [Co-Student](https://ecva.net/papers/eccv_2024/papers_ECCV/html/6407_ECCV_2024_paper.php), [PECL](https://proceedings.neurips.cc/paper_files/paper/2024/hash/47a287298e7887d1c25d4aabb918bd54-Abstract.html) | ECCV 2024, NeurIPS 2024 | abstract | gated teacher–student | random | not funded |
| [Sketchpose](https://arxiv.org/html/2508.17798) | MELBA 2025 | full | valid-set masked flow loss | random | instance-head option |
| [ImPartial](https://pmc.ncbi.nlm.nih.gov/articles/PMC13637462) | MICCAI 2026 | full | scribble loss + blind-spot | random | self-supervised regulariser option |
| [ReCo-I2P](https://papers.miccai.org/miccai-2025/0755-Paper3517.html) | MICCAI 2025 | full | reliability-weighted co-teaching | size-biased | pseudo-positive option |
| [FedIA](https://papers.miccai.org/miccai-2024/321-Paper1182.html) | MICCAI 2024 | page + reviews | confidence-0.8 relabel | random | reviewer expectations |
| [Bekker & Davis SAR-EM](https://arxiv.org/html/1808.08755) | LIDTA 2018 workshop; ECML-PKDD 2019 version not read | full | propensity + EM | SAR | estimator origin |
| [Coudray et al.](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf) | JMLR 2023 | §3–4 | SAR risk and bounds | SAR | theory |
| [PUe](https://arxiv.org/html/2607.13428v1) | NeurIPS 2023 | full | normalised IPW | SAR | weight normalisation |
| [CADR](https://arxiv.org/html/2206.14923) | ICLR 2022 | full | class propensity + DR | class-MNAR | DR template only |
| [Zhang CM](https://ar5iv.labs.arxiv.org/html/2007.15963) | NeurIPS 2020 | full | pixel confusion matrix + trace | pixel noise | do not use (degenerates) |
| [Pionono](https://arxiv.org/html/2307.11397v1) | ICCV 2023 | methods | rater codes + gold code | style | protocol-code option |
| [MultiTalent](https://arxiv.org/html/2303.14444v2) | MICCAI 2023 | methods | per-protocol sigmoid heads | missing classes | protocol-head option |
| [Shi marginal loss](https://arxiv.org/pdf/2007.03868) | MedIA 2021 | abstract | marginalised background | missing classes | abnormal-class option |
| [Parikh et al.](https://arxiv.org/abs/2605.06891) | arXiv 2026, **preprint** (NeurIPS unconfirmed) | full | style-conditioned decoder | boundary bias | evaluation rule |
| [LVIS](https://arxiv.org/abs/1908.03195) | CVPR 2019 | abstract | federated precision | clustered | evaluation template |
| Bukas et al. ([local text](../references/papers/bukas2026.txt)) | MICCAI-W 2026 | full | — | — | pseudo-label batch-effect risk |

## 2. Our omission is two mechanisms, and labels alone identify only one

C3's labels hide two different processes behind the same 0 in the mask:

- **Random misses of normal fibers**, about 4–8% ([02-challenge-and-data §3](../notes/02-challenge-and-data.md)). The propensity is high and roughly constant.
- **Deliberate exclusion of abnormal fibers.** About 25% of ring-like structures are left as background, mostly decompacted, split or redundant myelin ([02-challenge-and-data §3](../notes/02-challenge-and-data.md)). The propensity is near zero by protocol. The local completeness audit rated about 60% of 48 unlabelled candidates as abnormal: onion-like thick myelin (low g), large axons with thin myelin (high g), and oblique profiles ([Overlooked gaps validation §1](Overlooked%20gaps%20validation.md)).

Inference: the second component violates positivity. Fibers with e ≈ 0 are not recoverable from labels by any reweighting, which is consistent with the 1/e_min bound above. Kato's order invariance and Gerych's probabilistic gap both fail here as well, because a degenerating fiber is unambiguous as a fiber (high η) yet never labelled.

The resolution is conceptual before it is technical. For the **protocol output**, the one the challenge scores, abnormal fibers are negatives *by definition*, so only the random component is a missing-label problem. The selection-aware task is then to decide which unlabelled candidates are negatives by protocol and which are missed positives. For a **morphometry output** ("all real fibers"), abnormal fibers can only be learned from explicit supervision: an expert-labelled abnormal class, which is Shi's marginal-loss setting. No reweighting of our labels can supply it.

The random component is identifiable from data we already hold. The overlapping tiles act as two labelling draws of the same fibers. The repeat-annotation result counts **TP 118, FP 7 and FN 11** ([05-results §2](../notes/05-results.md)), so 136 fibers were labelled at least once and 18 were labelled in only one tile. Inference: if both draws share one recall r and are independent, then P(discordant | labelled at least once) = 2(1−r)/(2−r). Solving with 18/136 gives **r ≈ 0.93, a random-miss rate of about 7%**. That sits inside the independent EDA estimate of 4–8% random misses of clearly normal fibers, a useful consistency check between two unrelated methods. Three caveats apply:

- Positively correlated misses (same annotator, same hard-looking fiber) make discordance rarer and bias r upward. Whether both tiles had the same annotator is unknown.
- Fibers missed in both tiles are invisible to this estimate. Under independence they number about (1−r)²·136 ≈ 0.7.
- The sample is small (136 fibers) and lies in a different region from the dev-val images ([05-results §2](../notes/05-results.md)), so r̂ carries a wide interval and should be swept, not fixed, in sensitivity analyses.

We found no published use of overlapping image regions as repeat annotations to estimate per-object omission ([note 04 §4](../research_notes/C3%20methodology%20design/04_annotator_protocol_models.md)). The underlying idea is classic dual-system (capture–recapture) estimation, so it must be cited as such once that literature is checked.

The protocol component a(x) = P(excluded | real fiber, x) is not identified by labels alone. Bekker & Davis's "ill-defined" problem applies, and Nguyen et al. report that one annotator's labels are "fundamentally insufficient" for instance-dependent identifiability ([NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95831); snippet only, verify before citing). It needs **anchors**. The strongest anchor is the expert normal/abnormal subset (T-20, currently deferred), which plays the role of Pionono's gold rater or a validation sample. Without it, the fallback is a definitional assumption: "abnormal" means "unlike the labelled fibers". That would be implemented as a SAMPD-style similarity or density score against labelled-fiber features. It is defensible as an operationalisation of the protocol, but it remains an assumption, and the paper must say so.

Theory also caps the expected gain. Coudray et al. restate Cannings et al. (2020): the naive classifier stays Bayes-optimal wherever e(x) ≥ 1/(2η(x)) ([JMLR 2023, Eq. 5](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf)). Inference: with e ≈ 0.93 for normal fibers, that condition holds for every clear normal fiber. **Any C3 benefit will concentrate on ambiguous fibers (thin, oblique, low-contrast) and on the calibration of the fiber probability that drives pairing**, and will be close to zero elsewhere. This agrees with LNQ's +3-point recall and PUe's small real-data gains. It also implies that a 10-point rise on official metrics is unrealistic.

## 3. Recommended method: fiber-level selection-gated loss masking in nnU-Net

### The design

The recommended method keeps B1's architecture, plans, trainer length and shared pairing step. It changes only the per-pixel loss weights, which are derived from a fiber-level selection model. All of the following is our design.

**Step 1: candidate mining (CPU).** Run the D-18 ring-evidence detector on the out-of-fold B1 probabilities from T-21. Out-of-fold probabilities are required because B1's dev model has already seen the training images ([07-plan D-28](../notes/07-plan.md)). Optionally keep only candidates on which zero-shot micro-sam or Cellpose-SAM agrees. Each candidate k gets a paired axon ∪ ring footprint, so that every later decision applies to a whole fiber, never half a ring.

**Step 2: fiber features (CPU).** Compute, for labelled fibers and candidates alike:

- equivalent inner and outer diameters, from which g and myelin thickness follow;
- eccentricity and solidity;
- ring-closure fraction and ring contrast;
- the mean out-of-fold fiber probability t_k;
- optionally, ResEnc encoder features mean-pooled over the ring.

**Step 3: selection model (CPU).** Model each fiber as s_k ~ Bernoulli(t_k · r · (1 − a_k)), where:

- **r** is fixed from the overlap capture–recapture (≈ 0.93);
- **a_k = a(x_k)** is the protocol-exclusion score. With the expert subset it is a logistic regression on 5–10 features. Without it, it is a calibrated novelty score against the labelled-fiber feature distribution, with SAMPD-style thresholds tuned on the 48-candidate audit;
- **t_k** comes from step 1.

The posterior then splits each unlabelled candidate's mass three ways. The probability that it is a randomly missed normal fiber is q_k = t_k(1−a_k)(1−r) / (1 − t_k(1−a_k)r). The probability that it is protocol-excluded is t_k·a_k / (1 − t_k(1−a_k)r), and the rest is "not a fiber". As a worked example: a confident, normal-looking candidate (t = 0.99, a = 0) has **q ≈ 0.87**. A candidate rated half-abnormal (a = 0.5) has q ≈ 0.06 and a probability of about 0.92 of being protocol-excluded. A low-capacity propensity model is deliberate. Bekker & Davis note that flexible models need calibration, and that EM can collapse to e ≡ 1 ([arXiv 1808.08755](https://arxiv.org/html/1808.08755)).

**Step 4: weights (nnU-Net trainer).** On each candidate footprint, weight the background target by **w_k = 1 − q_k**. Likely missed normal fibers are then effectively ignored, likely protocol-excluded fibers stay full-weight background (so the network learns the protocol), and uncertain cases are down-weighted. This is the LNQ weighting idea with a selection-derived weight, so cite LNQ. The hard-gated variant uses three bands, ignore / background / soft, following SAMPD.

Labelled fibers keep weight 1. Inference: inverse-propensity weighting of labelled fibers is redundant here, because r is roughly constant over normal fibers and a ≈ 0 on them. If it is ever used, apply PUe's normalisation and clip ê ≥ 0.2.

Paint each weight over the whole fiber, dilated 1–2 px into background only, so that deep-supervision downsampling does not leak into a touching neighbour ([note 05 §Q2](../research_notes/C3%20methodology%20design/05_architectures_and_evaluation.md)). nnU-Net's hard ignore needs no code: in nnU-Net format it is the highest label. Soft weights need a trainer subclass that carries a float weight map into the CE (reduction "none") and into the Dice `loss_mask`. That is about 50 lines, because `DC_and_CE_loss` already masks by `target != ignore_label` ([`compound_losses.py:37–53`](../../external/nnUNet/nnunetv2/training/loss/compound_losses.py)). Select the **final** checkpoint, because pseudo-Dice on unmodified validation folds penalises recovered fibers.

**Why this, and not pseudo-positives.** Writing recovered fibers into the label map as positives injects model-made shapes. Bukas et al. showed in TEM that mixing human and model labels creates batch effects "most evident in the eccentricity and solidity", and lowers precision ([local text](../references/papers/bukas2026.txt); MICCAI-W 2026, full). Those are exactly the shape features behind g-ratio. Ignore-type weighting adds no model shape to the targets. Gated pseudo-positives (SAMPD, ReCo-I2P, FedIA style) are kept as a later ablation, with a mandatory batch-effect check on per-fiber eccentricity, solidity and g.

### Ranked recommendation

| Rank | Technique | Origin (venue) | Verdict for C3 | Why |
|---|---|---|---|---|
| 1 | **Fiber-level selection-gated soft ignore** (above) | Built from SAR-PU ([Bekker 2018](https://arxiv.org/html/1808.08755), [Coudray JMLR 2023](https://www.jmlr.org/papers/volume24/22-067/22-067.pdf)) + [LNQ MELBA 2024](https://arxiv.org/abs/2406.03984) + [SAMPD AAAI 2025](https://arxiv.org/html/2501.02640v2) | **Adopt as the C3 method** | Only design that separates the two omission mechanisms; zero new architecture; one run per arm |
| 2 | Hard ignore of mined candidates (D-18) | [Gotkowski 2024](https://arxiv.org/abs/2403.12834); mine-then-ignore, MICCAI 2021 | **Primary baseline** (prior art) | Ignores abnormal and normal candidates alike, so it cannot learn the protocol |
| 3 | SCAR ablation of rank 1 (a ≡ 0) | [Elkan & Noto KDD 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf); [Zhao MELBA 2023](https://arxiv.org/abs/2302.08050) | **Required ablation** | Isolates the value of appearance dependence |
| 4 | Random-ignore control, same pixel budget | Our design (novelty report) | **Required control** | Separates "targeting helps" from "less supervision helps" |
| 5 | Oracle ignore (simulation only) | [BMVC 2021](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0979.pdf) | **Reference arm** | Ceiling for any mask-based method |
| 6 | LNQ constant weight 0.75; WDA-Net confident-negative mask; nnPU risk | [LNQ](https://arxiv.org/abs/2406.03984); [WDA-Net](https://arxiv.org/html/2404.00667) (**preprint**); [Kiryo NeurIPS 2017](https://arxiv.org/abs/1703.00593) | Baselines if budget allows | Reviewers will ask; nnPU is expected to fail because SCAR is violated |
| 7 | Gated pseudo-positives / co-teaching | SAMPD, [ReCo-I2P](https://papers.miccai.org/miccai-2025/0755-Paper3517.html), FedIA | Later ablation | Bukas batch-effect risk on g |
| 8 | Abnormal class with Shi marginal loss, or per-protocol heads | [Shi MedIA 2021](https://arxiv.org/pdf/2007.03868); [MultiTalent MICCAI 2023](https://arxiv.org/html/2303.14444v2) | Gate 1 option, needs expert labels | Only route to a morphometry ("all fibers") output; prior art Zepf, Parikh, MicroStructFormer |
| 9 | EMA-teacher consistency on ignored pixels | [SPOCO CVPR 2022](https://arxiv.org/html/2103.14572v3); [ImPartial MICCAI 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13637462) | Optional regulariser, unfunded | SPOCO's consistency term was worth about 0.15 AP; untested on nested rings |
| — | Zhang pixel CM with trace; class-propensity MNAR-SSL (CADR) as estimator; Mask2Former / SPOCO / Cellpose-SAM / InstanSeg retraining; plain uncertainty thresholds; PECL's RL | see §1 | **Do not adopt** | Degenerate prior; wrong missingness family; budget, and confounds C3 with an architecture change; remote-sensing negative result; cost |

Instance-native models deserve one more sentence, because the question asks about architectures. Query models are the only family with a native per-instance score (DETR's no-object class, down-weighted ×10 ([DETR, ECCV 2020](https://ar5iv.labs.arxiv.org/html/2005.12872); loss read)), so a learned labelling-propensity logit would fit there most naturally. Adopting one would discard B1's F1@0.5 of 0.904 and confound the C3 effect with an architecture change, doubling the runs. Inference: under this budget the per-instance score belongs in a **post-hoc CPU fiber classifier**, as in step 3. micro-sam's `object_classification.py` pattern shows the same idea, pooling embeddings per object ([note 05 §Q1](../research_notes/C3%20methodology%20design/05_architectures_and_evaluation.md)).

## 4. Evaluation rests on a TEM1 benchmark, because our data has 18 known positives

### Why the primary test must be simulated

The project's own known-positive set is the 18 overlap-disagreement fibers. A 10-point recall change there is about two fibers, so H3 is reported as secondary ([07-plan, H3](../notes/07-plan.md)). Official metrics on our data cannot decide either, because of the observed-label illusion (Parikh). C3 therefore needs the evaluation pattern reviewers already accept: delete objects from near-complete labels and score on the complete ones. That is the standard in SAOD, cells and the nnU-Net documentation, which recommends evaluating sparse-annotation strategies on densely annotated images ([local docs](../../external/nnUNet/documentation/ignore_label.md)). Two additions are needed, because reviewers challenge random-only deletion (FedIA reviews) and synthetic noise does not transfer to real label errors ([Penquitt et al., arXiv 2609.21822](https://arxiv.org/abs/2609.21822); **preprint**, abstract):

- the deletion is made **appearance-dependent and two-component**;
- **protocol fidelity** is scored alongside recovery.

ASTIH TEM1 is the substrate: 158 images, about 7,700 fibers, near-complete for normal and thin-myelin fibers, morphologically comparable to ours ([02-challenge-and-data §5](../notes/02-challenge-and-data.md)). It also omits degenerating and broken rings, so its protocol stratum must be a *proxy* ([Overlooked gaps validation §2](Overlooked%20gaps%20validation.md)).

### Benchmark protocol (our design)

**Split.** Use the TEM1 fold that holds out the remyelinating-like mice. The H8 reference run needs that fold anyway, and the complete-label arm here is the same run as H8's reference ([07-plan §6 audit](../notes/07-plan.md)). Bootstrap by mouse.

**Simulated protocol, applied to training labels only.**

1. Define a protocol stratum from measurable appearance proxies. It is two-sided, to match our mixed abnormal set: the high-g tail (thin myelin), the low-g tail (thick myelin), and high-eccentricity oblique profiles. Delete those fibers with probability 0.9.
2. Delete the remaining fibers at random at 1 − r ≈ 7%.
3. Simulate the side information at our real size: a second, independent labelling draw on about 136 fibers (the overlap analogue), plus an "expert" subset of known stratum membership.

This tests the identification strategy at realistic power, not just the loss.

**Test labels.** Protocol truth is the full labels minus the protocol stratum. The randomly deleted test fibers stay labelled, as known positives.

**Endpoints.**

- **Primary:** recall of normal fibers against protocol truth. Report it overall, and for the "hard normal" fibers whose features lie nearest the stratum boundary.
- **Co-primary guard:** the protocol-violation rate, i.e. the fraction of stratum fibers predicted as fibers.
- **Secondary:** F1@0.5/0.75 and PQ against protocol truth; g-MAE with fiber recall beside it, plus the PQ-style g-quality metric (D-34); and recovery of the simulated mechanism (AUC of â against the true stratum, error of r̂).

**Decision rule, to pre-register before any run.** The selection-gated arm must beat **both** the hard-ignore arm and the random-ignore control on normal-fiber recall, with paired mouse-bootstrap CI > 0. Its protocol-violation rate must also be no more than 0.05 above the naive arm's.

**On our data.** Compare the selection-gated arm with the existing B1 on the dev fold, scoring:

- the 18 overlap-disagreement fibers (H3, secondary);
- the H8 thin-fiber readout on outside data, framed as "recovers thin fibers / does no harm";
- the false-positive split between `unlabelled_real` and `other_bg` from our taxonomy;
- the predicted rate on audited abnormal candidates (protocol fidelity);
- LVIS-style federated precision, counting false positives only where exhaustiveness is verified, i.e. on the expert subset if T-20 runs ([LVIS](https://arxiv.org/abs/1908.03195)).

Report official metrics, but never use them as the decision rule.

### Compute plan inside $34

The costs below are our arithmetic. B1 used **6.34 L4-hours for $7.69** over 150 epochs ([06-compute §3](../notes/06-compute-and-reproduce.md)), which is about **$1.21 per L4-hour and about 152 s per epoch**. An nnU-Net epoch is a fixed 250 iterations ([`nnUNetTrainer.py:177`](../../external/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py)), so cost tracks epochs, not dataset size. That holds only if TEM1's plans give a similar patch configuration, which a 5-epoch timing run must confirm.

At the B1 length, $34 buys only four runs. Six to ten runs therefore require shorter training. **80 epochs costs about $4.1 per run.** B1's best epoch was 134 and early stopping never fired, so 80-epoch arms will be undertrained in absolute terms. The comparison stays fair because every arm is trained identically.

| Step | Run | Data | Epochs | Est. cost | Funded by |
|---|---|---|---|---|---|
| 0 | Timing run, TEM1 plans | TEM1 | 5 | ≈ $0.4 | tranche B |
| — | Complete-label reference (= H8 reference) | TEM1 | 150 | ≈ $7.7 | reserve earmark (D-34), not the $34 |
| Pilot (gate) | R1 naive on simulated labels | TEM1 | 80 | ≈ $4.1 | tranche B |
| | R2 hard ignore (D-18 / Gotkowski) | TEM1 | 80 | ≈ $4.1 | tranche B |
| | R3 random-ignore control | TEM1 | 80 | ≈ $4.1 | tranche B |
| | R4 selection-gated soft ignore | TEM1 | 80 | ≈ $4.1 | tranche B |
| Full (only if R4 > R2 and R3) | R5 SCAR ablation (a ≡ 0) | TEM1 | 80 | ≈ $4.1 | tranche C |
| | R6 oracle ignore | TEM1 | 80 | ≈ $4.1 | tranche C |
| | R7 selection-gated, our data, vs existing B1 | challenge dev | 150 | ≈ $7.7 | tranche C |
| | **Total within the $34** | | | **≈ $32.8** | B ≈ $16.8, C ≈ $15.9 |

Three alternatives are open to the owner. A 60-epoch recipe (about $3.1 per run) would fit about nine TEM1 arms, adding LNQ-constant, WDA-style and nnPU baselines, at the cost of heavier undertraining. The roughly $20 of credit left above the $60 ceiling could fund the rank-6 baselines that reviewers will ask for. R7 depends on T-21's out-of-fold probabilities (tranche A); without them, our-data masks are biased and R7 must not run. The selection model, candidate mining, feature extraction and all scoring run locally on the M4, in line with the GPU-only rule for Modal. Every launch keeps the `--max-gpu-hours` cap and the `*_ckpt10` resumable trainers.

## 5. What C3 can claim, and what will sink it

**Can be claimed, within the reach of our searches.** Coverage of the 2025–26 MICCAI, MedIA and TMI proceedings was partial, apart from a full MICCAI 2024–26 title scan.

1. The **first application to segmentation of an instance-level SAR selection model**: a fiber-level labelling propensity that conditions the ignore decision. It must be framed as applying Bekker/Coudray SAR-PU together with LNQ/Gotkowski masking.
2. A **two-component omission model**, random plus protocol, whose random part is identified from **repeat annotation in overlapping tiles**. No such use of overlapping regions was found. The capture–recapture origin must still be cited.
3. The **first appearance-dependent, two-component omission benchmark** in EM segmentation that scores both recovery and protocol fidelity. Every EM study found simulates random omission, and the only non-random simulations elsewhere are size-based (SAMPD, BMVC 2021, ReCo-I2P).
4. **Known-positive and federated evaluation** of an incomplete-label method in EM, combined with the observed-label-illusion rule.

**Cannot be claimed.**

- A new loss. Ignore is nnU-Net's; soft weighting is LNQ's; PU and SAR risks belong to du Plessis, Kiryo, Bekker and Coudray; the doubly robust template is CADR's.
- A new gate. The three-way gate is SAMPD's.
- An abnormal or protocol class or style conditioning. Zepf ICLR 2023, Parikh 2026 and MicroStructFormer 2026 already cover it.
- "MNAR". The mechanism is SAR within positives, plus a zero-propensity stratum. Calling it MNAR invites a correction from any PU reviewer.
- Recovery of deliberately excluded fibers by the protocol-trained model. It is not identifiable, and the outside sets omit degenerating fibers too.

| Risk | Consequence | Mitigation |
|---|---|---|
| Small headroom (e ≈ 0.93; Cannings condition; real-data SAR gains of +0.2–1.8 points in PUe) | Official metrics barely move; H3's ≥ 10 points fails | Pre-register recall on ambiguous fibers as the primary endpoint; report against H3 without editing it |
| a(x) not identifiable without the expert subset (T-20 deferred) | The selection term rests on a "unlike labelled fibers" assumption that reviewers can contest | Run T-20; report a sensitivity sweep over the gate thresholds; state the assumption |
| The TEM1 proxy stratum is not our abnormal set (TEM1 also omits degenerating fibers; domain shift) | The benchmark validates the mechanism, not our data | Fit the proxy two-sided, matching our mixed audit; keep the our-data arm (R7) as confirmation |
| Overlap-tile misses are correlated, or by the same annotator | r is biased upward, so too few candidates are ignored | Ask the organisers (A-questions); sweep r between 0.88 and 0.96 |
| The candidate detector misses thin myelin | The selection model never sees the hardest stratum | Add zero-shot micro-sam or Cellpose-SAM candidates; report detector recall per stratum |
| Undertraining at 80 epochs | Absolute numbers fall below B1 | Compare arms only with each other; keep R7 at full length |
| Observed-label illusion | C3 looks worse on official precision | Decision rules use known positives only (D-34) |
| Pseudo-label batch effects (Bukas) | g-ratio is biased by model-made shapes | No pseudo-positives in the main method; batch-effect check in any ablation |
| Late prior art (SIPAIM entries after 18–20 Nov; 2026 proceedings) | Novelty claims 1–3 weaken | Run the planned novelty recheck before writing |
| The whole route exceeds the credit if baselines grow | Reviewer-demanded baselines go unfunded | Owner decision on the $20 above the ceiling, or the 60-epoch recipe |

## Conclusion

The literature changes what C3 should *be*. Ignoring confident candidates (D-18) is not a contribution; it is nnU-Net's ignore label. Worse, applied blindly it erases the very protocol signal the challenge scores, because it ignores abnormal fibers alongside missed normal ones. The defensible method is to ask, for every unlabelled fiber, whether the annotator missed it or was told to skip it, and to make the answer explicit. The overlap tiles answer the first half (r ≈ 0.93). The expert subset, or a stated assumption, answers the second. That reframing also explains in advance why gains will be small on clean fibers and visible only at the ambiguous boundary of the protocol.

The practical implication is to sequence decisions around identification, not architecture. If T-20's expert subset is not run, C3's selection term rests on an assumption, and the paper becomes mostly a benchmark contribution. If it is run, C3 gains both an anchored propensity model and the only route to a morphometry output covering all fibers (the Shi-style abnormal class). The four-run TEM1 pilot is the cheapest way to learn which of those papers exists, and it should run before any other C3 spending.
