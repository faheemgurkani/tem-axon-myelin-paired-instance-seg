# Training from incomplete or selective object labels in electron microscopy (15 Nov 2026)

**Scope:** EM only (TEM, SEM, FIB-SEM, ssTEM, vEM). Axon and myelin work first, then connectomics and organelles. This note does not repeat the items already in our notes (nnU-Net ignore label, LNQ 2023, Zhao PU, Pototzky BMVC 2021, ReCo-I2P, Vădineanu, Komura, Bukas 2026, MicroStructFormer, Parikh 2026). For those, see `docs/reports/C3 incomplete label learning novelty.md` and `docs/research_notes/Phase 0b evidence/03_c3_unread_papers.md`, `08_bukas2026.md` and `11_overlooked_gaps.md`. They are cited here only where they bear on an EM-specific point.

**Search effort:** about 20 tool calls, made up of local notes, local code in `external/`, 6 web searches and 9 fetches. Read depth for each source is stated inline. Several pages were blocked:
- the bioRxiv full texts (403);
- the ScienceDirect optic-nerve paper (403);
- the Nature Methods LSD paper (login redirect).

For those, only search snippets were used, and this is marked where it applies.

## Q1. Axon and myelin EM: does anyone train with incomplete fiber labels or ignore regions, or model abnormal fibers?

### Takeaway
No axon or myelin EM paper or code base we found trains with ignore regions, positive-unlabelled losses or any other correction for unlabelled fibers. They all treat unlabelled pixels as background. The one recent in-domain paper that states a selection rule (optic-nerve TEM, 2026) leaves degenerated profiles out of annotation **and** training, which is the same selective protocol as ours, and makes no correction for it. Only MicroStructFormer (preprint) models abnormal myelin explicitly, and it does so as an extra class, not as a missing-label correction.

### Cited Findings
- **AxonPath (code, local copy).**
  - Its training loss is plain DiceCE on a 3-class semantic map (bg / myelin / inner cylinder) plus MSE on fibre and axon signed distance maps: `sem_loss + mse1 + mse2`.
  - There is no ignore index, no loss mask and no weighting of unlabelled regions. Its only `valid_mask` restricts the watershed at inference.
  - Source: [external/axonpath/axonpath/utils/losses.py](../../../external/axonpath/axonpath/utils/losses.py) and `axonpath/inference/post_processing.py`.
- **ASTIH (code, local copy).** The repo has only data download, model application and evaluation scripts. The single "ignore" hit is a pandas `ignore_index`. Nothing handles missing labels.
  - Our TEM2 completeness check found that the ASTIH paper (MIDL 2026) reports **no completeness audit** and **no inclusion rule for abnormal profiles**.
  - TEM1 labels came from a threshold + connected-component macro that kept only full myelin rings. TEM2 labels are corrected AxonDeepSeg pseudo-labels.
  - Sources: [external/ASTIH/scripts/](../../../external/ASTIH/scripts/) and [TEM2 label completeness check](../../reports/TEM2%20label%20completeness%20check.md).
- **AxonDeepSeg (code, local copy).** No ignore or partial-label logic in the package. The "partial" hits concern only morphometrics, where a bounding box touching the border flags a "partial axonmyelin object". Source: [external/axondeepseg/AxonDeepSeg/morphometrics/compute_morphometrics.py](../../../external/axondeepseg/AxonDeepSeg/morphometrics/compute_morphometrics.py).
- **Ma, Hao, Li, … Shyu, Lee, "Deep Learning–Driven TEM Analysis of Murine Optic Nerve Myelinated Axons", *Ophthalmology Science* 6(5), March 2026 (peer-reviewed journal).**
  - Data: nnU-Net on 368 optic-nerve TEM images from 23 mice.
  - F1: 0.771 for inner axons and 0.697 for outer myelinated fibers. Outputs include g-ratio.
  - Source: [U. Miami repository record](https://scholarship.miami.edu/esploro/outputs/journalArticle/Deep-LearningDriven-Transmission-Electron-Microscopy-Analysis/991033019968402976).
  - According to the search snippet, **"degenerated profiles were excluded from manual annotation and model training"**, and post-processing removes incomplete axons. Source: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666914526000795).
  - *Only the abstract and snippet were read; the full text returned 403.* So we cannot say whether the exclusion is done by masking (ignore) or by leaving the profiles as background.
- **gACSON** (Abdollahzadeh, Sierra group) is **unsupervised** 3D myelinated-axon segmentation: it "requires no manually annotated training data". So label completeness does not arise. Source: [arXiv 2112.06476](https://arxiv.org/html/2112.06476) (snippet only).
- **Volume EM of oligodendrocyte axon selection** (*iScience*, 2026, based on the S2589-0042 ISSN prefix): axons were hand-segmented "every 10 slices" and then interpolated. This is slice-sparse annotation used to build ground truth, not a learning method. Source: [ScienceDirect S2589004226031548](https://www.sciencedirect.com/science/article/pii/S2589004226031548) (snippet only).
- **MicroStructFormer** (Deng … Khadra, bioRxiv May 2026, **preprint**; from our notes) segments an EM **"abnormality" class** (uncompacted membrane), assigned per fiber by geodesic growth. It reports per-fiber "inclusive" and "exclusive" measures, and uses iterative model-assisted labels. Sources: [Overlooked gaps validation](../../reports/Overlooked%20gaps%20validation.md) and [TEM axon myelin papers 2025 2026](../../reports/TEM%20axon%20myelin%20papers%202025%202026.md).
- **SnapG** (Bekku … Salzer, bioRxiv Jan 2026, **preprint**; from our notes) detected **126% of the manually counted** CNS axons. This is direct in-domain evidence that manual counts are selective. Source: [TEM axon myelin papers 2025 2026](../../reports/TEM%20axon%20myelin%20papers%202025%202026.md).

### Inferences
- **In-domain, C3 is the first to treat unlabelled fibers as anything other than background during training,** as far as this search reaches. The optic-nerve paper shows that the selective protocol (degenerated profiles out) is standard practice in the field, and that it is passed into training without any correction. That is a direct "problem exists, nobody corrects it" citation for C3's motivation.
- Explicit abnormal-fiber modelling exists only as a *class* (MicroStructFormer). There it is a supervised label, not inferred from missingness. A C3 variant with a "protocol-excluded fiber" class would be a design option close to MicroStructFormer's, not a new idea in itself.

### Gaps
- DeepACSON (Abdollahzadeh et al., *Commun Biol* 2021), EM-SAM, Ruthig and other optic-nerve TEM tools: their training-label handling was not checked in this run. DeepACSON trained on automatically generated labels from low-resolution data, which is a noisy-label setting, not a missing-label one; that is from prior knowledge, not re-verified here.
- The optic-nerve paper's exact handling of degenerated profiles (ignore or background) is unknown. Its full text was blocked.

## Q2. Connectomics and organelle EM: which mechanisms treat "unlabelled" as different from "background"?

### Takeaway
Four EM mechanisms exist:
1. **loss masking** to labelled regions (LSD and the sparse-to-dense LSD work, nnU-Net ignore);
2. **object-level positive-unlabelled embedding training** with consistency on the unlabelled region (SPOCO, CVPR 2022: the origin technique and the strongest EM precedent);
3. **confidence-masked pseudo-labels plus a "known-negative" weight map** (WDA-Net, preprint);
4. **teacher–student pseudo-labelling** on annotator-partial instances (SomaNet, preprint Sep 2026). SomaNet still sets unlabelled voxels to background in its supervised loss.

In all of them the missingness is **random or not analysed**. None ties omission to object appearance.

### Cited Findings
- **SPOCO: Wolny, Yu, Pape, Kreshuk, "Sparse object-level supervision for instance segmentation with pixel embeddings", CVPR 2022.**
  - Venue: the arXiv comments say "CVPR 2022 camera-ready". *Methods and experiments read in full (arXiv HTML v3).*
  - Code: [github.com/kreshuklab/spoco](https://github.com/kreshuklab/spoco). Source: [arXiv 2103.14572](https://arxiv.org/html/2103.14572v3).
  - **Setting:** "mask annotations for a subset of instances in the image, leaving the rest of the pixels unlabeled". The background is **also** unlabelled, so this is explicitly positive-unlabelled.
  - **Head:** non-spatial pixel embeddings, with differentiable instance extraction. Given an anchor embedding a_k, the soft mask is S_k = exp(−‖e_i − a_k‖²/2σ²), with σ² = −δ_v²/(2 ln t), t = 0.9.
  - **Losses on labelled objects:**
    - pull: L_pull = (1/C)Σ_k(1/N_k)Σ_i[‖μ_k − e_i‖ − δ_v]₊²;
    - push: L_push = (1/C²)Σ_kΣ_l[2δ_d − ‖μ_k − μ_l‖]₊², with δ_v = 0.5 and δ_d = 2.0;
    - instance Dice: L_obj = (1/C)Σ_k D(S_k, I_k).
  - **Losses on unlabelled pixels:**
    - unlabelled push, L_U_push = (1/C)Σ_k(1/N_U)Σ_i[δ_d − ‖μ_k − e_i‖]₊², which pushes labelled clusters away from unlabelled pixels;
    - consistency, L_U_con = (1/K)Σ_k D(S^f_k, S^g_k), using anchors sampled in the unlabelled region and two augmented views. g is an EMA teacher (m = 0.999).
    - Total: L = L_SO + L_U_push + L_U_con, with all weights 1.
  - **EM experiment:** mitochondria, with VNC (Drosophila ssTEM) as source and **MitoEM-R** as target. Sparsity is **simulated by random sampling** of 1%, 5% or 10% of instances.
    - AP@0.5 / mAP, MitoEM only: 0.088/0.046 at 1%, 0.403/0.280 at 5%, 0.481/0.340 at 10%.
    - Fully supervised SPOCO: 0.560/0.429.
    - Removing the consistency loss lowers AP to about 0.30–0.32 (fine-tuning setting).
    - Background is never sampled as an object in the weak setting.
- **WDA-Net (weakly supervised domain adaptation for EM mitochondria with sparse points).** [arXiv 2404.00667](https://arxiv.org/html/2404.00667), **preprint**. It describes itself as an extension of a BIBM 2022 conference paper; no journal venue is stated. *Method read (HTML).*
  - **Labels:** centre points on a **"randomly-sampled"** 15% of target mitochondria.
  - **Unlabelled handling:**
    1. Segmentation pseudo-labels are kept for the top 80% most confident pixels by normalised entropy. Other unlabelled pixels are **ignored in backprop**.
    2. **Centre-detection weight map:** weight is zero everywhere except at labelled regions and pixels predicted as background with high confidence (p < 0.1). This is a "known positive + confident negative" mask.
    3. A **counting prior:** a source-trained counter estimates the total count. A soft margin loss (ε = 3) pulls the heatmap's count towards it.
    4. Cut-and-paste of labelled points.
  - **Data:** EPFL FIB-SEM, Drosophila VNC ssTEM, Kasthuri++ and MitoEM-R.
  - **Dice with 15% points vs fully supervised:** 90.7 vs 93.6, 93.2 vs 94.6 and 90.1 vs 93.6. Labelling took about 15 min for the points vs about 323 min for full masks.
- **SomaNet: Khateri, Ghahremani, Tohka, Sierra, "Weakly Supervised Learning for Instance Soma Segmentation in 3D EM with Partial Annotations"**, arXiv 2609.23019, 19 Sep 2026, **preprint**. *Methods and results read (HTML v1).* Code: [github.com/mkhateri/SomaNet](https://github.com/mkhateri/SomaNet). Source: [arXiv HTML](https://arxiv.org/html/2609.23019v1).
  - **Data:** EMADS, adult Drosophila ssTEM at 16×16×40 nm³. Ten blocks are "partially annotated at the voxel level", with 204 somas (about 20 per block). The partiality is **natural**, not simulated, but how the subset was chosen is not described.
  - **Unlabelled handling:**
    - The supervised mask loss is averaged over all voxels **with target 0 for unannotated voxels**, i.e. treated as background. The appendix acknowledges the resulting false-negative bias.
    - The affinity MSE (λ = 0.005) treats unannotated voxels the same way.
    - Correction comes only from an EMA teacher (α = 0.999). Its foreground is thresholded at τ = 0.5 into pseudo-labels.
    - A consistency CE over K augmented views (λ_u = 0.5), plus feature distillation (λ = 0.1).
  - **Results:** evaluated on two "nearly dense" held-out blocks. PQ is 0.740 (DINOv2 backbone) and 0.818 (SwinIR), vs 0.603/0.659 for the EMADS baselines.
  - There is **no ablation without pseudo-labels and no nnU-Net baseline**.
  - Relevance: it comes from the gACSON and DeepACSON group, so it is the closest same-lab precedent in EM white matter.
- **Local Shape Descriptors (Sheridan et al.; bioRxiv 2021.01.18.427039, later *Nature Methods* 2023, venue from prior knowledge):**
  - "Data was randomly sampled … using a labels mask to ensure every batch contained at least 50% of ground-truth data".
  - "segmentations were restricted to a foreground mask generated from the ground truth", because some GT was incomplete.
  - Source: [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.01.18.427039v2.full) and [LSD project page](https://localshapedescriptors.github.io/). *Snippet only: the full text returned 403 and Nature Methods redirected to a login.*
- **Sparse-to-dense 3D from sparsely annotated 2D (bioRxiv 2024.06.14.599135, preprint, LSD-based):** "Unannotated areas were excluded when computing the loss", using a weighted MSE on LSDs and affinities. Sparse annotations were placed near object boundaries and inside objects. Source: [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v1.full) (snippet only; 403). Authors and later venue not confirmed.
- **micro-sam (Archit et al.; venue not re-verified in this run).** The FAQ says:
  - "it's okay to use sparse segmentations (i.e. few objects per image are annotated)" when fine-tuning SAM without the decoder;
  - but "it's expected to provide dense segmentations (i.e. all objects per image are annotated)" for the instance-segmentation (AIS) decoder.
  - The mechanism is prompt sampling from labelled objects only. Unlabelled objects are simply never prompted, so they create no negative gradient.
  - Source: [micro-sam docs](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html).
  - Local training code crops Platynereis EM ROIs to `labels != ignore_label`, i.e. the annotated extent. Source: [obtain_boundaries_em_datasets.py](../../../external/micro-sam/finetuning/generalists/training/electron_microscopy/boundaries/obtain_boundaries_em_datasets.py).
- **ε-Seg (Rahnamai Kordasiabi, Dalle Nogare, Jug):** labels on "0.05% of the total image data or less" in two dense EM datasets. It uses masked-centre inpainting, sparse-label contrastive learning, a GMM latent prior and an MLP head. Pixel-sparse, not object-selective. Source: [arXiv 2510.18637](https://arxiv.org/abs/2510.18637). **Preprint; abstract only.**
- **EM survey (Aswath et al., *Medical Image Analysis* 2023):** notes a study where "all unlabeled mitochondria and other unlabeled structures were treated as background" (the default assumption), and predicts more semi- and self-supervised learning. Source: [ScienceDirect S1361841523001809](https://www.sciencedirect.com/science/article/pii/S1361841523001809) (snippet only).
- **MitoNet / CEM-MitoLab (Conrad & Narayan; bioRxiv 2022, later in a journal, PMC9883049):** built from existing labels plus crowdsourced annotations. Whether images were partially annotated, and how that was handled, could **not** be confirmed from the snippets. Source: [PMC9883049](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883049/).
- **Vădineanu et al., *J Imaging* 2024 (from our notes):** uses EPFL hippocampus FIB-SEM mitochondria with **random** instance omission at 30–70%. An "upgrade network" repairs the labels. Source: [03_c3_unread_papers.md](../Phase%200b%20evidence/03_c3_unread_papers.md).
- **Bukas et al., MICCAI-W 2026 (from our notes):** in rat-liver mitochondria TEM, model-seeded labels form provenance-dependent shape clusters, and models learn the boundary style. Source: [08_bukas2026.md](../Phase%200b%20evidence/08_bukas2026.md).

### Inferences
- The EM field already has three plug-in answers to "unlabelled ≠ background":
  - **(a) hard masking**: LSD, sparse-to-dense, nnU-Net ignore; C3's primary arm;
  - **(b) known-negative masking**: WDA-Net's p < 0.1 background map. This is a direct, cheap C3 baseline: weight 1 on labelled fibers and on confident background, 0 elsewhere;
  - **(c) teacher pseudo-labels**: SomaNet, WDA-Net, and Pototzky outside EM.
- **SPOCO is the method-level EM precedent for object-level PU instance segmentation.** It needs an embedding head, so it is not a drop-in for our nnU-Net semantic + watershed pipeline. Its useful idea is **consistency on unlabelled regions** under an EMA teacher, which can be added to nnU-Net as a regulariser on ignored pixels.
- **SomaNet is a cautionary precedent.** It still uses unlabelled = 0 in its supervised loss and relies on pseudo-labels to recover. With no ablation, its gain cannot be attributed to the pseudo-labels, and it does not separate deliberate from accidental omission.
- **Mapping to nested fibers.** For our semantic classes (axon / myelin / bg / ignore), any of (a)–(c) must act on **whole fibers**: axon and ring together (X ∪ X+1). Otherwise a ring candidate could be half-ignored, half-background, and the watershed pairing would learn inconsistent topology. None of the EM papers deals with nested objects.

### Gaps
- CREMI and SNEMI3D sparse skeleton training, flood-filling networks (FFN) with sparse GT, Synful (synapses) and CEM500K were **not read in this run**. CEM500K is self-supervised pretraining on unlabelled data, which is a different problem. FFN and LSD training are, as far as known, done on dense subvolumes with masks, but this was not re-verified.
- No 2025–26 MICCAI, ISBI or TMI EM paper on PU or partial-instance learning was found. Coverage of the 2026 proceedings via web search is incomplete.
- The MExConn preprint (bioRxiv 2025.11.19.689394, multi-organelle connectomics) appeared in results, but its label handling was not checked.

## Q3. Does any EM paper model *why* objects are unlabelled, or evaluate on known-positive held-out objects?

### Takeaway
No. Every EM study of incomplete labels we found does one of two things:
- **simulates random omission**: SPOCO (random instance fractions), WDA-Net (random 15%), Vădineanu (random 30–70%);
- **uses an unanalysed, naturally partial set**: SomaNet's EMADS, LSD's incomplete GT.

They all evaluate on dense or near-dense test volumes. None models appearance-dependent omission. None scores recovery on objects known to be real but unlabelled in the training-protocol sense. The only in-domain hints of selection bias are descriptive: the optic-nerve exclusion of degenerated profiles, and SnapG's 126% count.

### Cited Findings
- SPOCO: "randomly sampling a fraction of ground-truth instances", 1/5/10% on MitoEM-R. Evaluated against full MitoEM GT — [arXiv 2103.14572](https://arxiv.org/html/2103.14572v3).
- WDA-Net: "randomly-sampled" 15% point subsets. Evaluated with Dice against full masks — [arXiv 2404.00667](https://arxiv.org/html/2404.00667).
- SomaNet: partial labels from the dataset itself, selection rule not described. Evaluated on "nearly dense" held-out blocks. Somas touching volume boundaries excluded from GT — [arXiv 2609.23019](https://arxiv.org/html/2609.23019v1).
- Vădineanu 2024: random instance deletion on FIB-SEM mitochondria — [03_c3_unread_papers.md](../Phase%200b%20evidence/03_c3_unread_papers.md).
- Optic-nerve TEM 2026: degenerated profiles excluded "from manual annotation and model training". No analysis of the effect — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666914526000795) (snippet).
- Outside EM, the only appearance-tied omission evidence in our notes is Komura (*Patterns* 2023, histology, a labelling remedy only) and the size-based simulated omission of Pototzky 2021 and ReCo-I2P 2025 — [C3 novelty report](../../reports/C3%20incomplete%20label%20learning%20novelty.md).

### Inferences
- C3's two claims survive the EM-specific search intact:
  1. characterising **MNAR, appearance-dependent omission** in EM;
  2. a **known-positive evaluation**: overlap-disagreement fibers, plus the outside thin-myelin sets under H8.

  The mechanism claim does not survive (see the earlier report).
- **The identifiability problem is unique to our setting and unaddressed anywhere.** A missed normal fiber and a deliberately excluded abnormal fiber are both 0 in the mask. Every EM method above assumes all unlabelled objects are positives to recover (SPOCO, WDA-Net, SomaNet pseudo-labels). Applied naively to our data, each would learn to predict protocol-excluded fibers and lower official precision. To separate the two kinds of missingness, a selection model is needed: an appearance or abnormality score per candidate. That is the potentially novel *method* element, if C3 wants one beyond evaluation.

### Gaps
- Whether any 2025–26 EM paper estimates a class prior or selection probability per object could not be ruled out with certainty. Searches for PU or MNAR in EM returned nothing.

## Q4. Evaluation designs for incomplete labels in EM

### Takeaway
EM papers evaluate in two ways:
- **(i) dense or near-dense held-out test sets**, while training on simulated random subsets (SPOCO, WDA-Net, Vădineanu) or naturally partial data (SomaNet);
- **(ii) masking evaluation to the annotated foreground**: LSD restricts segmentations to a foreground mask from GT.

None uses the train-label-biased test set *and* a separate known-positive set, which is what C3 needs because our test labels follow the same selective protocol.

### Cited Findings
- **Simulated sparsity curves:** SPOCO reports AP at 1/5/10% of instances against the full label — [arXiv 2103.14572](https://arxiv.org/html/2103.14572v3). WDA-Net compares 15% points vs full supervision — [arXiv 2404.00667](https://arxiv.org/html/2404.00667).
- **Ablations:** SPOCO removes its consistency loss (AP drops to about 0.30). SomaNet has no component ablation — [arXiv 2609.23019](https://arxiv.org/html/2609.23019v1).
- **Evaluation masked to GT:** LSD restricted its segmentations "to a foreground mask generated from the ground truth" — [LSD bioRxiv](https://www.biorxiv.org/content/10.1101/2021.01.18.427039v2.full) (snippet).
- **Metrics:** SomaNet uses Dice, IoU, PQ, ARAND and VOI. SPOCO uses AP@0.5 and mAP. WDA-Net uses Dice. No paper reports a precision split by "unlabelled-but-real" vs "true FP".
- The "observed-label illusion" (Parikh et al. 2026, arXiv, **preprint**; from our notes): a correction looks like a failure when scored on the same biased labels — [C3 novelty report / 03 notes](../Phase%200b%20evidence/03_c3_unread_papers.md).

### Inferences
- C3's evaluation should borrow the **simulated-omission curve** from SPOCO and WDA-Net as a *control*: delete random fibers from our labels at 5–20% and check that the ignore or pseudo-label arm recovers them. Our real omission is selective, so the control shows **how much harder** MNAR omission is than the random omission the literature tests. This contrast is itself a publishable result.
- Pair it with a **selective-omission simulation**: delete fibers by a fiber-level appearance score (thin myelin / high g, oblique). This turns the BMVC 2021 and ReCo-I2P size-based design into an appearance-based one in EM.
- Report precision split into `unlabelled_real` vs `other_bg` (our taxonomy). No EM paper does this.

### Gaps
- We found no EM dataset with an explicitly *exhaustive* relabel of a selectively annotated set that could serve as an external known-positive benchmark. ASTIH TEM1 and TEM2 completeness is unaudited (TEM2 completeness report).

## Q5. Method details and how each could be adapted to nested axon–myelin instances

### Takeaway
The cheapest in-domain arms to add to C3 are:
- nnU-Net ignore (primary);
- WDA-Net-style "known positive + confident negative" masking;
- a SPOCO- or SomaNet-style EMA-teacher consistency term on ignored pixels.

SPOCO's embedding PU loss needs a new head. That means an instance-embedding model rather than the B1 pipeline, so it is a secondary comparator at best.

### Cited Findings (method → adaptation)

| Method (venue, status) | Unlabelled treatment | Reported gain | Code | Adaptation to our nested fibers |
|---|---|---|---|---|
| nnU-Net ignore label (Gotkowski 2024, arXiv; from our notes) | Loss only on annotated pixels | (see earlier report) | nnU-Net | Mark candidate fiber (axon ∪ ring) as label 1 (ignore) in the training mask. Already our encoding. **Primary C3 arm.** |
| LSD / sparse-to-dense (Nature Methods 2023 / bioRxiv 2024 preprint) | Labels-mask sampling (≥ 50% labelled per batch); loss masked to annotated area | Not quantified for masking alone | [LSD](https://localshapedescriptors.github.io/) | Patch sampler: oversample patches with few ignored candidates. nnU-Net's ignore already excludes them from the loss. |
| SPOCO ([CVPR 2022](https://arxiv.org/html/2103.14572v3)) | Unlabelled = unknown (including background); push labelled clusters away from unlabelled pixels; EMA-teacher consistency on unlabelled anchors | MitoEM-R 10% instances: AP 0.481 vs 0.560 full; without consistency about 0.30 | [spoco](https://github.com/kreshuklab/spoco) | Needs an embedding head and full-fiber instances. Taken alone, the consistency idea can regularise ignored candidate pixels in nnU-Net (two augmentations, EMA teacher, Dice between soft predictions). The unlabelled push term would push abnormal fibers *towards* "not like labelled fibers", which is the direction the protocol wants, but it also pushes missed normals away. **Unsafe for MNAR without a selection model.** |
| WDA-Net ([arXiv 2404.00667](https://arxiv.org/html/2404.00667), preprint; BIBM 2022 extension) | Pixel weight = 1 on labelled + confident background (p < 0.1), 0 elsewhere; top-80% entropy pseudo-labels; counting prior | 15% points: Dice 90.1–93.2 vs 93.6–94.6 full | Not checked | **Direct baseline:** weight 1 on labelled fibers and on pixels an out-of-fold model calls background with p(fiber) < 0.1; ignore the rest. It cannot tell excluded abnormal fibers from missed ones (both are uncertain), so it ignores both. A **count prior** per image (e.g. from ring-candidate counts) is a novel-for-fibers twist. |
| SomaNet ([arXiv 2609.23019](https://arxiv.org/html/2609.23019v1), preprint Sep 2026) | Unlabelled = 0 in supervised loss; EMA teacher (α = 0.999) pseudo-labels at τ = 0.5; multi-view consistency (λ_u = 0.5) + feature distillation | PQ 0.74–0.82 vs 0.60–0.66 baselines; no ablation | [SomaNet](https://github.com/mkhateri/SomaNet) | Teacher-student arm. Pseudo-labels must be whole-fiber and filtered by an abnormality score, or the student will learn protocol-excluded fibers. Mixes model and human boundary styles (Bukas risk), so harmonise or stratify morphometry by label provenance. |
| micro-sam fine-tuning (FAQ) | Only labelled objects are prompted, so unlabelled ones give no gradient; AIS decoder needs dense labels | n/a | [micro-sam](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html) | Prompt-only fine-tuning is naturally robust to missing fibers but gives no automatic instance output. The decoder path would need the same ignore treatment, which it does not support per the FAQ. |
| ε-Seg ([arXiv 2510.18637](https://arxiv.org/abs/2510.18637), preprint) | Pixel-sparse; inpainting + contrastive + GMM prior | Not given in abstract | Not linked | Not object-selective, so low relevance. |

### Inferences
- **Recommended C3 baseline set, adding EM precedents to the earlier report's list:**
  1. nnU-Net hard ignore on candidates (Gotkowski);
  2. random-ignore control with the same pixel budget;
  3. WDA-Net-style confident-negative masking (ignore every uncertain pixel, no candidate miner);
  4. optional EMA-teacher consistency on ignored pixels (SPOCO/SomaNet);
  5. LNQ soft weight and PU, as in the earlier report.

  Arm 3 is important. It tests whether our *ring-evidence* candidate miner adds anything over generic uncertainty masking, which is the obvious reviewer question from the EM literature.
- **The candidate novelty is a selection-aware mask.** Split unlabelled candidates into likely-missed-normal (ignore or pseudo-positive) and likely-abnormal (keep as background, or a separate class à la MicroStructFormer), using a fiber-level appearance score. We found no EM or natural-image method that conditions the ignore or pseudo-label decision on a learned P(omitted | appearance). This is the in-domain method contribution, if one is wanted, and it should be framed as applying MNAR selection modelling (BiasCorr-style, tabular origin) to instance masks.
- **Expectation setting:** EM sparse-label methods close most but not all of the gap to full supervision (SPOCO about 86% of full AP at 10% instances; WDA-Net 96–99% of full Dice). Our labels are about 92–96% complete, so the headroom on *random* misses is small. Most of any C3 effect will show on the known-positive set, not on official metrics. This agrees with the earlier report's "modest gains" warning about H3.

### Gaps
- Code-level checks of WDA-Net and SomaNet loss masking were not done (only paper text was read). SomaNet's Eq. 8 consistency term appears to sum over all voxels, but whether it is restricted to unannotated voxels is unclear from the text.
- Not verified here: the peer-reviewed venues for LSD (Nature Methods 2023), micro-sam (Nature Methods 2025) and MitoNet (Cell Systems 2023). They come from prior knowledge and should be confirmed before citing.
