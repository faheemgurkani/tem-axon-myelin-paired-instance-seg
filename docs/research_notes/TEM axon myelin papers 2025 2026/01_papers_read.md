# Papers read (full text, browser), 13 Nov 2026

## Ma, Hao, ... Shyu, Lee. "Deep Learning-Driven TEM Analysis of Murine Optic Nerve Myelinated Axons". Ophthalmology Science 6(5):101141, 3 Mar 2026. doi:10.1016/j.xops.2026.101141 (PMC13091109, OA). ARVO 2025 abstract precedes it.
- Data: 368 TEM images, 23 mice (C57BL/6J, DBA/1J), 4 conditions incl. silicone-oil ocular hypertension; JEOL JEM-1400 at **x3000** (same magnification as ours). Split 50/25/25 **by image** (no animal-level split stated -> likely leakage across animals' images).
- Annotation: NeuronJ traces of inner (axon) and outer (fiber) circumferences, binary masks; only complete fibers; **degenerated profiles excluded from GT** (same "selective annotation" issue as our data).
- Model: nnU-Net "conceptualised as" 1 encoder + 2 decoders (axon: Dice+BCE; fiber/boundary: boundary-weighted Dice). Claims fewer merges than single decoder, no numbers. Post-processing: soft threshold, CC filtering, morphology, border pruning. Compared U-Net, Swin-UNet, DeepLabV3, ConvNeXt-UNet (Table S1).
- Metrics: **pixel-level** precision/recall/F1 (inner F1 0.771, outer 0.697 test; text says 0.763/0.691 - internal inconsistency). No instance/fiber detection metric, no pairing metric.
- Morphometry: d, D = **mean centroid-to-boundary distance** (not equivalent circle); R2 vs 3 graders >0.96 (d, D, area), 0.915 thickness, **0.805 g**; only 5 representative axons in the grader comparison (Tables S2-S5) - very small morphometry validation.
- Code + subset of data: github.com/umddm/tem-axon-quantification.
- Relevance: closest published journal work. Supports our C1 points: (i) field reports pixel metrics only; (ii) g is the weakest-reproduced quantity; (iii) g-definition differs from the challenge's (equivalent circle) -> our g-definition-sensitivity analysis matters. Pairing still implicit (connected components of two masks).

## Ahmad, Chowdhury, Boullerne, Gow, Feinstein. "MyeliMetric: A Python-Based Toolbox for Standardized G-Ratio Analysis". ASN Neuro 18(1):2603411, 23 Dec 2025 (bioRxiv Jul 2025). PMC12818378.
- Post-segmentation statistics only (Excel in: axon diameter + myelin thickness). Filters axon < 0.15 um, myelin < 0.03 um, g outside [0.5, 0.9]; 6 diameter bins from control distribution; grand-mean g; Shapiro-Wilk; 2-way ANOVA; warns against **pseudoreplication** (fibers are not independent: cluster by animal) and against interpreting g-vs-diameter regression slopes.
- Cites Gow 2025 (ASN Neuro 17:2445624, "Understanding the myelin g ratio from first principles... artifacts") and Gow et al. 2025 (ASN Neuro 17:2447336, axomyelin unit model) - **to read**: they define the biologically correct way to summarise g, relevant to our g-MAE interpretation and the gaming curve.
- Relevance: supports cluster-level statistics (we bootstrap over overlap groups; animal IDs unknown). Its hard filters would drop real remyelinating fibers (our data has g up to 0.94) - selection again changes g summaries.

## Deng, Miao, Bagheri, Peterson, Khadra. "Deep learning-based decoding of axonal ultrastructure in gene-edited mice using EM imaging" (MicroStructFormer). bioRxiv 10.64898/2026.05.26.727755, May 2026 (preprint; McGill).
- Spinal cord EM, Mbp enhancer-edited lines (graded hypomyelination), 3 tracts x 3 ages.
- Mask2Former (Swin-L, ADE20K init, EM pre-training) at **384x384 resized input** for organelles/PAS/abnormal myelin; **AxonDeepSeg** for axon/myelin; outputs merged.
- Instances by rules: myelin-dilated barrier + distance-max seeds -> marker watershed for axons; **geodesic assignment** of myelin/PAS/abnormal to nearest axon with radius scaling; mitochondria by contact.
- g from equivalent-circle diameters (same as challenge); "exclusive" vs "inclusive" fiber (PAS/abnormal in or out) - another g-definition switch.
- Evaluation: mIoU (weighted) only; no fiber-level detection/pairing metrics; admits reduced sensitivity on abnormal myelin in severe KOs.
- Cites "**Mysh**" (ref 37): "improved per-fiber segmentation accuracy through cascaded architectures" - new lead, to check.
- Data/code "being organised for public release".
- Relevance: another rule-based pairing; reinforces that per-fiber pairing quality is never measured.

## Bekku, Cai, Xin, Sall, Kestel, Liang, Salzer. "SnapG: An Automated Tool for Myelin g-ratio Measurement with Batch Processing". bioRxiv 10.64898/2026.01.14.699516, 15 Jan 2026 (preprint; NYU). Code github.com/KK201431873/SnapG.
- Classical OpenCV: blur, global threshold, morphology, contours; filters edge-touching, area, convexity, circularity >= 0.6. Axon = inner contour; myelin thickness = user-chosen **percentile (30-70%)** of distances from the inner contour to the nearest outside boundary pixels; r = perimeter/2pi; g = r/(r + t). A **fourth g definition** (cf. equivalent circle, centroid mean radius).
- Validation: mean g only (PNS 201 axons: 0.675 vs manual 0.669; optic nerve 0.757 vs 0.748); CC P90 1 mouse, 26 FOVs (~6.8 um FOV, similar to ours). **Detected 126% of manually counted CNS axons** (273 vs 245) - annotator selection vs automated detection again; **24.6% of CNS axons with periaxonal space had to be traced manually** (our data folds tongue/PAS into axon by convention).
- In dense CNS the percentile picks up neighbouring sheaths (touching fibers = grouping problem; they lower the percentile to ~30% as a workaround).
- Relevance: (i) g-definition and inner-boundary convention are lab-specific -> supports our g-definition sensitivity analysis (T-16'); (ii) "counts disagree with manual by 26%" mirrors our selection finding; (iii) no per-fiber agreement, only means.

## Fields & Dutta. "G-Ratio Commentary - Why You've Been Doing It Wrong". ASN Neuro 17(1):2486962, 7 Apr 2025 (PMC12140491) - commentary on Gow 2025 (ASN Neuro 17:2445624) and Gow et al. 2025 (17:2447336).
- Gow: the common **positive slope of g vs axon diameter is mostly a measurement + sampling artifact** (unequal sampling across sizes, resolution-limited errors that differ by caliber, rank-order mismatch between axon and fiber diameters). Plotting **g vs total fiber diameter** gives ~zero slope, smaller residuals, intercept through origin ("axomyelin unit": myelin thickness proportional to fiber diameter).
- Pseudoreplication: per-axon n inflates significance; use per-field (or per-animal) means as n=1. Tissue shrinkage up to 65% during preparation.
- **Action for us:** notes/02 §2 calls "g rises with axon size (r = 0.65) ... textbook pattern" - per Gow this is partly an artifact; rephrase, and bin by **fiber** diameter (not axon) in T-16' error-by-size analyses. Our fiber-level bootstrap over overlap groups is the right instinct (no animal IDs available).

## Spalinska, Kopka, Kopka, Pascal, Spalinski, Jasionowska-Skop, Przelaskowski. "Computational Morphometry of Peripheral Nerves: A Pipeline Perspective on Reproducibility and Generalization". Neuroinformatics 24:36, 10 Jun 2026 (review). doi:10.1007/s12021-026-09777-2. Read via institutional access.
- Thesis: morphometric validity is **"pipeline-conditional"** (acquisition, preparation, annotation, segmentation, metric extraction), not segmentation accuracy alone. Peripheral nerve focus, but framing transfers to CC TEM.
- Ground-truth ambiguity: **Kotyk et al. 2025** - manual ICC >= 0.9 for diameters in healthy nerve, but **ICC 0.58 for g-ratio in pathological tissue**; also proposes a geometric-center-based diameter for irregular fibers (yet another g definition). -> to find.
- Annotation strategy varies "from exhaustive segmentation of all visible fibers to selective labeling" - names the selective-annotation issue but offers no method for it.
- Deep models agree on population means; errors larger for per-fiber diameter/thickness; Moiseev 2019 needed manual correction of 20-30% of fibers.
- Orfahli 2021: strong method correlations (rs 0.91) yet systematic count overestimation -> "agreement != equivalence" (supports Bland-Altman in T-16').
- Calls for **benchmarks that assess stability of derived morphometrics and expected expert-agreement ranges** - exactly C1 (human ceilings from overlaps + decomposition). Mentions Ono et al. 2024 unified instance-segmentation pipeline for human nerve biopsies; Ait-Ahmad 2024 (axons in tumour microenvironments).
- Authors' own instance pipeline: Spalinski et al., AIME 2025 (LNCS, pp. 389-394): LM peripheral nerve, U-Net + connected-component labelling (rule-based). Not EM.

## Chuter, Kim, ... Jablonski. "A Systematic Review and Independent Benchmarking of Automated Nerve Morphometry Methods". bioRxiv 10.64898/2026.06.15.731646 v2, 31 Aug 2026 (PMC13307995 abstract only; full text not archived in PMC).
- PRISMA review to Jan 2026: 71 studies (1999-2026), 38% deep learning (>55% after 2020); 73% report axon counts, **only 35% report g-ratio**.
- Benchmarked 18 tools (8 DL, 10 classical) on PPD-stained **light-microscopy** mouse (85) and rat (44) optic-nerve images, counts only: best MAPE 32.9% (a 2010 classical tool); top 5 statistically indistinguishable; rankings flip across datasets (AxonJ 27.7% rat vs 438.6% mouse).
- Relevance: independent benchmarking shows **dataset-dependent rank instability** -> supports H4 (rank stability on ASTIH TEM1) and the C1 claim that per-dataset leaderboards are fragile. Not EM, counts only.

## Bukas, Engler, Pelin, Hagenberg, Khan, Campi, Eberhagen, Zischka, Piraud. "AI-assisted labeling and its pitfalls: A case study in Electron Microscopy segmentation". MICCAI 2026 Workshops (HAIC), LNCS 17279. Abstract read; PDF https://papers.miccai.org/miccai-2026-sat/paper/HAIC26_001.pdf (browser viewer did not render; not downloaded). OpenReview ZJzezbObr1.
- TEM dataset annotated semi-automatically (their BATS tool); **mixing human and model-generated labels creates batch effects** that bias downstream analysis; they propose a mitigation.
- Relevance to **C3**: our plan ignores/pseudo-labels unlabelled rings using model confidence -> risk of the same model-vs-human label batch effect; evaluate C3 only on human labels and on overlap-disagreement fibers (already planned), and read their mitigation before T-15'.

## Leads found but not read (low priority)
- Frontiers Neuroinform 2025 (Shank3B, EM-SAM): already in report B notes; serial SEM, semantic only.
- NeuroImage 2025 human superficial WM multibeam SEM (S1053811925002150): 3D, 128k axons, not TEM 2D.
- Frontiers Cell Neurosci 2026 review of myelin imaging (Niigata; 10.3389/fncel.2026.1771168): general review.
- Kotyk et al. 2025 (inter-rater ICC 0.58 for g in pathological nerve): could not locate; cited by the Neuroinformatics 2026 review.
- Gow 2025 ASN Neuro 17:2445624 and Gow et al. 2025 17:2447336: primary sources behind the g-slope artifact (read via commentary only).
- Zou et al. bioRxiv 2025.11.05.686793: kidney TEM (TEAMKidney), not relevant.
- AxonPath preprint: still not posted (Zenodo says "in preparation"); models v0.1.0 Zenodo 22982902.
