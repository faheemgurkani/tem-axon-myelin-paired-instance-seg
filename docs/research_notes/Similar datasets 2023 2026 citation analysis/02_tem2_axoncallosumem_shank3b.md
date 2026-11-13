# ASTIH TEM2, AxonCallosumEM, Shank3B

## ASTIH TEM2 (DANDI 001350)
- **Source:** Iram, Garcia, ... Wyss-Coray, Kern, Zuchero. "SRF transcriptionally regulates the oligodendrocyte cytoskeleton during CNS myelination." PNAS 121(12), Mar 2024 (PMC10962977). Stanford/UC Berkeley EM lab.
- Biology: SRF-cKO (Olig2-Cre) -> **hypomyelination**: about half as many myelinated axons; CC g-ratio 0.760 (cKO) vs 0.703 (WT); myelin thickness unchanged, g rises because **myelinated axons are larger** in cKO. Manual ImageJ straight-line measurements by an expert, 5 cKO + 5 control mice.
- Imaging: HPF-FS, OTO staining, 90 nm sections, JEOL JEM-1400 120 kV; 4.93 nm/px; 86 images, only 10 with GT, large FOV (>= 1000 um^2), avg 4135x4016.
- **ASTIH annotation:** ADS pseudo-labels manually corrected; >= 2-px gaps between fibers; **unmyelinated axons, OL nuclei and OL processes labelled as extra classes** (26,629 axons; 4,372 myelinated).
- **ASTIH results:** nnU-Net Dice 0.971 / 0.929; **F1@0.5 0.949 (P 0.945, R 0.958)**; Cellpose F1@0.5 0.902 (P 0.996, R 0.829). Collin 2024 generalist on TEM2 (OOD): 0.697 / 0.706.
- **Key contrast with TEM1:** same lab, same pipeline, but F1@0.5 0.949 vs 0.791. TEM2 labels came from **model pseudo-labels corrected by hand** (near-exhaustive) whereas TEM1 started from threshold-segmented "full rings". Consistent with our claim that **annotation selection, not the segmenter, drives fiber-level F1** (not proven: TEM2 also has a single 'healthy-ish' phenotype and larger pixel size).

## AxonCallosumEM (arXiv 2307.02464, 2023; preprint, never peer-reviewed as far as indexed)
- Cheng, Zhao, Wang, Zhang (Soochow Univ. / SIBET Suzhou). Mecp2 (Rett) mouse, **whole CC sagittal cross-section 1.83 x 5.76 mm at 4 nm/px**, > 600,000 1024^2 patches; semantic GT (myelinated axon, myelin) built semi-automatically: manual region -> fine-tuned SAM predictions -> proofreading in VAST, iteratively.
- Benchmark: mIoU only. U-Net 0.919, ViT-B 0.947, MAE 0.966, BEiT 0.962, **EM-SAM 0.984** (split by region: genu / body / splenium for train / val / test).
- Failure modes named: **gaps inside the myelin sheath and thick-membrane nuclei predicted as myelinated axons** (= our "gap" FP class), unclear boundaries, nodes of Ranvier, long oblique axons.
- g-ratio only as an aggregate map from MVF/AVF (~0.7).
- No download link in the paper.

## Shank3B (Frontiers Neuroinform 2025, PMC12021825)
- Same Suzhou group; WT vs Shank3B+/- mice; serial SEM 4 nm, whole sagittal CC. EM-SAM (fine-tuned) -> semantic -> **instances by connected components + dilation/erosion**; **four months of manual refinement** of the predictions.
- Findings: higher myelinated-axon density, thinner myelin, altered diameter distribution and g-ratio in Shank3B. Data on request.
- No segmentation accuracy at fiber level reported (Dice/mIoU in supplement).
