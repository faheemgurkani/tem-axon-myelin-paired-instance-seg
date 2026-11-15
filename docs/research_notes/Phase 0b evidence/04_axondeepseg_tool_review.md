# AxonDeepSeg as a tool (D2), reviewed 15 Nov 2026

**Sources:**
- Zaimi et al., *Sci Rep* 8:3816 (2018), local `docs/references/papers/zaimi2018_axondeepseg_scirep.pdf` (+ `zaimi2018.txt`);
- code at `external/axondeepseg` @ 05a8dd1 (30 Oct 2026): `AxonDeepSeg/morphometrics/compute_morphometrics.py`, `postprocessing.py`;
- model repos `axondeepseg/default-TEM-model` and `axondeepseg-pipeline/model_seg_unmyelinated_tem` (GitHub API, 15 Nov);
- Collin et al. 2024 generalist (local `generalist_2409.11552.txt`; numbers from our 11–13 Nov reports).

## Method
- **2018:** a U-Net for 3-class semantic segmentation (background / axon / myelin). Inputs are resampled to a fixed working pixel size (0.01 µm for TEM, 0.1 µm for SEM) and cut into 512×512 patches.
  - **Training/validation was split at the patch level** (random ~70/30). Separate test images were used for the reported numbers.
- **2024 onward:** the models were retrained with nnU-Net v2, one per dataset (TEM1 = `default-TEM-model` r20240403, 5-fold CV, 3 folds + `fold_all` released). There is also a multi-domain generalist. The ASTIH paper (MIDL 2026) benchmarks these models.
- **No instance model.** Fibers are formed after the network, as below.

## How a fiber is formed, and how g is computed (code)
1. **Instances:** `get_watershed_segmentation` seeds a watershed at the **centroid** of each axon connected component, on −EDT of the **axon** mask, restricted to axon ∪ myelin.
   - Our shared rule (D-11) differs: it uses whole axon components as seeds and −EDT of **axon ∪ myelin**.
   - **T-17 must use ADS's own rule to reproduce ASTIH numbers.**
2. **Axon → fiber link:** the fiber label read at the axon's **centroid pixel**. This fails when the centroid falls outside a concave axon, which is the failure our D-11 avoids.
3. **Per-fiber g, default `axon_shape="circle"`:** axon diameter = equivalent diameter; thickness = (D_fiber,eq − D_axon,eq)/2; g = (D_A/2)/(D_A/2 + thickness) = **D_A,eq / D_F,eq**. This is **identical to the challenge's equivalent-circle g**.
   - `axon_shape="ellipse"` uses the **minor-axis** length instead. That is the minor-axis switch already planned for T-16′.
4. **Aggregate g:** √(1/(1 + MVF/AVF)) from area fractions, the MRI-style aggregate. This is not a mean of per-fiber g.
5. It reports eccentricity, orientation and solidity per fiber. These are usable for the oblique-fiber switch (see `02_gow_g_ratio.md`).

## Evaluation practice
- **2018:** pixel Dice (axon, myelin), pixel accuracy, plus **centroid-based** detection sensitivity and precision.
  - Mouse TEM test: Dice 0.949 / 0.855, sensitivity 0.960, precision 0.965. Macaque TEM: 0.907 / 0.752.
  - **No IoU-matched fiber F1, no pairing check, no per-fiber g error.**
- **ASTIH 2026** adds fiber F1@IoU 0.5 through `stardist.matching`. There is still no pairing or g-error analysis.
- **Generalisation** (Collin 2024): out-of-domain TEM2 axon/myelin Dice 0.697 / 0.706 for the generalist, with modest gains from multi-domain training.
- The 2026 `unmyelinated-TEM` v3.0.0 model was trained on TEM2 **plus an unpublished dataset**. Never use it to score TEM2 (possible leakage, and its training set is unknown).

## What it means for us
- ADS is the reference *tool* lineage behind TEM1/TEM2 and ASTIH. Our evaluator's g equals ADS's default g, so g numbers are comparable.
- **Its known failure point is centroid-based linking** (step 2). It is a natural H5-style error source for weaker pixels, but irrelevant for our D-11 rule (R1: lossless on perfect pixels).
- **Related-work sentence:** "AxonDeepSeg pairs by centroid lookup after a centroid-seeded watershed and reports centroid detection and pixel Dice; no per-fiber pairing or g error is reported."
