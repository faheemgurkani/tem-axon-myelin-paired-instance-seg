# Ruthig et al. 2025, PLoS Biology (PMC12410883): human CC vs superficial WM, 2D TEM

- Leipzig (Morawski / Mohammadi groups). 5 elderly human donors (60-78 y), postmortem; LEO 912 TEM, 80 kV, **4.32 nm/px**, tiled large FOV (80 x 800 um).
- ~16,000 fibers labelled for training; **DenseNet semantic segmentation (bg / axon / myelin) + watershed for instances** (reviewer summary: "U-Net semantic segmentation followed by watershed").
- Validation (held-out images): CC IoU 0.89 / 0.83, **Dice 0.94 / 0.91**, HD95 0.04 / 0.06 um; SWM much worse (Dice 0.77 / 0.70).
- Annotation: **"we 'idealized' obvious artifacts in the training data (e.g., partly broken-down myelin sheaths)"** - an explicit label-cleaning convention, admitted as a possible systematic error.
- g-ratio from the **minor axis of fitted ellipses** (chosen as "most reliable" in their validation) -> fifth g definition we have seen. Result: CC fibers ~25% larger and more myelinated than SWM, but **g equal (0.54)** in both.
- Reviewer flagged pseudoreplication (pooling fibers across 5 donors).
- Data: per-fiber CSVs + sample images public (GitHub / Zenodo 15601201, 15720452); full images on request.
- Citers (6): CARS uncinate (own data), Mordhorst 2026 (own 3D/LM data), conduction-velocity sEEG, SWM atlas (MRI), Gow 2026, an unrelated pharmacology paper. **No reuse of the images.**
