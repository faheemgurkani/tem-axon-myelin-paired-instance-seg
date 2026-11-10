# 07 — Datasets

## Primary: SIPAIM 2026 TEM challenge (ICM)

400 TEM images of mouse corpus callosum, 4096×3008, 2.15 nm/px; 266 annotated (train) + 134 unannotated (test).
uint16 paired-instance masks. Full spec in [01](01-challenge-spec.md). Access is via the ICM OneDrive linked from the
challenge site. **Licence and redistribution terms are not stated. Never commit the data to this repo, and ask the
organizers before any redistribution or publication use.**

## External (pretraining / external validation)

| Dataset | Modality, species, region | Resolution | Size / annotation | Access |
|---|---|---|---|---|
| ASTIH TEM1 (DANDI 001436) | TEM, mouse CC splenium, control + cuprizone | 2.36 nm/px | 158 images, 20 mice; axon/myelin/axonmyelin + inter-fiber delineation | CC-BY-4.0. **Check overlap with the challenge before any use** |
| ASTIH TEM2 (DANDI 001350) | TEM, mouse CC, Flox/SRF-cKO, HPF | ? | ~10 images; myelinated + **unmyelinated axons** | CC-BY-4.0; hard negatives |
| ASTIH SEM1 | SEM, rat cervical spinal cord | ~0.1 µm/px | 10 samples, 8 rats | CC-BY-4.0 |
| ASTIH BF1/BF2 | Bright-field, rat/rabbit PNS | 0.1–0.2 µm/px | ? | ASTIH |
| White Matter Microscopy DB (OSF yp4qg) | TEM/SEM/BF incl. macaque TEM | various | Generalist's training source | OSF |
| AimSeg GT (Zenodo 8351731) | TEM, mouse CC, remyelination | downsampled ×4 | 8 images, hierarchical incl. inner tongue | CC-BY-4.0 |
| AxonCallosumEM | Serial SEM, mouse CC (Mecp2) | 4 nm/px | > 600k 1024² patches, semantic | Per paper |
| AxonEM | 3D EM, mouse/human cortex | ~8 nm | 3D axon instances (no myelin pairs) | MICCAI 2021 |
| CEM500K | Unlabeled cellular EM | various | ~500k images for SSL | eLife 2021 |

## Notes

- Resolution harmonization: resample external sets to ~2.15 nm/px, or train resolution-aware (as the generalist does).
- Label-convention mismatch: ASTIH/AimSeg may treat the inner tongue and periaxonal space differently. Document the mapping per dataset.
