# 07 — Datasets

## Primary: SIPAIM 2026 TEM challenge (ICM)

400 TEM images of mouse corpus callosum, 4096×3008, 2.15 nm/px; 266 annotated (train) + 134 unannotated (test).
uint16 paired-instance masks. Full spec in [01](01-challenge-spec.md). Access is via the ICM OneDrive linked from the
challenge site. **Licence and redistribution terms are not stated. Never commit the data to this repo, and ask the
organizers before any redistribution or publication use.**

## External (pretraining / external validation), verified 2026-11-11

Source: `../research_notes/TEM axon myelin segmentation review/datasets_provenance.md` (ASTIH paper, MIDL 2026, PMLR 315, and the DANDI API).
All ASTIH sets are CC-BY-4.0, in BIDS layout, with 8-bit PNG images and 8-bit PNG **semantic** masks (axon / myelin / axonmyelin).
Adjacent fibers are separated by a 1–2 px gap, and every myelin region contains exactly one axon, so fiber instances are recoverable by
connected components. No instance-ID files are shipped.

| Dataset | Content | Pixel size | Images | Fibers | Notes |
|---|---|---|---|---|---|
| ASTIH TEM1 (DANDI 001436) | Mouse CC splenium, control + cuprizone, 20 mice (NYU; Jelescu & Fieremans 2016; Philips CM12) | 2.36 nm | 158, all annotated, ~3762×2286 | 7,711 | Test subject `sub-nyuMouse26`; ×1.097 to reach 2.152 nm |
| ASTIH TEM2 (DANDI 001350) | Mouse CC, WT vs SRF-cKO, 10 mice, HPF, 8000× | 4.93 nm | 86, 10 annotated | 26,629 (4,372 myelinated) | **Unmyelinated axons** labelled (hard negatives); axon labels are corrected pseudo-labels |
| ASTIH SEM1 (DANDI 001442) | Rat cervical spinal cord | 0.05–0.18 µm | 10 (10 subjects) | 7,948 | |
| ASTIH BF1 (DANDI 001440) | Rat PNS, bright-field | 0.1 µm | 8 | 17,992 | |
| ASTIH BF2 (DANDI 001630) | Rabbit tibial nerve, bright-field | 0.211 µm | 14, 8 annotated | 9,558 | |
| **AxonPath (Zenodo 23084690, 1 Oct 2026)** | Edinburgh/AimSeg group; includes **20 mouse CC TEM at ~2.10 nm** (2,074 fibers) plus optic nerve, sciatic and fimbria TEM/STEM sets (47 images and 9,899 fibers in total) | ~2.10 nm | 47 | 9,899 | CC-BY-4.0; **fiber instance IDs** + axon / inner-tongue / myelin; QuPath GeoJSON. **Closest match to the challenge** |
| AimSeg GT (Zenodo 8351731) | Mouse CC TEM, remyelination | 8.42 nm (×4 downsampled) | 8 | — | CC-BY-4.0; superseded by AxonPath (8 images are revised versions) |
| White Matter Microscopy DB (OSF yp4qg) | Mixed TEM/SEM/LM, mostly unlabelled or automated maps; macaque entry probably Stikov 2015 (9.144 nm) | various | — | — | MIT on OSF; generalist's "TEM2" macaque OOD set |
| AxonCallosumEM | Serial **SEM** mouse CC (Mecp2), 4 nm | 4 nm | > 600k patches | — | No data release or licence found |
| AxonEM | 3D EM cortex, axon instances, no myelin | ~8 nm | — | — | Hugging Face `pytc/AxonEM` |
| CEM500K (EMPIAR-10592), CEM1.5M (EMPIAR-11035), EMCF (~4M) | Unlabelled cellular EM for SSL | various | 0.5M / 1.59M / ~4M | — | Licences unverified |

### Overlap with the challenge data: **unlikely** (moderate–high confidence)

- TEM1 was produced at NYU on a Philips CM12 at 2.36 nm with ~3762×2286 images. The challenge images are from ICM at 2.152 nm, 4096×3008, which is the sensor size of
  the AMT NanoSprint12.
- Published ICM papers used a Hitachi HT7700 with 2048² AMT cameras. The challenge images are probably recent and unpublished acquisitions.
- The only ICM–NeuroPoly link is Lousada et al. 2021 (striatum, not in ASTIH). "ICM" in the ASTIH SEM1 text means the Montreal Heart Institute.
- The challenge data has no licence, DOI, animal count or source paper.

### Conversion to the X/X+1 format

- **Direct:** AxonPath and AimSeg (fiber instance IDs). First decide whether the challenge's axon includes the **inner tongue**. Check this in the 266 training masks.
- **Via connected components:** ASTIH sets (TEM1 best). The inter-fiber gap pixels need a policy: grow myelin into them, or mark them as ignore.
- **Not convertible:** WMMD, AxonCallosumEM, AxonEM, unlabelled corpora.

## Notes

- Resolution harmonization: resample external sets to ~2.15 nm/px, or train resolution-aware (as the generalist does).
- Label-convention mismatch: ASTIH/AimSeg may treat the inner tongue and periaxonal space differently. Document the mapping per dataset.
