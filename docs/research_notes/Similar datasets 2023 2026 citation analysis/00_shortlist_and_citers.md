# Similar datasets published 2023+ : shortlist, citers, screening (13 Nov 2026)

Criteria: published 2023 or later; EM (or EM-like) images of myelinated axons with axon + myelin annotation, usable for paired axon-myelin segmentation / g-ratio. "Popularity" = citations of the introducing paper (OpenAlex, 13 Nov 2026) + whether data is public. ASTIH TEM1 is done separately (report "ASTIH TEM1 citation analysis"); ASTIH TEM2 is included here.

| Dataset | Introducing paper (year) | Modality / tissue | Public? | Citers |
|---|---|---|---|---|
| **AimSeg CC TEM** (Zenodo 8351731, 6327209, 8021007) | Carrillo-Barbera et al., PLoS Comput Biol 2023 | TEM, mouse CC, remyelinating + healthy; fiber / inner-region / axon instances | Yes, CC-BY | 13 |
| **ASTIH TEM2** (DANDI 001350) | Iram et al., PNAS 2024 (images); ASTIH MIDL 2026 (release) | TEM, mouse CC, SRF-cKO vs control; myelinated + unmyelinated axons | Yes | 7 (+0 for ASTIH) |
| **AxonCallosumEM** | Cheng, Zhao, Wang, Zhang, arXiv 2307.02464 (2023, preprint only; Suzhou) | serial SEM, whole mouse CC cross-section (Rett model), 4 nm; **semantic only** | No download link in the paper | 5 |
| EM-SAM / Shank3B CC | Frontiers Neuroinform 2025 (same Suzhou group) | serial SEM 4 nm, mouse CC sagittal (WT vs Shank3B) | **On request** | 3 |
| Human superficial WM multibeam SEM | NeuroImage 2025 (PMC12312701); **a derivative of H01** (Shapson-Coe et al., Science 2024, 621 citers) | 3D multibeam SEM 4x4x33 nm, human SWM | Yes (H01 release) | 14 |
| Human CC + SWM 2D TEM (Ruthig et al.) | PLoS Biol 2025 (PMC12410883) | 2D TEM 4.32 nm, 5 human donors, ~400k fibers | **Images on request**; sample images + per-fiber CSVs on GitHub/Zenodo | (new; see 04) |
| **AxonPath datasets** (Zenodo 23084690) | Zenodo 2026 (preprint not out) | TEM/STEM + BF; mouse CC + other nerves; **fiber instances + inner cylinder + axon** | Yes, CC-BY | 0 |
| Optic nerve TEM subset | Ma et al., Ophthalmol Sci 2026 | TEM x3000, mouse optic nerve | Subset on GitHub | 0 |

Excluded (pre-2023): WMMD 2016, ADS TEM/SEM 2018, AxonEM 2021, DeepACSON 2021.

## Screening of every citer (OpenAlex lists, 13 Nov 2026) - who actually used the images?
| Dataset | Citers | Used the images | Everyone else |
|---|---|---|---|
| AimSeg CC TEM | 13 | **AxonPath datasets (Zenodo 2026)**: includes 8 AimSeg CC images (same first author); **MicroStructFormer (bioRxiv 2026)**: AimSeg data cited (ref 35) as public EM data for domain pre-training; also AxonPath workshop images (Zenodo, likely derived) | reviews (Neuroinformatics 2026, Frontiers 2026, Toxicol Pathol 2025, Methods Microsc 2024), LM/other tools (Spalinski AIME 2025, Chuter 2026), 3D XNH (Amirrashedi 2025, own data), SARM1 2023 (biology), Biomaterials 2025 (spinal cord injury; not verified, no OA text - tool or citation only) |
| ASTIH TEM2 (Iram 2024) | 7 (+ASTIH) | **ASTIH 2026** only | all biology / reviews (myosin, Tfii-i, aging OLs, MBP, tilapia) |
| AxonCallosumEM | 5 | **Shank3B 2025** (same group; reuses the EM-SAM pipeline, new data) | foundation-model surveys, SAM-in-remote-sensing, retinal MAE-SAM2 (ISBI 2026), Collin 2024 (cites only) |
| Shank3B | 3 | none | ketogenic diet, optic nerve ASD, Gow 2026 (g-ratio models) |
| SWM multibeam / H01 | 14 / 621 | SWM paper itself uses H01; its 14 citers are MRI / conduction-velocity / other-histology papers (Ruthig 2025 = own TEM data; Mordhorst = own LM/3D data). H01 citers with full-text "g-ratio"+"myelin": 0 | — |
| AxonPath datasets | 0 | — | — |
| Optic nerve TEM (Ma 2026) | 0 | — | — |

**Pattern (same as TEM1): every reuse is by the creators or a close collaborator. No independent benchmark exists on any of these datasets.**
