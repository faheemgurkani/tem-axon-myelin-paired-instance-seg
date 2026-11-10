# Public axon/myelin EM datasets and provenance/overlap check for the SIPAIM 2026 TEM challenge (as of 10–11 Nov 2026)

Scope note: metadata was pulled directly from the DANDI, Zenodo, OSF, EMPIAR and Europe PMC APIs, from a shallow clone of `axondeepseg/ASTIH` (commit 089896e, 7 Oct 2026, cloned to `external/ASTIH/`), and from the ASTIH MIDL 2026 PDF (saved as `docs/references/papers/collin26a_ASTIH_MIDL2026.pdf`). No image data was downloaded. "UNVERIFIED" marks fields I could not confirm from a primary source.

## Q1. ASTIH datasets: verified catalogue

### Takeaway
ASTIH (NeuroPoly, Polytechnique Montréal) now has **five** DANDI datasets: TEM1 001436, TEM2 001350, SEM1 001442, BF1 001440 and BF2 001630. All five are CC-BY-4.0 and BIDS-Microscopy, with 8-bit PNG images and 8-bit PNG semantic masks (axon, myelin, axonmyelin). Adjacent fibres are separated by 1–2 px gaps, so each fibre can be recovered as its own instance, but no instance-ID files are shipped. The prior review's TEM1 and TEM2 identifications are correct. For TEM2, the review's "~10 annotated images" is right, and the pixel size is 4.93 nm.

### Cited Findings
**Collection-level**
- ASTIH = "Axon Segmentation Training Initiative for Histology"; paper: A. Collin, M. Boudreau, J. Cohen-Adad, "ASTIH: A collection of axon and myelin segmentation datasets from multiple histology studies", MIDL 2026, PMLR vol. 315, pp. 3608–3627. The paper claims 5 datasets, >69,000 manually segmented fibres, 3 modalities, 3 species, 0.2–0.002 µm/px. — [PMLR](https://proceedings.mlr.press/v315/collin26a.html); [PDF](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf); [ASTIH README](https://github.com/axondeepseg/ASTIH)
- Format: source images are "8-bit grayscale PNG", semantic masks are "8-bit PNG" with suffixes `seg-axon-manual`, `seg-myelin-manual` (+ `seg-axonmyelin-manual`). Layout is BIDS-Microscopy, with JSON sidecars per image and `participants.tsv`/`samples.tsv`. — [ASTIH paper §2.1.2](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- Instance-ability: annotation followed a "boundary delineation protocol" separating adjacent fibres. An automated script checked that "every myelin region contained exactly one axon region", and violations were corrected manually. The legacy AxonDeepSeg models were trained on an earlier version in which "myelin sheaths of adjacent, densely packed axons were often merged". — [ASTIH paper §2.1.2](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- Official splits (from `scripts/dataset_list.json`): TEM1 test = `sub-nyuMouse26`. TEM2 test = an external zip (`axondeepseg/data_axondeepseg_srf_testing` tag `r20250513-neurips2025`). SEM1 test = `sub-rat6`. BF1 test = `sub-uoftRat02`, `sub-uoftRat07`. BF2 test = `sub-22G132040x3`. Splits are subject-level except TEM2, whose test images come from every mouse. — [dataset_list.json](https://github.com/axondeepseg/ASTIH/blob/main/scripts/dataset_list.json); [ASTIH paper §2.1.3](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- Baselines: nnU-Net v2.2 semantic models, followed by AxonDeepSeg watershed post-processing for instances, plus Cellpose (cyto3 fine-tuned) single-class fibre-instance models. For TEM1, nnU-Net Dice is axon 0.974 and myelin 0.950. — [ASTIH paper §2.2, Table 3](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)

**Per-dataset table** (counts from ASTIH Tables 1–2; metadata from DANDI API)

| Field | TEM1 (DANDI 001436) | TEM2 (DANDI 001350) | SEM1 (DANDI 001442) | BF1 (DANDI 001440) | BF2 (DANDI 001630) |
|---|---|---|---|---|---|
| Title (DANDI) | TEM Images of Corpus Callosum in Control and Cuprizone-Intoxicated Mice with Axon and Myelin Segmentations | TEM Images of Corpus Callosum in Flox/SRF-cKO Mice | SEM Images of Rat Spinal Cord with Axon and Myelin Segmentations | Bright-Field Images of Rat Nerves at Different Regeneration Stages… | Bright-Field Images of Rabbit Nerves from Nerve Repair Study… |
| Version cited by ASTIH | 0.250512.1625 | 0.250511.1527 | 0.250512.1626 | 0.250509.1913 | 0.251127.1424 |
| Modality | TEM (Philips CM12) | TEM, 8000×, HPF/freeze-substitution (JEOL JEM-1400 120 kV) | SEM (JEOL JSM-7600F) | Bright-field (Zeiss Axiovert 200M, 63×/1.4 oil) | Bright-field, 40×, toluidine blue |
| Species / region | Mouse / corpus callosum, splenium | Mouse / corpus callosum | Rat / cervical spinal cord | Rat / tibial, median, common peroneal nerves | Rabbit (NZ white) / tibial nerve |
| Condition | Control + cuprizone (acute 6 wk and chronic 12 wk) | WT vs SRF conditional KO (5+5) | Healthy; varied fixatives (3%/3%, 4%/0%, 4%/2% FA/GA) | Nerve repair, various regeneration stages | Nerve repair (Nerve Tape vs conduit vs suture), 16 wk |
| Pixel size | **2.36 nm** (0.00236 µm; the 2018 paper's "0.002 µm" was a rounding) | **4.93 nm** | 0.1 µm nominal (0.05–0.18 µm per image) | 0.1 µm | 0.211 µm |
| Images / annotated | 158 / 158 | 86 / 10 | 10 / 10 | 8 / 8 | 14 / 8 |
| Avg image size (px) | 3762×2286 | 4135×4016 | 1225×1064 | 5946×5805 | 1440×1024 |
| Subjects | 20 | 10 | 8 rats (ASTIH Table 1 says 10; DANDI says 8 subjects, 10 samples) | 8 | 14 |
| Fibres (ASTIH Table 1) | 7,711 (≈49/img) | 26,629 incl. unmyelinated; 4,372 myelinated (≈437 myelinated/img; ≈2,782 unmyelinated/img) | 7,948 | 17,992 | 9,558 |
| Classes | axon, myelin, axonmyelin; unmyelinated axons visible but **unlabelled** | myelinated axon, myelin, axonmyelin, **unmyelinated axon (`seg-uaxon`)**, OL nucleus, OL process | axon, myelin, axonmyelin | axon, myelin, axonmyelin | axon, myelin, axonmyelin |
| Inter-fibre delineation | 2 px | ≥2 px | 1 px | 1 px | 1 px |
| Instance vs semantic | Semantic masks; instance by connected components | Same; axon labels are proofread AxonDeepSeg pseudo-labels | Same | Same | Same |
| Files / size (DANDI) | 660 files, 1.04 GB | 134 files, 1.04 GB | 57 files, 12.5 MB | 48 files, 144 MB | 60 files, 64 MB |
| Licence | CC-BY-4.0 | CC-BY-4.0 | CC-BY-4.0 | CC-BY-4.0 | CC-BY-4.0 |
| Originating lab | NYU School of Medicine, Center for Biomedical Imaging (Jelescu, Fieremans); NYU Langone Microscopy Core; curated by NeuroPoly | Zuchero lab, Stanford (Iram et al.); imaged at UC Berkeley; curated by NeuroPoly | NeuroPoly; Montreal Heart Institute (French acronym "ICM") ethics | Univ. Toronto (Borschel, Gordon; Daeschler) | Bendale et al. 2023 (nerve-tape study) |
| Source paper | Jelescu et al., NeuroImage 2016; Zaimi et al., Sci Rep 2018 | Iram et al., PNAS 2024 | Zaimi et al. 2018; Bourget et al. 2022 (BIDS) | Daeschler et al., Sci Rep 2022 | Bendale et al. 2023 (PMC10517328) |

Sources for the table: [DANDI 001436 API](https://api.dandiarchive.org/api/dandisets/001436/versions/draft/info/), [001350](https://api.dandiarchive.org/api/dandisets/001350/versions/draft/info/), [001442](https://api.dandiarchive.org/api/dandisets/001442/versions/draft/info/), [001440](https://api.dandiarchive.org/api/dandisets/001440/versions/draft/info/), [001630](https://api.dandiarchive.org/api/dandisets/001630/versions/draft/info/); [ASTIH paper Tables 1–2 and Appendix A](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf); TEM2 pixel size and label counts (10 axonmyelin, 10 uaxon, 7 nuclei, 4 process; uint8 4675×4675 sample) from the JSON sidecar reads recorded in [mia-agentic-search PR #772](https://github.com/AI-HHMI/mia-agentic-search/pull/772).
- The TEM1 DANDI description says: "Every individual axon is separated from neighboring fibers with a 2 pixel delineation, such that this semantic segmentation dataset also doubles as an instance segmentation dataset". — [DANDI 001436](https://api.dandiarchive.org/api/dandisets/001436/versions/draft/info/)
- TEM1 subjects are `sub-nyuMouse07`–`sub-nyuMouse36` (20 IDs), with up to 8–11 samples each. — [DANDI 001436 assets](https://api.dandiarchive.org/api/dandisets/001436/versions/0.250512.1625/assets/paths/)
- The OSF mirror of the original NYU data has the condition folders `ctl_12wks`, `ctl_18wks`, `cpz6_treatment_6wks`, `cpz6_treatement_12wks`, `cpz6_treatment_6wks_recovery_6wks` and `cpz6_treatment_12wks_recovery_6wks`. The README says "Mouse brain (splenium), 0.002 micrometer pixel size, FOV = 6x9 µm²". — [OSF node adrcs (WMMD child)](https://osf.io/adrcs/)
- Two other AxonDeepSeg repos exist: the older Zenodo copy of SEM1 (`data_axondeepseg_sem`, Zenodo 5575558, listed as MIT licence) and the TEM2 model repo (`model_seg_unmyelinated_tem`). — [mia yaml zenodo-5575558](https://github.com/AI-HHMI/mia-agentic-search/blob/main/datasets/2D/other/zenodo-5575558-axondeepseg-sem-rat-spinal-cord.yaml); [DANDI 001350](https://api.dandiarchive.org/api/dandisets/001350/versions/draft/info/)

### Inferences
- TEM1 is the closest public match to the challenge (mouse corpus callosum TEM, ~2.4 nm/px, dense manual axon+myelin). Its pixel size is ~10% coarser than the challenge's 2.152 nm, and its images are smaller (3762×2286 vs 4096×3008). Rescaling by about 1.097 would match the physical scale.
- TEM2 is useful mainly as a source of unmyelinated-axon "hard negatives" and of OL-process confounders. Only 10 of its 86 images are labelled, and its axon labels are proofread pseudo-labels, not fully manual.
- SEM1, BF1 and BF2 come from different modalities and scales (0.1–0.2 µm/px, about 50–100× coarser). They are of limited direct use beyond multi-domain pretraining.

### Gaps
- Exact per-subject condition (control/cuprizone/recovery) in the DANDI TEM1 release: UNVERIFIED. I did not open `participants.tsv`; it may encode the group.
- Number of TEM1 test images in `sub-nyuMouse26`: UNVERIFIED. The samples.tsv listing suggests about 8.
- SEM1 subject count conflict: ASTIH Table 1 says 10 subjects, while DANDI and the ASTIH description say 8 rats with 10 samples.
- BF2 originating lab affiliations: UNVERIFIED. DANDI lists names only.

## Q2. Provenance and overlap: ASTIH/public data vs the SIPAIM 2026 ICM challenge data

### Takeaway
Overlap is **unlikely**. ASTIH TEM1 comes from NYU (Jelescu/Fieremans, 2016 cuprizone study, Philips CM12, 2.36 nm/px). The challenge data are ICM Paris micrographs taken with an AMT NanoSprint12 camera at 2.152 nm/px and 4096×3008 px. Every public dataset I checked differs from the challenge in lab, microscope and pixel size, and the challenge data have no public release, DOI or descriptive paper. The only documented ICM/Nait-Oumesmar ↔ NeuroPoly link is a 2021 collaboration that applied AxonDeepSeg to ICM striatal TEM. That data is not in ASTIH or the White Matter Microscopy Database (WMMD).

### Cited Findings
**Challenge facts (primary: organiser GitHub)**
- The challenge data are 400 TEM images (266 train + 134 test) of mouse corpus callosum acquired at the Paris Brain Institute (ICM): 4096×3008, 100 kV, 3000×, AMT NS12 camera, 0.002152 µm/px, 4 × 480 ms frames, gamma 1.0. Masks are uint16 (0 bg, 1 artifact, even X axon, X+1 myelin). — [data.md](https://github.com/alexandrojim/tem-challenge/blob/main/content/data.md)
- data.md states no licence, citation, DOI, animal count, experimental condition (e.g. cuprizone/LPC/control) or source paper (grep for licen/doi/cite/cuprizone/lysolecithin/subject returned nothing). — [data.md](https://github.com/alexandrojim/tem-challenge/blob/main/content/data.md)
- Distribution is only via ICM OneDrive links under `gabriel_jimenez_icm-institute_org`. The training set is "~4-5 GB", and `IS_TEST_DATA_RELEASED: false` at the last commit. — [js/config.js](https://github.com/alexandrojim/tem-challenge/blob/main/js/config.js)
- Data donors: Brahim Nait-Oumesmar (ICM PI), Bassem Hassan (ICM PI), Daniel Racoceanu (ICM–Sorbonne PI), Tala Karam (ICM PhD student), Litsa-Maria Ghayad (ICM PhD student) and Karim Ourahmoun (ICM.Quant). Annotators: Sara Rajabzadeh and Karim Ourahmoun (ICM.Quant), and Diego Jimenez (PUCP, DSP Laboratory). — [acknowledgments.md](https://github.com/alexandrojim/tem-challenge/blob/main/content/acknowledgments.md)
- The repo was created 17 Oct 2026 and last pushed 2 Nov 2026. The deadlines disagree: challenge.md says submission closes 13 Nov 2026 23:59 CET, while config.js has `SUBMISSION_DEADLINE: "2026-11-14T23:59:59"`. — [challenge.md](https://github.com/alexandrojim/tem-challenge/blob/main/content/challenge.md); [config.js](https://github.com/alexandrojim/tem-challenge/blob/main/js/config.js)
- AMT NanoSprint12 is a 12 MP CMOS TEM camera with a 4096×3008 sensor (variants 12S/12M/12L). This matches the challenge image size exactly. — [AMT NanoSprint12 datasheet (Milexia)](https://azure.milexia.com/wp-content/uploads/sites/7/2023/03/AMT-NanoSprint12rev5.pdf); [Deben listing](https://deben.co.uk/?p=77)

**ASTIH TEM1 provenance (facts)**
- TEM1 "Reference for the origin of the data: Jelescu, I. O. et al. … Neuroimage 132, 104–114 (2016) … Center for Biomedical Imaging, Department of Radiology, New York University School of Medicine". The DANDI contributors are Jelescu and Fieremans (NYU), the NYU Langone Microscopy Core (DataCollector), and Collin, Foias, Guenther and Cohen-Adad (Polytechnique Montréal). Funders are NINDS R01 NS088040 and CIHR FDN-143263. — [DANDI 001436](https://api.dandiarchive.org/api/dandisets/001436/versions/draft/info/)
- TEM1 images were "imaged using a Phillips CM12 transmission electron microscope" with ~6×9 µm² per micrograph. The original masks came from an ImageJ semi-automated macro and were later refined for ASTIH. — [ASTIH paper App. A.1](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf); [Jelescu et al. 2016](https://doi.org/10.1016/j.neuroimage.2016.02.004)
- The SEM1 text says it was approved by the "Ethics Committee of the Montreal Cardiology Institute (ICM)". This "ICM" is the Montreal Heart Institute, not the Paris Brain Institute. — [ASTIH paper App. A.3](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)

**ICM donor publications with corpus callosum / myelin TEM (facts)**
- Huré, Foucault, **Ghayad** … **Hassan** … Parras (Nat Commun 2024) used LPC focal demyelination of the **corpus callosum** at ICM, with TEM g-ratio and myelinated-axon quantification. Sections were imaged on a "Hitachi HT7700 … operating at 100 kV. Images (2048 × 2048 pixels) were taken with an AMT41B camera". Data availability: "provided in the main article, Supplementary … source data". No image repository is cited. — [Huré et al. 2024](https://doi.org/10.1038/s41467-024-54003-9) (PMC11561360)
- Lousada, Boudreau, **Cohen-Adad**, **Nait Oumesmar**, Burguière, Schreiweis (Brain Sci 2021) applied **AxonDeepSeg** to ICM TEM of striatum and cortex. Images were taken on a "HITACHI 120 kV HT7700, camera AMT XR41-B" at 100.0 k× and 6.2 k×. — [Lousada et al. 2021](https://doi.org/10.3390/brainsci11101353) (PMC8570333)
- Masson, Graciarena, Porte, Nait-Oumesmar (iScience 2025) is zebrafish work with TEM on a Hitachi HT7700 via ICM.Quant. — [Masson et al. 2025](https://doi.org/10.1016/j.isci.2025.113624)
- Gacem … **Karam** … **Nait-Oumesmar**, Martino (Sci Transl Med 2026, bavisant) validated remyelination in LPC, cuprizone and EAE models. Its EM methods were not accessible (no PMC full text). — [Gacem et al. 2026](https://doi.org/10.1126/scitranslmed.ads0633)
- Izagirre-Urizar … Nait-Oumesmar … Tepavčević (Nat Commun 2026) did its TEM on a FEI Tecnai G2 Spirit with a Xarosa camera (non-ICM). Data are "available from the corresponding author upon request". — [Izagirre-Urizar et al. 2026](https://doi.org/10.1038/s41467-026-74488-w)
- Europe PMC found no paper co-authored by Ghayad and Nait-Oumesmar, and no myelin/EM paper by Racoceanu with G. Jimenez. — [Europe PMC search](https://europepmc.org/search?query=AUTH%3A%22Racoceanu%20D%22%20AND%20AUTH%3A%22Jimenez%20G%22)

**Organiser (Gabriel Jimenez) prior work (facts)**
- G. Jiménez and D. Racoceanu's prior papers are on computational pathology: mitosis segmentation vs classification (Front Bioeng Biotechnol 2019), "Computational Pathology for Brain Disorders" (2023 book chapter), and deep-learning segmentation of neuritic plaques and tangles in Alzheimer histology (Ingrassia et al., J Neuropathol Exp Neurol 2024). None of these involves EM myelin data. — [Jiménez & Racoceanu 2019](https://doi.org/10.3389/fbioe.2019.00145); [Ingrassia et al. 2024](https://doi.org/10.1093/jnen/nlae048); [PMID 37988535](https://pubmed.ncbi.nlm.nih.gov/37988535/)
- Web searches found no public paper, preprint or dataset describing the SIPAIM 2026 TEM challenge data. — [search: SIPAIM 2026 TEM challenge](https://proceedings.mlr.press/v315/collin26a.html) (only ASTIH and AxonDeepSeg returned)

**Other Paris myelin EM work (not ICM)**
- Le Couédic et al. (IPTA 2020; ISEP + Institute of Psychiatry and Neuroscience of Paris) used 23 TEM images of aged-mouse corpus callosum from a Tecnai Spirit at 6500× (5.96 nm/px, 1336×888, 16-bit). No public release was found. — [Rajani/Le Couédic IPTA 2020](https://discovery.ucl.ac.uk/10121668/7/Rajani_1IPTA_2020_Myelin_Hal.pdf)

### Inferences (clearly separated from facts)
- **Overlap verdict: UNLIKELY (moderate-to-high confidence).** Reasons:
  1. TEM1 has a different lab (NYU, USA), microscope (Philips CM12), pixel size (2.36 vs 2.152 nm), image size (≈3762×2286 vs 4096×3008), bit depth (8-bit PNG vs camera-native) and model (cuprizone splenium). Its funders and contributors have no ICM affiliation.
  2. The challenge's 4096×3008 frame exactly matches the AMT NanoSprint12 sensor. Every ICM paper I found used Hitachi HT7700 + AMT XR41-B/AMT41B (2048×2048), an older AMT camera. The NS12 is probably a newer camera on ICM.Quant's TEM. That points to recent acquisitions not described in any paper found.
  3. None of the ASTIH or WMMD folders (DANDI, OSF) comes from Paris or ICM.
  4. AimSeg and AxonPath corpus callosum images are 2.1046 nm/px (very close to 2.152) but come from Edinburgh (JEOL JEM-1400 Plus, 4096×4096). Their pixel size and sensor geometry differ from the challenge.
- The Lousada 2021 collaboration shows that the Nait-Oumesmar lab has used AxonDeepSeg, so ICM teams may have used ASTIH-trained models for pseudo-labelling. That would be model-level, not data-level, contamination. It is speculative and has no evidence.
- The likely biological source of the challenge images is ICM demyelination/remyelination studies (LPC and/or cuprizone in corpus callosum) by the Nait-Oumesmar, Parras/Hassan (Ghayad) and Karam groups. The challenge page does not confirm this, so it is UNVERIFIED. If they come from LPC lesions, expect remyelinated thin-myelin fibres and lesion debris (consistent with the "artifact" class).
- Practical risk: there is no evidence of test leakage via public data. The challenge data have no public licence, so they should not be redistributed or used outside the challenge without asking the organisers.

### Gaps
- Exact TEM model at ICM.Quant carrying the AMT NanoSprint12 camera: not found. ICM.Quant equipment page not fetched.
- Experimental conditions, animal numbers and acquisition years of the 400 challenge images: not disclosed. Contact: gabriel.jimenez@icm-institute.org.
- Gacem et al. 2026 (Karam co-author) EM methods: paywalled and not checked.
- A public paper by Tala Karam: none found by web search.
- Whether a challenge summary paper is planned for SIPAIM 2026 proceedings: unknown.

## Q3. White Matter Microscopy Database (OSF yp4qg)

### Takeaway
WMMD is a 2016–2017 OSF aggregation curated by Cohen-Adad, Foias and Dyrby, licensed MIT on OSF. It holds mixed TEM, SEM, CARS and light data from 7+ centres. Most entries have images only or automated semantic masks, not manual instance labels. The macaque TEM is 8 corpus callosum samples × 3 images as NIfTI, with automated `myelin` and `avf` maps.

### Cited Findings
- Node yp4qg "White Matter Microscopy Database" was created 28 Sep 2016 and modified 14 May 2025. Its stated purpose is a "freely-accessible and curated microscopy data of white matter axons" for segmentation training/testing and qMRI validation. Copyright holders are Julien Cohen-Adad, Alexandru Foias and Tim B. Dyrby (2017). The OSF API licence is "MIT License". — [OSF API yp4qg](https://api.osf.io/v2/nodes/yp4qg/); [OSF](https://osf.io/yp4qg/)
- Top-level folders: DanishResearchCentreMagneticResonance (MRI only), PolytechniqueMontreal (rat spinal cord 2014/2015/2016/2017, cat and dog spinal cord, human, rat atlas), MontrealNeurologicalInstitute/histology/Macaque, VanderbiltUniversity/Mice2016, UniversityOfSouthernCalifornia/MouseCorpusCallosum. Child nodes are NewYorkUniversitySchoolOfMedicine (adrcs) and UniversidadNacionalAutonomaDeMexico (8wsda, RatOptic2019). — [OSF file API](https://api.osf.io/v2/nodes/yp4qg/files/osfstorage/)
- Macaque (MNI): `data/sample1..8/{01,03,05,07}.nii` (3 images per sample, ~28 MB each) and `segmentation/sample1..8/{avf.nii, myelin.nii}`. No description JSON sits in the folder. — [OSF file API](https://api.osf.io/v2/nodes/yp4qg/files/osfstorage/)
- Stikov et al. (Data in Brief 2015) describe macaque corpus callosum EM. Samples come from 8 corpus callosum segments at 1900× magnification, "21×28 µm² with 9.144 nm/pixel", segmented with "an automated method for large scale histology". — [Stikov et al. 2015](https://doi.org/10.1016/j.dib.2015.05.019)
- Vanderbilt Mice2016 is TEM of mid corpus callosum in control, Pten cKO, Rictor cKO and Tsc2 cKO mice, 3 TIFFs per group. Pixel size is "0.004um, 0.009um, 0.022um". Authors are West, Kelm, Carson, Gochberg, Ess and Does (NeuroImage 2016). The folder listing shows no label files. — [OSF description_histology.json (Vanderbilt)](https://osf.io/yp4qg/); [West et al.](http://www.sciencedirect.com/science/article/pii/S1053811916307935)
- USC MouseCorpusCallosum is **SEM**, Black-Gold II stained, 0.023 µm/px, with genu and body tiles (Sepehrband AxonDiameter). — [OSF (USC description JSON)](https://osf.io/yp4qg/)
- The ASTIH authors note that in WMMD "annotations are not consistently available across constituent datasets and the provided segmentation masks were not created manually". — [ASTIH paper, related work](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)

### Inferences
- The MNI macaque folder (8 samples) is very probably the Stikov 2015 macaque corpus callosum EM data (same group, 8 segments, automated segmentation). The 9.144 nm/px figure is from the paper, not the OSF folder. Treat it as an inference.
- WMMD is useful for unlabelled TEM domain diversity (Vanderbilt mouse CC TEM, macaque TEM, NYU cuprizone originals) but not as instance ground truth.

### Gaps
- Per-image pixel sizes and NIfTI headers of the macaque files: not inspected.
- Whether `avf.nii` is an axon-volume-fraction map or an axon mask: UNVERIFIED.

## Q4. AimSeg GT, AxonPath (new), AxonCallosumEM, AxonEM and unlabelled EM corpora

### Takeaway
The best new find is **AxonPath (Zenodo 23084690, 1 Oct 2026, CC-BY-4.0)**: 20 mouse corpus callosum TEM images at ~2.10 nm/px, with fibre instance labels plus axon, inner-tongue and myelin masks, and further CNS/PNS TEM/STEM sets. It supersedes the AimSeg GT (Zenodo 8351731), whose 8 images it includes after revision. AxonCallosumEM has no public data release that I could find. AxonEM is 3D axon instances without myelin. CEM500K, CEM1.5M and EMCF are the main unlabelled pretraining corpora.

### Cited Findings
**AimSeg ground truth and classifiers (Zenodo 8351731)**
- Rondelli, Carrillo-Barberà; published 16 Sep 2023; CC-BY-4.0. Files: `Validation_Dataset_v1.rar` (14.6 MB), `Control_Dataset.rar` (8.8 MB), `Classifiers_v1.rar` (56.6 MB, ilastik), `_preview.png`. Content: TEM of adult mouse corpus callosum in a remyelinating validation set (5 images, 4 mice) and a healthy control set (3 images). GT covers "instance segmentation (identifying individual myelinated axons) and semantic segmentation (discerning axons, inner cytoplasmic tongue, and myelin)". — [Zenodo 8351731](https://zenodo.org/records/8351731); [mia PR #385](https://github.com/AI-HHMI/mia-agentic-search/pull/385)
- Resolution: original 4096×4096 at 2.1046 nm on a JEOL JEM-1400 Plus. Training/validation data were resized 4× to 1024×1024 at 8.4182 nm. — [AimSeg paper](https://doi.org/10.1371/journal.pcbi.1010845) via [mia PR #385](https://github.com/AI-HHMI/mia-agentic-search/pull/385)
- An earlier "AimSeg validation dataset" (Zenodo 6327209, 30 Nov 2022, 287 MB, CC-BY-4.0) holds 5 images with instance and semantic GT TIFFs. — [mia yaml zenodo-6327209](https://github.com/AI-HHMI/mia-agentic-search/blob/main/datasets/2D/TEM/zenodo-6327209-aimseg-tem-axon-myelin.yaml); [Zenodo 6327209](https://zenodo.org/records/6327209)

**AxonPath datasets (Zenodo 23084690)**
- Carrillo-Barberà, Poveda Sabuco, Sgattoni, Gomez-Sanchez, Rondelli, Williams; published 1 Oct 2026; CC-BY-4.0; 1.84 GB. It has 47 images and 9,899 annotated fibres in 7 sets:
  - CorpusCallosum_Mus_TEM_000: 20 images, 0.00175–0.00211 µm, 2,074 fibres
  - Fimbria_Rat_STEM_000: 1 image, 6.1 nm, 420 fibres
  - OpticNerve_Mus_STEM_000: 1 image, 433 fibres
  - OpticNerve_Mus_TEM_000: 10 images, 4.3 nm, 1,084 fibres
  - SciaticNerve_Mus_STEM_000: 3 images, 15.2 nm, 470 fibres
  - SciaticNerve_Mus_TEM_000: 9 images, ~17 nm, 921 fibres
  - TibialNerve_Mus_BF_000: 3 images, 71 nm, 4,497 fibres
  
  — [Zenodo 23084690 README](https://zenodo.org/records/23084690)
- Annotations are QuPath GeoJSON (`Fibre`, `InnerCylinder` = axon + inner tongue, `Axon`, ROI) at full resolution. Training exports include a semantic mask (0 bg, 1 myelin, 2 inner tongue, 3 axon) and **fibre instance labels** (one integer per fibre). Only fibres inside the ROI are annotated. — [Zenodo 23084690 README](https://zenodo.org/records/23084690)
- Per-image corpus callosum raw pixel sizes are mostly 0.0021046–0.0021087 µm, with one at 0.0017537 µm. The training exports are downsampled 4× (~8.42 nm). — [pixel_sizes.csv](https://zenodo.org/records/23084690)
- SOURCES.csv: "8 of the 20 images were previously published, 4× downsampled, in the AimSeg dataset; their annotations were reviewed and revised for AxonPath. The other 12 are newly annotated." — [SOURCES.csv](https://zenodo.org/records/23084690)
- A related "AxonPath workshop images (SPAOM2026)" record exists (Zenodo 23066979, 30 Sep 2026, CC-BY-4.0); its contents are not inspected. — [Zenodo search](https://zenodo.org/records/23066979)

**AxonCallosumEM (arXiv 2307.02464)**
- Cheng, Zhao, Wang, Zhang. It covers a whole corpus callosum cross-section of a Rett-syndrome (Mecp2) mouse model, "1.83 × 5.76 mm" EM, with >600,000 proofread 1024×1024 patches of myelinated axons and myelin sheaths (semantic). The arXiv page gives no data link or dataset licence (only an arXiv licence for the paper). v1 is from 5 Jul 2023 with no later versions. — [arXiv 2307.02464](https://arxiv.org/abs/2307.02464)
- A GitHub repository search for "AxonCallosumEM" returned no repos. — [GitHub search](https://github.com/search?q=AxonCallosumEM&type=repositories)

**AxonEM (MICCAI 2021)**
- It has two 30×30×30 µm serial-section EM volumes: human temporal cortex L2 and mouse V1 L2/3. Labels are proofread 3D axon instances only, with no myelin class. HDF5 data are on Hugging Face `pytc/AxonEM` (~3.7 GB of train volumes) and the project page. — [arXiv 2107.05451](https://arxiv.org/abs/2107.05451); [HF pytc/AxonEM](https://huggingface.co/datasets/pytc/AxonEM); [mia yaml](https://github.com/AI-HHMI/mia-agentic-search/blob/main/datasets/3D/other/hf-pytc-axonem-axon-segmentation.yaml)

**Unlabelled pretraining corpora**
- CEM500K (EMPIAR-10592, released 21 Dec 2020): 496,544 unlabelled 224×224 cellular EM patches from 101 projects, 16.6 GB. — [EMPIAR-10592](https://www.ebi.ac.uk/empiar/EMPIAR-10592/); [Conrad & Narayan, eLife 2021](https://doi.org/10.7554/eLife.65894)
- CEM1.5M (EMPIAR-11035, released 20 May 2022): 1,592,753 unlabelled 224×224 patches from 651 image sets, 57.6 GB. — [EMPIAR-11035](https://www.ebi.ac.uk/empiar/EMPIAR-11035/); [Conrad & Narayan, Cell Systems 2023](https://doi.org/10.1016/j.cels.2022.12.006)
- EMCF (ScienceDB 32271, 16 Dec 2025): 4,002,802 micrographs from 14 EM technologies, compiled from the authors' acquisitions and public repositories (EMPIAR, BossDB, OpenOrganelle, Nanotomy), ~593 GB, for EM foundation-model pretraining. — [ScienceDB DOI 10.57760/sciencedb.32271](https://doi.org/10.57760/sciencedb.32271); [mia yaml](https://github.com/AI-HHMI/mia-agentic-search/blob/main/datasets/2D/TEM/sciencedb-32271-emcf-electron-microscopy-foundation.yaml)
- OpenOrganelle / CellMap volumes include mouse cortex and nucleus accumbens FIB-SEM (e.g. jrc_mus-cortex-2/4, jrc_mus-nacc-1) with organelle labels. Myelin labels in them are UNVERIFIED. — [mia index tree](https://github.com/AI-HHMI/mia-agentic-search/tree/main/datasets/3D/FIB-SEM)

### Inferences
- AxonPath CorpusCallosum_Mus_TEM_000 at full resolution (~2.105 nm/px) is the closest public data in physical scale to the challenge (2.152 nm/px, about 2% difference). It is the best external instance-labelled validation and fine-tuning set. Note that its "Axon" excludes the inner tongue (InnerCylinder includes it). The challenge's axon definition (X) is not stated relative to the inner tongue; this needs checking against challenge masks.
- Do not count AimSeg 8351731 and AxonPath CC as independent. 8 images overlap, and AxonPath's annotations of them are revised.
- CEM1.5M is the default large unlabelled corpus for SSL pretraining (e.g. MoCo/DINO/MAE at 224 px). Its patches are small, so long-range context for large fibres at 2 nm/px is limited. EMCF is larger but its licence and access are UNVERIFIED.

### Gaps
- Licences of CEM500K and CEM1.5M (EMPIAR defaults are usually CC0): UNVERIFIED in this session.
- AxonEM licence: UNVERIFIED.
- AxonCallosumEM: no public data or licence found. Treat it as unavailable unless the authors are contacted.
- AimSeg TEM camera model: not checked.

## Q5. mia-agentic-search PRs and other public annotated myelin EM datasets

### Takeaway
The four PRs index ASTIH TEM1, TEM2 and SEM1 plus the AimSeg GT. The PR numbers map to different titles than the brief assumed. The same index also lists AxonPath (Zenodo 23084690), the AimSeg validation set (Zenodo 6327209) and the ADS SEM Zenodo copy. Outside those, public annotated myelin EM data are scarce. The only other relevant TEM resource is the rat vagus/pelvic **unmyelinated**-axon set (Plebani 2022, Pennsieve). Connectomics volumes (MICrONS, H01, AxonEM) carry neuron/axon instances, not 2D paired axon-myelin labels.

### Cited Findings
- **PR #385** (merged 1 Oct 2026), title "Mouse corpus callosum **myelinated axons** (TEM)": indexes **AimSeg GT, Zenodo 8351731** (`datasets/2D/TEM/zenodo-8351731-aimseg-corpus-callosum-tem.yaml`). — [PR #385](https://github.com/AI-HHMI/mia-agentic-search/pull/385)
- **PR #770** (merged 8 Oct 2026), title "Mouse corpus callosum axons/myelin (TEM)": indexes **DANDI 001436 (ASTIH TEM1)**. It notes the dandiset is a draft in the API (no datePublished or DOI) while ASTIH cites DOI 10.48324/dandi.001436/0.250512.1625, and that 2.36 nm is confirmed in the `_TEM.json` sidecars. — [PR #770](https://github.com/AI-HHMI/mia-agentic-search/pull/770)
- **PR #772** (merged 8 Oct 2026), title "Mouse corpus callosum unmyelinated axons (TEM)": indexes **DANDI 001350 (ASTIH TEM2)**: 86 PNGs, 4.93 nm, 10 annotated, axon labels are "manually corrected AxonDeepSeg pseudo-labels". — [PR #772](https://github.com/AI-HHMI/mia-agentic-search/pull/772)
- **PR #1113** (merged 9 Oct 2026), title "Rat spinal cord axons/myelin (SEM)": indexes **DANDI 001442 (ASTIH SEM1)**. Per-image sidecar pixel sizes are 0.07–0.18 µm. — [PR #1113](https://github.com/AI-HHMI/mia-agentic-search/pull/1113)
- The mia index also contains these myelin/axon entries: `zenodo-23084690-axonpath-myelinated-fibres`, `zenodo-6327209-aimseg-tem-axon-myelin`, `dandi-001440` and `dandi-001630` (BF), `zenodo-5575558-axondeepseg-sem-rat-spinal-cord`, `axonem-*`, `micronsgt-zenodo-5760218-em-neuron-segmentation`, `microns-minnie65-cortical-mm3-em`, `zenodo-10864205-h01-synapse-gt`, `empiar-12244-cryo-sem-mouse-cortex-axon-seg`, and confocal myelin GT (`idr0100`, `s-biad493`). — [mia-agentic-search tree](https://github.com/AI-HHMI/mia-agentic-search/tree/main/datasets)
- Plebani et al. (Sci Rep 2022) used rat vagus and pelvic nerve TEM mosaics at 11.9 and 13.7 nm/px. "All unmyelinated axons … individually annotated" in Neurolucida 360. Data are on Pennsieve/SPARC, DOI 10.26275/K0MX-JCTH. — [Plebani et al. 2022](https://doi.org/10.1038/s41598-022-04854-3)
- EMPIAR-12244: one cryo FIB-SEM volume of mouse cortex (6.34 nm px) with partial manual axon and microglia semantic labels. — [EMPIAR-12244](https://www.ebi.ac.uk/empiar/EMPIAR-12244/)
- EMPIAR-12850: mouse brain TEM benchmark (JEOL JEM-1400 Flash) with nuclei masks only. — [EMPIAR-12850](https://www.ebi.ac.uk/empiar/EMPIAR-12850/)
- DANDI 001785 (Fujimoto, Rudebeck; Nat Neurosci 2026) includes "Electron microscopy (EM) data for myelin analysis" of macaque cingulum after DBS. It is CC-BY-4.0 and 7.1 GB, with annotation status unknown. — [DANDI 001785](https://api.dandiarchive.org/api/dandisets/001785/versions/draft/info/)
- Zenodo keyword searches ("myelin electron microscopy segmentation", "axon myelin TEM ground truth", "optic nerve electron microscopy myelin") surfaced only AimSeg, AxonPath and the AxonPath workshop set as annotated axon/myelin EM. — [Zenodo search API](https://zenodo.org/api/records?q=myelin%20electron%20microscopy%20segmentation&type=dataset)

### Inferences
- The brief's PR-title mapping is off by one. #385 is the AimSeg "myelinated axons" entry, and the "axons/myelin" title belongs to #770 (DANDI 001436).
- For optic nerve TEM with myelin labels, the only public source found is AxonPath OpticNerve_Mus_TEM_000 (10 images, 4.3 nm, 1,084 fibres).
- MICrONS and H01 could in principle yield myelin from their 3D segmentations, but they come without curated 2D axon–myelin pairs. The conversion effort is high, and the modality (serial-section TEM at 4–8 nm, cortex) differs. Low priority.

### Gaps
- Whether MICrONS minnie65 or H01 release an explicit myelin segmentation layer: UNVERIFIED (not checked in this session).
- BossDB and Figshare were not searched systematically. EMPIAR keyword search for "myelin" was not run directly; only the mia index was used.
- DANDI 001785: no annotation files inspected.

## Q6. Which datasets carry instance-level (paired) labels convertible to X/X+1?

### Takeaway
Directly convertible: **AxonPath** (fibre instance labels plus axon/inner-tongue/myelin semantics) and **AimSeg GT** (instance + semantic). Convertible via connected components, because fibres are separated and topology-checked: **ASTIH TEM1** (best), TEM2 myelinated subset, SEM1, BF1 and BF2. Not convertible to pairs: WMMD automated maps, AxonCallosumEM (semantic and unavailable), AxonEM (3D axons, no myelin), Plebani (unmyelinated only), and CEM*/EMCF (unlabelled).

### Cited Findings
- ASTIH masks separate touching fibres by 1–2 px, and an automated check enforced "every myelin region contained exactly one axon region". — [ASTIH paper §2.1.2](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf); [DANDI 001436](https://api.dandiarchive.org/api/dandisets/001436/versions/draft/info/)
- ASTIH's own instance pipeline uses AxonDeepSeg watershed post-processing on semantic masks, plus Cellpose on combined axon+myelin masks. — [ASTIH paper §2.2](https://raw.githubusercontent.com/mlresearch/v315/main/assets/collin26a/collin26a.pdf)
- AxonPath ships `training/labels/*.tif` fibre instance labels and `training/masks/*.tif` (1 myelin, 2 inner tongue, 3 axon), plus GeoJSON polygons per fibre (Fibre ⊃ InnerCylinder ⊃ Axon). — [Zenodo 23084690 README](https://zenodo.org/records/23084690)
- AimSeg GT holds instance masks (axon, inner region, fibre) and a semantic mask (axon, inner tongue, compact myelin) per image. — [mia PR #385](https://github.com/AI-HHMI/mia-agentic-search/pull/385)
- The challenge encoding is: axon = even X ≥ 2, myelin = X+1, 1 = ignored artifact. — [data.md](https://github.com/alexandrojim/tem-challenge/blob/main/content/data.md)

### Inferences (conversion recipes)
- **ASTIH (TEM1 etc.):**
  1. Label connected components of `axonmyelin > 0` (or myelin ∪ axon) to get fibre IDs k.
  2. For each k, set axon pixels to 2k and myelin pixels to 2k+1.
  3. Gap pixels stay 0 (background), whereas challenge GT may assign touching boundaries to myelin. Consider a 1–2 px myelin dilation within each fibre's Voronoi cell, or mask the gaps as ignore (label 1) during training.
  4. Drop axon components with no myelin.
  5. TEM2: map `seg-uaxon` to background, or keep it as an auxiliary negative class.
- **AxonPath / AimSeg:**
  1. Set the fibre instance to the X+1 region minus the inner region.
  2. Decide whether X = `Axon` only (inner tongue → myelin, background or ignore) or X = `InnerCylinder`. The challenge text defines the axon as "axoplasm … lighter internal region" enclosed by the "dark" myelin ring. This suggests X ends at the inner edge of compact myelin, which would be closer to InnerCylinder. UNVERIFIED; check against the challenge training masks.
  3. Use full-resolution GeoJSON (2.10 nm) rather than the 4×-downsampled exports.
  4. Annotations cover only the ROI, so mark outside-ROI pixels as ignore (label 1).
- **Scale harmonisation:** to reach 2.152 nm/px, rescale ASTIH TEM1 by 2.36/2.152 ≈ 1.097×, AxonPath/AimSeg full-res by 2.105/2.152 ≈ 0.978×, and AxonPath optic nerve TEM (4.3 nm) by ≈ 2.0×.

### Gaps
- Whether challenge GT myelin includes the inner tongue, and how touching sheaths are split: not documented on the challenge site. Inspect the 266 training masks.
- ASTIH mask pixel values (0/255 vs 0/1) and whether `axonmyelin` is a 3-level image: UNVERIFIED (no mask downloaded).
