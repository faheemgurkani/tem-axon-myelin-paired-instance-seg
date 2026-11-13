# Dataset source (checked 13 Nov 2026)

- Challenge repo alexandrojim/tem-challenge, latest commit 862d6ff (11 Nov 2026, "update challenge"). No paper, DOI, citation or licence anywhere in content/, js/ or the README. Data is shared only through ICM OneDrive links (js/config.js).
- Data page: 400 TEM images (266 train + 134 test), mouse corpus callosum, Paris Brain Institute (ICM); AMT NS12, 100 kV, x3000, 2.152 nm/px, 480 ms x 4 frames averaged, gamma 1.0. Says the acquisition values are "read directly from the embedded metadata" (our copies carry no TIFF metadata; see notes/02).
- Image donors (acknowledgments.md): B. Nait-Oumesmar (ICM PI, myelin repair / MS), B. Hassan (ICM PI, brain wiring), D. Racoceanu (ICM-Sorbonne PI), T. Karam and L.-M. Ghayad (ICM PhD students), K. Ourahmoun (ICM.Quant).
- Annotators: S. Rajabzadeh, K. Ourahmoun (ICM.Quant), D. Jimenez (PUCP).
- Web searches (organisers, donors, ICM.Quant + axon/myelin/TEM) found **no publication introducing or using this dataset**. Conclusion: the dataset is new and unpublished; the challenge is its first release. Likely origin: the donors' own biology studies (Nait-Oumesmar: lysolecithin/demyelination corpus-callosum models), consistent with the pathology seen in EDA. Unconfirmed: ask the organisers (already in 07-plan open questions).
- A web-search summary claimed the challenge's CC images "come from the AxonPath training dataset". This is **not supported**: it conflates a HHMI PR about AxonPath data (Zenodo 23084690, Edinburgh/Valencia, ~2.10 nm/px, labels 0-3) with the challenge. The challenge site credits ICM donors only.

## Second, deeper check (13 Nov 2026, later): introducing paper and its citers
Goal: find the paper that introduced the data, then the papers citing it that used it.
- Organiser (G. Jimenez Garay, ICM.Quant) Google Scholar, sorted by date: latest items are 2026 (TMI ConfIC-RCA, BSPC breast SAM), 2025 (ADNP-15 histopathology dataset, SPIE AD lesions). **No TEM / myelin / axon paper.**
- Scholar `author:racoceanu` + myelin/axon/EM since 2023: only a visual-prostheses review. **Nothing.**
- Nait-Oumesmar EM papers since 2024 (candidate biological source of the images):
  - Baudouin et al., Glia 2024, doi:10.1002/glia.24570 (PAK1/NF2): EM on a **Hitachi 7000 at 44.0k x / 11.0k x**, anterior commissure + CC (Fig. S8). Not ours (ours: AMT NS12 at 3000x, 2.152 nm/px).
  - Izagirre-Urizar et al., Nat Commun 17:7716, 2026 (MCT2): TEM of **spinal cord** dorsal funiculus, immunogold. Not ours.
- Scholar exact phrases "tem-challenge" / "TEM Axon & Myelin Segmentation Challenge" / "SIPAIM 2026" + axon myelin: **0 results**. Web search: none. GitHub code/repos referencing tem-challenge: none besides ours.
- HHMI mia-agentic-search dataset catalogue (searched for corpus callosum TEM): lists AimSeg (Zenodo 8351731, 6327209), ASTIH TEM1 (DANDI 001436), TEM2 (DANDI 001350), AxonPath (Zenodo 23084690). **The challenge dataset is not listed.**
- **Conclusion: no introducing publication exists, so there are no citing papers and no published use of the data.** The data was first released with the challenge (train 10-12 Nov, test 9-11 Nov 2026); participant papers, if any, would appear in SIPAIM 2026 (Lima, 18-20 Nov 2026) proceedings, which are not yet out. Re-check after the conference.
- Nearest published CC-TEM datasets that do have papers and citers: AimSeg (PLoS CB 2023) and ASTIH TEM1 (MIDL 2026). They are proxies, not this data.
