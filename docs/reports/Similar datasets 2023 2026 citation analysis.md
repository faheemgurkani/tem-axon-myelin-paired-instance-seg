# Datasets like ours (2023–2026): who used them, what they found, how it compares (13 Nov 2026)

**Method:** the same as the TEM1 analysis.
1. Find each dataset's introducing paper.
2. List every citing work (OpenAlex) and screen it.
3. Read whatever could be a real use of the images (full text through PMC, Europe PMC or arXiv).

Notes and evidence: `docs/research_notes/Similar datasets 2023 2026 citation analysis/` (00 shortlist and screening; 01–04 per dataset; 05 comparison). TEM1 has its own report.

## 1. The datasets

| Dataset | Introduced by | What it is | Public | Citers |
|---|---|---|---|---|
| **AimSeg CC TEM** | *PLoS Comput Biol* 2023 | Mouse CC TEM at 2.10 nm; remyelinating + healthy; **per-fiber hierarchy: fiber ⊃ inner region ⊃ axon** | Yes (CC-BY) | 13 |
| **AxonPath CC TEM** | Zenodo, Oct 2026 (preprint pending) | 20 mouse CC TEM images, 2,074 fibers; same hierarchy; **includes 8 AimSeg images** | Yes (CC-BY) | 0 |
| **ASTIH TEM2** | Iram et al., *PNAS* 2024; released by ASTIH 2026 | Mouse CC TEM at 4.93 nm; SRF-knockout hypomyelination; also labels unmyelinated axons | Yes (DANDI) | 7 |
| AxonCallosumEM | arXiv 2023 (preprint) | Whole mouse CC section, SEM at 4 nm (Rett model); semantic labels only | No link given | 5 |
| Shank3B CC | *Front Neuroinform* 2025 (same group) | Mouse CC SEM at 4 nm (autism model) | On request | 3 |
| Human CC + superficial WM | Ruthig et al., *PLoS Biol* 2025 | Human TEM at 4.32 nm, 5 donors, ~400k fibers | Sample images + per-fiber tables only | 6 |
| Human superficial-WM multibeam (from **H01**, *Science* 2024) | *NeuroImage* 2025 | 3D human SEM at 4×4×33 nm | Yes | 14 (H01: 621) |
| Optic-nerve TEM | *Ophthalmol Sci* 2026 | Mouse optic nerve TEM ×3000 | Subset on GitHub | 0 |

**Popularity:**
- All of these are young and lightly cited (0–14).
- Only H01 is widely cited, and none of its citers studies myelin or g-ratio.

## 2. Who actually used the images

**Same pattern as TEM1:** every reuse comes from the creators or close collaborators.
- **AimSeg →** AxonPath (same first author), and MicroStructFormer (pre-training only, no AimSeg-specific results).
- **TEM2 →** ASTIH only.
- **AxonCallosumEM →** the same group's Shank3B study, which used their model on new data.
- **Ruthig, Shank3B, the multibeam set, AxonPath and the optic-nerve set:** nobody has reused the images.

No independent benchmark exists on any EM axon–myelin dataset from 2023 onwards.

## 3. What they found

| Dataset | Key results |
|---|---|
| AimSeg | Fiber F1 averaged over IoU 0.5–0.9: 0.85 automatic, 0.88 assisted. **Myelin g CCC 0.83, limits of agreement ±0.07**; axon g CCC 0.75. The inner tongue can change g without changing compact myelin. Ground truth excludes degenerated axons and myelin debris |
| TEM2 (ASTIH) | nnU-Net Dice 0.971 / 0.929, **F1@0.5 0.949**, against **0.791** on TEM1 with the same lab and pipeline. Biology: g 0.760 vs 0.703 because axons are larger, not because myelin is thinner |
| AxonCallosumEM | mIoU only (EM-SAM 0.984). Named error: **gaps inside the myelin sheath predicted as axons** |
| Shank3B | Denser, thinner-myelinated callosal fibers in Shank3B mice. Instances from connected components after four months of manual fixing |
| Ruthig (human) | Dice 0.94 / 0.91 in CC. Broken sheaths were **"idealized"** in the training labels. g = 0.54 in both regions, using a fitted-ellipse minor axis |

## 4. How it compares with ours

1. **Selection is universal and unmeasured.** Every dataset has an inclusion rule (excluded debris, idealized sheaths, complete rings only, abnormal fibers as background), and no paper quantifies its effect. Our selection finding and C3 have no competitor.
2. **Fiber F1 seems to follow how complete the labels are, not the model.** TEM2 (pseudo-labels corrected by hand) reaches 0.949 against TEM1's 0.791 (threshold-based "complete rings") with an identical pipeline. This is circumstantial external support for our "78% of errors are unlabelled real fibers"; T-17 can test it.
3. **Pairing is always a rule and never evaluated.** The rules are watershed, connected components, seeded expansion and a containment rule. No dataset paper reports grouping as a problem, consistent with our H1 NO-GO.
4. **The "gap" false positive appears in AxonCallosumEM too.**
5. **g agrees worst wherever it is checked per fiber.**
   - AimSeg: limits of agreement ±0.07.
   - Optic nerve: R² 0.805.
   - Five g definitions are in use across these datasets.
   - This supports our g-definition sensitivity analysis, Bland–Altman reporting and H5.
6. **Our dataset is the only one with:**
   - paired per-fiber ground truth at this scale (7,968 fibers) with an ignore class;
   - a human ceiling;
   - an error decomposition.

   Its weaknesses are no subject IDs and an unknown licence.

## 5. Changes suggested for the plan

- **T-17:** add **AxonPath CC TEM** (CC-BY, 20 images, annotated pairs, ~2 nm/px) next to TEM1. Its axon↔fiber pairs are annotated, not watershed-derived, so it is the only external set that can test the pairing metric and H5. Map its inner cylinder to our axon.
- **T-18 (paper):** use the TEM1-vs-TEM2 contrast and AimSeg's g agreement as context.
- **g reporting:** read Gow 2026, *ASN Neuro* ("the model is in the details"); it argues the standard g model is internally inconsistent. I have only read the abstract.
