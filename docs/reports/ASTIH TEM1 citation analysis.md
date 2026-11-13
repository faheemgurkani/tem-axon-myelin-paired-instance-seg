# ASTIH TEM1: who used it, what they found, how it compares with us (13 Nov 2026)

Notes and evidence: `docs/research_notes/ASTIH TEM1 citation analysis/` (00 = lineage, citers, per-paper verdicts; 01 = findings and comparison).

## 1. Where TEM1 comes from

| Step | Paper | Role |
|---|---|---|
| 1 | Jelescu et al., *NeuroImage* 2016 (NYU) | **Acquired the images.** Diffusion MRI validated by EM in cuprizone mice; splenium; Philips CM12, 2.36 nm/px |
| 2 | Zaimi et al., AxonDeepSeg, *Sci Rep* 2018 | Curated them into a deep-learning dataset; trained the TEM model |
| 3 | Collin et al., arXiv 2024 | Named them "TEM1"; used in multi-domain training |
| 4 | Collin, Boudreau, Cohen-Adad, **ASTIH**, MIDL 2026 | Re-annotated (2-px gaps between fibers), released on DANDI 001436 (158 images, 20 mice, 7,711 axons), benchmarked |

## 2. Who cites it, and who actually used the images

**Citers (OpenAlex):**
- Jelescu 2016: 192;
- AxonDeepSeg: 173;
- Collin 2024: 3;
- ASTIH: 0 (Google Scholar).

**Screening:**
- all titles;
- a full-text search of the citers for the dataset's identifiers ("001436", "TEM1", "White Matter Microscopy Database", "cuprizone"), which found none outside the lineage papers and Boschet 2024;
- a reading of every plausible candidate.

**Five works used the TEM1 images,** and all come from the creators' own chain (NYU → NeuroPoly):
- the four lineage papers;
- Boschet et al. 2024 (MICCAI workshop), which uses TEM1 as the labelled source for pseudo-labelling by modality translation.

The other ~360 citers cite the MRI method or apply the AxonDeepSeg *tool* to their own data. **No independent group has published results on TEM1.**

## 3. What they found on TEM1

| Work | Main result on TEM1 |
|---|---|
| Jelescu 2016 | Patchy (~5 µm) demyelination; small axons demyelinate first; only **"full myelin rings"** were segmented (threshold + connected components); g is a per-image aggregate; unmyelinated axons not segmented |
| AxonDeepSeg 2018 | Mouse test: Dice 0.949 / 0.855; centroid-based detection sensitivity 0.960, precision 0.965; macaque TEM 0.907 / 0.752 |
| Collin 2024 | Multi-domain training ≥ single-domain; out-of-domain TEM2 rises from 0.64 / 0.60 to 0.70 / 0.71 |
| Boschet 2024 | A TEM1 model on macaque TEM reaches 0.895 / 0.820 without adaptation; translation helps only across modalities. Calls TEM1 "healthy", which is wrong (it includes cuprizone mice) |
| ASTIH 2026 | nnU-Net Dice 0.974 / 0.924, but **fiber F1@0.5 0.791 (precision 0.747)**. Cellpose F1@0.5 0.882 (precision 0.990, recall 0.797). Same watershed pairing as ours. No g-ratio, pairing or error analysis |

## 4. How it compares with ours

**Matches:**
- **Same pipeline:** nnU-Net, then the seeded −EDT watershed.
- **Same conventions:** unmyelinated axons are background.
- **Selective annotation is likely shared.** TEM1's labels started from "full myelin rings" only, so incomplete or demyelinating rings are probably under-labelled, as in our data.
- **Instance-native models are conservative there too.**

**Opposite symptom, likely related cause:**
- ASTIH: high Dice (0.974) but low F1 (0.791), because precision is low.
- Ours: lower Dice (0.88 / 0.86) but higher F1 (0.904).
- Our Dice is pulled down by predictions on unlabelled real fibers. Nobody has diagnosed TEM1's low precision.

**Not done by anyone on TEM1:**
- an error decomposition;
- a pairing metric;
- a human ceiling;
- per-fiber g-ratio error.

C1 would be new there too. TEM1 can't test pairing, though, because its instance ground truth is built by the same watershed.

**TEM1's advantage over our data:** it has 20 subject IDs, so it allows animal-level splits and bootstrap.

## 5. What this adds to T-17

- Run our taxonomy on TEM1 to test whether ASTIH's low precision is **selection** (an external replication of our main finding) or **low_iou** (thin myelin).
- Expect a small separation share again.
- Use subject-level bootstrap.
- Still reproduce Dice 0.974 / 0.924 and F1@0.5 0.791 first, with the IoU threshold set explicitly.
