# ASTIH TEM1: lineage, citers, and who actually used the images (13 Nov 2026)

## Lineage of the TEM1 images (from ASTIH App. A.1 + the papers themselves)
1. **Jelescu et al. 2016**, NeuroImage 132:104-114, doi:10.1016/j.neuroimage.2016.02.004 (PMC4851889). NYU (Fieremans lab). **Acquisition paper.** dMRI (WMTI) validated by EM in cuprizone mice. 36 female C57BL/6, 8 wk; Ctl / Cpz6 / Cpz12 (0.2% cuprizone, 6 or 12 wk, then 6 wk recovery); EM at 6, 12, 18 wk on 3 mice per group x timepoint, **8 micrographs per mouse**, splenium of CC; Philips CM12 + Gatan 4k x 2.7k, **2.36 nm/px, ~6 x 9 um** per image.
2. **Zaimi et al. 2018**, AxonDeepSeg, Sci Rep 8:3816. **Curated the images into a DL dataset**; TEM model trained on them.
3. **Collin et al. 2024**, arXiv 2409.11552 (multi-domain generalist). Used as "TEM1" (from the White Matter Microscopy Database).
4. **Collin, Boudreau, Cohen-Adad, ASTIH, MIDL 2026** (PMLR 315:3608-3627). Re-annotated (2-px gaps between touching fibers), released as DANDI 001436: 158 images, 20 subjects, 7,711 axons, avg 3762x2286.

## Citation counts (OpenAlex, 13 Nov 2026)
| Paper | Citing works |
|---|---|
| Jelescu 2016 | 192 |
| AxonDeepSeg 2018 | 173 |
| Collin 2024 (arXiv) | 3 |
| ASTIH MIDL 2026 | not indexed in OpenAlex (checked Scholar separately; see 02) |
Total ~368 citing works (with overlaps).

## Screening method
- Title screen (segmentation / EM / microscopy / g-ratio / dataset keywords): 21 of Jelescu's citers, 90 of AxonDeepSeg's.
- OpenAlex full-text search inside the citers for dataset identifiers: "001436", "data_axondeepseg_tem", "DANDI:001436" -> **0 hits**; "cuprizone" inside ADS citers -> 12 (all checked by title/role, none reuse TEM1 images except as noted in 01).
- Most Jelescu citers are diffusion-MRI papers citing the MRI result; most ADS citers apply the ADS *tool* to their own data.

## Additional checks
- ASTIH (MIDL 2026) on Google Scholar: no "Cited by" link -> **0 citers** as of 13 Nov 2026.
- Full-text "White Matter Microscopy Database" (the pre-ASTIH host of TEM1): 13 works. Besides the lineage papers, the rest are diffusion/T2-MRI papers (Dmipy, T2 relaxometry, axon-diameter MRI); OpenAlex full-text search for "cuprizone" OR "transmission electron" in each returned 0 -> they used other WMMD subsets, not TEM1.
- Full-text "TEM1" among ADS citers: only Collin 2024.

## Verdict per candidate (read in full or abstract + methods)
| Work | Used TEM1 images? | Evidence |
|---|---|---|
| Jelescu 2016, NeuroImage | **Yes (acquired them)** | §2.4 EM methods (PMC4851889) |
| Zaimi 2018, AxonDeepSeg, Sci Rep | **Yes (train/test of TEM model)** | Table 1: 8x17 mice train/val, 8x3 mice test, splenium, 0.002 um/px |
| Collin 2024, arXiv 2409.11552 | **Yes (training source)** | Table 1 + "datasets used for training were TEM1, ..." |
| Boschet, Collin, Katoch, Cohen-Adad 2024, MICCAI-W (arXiv 2412.02858) | **Yes (labelled source domain, 158 images)** | Table 1 "TEM, mouse, 0.00236, 158 images" |
| Collin, Boudreau, Cohen-Adad 2026, ASTIH, MIDL | **Yes (released + benchmarked)** | Tables 1-4 |
| Le Couedic 2020, IPTA | No: quotes ADS-TEM published numbers only; own Tecnai data | Table III |
| Deep active learning 2019 (arXiv 1907.05143) | No: TEM *spinal cord* 24 images, 2.4 nm (different WMMD set) | §Datasets |
| Papazoglou 2023, NMR Biomed | No: own Tsc2 EM; cites Jelescu's unmyelinated % | text |
| Zhou 2020, NeuroImage | No: postmortem human CC, staining | abstract |
| Hill 2020, NeuroImage (ICM Paris, cuprizone) | No: own mice/histology | abstract |
| Nguyen 2022, AIPR meta-learning | No: own hypoglossal nerve EM | abstract |
| AxonSynth 2026 | No: 3D light-sheet | abstract |
| Other ~350 citers | No (title screen: MRI methods, or ADS tool applied to own data) | title + full-text keyword screen |

**Result: 5 works used TEM1; all 5 are by the creators' chain (NYU -> NeuroPoly/Polytechnique Montreal). No independent group has published results on TEM1.**
