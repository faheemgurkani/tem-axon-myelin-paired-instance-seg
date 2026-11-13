# Cross-dataset comparison with our work (challenge data, B1 results, C1/C3/H5)

| | Ours (challenge) | TEM1 | TEM2 | AimSeg / AxonPath CC | AxonCallosumEM / Shank3B | Ruthig human CC |
|---|---|---|---|---|---|---|
| Modality, px | TEM, 2.152 nm | TEM, 2.36 nm | TEM, 4.93 nm | TEM, 2.10 nm (AxonPath 1.75-2.11) | sSEM, 4 nm | TEM, 4.32 nm |
| Pathology | de/remyelination (inferred) | cuprizone | SRF-cKO hypomyelination | toxin lesion remyelination + healthy | Rett / Shank3B | aged human |
| Per-fiber pairing in GT | **yes (X / X+1)** | no (semantic; watershed) | no (semantic; watershed) | **yes (fiber ⊃ inner cyl. ⊃ axon)** | no (semantic; CC) | no (semantic; watershed) |
| Ignore class | label 1 | none | none | ROI-only annotation | none | none |
| Selective annotation | abnormal fibers = bg; ~4-8% misses | "full rings" origin | pseudo-label + correction (near-exhaustive) | degenerated axons + myelin debris excluded | n/a | broken sheaths "idealized" |
| Fiber-level metric reported | F1@0.5 0.904 (B1) + taxonomy | F1@0.5 0.791 | F1@0.5 0.949 | F1 avg IoU .5-.9: 0.85-0.88 | none (mIoU 0.984) | none (Dice 0.94/0.91) |
| Per-fiber g agreement | g-MAE 0.016 (B1), human 0.0067 | none | none | **myelin g CCC 0.83, LoA ±0.07** | none | none |
| Human ceiling | yes (overlap tiles) | no | no | no | no | no |
| g definition | equivalent circle | aggregate | manual line | equiv. circle; "myelin" vs "axon" g | aggregate (MVF/AVF) | ellipse minor axis |
| Subject IDs | **no** | 20 mice | 10 mice | 4+1 mice | 2 animals | 5 donors |
| Reused independently? | — | no | no | no | no | no |

## What matches our findings
1. **Selection is everywhere and unmodelled.** Every dataset has an explicit or implicit inclusion rule (AimSeg excludes degenerated axons/debris; Ruthig "idealizes" broken sheaths; TEM1 began from full rings; ours leaves abnormal fibers as background). None of the papers measures its effect. Our C1 selection finding and C3 have no competitor.
2. **Detection F1 tracks annotation exhaustiveness, not the model.** Same lab, same nnU-Net + watershed: TEM2 (pseudo-label-then-correct) F1@0.5 0.949 vs TEM1 (threshold "full rings") 0.791. This is external, circumstantial support for our selection result (78% of B1 errors are unlabelled real fibers). Testable in T-17.
3. **Rule-based pairing is universal and never evaluated.** Watershed (TEM1/2, Ruthig), connected components (Shank3B), seeded expansion (AimSeg), containment rule (AxonPath). Consistent with our H1 NO-GO: no dataset paper reports grouping errors as a problem; AimSeg's hierarchical GT is the only other source of true pairs.
4. **"Gap" false positives are a named failure mode** elsewhere (AxonCallosumEM: gaps inside sheaths predicted as axons) - matches our taxonomy class "gap".
5. **g is the weakest-agreeing quantity** wherever it is measured per fiber: AimSeg myelin-g CCC 0.83 (LoA ±0.07, i.e. SD ≈ 0.035 and an implied MAE ≈ 0.028 if errors are normal - roughly 4x our human-vs-human g-MAE of 0.0067 and ~2x B1's 0.016; different data and tool, so indicative only), axon-g CCC 0.75; optic nerve R2 0.805. Supports reporting g with Bland-Altman (T-16') and H5's focus on g.
6. **g definitions disagree across datasets** (equivalent circle, ellipse minor axis, aggregate MVF/AVF, AimSeg's myelin vs axon g, manual line). The inner-tongue convention matters: our data counts the tongue as axon (= AimSeg "myelin g"/inner region), AimSeg shows tongue size can change g without changing compact myelin.

## Where ours differs / is stronger or weaker
- **Stronger:** the only dataset with per-fiber paired instance GT at this scale (7,968 fibers) plus an ignore class; the only one with a human ceiling (overlap tiles) and an error decomposition.
- **Weaker:** no subject IDs (all others have them); licence unknown (AimSeg, AxonPath, TEM1/2 are CC-BY or DANDI).
- **Closest external match for pairing claims: AimSeg / AxonPath CC** (TEM ~2.1 nm, remyelination, hierarchical per-fiber GT incl. inner tongue). Better than TEM1 for testing C1's pairing metric and H5, because its pairs are annotated, not watershed-derived.

## Implications for the plan
- T-17: keep TEM1 (published benchmark) but **add AxonPath CC TEM (20 images, 2,074 fibers, CC-BY)** as the pairing-capable external set; map inner cylinder -> our axon (tongue counted as axon, as already noted in notes 02 §5).
- Paper framing (T-18): "fiber-level F1 is dominated by annotation selection" can cite the TEM1 vs TEM2 contrast as motivation, and AimSeg's g CCC as the field's current per-fiber g agreement.
- g reporting: add Gow 2026 ("Myelin g ratios: the model is in the details", ASN Neuro) to the g-definition discussion - argues the canonical g model is internally inconsistent (abstract only read).
