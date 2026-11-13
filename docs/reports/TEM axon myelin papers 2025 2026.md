# TEM axon–myelin papers, 2025 – 13 Nov 2026

Scope: papers from 2025 up to 13 Nov 2026 that are relevant to our data (mouse corpus-callosum TEM, paired axon–myelin instances, g-ratio). Peer-reviewed venues come first; preprints are marked as such. Notes with full details: `docs/research_notes/TEM axon myelin papers 2025 2026/`. This report adds to the two earlier reports (A: 11 Nov, B: 12 Nov) and does not repeat them.

## 1. Where the dataset comes from

- **No paper introduces or uses this dataset.** The challenge repository (`alexandrojim/tem-challenge`, latest commit 862d6ff, 11 Nov 2026) cites no paper or DOI and gives no licence. The data is shared only through ICM OneDrive links.
- **Origin:** Paris Brain Institute (ICM). The image donors are the Nait-Oumesmar lab (myelin repair / MS) and the Hassan and Racoceanu labs, plus ICM.Quant. Annotation was done by ICM.Quant and PUCP.
  - The remyelination-like pathology we saw in the EDA fits a demyelination model from the Nait-Oumesmar lab. This is unconfirmed; the question is already on the organisers list in [07](../notes/07-plan.md).
- **Not the AxonPath data.** One search summary claimed the images come from the AxonPath training set (Zenodo 23084690). Nothing supports this: AxonPath is a different lab, with ~2.10 nm/px and a 4-class label scheme.
- **Consequence:** there is no published baseline on this data. Our day 1–5 numbers are the first ones on it. Any publication still needs the organisers' licence.

## 2. Papers read

| Paper | Venue | What it does | What it means for us |
|---|---|---|---|
| Ma, Hao … Lee, *optic nerve TEM* | Ophthalmology Science 6(5), Mar 2026 (peer-reviewed) | nnU-Net with an axon decoder and a fiber/boundary decoder; ×3000 TEM, 368 images / 23 mice; degenerated profiles excluded from the ground truth | **Pixel** F1 only (0.77 / 0.70); no fiber or pairing metric. g = mean centroid-to-boundary radius; g R² 0.805 on only 5 axons per grader. Split by image. Strengthens the C1 gap |
| Ahmad … Gow, Feinstein, *MyeliMetric* | ASN Neuro, Dec 2025 | g-ratio statistics after segmentation: hard filters, 6 diameter bins, pseudoreplication warnings | Cluster-level statistics are right. Its filters (g ≤ 0.9) would drop real remyelinating fibers, so selection changes summary g |
| Fields & Dutta, *G-ratio commentary* (on Gow 2025 ×2) | ASN Neuro, Apr 2025 | Positive slope of g vs **axon** diameter is mostly a measurement/sampling artifact; g vs **fiber** diameter is flat | **Fix needed:** [02](../notes/02-challenge-and-data.md) §2 calls the r = 0.65 g-vs-axon trend "textbook". Bin T-16′ error analyses by fiber diameter |
| Spalińska … Przelaskowski, *pipeline-conditional morphometry* | Neuroinformatics 24:36, Jun 2026 (review) | Morphometry validity depends on the whole pipeline; ground truth is ambiguous (manual g ICC 0.58 in pathology); calls for benchmarks of morphometric stability and expert-agreement ranges | Direct external support for C1 (decomposition and human ceilings) |
| Chuter … Jablonski, *benchmark of 18 nerve-morphometry tools* | bioRxiv, Aug 2026 (preprint) | Independent benchmark on light-microscopy optic nerve; tool ranks flip between mouse and rat | Supports H4: rankings are dataset-dependent |
| Bukas … Piraud, *AI-assisted labelling pitfalls* | MICCAI 2026 workshop (HAIC) | Mixing human and model labels in TEM creates batch effects in downstream analysis | **Risk for C3:** model-derived ignore masks could add the same bias. Evaluate C3 only against human labels; read their mitigation before T-15′ |
| Deng … Khadra, *MicroStructFormer* | bioRxiv, May 2026 (preprint) | Mask2Former (384 px input) + AxonDeepSeg; myelin-barrier watershed + geodesic assignment; "inclusive" vs "exclusive" fiber | Another rule-based pairing with no pairing metric. Adds another g-definition switch |
| Bekku … Salzer, *SnapG* | bioRxiv, Jan 2026 (preprint) | Classical thresholding; g from perimeter radius + percentile thickness | Detected 126% of manually counted CNS axons (selection again). 25% of CNS axons needed manual tracing because of the periaxonal space. A fourth g definition |

**Not found:**
- no 2025–26 paper at MICCAI, ISBI, MIDL (other than ASTIH, already covered), TMI or MedIA on paired axon–myelin instance segmentation in TEM;
- no AxonPath preprint yet;
- no Kotyk et al. 2025 full text.

## 3. What changes for us

1. **The gap holds and narrows towards C1.** Every 2025–26 paper reports pixel metrics or summary g. None measures fiber detection, pairing correctness or annotator agreement. Two independent sources now call for exactly this: the *Neuroinformatics* review and the benchmark preprint.
2. **g definitions differ across labs.** Recent work uses four: equivalent circle (challenge, MicroStructFormer), mean centroid radius (optic nerve), perimeter radius + percentile thickness (SnapG), and Feret/geometric centre (review). Our g-definition sensitivity analysis (T-16′) is therefore needed, not optional.
3. **Interpreting g by size:**
   - report errors by **fiber** diameter, not axon diameter (Gow 2025);
   - reword the EDA statement in notes/02;
   - keep cluster-level statistics.
4. **C3 caution:** the labelling-pitfalls paper is direct prior evidence that model-assisted labels bias downstream results. Add it to the C3 risks (D-18).
5. **Selective annotation is common but never modelled.** The optic-nerve paper excludes degenerated profiles, and SnapG's counts disagree with manual by 26%. This supports the selection finding (78% of B1's errors) and C3's motivation.

## 4. Downloads

- The curl download was blocked by the permission system.
- In the browser, the Springer "Download PDF" click and the MICCAI open-access PDF produced no file in `~/Downloads` after 40+ s, and Chrome's PDF viewer stayed blank.
- All papers above were read as full-text HTML (PMC, bioRxiv, Springer through institutional access), except Bukas et al. and Chuter et al., for which only the abstract was read.
- No PDFs were added to `docs/references/papers/`.

## 5. Addendum: is there a paper that introduced the dataset, and who cites it?

**No.** A second, deeper search covered:
- the organiser's and PI's Google Scholar records;
- the donor lab's 2024–26 EM papers;
- exact-phrase searches for the challenge on Scholar, the web and GitHub;
- the HHMI dataset catalogue.

None of them turned up a publication that introduces or uses this data. The two donor-lab EM papers use different instruments and tissue:
- *Glia* 2024: Hitachi 7000 at 44,000×;
- *Nature Communications* 2026: spinal cord.

So there are no citing papers to analyse, and no published findings to compare with ours. The first papers on this data would be SIPAIM 2026 challenge entries (conference 18–20 Nov 2026); re-check after it. Details: `research_notes/…/00_dataset_source.md`.
