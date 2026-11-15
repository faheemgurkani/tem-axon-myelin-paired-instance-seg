# How big a g-ratio change matters biologically (B5), 15 Nov 2026

**Goal:** a yardstick to judge whether a g error or a g-MAE difference between methods is biologically meaningful.

## Reported effects in mouse white matter (EM)

| Contrast | Mean g (control → condition) | Δg | Source and confidence |
|---|---|---|---|
| SRF-cKO hypomyelination, CC (= TEM2) | 0.703 → 0.760 | **+0.057** | Iram et al., *PNAS* 2024 (our 13 Nov note `../Similar datasets…/02_tem2…md`). Manual expert measurements, 5+5 mice. **Solid** |
| Msh2-null dysmyelination, CC | — | **~+0.10** (95% CI of the difference ≈ 0.10–0.11) | *Sci Rep* 2016 (srep30757, PMC4967871). From a search snippet; the full text was not open via Europe PMC. **Unverified** |
| Cuprizone 0.3% vs normal diet, CC | 0.78 → 0.84 | **+0.06** | A patent (US 11324742). Internally inconsistent. **Low quality**; context only |
| Adaptive myelination: intermittent theta-burst stimulation, CC | ~5% lower g | **≈ −0.03** | Cullen et al., bioRxiv 2019 (726760). **Preprint**; check for a published version before citing |
| Chronic sleep loss | higher g (thinner myelin) | not extracted | *Sleep* 41(5) 2018. Direction only |
| Cuprizone 6 and 12 weeks, splenium (= TEM1 origin) | g correlated with extra-axonal RD (ρ = 0.48) | values not in the abstract | Jelescu 2016, PMC4851889. Europe PMC full text returned a 500 error |

**Measurement noise for comparison** (ours and the literature):
- human vs human per-fiber g-MAE **0.0067** (our ceiling, [05](../../notes/05-results.md) §2);
- B1 per-fiber g-MAE **0.016**, bias **−0.0002**;
- AimSeg manual-vs-auto limits of agreement **±0.07** (CCC 0.83);
- different g formulas shift mean g by up to **0.065** ([02](../../notes/02-challenge-and-data.md) §3).

## Yardstick (proposed; to use in T-16′, H5 and T-18)
- **Smallest biologically meaningful group-level shift: ≈ 0.03** (adaptive myelination). Disease models give **0.05–0.10**.
- **What matters for a group comparison is *systematic* error (bias and differential bias between conditions), not per-fiber MAE.** Random per-fiber error averages out over hundreds of fibers.
  - B1's bias (−0.0002) is ~150× below 0.03, so B1 is already adequate for group means on this data.
  - Per-fiber MAE matters for per-fiber analyses: g-vs-diameter distributions and outlier fibers.
- **So H5's practical importance must be argued per fiber** (for example, fraction of fibers with |g error| > 0.03), not through group means. Add this readout to H5's report without changing the H5a–c rules.

## Insight that follows: selection can bias g more than segmentation does
- The protocol leaves out abnormal fibers (thin myelin, so **high g**).
- Any group mean of g from the labelled fibers is therefore biased low relative to all fibers, and the bias grows with pathology (more abnormal fibers). That is a **differential bias between conditions**.
- **Proposed T-16′ analysis (CPU):** compare the g of unlabelled ring candidates with that of labelled fibers, and estimate the shift in mean g if they were included. If it is ≥ 0.03, the selection finding (C1) has a direct biological consequence: the protocol itself can mask a disease effect.
  - This needs only GT and the candidate detector. It is cheap and potentially the strongest sentence in the C1 paper.
  - Caveat: the candidate detector misses thin-myelin fibers ([05](../../notes/05-results.md) §5), so the estimate is a lower bound.
