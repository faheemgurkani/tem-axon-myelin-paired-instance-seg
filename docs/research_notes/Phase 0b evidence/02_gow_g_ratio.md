# Gow 2025–2026: what the g-ratio critique means for our g definition (B4)

**Papers:**
- **Gow A. "Myelin g Ratios: The Model Is in the Details."** *ASN Neuro* 18(1), online 12 Jan 2026. doi:10.1080/17590914.2025.2612034, PMID 41525149, PMC12802999 (CC BY-NC). **Full text read** 15 Nov via Europe PMC.
- Gow A. "Understanding the Myelin g Ratio from First Principles, Its Derivation, Uses and Artifacts." *ASN Neuro* 2025, PMC11877616 (= Gow 2025a in the 2026 paper). Not re-read; summarised through the 2026 paper and the commentary below.
- Gow et al. "Demystifying the Myelin g Ratio: Its Origin, Derivation and Interpretation." *ASN Neuro* 2025, PMC12360196. Not re-read.
- Fields RD, Dutta DJ. "G-Ratio Commentary—Why You've Been Doing It Wrong." *ASN Neuro* 17(1):2486962, 2025, PMC12140491. **Full text read** 15 Nov.

The 13 Nov report had only the 2026 abstract. Its claim "argues the standard g model is internally inconsistent" is confirmed, but the inconsistency is about **population-level** g estimation, not about how a single fiber's g is measured (see below).

## What Gow actually argues

1. **The principal data are per-fiber axon diameter D_A and fiber diameter D_F.** In wild-type mouse optic nerve (3 mice, 481 fibers, EM at 8 nm/px), D_A vs D_F is linear with r > 0.99.
2. **The axomyelin unit model** says D_A = γ·D_F. The population g ratio γ is the slope of that line (the "g' cline").
3. **The canonical approach is internally inconsistent:** compute g = D_A/D_F per fiber, then regress g on diameter (linear, semi-log, or just the mean). When the D_A–D_F fit has a non-zero intercept Y_int (mostly measurement error), g = slope + Y_int/D_F. So per-fiber g depends on D_F by construction, and the fitted g-vs-diameter slope is an artifact.
   - In his data the canonical intercepts were 0.80–0.85, against true slopes of 0.86–0.87.
4. **Consistent estimators:**
   - the slope of D_A on D_F (the g' cline);
   - the asymptote of the reciprocal model;
   - the corrected g_c = (D_A − Y_int)/D_F averaged over fibers.
   These three agree.
5. **Resolution:** the Y_int error term shrinks as D_F grows, and is negligible above about 1 µm. Light microscopy gives curvilinear artifacts; EM at single-digit nm resolution is adequate.
6. **Orientation:** manual EM measures **only transversely cut fibers** (judged by circular microtubule profiles). Semi-automated algorithms are vulnerable to orientation artifacts because they can't check this.
7. **Statistics (Fields & Dutta):** treating each fiber as n is pseudo-replication; use the field or animal as the unit. EM also samples fiber sizes unevenly, and tissue shrinkage (up to 65%) affects absolute values.

## Decision for our project (B4)

- **Primary per-fiber g stays the challenge's equivalent-circle g** = D_A/D_F with D = 2√(A/π). Gow does not dispute per-fiber D_A/D_F as data. What he disputes is fitting a second model to g. Our metric compares predicted and GT g *for the same fiber*, which is a measurement error, not a population model, so the critique doesn't touch it.
- **Report D_A and D_F errors separately** next to g error (Bland–Altman, T-16′). They are the principal data, and a ratio can be right while both parts are wrong.
- **For population or biological summaries** (B5, and any "does the model change the biology" claim), use the **g' cline** (the D_A-on-D_F slope with its intercept) and g_c. Never use the mean g, or a regression of g on diameter.
- **Bin errors by fiber diameter D_F, not axon diameter.** This agrees with Fields & Dutta.
- **Orientation switch:** report every g analysis with and without oblique fibers (eccentricity > 0.9; 6% of axons, [02](../../notes/02-challenge-and-data.md) §3). Our interior label-1 fibers are mostly large or oblique ones the annotators excluded, which is consistent with the classic "transverse only" EM measurement practice. Note this when interpreting selection (A3).
- **Statistical unit:** cluster bootstrap over overlap groups (D-22) already avoids per-fiber pseudo-replication for our CIs. External sets with animal IDs (TEM1, TEM2) use the animal as the unit.
- **Sensitivity definitions** (T-16′) stay as switches only: minor axis, Feret, thickness-based, mean centroid radius (optic nerve), perimeter radius (SnapG).
