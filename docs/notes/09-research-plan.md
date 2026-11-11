# 09 — Research Plan

We are **not** bound by the challenge timeline (registration closed 9 Nov 2026; submission 13 Nov 2026). The
challenge is our problem statement, benchmark protocol and research gap.

## Phases

| Phase | Goal | Output | Exit criterion |
|---|---|---|---|
| **P0 Setup** | Repo, environment, data access, organizer questions ([08](08-open-questions-and-risks.md)) | This repo; data under `data/raw/` | Data loads; label encoding validated on all 266 masks |
| **P1 Evaluation framework** | Official-metric replica + diagnostics + morphometry + frozen splits ([05](05-evaluation-framework.md)) | `evaluation/`, `configs/splits/` | Unit tests pass; GT-vs-GT scores perfect; perturbed-GT behaves sensibly |
| **P2 Baselines** | B1–B4 (+B5) with shared pairing ([04](04-baselines.md)) | `baselines/*`, results table | Every baseline scored on the same split and settings |
| **P3 Error analysis** | Failure-type × fiber-size breakdown | `docs/notes/10-error-analysis.md` | 1–2 dominant failure modes identified |
| **P4 Our method** | Targeted method ([06](06-research-directions.md)) | `src/axon_myelin_seg/models/` | Beats B1 on HD95 and g-MAE with CIs; external-set results |
| **P5 Write-up** | SIPAIM-style 2–4 page note → full paper | `paper/` | Draft ready for target venue |

## Compressed challenge-style schedule (reference, from the initial study)

| Days | Work |
|---|---|
| 1 | Evaluator + train/val split |
| 2–3 | 3 baselines with shared pairing |
| 4 | Error analysis → choose 1–2 targeted fixes |
| 5–6 | Fixes, TTA, ensemble |
| 7 | Test inference, package code and weights |
| 8 | Method PDF |

## Log

- 2026-11-10: repo scaffolded; challenge site crawled; literature review preserved.
- 2026-11-11: deep-research pass done (5 research streams plus a final report in `docs/reports/`). Notes 03–08 and `configs/default.yaml` updated. **Next:** inspect the 266 training masks (inner tongue, sheath splitting, scale bar, filename metadata), then implement `evaluation/official.py` and validate it against ASTIH TEM1 (B0).
- 2026-11-11: feasibility check for a 2-week budget. The gap still holds: a fresh search found no paper on nested/paired axon–myelin instances or g-ratio-trained segmentation. **Biggest risk:** access to the challenge data, since we are not registered. **Fallback:** run the whole study on public CC-BY data (ASTIH TEM1 + AxonPath). Two-week scope: evaluator + B1/B3 + oracle-pairing (week 1), then the nested decoder + g-ratio loss vs B1 (week 2). The dual-radius head, SSL pretraining and μSAM are deferred.
- 2026-11-12: gap-validation pass (`docs/reports/Paired axon myelin gap validation.md`). The gap is open but narrower than first stated. AxonPath (Sep 2026) is a multi-head joint model, but it pairs with a 90% overlap rule. NPS-Net, Alblas 2022 and Bi-Polar Mask already cover nested polar contours, so the dual-radius head is no longer the novelty. Meng 2022 is prior art for ratio losses. Novelty now rests on dense multi-instance nesting by construction, a per-fiber g-ratio loss and a pairing-correctness metric. Plan changes: add AxonPath as a week-1 baseline; oracle pairing becomes the go/no-go experiment; still need the Bi-Polar Mask full text.
- 2026-11-12: training data received (266 images + masks) and EDA done ([10](10-eda.md)). Conventions are now settled: border-cut fibers are label 1, the inner tongue is part of the axon, IDs are not consecutive, and 47 orphan labels exist. **New risk:** selective annotation. Abnormal myelinated fibers are background (~25% of ring-like structures), so fiber *selection* becomes a hidden task, and the go/no-go experiment must separate selection errors from pairing errors. Fibers are large (median axon 562 nm), so g is less sensitive than the report assumed.
- 2026-11-12: deep EDA ([11](11-deep-eda.md)). Found **leakage**: 22 overlapping tile pairs in 18 groups (40 images), saved in `configs/splits/overlap_groups.json` and to be used as split groups. **FP patterns:** ~4–8% of real fibers are unlabelled (random misses, not a rule), interstitial gaps between touching fibers act as pseudo-fibers, and interior label 1 marks large or oblique fibers. **Ceilings from re-annotated overlaps:** myelin Dice ~0.93–0.96, g-MAE floor ~0.006. The g definition spread (0.065) is 10× annotation noise. Next: an evaluator with an FP taxonomy (unlabelled fiber / interstitial gap / label 1 / other).
