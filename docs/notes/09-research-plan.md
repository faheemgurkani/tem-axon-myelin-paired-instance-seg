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
