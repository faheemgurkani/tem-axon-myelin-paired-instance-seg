# Results (committed): metric tables and logs only

No images, masks or predictions are kept here. Each file is written by the script named below. The numbers are explained in `docs/notes/05-results.md`.

| Folder | Files | Written by |
|---|---|---|
| `day1/` | `images.csv` (per-image load check, fiber counts, density), `cleaning_summary.json` (orphans / defects set to ignore), `*_v1_superseded.json` (old leaky splits, kept for traceability) | `scripts/run/day1_data.py` |
| `day2/` | `ceilings_pairs.csv`, `ceilings_pooled.json` (human-vs-human ceiling), `registration_scan_all710.csv` (all ORB candidate pairs re-registered), `*_pre_violation_fix.csv`, run logs | `scripts/run/day2_ceilings.py`, `day2_ceilings_pool.py` |
| `day3/` | `r1_*` (pairing rule on GT pixels), `nnunet_label_fractions.csv`, `nnUNetResEncUNetMPlans.json` (plans with the `2d_bigpatch` config), AxonPath and Cellpose run logs | `day3_r1_pairing_on_gt.py`, `baselines/nnunet/convert.py`, `baselines/axonpath/run.py`, `baselines/cellpose_sam/zeroshot.py` |
| `day4/` | `scores_<baseline>.csv` (per image), `summary_<baseline>.json` (pooled F1 with CIs, taxonomy counts), `oracles_<baseline>.{csv,json}` (oracle gains with paired CIs, H1 flags) | `score_predictions.py`, `day4_oracles.py` |
| `day5/` | `paired_differences.json`, `fiber_analysis_b1.json` (ring topology vs g error), `ceiling_gap.json`. The per-fiber `fibers_*.csv` stays local (derived from the GT) | `day5_analysis.py` |

Baseline names: `axonpath` (zero-shot), `b1` (nnU-Net + watershed), `cellpose_b1` (Cellpose-SAM zero-shot regions split by B1's pixels).
