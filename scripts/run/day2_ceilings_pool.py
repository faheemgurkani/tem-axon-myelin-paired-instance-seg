"""Pool T-07 per-pair ceilings into results/day2/ceilings_pooled.json (pairs with NCC ≥ 0.8)."""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
d = pd.read_csv(REPO / "results/day2/ceilings_pairs.csv")
u = d[d.ncc >= 0.8]
s3 = u[u.n_gt >= 3]
tot = {t: u[f"{t}@0.5"].sum() for t in ("tp", "fp", "fn")}
w = u.n_matched.sum()
out = {"pairs_ncc_ge_0.8": len(u), "pairs_with_ge3_gt_fibers": len(s3),
       "pooled_fibers": {**{k: int(v) for k, v in tot.items()},
                         "f1@0.5": float(2 * tot["tp"] / (2 * tot["tp"] + tot["fp"] + tot["fn"])),
                         "f1@0.75": float(2 * u["tp@0.75"].sum() / (2 * u["tp@0.75"].sum() + u["fp@0.75"].sum()
                                                                    + u["fn@0.75"].sum()))},
       "median_pairs_ge3": {c: float(s3[c].median()) for c in ("dice_axon", "dice_myelin", "hd95_axon_nm",
                                                                "hd95_myelin_nm")},
       "iqr_pairs_ge3": {c: [float(s3[c].quantile(.25)), float(s3[c].quantile(.75))] for c in ("dice_axon",
                                                                                              "dice_myelin")},
       "g_mae_fiber_weighted": float((u.g_mae.fillna(0) * u.n_matched).sum() / w),
       "g_bias_fiber_weighted": float((u.g_bias.fillna(0) * u.n_matched).sum() / w),
       "pairing_correct_pooled": float(u.n_pairing_correct.sum() / w),
       "violation_rate_pooled": float(u.n_violation.sum() / u.n_pred_objects.sum()),
       "n_matched_fibers": int(w), "merges": int(u.n_merges.sum()), "splits": int(u.n_splits.sum())}
(REPO / "results/day2/ceilings_pooled.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
