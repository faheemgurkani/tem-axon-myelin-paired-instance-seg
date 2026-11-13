"""T-14 support analyses (day 5): the numbers behind the GO/NO-GO write-up beyond the H1 rule itself.

1. Paired cluster-bootstrap differences between baselines on F1@0.5/0.75 and fiber-weighted g-MAE (same resampled
   groups; D-22), and O-sep vs base on g-MAE.
2. Per matched fiber of one baseline: g error, size, and the topology flags of the *predicted* fiber
   (ring Euler ≠ 0, axon touching background, axon in > 1 component). Answers: are the broken predicted rings
   where the g error is, and is g error concentrated in small fibers?
3. Gap of each baseline to the human ceiling (results/day2/ceilings_pooled.json).

Usage: python scripts/run/day5_analysis.py [--fiber-baseline b1]
Outputs: results/day5/paired_differences.json, fibers_<name>.csv, fiber_analysis_<name>.json, ceiling_gap.json
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from axon_myelin_seg.data.io import load_sample, read_mask
from axon_myelin_seg.evaluation.diagnostics import fiber_checks
from axon_myelin_seg.evaluation.official import PIXEL_NM, fiber_table, g_ratio, match
from axon_myelin_seg.evaluation.stats import micro_f1, paired_bootstrap

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/day5"
BASELINES = ("axonpath", "cellpose_b1", "b1")
PAIRS = (("axonpath", "b1"), ("cellpose_b1", "b1"), ("axonpath", "cellpose_b1"))


def g_mae_weighted(d: pd.DataFrame) -> float:
    """Fiber-weighted g-MAE: per-image g-MAE weighted by its number of matched fibers."""
    w = d["n_matched"].to_numpy(float)
    v = d["g_mae"].to_numpy(float)
    ok = w > 0
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def fibers_one(args):
    name, pred_dir = args
    _, gt = load_sample(name)
    pred = read_mask(Path(pred_dir) / f"{name}.tif")
    t = fiber_table(pred, gt)
    pc = fiber_checks(pred)
    rows = []
    for kg, kp, iou in match(t, 0.5):
        g_gt = g_ratio(t.gt_axon_area[kg], t.gt_area[kg])
        g_p = g_ratio(t.pred_axon_area[kp], t.pred_area[kp])
        r = pc.get(kp, {})
        rows.append({
            "name": name, "gt_k": kg, "pred_k": kp, "iou": iou, "g_gt": g_gt, "g_pred": g_p, "g_err": g_p - g_gt,
            "fiber_diam_nm": 2 * np.sqrt(t.gt_area[kg] / np.pi) * PIXEL_NM,
            "ring_euler_bad": bool(r.get("ring_euler_bad", False)),
            "axon_touches_bg": bool(r.get("axon_touches_bg", False)),
            "axon_multi_cc": r.get("axon_n_cc", 1) > 1,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiber-baseline", default="b1")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. paired differences
    sc = {b: pd.read_csv(REPO / f"results/day4/scores_{b}.csv") for b in BASELINES}
    diffs = {}
    for x, y in PAIRS:
        diffs[f"{y}_minus_{x}"] = {thr: paired_bootstrap(sc[x], sc[y], lambda d, t=thr: micro_f1(d, t))
                                   for thr in ("0.5", "0.75")}
        diffs[f"{y}_minus_{x}"]["g_mae"] = paired_bootstrap(sc[x], sc[y], g_mae_weighted)
    # oracle O-sep vs base on g-MAE (same images, rows from oracles_<b>.csv)
    for bl in ("b1", "cellpose_b1"):
        o = pd.read_csv(REPO / f"results/day4/oracles_{bl}.csv")
        diffs[f"{bl}_o_sep_minus_base"] = {"g_mae": paired_bootstrap(o[o.cond == "base"], o[o.cond == "o_sep"],
                                                                     g_mae_weighted)}
    (OUT / "paired_differences.json").write_text(json.dumps(diffs, indent=1))
    print(json.dumps(diffs, indent=1))

    # 2. per-fiber topology vs g error
    b = a.fiber_baseline
    names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    with Pool(4) as p:
        res = p.map(fibers_one, [(n, REPO / "data/predictions" / b) for n in names])
    df = pd.DataFrame([r for rows in res for r in rows])
    df["any_topology_defect"] = df.ring_euler_bad | df.axon_touches_bg | df.axon_multi_cc
    df.to_csv(OUT / f"fibers_{b}.csv", index=False)
    ae = df.g_err.abs()
    bins = pd.qcut(df.fiber_diam_nm, 4)
    fa = {
        "n_matched": len(df),
        "g_mae": float(ae.mean()), "g_bias": float(df.g_err.mean()),
        "share_topology_defect": float(df.any_topology_defect.mean()),
        "share_ring_euler_bad": float(df.ring_euler_bad.mean()),
        "share_axon_touches_bg": float(df.axon_touches_bg.mean()),
        "share_axon_multi_cc": float(df.axon_multi_cc.mean()),
        "g_mae_defect": float(ae[df.any_topology_defect].mean()),
        "g_mae_clean": float(ae[~df.any_topology_defect].mean()),
        "share_of_total_abs_g_error_from_defect": float(ae[df.any_topology_defect].sum() / ae.sum()),
        "g_mae_by_diam_quartile": {str(k): float(v) for k, v in ae.groupby(bins, observed=True).mean().items()},
        "g_bias_by_diam_quartile": {str(k): float(v) for k, v in df.g_err.groupby(bins, observed=True).mean().items()},
        "defect_share_by_diam_quartile": {str(k): float(v) for k, v in
                                          df.any_topology_defect.groupby(bins, observed=True).mean().items()},
        "iou_median": float(df.iou.median()),
        "share_g_err_gt_0.05": float((ae > 0.05).mean()),
    }
    (OUT / f"fiber_analysis_{b}.json").write_text(json.dumps(fa, indent=1))
    print(json.dumps(fa, indent=1))

    # 3. gap to the human ceiling
    ceil = json.loads((REPO / "results/day2/ceilings_pooled.json").read_text())
    gap = {"ceiling": {"f1@0.5": ceil["pooled_fibers"]["f1@0.5"], "dice_axon_median": ceil["median_pairs_ge3"]["dice_axon"],
                       "dice_myelin_median": ceil["median_pairs_ge3"]["dice_myelin"],
                       "g_mae": ceil["g_mae_fiber_weighted"],
                       "violation_rate": ceil["violation_rate_pooled"]}}
    for bl in BASELINES:
        s = json.loads((REPO / f"results/day4/summary_{bl}.json").read_text())
        d = sc[bl]
        gap[bl] = {"f1@0.5": s["f1@0.5"]["est"], "dice_axon_median": float(d.dice_axon.median()),
                   "dice_myelin_median": float(d.dice_myelin.median()), "g_mae": s["g_mae_fiber_weighted"],
                   "violation_rate": s["violation_rate_pooled"]}
    (OUT / "ceiling_gap.json").write_text(json.dumps(gap, indent=1))
    print(json.dumps(gap, indent=1))


if __name__ == "__main__":
    main()
