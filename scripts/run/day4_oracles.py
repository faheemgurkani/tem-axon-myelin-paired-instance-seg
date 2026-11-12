"""T-13: oracle experiments on one baseline's dev-val predictions.

For each image: score the baseline, O-sel (strict and broad), O-sep and O-bnd (src/.../evaluation/oracles.py).
Then pool, with *paired* cluster-bootstrap CIs of each oracle's gain over the baseline (same resampled groups;
D-22). The H1 numbers come from here.

Usage: python scripts/run/day4_oracles.py --pred-dir data/predictions/<name> --name <name> [--sem-dir <png dir>]
Outputs: results/day4/oracles_<name>.csv (per image × condition) and oracles_<name>.json
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

from axon_myelin_seg.data.io import list_samples, load_sample, read_mask
from axon_myelin_seg.data.splits import image_groups
from axon_myelin_seg.evaluation.oracles import o_bnd, o_sel, o_sep
from axon_myelin_seg.evaluation.runner import score_image
from axon_myelin_seg.evaluation.stats import micro_f1, paired_bootstrap
from axon_myelin_seg.evaluation.taxonomy import ring_candidates

REPO = Path(__file__).resolve().parents[2]
CONDS = ("base", "o_sel", "o_sel_broad", "o_sep", "o_bnd")


def one(args):
    name, pred_dir, sem_dir = args
    img, gt = load_sample(name)
    pred = read_mask(Path(pred_dir) / f"{name}.tif")
    sem = None
    if sem_dir:
        sem = iio.imread(Path(sem_dir) / f"{name}.png")
    cand = ring_candidates(img, gt)
    base = score_image(pred, gt, candidates=cand, return_classes=True)
    cls = base.pop("_classes")
    preds = {"o_sel": o_sel(pred, cls), "o_sel_broad": o_sel(pred, cls, broad=True),
             "o_sep": o_sep(pred, gt, sem), "o_bnd": o_bnd(pred, gt)}
    rows = [{"name": name, "cond": "base", **base}]
    for c, p in preds.items():
        rows.append({"name": name, "cond": c, **score_image(p, gt, candidates=cand)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sem-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    with Pool(a.workers) as p:
        res = p.map(one, [(n, a.pred_dir, a.sem_dir) for n in names])
    df = pd.DataFrame([r for rows in res for r in rows])
    df["group"] = df.name.map(image_groups(list_samples()))
    out = REPO / "results" / "day4"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"oracles_{a.name}.csv", index=False)

    base = df[df.cond == "base"]
    summ = {"baseline": a.name, "n_images": len(base)}
    for c in CONDS:
        d = df[df.cond == c]
        w = d.n_matched.sum()
        summ[c] = {"f1@0.5": micro_f1(d), "f1@0.75": micro_f1(d, "0.75"),
                   "tp": int(d["tp@0.5"].sum()), "fp": int(d["fp@0.5"].sum()), "fn": int(d["fn@0.5"].sum()),
                   "dice_axon": float(d.dice_axon.mean()), "dice_myelin": float(d.dice_myelin.mean()),
                   "hd95_myelin_nm": float(d.hd95_myelin_nm.mean()),
                   "g_mae": float((d.g_mae.fillna(0) * d.n_matched).sum() / max(w, 1)),
                   "n_merges": int(d.n_merges.sum()), "n_splits": int(d.n_splits.sum())}
        if c != "base":
            summ[c]["gain_f1@0.5"] = paired_bootstrap(base, d, micro_f1)
            summ[c]["gain_f1@0.75"] = paired_bootstrap(base, d, lambda x: micro_f1(x, "0.75"))
    errs = base["fp@0.5"].sum() + base["fn@0.5"].sum()
    sep = sum(base[c].sum() for c in ("pred_gap", "pred_merge", "pred_split_piece", "gt_merged", "gt_split"))
    summ["separation_share_of_errors"] = float(sep / errs) if errs else float("nan")
    summ["errors_total"] = int(errs)
    g_sep = summ["o_sep"]["gain_f1@0.5"]
    g_sel = summ["o_sel"]["gain_f1@0.5"]
    summ["H1"] = {
        "sep_gain_ge_0.03": g_sep["diff"] >= 0.03,
        "sep_gain_ge_sel_gain": g_sep["diff"] >= g_sel["diff"],
        "sep_gain_ge_sel_broad_gain": g_sep["diff"] >= summ["o_sel_broad"]["gain_f1@0.5"]["diff"],
        "separation_share_ge_0.30": summ["separation_share_of_errors"] >= 0.30,
    }
    summ["H1"]["GO"] = bool(summ["H1"]["sep_gain_ge_0.03"] and summ["H1"]["sep_gain_ge_sel_gain"]
                            and summ["H1"]["separation_share_ge_0.30"])
    (out / f"oracles_{a.name}.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
