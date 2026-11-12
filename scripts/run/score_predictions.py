"""T-12: score a folder of uint16 challenge-encoded predictions on the dev-val images.

Per image: official metrics + diagnostics + taxonomy counts (src/axon_myelin_seg/evaluation/runner.py).
Pooled: micro F1@0.5/0.75 with cluster-bootstrap CIs over overlap groups (D-22), mean Dice/HD95, fiber-weighted
g-MAE, and the error-family shares used by H1.

Usage: python scripts/run/score_predictions.py --pred-dir data/predictions/<name> --name <name> [--workers 4]
Outputs: results/day4/scores_<name>.csv, results/day4/summary_<name>.json
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from axon_myelin_seg.data.io import list_samples, load_sample, read_mask
from axon_myelin_seg.data.splits import image_groups
from axon_myelin_seg.evaluation.runner import score_image
from axon_myelin_seg.evaluation.stats import cluster_bootstrap, micro_f1

REPO = Path(__file__).resolve().parents[2]
SEP_PRED = ("pred_gap", "pred_merge", "pred_split_piece")
SEP_GT = ("gt_merged", "gt_split")


def one(args):
    name, pred_dir = args
    img, gt = load_sample(name)
    pred = read_mask(Path(pred_dir) / f"{name}.tif")
    r = score_image(pred, gt, image=img)
    r["name"] = name
    return r


def summarise(df: pd.DataFrame) -> dict:
    errs = df["fp@0.5"].sum() + df["fn@0.5"].sum()
    sep = sum(df[c].sum() for c in SEP_PRED + SEP_GT)
    w = df.n_matched.sum()
    out = {
        "n_images": len(df), "n_gt": int(df.n_gt.sum()), "n_pred": int(df.n_pred.sum()),
        "n_pred_ignored_label1": int(df.n_pred_ignored.sum()),
        "f1@0.5": cluster_bootstrap(df, micro_f1), "f1@0.75": cluster_bootstrap(df, lambda x: micro_f1(x, "0.75")),
        "recall@0.5": float(df["tp@0.5"].sum() / df.n_gt.sum()),
        "precision@0.5": float(df["tp@0.5"].sum() / max(df.n_pred.sum(), 1)),
        "dice_axon_mean": float(df.dice_axon.mean()), "dice_myelin_mean": float(df.dice_myelin.mean()),
        "hd95_axon_nm_mean": float(df.hd95_axon_nm.mean()), "hd95_myelin_nm_mean": float(df.hd95_myelin_nm.mean()),
        "g_mae_fiber_weighted": float((df.g_mae.fillna(0) * df.n_matched).sum() / max(w, 1)),
        "g_bias_fiber_weighted": float((df.g_bias.fillna(0) * df.n_matched).sum() / max(w, 1)),
        "pq_mean": float(df.pq.mean()),
        "pairing_correct_pooled": float(df.n_pairing_correct.sum() / max(w, 1)),
        "violation_rate_pooled": float(df.n_violation.sum() / max(df.n_pred_objects.sum(), 1)),
        "n_merges": int(df.n_merges.sum()), "n_splits": int(df.n_splits.sum()),
        "errors_total": int(errs), "separation_errors": int(sep),
        "separation_share": float(sep / errs) if errs else float("nan"),
    }
    for c in [c for c in df.columns if c.startswith(("pred_", "gt_"))]:
        out[c] = int(df[c].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    with Pool(a.workers) as p:
        rows = p.map(one, [(n, a.pred_dir) for n in names])
    df = pd.DataFrame(rows)
    df["group"] = df.name.map(image_groups(list_samples()))
    out = REPO / "results" / "day4"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"scores_{a.name}.csv", index=False)
    s = summarise(df)
    (out / f"summary_{a.name}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
