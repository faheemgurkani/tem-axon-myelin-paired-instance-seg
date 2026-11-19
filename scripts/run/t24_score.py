"""T-24 step 6 (C3): score the arms on the 8 held-out TEM1 mice and decide H9 (07-plan §7.6; D-42).

Written 19 Nov before any arm was trained. Predictions: semantic PNGs on the 2.152 nm grid from each arm's final
validation (fold 0 = the test mice), in data/predictions/t24/<arm>/<case>.png; the complete-label reference
(Dataset502) in data/predictions/tem1ref/. Pairing: D-11 `semantic_to_paired_instances` (B1's rule).
GT: full test labels (data/processed/t24/gt_pairs, 2.152 grid); stratum per fiber from fibers.csv.

Per arm:
- primary: normal-fiber recall at IoU > 0.5 (normal = not in the stratum = known positives);
- guard: protocol-violation rate = share of stratum test fibers matched by a predicted fiber;
- hard-normal recall: the tercile of normal test fibers with the highest â (the D-42 a(x) model applied to
  same-pipeline features from B1 zero-shot semantics inside the GT footprint; fixed across arms);
- protocol-truth F1@0.5 (stratum fibers removed from GT, so predicting them counts as FP), g-MAE with recall.
H9: R4 − R2 > 0 and R4 − R3 > 0 on the primary (paired mouse bootstrap, 1,000 resamples, CI lower bounds > 0)
and violation(R4) ≤ violation(R1) + 0.05.

    .venv/bin/python scripts/run/t24_score.py [--arms R1 R2 R3 R4 ref]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from t24_candidates import footprint_features

from axon_myelin_seg.evaluation.official import fiber_table, g_ratio, match
from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances

REPO = Path(__file__).resolve().parents[2]
T24 = REPO / "data/processed/t24"
RES = REPO / "results/phase3b"
PRED = {a: REPO / "data/predictions/t24" / a for a in ("R1", "R2", "R3", "R4", "R5", "R6")}
PRED["ref"] = REPO / "data/predictions/tem1ref"


def a_hat(case: str, ks: list[int], model: dict) -> dict[int, float]:
    gtp = iio.imread(T24 / "gt_pairs" / f"{case}.png").astype(np.int64)
    sem = iio.imread(REPO / "data/predictions/b1_ext/tem1" / f"{case}.png")
    img = iio.imread(REPO / "data/nnunet/phase2_inputs/tem1" / f"{case}_0000.png")
    img = img if img.ndim == 2 else img[..., 0]
    fid = np.where(gtp >= 2, gtp // 2, 0)
    out = {}
    for p in regionprops(fid):
        if p.label not in ks:
            continue
        m = fid[p.slice] == p.label
        f = footprint_features(p, sem[p.slice][m], img[p.slice][m])
        x = np.array([f[c] if np.isfinite(f[c]) else model["median"][c] for c in model["feats"]])
        z = (x - np.array([model["mean"][c] for c in model["feats"]])) / np.array([model["sd"][c] for c in model["feats"]])
        out[p.label] = float(1 / (1 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"]))))
    return out


def score_case(args):
    arm, case, normal_k, stratum_k = args
    sem = iio.imread(PRED[arm] / f"{case}.png")
    gt = iio.imread(T24 / "gt_pairs" / f"{case}.png").astype(np.int64)
    pred = semantic_to_paired_instances(sem == 1, sem == 2)
    t = fiber_table(pred, gt)
    matched = {int(kg) for kg, _, _ in match(t, 0.5)}
    # protocol truth: stratum fibers -> background
    fid = np.where(gt >= 2, gt // 2, 0)
    gp = np.where(np.isin(fid, stratum_k), 0, gt).astype(np.uint16)
    tp_ = fiber_table(pred, gp)
    mp = match(tp_, 0.5)
    gerr = [abs(g_ratio(tp_.pred_axon_area[kp], tp_.pred_area[kp]) - g_ratio(tp_.gt_axon_area[kg], tp_.gt_area[kg]))
            for kg, kp, _ in mp]
    return ({"arm": arm, "case": case, "tp": len(mp), "fp": len(tp_.pred_ids) - len(mp), "fn": len(tp_.gt_ids) - len(mp),
             "g_abs_sum": float(np.sum(gerr)), "n_g": len(gerr)},
            [{"arm": arm, "case": case, "k": k, "matched": k in matched} for k in list(normal_k) + list(stratum_k)])


def stats(fb: pd.DataFrame, im: pd.DataFrame) -> dict:
    n = fb[~fb.stratum]
    tp, fp, fn = im.tp.sum(), im.fp.sum(), im.fn.sum()
    return {"normal_recall": float(n.matched.mean()), "violation": float(fb[fb.stratum].matched.mean()),
            "hard_normal_recall": float(n[n.hard].matched.mean()), "f1_protocol": float(2 * tp / (2 * tp + fp + fn)),
            "g_mae": float(im.g_abs_sum.sum() / max(im.n_g.sum(), 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["R1", "R2", "R3", "R4", "ref"])
    arms = [a for a in ap.parse_args().arms if any(PRED[a].glob("*.png"))]
    F = pd.read_csv(T24 / "fibers.csv")
    F = F[F.test]
    model = json.loads((RES / "t24_selection_summary.json").read_text())["a_model"]
    A = {}
    for case, d in F[~F.stratum].groupby("case"):
        A.update({(case, k): v for k, v in a_hat(case, list(d.k), model).items()})
    F["a_hat"] = [A.get((c, k), np.nan) for c, k in zip(F.case, F.k, strict=True)]
    thr = F.loc[~F.stratum, "a_hat"].quantile(2 / 3)
    F["hard"] = ~F.stratum & (F.a_hat >= thr)
    jobs = [(a, c, list(d[~d.stratum].k), list(d[d.stratum].k)) for a in arms for c, d in F.groupby("case")]
    with Pool(4) as p:
        res = p.map(score_case, jobs)
    IM = pd.DataFrame([r for r, _ in res])
    FB = pd.DataFrame([x for _, xs in res for x in xs]).merge(F[["case", "k", "subject", "stratum", "hard"]],
                                                            on=["case", "k"])
    IM = IM.merge(F[["case", "subject"]].drop_duplicates(), on="case")
    FB.to_csv(T24 / "score_fibers.csv", index=False)
    S = {"arms": {a: stats(FB[FB.arm == a], IM[IM.arm == a]) for a in arms},
         "n_normal": int((~F.stratum).sum()), "n_stratum": int(F.stratum.sum()), "n_hard": int(F.hard.sum())}
    mice = sorted(F.subject.unique())
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        pick = rng.choice(mice, len(mice))
        fb = pd.concat([FB[FB.subject == m] for m in pick])
        im = pd.concat([IM[IM.subject == m] for m in pick])
        boots.append({a: stats(fb[fb.arm == a], im[im.arm == a]) for a in arms})
    if {"R1", "R2", "R3", "R4"} <= set(arms):
        def ci(fn):
            v = [fn(b) for b in boots]
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        st = S["arms"]
        d42 = st["R4"]["normal_recall"] - st["R2"]["normal_recall"]
        d43 = st["R4"]["normal_recall"] - st["R3"]["normal_recall"]
        c42 = ci(lambda b: b["R4"]["normal_recall"] - b["R2"]["normal_recall"])
        c43 = ci(lambda b: b["R4"]["normal_recall"] - b["R3"]["normal_recall"])
        guard = st["R4"]["violation"] - st["R1"]["violation"]
        S["H9"] = {"R4_minus_R2": d42, "R4_minus_R2_ci95": c42, "R4_minus_R3": d43, "R4_minus_R3_ci95": c43,
                   "violation_R4_minus_R1": guard,
                   "holds": bool(c42[0] > 0 and c43[0] > 0 and guard <= 0.05),
                   "hard_normal_R4_minus_R2_ci95": ci(lambda b: b["R4"]["hard_normal_recall"]
                                                      - b["R2"]["hard_normal_recall"])}
    (RES / "t24_scores.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
