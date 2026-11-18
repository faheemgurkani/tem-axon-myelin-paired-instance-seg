"""Phase 3a: H5a/b/c confirmation on the 213 non-dev images (D-29, 07-plan §7.2) + D-03 3-fold C1 numbers.

Input: T-21 out-of-fold B1 semantics, data/predictions/b1_cv3_sem/<name>.png (0 bg / 1 axon / 2 myelin; each image
predicted by the cv3 fold model that did not train on it).
Conditions per image:
  base     D-11 `semantic_to_paired_instances`
  h5c      frozen `semantic_to_paired_instances_h5c` (commit 7c673ca)
  o_sep    O-sep on base (GT grouping of the model's own pixels)
  o_sel, o_sel_broad   selection oracles on base (C1 numbers only)
H5b reduction share: defined in 07-plan §6 (19 Nov) before scoring.

Outputs: results/phase3a/h5_images.csv, h5_fibers.csv (per GT fiber, base vs o_sep vs h5c), h5_summary.json,
c1_cv3_summary.json.

    .venv/bin/python scripts/run/phase3a_h5.py [--workers 4]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

from axon_myelin_seg.data.io import list_samples, load_sample
from axon_myelin_seg.data.splits import image_groups
from axon_myelin_seg.evaluation.diagnostics import fiber_checks
from axon_myelin_seg.evaluation.official import fiber_table, g_ratio, match
from axon_myelin_seg.evaluation.oracles import o_sel, o_sep
from axon_myelin_seg.evaluation.runner import score_image
from axon_myelin_seg.evaluation.stats import micro_f1, paired_bootstrap
from axon_myelin_seg.evaluation.taxonomy import ring_candidates
from axon_myelin_seg.postprocessing.pairing import (
    semantic_to_paired_instances,
    semantic_to_paired_instances_h5c,
)

REPO = Path(__file__).resolve().parents[2]
SEM = REPO / "data/predictions/b1_cv3_sem"
OUT = REPO / "results/phase3a"
H5_CONDS = ("base", "h5c", "o_sep")


def per_fiber(pred: np.ndarray, gt: np.ndarray, defects: bool) -> dict[int, dict]:
    t = fiber_table(pred, gt)
    pc = fiber_checks(pred) if defects else {}
    out = {}
    for kg, kp, _ in match(t, 0.5):
        e = g_ratio(t.pred_axon_area[kp], t.pred_area[kp]) - g_ratio(t.gt_axon_area[kg], t.gt_area[kg])
        r = pc.get(kp, {})
        out[int(kg)] = {"err": float(e), "defect": bool(r.get("ring_euler_bad", False) or r.get("axon_touches_bg", False)
                                                        or r.get("axon_n_cc", 1) > 1)}
    return out


def one(name: str):
    img, gt = load_sample(name)
    sem = iio.imread(SEM / f"{name}.png")
    base = semantic_to_paired_instances(sem == 1, sem == 2)
    preds = {"base": base, "h5c": semantic_to_paired_instances_h5c(sem == 1, sem == 2), "o_sep": o_sep(base, gt, sem)}
    cand = ring_candidates(img, gt)
    b = score_image(base, gt, candidates=cand, return_classes=True)
    cls = b.pop("_classes")
    preds["o_sel"], preds["o_sel_broad"] = o_sel(base, cls), o_sel(base, cls, broad=True)
    rows = [{"name": name, "cond": "base", **b}]
    for c, p in preds.items():
        if c != "base":
            rows.append({"name": name, "cond": c, **score_image(p, gt, candidates=cand)})
    fb = {c: per_fiber(preds[c], gt, defects=(c == "base")) for c in H5_CONDS}
    ids = set().union(*fb.values())
    fibers = []
    for k in sorted(ids):
        r = {"name": name, "gt_k": k}
        for c in H5_CONDS:
            x = fb[c].get(k)
            r[f"matched_{c}"] = x is not None
            r[f"err_{c}"] = x["err"] if x else np.nan
        r["defect_base"] = fb["base"][k]["defect"] if k in fb["base"] else np.nan
        fibers.append(r)
    return rows, fibers


def gmae(d: pd.DataFrame) -> float:
    w = d.n_matched.to_numpy(float)
    v = d.g_mae.fillna(0).to_numpy(float)
    return float((v * w).sum() / max(w.sum(), 1))


def recall(d: pd.DataFrame) -> float:
    tp, fn = d["tp@0.5"].sum(), d["fn@0.5"].sum()
    return float(tp / (tp + fn))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in SEM.glob("*.png"))
    dev = set(json.loads((REPO / "configs/splits/dev.json").read_text())["val"])
    with Pool(a.workers) as p:
        res = p.map(one, names, chunksize=1)
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    df["group"] = df.name.map(image_groups(list_samples()))
    df["dev"] = df.name.isin(dev)
    F = pd.DataFrame([f for _, fs in res for f in fs])
    F["dev"] = F.name.isin(dev)
    df.to_csv(OUT / "h5_images.csv", index=False)
    F.to_csv(OUT / "h5_fibers.csv", index=False)

    # ---- H5 on the 213 non-dev images
    nd = df[~df.dev]
    by = {c: nd[nd.cond == c] for c in H5_CONDS}
    g = {c: gmae(by[c]) for c in H5_CONDS}
    d_osep = paired_bootstrap(by["base"], by["o_sep"], gmae)
    d_h5c = paired_bootstrap(by["base"], by["h5c"], gmae)
    df1 = paired_bootstrap(by["base"], by["h5c"], micro_f1)
    drec = paired_bootstrap(by["base"], by["h5c"], recall)
    rel = (g["base"] - g["o_sep"]) / g["base"]
    capture = (g["base"] - g["h5c"]) / (g["base"] - g["o_sep"]) if g["base"] != g["o_sep"] else np.nan
    f = F[~F.dev]
    fb = f[f.matched_base]
    ae = fb.err_base.abs()
    dshare = float(fb.defect_base.astype(bool).mean())
    derr = float(ae[fb.defect_base.astype(bool)].sum() / ae.sum())
    both = f[f.matched_base & f.matched_o_sep]
    red = both.err_base.abs() - both.err_o_sep.abs()
    red_share = float(red[both.defect_base.astype(bool)].sum() / red.sum()) if red.sum() else np.nan
    S = {
        "n_images": len(by["base"]), "n_groups": int(by["base"].group.nunique()),
        "g_mae": g,
        "f1@0.5": {c: micro_f1(by[c]) for c in H5_CONDS}, "recall": {c: recall(by[c]) for c in H5_CONDS},
        "H5a": {"rel_reduction": rel, "delta_gmae_osep_minus_base": d_osep,
                "holds": bool(rel >= 0.25 and d_osep["hi"] < 0)},
        "H5b": {"defective_share_of_matched": dshare, "defective_share_of_abs_g_error": derr,
                "o_sep_reduction_share_from_defective": red_share, "n_fibers_both": len(both),
                "only_base_matched": {"n": int((f.matched_base & ~f.matched_o_sep).sum()),
                                      "abs_err_sum": float(f.loc[f.matched_base & ~f.matched_o_sep, "err_base"].abs().sum())},
                "only_osep_matched": {"n": int((~f.matched_base & f.matched_o_sep).sum()),
                                      "abs_err_sum": float(f.loc[~f.matched_base & f.matched_o_sep, "err_o_sep"].abs().sum())},
                "holds": bool(dshare <= 0.35 and derr >= 0.40 and red_share >= 0.50)},
        "H5c": {"capture_of_osep_gain": capture, "delta_gmae_h5c_minus_base": d_h5c, "delta_f1": df1,
                "delta_recall": drec,
                "holds": bool(capture >= 0.5 and d_h5c["hi"] < 0 and df1["lo"] >= -0.005 and drec["lo"] >= -0.005)},
        "per_fiber_relevance_share_abs_err_gt_0.03": {
            c: float((f.loc[f[f"matched_{c}"], f"err_{c}"].abs() > 0.03).mean()) for c in H5_CONDS},
    }
    a5, c5 = S["H5a"]["holds"], S["H5c"]["holds"]
    S["gate_3a"] = ("H5 rejected (dev finding does not generalise)" if not a5 else
                    "measurement finding: grouping matters for g and a cheap rule fixes most of it" if c5 else
                    "C2 for morphometry becomes a candidate (new budget decision)")
    (OUT / "h5_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))

    # ---- D-03 3-fold C1 numbers on all 266 out-of-fold images
    C = {"n_images": int((df.cond == "base").sum())}
    for scope, d in (("all_266", df), ("non_dev_213", df[~df.dev]), ("dev_53", df[df.dev])):
        bs = d[d.cond == "base"]
        e = bs["fp@0.5"].sum() + bs["fn@0.5"].sum()
        sep = sum(bs[c].sum() for c in ("pred_gap", "pred_merge", "pred_split_piece", "gt_merged", "gt_split"))
        x = {"f1@0.5": micro_f1(bs), "errors": int(e),
             "strict_selection_share": float(bs.pred_unlabelled_real.sum() / e),
             "broad_selection_share": float((bs.pred_unlabelled_real.sum() + bs.pred_other_bg.sum()) / e),
             "separation_share": float(sep / e), "g_mae": gmae(bs)}
        for c in ("o_sel", "o_sel_broad", "o_sep"):
            x[f"gain_f1_{c}"] = paired_bootstrap(bs, d[d.cond == c], micro_f1)
        C[scope] = x
    (OUT / "c1_cv3_summary.json").write_text(json.dumps(C, indent=1, default=float))
    print(json.dumps(C, indent=1, default=float))


if __name__ == "__main__":
    main()
