"""T-28 G6 / H4 (re-registered 25 Nov, 07-plan §7.8, commit dae7a63 before any score below existed): does the
evaluator rank methods the same way on another lab's data?

Data: the 64 held-out TEM1 images (8 mice; H8's set). Inputs on the 2.152 nm grid
(`data/nnunet/phase2_inputs/tem1`), GT on the native grid (ASTIH protocol, phase2_t22_h8.load_gt).
Models:
  b1           B1 zero-shot semantics (data/predictions/b1_ext/tem1) + D-11 pairing        (exists)
  ref          TEM1 complete-label reference (data/predictions/tem1ref) + D-11 pairing        (exists; not in the rule)
  cellpose_b1  Cellpose-SAM cpsam_v2 zero-shot at 1/6 scale, regions split by B1's pixels (D-26, unchanged)
  axonpath     AxonPath `em` weights zero-shot at 8 nm/px, inner cylinder as axon (D-13, unchanged)
Steps:
  --predict    writes data/predictions/h4_tem1/{cellpose_b1,axonpath}/<case>.tif (2.152 grid, local MPS)
  --score      per image: F1@0.5 counts, g errors of matched fibers, taxonomy; mouse-bootstrap paired CIs;
               H4 verdict as registered; GQ_0.1 (secondary). Writes results/paper/h4_tem1.json and per-image CSV.

    PYTHONPATH=src:scripts/run .venv/bin/python scripts/run/paper_h4_tem1.py --predict [--limit 3]
    PYTHONPATH=src:scripts/run .venv/bin/python scripts/run/paper_h4_tem1.py --score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "baselines/axonpath"))
sys.path.insert(0, str(REPO / "scripts/run"))
INP = REPO / "data/nnunet/phase2_inputs/tem1"
PRED = REPO / "data/predictions/h4_tem1"
OUT = REPO / "results/paper"
TEST_MICE = {f"sub-nyuMouse{n}" for n in (13, 14, 31, 32, 33, 34, 26, 27)}
MODELS = ["b1", "cellpose_b1", "axonpath", "ref"]
B = 1000


def manifest() -> pd.DataFrame:
    m = pd.read_csv(INP / "manifest.csv")
    return m[m.subject.isin(TEST_MICE)].reset_index(drop=True)


def predict(limit: int) -> None:
    import torch
    from cellpose import models as cpm
    from day4_postprocess import cellpose_split
    from run import SCALE, to_challenge

    import axonpath
    from axonpath.inference.inference import load_model, run_inference
    from axonpath.utils.image_processing import normalize

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ap_model = load_model(str(Path(axonpath.__file__).parent / "weights" / "em.pth"), device=dev)
    cp_model = cpm.CellposeModel(gpu=True, pretrained_model="cpsam_v2")
    for d in ("cellpose_b1", "axonpath"):
        (PRED / d).mkdir(parents=True, exist_ok=True)
    rows = manifest()
    rows = rows.head(limit) if limit else rows
    for r in rows.itertuples():
        t0 = time.time()
        img = iio.imread(INP / f"{r.case}_0000.png")
        img = img if img.ndim == 2 else img[..., 0]
        H, W = img.shape
        # AxonPath
        small = cv2.resize(img, (round(W * SCALE), round(H * SCALE)), interpolation=cv2.INTER_AREA)
        fibre, axon, inner, _ = run_inference(normalize(small.astype(np.float32)), ap_model, device=dev, min_diameter=20.0)
        up = [cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
              for x in (fibre, inner, axon)]
        pred, _, _ = to_challenge(*up)
        tifffile.imwrite(PRED / "axonpath" / f"{r.case}.tif", pred, compression="zlib")
        t1 = time.time()
        # Cellpose + B1
        small = cv2.resize(img, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
        masks = cp_model.eval(small, diameter=None)[0]
        inst = cv2.resize(masks.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        sem = iio.imread(REPO / "data/predictions/b1_ext/tem1" / f"{r.case}.png").astype(np.uint8)
        tifffile.imwrite(PRED / "cellpose_b1" / f"{r.case}.tif", cellpose_split(inst, sem), compression="zlib")
        print(f"{r.case} axonpath {t1 - t0:.1f}s cellpose {time.time() - t1:.1f}s", flush=True)


def load_pred(model: str, row) -> np.ndarray:
    from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances
    if model in ("b1", "ref"):
        p = (REPO / "data/predictions/b1_ext/tem1" if model == "b1" else REPO / "data/predictions/tem1ref") / f"{row.case}.png"
        sem = cv2.resize(iio.imread(p), (row.w, row.h), interpolation=cv2.INTER_NEAREST)
        return semantic_to_paired_instances(sem == 1, sem == 2)
    inst = tifffile.imread(PRED / model / f"{row.case}.tif")
    return cv2.resize(inst.astype(np.float32), (row.w, row.h), interpolation=cv2.INTER_NEAREST).astype(np.uint16)


def score_one(args):
    from axon_myelin_seg.evaluation.official import fiber_table, g_ratio, match
    from axon_myelin_seg.evaluation.taxonomy import classify_errors, ring_candidates
    from phase2_t22_h8 import load_gt
    row = args
    img, gt = load_gt(row)
    cand = ring_candidates(img, gt, ds=4, px_nm=row.px_nm)
    out = []
    for model in MODELS:
        pred = load_pred(model, row)
        t = fiber_table(pred, gt)
        m = match(t, 0.5)
        dg = [g_ratio(t.pred_axon_area[b], t.pred_area[b]) - g_ratio(t.gt_axon_area[a], t.gt_area[a]) for a, b, _ in m]
        c = classify_errors(pred, gt, cand)["counts"]
        out.append({"model": model, "case": row.case, "subject": row.subject, "tp": len(m),
                    "fp": len(t.pred_ids) - len(m),  # same convention as phase2_t22_h8 (pred_ids exclude label-1 fibers)
                    "fn": len(t.gt_ids) - len(m), "abs_dg_sum": float(np.sum(np.abs(dg))),
                    "gq_sum": float(np.sum(np.maximum(0, 1 - np.abs(dg) / 0.10))), **c})
    return out


def agg(d: pd.DataFrame) -> dict:
    tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
    f1 = 2 * tp / (2 * tp + fp + fn)
    err = d[[c for c in d.columns if c.startswith(("pred_", "gt_")) and c != "pred_on_label1"]].sum().sum()
    return {"f1": f1, "g_mae": d.abs_dg_sum.sum() / tp, "recall": tp / (tp + fn), "gq": f1 * d.gq_sum.sum() / tp,
            "broad_sel": (d.pred_unlabelled_real.sum() + d.pred_other_bg.sum()) / err,
            "separation": d[["pred_gap", "pred_merge", "pred_split_piece", "gt_merged", "gt_split"]].sum().sum() / err}


def score() -> None:
    rows = [SimpleNamespace(**r) for r in manifest().to_dict("records")]
    with Pool(4) as p:
        res = [x for xs in p.map(score_one, rows) for x in xs]
    D = pd.DataFrame(res)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "h4_tem1_images.csv", index=False)
    mice = sorted(D.subject.unique())
    by = {(m, s): g for (m, s), g in D.groupby(["model", "subject"])}

    def stats(sel):
        return {m: agg(pd.concat([by[(m, s)] for s in sel])) for m in MODELS}

    pt = stats(mice)
    rng = np.random.default_rng(0)
    draws = [stats(list(rng.choice(mice, len(mice)))) for _ in range(B)]

    def ci(f):
        v = np.array([f(x) for x in draws])
        return [float(f(pt)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    diffs = {"f1_b1_minus_cellpose": ci(lambda s: s["b1"]["f1"] - s["cellpose_b1"]["f1"]),
             "f1_cellpose_minus_axonpath": ci(lambda s: s["cellpose_b1"]["f1"] - s["axonpath"]["f1"]),
             "gmae_b1_minus_cellpose": ci(lambda s: s["b1"]["g_mae"] - s["cellpose_b1"]["g_mae"]),
             "gmae_axonpath_minus_cellpose": ci(lambda s: s["axonpath"]["g_mae"] - s["cellpose_b1"]["g_mae"])}
    f1_order = pt["b1"]["f1"] > pt["cellpose_b1"]["f1"] > pt["axonpath"]["f1"]
    g_order = pt["cellpose_b1"]["g_mae"] < pt["b1"]["g_mae"] < pt["axonpath"]["g_mae"]
    all_ci = all(v[1] > 0 for v in diffs.values())  # all four registered differences are oriented to be > 0
    verdict = "holds" if f1_order and g_order and all_ci else ("holds partially" if f1_order and g_order else "fails")
    S = {"registered": "07-plan §7.8, commit dae7a63", "n_images": len(rows), "n_mice": len(mice),
         "per_model": {m: {k: float(v) for k, v in pt[m].items()} for m in MODELS},
         "per_model_ci95": {m: {k: ci(lambda s, m=m, k=k: s[m][k]) for k in pt[m]} for m in MODELS},
         "differences_ci95": diffs, "f1_order_reproduced": f1_order, "gmae_order_reproduced": g_order,
         "H4": verdict}
    (OUT / "h4_tem1.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: S[k] for k in ("per_model", "differences_ci95", "f1_order_reproduced",
                                        "gmae_order_reproduced", "H4")}, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.predict:
        predict(a.limit)
    if a.score:
        score()


if __name__ == "__main__":
    main()
