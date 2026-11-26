"""T-30 (G1, H11; 07-plan §7.9): inference and the H11 verdict for CP-FT, Cellpose-SAM fine-tuned on our labels.

--predict dev   the 53 dev-split val images  -> data/predictions/cellpose_ft/<name>.tif   (local MPS)
--predict tem1  the 64 held-out TEM1 images  -> data/predictions/h11_tem1/<case>.tif      (2.152 nm inputs)
Both: image /6 (INTER_AREA), fiber and axon models, instance maps upsampled (nearest); pairing by containment
(day4_postprocess.cellpose_split) with sem = 1 on the axon model's mask and 2 on the rest of the fiber mask. No B1 pixel.
--h11           H11a from results/day4/scores_cellpose_ft.csv (run score_predictions.py first) with an overlap-group
                bootstrap of the shares, plus paired CP-FT - B1 differences; H11b on TEM1 against the reference
                (H8 rule). Writes results/paper/h11.json.

    PYTHONPATH=src:scripts/run .venv/bin/python scripts/run/g1_cellpose_eval.py --predict dev
    .venv/bin/python scripts/run/score_predictions.py --pred-dir data/predictions/cellpose_ft --name cellpose_ft
    .venv/bin/python scripts/run/day4_oracles.py --pred-dir data/predictions/cellpose_ft --name cellpose_ft
    PYTHONPATH=src:scripts/run .venv/bin/python scripts/run/g1_cellpose_eval.py --predict tem1
    PYTHONPATH=src:scripts/run .venv/bin/python scripts/run/g1_cellpose_eval.py --h11
"""
from __future__ import annotations

import argparse
import json
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
MODELS = REPO / "data/models/g1_cellpose"
OUT = REPO / "results/paper"
TEST_MICE = {f"sub-nyuMouse{n}" for n in (13, 14, 31, 32, 33, 34, 26, 27)}
SEP = ["pred_gap", "pred_merge", "pred_split_piece", "gt_merged", "gt_split"]
DS = 6
B = 1000


def predict(which: str, limit: int) -> None:
    from cellpose import models
    from day4_postprocess import cellpose_split

    from axon_myelin_seg.data.io import load_sample, write_mask
    fm = models.CellposeModel(gpu=True, pretrained_model=str(MODELS / "fiber_cpsam_ft"))
    am = models.CellposeModel(gpu=True, pretrained_model=str(MODELS / "axon_cpsam_ft"))
    if which == "dev":
        names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
        out = REPO / "data/predictions/cellpose_ft"
        items = [(n, None) for n in names]
    else:
        man = pd.read_csv(REPO / "data/nnunet/phase2_inputs/tem1/manifest.csv")
        man = man[man.subject.isin(TEST_MICE)]
        out = REPO / "data/predictions/h11_tem1"
        items = [(r.case, REPO / "data/nnunet/phase2_inputs/tem1" / f"{r.case}_0000.png") for r in man.itertuples()]
    out.mkdir(parents=True, exist_ok=True)
    items = items[:limit] if limit else items
    for name, path in items:
        t0 = time.time()
        img = load_sample(name)[0] if path is None else iio.imread(path)
        img = img if img.ndim == 2 else img[..., 0]
        H, W = img.shape
        small = cv2.resize(img, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
        fi = fm.eval(small, diameter=None)[0]
        ai = am.eval(small, diameter=None)[0]
        up = [cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32) for x in (fi, ai)]
        sem = np.zeros((H, W), np.uint8)
        sem[up[0] > 0] = 2
        sem[up[1] > 0] = 1
        pred = cellpose_split(up[0], sem)
        if path is None:
            write_mask(out / f"{name}", pred) if name.endswith(".tif") else write_mask(out / f"{name}.tif", pred)
        else:
            tifffile.imwrite(out / f"{name}.tif", pred, compression="zlib")
        print(f"{name} fibers {int(fi.max())} axons {int(ai.max())} paired {int(pred.max() // 2)} {time.time() - t0:.1f}s",
              flush=True)


def shares(d: pd.DataFrame) -> dict:
    err = d["fp@0.5"].sum() + d["fn@0.5"].sum()
    s = d.pred_unlabelled_real.sum()
    return {"errors": int(err), "strict": s / err, "broad": (s + d.pred_other_bg.sum()) / err,
            "separation": d[SEP].sum().sum() / err,
            "f1": 2 * d["tp@0.5"].sum() / (2 * d["tp@0.5"].sum() + err)}


def tem1_fiber(args):
    from axon_myelin_seg.evaluation.official import fiber_table, match
    from phase2_t22_h8 import load_gt
    row = args
    _, gt = load_gt(row)
    inst = tifffile.imread(REPO / "data/predictions/h11_tem1" / f"{row.case}.tif")
    pred = cv2.resize(inst.astype(np.float32), (row.w, row.h), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
    t = fiber_table(pred, gt)
    m = match(t, 0.5)
    matched = {kg for kg, _, _ in m}
    return [{"case": row.case, "subject": row.subject, "gt_k": int(k), "matched_cp": int(k) in matched}
            for k in t.gt_ids if t.gt_axon_area[int(k)] > 0]


def h11() -> None:
    rng = np.random.default_rng(0)
    S: dict = {"registered": "07-plan §7.9, commit ab7c273"}
    cp = pd.read_csv(REPO / "results/day4/scores_cellpose_ft.csv")
    b1 = pd.read_csv(REPO / "results/day4/scores_b1.csv")
    cp["group"] = cp.group.fillna(cp.name)
    pt = shares(cp)
    groups = cp.group.unique()
    idx = {g: cp.index[cp.group == g] for g in groups}
    bm = b1.set_index("name")
    draws, dF1 = [], []
    for _ in range(B):
        pick = rng.choice(groups, len(groups))
        rows = np.concatenate([idx[g] for g in pick])
        d = cp.loc[rows]
        draws.append(shares(d))
        e = bm.loc[d.name]
        f1b = 2 * e["tp@0.5"].sum() / (2 * e["tp@0.5"].sum() + e["fp@0.5"].sum() + e["fn@0.5"].sum())
        dF1.append(draws[-1]["f1"] - f1b)
    ci = {k: [float(pt[k]), *np.percentile([x[k] for x in draws], [2.5, 97.5]).tolist()]
          for k in ("strict", "broad", "separation", "f1")}
    b1pt = shares(b1)
    S["H11a"] = {"cp_ft": ci, "errors": pt["errors"], "b1_dev": {k: float(v) for k, v in b1pt.items()},
                 "f1_cp_minus_b1": [float(np.mean(pt["f1"] - b1pt["f1"])), *np.percentile(dF1, [2.5, 97.5]).tolist()]}
    inconclusive = pt["f1"] < 0.80
    holds = pt["broad"] >= 0.50 and ci["broad"][1] >= 0.40 and pt["separation"] <= 0.15
    S["H11a"]["verdict"] = "inconclusive (F1 < 0.80)" if inconclusive else ("holds" if holds else "fails")
    # H11b
    man = pd.read_csv(REPO / "data/nnunet/phase2_inputs/tem1/manifest.csv")
    rows = [SimpleNamespace(**r) for r in man[man.subject.isin(TEST_MICE)].to_dict("records")]
    with Pool(4) as p:
        F = pd.DataFrame([x for xs in p.map(tem1_fiber, rows) for x in xs])
    ref = pd.read_csv(REPO / "data/processed/phase2/h8_fibers_ref.csv")[["case", "gt_k", "g", "matched"]]
    m = F.merge(ref, on=["case", "gt_k"])
    assert len(m) == len(F) == len(ref), (len(m), len(F), len(ref))

    def gap(z, col):
        return z[z.g > 0.85][col].mean() - z[z.g <= 0.85][col].mean()

    def diff(z):
        return gap(z, "matched_cp") - gap(z, "matched")

    mice = sorted(m.subject.unique())
    by = {u: m[m.subject == u] for u in mice}
    bs = [diff(pd.concat([by[u] for u in rng.choice(mice, len(mice))])) for _ in range(B)]
    est = diff(m)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    S["H11b"] = {"recall_thin_cp": float(m[m.g > 0.85].matched_cp.mean()),
                 "recall_normal_cp": float(m[m.g <= 0.85].matched_cp.mean()),
                 "recall_thin_ref": float(m[m.g > 0.85].matched.mean()),
                 "recall_normal_ref": float(m[m.g <= 0.85].matched.mean()),
                 "gap_cp": float(gap(m, "matched_cp")), "gap_ref": float(gap(m, "matched")),
                 "gap_cp_minus_gap_ref": float(est), "ci95": [float(lo), float(hi)],
                 "verdict": "holds" if est <= -0.10 and hi < 0 else "fails"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h11.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", choices=["dev", "tem1"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--h11", action="store_true")
    a = ap.parse_args()
    if a.predict:
        predict(a.predict, a.limit)
    if a.h11:
        h11()


if __name__ == "__main__":
    main()
