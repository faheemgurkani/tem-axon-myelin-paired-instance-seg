"""Phase 2, T-17 (evaluator validation) and H6a/H6b/H6c (D-31): ASTIH's released models on the leakage-free test sets.

Inference: the released nnU-Net v2 model via `nnUNetPredictor` (tile step 0.5, mirroring on, fold "all",
checkpoint_final; the ASTIH-era ADS pipeline), on the M4 (MPS). Small: TEM1 8 images 2286×3762, TEM2 21 ROIs 600×600.
Predictions are cached in data/predictions/astih_<set>/ (git-ignored). Label values: myelin 1, axon 2.

Step 1, reproduction (ASTIH's evaluate_models.py, re-implemented without ADS/MONAI/stardist):
- Dice per image per class on the binary masks, averaged over images;
- instances for BOTH prediction and GT from ADS's centroid watershed (`get_watershed_segmentation`: seeds =
  int centroids of 8-connected axon components, landscape −EDT(axon), mask axon ∪ myelin), matched at IoU ≥ 0.5,
  per-image precision/recall/F1 averaged over images (ASTIH aggregates with mean).
- Gate (D-31): |Dice − published| ≤ 0.01 and |F1 − published| ≤ 0.03. Published TEM1: Dice 0.974 / 0.924 (axon /
  myelin), F1 0.791. TEM2: 0.971 / 0.929, F1 0.949. If the gate fails, H6 is NOT run for that set.

Step 2, H6 (only when step 1 passes): prediction = ADS watershed instances encoded as our pairs (axon 2k, myelin 2k+1);
GT = our `astih_to_pairs` (ASTIH convention: border fibers scored; TEM2 also with border_to_ignore=True).
Our evaluator + taxonomy (ring candidates at ds 4 for TEM1 / 2 for TEM2, sensitivity ds±1).
- strict share = unlabelled_real / (FP + FN); broad = (unlabelled_real + other_bg) / (FP + FN).
- H6a / H6c: strict ≤ 5% and broad ≤ 39%. H6b: the most frequent FP class is not unlabelled_real.
- Image-level bootstrap (one test animal on TEM1; a limitation, not animal-level inference).

    .venv/bin/python scripts/run/phase2_t17_h6.py --set TEM1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from skimage.segmentation import watershed

from axon_myelin_seg.data.external import astih_to_pairs
from axon_myelin_seg.evaluation.taxonomy import classify_errors, ring_candidates

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "data/external/models"
OUT = REPO / "results/phase2"
SETS = {
    "TEM1": {"model": "model_seg_mouse_axon-myelin_tem_light", "px_nm": 2.36, "ds": 4,
             "published": {"dice_axon": 0.974, "dice_myelin": 0.924, "f1": 0.791}},
    "TEM2": {"model": "model_seg_unmyelinated_stanford_tem_best", "px_nm": 4.93, "ds": 2,
             "published": {"dice_axon": 0.971, "dice_myelin": 0.929, "f1": 0.949}},
}


def test_items(name: str) -> list[dict]:
    """[{image, axon_gt, myelin_gt}] paths for the leakage-free test set."""
    items = []
    if name == "TEM1":
        root = REPO / "data/external/001436"
        for p in sorted((root / "sub-nyuMouse26/micr").glob("*_TEM.png")):
            lab = root / "derivatives/labels/sub-nyuMouse26/micr" / p.name[:-4]
            items.append({"image": p, "axon": Path(f"{lab}_seg-axon-manual.png"),
                          "myelin": Path(f"{lab}_seg-myelin-manual.png")})
    else:
        root = next((REPO / "data/external/tem2_testset").glob("data_axondeepseg_srf_testing*"))
        for p in sorted(root.glob("sub-*/micr/*_TEM.png")):
            lab = root / "derivatives/labels" / p.parent.parent.name / "micr" / p.name[:-4]
            items.append({"image": p, "axon": Path(f"{lab}_seg-axon-manual.png"),
                          "myelin": Path(f"{lab}_seg-myelin-manual.png"),
                          "axonmyelin": Path(f"{lab}_seg-axonmyelin-manual.png")})
    return items


def gray(p: Path) -> np.ndarray:
    im = iio.imread(p)
    return im[..., 0] if im.ndim == 3 else im


def predict(name: str, items: list[dict]) -> Path:
    out = REPO / "data/predictions" / f"astih_{name.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (out / it["image"].name).exists()]
    if not todo:
        return out
    for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(k, "/tmp/unused")
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pr = nnUNetPredictor(tile_step_size=0.5, use_mirroring=True, device=dev, verbose=False, allow_tqdm=False)
    pr.initialize_from_trained_model_folder(str(MODELS / SETS[name]["model"]), use_folds=("all",),
                                            checkpoint_name="checkpoint_final.pth")
    for it in todo:
        img = gray(it["image"]).astype(np.float32)[None, None]
        seg = pr.predict_single_npy_array(img, {"spacing": [999.0, 1.0, 1.0]})
        iio.imwrite(out / it["image"].name, np.asarray(seg[0]).astype(np.uint8))
        print("predicted", it["image"].name, flush=True)
    return out


def ads_watershed(ax: np.ndarray, my: np.ndarray) -> np.ndarray:
    """ADS get_watershed_segmentation with centroid seeds (as ASTIH evaluate_models.apply_watershed)."""
    props = measure.regionprops(measure.label(ax))
    seeds = np.zeros(ax.shape, np.int32)
    for i, p in enumerate(props, start=1):
        seeds[int(p.centroid[0]), int(p.centroid[1])] = i
    return watershed(-ndi.distance_transform_edt(ax), seeds, mask=(ax.astype(int) + my.astype(int)) > 0)


def ads_watershed_astih_as_run(ax: np.ndarray, my: np.ndarray) -> np.ndarray:
    """ADS watershed exactly as ASTIH's evaluate_models.py runs it: masks come from cv2.imread(...)[None], i.e. shape
    (1, H, W). regionprops' centroid is then (z=0, y, x) and get_watershed_segmentation's
    `im_centroid[c[0], c[1]] = i + 1` writes seed i over the WHOLE ROW y, not a point. Found 16 Nov (D-38)."""
    ax3, my3 = ax[None].astype(np.float32), my[None].astype(np.float32)
    props = measure.regionprops(measure.label(ax3))
    seeds = np.zeros(ax3.shape, np.uint16)
    for i, p in enumerate(props):
        seeds[int(p.centroid[0]), int(p.centroid[1])] = i + 1
    return watershed(-ndi.distance_transform_edt(ax3), seeds, mask=ax3 + my3)[0]


def det_stats(gt_inst: np.ndarray, pr_inst: np.ndarray, thr: float = 0.5) -> dict:
    """Matching at IoU ≥ thr (unique for thr ≥ 0.5, so equal to stardist's optimal matching)."""
    g, p = gt_inst.ravel(), pr_inst.ravel()
    ng, npr = int(g.max()), int(p.max())
    m = (g > 0) & (p > 0)
    inter = np.bincount(g[m].astype(np.int64) * (npr + 1) + p[m], minlength=(ng + 1) * (npr + 1))
    inter = inter.reshape(ng + 1, npr + 1)
    ag, ap = np.bincount(g, minlength=ng + 1), np.bincount(p, minlength=npr + 1)
    gi, pi = np.nonzero(inter)
    gi, pi = gi[(gi > 0) & (pi > 0)], pi[(gi > 0) & (pi > 0)]
    iou = inter[gi, pi] / (ag[gi] + ap[pi] - inter[gi, pi])
    n_g = int((ag[1:] > 0).sum())
    n_p = int((ap[1:] > 0).sum())
    tp = int((iou >= thr).sum())
    fp, fn = n_p - tp, n_g - tp
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(n_p, 1), "recall": tp / max(n_g, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def encode_pairs(inst: np.ndarray, ax: np.ndarray, my: np.ndarray) -> np.ndarray:
    out = np.zeros(inst.shape, np.uint16)
    out[(inst > 0) & my] = (2 * inst[(inst > 0) & my] + 1).astype(np.uint16)
    out[(inst > 0) & ax] = (2 * inst[(inst > 0) & ax]).astype(np.uint16)
    return out


def gt_masks(name: str, it: dict) -> tuple[np.ndarray, np.ndarray]:
    ax, my = gray(it["axon"]) > 0, gray(it["myelin"]) > 0
    return ax, my & ~ax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), required=True)
    a = ap.parse_args()
    cfg = SETS[a.set]
    OUT.mkdir(parents=True, exist_ok=True)
    items = test_items(a.set)
    pdir = predict(a.set, items)
    rep, h6 = [], []
    for it in items:
        sem = iio.imread(pdir / it["image"].name)
        p_ax, p_my = sem == 2, sem == 1
        g_ax, g_my = gt_masks(a.set, it)
        rep.append({"image": it["image"].name, "dice_axon": dice(p_ax, g_ax), "dice_myelin": dice(p_my, g_my),
                    **det_stats(ads_watershed(g_ax, g_my), ads_watershed(p_ax, p_my)),
                    **{f"{k}_as_run": v for k, v in det_stats(ads_watershed_astih_as_run(g_ax, g_my),
                                                               ads_watershed_astih_as_run(p_ax, p_my)).items()}})
        # H6: our evaluator on our pair GT
        pred = encode_pairs(ads_watershed(p_ax, p_my), p_ax, p_my)
        img = gray(it["image"])
        for border_ignore in ((False, True) if a.set == "TEM2" else (False,)):
            gt, _ = astih_to_pairs(gray(it["axon"]), gray(it["myelin"]), border_to_ignore=border_ignore)
            for ds in (cfg["ds"] - 1, cfg["ds"], cfg["ds"] + 1):
                cand = ring_candidates(img, gt, ds=ds, px_nm=cfg["px_nm"])
                r = classify_errors(pred, gt, cand)
                t = r["table"]
                m = r["matches"]
                c = r["counts"]
                # errors on border-touching GT fibers / preds
                edge = np.zeros_like(gt, bool)
                edge[[0, -1], :] = edge[:, [0, -1]] = True
                gt_edge = set((gt[edge] // 2).tolist()) - {0}
                pr_edge = set((pred[edge] // 2).tolist()) - {0}
                n_fn_edge = sum(1 for k in r["gt"] if k in gt_edge)
                n_fp_edge = sum(1 for k in r["pred"] if k in pr_edge)
                h6.append({"image": it["image"].name, "border_ignore": border_ignore, "ds": ds,
                           "tp": len(m), "fp": len(t.pred_ids) - len(m), "fn": len(t.gt_ids) - len(m),
                           "errors_on_border": n_fn_edge + n_fp_edge, **c})
        print("scored", it["image"].name, flush=True)
    R = pd.DataFrame(rep)
    H = pd.DataFrame(h6)
    R.to_csv(OUT / f"t17_{a.set}_images.csv", index=False)
    H.to_csv(OUT / f"h6_{a.set}_images.csv", index=False)
    pub = cfg["published"]
    repro = {"dice_axon": float(R.dice_axon.mean()), "dice_myelin": float(R.dice_myelin.mean()),
             "f1_mean": float(R.f1.mean()), "precision_mean": float(R.precision.mean()),
             "recall_mean": float(R.recall.mean()),
             "f1_micro": float(2 * R.tp.sum() / (2 * R.tp.sum() + R.fp.sum() + R.fn.sum())),
             "f1_mean_astih_as_run": float(R.f1_as_run.mean()),
             "precision_mean_astih_as_run": float(R.precision_as_run.mean()),
             "recall_mean_astih_as_run": float(R.recall_as_run.mean())}
    dice_ok = (abs(repro["dice_axon"] - pub["dice_axon"]) <= 0.01
               and abs(repro["dice_myelin"] - pub["dice_myelin"]) <= 0.01)
    f1_ok = abs(repro["f1_mean"] - pub["f1"]) <= 0.03
    f1_ok_as_run = abs(repro["f1_mean_astih_as_run"] - pub["f1"]) <= 0.03
    # D-38: the gate passes if Dice reproduces and F1 reproduces either directly or through ASTIH's own code path
    passed = dice_ok and (f1_ok or f1_ok_as_run)
    S = {"set": a.set, "n_images": len(R), "published": pub, "reproduced": repro, "dice_ok": bool(dice_ok),
         "f1_ok_correct_watershed": bool(f1_ok), "f1_ok_astih_as_run": bool(f1_ok_as_run),
         "t17_gate_passed": bool(passed)}
    if passed:
        rng = np.random.default_rng(0)
        res = {}
        for (bi, ds), d in H.groupby(["border_ignore", "ds"]):
            def shares(x: pd.DataFrame) -> dict:
                e = x.fp.sum() + x.fn.sum()
                return {"strict": x.pred_unlabelled_real.sum() / e if e else np.nan,
                        "broad": (x.pred_unlabelled_real.sum() + x.pred_other_bg.sum()) / e if e else np.nan}
            est = shares(d)
            bs = [shares(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
            fp_cls = {k: int(d[f"pred_{k}"].sum()) for k in
                      ("gap", "unlabelled_real", "merge", "split_piece", "low_iou", "other_bg", "other")}
            gt_cls = {k: int(d[f"gt_{k}"].sum()) for k in ("missed", "merged", "split", "low_iou")}
            e = int(d.fp.sum() + d.fn.sum())
            res[f"border_ignore={bi},ds={ds}"] = {
                "tp": int(d.tp.sum()), "fp": int(d.fp.sum()), "fn": int(d.fn.sum()), "errors": e,
                "f1": float(2 * d.tp.sum() / (2 * d.tp.sum() + e)),
                "strict_share": float(est["strict"]), "broad_share": float(est["broad"]),
                "strict_ci95": [float(np.nanpercentile([b["strict"] for b in bs], q)) for q in (2.5, 97.5)],
                "broad_ci95": [float(np.nanpercentile([b["broad"] for b in bs], q)) for q in (2.5, 97.5)],
                "fp_classes": fp_cls, "gt_classes": gt_cls, "pred_on_label1": int(d.pred_on_label1.sum()),
                "share_errors_on_border": float(d.errors_on_border.sum() / e) if e else np.nan,
                "H6a_or_c": bool(est["strict"] <= 0.05 and est["broad"] <= 0.39),
                "H6b_top_fp_class": max(fp_cls, key=fp_cls.get),
                "H6b": max(fp_cls, key=fp_cls.get) != "unlabelled_real"}
        S["h6"] = res
    else:
        S["h6"] = "NOT RUN: T-17 reproduction gate failed (D-31)"
    (OUT / f"t17_h6_{a.set}.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
