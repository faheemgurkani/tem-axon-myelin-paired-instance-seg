"""Phase 1, T-23 (H7): does the labelling protocol bias the endpoint? Tooling + a PROVISIONAL run.

Steps (all CPU, local):
1. Outer-ring tracer: from an inner hole, grow into the dark myelin band (intensity below the image's
   myelin/background midpoint, opened with a 5-px disk to drop thin membranes; revised 16 Nov after the first
   version failed validation with bias −0.05), capped at the distance that gives g ≥ 0.5 and excluding other labelled fibers.
   g = sqrt(A_hole / A_hole∪band).
2. Tracer validation on LABELLED fibers (GT g known): run the same tracer from each GT axon, other fibers excluded;
   report bias and limits of agreement of traced g vs GT g. If |bias| > 0.02 or LoA half-width > 0.08 the tracer is
   not good enough for H7a/H7c, and only H7b (counts) is reported.
3. Detector recall on labelled fibers (are their axons found as dark-ringed holes?), used to correct the skip rate.
4. Provisional H7 on our data: the unlabelled ring candidates of the completeness audit, with the real-fiber share
   0.83 from Claude's single non-expert rating (note 07 §4). CONFIRMATORY H7 uses the expert-confirmed set (D-33);
   rules (07-plan §7.5): H7a |Δ mean g| ≥ 0.03; H7b count bias ≥ 5% (CI lower bound); H7c |Δ share g > 0.85| ≥ 2 pp.
Bootstrap over overlap groups (D-22). Outputs: results/phase1/h7_*.json|csv (per-fiber values stay in data/processed).

    .venv/bin/python scripts/run/phase1_h7_protocol_bias.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from axon_myelin_seg.data.io import list_samples, load_sample
from axon_myelin_seg.evaluation.official import paired_fiber_map
from axon_myelin_seg.evaluation.taxonomy import ring_candidates

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/phase1"
PROC = REPO / "data/processed/phase1"
REAL_SHARE_PROVISIONAL = 0.83  # Claude's single rating of 48 audit candidates (note 07 §4); replaced by D-33 ratings
B8 = np.ones((3, 3), bool)
OPEN_SE = np.hypot(*np.mgrid[-5:6, -5:6]) <= 5  # disk, radius 5 px (≈ 11 nm)


def trace_g(img_s: np.ndarray, thr: float, hole: np.ndarray, forbid: np.ndarray) -> float:
    """g of one fiber from its hole (bool, crop) by growing into the dark band; forbid = other fibers (crop)."""
    a = hole.sum()
    if a < 20:
        return np.nan
    r = np.sqrt(a / np.pi)
    d = ndi.distance_transform_edt(~hole)
    # compact myelin is a thick dark band (~44 px); an opening removes thin membranes (2–4 px) before growing
    dark = ndi.binary_opening(img_s < thr, structure=OPEN_SE) & ~forbid & (d <= r)  # g ≥ 0.5
    lab, _ = ndi.label(dark | hole, structure=B8)
    keep = np.unique(lab[hole])
    band = np.isin(lab, keep[keep > 0]) & ~hole
    outer = ndi.binary_fill_holes(hole | band)
    return float(np.sqrt(a / outer.sum())) if outer.sum() > a else np.nan


def per_image(name: str):
    img, gt = load_sample(name)
    img_s = ndi.gaussian_filter(img.astype(np.float32), 1.5)
    my, bg = (gt >= 3) & (gt % 2 == 1), gt == 0
    thr = (np.median(img_s[my]) + np.median(img_s[bg])) / 2 if my.any() and bg.any() else np.nan
    fib = paired_fiber_map(gt)
    ax = (gt >= 2) & (gt % 2 == 0)
    rows_lab, rows_cand = [], []
    # 2. validation on labelled fibers + 3. detector recall on them
    cand_any = ring_candidates(img, gt, skip_labelled_filter=True)  # detector recall on labelled fibers
    for k, sl in enumerate(ndi.find_objects(fib), start=1):
        if sl is None:
            continue
        axk = ax[sl] & (fib[sl] == k)
        if not axk.any() or not ((fib[sl] == k) & ~ax[sl]).any():
            continue
        g_gt = np.sqrt(axk.sum() / (fib[sl] == k).sum())
        pad = int(2 * np.sqrt(axk.sum() / np.pi)) + 10
        ys = slice(max(sl[0].start - pad, 0), sl[0].stop + pad)
        xs = slice(max(sl[1].start - pad, 0), sl[1].stop + pad)
        hole = ndi.binary_fill_holes(ax[ys, xs] & (fib[ys, xs] == k))
        forbid = (fib[ys, xs] > 0) & (fib[ys, xs] != k) | (gt[ys, xs] == 1)
        rows_lab.append({"image": name, "k": k, "g_gt": g_gt, "g_trace": trace_g(img_s[ys, xs], thr, hole, forbid),
                         "detected": bool((cand_any[ys, xs][hole] > 0).mean() >= 0.5)})
    # 4. unlabelled candidates (same detector as the audit)
    cand = ring_candidates(img, gt)
    for c, sl in enumerate(ndi.find_objects(cand), start=1):
        if sl is None:
            continue
        hole = cand[sl] == c
        r = np.sqrt(hole.sum() / np.pi)
        pad = int(r) + 10
        ys = slice(max(sl[0].start - pad, 0), sl[0].stop + pad)
        xs = slice(max(sl[1].start - pad, 0), sl[1].stop + pad)
        hole_c = cand[ys, xs] == c
        # refine the blocky (4×-downsampled) hole to the bright interior
        hole_c = ndi.binary_fill_holes(ndi.binary_opening(hole_c & (img_s[ys, xs] >= thr), B8)) | \
            (ndi.binary_erosion(hole_c, iterations=3))
        forbid = (fib[ys, xs] > 0) | (gt[ys, xs] == 1)
        rows_cand.append({"image": name, "c": c, "g_trace": trace_g(img_s[ys, xs], thr, hole_c, forbid)})
    return rows_lab, rows_cand


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    names = list_samples()
    groups = json.loads((REPO / "configs/splits/overlap_groups.json").read_text())
    gmap = {}
    for gi, members in enumerate(groups):
        for mname in members:
            gmap[mname] = gmap[mname.removesuffix(".tif")] = f"g{gi}"
    with Pool(6) as p:
        res = p.map(per_image, names)
    lab = pd.DataFrame([r for a, _ in res for r in a])
    cand = pd.DataFrame([r for _, b in res for r in b])
    for d in (lab, cand):
        d["group"] = d.image.map(lambda n: gmap.get(n, n))
    lab.to_csv(PROC / "h7_labelled_fibers.csv", index=False)
    cand.to_csv(PROC / "h7_candidates.csv", index=False)
    v = lab.dropna(subset=["g_trace"])
    diff = v.g_trace - v.g_gt
    val = {"n": len(v), "traced_share": float(lab.g_trace.notna().mean()), "bias": float(diff.mean()),
           "loa_half": float(1.96 * diff.std()), "mae": float(diff.abs().mean())}
    tracer_ok = abs(val["bias"]) <= 0.02 and val["loa_half"] <= 0.08
    det_recall = float(lab.detected.mean())
    # provisional H7 with group bootstrap
    rng = np.random.default_rng(0)
    grp = sorted(set(lab.group))

    def stats(lb: pd.DataFrame, cd: pd.DataFrame) -> dict:
        n_l = len(lb)
        n_real = REAL_SHARE_PROVISIONAL * len(cd) / max(det_recall, 1e-6)  # detector-recall corrected
        cg = cd.g_trace.dropna()
        g_l = lb.g_gt.mean()
        g_u = cg.mean() if len(cg) else g_l
        w = n_real / (n_l + n_real)
        tail_l = (lb.g_gt > 0.85).mean()
        tail_u = (cg > 0.85).mean() if len(cg) else tail_l
        return {"count_bias": w, "d_mean_g": w * (g_u - g_l), "d_tail_085": w * (tail_u - tail_l),
                "g_labelled": g_l, "g_skipped_traced": g_u}

    est = stats(lab, cand)
    boots = []
    for _ in range(1000):
        pick = rng.choice(grp, len(grp))
        lb = pd.concat([lab[lab.group == x] for x in pick])
        cd = pd.concat([cand[cand.group == x] for x in pick])
        boots.append(stats(lb, cd))
    B = pd.DataFrame(boots)
    ci = {k: [float(B[k].quantile(0.025)), float(B[k].quantile(0.975))] for k in B.columns}
    out = {"status": "PROVISIONAL (real share 0.83 from one non-expert rater; confirmatory run needs D-33)",
           "tracer_validation": val, "tracer_ok_for_g": tracer_ok, "detector_recall_on_labelled": det_recall,
           "n_candidates": len(cand), "n_labelled": len(lab), "estimate": est, "ci95": ci,
           "rules": {"H7a": "|d_mean_g| >= 0.03, CI excludes 0", "H7b": "count_bias CI lower >= 0.05",
                     "H7c": "|d_tail_085| >= 0.02, CI excludes 0"}}
    if not tracer_ok:
        out["note"] = "Tracer failed its validation gate: H7a/H7c not reportable from traced g; H7b only."
    (OUT / "h7_provisional.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
