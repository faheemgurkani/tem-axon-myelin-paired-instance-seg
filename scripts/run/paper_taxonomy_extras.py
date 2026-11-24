"""T-28 G3 + G5 (reviewer audit, 25 Nov): protocol realism and taxonomy sensitivity, on B1's 3-fold out-of-fold
predictions for all 266 images (D-11 pairing, as in Phase 3a). CPU only.

G5 sensitivity: error-family shares (strict selection, broad selection, separation) under
  - the ring-candidate detector's downsample factor ds in {3, 4, 5} (default 4) and minimum hole diameter
    min_d_nm in {200, 300, 400} (default 300);
  - the matching threshold IoU > 0.5 (default) vs > 0.75 (default detector).
  The broad share does not use the detector; only the strict share can move with it.
G3 realism of the simulated T-24 stratum (g > 0.85 or g < 0.65 or eccentricity > 0.9): share in the stratum of
  (a) labelled GT fibers, (b) B1 predictions matched to them, (c) unmatched predictions classed unlabelled_real,
  (d) unmatched predictions classed other_bg. If (c)/(d) are enriched relative to (a), the simulated protocol mirrors
  what the real protocol leaves out.

Outputs (committed): results/paper/taxonomy_sensitivity.json, results/paper/protocol_realism.json.

    .venv/bin/python scripts/run/paper_taxonomy_extras.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from skimage.measure import regionprops

from axon_myelin_seg.data.io import list_samples, load_sample
from axon_myelin_seg.evaluation.official import paired_fiber_map
from axon_myelin_seg.evaluation.taxonomy import classify_errors, ring_candidates
from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances

REPO = Path(__file__).resolve().parents[2]
SEM = REPO / "data/predictions/b1_cv3_sem"
OUT = REPO / "results/paper"
SETTINGS = [(ds, d) for ds in (3, 4, 5) for d in (200, 300, 400)]
SEP = ["pred_gap", "pred_merge", "pred_split_piece", "gt_merged", "gt_split"]


def fiber_feats(lab: np.ndarray) -> dict[int, tuple[float, float]]:
    """fiber id -> (equivalent-circle g, eccentricity of the fiber) for a paired label map (axon 2k, ring 2k+1)."""
    fid = paired_fiber_map(lab)
    ax = np.where((lab >= 2) & (lab % 2 == 0), fid, 0)
    a_ax = np.bincount(ax.ravel(), minlength=fid.max() + 1)
    out = {}
    for p in regionprops(fid):
        if p.area == 0:
            continue
        out[p.label] = (float(np.sqrt(a_ax[p.label] / p.area)), float(p.eccentricity))
    return out


def in_stratum(g: float, e: float) -> bool:
    return g > 0.85 or g < 0.65 or e > 0.9


def one(name: str) -> dict:
    img, gt = load_sample(name)
    sem = iio.imread(SEM / f"{name}.png")
    pred = semantic_to_paired_instances(sem == 1, sem == 2)
    r = {"name": name, "sens": {}}
    for ds, d in SETTINGS:
        cand = ring_candidates(img, gt, ds=ds, min_d_nm=d)
        tax = classify_errors(pred, gt, cand)
        r["sens"][f"ds{ds}_d{d}"] = tax["counts"] | {"tp": len(tax["matches"])}
        if (ds, d) == (4, 300):
            base = tax
            cand0 = cand
    t75 = classify_errors(pred, gt, cand0, iou_threshold=0.75)
    r["sens"]["iou0.75"] = t75["counts"] | {"tp": len(t75["matches"])}
    # realism
    fp, fg = fiber_feats(pred), fiber_feats(np.where(gt == 1, 0, gt))
    matched_pred = {b for _, b, _ in base["matches"]}
    rows = {"gt_labelled": [fg[k] for k in fg],
            "pred_matched": [fp[k] for k in matched_pred if k in fp],
            "pred_unlabelled_real": [fp[k] for k, c in base["pred"].items() if c == "unlabelled_real" and k in fp],
            "pred_other_bg": [fp[k] for k, c in base["pred"].items() if c == "other_bg" and k in fp]}
    r["real"] = rows
    return r


def shares(c: dict) -> dict:
    err = sum(v for k, v in c.items() if k.startswith(("pred_", "gt_")) and k != "pred_on_label1")
    return {"errors": err, "strict": c["pred_unlabelled_real"] / err,
            "broad": (c["pred_unlabelled_real"] + c["pred_other_bg"]) / err, "separation": sum(c[k] for k in SEP) / err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    names = list_samples()
    names = names[: a.limit] if a.limit else names
    with Pool(6) as p:
        res = p.map(one, names, chunksize=2)
    OUT.mkdir(parents=True, exist_ok=True)
    sens = {}
    for key in res[0]["sens"]:
        tot: dict = {}
        for r in res:
            for k, v in r["sens"][key].items():
                tot[k] = tot.get(k, 0) + v
        sens[key] = shares(tot) | {"counts": tot}
    real = {}
    for cls in res[0]["real"]:
        f = [x for r in res for x in r["real"][cls]]
        g = np.array([x[0] for x in f])
        e = np.array([x[1] for x in f])
        real[cls] = {"n": len(f), "share_stratum": float(np.mean([in_stratum(*x) for x in f])) if f else None,
                     "share_g_gt_0.85": float(np.mean(g > 0.85)) if f else None,
                     "share_g_lt_0.65": float(np.mean(g < 0.65)) if f else None,
                     "share_ecc_gt_0.9": float(np.mean(e > 0.9)) if f else None,
                     "median_g": float(np.median(g)) if f else None}
    tag = "" if not a.limit else f"_limit{a.limit}"
    (OUT / f"taxonomy_sensitivity{tag}.json").write_text(json.dumps({"n_images": len(res), "settings": sens}, indent=1))
    (OUT / f"protocol_realism{tag}.json").write_text(json.dumps({"n_images": len(res), "classes": real,
                                                                 "stratum": "g>0.85 or g<0.65 or ecc>0.9"}, indent=1))
    for k, v in sens.items():
        print(f"{k:10s} err={v['errors']:5d} strict={v['strict']:.3f} broad={v['broad']:.3f} sep={v['separation']:.3f}")
    for k, v in real.items():
        print(k, v)


if __name__ == "__main__":
    main()
