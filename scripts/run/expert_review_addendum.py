"""Addendum to the blind expert review (D-33 note 10; proposed 15 Nov, built 16 Nov): the overlap-disagreement fibers.

Reproduces the repeat-annotation comparison of `scripts/run/day2_ceilings.py` for every used pair (NCC ≥ 0.8), using
its stored warp matrix M: B's labels warped into A's frame are the "prediction", label 1 in either tile and the area
outside the shared view are ignored. Unmatched fibers (IoU ≤ 0.5) are the disagreements:
  - FN: labelled in A only (outline = A's fiber);
  - FP: labelled in B only (outline = B's warped fiber).
Expected count: 18 (FP 7 + FN 11; 05-results §2). Each is rated with the same five categories as the main packet;
normal vs abnormal splits C3's r ≈ 0.93 into random misses vs inconsistent protocol judgements. Descriptive, no pass rule.

Output (git-ignored, contains challenge crops; never upload): data/processed/expert_review_addendum/
    .venv/bin/python scripts/run/expert_review_addendum.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from axon_myelin_seg.data.io import load_sample
from axon_myelin_seg.evaluation.official import fiber_table, match, paired_fiber_map

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/processed/expert_review_addendum"
spec = importlib.util.spec_from_file_location("pk", REPO / "scripts/run/expert_review_packet.py")
pk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pk)


def warp(arr, M, shape):
    return cv2.warpAffine(arr, M, (shape[1], shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)


def main() -> None:
    pairs = pd.read_csv(REPO / "results/day2/ceilings_pairs.csv")
    pairs = pairs[pairs.ncc >= 0.8]
    items = []
    for r in pairs.itertuples():
        ia, ma = load_sample(r.a)
        _, mb = load_sample(r.b)
        H, W = ia.shape
        M = np.array(json.loads(r.M), np.float32)
        valid = warp(np.ones_like(mb, np.uint8), M, (H, W)).astype(bool)
        valid = ndi.binary_erosion(valid, iterations=20)
        mb_w = warp(mb, M, (H, W))
        gt = ma.copy()
        gt[~valid] = 1
        gt[mb_w == 1] = 1
        t = fiber_table(mb_w, gt)
        m = match(t, 0.5)
        mg, mp = {kg for kg, _, _ in m}, {kp for _, kp, _ in m}
        fg, fp = paired_fiber_map(gt), paired_fiber_map(mb_w)
        for kind, ids, fmap in (("FN_A_only", [k for k in t.gt_ids if k not in mg], fg),
                                ("FP_B_only", [k for k in t.pred_ids if k not in mp], fp)):
            for k in ids:
                f = fmap == k
                ys, xs = np.nonzero(f)
                items.append({"pair": f"{r.a}|{r.b}", "kind": kind, "fiber": int(k), "cy": int(ys.mean()),
                              "cx": int(xs.mean()), "png": pk.crop(ia, f, int(ys.mean()), int(xs.mean()), 2.152)})
    rng = np.random.default_rng(0)
    order = rng.permutation(len(items))
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    keys = []
    for j, i in enumerate(order, start=1):
        it = dict(items[i])
        cv2.imwrite(str(OUT / "img" / f"{j:03d}.png"), it.pop("png"))
        keys.append({"item": f"{j:03d}", **it})
    pd.DataFrame(keys).to_csv(OUT / "key.csv", index=False)
    html = pk.HTML.replace("__N__", str(len(items))).replace("__OPTS__", json.dumps(pk.OPTIONS))
    html = html.replace("Myelinated-fiber review", "Myelinated-fiber review: addendum").replace("'fr'", "'fr_add'")
    (OUT / "index.html").write_text(html)
    print(len(items), "items;", pd.DataFrame(keys).kind.value_counts().to_dict())


if __name__ == "__main__":
    main()
