"""Visual check of unmatched predictions by taxonomy class (T-12/T-13 sanity check).

Draws N random examples of one class from one baseline's dev-val predictions:
- left: the image crop;
- right: the same crop with the prediction outline in white and GT fibers tinted.

Usage: python scripts/run/day4_fp_gallery.py --pred-dir data/predictions/<name> --cls other_bg --n 16 --out <png>
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from axon_myelin_seg.data.io import load_sample, read_mask
from axon_myelin_seg.evaluation.runner import score_image

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--pred-dir", required=True)
ap.add_argument("--cls", required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)
names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
random.shuffle(names)
tiles = []
for n in names:
    if len(tiles) >= a.n:
        break
    img, gt = load_sample(n)
    pred = read_mask(Path(a.pred_dir) / f"{n}.tif")
    r = score_image(pred, gt, image=img, return_classes=True)
    ks = [k for k, c in r["_classes"]["pred"].items() if c == a.cls]
    random.shuffle(ks)
    for k in ks[:2]:
        f = (pred == 2 * k) | (pred == 2 * k + 1)
        ys, xs = np.nonzero(f)
        cy, cx = int(ys.mean()), int(xs.mean())
        h = 350
        y0, x0 = max(cy - h, 0), max(cx - h, 0)
        sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
        crop = cv2.cvtColor(img[sl], cv2.COLOR_GRAY2BGR)
        ov = crop.copy()
        g = gt[sl]
        ov[(g >= 2)] = (0.6 * ov[(g >= 2)] + 0.4 * np.array([0, 160, 0])).astype(np.uint8)
        ov[g == 1] = (0.6 * ov[g == 1] + 0.4 * np.array([160, 0, 160])).astype(np.uint8)
        cnts, _ = cv2.findContours(f[sl].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, (255, 255, 255), 3)
        t = cv2.resize(np.hstack([crop, ov]), (560, 280))
        cv2.putText(t, f"{n[-4:]} k={k}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(t)
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
cv2.imwrite(a.out, np.vstack(rows))
print("wrote", a.out, len(tiles), "examples")
