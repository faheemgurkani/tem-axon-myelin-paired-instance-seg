"""T-10 (reduced, D-12 revised 13 Nov): Cellpose-SAM *zero-shot* fiber instances on dev-val, locally on MPS.

The fine-tuned B3 of the plan does not fit the $10 GPU budget once B1 is paid for (docs/notes/16-day3-baselines.md
§T-10). This zero-shot run costs nothing and asks one question: does an instance-first generalist find the fibers?

- Images are downscaled ×1/6 (2.152 → 12.9 nm/px), so the median fiber is ~59 px across, inside Cellpose-SAM's
  7.5–120 px training range (D-12).
- `cpsam_v2`, diameter=None (the SAM backbone is size-invariant), default flow and cellprob thresholds.
- Masks are upscaled back with nearest neighbour.
- Cellpose segments whatever looks like "a cell". In TEM that could be whole fibers (ring + axon) or only the
  light axon interiors. Which one it does is reported (doc 16).
- The axon/myelin split inside each instance needs B1's semantics. It is done in scripts/run/day4_cellpose_split.py
  after B1 finishes.

Output: data/predictions/cellpose_zs_instances/<name>.tif (int32 instance map, git-ignored).
Usage: python baselines/cellpose_sam/zeroshot.py [--limit N]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

from axon_myelin_seg.data.io import load_sample

REPO = Path(__file__).resolve().parents[2]
DS = 6

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
from cellpose import models  # noqa: E402

model = models.CellposeModel(gpu=True, pretrained_model="cpsam_v2")
names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
if a.limit:
    names = names[: a.limit]
out = REPO / "data" / "predictions" / "cellpose_zs_instances"
out.mkdir(parents=True, exist_ok=True)
log = []
for n in names:
    t0 = time.time()
    img, _ = load_sample(n)
    H, W = img.shape
    small = cv2.resize(img, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
    masks, _flows, _ = model.eval(small, diameter=None)[:3]
    full = cv2.resize(masks.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    tifffile.imwrite(out / f"{n}.tif", full, compression="zlib")
    areas = np.bincount(masks.ravel())[1:]
    areas = areas[areas > 0]
    log.append({"name": n, "n_instances": int(areas.size), "median_diam_px_small": float(2 * np.sqrt(np.median(areas) / np.pi)) if areas.size else 0.0,
                "seconds": time.time() - t0})
    print(log[-1], flush=True)
(REPO / "results" / "day3").mkdir(parents=True, exist_ok=True)
pd.DataFrame(log).to_csv(REPO / "results" / "day3" / f"cellpose_zs_runlog{'_test' if a.limit else ''}.csv", index=False)
