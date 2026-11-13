"""T-11 (D-13): AxonPath zero-shot on the dev-val images, converted to the challenge encoding.

AxonPath (paucabar/axonpath @ 6d633bf, bundled `em` weights) was trained on EM at **8 nm/px**. Our pixels are
2.152 nm, so images are downscaled by 2.152 / 8 = 0.269 (4096×3008 → 1102×809) before inference. The labels are
upscaled back to full resolution with nearest neighbour.

Mapping AxonPath outputs to our encoding, per AxonPath fibre k:
- axon (2k) = AxonPath *inner cylinder* k (axon + inner tongue). Our GT counts the inner tongue as axon
  (docs/notes/02-challenge-and-data.md §3), so this is the like-for-like choice. AxonPath's "axon" excludes the tongue.
- myelin (2k+1) = fibre k minus inner cylinder k.
- If a fibre has no inner cylinder, AxonPath's own *axon* instance for that fibre is used instead (fallback).
  This was added after the first run (v1) dropped 834 of 3,277 fibres (25%) for lacking an inner cylinder;
  see docs/notes/05-results.md §3. A fibre with neither is dropped, since it has no axon and is therefore not
  a fiber in our encoding. Both counts are logged.

AxonPath pairs axons to fibres with its own rule (map_axon_labels_to_fibres / segment_inner_cylinders), not with
our watershed. That makes it the "explicit-fibre-instance" baseline.

Runs locally on MPS (Apple GPU). Output: data/predictions/axonpath/<name>.tif (git-ignored) and
results/day3/axonpath_runlog.csv.
Usage: python baselines/axonpath/run.py [--split val|all]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from axon_myelin_seg.data.io import list_samples, load_sample, write_mask

REPO = Path(__file__).resolve().parents[2]
SCALE = 2.152 / 8.0


def to_challenge(fibre: np.ndarray, inner: np.ndarray, axon: np.ndarray) -> tuple[np.ndarray, int, int]:
    out = np.zeros(fibre.shape, np.uint16)
    ids = np.unique(fibre)
    ids = ids[ids > 0]
    dropped, fallback, k = 0, 0, 0
    for i in ids:
        f = fibre == i
        ic = (inner == i) & f
        if not ic.any() or ic.sum() == f.sum():
            ic = (axon == i) & f
            if not ic.any() or ic.sum() == f.sum():
                dropped += 1
                continue
            fallback += 1
        k += 1
        out[f] = 2 * k + 1
        out[ic] = 2 * k
    return out, dropped, fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "all"])
    ap.add_argument("--min-diameter", type=float, default=20.0)
    a = ap.parse_args()
    import axonpath
    from axonpath.inference.inference import load_model, run_inference
    from axonpath.utils.image_processing import normalize

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(str(Path(axonpath.__file__).parent / "weights" / "em.pth"), device=device)
    names = list_samples() if a.split == "all" else json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    out_dir = REPO / "data" / "predictions" / "axonpath"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for n in names:
        t0 = time.time()
        img, _ = load_sample(n)
        H, W = img.shape
        small = cv2.resize(img, (round(W * SCALE), round(H * SCALE)), interpolation=cv2.INTER_AREA)
        x = normalize(small.astype(np.float32))
        fibre, axon, inner, _sem = run_inference(x, model, device=device, min_diameter=a.min_diameter)
        fibre = cv2.resize(fibre.astype(np.int32).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        inner = cv2.resize(inner.astype(np.int32).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        axon = cv2.resize(axon.astype(np.int32).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        pred, dropped, fallback = to_challenge(fibre, inner, axon)
        write_mask(out_dir / f"{n}.tif", pred)
        log.append({"name": n, "n_fibres_axonpath": int(len(np.unique(fibre)) - 1), "n_fibers_written": int(pred.max() // 2),
                    "dropped_no_axon": dropped, "axon_fallback": fallback, "seconds": time.time() - t0})
        print(log[-1], flush=True)
    out = REPO / "results" / "day3"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out / "axonpath_runlog.csv", index=False)


if __name__ == "__main__":
    main()
