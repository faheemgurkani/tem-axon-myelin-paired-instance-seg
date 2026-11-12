"""T-09/T-10 applied to B1's output: semantics → paired instances, plus the Cellpose(+B1) variant.

Input:
- B1's nnU-Net validation segmentations, downloaded from the Modal Volume to
  data/nnunet/results_modal/<...>/fold_0/validation/<name>.png (labels 0 bg, 1 axon, 2 myelin).

Outputs (all git-ignored):
- data/predictions/b1_sem/<name>.png: B1 semantics (copy, for O-sep)
- data/predictions/b1/<name>.tif: B1 instances via the shared watershed (D-11)
- data/predictions/cellpose_b1/<name>.tif: Cellpose-SAM zero-shot instances, split into axon/myelin by B1 (D-12).
  For each instance:
  - axon = the largest 8-connected component of B1-axon pixels inside it, holes filled, kept inside the instance;
  - myelin = B1-myelin pixels inside it, minus the axon;
  - an instance with no axon (≥ 50 px) or no myelin is dropped (background "cells" fall out here).

Usage: python scripts/run/day4_postprocess.py --val-dir <validation dir>
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile
from scipy import ndimage as ndi

from axon_myelin_seg.data.io import write_mask
from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances

REPO = Path(__file__).resolve().parents[2]
B8 = np.ones((3, 3), bool)


def cellpose_split(inst: np.ndarray, sem: np.ndarray, min_axon_px: int = 50) -> np.ndarray:
    out = np.zeros(inst.shape, np.uint16)
    k = 0
    for i, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        reg = inst[sl] == i
        s = sem[sl]
        ax = reg & (s == 1)
        lab, n = ndi.label(ax, structure=B8)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        if sizes.max() < min_axon_px:
            continue
        a = ndi.binary_fill_holes(lab == (np.argmax(sizes) + 1)) & reg
        m = reg & (s == 2) & ~a
        if not m.any():
            continue
        k += 1
        sub = out[sl]
        sub[m] = 2 * k + 1
        sub[a] = 2 * k
    return out


def one(args):
    name, val_dir = args
    sem = iio.imread(Path(val_dir) / f"{name}.png").astype(np.uint8)
    (REPO / "data/predictions/b1_sem").mkdir(parents=True, exist_ok=True)
    iio.imwrite(REPO / "data/predictions/b1_sem" / f"{name}.png", sem)
    inst = semantic_to_paired_instances(sem == 1, sem == 2)
    write_mask(REPO / "data/predictions/b1" / f"{name}.tif", inst)
    cp = REPO / "data/predictions/cellpose_zs_instances" / f"{name}.tif"
    if cp.exists():
        write_mask(REPO / "data/predictions/cellpose_b1" / f"{name}.tif", cellpose_split(tifffile.imread(cp), sem))
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True)
    a = ap.parse_args()
    for d in ("b1", "cellpose_b1"):
        (REPO / "data/predictions" / d).mkdir(parents=True, exist_ok=True)
    names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    with Pool(4) as p:
        print(len(p.map(one, [(n, a.val_dir) for n in names])), "images")
