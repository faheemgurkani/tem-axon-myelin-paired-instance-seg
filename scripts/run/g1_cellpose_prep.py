"""T-30 (G1, H11): training data for a second, non-nnU-Net model trained on our labels (Cellpose-SAM fine-tune). CPU.

Cellpose works on instance maps of one object type, so two targets are written per image, both from the cleaned
training mask (orphans and topology-defect fibers -> label 1, as for B1):
  fiber: instance id = X // 2 on axon X and its ring X+1 (whole fiber);
  axon:  instance id = X // 2 on axon pixels only.
Cellpose has no ignore label, so label 1 (border-cut and deliberately excluded fibers, cleaned fibers) becomes
background, i.e. the model is trained on the selective labels as given; predictions on GT label 1 are ignored at
evaluation, as for every model. Images and labels are downscaled by 6 (2.152 -> 12.9 nm/px; fibers ~58 px, the scale
of the zero-shot Cellpose+B1 baseline, D-26): image INTER_AREA, labels nearest.
Split: configs/splits/dev.json (213 train / 53 val), the split B1 used.

Output (git-ignored): data/processed/g1_cellpose/{train,val}/<name>_{img,fiber,axon}.tif

    PYTHONPATH=src .venv/bin/python scripts/run/g1_cellpose_prep.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import tifffile

from axon_myelin_seg.data.cleaning import clean_training_mask
from axon_myelin_seg.data.io import load_sample

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/processed/g1_cellpose"
DS = 6


def one(args: tuple[str, str]) -> tuple[str, int, int]:
    name, part = args
    img, mask = load_sample(name)
    m, _ = clean_training_mask(mask)
    H, W = img.shape
    size = (W // DS, H // DS)
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    fiber = np.where(m >= 2, m // 2, 0).astype(np.int32)
    axon = np.where((m >= 2) & (m % 2 == 0), m // 2, 0).astype(np.int32)
    fs = cv2.resize(fiber.astype(np.float32), size, interpolation=cv2.INTER_NEAREST).astype(np.uint16)
    ax = cv2.resize(axon.astype(np.float32), size, interpolation=cv2.INTER_NEAREST).astype(np.uint16)
    d = OUT / part
    stem = name.removesuffix(".tif")
    tifffile.imwrite(d / f"{stem}_img.tif", small)
    tifffile.imwrite(d / f"{stem}_fiber.tif", fs, compression="zlib")
    tifffile.imwrite(d / f"{stem}_axon.tif", ax, compression="zlib")
    return name, int(len(np.unique(fs)) - 1), int(len(np.unique(ax)) - 1)


def main() -> None:
    split = json.loads((REPO / "configs/splits/dev.json").read_text())
    for part in ("train", "val"):
        (OUT / part).mkdir(parents=True, exist_ok=True)
    jobs = [(n, "train") for n in split["train"]] + [(n, "val") for n in split["val"]]
    with Pool(6) as p:
        res = p.map(one, jobs)
    print(len(res), "images; fibers", sum(r[1] for r in res), "axons", sum(r[2] for r in res))


if __name__ == "__main__":
    main()
