"""T-08: convert the challenge training set to nnU-Net v2 raw format (Dataset501_TEMAxonMyelin).

Semantic labels (nnU-Net requires the ignore label to be the highest integer and to be named "ignore"):
- 0 background: GT 0, which includes unmyelinated axons and unlabelled fibers
- 1 axon: GT even X ≥ 2
- 2 myelin: GT odd X ≥ 3
- 3 ignore: GT 1 (border-cut and deliberately excluded fibers), orphan IDs, and topology-defect fibers (T-02, D-21)

Images are written as 8-bit grayscale PNG (lossless) with the ``_0000`` channel suffix. With PNG, nnU-Net uses
NaturalImage2DIO and spacing (999, 1, 1). Our pixels are square (2.152 nm), so the spacing is irrelevant.
Channel name "TEM" → nnU-Net's default per-image z-score normalisation (anything that is not "CT").

splits_final.json (written into the *preprocessed* folder after planning, see write_splits):
- fold 0 = configs/splits/dev.json (the 213/53 dev split)
- folds 1–3 = configs/splits/cv3.json folds 0–2 (used for out-of-fold predictions in C3)

Usage:
    export nnUNet_raw=data/nnunet/raw nnUNet_preprocessed=data/nnunet/preprocessed
    python baselines/nnunet/convert.py convert [--workers 4]
    nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEncM -np 4
    python baselines/nnunet/convert.py splits
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from axon_myelin_seg.data.cleaning import clean_training_mask
from axon_myelin_seg.data.io import list_samples, load_sample

REPO = Path(__file__).resolve().parents[2]
NAME = "Dataset501_TEMAxonMyelin"


def semantic(mask_clean: np.ndarray) -> np.ndarray:
    out = np.zeros(mask_clean.shape, np.uint8)
    out[(mask_clean >= 2) & (mask_clean % 2 == 0)] = 1
    out[(mask_clean >= 3) & (mask_clean % 2 == 1)] = 2
    out[mask_clean == 1] = 3
    return out


def _one(args):
    name, root = args
    img, mask = load_sample(name)
    clean, log = clean_training_mask(mask)
    lab = semantic(clean)
    iio.imwrite(root / "imagesTr" / f"{name}_0000.png", img)
    iio.imwrite(root / "labelsTr" / f"{name}.png", lab)
    counts = np.bincount(lab.ravel(), minlength=4)
    return {"name": name, **{f"frac_{k}": float(c / lab.size) for k, c in zip(("bg", "axon", "myelin", "ignore"),
                                                                               counts, strict=True)},
            "n_orphan_ids": log["n_orphan_ids"], "n_defect_fibers": log["n_defect_fibers"]}


def convert(workers: int) -> None:
    root = Path(os.environ["nnUNet_raw"]) / NAME
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    names = list_samples()
    with Pool(workers) as p:
        rows = p.map(_one, [(n, root) for n in names])
    ds = {"channel_names": {"0": "TEM"}, "labels": {"background": 0, "axon": 1, "myelin": 2, "ignore": 3},
          "numTraining": len(names), "file_ending": ".png", "name": NAME,
          "description": "SIPAIM 2026 TEM axon/myelin training set, semantic labels; ignore = GT label 1 + orphan "
                         "IDs + topology-defect fibers (tem-axon-myelin-paired-instance-seg T-08)",
          "licence": "unstated by the challenge organisers; private use only"}
    (root / "dataset.json").write_text(json.dumps(ds, indent=1))
    import pandas as pd

    df = pd.DataFrame(rows)
    out = REPO / "results" / "day3"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "nnunet_label_fractions.csv", index=False)
    print(df.drop(columns="name").sum(numeric_only=True).to_dict() | {"mean_frac": df.filter(like="frac").mean().to_dict()})


def write_splits() -> None:
    dev = json.loads((REPO / "configs/splits/dev.json").read_text())
    cv3 = json.loads((REPO / "configs/splits/cv3.json").read_text())["folds"]
    folds = [{"train": dev["train"], "val": dev["val"]}] + [{"train": f["train"], "val": f["val"]} for f in cv3]
    target = Path(os.environ["nnUNet_preprocessed"]) / NAME / "splits_final.json"
    target.write_text(json.dumps(folds, indent=1))
    print(f"wrote {target}: folds", [(len(f["train"]), len(f["val"])) for f in folds])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "splits"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    convert(a.workers) if a.cmd == "convert" else write_splits()
