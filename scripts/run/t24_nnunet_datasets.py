"""T-24 step 4 (D-42): the four arms as nnU-Net datasets 503–506, preprocessed locally with Dataset502's plans.

- Raw: imagesTr are symlinks to Dataset502's (identical pixels); labelsTr from data/processed/t24/arms/R*/.
- Preprocessed: 502's plans (dataset name swapped), fingerprint and dataset.json copied in; `nnUNetv2_preprocess -c 2d`;
  splits_final.json = 502's (train = 12 mice, val = the 8 test mice).
- Round 2 (H10): R6 oracle (507), R4b PU soft weights (508, weight channel from R4b_d), R4h PU-gated ignore (509).
- R4 (506) and R4b (508): each `_seg.b2nd` is rewritten with a 2nd channel, the down-weight level d (0..20; trainer
  `nnUNetTrainer_80epochs_ckpt10_softw`), cropped with the case's `bbox_used_for_cropping`.
- Check: the preprocessed image of one case must be byte-identical to 502's, so on the Volume the image files can be
  copied server-side from 502 and only seg/pkl/json files uploaded.

    export nnUNet_raw=data/nnunet/raw nnUNet_preprocessed=data/nnunet/preprocessed nnUNet_results=data/nnunet/results
    .venv/bin/python scripts/run/t24_nnunet_datasets.py [--arms R1 R2 R3 R4]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import blosc2
import imageio.v3 as iio
import numpy as np

REPO = Path(__file__).resolve().parents[2]
T24 = REPO / "data/processed/t24/arms"
RAW = Path(os.environ.get("nnUNet_raw", REPO / "data/nnunet/raw"))
PRE = Path(os.environ.get("nnUNet_preprocessed", REPO / "data/nnunet/preprocessed"))
SRC = "Dataset502_TEM1Ref"
ARMS = {"R1": "Dataset503_T24R1naive", "R2": "Dataset504_T24R2hardignore", "R3": "Dataset505_T24R3randignore",
        "R4": "Dataset506_T24R4softw",
        # round 2 (H10, D-44–D-47; labels from t24_build_arms_v2.py)
        "R6": "Dataset507_T24R6oracle", "R4b": "Dataset508_T24R4bPUsoftw", "R4h": "Dataset509_T24R4hPUgated"}
WEIGHT_ARMS = {"R4": "R4_d", "R4b": "R4b_d"}  # arms whose seg gets the down-weight channel
PLANS = "nnUNetResEncUNetMPlans"


def build(arm: str) -> None:
    name = ARMS[arm]
    num = int(name[7:10])
    raw = RAW / name
    (raw / "labelsTr").mkdir(parents=True, exist_ok=True)
    (raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    for f in sorted((RAW / SRC / "imagesTr").glob("*.png")):
        dst = raw / "imagesTr" / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
    for f in sorted((T24 / arm).glob("*.png")):
        shutil.copy(f, raw / "labelsTr" / f.name)
    ds = json.loads((RAW / SRC / "dataset.json").read_text())
    ds["name"] = name
    ds["description"] = f"T-24 arm {arm} (C3 pilot, D-35/D-42): ASTIH TEM1 with simulated two-part omission on the " \
                        f"training mice; images identical to {SRC}"
    (raw / "dataset.json").write_text(json.dumps(ds, indent=1))

    pre = PRE / name
    pre.mkdir(parents=True, exist_ok=True)
    plans = json.loads((PRE / SRC / f"{PLANS}.json").read_text())
    plans["dataset_name"] = name
    (pre / f"{PLANS}.json").write_text(json.dumps(plans, indent=1))
    for f in ("dataset_fingerprint.json", "splits_final.json"):
        shutil.copy(PRE / SRC / f, pre / f)
    shutil.copy(raw / "dataset.json", pre / "dataset.json")
    subprocess.run(["nnUNetv2_preprocess", "-d", str(num), "-plans_name", PLANS, "-c", "2d", "-np", "4"], check=True,
                   env={**os.environ, "nnUNet_raw": str(RAW), "nnUNet_preprocessed": str(PRE)},
                   cwd=REPO)
    if arm in WEIGHT_ARMS:
        add_weight_channel(pre / "nnUNetPlans_2d", WEIGHT_ARMS[arm])
    a = iio.imread(raw / "labelsTr" / "tem1_0000.png")
    same = (pre / "nnUNetPlans_2d/tem1_0000.b2nd").read_bytes() == (PRE / SRC / "nnUNetPlans_2d/tem1_0000.b2nd").read_bytes()
    print(f"{arm}: {name} done; image b2nd identical to 502: {same}; label values {np.unique(a)}")


def add_weight_channel(folder: Path, d_dir: str) -> None:
    for seg_f in sorted(folder.glob("*_seg.b2nd")):
        case = seg_f.name[:-len("_seg.b2nd")]
        props = pickle.loads((folder / f"{case}.pkl").read_bytes())
        bb = props["bbox_used_for_cropping"]
        seg = blosc2.open(str(seg_f), mode="r")
        chunks, blocks = seg.chunks, seg.blocks
        s = seg[:]
        d = iio.imread(T24 / d_dir / f"{case}.png").astype(np.int16)[bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]
        d = np.where(s[0, 0] < 0, 0, d)
        out = np.concatenate([s, d[None, None]], axis=0)
        seg_f.unlink()
        blosc2.asarray(np.ascontiguousarray(out), urlpath=str(seg_f), chunks=(1, *chunks[1:]), blocks=(1, *blocks[1:]),
                       cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 5, "nthreads": 4})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    for arm in ap.parse_args().arms:
        build(arm)


if __name__ == "__main__":
    main()
