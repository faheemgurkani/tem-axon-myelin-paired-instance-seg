"""T-22 / T-24: TEM1 (ASTIH, DANDI 001436, CC-BY) as nnU-Net dataset 502, for the complete-label reference arm.

- Images: the same 2.152 nm/px rescaled PNGs B1 is run on (data/nnunet/phase2_inputs/tem1, from
  scripts/run/phase2_t22_prep_inputs.py), so B1 and the reference see identical pixels on the held-out mice.
- Labels: `astih_to_pairs` (ASTIH protocol, border fibers labelled) → semantic 0 bg / 1 axon / 2 myelin / 3 ignore
  (components lacking an axon or a myelin part, label 1 in the pair map), resized to the rescaled grid (nearest).
- Split (fold 0, written to the preprocessed folder): val = the H8/T-24 test mice = proxy cluster 2 (13, 14, 31, 32,
  33, 34) + control-like 26 (ASTIH's test mouse) and 27 (results/phase0b/tem1_mouse_proxy.csv); train = the other 12.
  The reference trains with the fixed-150-epoch trainer (D-39), so nothing is selected on the val (= test) mice.
- Plans: B1's own plans (nnUNetResEncUNetMPlans incl. 2d_bigpatch) are copied from dataset 501, so the network and
  patch are identical (the "B1 recipe"). Preprocessing runs locally.

    export nnUNet_raw=data/nnunet/raw nnUNet_preprocessed=data/nnunet/preprocessed
    python baselines/nnunet/convert_tem1.py convert
    nnUNetv2_extract_fingerprint -d 502 -np 4
    nnUNetv2_move_plans_between_datasets -s 501 -t 502 -sp nnUNetResEncUNetMPlans -tp nnUNetResEncUNetMPlans
    nnUNetv2_preprocess -d 502 -plans_name nnUNetResEncUNetMPlans -c 2d -np 4
    python baselines/nnunet/convert_tem1.py splits
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd

from axon_myelin_seg.data.external import astih_to_pairs

REPO = Path(__file__).resolve().parents[2]
NAME = "Dataset502_TEM1Ref"
TEM1 = REPO / "data/external/001436"
INPUTS = REPO / "data/nnunet/phase2_inputs/tem1"
TEST_MICE = {f"sub-nyuMouse{n}" for n in (13, 14, 31, 32, 33, 34, 26, 27)}


def semantic(pairs: np.ndarray) -> np.ndarray:
    out = np.zeros(pairs.shape, np.uint8)
    out[(pairs >= 2) & (pairs % 2 == 0)] = 1
    out[(pairs >= 3) & (pairs % 2 == 1)] = 2
    out[pairs == 1] = 3
    return out


def convert() -> None:
    root = Path(os.environ["nnUNet_raw"]) / NAME
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(INPUTS / "manifest.csv")
    rows = []
    for r in man.itertuples():
        lab_dir = TEM1 / "derivatives/labels" / r.subject / "micr"
        stem = r.source[:-4]
        pairs, _ = astih_to_pairs(iio.imread(lab_dir / f"{stem}_seg-axon-manual.png"),
                                  iio.imread(lab_dir / f"{stem}_seg-myelin-manual.png"))
        sem = cv2.resize(semantic(pairs), (r.w_in, r.h_in), interpolation=cv2.INTER_NEAREST)
        img = iio.imread(INPUTS / f"{r.case}_0000.png")
        assert img.shape == sem.shape, (r.case, img.shape, sem.shape)
        iio.imwrite(root / "imagesTr" / f"{r.case}_0000.png", img)
        iio.imwrite(root / "labelsTr" / f"{r.case}.png", sem)
        rows.append({"case": r.case, "subject": r.subject, "test": r.subject in TEST_MICE,
                     **{f"frac_{k}": float(c / sem.size) for k, c in
                        zip(("bg", "axon", "myelin", "ignore"), np.bincount(sem.ravel(), minlength=4), strict=True)}})
    ds = {"channel_names": {"0": "TEM"}, "labels": {"background": 0, "axon": 1, "myelin": 2, "ignore": 3},
          "numTraining": len(rows), "file_ending": ".png", "name": NAME,
          "description": "ASTIH TEM1 (DANDI 001436, CC-BY-4.0) rescaled to 2.152 nm/px; complete-label reference for "
                         "T-22 (H8) and T-24 (tem-axon-myelin-paired-instance-seg)",
          "licence": "CC-BY-4.0 (Jelescu et al. 2016; ASTIH, Collin et al. 2026)"}
    (root / "dataset.json").write_text(json.dumps(ds, indent=1))
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/phase2/tem1ref_cases.csv", index=False)
    print(df.groupby("test").agg(n=("case", "size"), mice=("subject", "nunique")))


def write_splits() -> None:
    df = pd.read_csv(REPO / "results/phase2/tem1ref_cases.csv")
    fold = {"train": df.case[~df.test].tolist(), "val": df.case[df.test].tolist()}
    target = Path(os.environ["nnUNet_preprocessed"]) / NAME / "splits_final.json"
    target.write_text(json.dumps([fold], indent=1))
    print(f"wrote {target}: train {len(fold['train'])}, val (= test mice) {len(fold['val'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "splits"])
    a = ap.parse_args()
    convert() if a.cmd == "convert" else write_splits()
