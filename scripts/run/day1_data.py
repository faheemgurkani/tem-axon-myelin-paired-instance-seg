"""Day 1 (T-01, T-02, T-03): load and validate every sample, count cleaning actions, build frozen splits.

Outputs (committed, numbers only):
- results/day1/images.csv: per image: shape/dtype OK, #fibers, density, orphans, defects, ignored px
- results/day1/cleaning_summary.json
- configs/splits/dev.json, configs/splits/cv3.json

Usage: python scripts/run/day1_data.py [--workers 4]
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from axon_myelin_seg.data.cleaning import clean_training_mask
from axon_myelin_seg.data.encoding import fiber_ids
from axon_myelin_seg.data.io import list_samples, load_sample
from axon_myelin_seg.data.splits import check_split, image_groups, make_splits

REPO = Path(__file__).resolve().parents[2]
PX_UM2 = (2.152e-3) ** 2


def per_image(name):
    try:
        img, mask = load_sample(name)
        ok, err = True, ""
    except ValueError as e:  # keep going; report
        return {"name": name, "ok": False, "error": str(e)}
    _, log = clean_training_mask(mask)
    n_f = len(fiber_ids(mask))
    valid_px = int((mask != 1).sum())
    return {"name": name, "ok": ok, "error": err, "img_dtype": str(img.dtype), "mask_dtype": str(mask.dtype),
                "n_fibers": n_f, "density_per_um2": n_f / (valid_px * PX_UM2),
                "frac_ignore_gt": float((mask == 1).mean()),
                "n_orphan_ids": log["n_orphan_ids"], "n_defect_fibers": log["n_defect_fibers"],
                "cleaned_extra_ignore_px": log["ignored_px"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    names = list_samples()
    with Pool(args.workers) as p:
        rows = p.map(per_image, names)
    df = pd.DataFrame(rows)
    out = REPO / "results" / "day1"
    out.mkdir(parents=True, exist_ok=True)
    groups = image_groups(names)
    df["group"] = df.name.map(groups)
    df.to_csv(out / "images.csv", index=False)

    summary = {"n_samples": len(names), "n_load_ok": int(df.ok.sum()),
                   "n_fibers": int(df.n_fibers.sum()),
                   "n_orphan_ids": int(df.n_orphan_ids.sum()), "n_defect_fibers": int(df.n_defect_fibers.sum()),
                   "images_with_defects": int((df.n_defect_fibers > 0).sum()),
                   "cleaned_extra_ignore_px": int(df.cleaned_extra_ignore_px.sum()),
                   "n_groups": int(df.group.nunique()),
                   "density_min": float(df.density_per_um2.min()), "density_max": float(df.density_per_um2.max())}
    (out / "cleaning_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    density = dict(zip(df.name, df.density_per_um2, strict=True))
    dev = make_splits(names, density, n_splits=5, seed=0)[0]
    cv3 = make_splits(names, density, n_splits=3, seed=0)
    check_split(dev, names)
    for f in cv3:
        check_split(f, names)
    meta = {"created": "2026-11-13", "seed": 0, "method": "StratifiedGroupKFold over overlap groups, strata = "
                "fiber-density tercile of the group mean (src/axon_myelin_seg/data/splits.py)"}
    splits = REPO / "configs" / "splits"
    (splits / "dev.json").write_text(json.dumps({**meta, "kind": "dev 80/20 (fold 0 of 5)", **dev}, indent=1))
    (splits / "cv3.json").write_text(json.dumps({**meta, "kind": "3-fold", "folds": cv3}, indent=1))
    d = df.set_index("name").density_per_um2
    for nm, f in [("dev", dev)] + [(f"cv3_{i}", f) for i, f in enumerate(cv3)]:
        print(nm, len(f["train"]), len(f["val"]),
              "val density median {:.3f} vs train {:.3f}".format(np.median(d[f["val"]]), np.median(d[f["train"]])),
              "val fibers", int(df.set_index("name").n_fibers[f["val"]].sum()))


if __name__ == "__main__":
    main()
