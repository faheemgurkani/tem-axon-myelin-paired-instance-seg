"""Phase 0b: inventory of external datasets (ASTIH TEM1 / TEM2) + label-mapping check + by-eye overlays.

No model is run and nothing is scored. Per image: subject, size, pixel size, conversion report
(``data.external.astih_to_pairs``), fiber count, density, median fiber diameter and median g (equivalent circle).

Outputs: results/phase0b/inventory_<dataset>.csv and inventory_summary.json (counts and morphometry only);
overlays for the by-eye test in data/processed/phase0b/ (git-ignored: they contain image pixels).

    .venv/bin/python scripts/run/phase0b_inventory.py
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from axon_myelin_seg.data.external import astih_to_pairs, split_axonmyelin

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data/external"
OUT = REPO / "results/phase0b"
VIS = REPO / "data/processed/phase0b"

DATASETS = {
    # name: (root, image glob relative to root, test subjects or None)
    "TEM1": (EXT / "001436", "sub-*/micr/*_TEM.png", {"sub-nyuMouse26"}),
    "TEM2_dandi": (EXT / "001350", "sub-*/micr/*_TEM.png", None),
    "TEM2_test": (EXT / "tem2_testset/data_axondeepseg_srf_testing-r20250513-neurips2025", "sub-*/micr/*_TEM.png", None),
}


def pixel_size_um(img_path: Path) -> float | None:
    subj = img_path.name.split("_")[0]
    js = img_path.parent / f"{subj}_TEM.json"
    if js.exists():
        return float(json.loads(js.read_text())["PixelSize"][0])
    return None


def masks_for(root: Path, img_path: Path) -> dict[str, Path]:
    subj = img_path.name.split("_")[0]
    d = root / "derivatives/labels" / subj / "micr"
    stem = img_path.name[: -len(".png")]
    return {p.name[len(stem) + len("_seg-"):].replace("-manual.png", ""): p for p in d.glob(f"{stem}_seg-*-manual.png")}


def morph(pairs: np.ndarray, px_um: float) -> tuple[float, float]:
    fib = pairs // 2
    fib[pairs < 2] = 0
    ids, area = np.unique(fib[fib > 0], return_counts=True)
    ax_ids, ax_area = np.unique(fib[(pairs >= 2) & (pairs % 2 == 0)], return_counts=True)
    ax = dict(zip(ax_ids, ax_area))
    g = [np.sqrt(ax.get(i, 0) / a) for i, a in zip(ids, area) if ax.get(i, 0) > 0]
    d_nm = 2 * np.sqrt(area / np.pi) * px_um * 1000
    return (float(np.median(d_nm)) if len(d_nm) else float("nan"), float(np.median(g)) if g else float("nan"))


def overlay(img: np.ndarray, pairs: np.ndarray, path: Path) -> None:
    rgb = np.stack([img] * 3, -1).astype(np.float32)
    ax = (pairs >= 2) & (pairs % 2 == 0)
    my = (pairs >= 3) & (pairs % 2 == 1)
    rgb[ax] = 0.6 * rgb[ax] + 0.4 * np.array([60, 200, 60])
    rgb[my] = 0.6 * rgb[my] + 0.4 * np.array([220, 60, 220])
    rgb[pairs == 1] = 0.6 * rgb[pairs == 1] + 0.4 * np.array([240, 200, 0])
    rgb[find_boundaries(pairs // 2, mode="outer")] = [255, 0, 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, rgb.clip(0, 255).astype(np.uint8))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (root, pattern, test_subj) in DATASETS.items():
        imgs = sorted(p for p in root.glob(pattern) if "derivatives" not in p.parts)
        rows = []
        for j, p in enumerate(imgs):
            m = masks_for(root, p)
            if "axon" in m and "myelin" in m:
                ax_m, my_m = iio.imread(m["axon"]), iio.imread(m["myelin"])
            elif "axonmyelin" in m:
                ax_m, my_m = split_axonmyelin(iio.imread(m["axonmyelin"]))
            else:
                continue
            img = iio.imread(p)
            img = img if img.ndim == 2 else img[..., 0]
            pairs, rep = astih_to_pairs(ax_m, my_m)
            px = pixel_size_um(p)
            d_med, g_med = morph(pairs, px)
            area_um2 = img.shape[0] * img.shape[1] * px * px
            subj = p.name.split("_")[0]
            rows.append({
                "image": p.name, "subject": subj, "split": "test" if test_subj and subj in test_subj else "train",
                "h": img.shape[0], "w": img.shape[1], "px_um": px, "area_um2": area_um2,
                "n_fibers": rep.n_fibers, "n_border": rep.n_border, "n_no_axon": rep.n_no_axon,
                "n_no_myelin": rep.n_no_myelin, "n_multi_axon": rep.n_multi_axon,
                "fibers_per_um2": rep.n_fibers / area_um2, "median_fiber_d_nm": d_med, "median_g": g_med,
                "extra_classes": ",".join(sorted(set(m) - {"axon", "myelin", "axonmyelin"})),
            })
            if j < 2:
                overlay(img, pairs, VIS / f"{name}_{p.stem}_overlay.png")
        df = pd.DataFrame(rows)
        if df.empty:
            print(name, "no annotated images found (download incomplete?) - skipped")
            continue
        df.to_csv(OUT / f"inventory_{name}.csv", index=False)
        summary[name] = {
            "images": len(df), "subjects": int(df.subject.nunique()),
            "px_um": sorted(df.px_um.unique().tolist()), "image_sizes": sorted({f"{h}x{w}" for h, w in zip(df.h, df.w)}),
            "fibers": int(df.n_fibers.sum()), "border_fibers": int(df.n_border.sum()),
            "no_axon": int(df.n_no_axon.sum()), "no_myelin": int(df.n_no_myelin.sum()),
            "multi_axon": int(df.n_multi_axon.sum()),
            "fibers_per_um2_median": float(df.fibers_per_um2.median()),
            "fiber_d_nm_median_of_images": float(df.median_fiber_d_nm.median()),
            "g_median_of_images": float(df.median_g.median()),
            "split_counts": df.split.value_counts().to_dict(),
            "extra_classes": sorted({c for s in df.extra_classes for c in s.split(",") if c}),
        }
        print(name, json.dumps(summary[name]))
    (OUT / "inventory_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
