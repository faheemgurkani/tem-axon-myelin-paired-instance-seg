"""Phase 0b completeness audit (replaces the C4 assumption): how many unlabelled ring-like structures each dataset has.

Same detector on every dataset (``taxonomy.ring_candidates``), with the downsampling chosen so the working pixel is
~9 nm everywhere. Per image: candidates, labelled fibers, GT background area. Per dataset: candidates per 100 µm² of
background, and the unlabelled share u = candidates / (candidates + labelled fibers) — a *lower bound* on
incompleteness (the detector misses thin-myelin rings) that also counts protocol exclusions and look-alikes, which is
why a by-eye check of a random sample follows (galleries in data/processed/phase0b/, git-ignored).

No model is run. Outputs: results/phase0b/completeness_<dataset>.csv, completeness_summary.json.

    .venv/bin/python scripts/run/phase0b_completeness_audit.py [--datasets ours TEM1 TEM2_test TEM2_dandi]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from axon_myelin_seg.data.external import (
    astih_to_pairs,
    axonpath_geojson_to_pairs,
    normalise_to_uint8,
    split_axonmyelin,
)
from axon_myelin_seg.data.io import list_samples, load_sample
from axon_myelin_seg.evaluation.taxonomy import ring_candidates

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data/external"
OUT = REPO / "results/phase0b"
VIS = REPO / "data/processed/phase0b"
TEM2_TEST = EXT / "tem2_testset/data_axondeepseg_srf_testing-r20250513-neurips2025"


def astih_iter(root: Path):
    for p in sorted(root.glob("sub-*/micr/*_TEM.png")):
        subj = p.name.split("_")[0]
        d = root / "derivatives/labels" / subj / "micr"
        stem = p.name[:-4]
        a, m = d / f"{stem}_seg-axon-manual.png", d / f"{stem}_seg-myelin-manual.png"
        am = d / f"{stem}_seg-axonmyelin-manual.png"
        if a.exists() and m.exists():
            ax_m, my_m = iio.imread(a), iio.imread(m)
        elif am.exists():
            ax_m, my_m = split_axonmyelin(iio.imread(am))
        else:
            continue
        img = iio.imread(p)
        img = img if img.ndim == 2 else img[..., 0]
        pairs, _ = astih_to_pairs(ax_m, my_m)
        yield p.name, subj, img, pairs


AXONPATH = EXT / "axonpath_zenodo/CorpusCallosum_Mus_TEM_000/raw"
# the 8 images AxonPath re-annotated from AimSeg (names in AimSeg's Zenodo 8351731 archives); "subject" column = source
AIMSEG = {"4946_3_1_72_18_0003", "7_EM1_3_10D_N_P_0002", "7_EM1_3_10D_N_P_0004", "8_E72_18_0014", "E72_21_0003",
          "exp69_72_EM10N_NOLPC_P5_0001", "exp69_72_EM10N_NOLPC_P5_0002", "exp69_72_EM10N_NOLPC_P5_0004"}


def axonpath_iter():
    for p in sorted((AXONPATH / "images").glob("*.tif")):
        img = normalise_to_uint8(iio.imread(p))
        pairs, _ = axonpath_geojson_to_pairs(AXONPATH / "annotations" / f"{p.stem}.geojson", img.shape)
        yield p.name, "aimseg" if p.stem in AIMSEG else "new", img, pairs


def ours_iter():
    for name in list_samples():
        img, gt = load_sample(name)
        yield name, "na", img, gt


DATASETS = {
    # name: (iterator factory, px_nm, ds)
    "ours": (ours_iter, 2.152, 4),
    "TEM1": (lambda: astih_iter(EXT / "001436"), 2.36, 4),
    "TEM2_test": (lambda: astih_iter(TEM2_TEST), 4.93, 2),
    "TEM2_dandi": (lambda: astih_iter(EXT / "001350"), 4.93, 2),
    "AxonPath": (axonpath_iter, 2.10, 4),  # 1.75–2.11 nm per image; 2.10 for all (one image at 1.75)
}


def gallery(crops: list[np.ndarray], path: Path, ncol: int = 6) -> None:
    if not crops:
        return
    s = crops[0].shape[0]
    nrow = int(np.ceil(len(crops) / ncol))
    canvas = np.full((nrow * (s + 6), ncol * (s + 6), 3), 255, np.uint8)
    for i, c in enumerate(crops):
        r, q = divmod(i, ncol)
        canvas[r * (s + 6): r * (s + 6) + s, q * (s + 6): q * (s + 6) + s] = c
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, canvas)


def crop_rgb(img: np.ndarray, gt: np.ndarray, cand: np.ndarray, cy: int, cx: int, half: int, out: int = 200):
    h, w = img.shape
    y0, y1, x0, x1 = max(0, cy - half), min(h, cy + half), max(0, cx - half), min(w, cx + half)
    g = img[y0:y1, x0:x1].astype(np.float32)
    rgb = np.stack([g] * 3, -1)
    lab = gt[y0:y1, x0:x1]
    rgb[(lab >= 3) & (lab % 2 == 1)] = 0.7 * rgb[(lab >= 3) & (lab % 2 == 1)] + 0.3 * np.array([220, 60, 220])
    edge = cand[y0:y1, x0:x1] > 0
    edge = edge & ~ndi.binary_erosion(edge, iterations=2)
    rgb[edge] = [255, 220, 0]
    pad = np.full((2 * half, 2 * half, 3), 255, np.float32)
    pad[: rgb.shape[0], : rgb.shape[1]] = rgb
    idx = (np.arange(out) * (2 * half) / out).astype(int)
    return pad[idx][:, idx].clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ours", "TEM1", "TEM2_test", "TEM2_dandi"])
    ap.add_argument("--n-gallery", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summ_path = OUT / "completeness_summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    for name in args.datasets:
        factory, px_nm, ds = DATASETS[name]
        rows, cand_pool = [], []
        for img_name, subj, img, gt in factory():
            cand = ring_candidates(img, gt, ds=ds, px_nm=px_nm)
            ids = np.unique(cand[cand > 0])
            fib = np.unique(gt[gt >= 2] // 2)
            bg_um2 = (gt == 0).sum() * (px_nm / 1000) ** 2
            rows.append({"image": img_name, "subject": subj, "n_candidates": len(ids), "n_fibers": len(fib),
                         "bg_um2": bg_um2, "img_um2": (gt != 1).sum() * (px_nm / 1000) ** 2})
            if len(ids):
                com = ndi.center_of_mass(cand > 0, cand, ids)
                cand_pool += [(img_name, int(cy), int(cx)) for cy, cx in com]
        df = pd.DataFrame(rows)
        if df.empty:
            print(name, "no images - skipped")
            continue
        df.to_csv(OUT / f"completeness_{name}.csv", index=False)
        nc, nf, bg = df.n_candidates.sum(), df.n_fibers.sum(), df.bg_um2.sum()
        summary[name] = {"images": len(df), "px_nm": px_nm, "ds": ds, "candidates": int(nc), "fibers": int(nf),
                         "candidates_per_100um2_bg": float(100 * nc / bg),
                         "fibers_per_100um2_img": float(100 * nf / df.img_um2.sum()),
                         "unlabelled_share_u": float(nc / (nc + nf)),
                         "u_per_image_median": float((df.n_candidates / (df.n_candidates + df.n_fibers)).median())}
        print(name, json.dumps(summary[name]))
        # gallery of a random sample for the by-eye check
        if cand_pool:
            pick = rng.choice(len(cand_pool), size=min(args.n_gallery, len(cand_pool)), replace=False)
            by_img: dict[str, list] = {}
            for i in pick:
                by_img.setdefault(cand_pool[i][0], []).append(cand_pool[i][1:])
            half = round(900 / px_nm)  # 1.8 µm field of view
            crops, index = [], []
            for img_name, img, gt in ((n, i, g) for n, _, i, g in factory() if n in by_img):
                cand = ring_candidates(img, gt, ds=ds, px_nm=px_nm)
                for cy, cx in by_img[img_name]:
                    crops.append(crop_rgb(img, gt, cand, cy, cx, half))
                    index.append({"k": len(index), "image": img_name, "cy": cy, "cx": cx})
            for j in range(0, len(crops), 24):
                gallery(crops[j: j + 24], VIS / f"audit_{name}_{j // 24}.png")
            pd.DataFrame(index).to_csv(VIS / f"audit_{name}_index.csv", index=False)
    summ_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
