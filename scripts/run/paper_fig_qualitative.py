"""T-28 G7 (reviewer audit, 25 Nov): qualitative figure from ASTIH TEM1 (DANDI:001436, CC-BY 4.0) only; challenge
images are never shown (licence unstated).

Picks held-out TEM1 crops centred on thin-myelin GT fibers (g > 0.85) that B1 misses and the complete-label reference
finds (from H8's per-fiber tables), one crop per mouse, ranked by the number of such fibers in the crop, and draws
GT / B1 / reference fiber outlines. Output: docs/paper/figures/qualitative_tem1.pdf (git-ignored folder).

    PYTHONPATH=src:scripts/run .venv/bin/python scripts/run/paper_fig_qualitative.py [--n 3] [--half 450]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from phase2_t22_h8 import load_gt  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

from axon_myelin_seg.evaluation.official import paired_fiber_map  # noqa: E402
from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data/processed/phase2"
OUT = REPO / "docs/paper/figures"


def pred(path: Path, w: int, h: int) -> np.ndarray:
    sem = cv2.resize(iio.imread(path), (w, h), interpolation=cv2.INTER_NEAREST)
    return paired_fiber_map(semantic_to_paired_instances(sem == 1, sem == 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--half", type=int, default=340)
    a = ap.parse_args()
    b = pd.read_csv(PROC / "h8_fibers_b1.csv")
    r = pd.read_csv(PROC / "h8_fibers_ref.csv")
    m = r.merge(b[["case", "gt_k", "matched"]].rename(columns={"matched": "b1"}), on=["case", "gt_k"])
    hits = m[(m.g > 0.85) & m.matched & ~m.b1]
    man = pd.read_csv(REPO / "data/nnunet/phase2_inputs/tem1/manifest.csv").set_index("case")
    # one candidate crop per mouse: the case with most such fibers
    pick = (hits.groupby(["subject", "case"]).size().rename("n").reset_index()
            .sort_values("n", ascending=False).drop_duplicates("subject").head(a.n))
    fig, axes = plt.subplots(len(pick), 3, figsize=(6.8, 2.3 * len(pick)))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(pick.itertuples()):
        mr = man.loc[row.case]
        R = SimpleNamespace(set="tem1", case=row.case, subject=mr.subject, source=mr.source, px_nm=mr.px_nm)
        img, gt = load_gt(R)
        fg = paired_fiber_map(np.where(gt == 1, 0, gt))
        fb = pred(REPO / "data/predictions/b1_ext/tem1" / f"{row.case}.png", img.shape[1], img.shape[0])
        fr = pred(REPO / "data/predictions/tem1ref" / f"{row.case}.png", img.shape[1], img.shape[0])
        ks = hits[hits.case == row.case].gt_k.to_numpy()
        # centre on the densest cluster of target fibers: the target with most other targets within the crop
        cents = np.array([np.argwhere(fg == k).mean(0) for k in ks])
        cnt = [(np.abs(cents - c) < a.half).all(1).sum() for c in cents]
        cy, cx = cents[int(np.argmax(cnt))].astype(int)
        y0 = int(np.clip(cy - a.half, 0, img.shape[0] - 2 * a.half))
        x0 = int(np.clip(cx - a.half, 0, img.shape[1] - 2 * a.half))
        sl = (slice(y0, y0 + 2 * a.half), slice(x0, x0 + 2 * a.half))
        target = np.isin(fg, ks)[sl]
        for j, (title, lab, col) in enumerate([("image + GT outlines", fg, "#56B4E9"),
                                               ("B1 (selective labels): fibers found", fb, "#E69F00"),
                                               ("reference (complete labels): found", fr, "#009E73")]):
            ax = axes[i, j]
            rgb = np.dstack([img[sl]] * 3).astype(np.float32) / 255
            c = np.array(matplotlib.colors.to_rgb(col))
            if j == 0:
                bd = binary_dilation(find_boundaries(lab[sl], mode="inner"), iterations=2)
                rgb[bd] = 0.5 * rgb[bd] + 0.5 * c
            else:  # translucent fill of every predicted fiber
                fill = lab[sl] > 0
                rgb[fill] = 0.6 * rgb[fill] + 0.4 * c
            tb = binary_dilation(find_boundaries(target, mode="outer"), iterations=3)
            rgb[tb] = np.array([1.0, 0.0, 1.0])
            ax.imshow(rgb)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{mr.subject.replace('sub-nyu', '')}", fontsize=7)
        um = 2 * a.half * mr.px_nm / 1000
        axes[i, 0].text(8, 2 * a.half - 14, f"{um:.1f} µm wide", color="w", fontsize=6)
    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qualitative_tem1.pdf", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "qualitative_tem1.png", dpi=110, bbox_inches="tight")
    print(pick.to_string(index=False), "\nthin missed by B1, found by ref (held-out):", len(hits))


if __name__ == "__main__":
    main()
