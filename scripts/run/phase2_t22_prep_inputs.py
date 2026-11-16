"""Phase 2, T-22 step 1 (tranche B0, opened by the owner 16 Nov): B1 zero-shot inputs for the outside datasets.

B1 was trained at 2.152 nm/px and nnU-Net's 2-D natural-image setting does no physical resampling, so every outside
image is resized to 2.152 nm/px here (bicubic, CPU) and predictions are resized back to the native grid (nearest) when
scored. Sets (annotated images only):
  tem1        158 images, 2.36 nm      (all 20 mice: H8 needs the held-out mice; t_k for T-24 needs all)
  tem2_dandi   10 images, 4.93 nm      (4675² frames become ~10.7k²)
  tem2_test    21 ROIs,   4.93 nm
  axonpath     20 images, 1.75–2.11 nm (per-image size from pixel_sizes.csv; 1–99 % normalised raw TIFFs)
Output (git-ignored): data/nnunet/phase2_inputs/<set>/<case>_0000.png and manifest.csv (case, source, shape, scale).
Upload: modal volume put tem-axon-myelin data/nnunet/phase2_inputs/<set> /phase2_inputs/<set>

    .venv/bin/python scripts/run/phase2_t22_prep_inputs.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd

from axon_myelin_seg.data.external import normalise_to_uint8

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data/external"
OUT = REPO / "data/nnunet/phase2_inputs"
B1_PX = 2.152
spec = importlib.util.spec_from_file_location("audit", REPO / "scripts/run/phase0b_completeness_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sources():
    for root, name in ((EXT / "001436", "tem1"), (EXT / "001350", "tem2_dandi"), (audit.TEM2_TEST, "tem2_test")):
        px = 2.36 if name == "tem1" else 4.93
        for fname, subj, img, _ in audit.astih_iter(root):
            yield name, fname, subj, img, px
    ps = pd.read_csv(EXT / "axonpath_zenodo/pixel_sizes.csv")
    ps = ps[ps.dataset == "CorpusCallosum_Mus_TEM_000"].set_index("image")
    for p in sorted((audit.AXONPATH / "images").glob("*.tif")):
        r = ps.loc[p.stem]
        px = 1000 * (r.pixel_size_x_um + r.pixel_size_y_um) / 2
        yield "axonpath", p.name, "aimseg" if p.stem in audit.AIMSEG else "new", normalise_to_uint8(iio.imread(p)), px


def main() -> None:
    rows = []
    for name, fname, subj, img, px in sources():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        s = px / B1_PX
        h, w = img.shape
        out = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
        case = f"{name}_{len([r for r in rows if r['set'] == name]):04d}"
        iio.imwrite(d / f"{case}_0000.png", out.astype(np.uint8))
        rows.append({"set": name, "case": case, "source": fname, "subject": subj, "px_nm": px, "scale": s,
                     "h": h, "w": w, "h_in": out.shape[0], "w_in": out.shape[1]})
    m = pd.DataFrame(rows)
    for name, d in m.groupby("set"):
        d.to_csv(OUT / name / "manifest.csv", index=False)
    print(m.groupby("set").agg(n=("case", "size"), px=("px_nm", "median"), h_in=("h_in", "median"),
                               w_in=("w_in", "median")))


if __name__ == "__main__":
    main()
