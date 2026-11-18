"""Phase 3a, exploratory (07-plan §7.2 "Secondary"): H5a/b/c measures on AxonPath CC TEM (hand-drawn pairs).

B1 zero-shot predictions (data/predictions/b1_ext/axonpath, 2.152 nm grid) resized back to native (nearest).
Conditions: base (D-11), h5c (frozen 7c673ca), o_sep. GT: `axonpath_geojson_to_pairs` (ROI only).
Context only, not confirmatory. Image bootstrap (no overlap groups on AxonPath), 1,000 resamples.

    .venv/bin/python scripts/run/phase3a_h5_axonpath.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
from phase2_t22_h8 import load_gt
from phase3a_h5 import H5_CONDS, per_fiber

from axon_myelin_seg.evaluation.oracles import o_sep
from axon_myelin_seg.postprocessing.pairing import (
    semantic_to_paired_instances,
    semantic_to_paired_instances_h5c,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/phase3a"


def one(row):
    sem = cv2.resize(iio.imread(REPO / "data/predictions/b1_ext/axonpath" / f"{row.case}.png"), (row.w, row.h),
                     interpolation=cv2.INTER_NEAREST)
    _, gt = load_gt(row)
    base = semantic_to_paired_instances(sem == 1, sem == 2)
    preds = {"base": base, "h5c": semantic_to_paired_instances_h5c(sem == 1, sem == 2), "o_sep": o_sep(base, gt, sem)}
    fb = {c: per_fiber(preds[c], gt, defects=(c == "base")) for c in H5_CONDS}
    rows = []
    for k in sorted(set().union(*fb.values())):
        r = {"case": row.case, "gt_k": k, "defect_base": fb["base"][k]["defect"] if k in fb["base"] else np.nan}
        for c in H5_CONDS:
            r[f"err_{c}"] = fb[c][k]["err"] if k in fb[c] else np.nan
        rows.append(r)
    return rows


def stats(f: pd.DataFrame) -> dict:
    g = {c: float(f[f"err_{c}"].abs().mean()) for c in H5_CONDS}
    rel = (g["base"] - g["o_sep"]) / g["base"]
    cap = (g["base"] - g["h5c"]) / (g["base"] - g["o_sep"])
    return {"g_mae": g, "osep_rel_reduction": rel, "h5c_capture": cap}


def main() -> None:
    man = pd.read_csv(REPO / "data/nnunet/phase2_inputs/axonpath/manifest.csv")
    with Pool(4) as p:
        F = pd.DataFrame([r for rs in p.map(one, [SimpleNamespace(**r) for r in man.to_dict("records")]) for r in rs])
    S = {"n_images": int(F.case.nunique()), "n_fibers_base": int(F.err_base.notna().sum()), **stats(F)}
    fb = F[F.err_base.notna()]
    ae = fb.err_base.abs()
    dfx = fb.defect_base.astype(bool)
    both = F[F.err_base.notna() & F.err_o_sep.notna()]
    red = both.err_base.abs() - both.err_o_sep.abs()
    S["defective_share_of_matched"] = float(dfx.mean())
    S["defective_share_of_abs_g_error"] = float(ae[dfx].sum() / ae.sum())
    S["o_sep_reduction_share_from_defective"] = float(red[both.defect_base.astype(bool)].sum() / red.sum())
    rng = np.random.default_rng(0)
    cases = F.case.unique()
    grp = {c: F[F.case == c] for c in cases}
    bs = [stats(pd.concat([grp[c] for c in rng.choice(cases, len(cases))])) for _ in range(1000)]
    for key in ("osep_rel_reduction", "h5c_capture"):
        S[f"{key}_ci95"] = [float(np.percentile([b[key] for b in bs], q)) for q in (2.5, 97.5)]
    S["note"] = "Exploratory; fiber-weighted over matched fibers; image bootstrap. Not part of the H5 rule."
    F.to_csv(OUT / "h5_axonpath_fibers.csv", index=False)
    (OUT / "h5_axonpath_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
