"""T-07: annotation-noise ceilings from overlapping tiles, scored with our evaluator.

Twenty-two pairs of training tiles show the same tissue, annotated independently (docs/notes/02-challenge-and-data.md §4).
Treating tile A's annotation as GT and tile B's annotation, warped into A's frame, as the "prediction" gives the
score of a *second annotator*. No model can be expected to beat it reliably.

Registration (fixes a bug in scripts/eda/overlap_agree.py, see docs/notes/05-results.md §2):
- Initialise from the ORB similarity A→B in data/processed/eda/overlap.csv.
- Refine with ECC (affine, 1/2 resolution) using template = A and input = B, so the warp maps A coords → B coords,
  the same direction as the initialisation.
- Keep whichever of {ORB, ECC} gives the higher NCC on the overlap. Pairs with NCC < 0.8 are reported but not used.

Pairs scored:
- the 22 EDA pairs (≥ 20 ORB inliers);
- every other pair from the full 710-row ORB scan that reached NCC ≥ 0.8 after the corrected ECC
  (results/day2/registration_scan_all710.csv). Those are *new* overlaps the EDA missed; see docs/notes/05-results.md §2.

Fairness rule: pixels that are label 1 in *either* annotation are ignored. A fiber cut by B's tile border is
label 1 in B but whole in A, and that is not annotator disagreement.

Outputs:
- results/day2/ceilings_pairs.csv: one row per pair, including the 2×3 transform;
- results/day2/ceilings_summary.json.
"""

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from axon_myelin_seg.data.io import load_sample
from axon_myelin_seg.evaluation.diagnostics import diagnose_image
from axon_myelin_seg.evaluation.official import evaluate_image

REPO = Path(__file__).resolve().parents[2]
NCC_MIN = 0.8


def warp_b_to_a(arr_b, M_ab, shape, interp):
    """dst(x_A) = B(M_ab · x_A)."""
    return cv2.warpAffine(arr_b, M_ab, (shape[1], shape[0]), flags=interp | cv2.WARP_INVERSE_MAP, borderValue=0)


def ncc(ia, ib_w, valid):
    return float(np.corrcoef(ia[valid].astype(float), ib_w[valid].astype(float))[0, 1])


def one(row):
    ia, ma = load_sample(row["a"])
    ib, mb = load_sample(row["b"])
    H, W = ia.shape
    ang = np.radians(row["angle"])
    M0 = np.array([[row["scale"] * np.cos(ang), -row["scale"] * np.sin(ang), row["tx"]],
                   [row["scale"] * np.sin(ang), row["scale"] * np.cos(ang), row["ty"]]], np.float32)

    def score(M):
        ib_w = warp_b_to_a(ib, M, (H, W), cv2.INTER_LINEAR)
        valid = warp_b_to_a(np.ones_like(ib), M, (H, W), cv2.INTER_NEAREST).astype(bool)
        valid = ndi.binary_erosion(valid, iterations=20)
        return (ncc(ia, ib_w, valid) if valid.sum() > 1e5 else -1.0), valid

    cands = {"orb": M0}
    s = 2
    Ms = M0.copy()
    Ms[:, 2] /= s
    try:
        _, Ms = cv2.findTransformECC(ia[::s, ::s].astype(np.float32), ib[::s, ::s].astype(np.float32), Ms,
                                     cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        Ms[:, 2] *= s
        cands["ecc"] = Ms
    except cv2.error:
        pass
    scored = {k: score(M) for k, M in cands.items()}
    best = max(scored, key=lambda k: scored[k][0])
    n_best, valid = scored[best]
    out = {"a": row["a"], "b": row["b"], "inliers": int(row["inliers"]), "ncc_orb": scored["orb"][0],
           "ncc_ecc": scored["ecc"][0] if "ecc" in scored else np.nan, "method": best, "ncc": n_best,
           "overlap_frac": float(valid.mean()), "M": json.dumps(np.asarray(cands[best]).round(5).tolist())}
    if n_best < NCC_MIN:
        return out
    mb_w = warp_b_to_a(mb, cands[best], (H, W), cv2.INTER_NEAREST)
    gt = ma.copy()
    gt[~valid] = 1                     # outside the shared field of view → ignore in both maps
    gt[mb_w == 1] = 1                  # label 1 in B (e.g. cut by B's border) → ignore, as in the EDA
    out.update(evaluate_image(mb_w, gt))
    out.update(diagnose_image(mb_w, gt))
    return out


def main():
    ov = pd.read_csv(REPO / "data/processed/eda/overlap.csv")
    ov = ov[ov.scale.between(0.95, 1.05)]
    scan = pd.read_csv(REPO / "results/day2/registration_scan_all710.csv")
    keep = set(zip(scan[scan.ncc >= NCC_MIN].a, scan[scan.ncc >= NCC_MIN].b, strict=True))
    ov = ov[(ov.inliers >= 20) | ov.apply(lambda r: (r.a, r.b) in keep, axis=1)]
    rows = ov.to_dict("records")
    with Pool(4) as p:
        res = p.map(one, rows)
    df = pd.DataFrame(res).sort_values("ncc", ascending=False)
    out = REPO / "results" / "day2"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ceilings_pairs.csv", index=False)
    ok = df[df.ncc >= NCC_MIN]
    cols = ["dice_axon", "dice_myelin", "hd95_axon_nm", "hd95_myelin_nm", "g_mae", "recall", "precision",
            "f1@0.5", "f1@0.75", "pq", "pairing_correct_rate", "violation_rate"]
    summ = {"n_pairs_total": len(df), "n_pairs_used": len(ok), "ncc_min": NCC_MIN,
            "n_used_via_ecc": int((ok.method == "ecc").sum()),
            "n_matched_fibers": int(ok.n_matched.sum()),
            "mean": {c: float(ok[c].mean()) for c in cols}, "median": {c: float(ok[c].median()) for c in cols},
            "min": {c: float(ok[c].min()) for c in cols}, "max": {c: float(ok[c].max()) for c in cols}}
    (out / "ceilings_summary.json").write_text(json.dumps(summ, indent=1))
    pd.set_option("display.width", 250)
    print(df[["a", "b", "ncc_orb", "ncc_ecc", "method", "ncc", "overlap_frac"] +
             [c for c in ["dice_axon", "dice_myelin", "g_mae", "f1@0.5", "n_gt", "n_matched"] if c in df]].to_string())
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
