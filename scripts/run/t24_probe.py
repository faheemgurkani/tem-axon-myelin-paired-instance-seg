"""T-24 post-hoc audit (24 Nov; exploratory, after H9 failed): where the selection-gated weights lost information.

Per training case (2.152 nm grid):
- coverage: share of each deleted GT fiber's pixels inside any painted candidate footprint (what R2/R4 can act on);
- label check: deleted fibers' pixels are background in labels_R1, kept fibers' pixels are not;
- every B1 zero-shot instance (D-11 pairing) with the same-pipeline features as the candidates (`footprint_features`)
  and its status: on a labelled fiber (>= 50% of its area; the instances t24_candidates.py excluded) or a candidate,
  plus its truth (best-IoU GT fiber >= 0.5: normal / stratum / none).
The labelled-fiber instances are what a PU (positive-unlabelled) selection model can learn "normal" from, without
the expert: random deletion makes missed normals a random sample of normals (D-44).

Outputs (git-ignored): data/processed/t24/probe_coverage.csv, data/processed/t24/b1_instances.csv.

    .venv/bin/python scripts/run/t24_probe.py
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from t24_candidates import footprint_features

from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances

REPO = Path(__file__).resolve().parents[2]
T24 = REPO / "data/processed/t24"
B1SEM = REPO / "data/predictions/b1_ext/tem1"
IMG = REPO / "data/nnunet/phase2_inputs/tem1"


def one(case: str) -> tuple[list[dict], list[dict], dict]:
    F = pd.read_csv(T24 / "fibers.csv")
    F = F[F.case == case].set_index("k")
    lab = iio.imread(T24 / "labels_R1" / f"{case}.png")
    gtp = iio.imread(T24 / "gt_pairs" / f"{case}.png").astype(np.int64)
    painted = iio.imread(T24 / "cands" / f"{case}.png")
    sem = iio.imread(B1SEM / f"{case}.png")
    img = iio.imread(IMG / f"{case}_0000.png")
    img = img if img.ndim == 2 else img[..., 0]
    fid = np.where(gtp >= 2, gtp // 2, 0)
    labelled = (lab == 1) | (lab == 2)

    # coverage and label check per GT fiber
    cov, chk = [], {"deleted_px_not_bg": 0, "kept_px_bg": 0}
    for p in regionprops(fid):
        k = p.label
        if k not in F.index:
            continue
        m = fid[p.slice] == k
        f = F.loc[k]
        if f.deleted:
            chk["deleted_px_not_bg"] += int((lab[p.slice][m] != 0).sum())
            cov.append({"case": case, "k": k, "kind": "stratum" if f.deleted_stratum else "random",
                        "area": int(m.sum()), "covered": float((painted[p.slice][m] > 0).mean()), "g": f.g,
                        "ecc": f.ecc})
        else:
            chk["kept_px_bg"] += int((lab[p.slice][m] == 0).sum())

    # all B1 instances with features, status and truth
    P = semantic_to_paired_instances(sem == 1, sem == 2).astype(np.int64)
    pf = np.where(P >= 2, P // 2, 0)
    gt_area = np.bincount(fid.ravel())
    rows = []
    for p in regionprops(pf):
        m = pf[p.slice] == p.label
        on_lab = float(labelled[p.slice][m].mean())
        ov = np.bincount(fid[p.slice][m].ravel(), minlength=gt_area.size)
        ov[0] = 0
        kg = int(ov.argmax())
        iou = ov[kg] / (p.area + gt_area[kg] - ov[kg]) if kg else 0.0
        if iou >= 0.5 and kg in F.index:
            f = F.loc[kg]
            truth = "stratum" if f.stratum else "normal"
            state = "deleted" if f.deleted else "kept"
        else:
            truth, state = "none", "none"
        rows.append({"case": case, "subject": F.subject.iloc[0] if len(F) else "", "iid": p.label,
                     "status": "labelled" if on_lab >= 0.5 else "candidate", "truth": truth, "gt_state": state,
                     "gt_k": kg if iou >= 0.5 else -1,
                     **footprint_features(p, sem[p.slice][m], img[p.slice][m])})
    return cov, rows, chk


def main() -> None:
    cases = sorted(pd.read_csv(T24 / "fibers.csv").query("~test").case.unique())
    with Pool(6) as p:
        res = p.map(one, cases)
    cov = pd.DataFrame([x for c, _, _ in res for x in c])
    inst = pd.DataFrame([x for _, r, _ in res for x in r])
    cov.to_csv(T24 / "probe_coverage.csv", index=False)
    inst.to_csv(T24 / "b1_instances.csv", index=False)
    print("label check:", {k: sum(c[k] for _, _, c in res) for k in res[0][2]})
    print("coverage of deleted fibers by painted footprints (pixel share):")
    print(cov.groupby("kind").covered.describe()[["count", "mean", "25%", "50%"]].round(3))
    print("share with coverage >= 0.8:", cov.groupby("kind").covered.apply(lambda s: round(float((s >= 0.8).mean()), 3)).to_dict())
    print("B1 instances:", inst.groupby(["status", "truth", "gt_state"]).size().to_dict())


if __name__ == "__main__":
    main()
