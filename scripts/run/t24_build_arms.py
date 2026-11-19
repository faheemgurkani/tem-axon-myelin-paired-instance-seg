"""T-24 step 3 (C3, H9; D-35, D-42): selection model, then the R1–R4 training label sets.

1. Simulated expert subset: 101 random training candidates (seed 0) with their true class (T-20's size).
2. t_k: Platt calibration of the raw B1 fiber probability on the subset (real fiber vs not).
3. a(x): L2 logistic regression, stratum vs normal, on the subset's real fibers; features from B1 semantics inside
   each footprint (same pipeline for every candidate): g_eq, d_out_nm, myelin_nm, ecc, solidity, contrast.
4. r̂ from the simulated second draw (results/phase3b/t24_sim_summary.json).
5. q_k = t(1−a)(1−r̂) / (1 − t(1−a)r̂); R4 down-weight level d = round(20 q_k) on the painted footprint.
Arms (training cases; test cases keep full labels and d = 0):
  R1 simulated labels · R2 R1 + painted footprints -> ignore (3) · R3 R2's footprints moved to random positions
  wholly on R1 background (same pixel budget, seed 0) · R4 R1 labels + d map.

Outputs: data/processed/t24/arms/R{1..4}/<case>.png, data/processed/t24/arms/R4_d/<case>.png,
data/processed/t24/candidates_scored.csv; committed: results/phase3b/t24_selection_summary.json.

    .venv/bin/python scripts/run/t24_build_arms.py
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
T24 = REPO / "data/processed/t24"
RES = REPO / "results/phase3b"
FEATS = ["g_eq", "d_out_nm", "myelin_nm", "ecc", "solidity", "contrast"]
N_EXPERT, SEED, LEVELS = 101, 0, 20


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def q_of(t, a, r):
    return t * (1 - a) * (1 - r) / (1 - t * (1 - a) * r)


def relocate(painted: np.ndarray, free: np.ndarray, rng) -> np.ndarray:
    """Move each painted footprint to a random position lying wholly on ``free`` pixels, without overlaps."""
    out = np.zeros_like(painted)
    occupied = ~free.copy()
    H, W = painted.shape
    for p_id, sl in enumerate(ndi.find_objects(painted), start=1):
        if sl is None:
            continue
        shape = painted[sl] == p_id
        h, w = shape.shape
        for _ in range(200):
            y, x = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
            win = occupied[y:y + h, x:x + w]
            if not (win & shape).any():
                out[y:y + h, x:x + w][shape] = p_id
                win |= shape
                break
    return out


def main() -> None:
    C = pd.read_csv(T24 / "candidates.csv")
    sim = json.loads((RES / "t24_sim_summary.json").read_text())
    r_hat = sim["second_draw"]["r_hat"]
    rng = np.random.default_rng(SEED)
    C["real"] = C.truth != "not_fiber"
    C["expert"] = False
    C.loc[rng.choice(C.index, N_EXPERT, replace=False), "expert"] = True
    E = C[C.expert]

    # t_k calibration (Platt on logit t)
    platt = LogisticRegression(C=1e6).fit(logit(E.t_raw.to_numpy())[:, None], E.real)
    C["t"] = platt.predict_proba(logit(C.t_raw.to_numpy())[:, None])[:, 1]

    # a(x): stratum vs normal on the subset's real fibers
    X = C[FEATS].copy()
    med = X[C.expert].median()
    X = X.fillna(med)
    mu, sd = X[C.expert].mean(), X[C.expert].std().replace(0, 1)
    Z = (X - mu) / sd
    Er = C.expert & C.real
    amodel = LogisticRegression(C=1.0).fit(Z[Er], C.stratum_true[Er])
    C["a"] = amodel.predict_proba(Z)[:, 1]
    C["q"] = q_of(C.t, C.a, r_hat)
    C["d"] = np.rint(LEVELS * C.q).astype(int)
    C.to_csv(T24 / "candidates_scored.csv", index=False)

    real = C[C.real & (C.truth != "labelled")]
    S = {"n_candidates": len(C), "by_truth": C.truth.value_counts().to_dict(),
         "by_source": C.source.value_counts().to_dict(), "expert_subset": {
             "n": N_EXPERT, "real": int(E.real.sum()), "stratum": int(E.stratum_true.sum())},
         "r_hat": r_hat,
         "auc_t_real_vs_not": {"raw": float(roc_auc_score(C.real, C.t_raw)), "calibrated": float(roc_auc_score(C.real, C.t))},
         "auc_a_stratum_vs_normal_real": float(roc_auc_score(real.truth == "deleted_stratum", real.a)),
         "a_coefficients": dict(zip(FEATS, map(float, amodel.coef_[0]), strict=True)),
         "a_model": {"feats": FEATS, "median": med.to_dict(), "mean": mu.to_dict(), "sd": sd.to_dict(),
                     "coef": amodel.coef_[0].tolist(), "intercept": float(amodel.intercept_[0])},
         "mean_q_by_truth": C.groupby("truth").q.mean().to_dict(),
         "r_sweep_mean_q_by_truth": {f"{r:.2f}": C.assign(q=q_of(C.t, C.a, r)).groupby("truth").q.mean().to_dict()
                                     for r in (0.88, 0.90, 0.92, 0.94, 0.96)}}

    fib = pd.read_csv(T24 / "fibers.csv")
    deleted = fib[fib.deleted & ~fib.test]
    found = set(zip(C.case, C.gt_k))
    S["candidate_recall_on_deleted"] = {
        "random": float(np.mean([(c, k) in found for c, k in zip(deleted[deleted.deleted_random].case,
                                                                 deleted[deleted.deleted_random].k, strict=True)])),
        "stratum": float(np.mean([(c, k) in found for c, k in zip(deleted[deleted.deleted_stratum].case,
                                                                  deleted[deleted.deleted_stratum].k, strict=True)]))}

    cases = pd.read_csv(REPO / "results/phase2/tem1ref_cases.csv")
    for a in ("R1", "R2", "R3", "R4", "R4_d"):
        (T24 / "arms" / a).mkdir(parents=True, exist_ok=True)
    dmap = {(c, k): d for c, k, d in zip(C.case, C.cid, C.d, strict=True)}
    budget = {"R2": 0, "R3": 0}
    for r in cases.itertuples():
        lab = iio.imread(T24 / "labels_R1" / f"{r.case}.png")
        d = np.zeros(lab.shape, np.uint8)
        r2, r3 = lab.copy(), lab.copy()
        if not r.test:
            painted = iio.imread(T24 / "cands" / f"{r.case}.png").astype(np.int32)
            on = (painted > 0) & (lab == 0)
            r2[on] = 3
            moved = relocate(np.where(on, painted, 0), lab == 0, rng)
            r3[moved > 0] = 3
            lut = np.zeros(painted.max() + 1, np.uint8)
            for k in range(1, lut.size):
                lut[k] = dmap.get((r.case, k), 0)
            d[on] = lut[painted[on]]
            budget["R2"] += int(on.sum())
            budget["R3"] += int((moved > 0).sum())
        iio.imwrite(T24 / "arms/R1" / f"{r.case}.png", lab)
        iio.imwrite(T24 / "arms/R2" / f"{r.case}.png", r2)
        iio.imwrite(T24 / "arms/R3" / f"{r.case}.png", r3)
        iio.imwrite(T24 / "arms/R4" / f"{r.case}.png", lab)
        iio.imwrite(T24 / "arms/R4_d" / f"{r.case}.png", d)
    S["ignored_pixels"] = budget
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "t24_selection_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
