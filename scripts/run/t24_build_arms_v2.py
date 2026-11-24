"""T-24 round 2 (H10; D-44–D-47): corrected selection model and the R6 / R4b / R4h label sets.

Fixes from the 24 Nov audit of H9 (07-plan §6, 24 Nov):
- D-44 PU normals: "normal" appearance is learned from the B1 instances that sit on still-labelled fibers
  (`data/processed/t24/b1_instances.csv`, from t24_probe.py). Random omission makes missed normals a random sample of
  normals, so these 3,278 instances are valid normal examples (they include the ~3% labelled stratum fibers, as real
  data would); "stratum" examples come from the expert subset (the same 101 candidates, seed 0, as round 1).
- D-45 model form: gradient-boosted trees (depth 3) on the round-1 features plus two-sided band terms
  ((g − 0.75)², max(ecc − 0.8, 0)); balanced classes, then a prior shift to π = stratum share among the expert
  subset's real candidates.
- D-46 calibration: q_raw = t·(1 − a) (D-43), recalibrated by one-input Platt on the expert subset. The stratum score of
  each expert candidate is cross-fitted (5 folds) so the calibration never sees its own training example. Chosen over
  no calibration and two-input Platt by expert-subset CV log-loss only (0.375 vs 0.591 / 0.387), never by benchmark truth.
Arms (training cases only; test cases keep full labels):
  R4b  R1 labels + down-weight d = round(20 q) on the painted footprint (trainer `_softw`, as R4).
  R4h  R1 labels + ignore (3) on the footprints of the top-K candidates by q, K = round(Σ q) (expected missed normals).
  R6   oracle: R1 labels + ignore on the footprints of candidates whose truth is a missed normal (uses benchmark truth;
       the ceiling of any selection model over these candidates; not a method).

Outputs: data/processed/t24/arms/{R4b,R4b_d,R4h,R6}/<case>.png, data/processed/t24/candidates_scored_v2.csv;
committed: results/phase3b/t24_selection_v2_summary.json.

    .venv/bin/python scripts/run/t24_build_arms_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
T24 = REPO / "data/processed/t24"
RES = REPO / "results/phase3b"
BASE = ["g_eq", "d_out_nm", "myelin_nm", "ecc", "solidity", "contrast"]
LEVELS = 20


def feats(d: pd.DataFrame) -> pd.DataFrame:
    x = d[BASE].copy()
    x["g_dev2"] = (x.g_eq - 0.75) ** 2
    x["ecc_hi"] = np.clip(x.ecc - 0.8, 0, None)
    return x


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    C = pd.read_csv(T24 / "candidates_scored.csv")  # round 1: expert flag (seed 0), calibrated t, truth, painted_px
    inst = pd.read_csv(T24 / "b1_instances.csv")
    L = inst[inst.status == "labelled"]
    E = C[C.expert]
    Er = E[E.truth.isin(["deleted_random", "deleted_stratum"])]
    pi = float((Er.truth == "deleted_stratum").mean())

    def fit(stratum_rows: pd.DataFrame):
        tr = pd.concat([feats(L).assign(y=0), feats(stratum_rows).assign(y=1)])
        y = tr.pop("y")
        med = tr.median()
        m = HistGradientBoostingClassifier(max_depth=3, class_weight="balanced", random_state=0).fit(tr.fillna(med), y)
        return lambda d: m.predict_proba(feats(d).fillna(med))[:, 1]

    s = pd.Series(fit(Er[Er.truth == "deleted_stratum"])(C), index=C.index)
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(E, E.truth):  # cross-fit the expert rows
        trE = E.iloc[tr_i]
        s[E.index[te_i]] = fit(trE[trE.truth == "deleted_stratum"])(E.iloc[te_i])
    C["a2"] = 1 / (1 + np.exp(-(logit(s) + np.log(pi / (1 - pi)))))
    C["q2_raw"] = C.t * (1 - C.a2)
    yE = (E.truth == "deleted_random").to_numpy()
    cal = LogisticRegression(C=1.0).fit(logit(C.q2_raw[E.index])[:, None], yE)
    C["q2"] = cal.predict_proba(logit(C.q2_raw)[:, None])[:, 1]
    C["d2"] = np.rint(LEVELS * C.q2).astype(int)
    K = round(float(C.q2.sum()))
    C["gate_h"] = C.q2.rank(ascending=False, method="first") <= K
    C.to_csv(T24 / "candidates_scored_v2.csv", index=False)

    nx = ~C.expert
    real = C[C.truth.isin(["deleted_random", "deleted_stratum"]) & nx]
    w = 1 - C.d2 / LEVELS
    px = C.painted_px
    S = {"pi_stratum_among_real_unlabelled": pi, "n_labelled_normal_source": len(L), "K_gate": K,
         "calibration": {"coef": float(cal.coef_[0, 0]), "intercept": float(cal.intercept_[0])},
         "auc_a2_stratum_vs_normal_nonexpert": float(roc_auc_score(real.truth == "deleted_stratum", real.a2)),
         "auc_q2_missed_normal_nonexpert": float(roc_auc_score(C.truth[nx] == "deleted_random", C.q2[nx])),
         "mean_q2_nonexpert": float(C.q2[nx].mean()),
         "actual_missed_normal_rate_nonexpert": float((C.truth[nx] == "deleted_random").mean()),
         "R4b_pixel_weight_by_truth": {k: float(np.average(w[g.index], weights=g.painted_px))
                                       for k, g in C.groupby("truth")},
         "R4h_ignored_pixel_share_by_truth": {k: float(px[C.gate_h & (C.truth == k)].sum() / px[C.truth == k].sum())
                                              for k in C.truth.unique()},
         "R4h_precision": float((C.truth[C.gate_h] == "deleted_random").mean())}

    cases = pd.read_csv(REPO / "results/phase2/tem1ref_cases.csv")
    for a in ("R4b", "R4b_d", "R4h", "R6"):
        (T24 / "arms" / a).mkdir(parents=True, exist_ok=True)
    budget = {"R4h": 0, "R6": 0, "R4b_downweighted": 0}
    for r in cases.itertuples():
        lab = iio.imread(T24 / "labels_R1" / f"{r.case}.png")
        d = np.zeros(lab.shape, np.uint8)
        r4h, r6 = lab.copy(), lab.copy()
        if not r.test:
            painted = iio.imread(T24 / "cands" / f"{r.case}.png").astype(np.int32)
            on = (painted > 0) & (lab == 0)
            cc = C[C.case == r.case].set_index("cid")
            n = painted.max() + 1
            lut_d, lut_h, lut_o = np.zeros(n, np.uint8), np.zeros(n, bool), np.zeros(n, bool)
            for k in cc.index:
                if k < n:
                    lut_d[k], lut_h[k], lut_o[k] = cc.d2[k], cc.gate_h[k], cc.truth[k] == "deleted_random"
            d[on] = lut_d[painted[on]]
            r4h[on & lut_h[painted]] = 3
            r6[on & lut_o[painted]] = 3
            budget["R4h"] += int((on & lut_h[painted]).sum())
            budget["R6"] += int((on & lut_o[painted]).sum())
            budget["R4b_downweighted"] += int((d > 0).sum())
        iio.imwrite(T24 / "arms/R4b" / f"{r.case}.png", lab)
        iio.imwrite(T24 / "arms/R4b_d" / f"{r.case}.png", d)
        iio.imwrite(T24 / "arms/R4h" / f"{r.case}.png", r4h)
        iio.imwrite(T24 / "arms/R6" / f"{r.case}.png", r6)
    S["pixels"] = budget
    (RES / "t24_selection_v2_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
