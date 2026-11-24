"""T-28 G4 (reviewer audit, 25 Nov): confidence intervals the C1 draft lacked. CPU only, from committed or local tables.

1. Selection and separation shares of B1's errors with cluster-bootstrap 95% CIs:
   - our data, 3-fold out-of-fold, all 266 images (`results/phase3a/h5_images.csv`, cond = base; overlap groups);
   - zero-shot outside sets (`results/phase2/h8_images_b1.csv`): TEM1 by mouse, TEM2 / TEM2 test / AxonPath by image.
2. Controlled-omission benchmark: normal recall and violation per arm, and paired differences against R1, with a
   mouse-bootstrap 95% CI (`data/processed/t24/score_fibers.csv`, written by t24_score.py; 8 held-out mice).
3. The visual-check share of real fibers among 16 'other background' predictions: Wilson 95% interval.

Output (committed): results/paper/cis.json.

    .venv/bin/python scripts/run/paper_cis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/paper"
B = 1000
SEP = ["pred_gap", "pred_merge", "pred_split_piece", "gt_merged", "gt_split"]


def shares(d: pd.DataFrame) -> dict[str, float]:
    err = d.fp.sum() + d.fn.sum() if "fp" in d else d["fp@0.5"].sum() + d["fn@0.5"].sum()
    strict = d.pred_unlabelled_real.sum()
    return {"errors": int(err), "strict": strict / err, "broad": (strict + d.pred_other_bg.sum()) / err,
            "separation": d[SEP].sum().sum() / err}


def boot_shares(d: pd.DataFrame, cluster: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    groups = d[cluster].unique()
    idx = {g: d.index[d[cluster] == g] for g in groups}
    pt = shares(d)
    draws = []
    for _ in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        draws.append(shares(d.loc[np.concatenate([idx[g] for g in pick])]))
    out = {"errors": pt["errors"], "n_clusters": len(groups), "cluster": cluster}
    for k in ("strict", "broad", "separation"):
        v = np.array([x[k] for x in draws])
        out[k] = [float(pt[k]), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [p, float(c - h), float(c + h)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    S: dict = {"bootstrap": f"{B} cluster resamples, 95% percentile", "selection_shares": {}}

    ours = pd.read_csv(REPO / "results/phase3a/h5_images.csv")
    ours = ours[ours.cond == "base"].reset_index(drop=True)
    ours["group"] = ours.group.fillna(ours.name)
    assert len(ours) == 266, len(ours)
    S["selection_shares"]["ours_cv3_266"] = boot_shares(ours, "group")

    ext = pd.read_csv(REPO / "results/phase2/h8_images_b1.csv")
    for s, d in ext.groupby("set"):
        d = d.reset_index(drop=True)
        S["selection_shares"][s] = boot_shares(d, "subject" if s == "tem1" else "case")

    # benchmark arms
    F = pd.read_csv(REPO / "data/processed/t24/score_fibers.csv")
    arms = list(F.arm.unique())
    per = F.groupby(["arm", "subject", "stratum"]).matched.agg(["sum", "count"]).reset_index()
    mice = sorted(F.subject.unique())

    def stats(sel_mice):
        r = {}
        for a in arms:
            g = per[(per.arm == a) & per.subject.isin(sel_mice)]
            # weight by resampled multiplicity
            m = pd.Series(sel_mice).value_counts()
            g = g.assign(w=g.subject.map(m))
            n = g[~g.stratum]
            st = g[g.stratum]
            r[a] = ((n["sum"] * n.w).sum() / (n["count"] * n.w).sum(), (st["sum"] * st.w).sum() / (st["count"] * st.w).sum())
        return r

    pt = stats(mice)
    rng = np.random.default_rng(0)
    draws = [stats(list(rng.choice(mice, len(mice), replace=True))) for _ in range(B)]
    T: dict = {"n_mice": len(mice), "arms": {}}
    for a in arms:
        rec = np.array([d[a][0] for d in draws])
        vio = np.array([d[a][1] for d in draws])
        e = {"normal_recall": [pt[a][0], *np.percentile(rec, [2.5, 97.5]).tolist()],
             "violation": [pt[a][1], *np.percentile(vio, [2.5, 97.5]).tolist()]}
        if a != "R1":
            dr = np.array([d[a][0] - d["R1"][0] for d in draws])
            dv = np.array([d[a][1] - d["R1"][1] for d in draws])
            e["recall_minus_R1"] = [pt[a][0] - pt["R1"][0], *np.percentile(dr, [2.5, 97.5]).tolist()]
            e["violation_minus_R1"] = [pt[a][1] - pt["R1"][1], *np.percentile(dv, [2.5, 97.5]).tolist()]
        T["arms"][a] = e
    S["t24"] = T
    S["visual_check_other_bg_real"] = {"k": 7, "n": 16, "wilson95": wilson(7, 16)}
    (OUT / "cis.json").write_text(json.dumps(S, indent=1, default=float))
    for k, v in S["selection_shares"].items():
        print(f"{k:14s} n_err={v['errors']:5d} clusters={v['n_clusters']:3d} broad={v['broad'][0]:.3f} "
              f"[{v['broad'][1]:.3f},{v['broad'][2]:.3f}] strict={v['strict'][0]:.3f} [{v['strict'][1]:.3f},"
              f"{v['strict'][2]:.3f}] sep={v['separation'][0]:.3f} [{v['separation'][1]:.3f},{v['separation'][2]:.3f}]")
    for a, e in T["arms"].items():
        print(a, {k: [round(x, 3) for x in v] for k, v in e.items()})
    print("visual check", S["visual_check_other_bg_real"])


if __name__ == "__main__":
    main()
