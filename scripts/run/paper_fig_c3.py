"""T-29 (25-26 Nov): figures and tables for the selection-aware training experiment (C3 / T-24, H9 and H10) in the
C1 paper. CPU only; reads the T-24 working tables (git-ignored) and committed summaries; writes figures into
docs/paper/figures (git-ignored) and a committed metrics file results/paper/c3_extras.json.

Figures:
  t24_selection.pdf     (a) score distributions of the round-1 and round-2 selection models by candidate truth;
                        (b) pixel-weighted training weight per truth class (round 1, round 2 uncalibrated, round 2 R4b);
                        (c) share of each class's footprint pixels ignored by R2, R4h and R6.
  t24_frontier_ci.pdf   recall-violation plane with mouse-bootstrap 95% CIs for every arm.
  t24_floor.pdf         share of stratum test fibers found vs depth inside the stratum (distance past the nearest cut),
                        for R1, R2, R6 and the complete-label reference; plus footprint coverage of deleted fibers.

    PYTHONPATH=src .venv/bin/python scripts/run/paper_fig_c3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
T24 = REPO / "data/processed/t24"
FIG = REPO / "docs/paper/figures"
RES = REPO / "results/paper"
LEVELS = 20
COL = {"deleted_random": "#009E73", "deleted_stratum": "#D55E00", "not_fiber": "#999999"}
LAB = {"deleted_random": "missed normal (215)", "deleted_stratum": "protocol-excluded (834)", "not_fiber": "not a fiber (120)"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.frameon": False, "savefig.bbox": "tight"})


def depth(g: np.ndarray, e: np.ndarray) -> np.ndarray:
    """How far a stratum fiber lies past its nearest cut (largest violated margin)."""
    return np.maximum.reduce([g - 0.85, 0.65 - g, e - 0.9])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    C = pd.read_csv(T24 / "candidates_scored_v2.csv")
    out: dict = {}

    # ---------------- selection diagnostics
    fig, ax = plt.subplots(1, 3, figsize=(8.0, 2.3), gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.5})
    bins = np.linspace(0, 1, 26)
    for k in ("deleted_random", "deleted_stratum", "not_fiber"):
        ax[0].hist(C.q[C.truth == k], bins=bins, histtype="step", color=COL[k], lw=1.0, ls=":", density=True)
        ax[0].hist(C.q2[C.truth == k], bins=bins, histtype="step", color=COL[k], lw=1.4, density=True, label=LAB[k])
    ax[0].set_xlabel("P(missed normal) per candidate")
    ax[0].set_ylabel("density")
    ax[0].set_title("(a) round 1 (dotted) vs round 2 (solid)", fontsize=7.5)
    ax[0].legend(fontsize=5.8, loc="upper right")
    px = C.painted_px
    w1 = 1 - C.d / LEVELS
    w2u = 1 - np.rint(LEVELS * C.q2_raw.clip(0, 1)) / LEVELS
    w2 = 1 - C.d2 / LEVELS
    rows = {}
    for k in ("deleted_random", "deleted_stratum", "not_fiber"):
        m = C.truth == k
        rows[k] = [float(np.average(w[m], weights=px[m])) for w in (w1, w2u, w2)]
    out["pixel_weight_by_truth"] = {"columns": ["round1_R4", "round2_uncalibrated", "round2_R4b"], **rows}
    x = np.arange(3)
    for i, k in enumerate(rows):
        ax[1].bar(x + (i - 1) * 0.26, rows[k], 0.26, color=COL[k])
    ax[1].set_xticks(x, ["R4\nround 1", "round 2\nuncalib.", "R4b\ncalib."], fontsize=6.2)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("mean training weight\n(1 = normal background)")
    ax[1].set_title("(b) weight on candidate pixels", fontsize=7.5)
    ign = {}
    gate = C.gate_h.astype(bool)
    for k in ("deleted_random", "deleted_stratum", "not_fiber"):
        m = C.truth == k
        ign[k] = [1.0, float(px[m & gate].sum() / px[m].sum()), float(k == "deleted_random")]
    out["ignored_pixel_share_by_truth"] = {"columns": ["R2", "R4h", "R6"], **ign}
    for i, k in enumerate(ign):
        ax[2].bar(x + (i - 1) * 0.26, ign[k], 0.26, color=COL[k])
    ax[2].set_xticks(x, ["R2\nall", "R4h\ntop 161", "R6\noracle"], fontsize=6.2)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("share of footprint pixels ignored")
    ax[2].set_title("(c) what each hard-ignore arm removes", fontsize=7.5)
    fig.savefig(FIG / "t24_selection.pdf")
    plt.close(fig)

    # ---------------- frontier with CIs
    cis = json.loads((RES / "cis.json").read_text())["t24"]["arms"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    r1, r2 = cis["R1"], cis["R2"]
    ax.plot([r1["violation"][0], r2["violation"][0]], [r1["normal_recall"][0], r2["normal_recall"][0]], "--",
            color="#555555", lw=0.8)
    sty = {"R1": ("#555555", "o"), "R1b": ("#555555", "o"), "R2": ("#555555", "s"), "R3": ("#0072B2", "^"),
           "R4": ("#D55E00", "v"), "R4b": ("#D55E00", "D"), "R4h": ("#D55E00", "P"), "R6": ("#009E73", "*"),
           "ref": ("k", "x")}
    for a, (c, m) in sty.items():
        v, r = cis[a]["violation"], cis[a]["normal_recall"]
        ax.errorbar(v[0], r[0], xerr=[[v[0] - v[1]], [v[2] - v[0]]], yerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt=m,
                    color=c, mfc="none" if a == "R1b" else c, ms=6 if a == "R6" else 4, lw=0.6, capsize=1.5,
                    label=a if a != "ref" else "complete labels")
    ax.set_xlabel("violation (stratum test fibers found)")
    ax.set_ylabel("normal-fiber recall")
    ax.legend(fontsize=5.6, ncol=3, loc="lower right", handletextpad=0.2, columnspacing=0.6)
    ax.set_ylim(0.93, 1.0)
    fig.savefig(FIG / "t24_frontier_ci.pdf")
    plt.close(fig)

    # ---------------- violation floor and coverage
    F = pd.read_csv(T24 / "fibers.csv")
    S = pd.read_csv(T24 / "score_fibers.csv").merge(F[["case", "k", "g", "ecc"]], on=["case", "k"])
    S = S[S.stratum].copy()
    S["depth"] = depth(S.g.to_numpy(), S.ecc.to_numpy())
    edges = [0, 0.02, 0.05, 0.10, 1.0]
    S["bin"] = pd.cut(S.depth, edges, include_lowest=True)
    floor = S.groupby(["arm", "bin"], observed=True).matched.agg(["mean", "size"]).reset_index()
    out["violation_by_depth"] = {a: {str(b): [float(m), int(n)] for b, m, n in zip(d.bin, d["mean"], d["size"],
                                                                                    strict=True)}
                                 for a, d in floor.groupby("arm")}
    cov = pd.read_csv(T24 / "probe_coverage.csv")
    out["footprint_coverage"] = {k: {"n": int(len(d)), "mean": float(d.covered.mean()),
                                     "share_ge_0.8": float((d.covered >= 0.8).mean())}
                                 for k, d in cov.groupby("kind")}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"wspace": 0.35})
    lab = ["0-0.02", "0.02-0.05", "0.05-0.10", "> 0.10"]
    for a, c in (("R1", "#555555"), ("R2", "#0072B2"), ("R6", "#009E73"), ("ref", "k")):
        d = floor[floor.arm == a]
        ax[0].plot(range(len(d)), d["mean"], "-o", color=c, ms=3, lw=1, label=a if a != "ref" else "complete labels")
    n = floor[floor.arm == "R1"]["size"].tolist()
    ax[0].set_xticks(range(len(lab)), [f"{x}\n(n={k})" for x, k in zip(lab, n, strict=True)], fontsize=6.5)
    ax[0].set_xlabel("depth inside the stratum (distance past the nearest cut)")
    ax[0].set_ylabel("share of stratum fibers found")
    ax[0].set_ylim(0.3, 1.02)
    ax[0].legend(fontsize=6)
    ax[0].set_title("(a) the protocol is learnable only far from its cuts", fontsize=7.5)
    for k, c in (("random", "#009E73"), ("stratum", "#D55E00")):
        ax[1].hist(cov.covered[cov.kind == k], bins=np.linspace(0, 1, 21), histtype="step", lw=1.3, color=c,
                   density=True, label=f"{k} deletion (mean {cov.covered[cov.kind == k].mean():.2f})")
    ax[1].set_xlabel("share of a deleted fiber covered by candidate footprints")
    ax[1].set_ylabel("density")
    ax[1].legend(fontsize=6, loc="upper left")
    ax[1].set_title("(b) candidates cover the deleted fibers", fontsize=7.5)
    fig.savefig(FIG / "t24_floor.pdf")
    plt.close(fig)
    (RES / "c3_extras.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    main()
