"""T-30 (G1, H11): figure for the second model family. (a) Error composition of B1 and CP-FT on the 53 dev images;
(b) thin-myelin vs normal recall on the 8 held-out TEM1 mice for B1, CP-FT and the complete-label reference.
Reads committed results only. Output: docs/paper/figures/h11.pdf (git-ignored folder).

    .venv/bin/python scripts/run/paper_fig_h11.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.frameon": False, "savefig.bbox": "tight"})
FAM = {"selection (strict)": ["pred_unlabelled_real"], "selection (other bg)": ["pred_other_bg"],
       "separation": ["pred_gap", "pred_merge", "pred_split_piece", "gt_merged", "gt_split"],
       "boundary (low IoU)": ["pred_low_iou", "gt_low_iou"], "GT fiber missed": ["gt_missed"]}
COL = ["#D55E00", "#F0B78F", "#0072B2", "#999999", "#000000"]


def main() -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"wspace": 0.45, "width_ratios": [1.15, 1]})
    for i, (m, lab) in enumerate((("b1", "B1 (nnU-Net)"), ("cellpose_ft", "CP-FT (Cellpose-SAM)"))):
        d = pd.read_csv(REPO / f"results/day4/scores_{m}.csv")
        left = 0
        for (fam, cols), c in zip(FAM.items(), COL, strict=True):
            v = int(d[cols].sum().sum())
            ax.barh(i, v, left=left, color=c, label=fam if i == 0 else None, height=0.55)
            if v >= 18:
                ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=6.5,
                        color="w" if c in ("#000000", "#D55E00", "#0072B2") else "k")
            left += v
    ax.set_yticks([0, 1], ["B1 (nnU-Net)", "CP-FT (Cellpose-SAM)"])
    ax.set_xlabel("fiber errors on the 53 dev images")
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.3))
    ax.set_title("(a) same labels, different error profile", fontsize=8)
    h8 = json.loads((REPO / "results/phase2/h8_compare.json").read_text())
    h11 = json.loads((REPO / "results/paper/h11.json").read_text())["H11b"]
    models = [("B1", h8["b1"]["recall_normal"], h8["b1"]["recall_thin"], "#D55E00"),
              ("CP-FT", h11["recall_normal_cp"], h11["recall_thin_cp"], "#CC79A7"),
              ("ref.", h8["ref"]["recall_normal"], h8["ref"]["recall_thin"], "#009E73")]
    for j, (lab, rn, rt, c) in enumerate(models):
        bx.bar([j - 0.18], [rn], 0.34, color=c, alpha=0.45)
        bx.bar([j + 0.18], [rt], 0.34, color=c)
        bx.text(j, max(rn, rt) + 0.02, f"gap {rt - rn:+.2f}", ha="center", fontsize=6.5)
    bx.set_xticks(range(3), [m[0] for m in models])
    bx.set_ylim(0, 1.1)
    bx.set_ylabel("recall, held-out TEM1 mice")
    bx.set_title("(b) normal (light) vs thin myelin (dark)", fontsize=8)
    out = REPO / "docs/paper/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "h11.pdf")
    fig.savefig(out / "h11.png", dpi=130)


if __name__ == "__main__":
    main()
