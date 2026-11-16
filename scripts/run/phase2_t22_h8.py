"""Phase 2, T-22 (H8, D-34): hidden stratification. Recall of thin-myelin vs normal fibers on near-complete outside data.

Predictions (semantic PNGs on the 2.152 nm grid; 0 bg / 1 axon / 2 myelin) are resized back to the native grid
(nearest) and paired with B1's standard D-11 rule (`semantic_to_paired_instances`), the post-processing B1 is defined
with. GT: `astih_to_pairs` (ASTIH protocol, border fibers scored) or `axonpath_geojson_to_pairs` (ROI only).
Per GT fiber: equivalent-circle g, thin = g > 0.85 (D-34), matched at IoU > 0.5 by our evaluator.

Models (--model):
  b1   B1 zero-shot, data/predictions/b1_ext/<set>/<case>.png (Modal phase2_preds/b1/<set>)
  ref  TEM1 complete-label reference (Dataset502, fold 0 = the 8 held-out mice), data/predictions/tem1ref/<case>.png
H8 rule (07-plan §7.5), on the 8 held-out TEM1 mice: gap = recall(thin) − recall(normal);
holds if gap_b1 − gap_ref ≤ −0.10 and the mouse-bootstrap 95% CI is below 0 (two-sided report). `--compare` computes
it once both models are scored. Also reported descriptively: B1's taxonomy (selection shares) per outside set.

Outputs: data/processed/phase2/h8_fibers_<model>.csv (per fiber), results/phase2/h8_<model>.json, h8_compare.json.

    .venv/bin/python scripts/run/phase2_t22_h8.py --model b1 [--sets tem1 tem2_test tem2_dandi axonpath]
    .venv/bin/python scripts/run/phase2_t22_h8.py --compare
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd

from axon_myelin_seg.data.external import astih_to_pairs, axonpath_geojson_to_pairs, split_axonmyelin
from axon_myelin_seg.evaluation.official import fiber_table, match
from axon_myelin_seg.evaluation.taxonomy import classify_errors, ring_candidates
from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "data/nnunet/phase2_inputs"
PROC = REPO / "data/processed/phase2"
OUT = REPO / "results/phase2"
PRED = {"b1": REPO / "data/predictions/b1_ext", "ref": REPO / "data/predictions"}
TEST_MICE = {f"sub-nyuMouse{n}" for n in (13, 14, 31, 32, 33, 34, 26, 27)}
DS = {"tem1": 4, "tem2_test": 2, "tem2_dandi": 2, "axonpath": 4}
spec = importlib.util.spec_from_file_location("audit", REPO / "scripts/run/phase0b_completeness_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def load_gt(row) -> tuple[np.ndarray, np.ndarray]:
    """(image uint8, GT pairs) on the native grid."""
    if row.set == "axonpath":
        img = audit.normalise_to_uint8(iio.imread(audit.AXONPATH / "images" / row.source))
        gt, _ = axonpath_geojson_to_pairs(audit.AXONPATH / "annotations" / f"{Path(row.source).stem}.geojson",
                                          img.shape)
        return img, gt
    root = {"tem1": REPO / "data/external/001436", "tem2_dandi": REPO / "data/external/001350",
            "tem2_test": audit.TEM2_TEST}[row.set]
    d = root / "derivatives/labels" / row.subject / "micr"
    stem = row.source[:-4]
    a, m = d / f"{stem}_seg-axon-manual.png", d / f"{stem}_seg-myelin-manual.png"
    if a.exists() and m.exists():
        ax, my = iio.imread(a), iio.imread(m)
    else:
        ax, my = split_axonmyelin(iio.imread(d / f"{stem}_seg-axonmyelin-manual.png"))
    img = iio.imread(root / row.subject / "micr" / row.source)
    img = img if img.ndim == 2 else img[..., 0]
    gt, _ = astih_to_pairs(ax, my)
    return img, gt


def one(args):
    model, row = args
    path = (PRED[model] / row.set / f"{row.case}.png") if model == "b1" else (PRED[model] / "tem1ref" / f"{row.case}.png")
    if not path.exists():
        return None
    sem = cv2.resize(iio.imread(path), (row.w, row.h), interpolation=cv2.INTER_NEAREST)
    pred = semantic_to_paired_instances(sem == 1, sem == 2)
    img, gt = load_gt(row)
    t = fiber_table(pred, gt)
    m = match(t, 0.5)
    matched = {kg for kg, _, _ in m}
    fibers = [{"set": row.set, "case": row.case, "subject": row.subject, "gt_k": int(k),
               "g": float(np.sqrt(t.gt_axon_area[k] / t.gt_area[k])), "area": int(t.gt_area[k]),
               "matched": int(k) in matched} for k in t.gt_ids if t.gt_axon_area[int(k)] > 0]
    r = classify_errors(pred, gt, ring_candidates(img, gt, ds=DS[row.set], px_nm=row.px_nm))
    img_row = {"set": row.set, "case": row.case, "subject": row.subject, "tp": len(m),
               "fp": len(t.pred_ids) - len(m), "fn": len(t.gt_ids) - len(m), **r["counts"]}
    return fibers, img_row


def recall_gap(f: pd.DataFrame) -> dict:
    thin, norm = f[f.g > 0.85], f[f.g <= 0.85]
    rt, rn = thin.matched.mean(), norm.matched.mean()
    return {"n_thin": len(thin), "n_normal": len(norm), "recall_thin": float(rt), "recall_normal": float(rn),
            "gap": float(rt - rn)}


def boot_by(f: pd.DataFrame, fn, key: str = "subject", n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    units = sorted(f[key].unique())
    groups = {u: f[f[key] == u] for u in units}
    return [fn(pd.concat([groups[u] for u in rng.choice(units, len(units))])) for _ in range(n)]


def score(model: str, sets: list[str]) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in sets:
        man = pd.read_csv(INPUTS / s / "manifest.csv")
        if model == "ref":
            man = man[man.subject.isin(TEST_MICE)]
        rows += [(model, SimpleNamespace(**r)) for r in man.to_dict("records")]  # picklable for Pool
    with Pool(4) as p:
        res = [x for x in p.map(one, rows) if x is not None]
    F = pd.DataFrame([f for fs, _ in res for f in fs])
    I = pd.DataFrame([i for _, i in res])
    F.to_csv(PROC / f"h8_fibers_{model}.csv", index=False)
    I.to_csv(OUT / f"h8_images_{model}.csv", index=False)
    S = {"model": model, "n_images": len(I), "missing_predictions": len(rows) - len(res)}
    for s, d in F.groupby("set"):
        gi = I[I.set == s]
        e = gi.fp.sum() + gi.fn.sum()
        x = {"images": len(gi), "fibers": len(d), "recall_all": float(d.matched.mean()),
             "f1": float(2 * gi.tp.sum() / (2 * gi.tp.sum() + e)) if e or gi.tp.sum() else np.nan,
             "strict_selection_share": float(gi.pred_unlabelled_real.sum() / e) if e else np.nan,
             "broad_selection_share": float((gi.pred_unlabelled_real.sum() + gi.pred_other_bg.sum()) / e) if e else np.nan,
             "errors": int(e), **recall_gap(d)}
        if s == "tem1":
            ho = d[d.subject.isin(TEST_MICE)]
            x["held_out_mice"] = recall_gap(ho)
            b = boot_by(ho, lambda z: recall_gap(z)["gap"])
            x["held_out_mice"]["gap_ci95"] = [float(np.percentile(b, q)) for q in (2.5, 97.5)]
        S[s] = x
    (OUT / f"h8_{model}.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


def compare() -> None:
    b = pd.read_csv(PROC / "h8_fibers_b1.csv")
    r = pd.read_csv(PROC / "h8_fibers_ref.csv")
    b = b[(b.set == "tem1") & b.subject.isin(TEST_MICE)]
    key = ["case", "gt_k"]
    m = b.merge(r[key + ["matched"]].rename(columns={"matched": "matched_ref"}), on=key)
    assert len(m) == len(b) == len(r), (len(m), len(b), len(r))

    def diff(z: pd.DataFrame) -> float:
        gb = recall_gap(z)["gap"]
        gr = recall_gap(z.assign(matched=z.matched_ref))["gap"]
        return gb - gr

    est = diff(m)
    bs = boot_by(m, diff)
    ci = [float(np.percentile(bs, q)) for q in (2.5, 97.5)]
    S = {"b1": recall_gap(m), "ref": recall_gap(m.assign(matched=m.matched_ref)), "gap_b1_minus_gap_ref": est,
         "ci95_mouse_bootstrap": ci, "n_mice": int(m.subject.nunique()),
         "H8_holds": bool(est <= -0.10 and ci[1] < 0), "rule": "gap_b1 - gap_ref <= -0.10 and CI upper < 0"}
    (OUT / "h8_compare.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(PRED))
    ap.add_argument("--sets", nargs="+", default=["tem1", "tem2_test", "tem2_dandi", "axonpath"])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    else:
        score(a.model, a.sets if a.model == "b1" else ["tem1"])


if __name__ == "__main__":
    main()
