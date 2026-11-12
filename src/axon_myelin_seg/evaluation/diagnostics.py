"""Instance and pairing diagnostics beyond the official metrics (T-05; contribution C1).

* **Fiber F1 / PQ:** F1@0.5 and F1@0.75 over paired fibers, and PQ = SQ × RQ (Kirillov 2019). Predicted fibers
  with ≥ 50% of their area on GT label 1 are *ignored*: they are neither TP nor FP, because the annotators
  deliberately left those fibers unscored.
* **Merge / split** (counted on all fibers, not just unmatched ones):
  - merge = a predicted fiber that covers ≥ 20% of the area of each of ≥ 2 GT fibers;
  - split = a GT fiber of which ≥ 2 predicted fibers each have ≥ 20% of *their own* area inside it.
* **Pairing correctness** for each matched fiber requires all three of:
  1. the predicted axon is one 8-connected component;
  2. it is enclosed by its own ring (no axon pixel touches background or another fiber);
  3. it has IoU ≥ 0.5 with the GT axon.

  Fiber IoU alone can be high while the axon is wrong (e.g. axon and ring swapped in part), and the g-ratio
  depends on exactly this.
* **Pairing violations** are checked on *every* predicted object, matched or not:
  - an axon without its ring;
  - a ring without its axon;
  - an axon touching background;
  - a ring whose Euler number ≠ 0 (not exactly one hole).

  These checks run on the raw prediction, not on the ignore-masked one. The same checks on GT give the
  "GT violation rate". For example, 2.2% of GT axons touch background
  (docs/notes/10-eda.md), so this is the floor a method should be compared against.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number
from skimage.measure import label as cc_label

from .official import apply_ignore, fiber_table, match

B8 = np.ones((3, 3), bool)


def _crop(labels: np.ndarray, sl: tuple, pad: int = 2) -> tuple:
    H, W = labels.shape
    return (slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)),
            slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)))


def fiber_checks(labels: np.ndarray) -> dict[int, dict]:
    """Per object (keyed by fiber index X // 2): pairing checks on one label map."""
    objs = ndi.find_objects(labels)
    out = {}
    for x in range(2, len(objs) + 1):
        if objs[x - 1] is None:
            continue
        k = x // 2
        if k in out:
            continue
        a, m = 2 * k, 2 * k + 1
        has_a = a <= len(objs) and objs[a - 1] is not None
        has_m = m <= len(objs) and objs[m - 1] is not None
        rec = {"axon_without_ring": has_a and not has_m, "ring_without_axon": has_m and not has_a,
                   "axon_touches_bg": False, "axon_touches_other": False, "ring_euler_bad": False, "axon_n_cc": 0}
        if has_a:
            sub = labels[_crop(labels, objs[a - 1])]
            A = sub == a
            nb = sub[ndi.binary_dilation(A, B8) & ~A]
            rec["axon_touches_bg"] = bool((nb == 0).any())
            rec["axon_touches_other"] = bool(((nb >= 2) & (nb != a) & (nb != m)).any())
            rec["axon_n_cc"] = int(cc_label(A, connectivity=2).max())
        if has_m:
            sub = labels[_crop(labels, objs[m - 1])]
            rec["ring_euler_bad"] = int(euler_number(sub == m, connectivity=2)) != 0
        rec["violation"] = bool(rec["axon_without_ring"] or rec["ring_without_axon"] or rec["axon_touches_bg"]
                                or rec["ring_euler_bad"])
        rec["enclosed"] = has_a and has_m and not rec["axon_touches_bg"] and not rec["axon_touches_other"]
        out[k] = rec
    return out


def axon_iou(pred: np.ndarray, gt: np.ndarray, kp: int, kg: int) -> float:
    a, b = pred == 2 * kp, gt == 2 * kg
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def diagnose_image(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Diagnostics for one image (raw challenge-encoded maps; D-05 applied inside)."""
    t = fiber_table(pred, gt)
    p, g = apply_ignore(pred, gt)
    out = {}
    for thr in (0.5, 0.75):
        m = match(t, thr)
        tp, fp, fn = len(m), len(t.pred_ids) - len(m), len(t.gt_ids) - len(m)
        out[f"tp@{thr}"], out[f"fp@{thr}"], out[f"fn@{thr}"] = tp, fp, fn
        out[f"f1@{thr}"] = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else float("nan")
        if thr == 0.5:
            sq = float(np.mean([i for *_, i in m])) if m else 0.0
            rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else float("nan")
            out.update(pq=sq * rq, sq=sq, rq=rq)
            matches = m

    gt_area = t.gt_area
    pred_area = t.pred_area
    cover_gt: dict[int, list] = {}
    cover_pred: dict[int, list] = {}
    for (a, b), n in t.inter.items():
        cover_gt.setdefault(b, []).append(n / gt_area[a])     # for merges: share of each GT covered by pred b
        cover_pred.setdefault(a, []).append(n / pred_area[b])  # for splits: share of pred b's area inside GT a
    out["n_merges"] = sum(sum(s >= 0.2 for s in v) >= 2 for v in cover_gt.values())
    out["n_splits"] = sum(sum(s >= 0.2 for s in v) >= 2 for v in cover_pred.values())

    # Pairing checks describe the prediction itself, so they use the *raw* prediction. On the ignore-masked map,
    # every predicted axon next to a GT label-1 region would wrongly "touch background".
    pc = fiber_checks(pred)
    ok = []
    for kg, kp, _ in matches:
        r = pc.get(kp, {})
        ok.append(bool(r.get("axon_n_cc", 0) == 1 and r.get("enclosed", False) and axon_iou(p, g, kp, kg) >= 0.5))
    out["n_pairing_correct"] = int(sum(ok))
    out["pairing_correct_rate"] = float(np.mean(ok)) if ok else float("nan")

    kept = [r for k, r in pc.items() if k not in t.pred_ignored]
    out["n_pred_objects"] = len(kept)
    for key in ("axon_without_ring", "ring_without_axon", "axon_touches_bg", "ring_euler_bad", "violation"):
        out[f"n_{key}"] = int(sum(r[key] for r in kept))
    out["violation_rate"] = out["n_violation"] / len(kept) if kept else float("nan")
    return out
