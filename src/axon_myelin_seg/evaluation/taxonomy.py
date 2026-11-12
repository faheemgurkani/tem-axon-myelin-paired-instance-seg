"""FP/FN taxonomy (T-06): why each predicted fiber is unmatched, and why each GT fiber is missed.

This is what separates **selection** errors from **separation** errors.
- Selection: the model found a real fiber that annotators chose not to label.
- Separation: the model grouped pixels into the wrong fibers.

Precedence is fixed and checked in this order. The first rule that fires wins.

Unmatched **predicted** fibers (not counting the ignored ones on GT label 1):

1. ``gap`` (interstitial gap). Two conditions:
   - ≥ 50% of the predicted axon lies on GT background;
   - in the 5-px band just outside it, ≥ 75% of labelled pixels are GT myelin, from ≥ 2 different GT fibers.

   This is the light triangle between touching sheaths that looks like "an axon inside a ring", although the ring
   belongs to the neighbours (docs/notes/11-deep-eda.md §B).
2. ``unlabelled_real``. Two conditions:
   - ≥ 50% of the predicted axon lies on GT background;
   - it overlaps a *ring candidate* by ≥ 50% of the smaller of the two areas.

   A ring candidate is an unlabelled dark closed band around a hole, d ≥ 300 nm. The detector is the same as
   ``scripts/eda/unl_loc.py``.

   These are real fibers the annotators skipped. ~4–8% of real fibers are unlabelled.
3. ``merge``: the predicted fiber covers ≥ 20% of each of ≥ 2 GT fibers.
4. ``split_piece``: ≥ 20% of its area lies inside a GT fiber that another prediction matched.
5. ``low_iou``: ≥ 20% of its area lies inside a GT fiber that nothing matched. This is a delineation error.
6. ``other_bg``: mostly on GT background, not a gap, no ring candidate. Either a real fiber the candidate
   detector missed, or a true FP: an organelle, a dark blob, or an abnormal myelin figure.
7. ``other``: anything else.

Predicted fibers with ≥ 50% of their area on GT label 1 are ``on_label1``. They are reported but never counted as
FP: annotators deliberately excluded those fibers.

Unmatched **GT** fibers:
1. ``missed``: predicted fibers cover < 10% of it.
2. ``merged``: a prediction covers ≥ 20% of it and ≥ 20% of another GT fiber.
3. ``split``: ≥ 2 predictions each have ≥ 20% of their own area inside it.
4. ``low_iou``: anything else (the best overlap is below the match threshold).
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from .official import apply_ignore, fiber_table, match, paired_fiber_map

PRED_CLASSES = ("gap", "unlabelled_real", "merge", "split_piece", "low_iou", "other_bg", "other")
GT_CLASSES = ("missed", "merged", "split", "low_iou")
SEPARATION_PRED = ("gap", "merge", "split_piece")
SEPARATION_GT = ("merged", "split")
SELECTION_PRED = ("unlabelled_real",)


def ring_candidates(image: np.ndarray, gt: np.ndarray, ds: int = 4, min_d_nm: float = 300.0) -> np.ndarray:
    """Label map (full resolution) of unlabelled ring-candidate *holes*. Same algorithm as scripts/eda/unl_loc.py.

    - Smooth the image.
    - Threshold half-way between the GT myelin and GT background medians.
    - Open the dark mask, then take the holes enclosed by dark bands.
    - Keep holes that are ≥ 300 nm in equivalent diameter, solid (solidity ≥ 0.8), and not mostly labelled.
    """
    img = image[::ds, ::ds].astype(np.float32)
    m = gt[::ds, ::ds]
    sm = ndi.gaussian_filter(img, 1.5)
    my, bg = (m >= 3) & (m % 2 == 1), m == 0
    if not my.any() or not bg.any():
        return np.zeros(gt.shape, np.int32)
    thr = (np.median(sm[my]) + np.median(sm[bg])) / 2
    dark = ndi.binary_opening(sm < thr, iterations=2)
    holes = ndi.binary_fill_holes(dark) & ~dark
    L = label(holes)
    keep = np.zeros(L.max() + 1, np.int32)
    k = 0
    for p in regionprops(L):
        d_nm = 2 * np.sqrt(p.area * ds * ds / np.pi) * 2.152
        if d_nm < min_d_nm or p.area * ds * ds > 1.2e6 or p.solidity < 0.8:
            continue
        vals = m[L == p.label]
        if (vals >= 2).mean() > 0.5 or (vals == 1).mean() > 0.5:
            continue
        k += 1
        keep[p.label] = k
    small = keep[L]
    full = np.kron(small, np.ones((ds, ds), np.int32))[: gt.shape[0], : gt.shape[1]]
    return full


def classify_errors(pred: np.ndarray, gt: np.ndarray, candidates: np.ndarray | None = None,
                    iou_threshold: float = 0.5) -> dict:
    """Return per-fiber classes for unmatched predicted and GT fibers, plus counts."""
    t = fiber_table(pred, gt)
    m = match(t, iou_threshold)
    matched_gt = {a for a, _, _ in m}
    matched_pred = {b for _, b, _ in m}
    gt_matched_to = {a: b for a, b, _ in m}
    p, g = apply_ignore(pred, gt)
    fg_map = paired_fiber_map(g)
    g_myelin_fiber = np.where((g >= 3) & (g % 2 == 1), fg_map, 0)

    by_pred: dict[int, list] = {}
    by_gt: dict[int, list] = {}
    for (a, b), n in t.inter.items():
        by_pred.setdefault(b, []).append((a, n))
        by_gt.setdefault(a, []).append((b, n))

    pred_cls = {}
    objs = ndi.find_objects(p)
    for kp in t.pred_ids:
        kp = int(kp)
        if kp in matched_pred:
            continue
        ov = by_pred.get(kp, [])
        area = t.pred_area[kp]
        ax_id = 2 * kp
        cls = None
        if ax_id <= len(objs) and objs[ax_id - 1] is not None:
            sl = objs[ax_id - 1]
            H, W = p.shape
            sl = (slice(max(sl[0].start - 6, 0), min(sl[0].stop + 6, H)),
                  slice(max(sl[1].start - 6, 0), min(sl[1].stop + 6, W)))
            A = p[sl] == ax_id
            gsub = g[sl]
            on_bg = float((gsub[A] == 0).mean()) if A.any() else 0.0
            if on_bg >= 0.5:
                band = ndi.binary_dilation(A, iterations=5) & ~A
                gb = gsub[band]
                lab = gb[gb >= 2]
                my_fibers = set((g_myelin_fiber[sl][band][g_myelin_fiber[sl][band] > 0]).tolist())
                if lab.size and ((lab % 2 == 1).mean() >= 0.75) and len(my_fibers) >= 2 \
                        and (gb >= 2).mean() >= 0.5:
                    cls = "gap"
                elif candidates is not None:
                    cs = candidates[sl][A]
                    cs = cs[cs > 0]
                    if cs.size:
                        cid = np.bincount(cs).argmax()
                        c_area = int((candidates[sl] == cid).sum())
                        if (cs == cid).sum() >= 0.5 * min(c_area, int(A.sum())):
                            cls = "unlabelled_real"
        if cls is None:
            big = [a for a, n in ov if n / t.gt_area[a] >= 0.2]
            if len(big) >= 2:
                cls = "merge"
            elif any(n / area >= 0.2 and a in matched_gt and gt_matched_to[a] != kp for a, n in ov):
                cls = "split_piece"
            elif any(n / area >= 0.2 for a, n in ov):
                cls = "low_iou"
            elif sum(n for _, n in ov) < 0.2 * area:
                cls = "other_bg"
            else:
                cls = "other"
        pred_cls[kp] = cls

    gt_cls = {}
    for kg in t.gt_ids:
        kg = int(kg)
        if kg in matched_gt:
            continue
        ov = by_gt.get(kg, [])
        if sum(n for _, n in ov) < 0.1 * t.gt_area[kg]:
            gt_cls[kg] = "missed"
        elif any(n / t.gt_area[kg] >= 0.2 and
                 sum(n2 / t.gt_area[a2] >= 0.2 for a2, n2 in by_pred.get(b, [])) >= 2 for b, n in ov):
            gt_cls[kg] = "merged"
        elif sum(n / t.pred_area[b] >= 0.2 for b, n in ov) >= 2:
            gt_cls[kg] = "split"
        else:
            gt_cls[kg] = "low_iou"

    counts = {f"pred_{c}": sum(v == c for v in pred_cls.values()) for c in PRED_CLASSES}
    counts.update({f"gt_{c}": sum(v == c for v in gt_cls.values()) for c in GT_CLASSES})
    counts["pred_on_label1"] = len(t.pred_ignored)
    return {"pred": pred_cls, "gt": gt_cls, "counts": counts, "matches": m, "table": t}
