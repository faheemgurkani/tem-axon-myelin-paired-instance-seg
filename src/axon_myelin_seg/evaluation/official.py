"""Replica of the official challenge metrics (T-04; docs/notes/02-challenge-and-data.md §1).

The challenge defines Dice (axon, myelin), HD95 in nm, and the g-ratio MAE over fibers matched by IoU on the fiber
mask (X ∪ X+1). It does not publish code, so the ambiguous details are decisions:

* **D-05:** ground-truth label 1 (artifact / border-cut / deliberately excluded fiber) is removed from *both* maps
  before anything is computed. A prediction there neither helps nor hurts.
* **D-06:** a "fiber" is an even ID X ≥ 2 whose X+1 is also present. Unpaired IDs count in Dice/HD95 but cannot
  match. Matching is one-to-one at IoU > 0.5; above 0.5 the match is unique, so greedy and Hungarian are identical.
  g = √(A_axon / (A_axon + A_myelin)), the equivalent-circle diameter ratio.
* **D-07:** HD95 is computed per image and per compartment as the max of the two directed 95th percentiles of
  boundary-to-boundary distances, × 2.152 nm. One empty side → the image diagonal (10,936 nm). Both empty → NaN.
* **D-08:** g-MAE only uses matched fibers, so a method can lower it by predicting fewer fibers. Recall is always
  returned next to it.

The fiber index trick used throughout: for a label L ≥ 2, ``L // 2`` is the same for axon X and myelin X+1. So
``labels // 2`` (with labels < 2 set to 0) is a fiber-instance map.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

PIXEL_NM = 2.152
DIAGONAL_NM = float(np.hypot(3008, 4096) * PIXEL_NM)  # 10,936 nm


def apply_ignore(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D-05: remove GT label-1 pixels from both maps."""
    ign = gt == 1
    p, g = pred.copy(), gt.copy()
    p[ign] = 0
    g[ign] = 0
    return p, g


def paired_fiber_map(labels: np.ndarray) -> np.ndarray:
    """Fiber-instance map (value = X // 2) restricted to paired IDs; everything else 0."""
    present = np.zeros(int(labels.max()) + 2, bool)
    present[np.unique(labels)] = True
    ids = np.arange(present.size)
    axon_ok = (ids >= 2) & (ids % 2 == 0) & np.r_[present[1:], False] & present
    paired = axon_ok | np.r_[False, axon_ok[:-1]]  # X and X+1 of every paired X
    lut = np.where(paired, ids // 2, 0).astype(np.int32)
    return lut[labels]


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = int(pred.sum()) + int(gt.sum())
    return float("nan") if s == 0 else 2.0 * int((pred & gt).sum()) / s


def _boundary(m: np.ndarray) -> np.ndarray:
    return m & ~ndi.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)


def hd95_nm(pred: np.ndarray, gt: np.ndarray, pixel_nm: float = PIXEL_NM) -> float:
    """D-07: max of the two directed 95th-percentile boundary distances, in nm."""
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() and not bg.any():
        return float("nan")
    if not bp.any() or not bg.any():
        return DIAGONAL_NM
    # crop to the joint bounding box (+ margin) to keep the distance transforms small
    rows = np.flatnonzero(bp.any(1) | bg.any(1))
    cols = np.flatnonzero(bp.any(0) | bg.any(0))
    sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    bp, bg = bp[sl], bg[sl]
    d_to_g = ndi.distance_transform_edt(~bg)
    d_to_p = ndi.distance_transform_edt(~bp)
    return float(max(np.percentile(d_to_g[bp], 95), np.percentile(d_to_p[bg], 95)) * pixel_nm)


@dataclass
class FiberTable:
    """Areas and pairwise intersections of paired fibers in a prediction and a ground truth."""

    pred_ids: np.ndarray            # fiber indices (X // 2) present in pred
    gt_ids: np.ndarray
    pred_area: dict                 # fiber index -> px
    gt_area: dict
    pred_axon_area: dict
    gt_axon_area: dict
    inter: dict                     # (gt_idx, pred_idx) -> px
    iou: np.ndarray                 # len(gt_ids) × len(pred_ids)
    pred_ignored: set = field(default_factory=set)  # pred fibers with ≥ 50% of their area on GT label 1


def fiber_table(pred: np.ndarray, gt: np.ndarray, ignore_frac: float = 0.5) -> FiberTable:
    """Build the fiber overlap table. ``pred``/``gt`` are raw label maps; D-05 is applied inside."""
    fp_raw = paired_fiber_map(pred)
    on_ign = gt == 1
    tot = np.bincount(fp_raw.ravel())
    ign = np.bincount(fp_raw[on_ign].ravel(), minlength=tot.size)
    pred_ignored = {int(i) for i in np.flatnonzero((tot > 0) & (ign >= ignore_frac * np.maximum(tot, 1))) if i > 0}

    p, g = apply_ignore(pred, gt)
    fp, fg = paired_fiber_map(p), paired_fiber_map(g)
    if pred_ignored:
        fp[np.isin(fp, list(pred_ignored))] = 0

    def areas(fm, lab):
        a = np.bincount(fm.ravel())
        ax = np.bincount(fm[(lab >= 2) & (lab % 2 == 0)].ravel(), minlength=a.size)
        ids = np.flatnonzero(a)
        ids = ids[ids > 0]
        return ids, {int(i): int(a[i]) for i in ids}, {int(i): int(ax[i]) for i in ids}

    pid, pa, pax = areas(fp, p)
    gid, ga, gax = areas(fg, g)
    both = (fp > 0) & (fg > 0)
    code = fg[both].astype(np.int64) * (int(fp.max()) + 1) + fp[both]
    u, c = np.unique(code, return_counts=True)
    inter = {(int(k // (int(fp.max()) + 1)), int(k % (int(fp.max()) + 1))): int(n) for k, n in zip(u, c, strict=True)}
    gi = {int(v): i for i, v in enumerate(gid)}
    pi = {int(v): i for i, v in enumerate(pid)}
    iou = np.zeros((len(gid), len(pid)))
    for (a, b), n in inter.items():
        iou[gi[a], pi[b]] = n / (ga[a] + pa[b] - n)
    return FiberTable(pid, gid, pa, ga, pax, gax, inter, iou, pred_ignored)


def match(table: FiberTable, iou_threshold: float = 0.5) -> list[tuple[int, int, float]]:
    """One-to-one matches (gt_idx, pred_idx, iou) with IoU > threshold. Unique for threshold ≥ 0.5."""
    if table.iou.size == 0:
        return []
    if iou_threshold >= 0.5:
        r, c = np.nonzero(table.iou > iou_threshold)
    else:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-table.iou)
        keep = table.iou[r, c] > iou_threshold
        r, c = r[keep], c[keep]
    return [(int(table.gt_ids[i]), int(table.pred_ids[j]), float(table.iou[i, j])) for i, j in zip(r, c, strict=True)]


def g_ratio(axon_area: int, fiber_area: int) -> float:
    return float(np.sqrt(axon_area / fiber_area)) if fiber_area > 0 else float("nan")


def evaluate_image(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Official metrics (+ recall/precision) for one image. ``pred``/``gt`` are uint16 challenge-encoded maps."""
    p, g = apply_ignore(pred, gt)
    pa, ga = (p >= 2) & (p % 2 == 0), (g >= 2) & (g % 2 == 0)
    pm, gm = (p >= 3) & (p % 2 == 1), (g >= 3) & (g % 2 == 1)
    t = fiber_table(pred, gt)
    m = match(t, iou_threshold)
    dg = [g_ratio(t.pred_axon_area[b], t.pred_area[b]) - g_ratio(t.gt_axon_area[a], t.gt_area[a]) for a, b, _ in m]
    n_gt, n_pred = len(t.gt_ids), len(t.pred_ids)
    return {
        "dice_axon": dice(pa, ga), "dice_myelin": dice(pm, gm),
        "hd95_axon_nm": hd95_nm(pa, ga), "hd95_myelin_nm": hd95_nm(pm, gm),
        "g_mae": float(np.mean(np.abs(dg))) if dg else float("nan"),
        "g_bias": float(np.mean(dg)) if dg else float("nan"),
        "n_gt": n_gt, "n_pred": n_pred, "n_pred_ignored": len(t.pred_ignored), "n_matched": len(m),
        "recall": len(m) / n_gt if n_gt else float("nan"), "precision": len(m) / n_pred if n_pred else float("nan"),
    }
