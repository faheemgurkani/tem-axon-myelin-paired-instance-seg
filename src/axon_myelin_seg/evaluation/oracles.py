"""Oracle experiments (T-13, D-14). Each oracle fixes exactly one error source using the GT and leaves the rest of
the prediction untouched. The gain over the baseline is the cost of that error source.

* **O-sel** (oracle selection): delete predicted fibers that the taxonomy calls ``unlabelled_real``. These are
  real fibers the annotators skipped, so the model is "right" but scored as FP.
  - The *broad* version also deletes ``other_bg``: on background, no ring candidate.
  - Fibers on GT label 1 are already ignored by the evaluator, so this oracle does not need to touch them.
* **O-sep** (oracle separation / grouping): keep the model's *pixels* (its axon/myelin semantics) but group them
  with the GT fiber instances.
  - Every predicted-foreground pixel inside GT fiber k is assigned to fiber k. Its compartment (axon vs myelin)
    stays as predicted.
  - Predicted fibers *outside* GT fibers keep the model's own instances. So selection FPs stay, and so do
    boundary errors and missed pixels.
  - A model instance that lies mostly (≥ 50% of its area) inside GT fibers was a GT fiber's prediction, so its
    outside remnant is dropped.
  - What changes: merges, splits, gap pseudo-fibers built from neighbours' myelin, and axon↔ring mis-pairing.
* **O-bnd** (oracle boundaries): every *matched* predicted fiber is replaced pixel-for-pixel by its GT fiber.
  Unmatched predictions are unchanged. What remains is selection + detection; the change in Dice, HD95 and g-MAE
  is the cost of boundary placement on the fibers the model found.
* **O-pair-GT** (= R1): GT semantics → the shared pairing rule. See scripts/run/day3_r1_pairing_on_gt.py.
"""

import numpy as np

from .official import fiber_table, match, paired_fiber_map


def o_sel(pred: np.ndarray, classes: dict, broad: bool = False) -> np.ndarray:
    drop = {k for k, c in classes["pred"].items() if c == "unlabelled_real" or (broad and c == "other_bg")}
    out = pred.copy()
    if drop:
        labs = [x for k in drop for x in (2 * k, 2 * k + 1)]
        out[np.isin(out, labs)] = 0
    return out


def o_sep(pred: np.ndarray, gt: np.ndarray, sem: np.ndarray | None = None) -> np.ndarray:
    """``sem``: 0 bg / 1 axon / 2 myelin. If None, it is derived from ``pred``'s encoding."""
    if sem is None:
        sem = np.where((pred >= 2) & (pred % 2 == 0), 1, np.where((pred >= 3) & (pred % 2 == 1), 2, 0))
    g_f = paired_fiber_map(gt)
    p_f = paired_fiber_map(pred)
    inside = g_f > 0
    # model instances that are mostly inside GT fibers are predictions *of* GT fibers → drop their remnant
    tot = np.bincount(p_f.ravel())
    ins = np.bincount(p_f[inside].ravel(), minlength=tot.size)
    mostly_in = (ins >= 0.5 * np.maximum(tot, 1)) & (tot > 0)
    mostly_in[0] = False
    out = np.zeros(pred.shape, np.int64)
    k0 = int(g_f.max()) + 1
    keep_out = (~inside) & (p_f > 0) & ~mostly_in[p_f]
    out[keep_out] = 2 * (p_f[keep_out] + k0) + (pred[keep_out] % 2)  # axon even, myelin odd, IDs offset
    ax = inside & (sem == 1)
    my = inside & (sem == 2)
    out[ax] = 2 * g_f[ax]
    out[my] = 2 * g_f[my] + 1
    # compact IDs into uint16 range
    ids = np.unique(out[out >= 2] // 2)
    lut = np.zeros(int(ids.max()) + 1 if ids.size else 1, np.int64)
    lut[ids] = np.arange(1, ids.size + 1)
    res = np.zeros(pred.shape, np.uint16)
    m = out >= 2
    res[m] = (2 * lut[out[m] // 2] + out[m] % 2).astype(np.uint16)
    return res


def o_bnd(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    t = fiber_table(pred, gt)
    m = match(t, 0.5)
    out = pred.copy()
    if not m:
        return out
    labs_p = [x for _, kp, _ in m for x in (2 * kp, 2 * kp + 1)]
    out[np.isin(out, labs_p)] = 0
    for kg, kp, _ in m:
        out[gt == 2 * kg] = 2 * kp
        out[gt == 2 * kg + 1] = 2 * kp + 1
    return out
