"""Training-label cleaning (T-02, D-21). Never applied to evaluation ground truth.

Two kinds of annotation are turned into ``ignore`` (label 1) for training:

* **Orphan IDs**: an axon ``X`` without its myelin ``X+1``, or a myelin without its axon. They cannot be scored
  as fibers, and teaching the model "axon with no ring" contradicts the encoding. (EDA: 4 + 43 = 47.)
* **Topology-defect fibers**: same definition as the EDA (scripts/eda/fiber_stats.py). There are 296 such fibers
  in 116 images. A fiber is a defect when any of these hold:
  - its myelin is split into several 8-connected components;
  - its myelin is not one ring (Euler number ≠ 0 with 8-connectivity);
  - its axon is not fully inside the filled myelin.

  These are mostly annotation slips. Supervising on them teaches shapes that contradict "axon inside its own ring".

Ignored pixels contribute no loss in nnU-Net, so the model is neither rewarded nor punished there.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number
from skimage.measure import label as cc_label

from .encoding import ARTIFACT


def fiber_defects(mask: np.ndarray) -> list[dict]:
    """Per paired fiber: the EDA topology checks. Returns one dict per axon ID with a paired myelin."""
    out = []
    objs = ndi.find_objects(mask)
    H, W = mask.shape
    for a in range(2, len(objs) + 1, 2):
        if objs[a - 1] is None or a + 1 > len(objs) or objs[a] is None:
            continue
        sa, sm = objs[a - 1], objs[a]
        r0 = max(min(sa[0].start, sm[0].start) - 2, 0)
        r1 = min(max(sa[0].stop, sm[0].stop) + 2, H)
        c0 = max(min(sa[1].start, sm[1].start) - 2, 0)
        c1 = min(max(sa[1].stop, sm[1].stop) + 2, W)
        sub = mask[r0:r1, c0:c1]
        A, M = sub == a, sub == a + 1
        n_cc_myelin = int(cc_label(M, connectivity=2).max())
        eul = int(euler_number(M, connectivity=2))
        enclosed = float((A & ndi.binary_fill_holes(M)).sum() / max(int(A.sum()), 1))
        out.append({"axon_id": a, "n_cc_myelin": n_cc_myelin, "myelin_euler": eul, "axon_enclosed": enclosed,
                        "defect": bool(n_cc_myelin > 1 or eul != 0 or enclosed < 1)})
    return out


def orphan_ids(mask: np.ndarray) -> list[int]:
    present = set(np.unique(mask).tolist())
    return sorted(x for x in present if x >= 2 and ((x % 2 == 0 and x + 1 not in present)
                                                    or (x % 2 == 1 and x - 1 not in present)))


def clean_training_mask(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Copy of ``mask`` with orphan IDs and topology-defect fibers set to ``ARTIFACT`` (ignore)."""
    out = mask.copy()
    orphans = orphan_ids(mask)
    defects = [d["axon_id"] for d in fiber_defects(mask) if d["defect"]]
    to_ignore = list(orphans) + [x for a in defects for x in (a, a + 1)]
    if to_ignore:
        out[np.isin(mask, to_ignore)] = ARTIFACT
    log = {"n_orphan_ids": len(orphans), "orphan_ids": orphans, "n_defect_fibers": len(defects), "defect_axon_ids": defects,
               "ignored_px": int((out == ARTIFACT).sum() - (mask == ARTIFACT).sum())}
    return out, log
