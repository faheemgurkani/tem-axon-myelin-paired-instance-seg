"""Shared semantic → paired-instance step (T-09, D-11). Every semantic baseline goes through this one function, so
baselines differ only in their pixel predictions, not in how pixels become fibers.

Algorithm (marker-controlled watershed):
1. **Seeds** = 8-connected components of the predicted axon mask, with holes filled. Components smaller than
   ``min_axon_px`` (50 px ≈ 0.23 µm²) are dropped as speckle.
   Seeds come from whole components, not centroids. AxonDeepSeg seeds from axon centroids, and a centroid can fall
   outside a concave axon (report A).
2. **Landscape** = −EDT of the fiber foreground (axon ∪ myelin). Touching fibers meet where their sheaths are
   thinnest, which is the valley between two EDT peaks.
3. **Mask** = axon ∪ myelin. Myelin that no seed can reach (a ring with no predicted axon) stays unassigned and is
   dropped. This is a deliberate rule: a ring without an axon is not a fiber in the challenge encoding.
4. Fiber k: axon = seed k, myelin = (watershed region k) ∩ predicted myelin. Fibers with an empty ring are dropped.
   Output uses the challenge encoding: axon 2k, myelin 2k+1.

**Known weakness (result R1, T-09):** where two sheaths touch, nothing in the semantic map says where one ends and
the next begins. The watershed line is a geometric guess. This module therefore cannot be better than the GT
semantics allow, and running it on *GT* semantics measures exactly that loss.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

B8 = np.ones((3, 3), bool)


def semantic_to_paired_instances(axon: np.ndarray, myelin: np.ndarray, min_axon_px: int = 50,
                                 threshold: float = 0.5) -> np.ndarray:
    """``axon``/``myelin``: boolean masks or probabilities (thresholded at ``threshold``). Returns uint16 labels."""
    if axon.dtype != bool:
        axon = axon > threshold
    if myelin.dtype != bool:
        myelin = myelin > threshold
    myelin = myelin & ~axon
    seeds_mask = ndi.binary_fill_holes(axon)
    seeds, n = ndi.label(seeds_mask, structure=B8)
    if n == 0:
        return np.zeros(axon.shape, np.uint16)
    sizes = np.bincount(seeds.ravel())
    small = sizes < min_axon_px
    small[0] = False
    seeds[small[seeds]] = 0
    fg = seeds_mask | myelin
    dist = ndi.distance_transform_edt(fg)
    regions = watershed(-dist, markers=seeds, mask=fg)
    ax_k = np.where(seeds > 0, seeds, 0)
    my_k = np.where(myelin & (seeds == 0), regions, 0)
    # keep fibers with a non-empty ring; renumber 1..K
    has_ring = np.bincount(my_k.ravel(), minlength=n + 1) > 0
    has_axon = np.bincount(ax_k.ravel(), minlength=n + 1) > 0
    keep = has_ring & has_axon
    keep[0] = False
    new = np.zeros(n + 1, np.int64)
    new[keep] = np.arange(1, int(keep.sum()) + 1)
    if keep.sum() > 32766:
        raise ValueError("too many fibers for uint16 encoding")
    out = np.zeros(axon.shape, np.uint16)
    a, m = new[ax_k], new[my_k]
    out[m > 0] = (2 * m[m > 0] + 1).astype(np.uint16)
    out[a > 0] = (2 * a[a > 0]).astype(np.uint16)
    return out
