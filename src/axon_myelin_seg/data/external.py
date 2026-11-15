"""Convert external axon/myelin datasets to the challenge's uint16 pair encoding (Phase 0b, label-mapping spec).

Encoding (as in ``encoding.py``): 0 background, 1 ignore, axon of fiber k = 2k, its myelin = 2k+1 (k >= 1).

ASTIH (TEM1 / TEM2) ground truth is semantic (binary axon and myelin PNGs, 0/255), but every fiber is separated
from its neighbours by >= 2 px (ASTIH §2.1.2), so a fiber instance is a connected component of axon ∪ myelin.
The spec, its checks and the by-eye test are documented in
``docs/research_notes/Phase 0b evidence/07_external_datasets.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class ConversionReport:
    n_fibers: int  # components of axon ∪ myelin with both an axon and a myelin part
    n_no_axon: int  # myelin-only components (ring without axon) -> set to ignore
    n_no_myelin: int  # axon-only components -> set to ignore
    n_multi_axon: int  # fibers whose axon part has > 1 connected component (kept; flagged)
    n_border: int  # fibers touching the image border


def astih_to_pairs(axon: np.ndarray, myelin: np.ndarray, border_to_ignore: bool = False) -> tuple[np.ndarray, ConversionReport]:
    """Semantic ASTIH masks -> uint16 paired instances.

    border_to_ignore: if True, fibers touching the image border become label 1, mirroring the challenge protocol
    (border-cut fibers = 1). Default False keeps ASTIH's own protocol (border fibers are scored).
    Components lacking an axon or a myelin part are written as label 1 (ignore), like orphans in our cleaner.
    """
    ax = axon > 0
    my = (myelin > 0) & ~ax
    lab, n = ndi.label(ax | my, structure=EIGHT)
    out = np.zeros(ax.shape, dtype=np.uint16)
    rep = ConversionReport(0, 0, 0, 0, 0)
    if n == 0:
        return out, rep
    objs = ndi.find_objects(lab)
    h, w = ax.shape
    k = 0
    for i, sl in enumerate(objs, start=1):
        comp = lab[sl] == i
        a = comp & ax[sl]
        m = comp & my[sl]
        touches = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w
        if not a.any() or not m.any():
            if not a.any():
                rep.n_no_axon += 1
            else:
                rep.n_no_myelin += 1
            out[sl][comp] = 1
            continue
        if touches:
            rep.n_border += 1
            if border_to_ignore:
                out[sl][comp] = 1
                continue
        if ndi.label(a, structure=EIGHT)[1] > 1:
            rep.n_multi_axon += 1
        k += 1
        if 2 * k + 1 > np.iinfo(np.uint16).max:
            raise OverflowError("too many fibers for uint16 encoding")
        out[sl][a] = 2 * k
        out[sl][m] = 2 * k + 1
    rep.n_fibers = k
    return out, rep
