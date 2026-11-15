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
from skimage.segmentation import watershed

EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class ConversionReport:
    n_fibers: int  # components of axon ∪ myelin with both an axon and a myelin part
    n_no_axon: int  # myelin-only components (ring without axon) -> set to ignore
    n_no_myelin: int  # axon-only components -> set to ignore
    n_multi_axon: int  # fibers whose axon part has > 1 connected component (kept; flagged)
    n_border: int  # fibers touching the image border
    n_split: int = 0  # merged components split into separate fibers by their myelin holes


def split_axonmyelin(am: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ASTIH combined mask -> (axon, myelin). Axon = 255, myelin = 127/128 (TEM2 also has a few 126s); checked on
    TEM1, where the combined mask agrees exactly with the separate axon/myelin masks."""
    if am.ndim == 3:  # one TEM2 mask is stored as grey + alpha
        am = am[..., 0]
    return am >= 192, (am >= 64) & (am < 192)


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
        parts = [(a, m)]
        if ndi.label(a, structure=EIGHT)[1] > 1:
            parts = _split_by_holes(comp, a, m)
            if len(parts) > 1:
                rep.n_split += 1
            else:
                rep.n_multi_axon += 1  # one ring, fragmented axon: kept as one fiber
        for pa, pm in parts:
            k += 1
            if 2 * k + 1 > np.iinfo(np.uint16).max:
                raise OverflowError("too many fibers for uint16 encoding")
            out[sl][pa] = 2 * k
            out[sl][pm] = 2 * k + 1
    rep.n_fibers = k
    return out, rep


def _split_by_holes(comp: np.ndarray, a: np.ndarray, m: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """A component with several axon pieces is several fibers only if its myelin encloses several axon-containing
    holes (touching rings drawn without the 2-px gap, seen in TEM2). Each such hole seeds one fiber; the rest of the
    component goes to the nearest hole (watershed on the distance to the holes)."""
    holes = ndi.binary_fill_holes(m) & ~m
    hl, nh = ndi.label(holes, structure=EIGHT)
    with_axon = [j for j in range(1, nh + 1) if (a & (hl == j)).any()]
    if len(with_axon) < 2:
        return [(a, m)]
    markers = np.zeros(comp.shape, np.int32)
    for r, j in enumerate(with_axon, start=1):
        markers[hl == j] = r
    region = watershed(ndi.distance_transform_edt(markers == 0), markers, mask=comp)
    return [(a & (region == r), m & (region == r)) for r in range(1, len(with_axon) + 1)
            if (a & (region == r)).any() and (m & (region == r)).any()]


def axonpath_geojson_to_pairs(path, shape: tuple[int, int]) -> tuple[np.ndarray, ConversionReport]:
    """AxonPath QuPath GeoJSON (Fibre ⊃ InnerCylinder ⊃ Axon, plus the annotated ROI 'unclassified') -> uint16 pairs.

    Our axon includes the inner tongue (D-13), so axon = InnerCylinder and myelin = Fibre − InnerCylinder.
    Each InnerCylinder goes to the Fibre it overlaps most. Outside the ROI, and fibres without an inner cylinder,
    become 1 (ignore): only the ROI was annotated.
    """
    import json

    import cv2

    def raster(geom) -> tuple[tuple[slice, slice], np.ndarray] | None:
        """Rasterise inside the polygon's bounding box (full-size masks per polygon are far too slow)."""
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        pts = np.concatenate([np.asarray(ring, float) for poly in polys for ring in poly[:1]])
        y0, x0 = max(int(np.floor(pts[:, 1].min())), 0), max(int(np.floor(pts[:, 0].min())), 0)
        y1, x1 = min(int(np.ceil(pts[:, 1].max())) + 1, shape[0]), min(int(np.ceil(pts[:, 0].max())) + 1, shape[1])
        if y1 <= y0 or x1 <= x0:
            return None
        sl = (slice(y0, y1), slice(x0, x1))
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for poly in polys:
            for j, ring in enumerate(poly):
                r = np.round(np.asarray(ring, float) - [x0, y0]).astype(np.int32)
                cv2.fillPoly(m, [r], 1 if j == 0 else 0)  # scanline fill; exterior fills, holes clear
        return sl, m.astype(bool)

    with open(path) as fh:
        g = json.load(fh)
    feats = g["features"] if isinstance(g, dict) else g
    by: dict[str, list] = {"Fibre": [], "InnerCylinder": [], "unclassified": []}
    for f in feats:
        name = ((f.get("properties") or {}).get("classification") or {}).get("name", "unclassified")
        if name in by and f.get("geometry"):
            r = raster(f["geometry"])
            if r is not None:
                by[name].append(r)
    out = np.ones(shape, np.uint16)  # outside the ROI = ignore
    for sl, m in by["unclassified"]:
        out[sl][m] = 0
    fib_lab = np.zeros(shape, np.int32)
    for i, (sl, m) in enumerate(by["Fibre"], start=1):
        fib_lab[sl][m] = i
    inner = np.zeros(shape, np.int32)  # fibre id that owns each inner-cylinder pixel
    for sl, m in by["InnerCylinder"]:
        ids, cnt = np.unique(fib_lab[sl][m & (fib_lab[sl] > 0)], return_counts=True)
        if len(ids):
            inner[sl][m] = int(ids[np.argmax(cnt)])
    rep = ConversionReport(0, 0, 0, 0, 0)
    k = 0
    for i, (sl, _) in enumerate(by["Fibre"], start=1):
        fm = fib_lab[sl] == i
        if not fm.any():
            continue
        a = fm & (inner[sl] == i)
        if not a.any():
            rep.n_no_axon += 1
            out[sl][fm] = 1
            continue
        m = fm & ~a
        if not m.any():
            rep.n_no_myelin += 1
            out[sl][fm] = 1
            continue
        k += 1
        out[sl][a] = 2 * k
        out[sl][m] = 2 * k + 1
    rep.n_fibers = k
    return out, rep


def normalise_to_uint8(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Percentile stretch for float images (AxonPath raw TIFFs are float32 with negative values)."""
    a, b = np.percentile(img, [lo, hi])
    return (np.clip((img - a) / max(b - a, 1e-6), 0, 1) * 255).astype(np.uint8)
