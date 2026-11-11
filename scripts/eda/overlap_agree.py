"""For overlapping image pairs: refine alignment at full res, then compare the two annotations of the same tissue.

Usage: python -I overlap_agree.py <train-dataset dir> <overlap.csv> <out csv> [min_inliers]
"""
import sys, os
import numpy as np, pandas as pd, tifffile, cv2
from scipy import ndimage as ndi

D, OV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MIN = int(sys.argv[4]) if len(sys.argv) > 4 else 20
O = pd.read_csv(OV)
O = O[(O.inliers >= MIN) & (O.scale.between(0.95, 1.05))]


def load(fn):
    return (tifffile.imread(os.path.join(D, "train_samples", fn)),
            tifffile.imread(os.path.join(D, "train_instance_labels", fn)))


def fibers(m):
    """fiber id map: axon id k (even) for both axon and its myelin."""
    f = np.where(m >= 2, m - (m % 2 == 1), 0)
    return f


rows = []
for _, r in O.iterrows():
    ia, ma = load(r.a)
    ib, mb = load(r.b)
    H, W = ia.shape
    # initial similarity from ORB (A->B), refine with ECC on 1/2 res
    ang = np.radians(r.angle)
    M = np.array([[r.scale * np.cos(ang), -r.scale * np.sin(ang), r.tx],
                  [r.scale * np.sin(ang), r.scale * np.cos(ang), r.ty]], np.float32)
    s = 2
    Ms = M.copy()
    Ms[:, 2] /= s
    try:
        _, Ms = cv2.findTransformECC(ib[::s, ::s].astype(np.float32), ia[::s, ::s].astype(np.float32), Ms,
                                     cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                     None, 5)
        refined = True
    except cv2.error:
        refined = False
    M = Ms.copy()
    M[:, 2] *= s
    # warp B into A's frame: inverse map
    Minv = cv2.invertAffineTransform(M)
    ib_w = cv2.warpAffine(ib, Minv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpAffine(np.ones_like(ib), Minv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
    valid = ndi.binary_erosion(valid, iterations=20)
    mb_w = cv2.warpAffine(mb, Minv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    ov = valid
    if ov.sum() < 1e5:
        continue
    ncc = float(np.corrcoef(ia[ov].astype(float), ib_w[ov].astype(float))[0, 1])
    # pixel-level agreement on overlap, excluding pixels that are label 1 in either
    ign = (ma == 1) | (mb_w == 1)
    use = ov & ~ign
    def cls(m):
        return np.where(m >= 2, np.where(m % 2 == 0, 1, 2), 0)
    ca, cb = cls(ma), cls(mb_w)
    dice = {}
    for k, nm in ((1, "axon"), (2, "myelin")):
        x, y = (ca == k) & use, (cb == k) & use
        dice[nm] = float(2 * (x & y).sum() / max(x.sum() + y.sum(), 1))
    # fiber-level: fibers fully inside the overlap in A, how are they labelled in B?
    fa, fb = fibers(ma), fibers(mb_w)
    stat = dict(matched=0, b_artifact=0, b_background=0, b_other=0, ious=[], dg=[])
    for k in np.unique(fa[ov]):
        if k == 0:
            continue
        reg = fa == k
        if not ov[reg].all():
            continue
        vals = mb_w[reg]
        if (vals == 1).mean() > 0.5:
            stat["b_artifact"] += 1
            continue
        if (vals == 0).mean() > 0.5:
            stat["b_background"] += 1
            continue
        kb = np.bincount(fb[reg][fb[reg] > 0]).argmax() if (fb[reg] > 0).any() else 0
        if kb == 0:
            stat["b_other"] += 1
            continue
        regb = fb == kb
        iou = (reg & regb).sum() / (reg | regb).sum()
        if iou < 0.5:
            stat["b_other"] += 1
            continue
        stat["matched"] += 1
        stat["ious"].append(iou)
        ga = np.sqrt((ma == k).sum() / reg.sum())
        gb = np.sqrt((mb_w == kb).sum() / regb.sum())
        stat["dg"].append(ga - gb)
    rows.append(dict(a=r.a, b=r.b, inliers=r.inliers, refined=refined, overlap_frac=float(ov.mean()), ncc=ncc,
                     dice_axon=dice["axon"], dice_myelin=dice["myelin"],
                     n_fibers_a=stat["matched"] + stat["b_artifact"] + stat["b_background"] + stat["b_other"],
                     matched=stat["matched"], b_artifact=stat["b_artifact"], b_background=stat["b_background"],
                     b_other=stat["b_other"], fiber_iou_med=float(np.median(stat["ious"])) if stat["ious"] else np.nan,
                     dg_mae=float(np.mean(np.abs(stat["dg"]))) if stat["dg"] else np.nan,
                     dg_bias=float(np.mean(stat["dg"])) if stat["dg"] else np.nan))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
