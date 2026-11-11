"""Find overlapping fields of view between images (incl. rotated / mirrored copies) with ORB + RANSAC.

Usage: python -I overlap.py <train-dataset dir> <out csv>
"""
import sys, os, itertools
import numpy as np, pandas as pd, tifffile, cv2
from multiprocessing import Pool

D, OUT = sys.argv[1], sys.argv[2]
S = 4  # downsample factor
files = sorted(os.listdir(os.path.join(D, "train_samples")))


def feats(fn):
    img = tifffile.imread(os.path.join(D, "train_samples", fn))[::S, ::S]
    img = cv2.createCLAHE(2.0, (8, 8)).apply(img)
    orb = cv2.ORB_create(1500)
    out = []
    for flip in (False, True):
        im = img[:, ::-1].copy() if flip else img
        kp, des = orb.detectAndCompute(im, None)
        out.append((np.float32([k.pt for k in kp]), des))
    return out


F = None


def init(f):
    global F
    F = f


def match(pair):
    i, j = pair
    best = (0, 0, None)
    pi, di = F[i][0]
    for flip in (0, 1):
        pj, dj = F[j][flip]
        if di is None or dj is None:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        ms = bf.knnMatch(di, dj, k=2)
        good = [a for a, b in (x for x in ms if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(good) < 12:
            continue
        src = pi[[g.queryIdx for g in good]]
        dst = pj[[g.trainIdx for g in good]]
        Mx, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4)
        n = int(inl.sum()) if inl is not None else 0
        if n > best[0]:
            best = (n, flip, Mx)
    n, flip, Mx = best
    if n == 0:
        return (i, j, 0, 0, np.nan, np.nan, np.nan, np.nan)
    sc = float(np.hypot(Mx[0, 0], Mx[1, 0]))
    ang = float(np.degrees(np.arctan2(Mx[1, 0], Mx[0, 0])))
    return (i, j, n, flip, sc, ang, float(Mx[0, 2] * S), float(Mx[1, 2] * S))


if __name__ == "__main__":
    with Pool(8) as p:
        f = p.map(feats, files)
    pairs = list(itertools.combinations(range(len(files)), 2))
    with Pool(8, initializer=init, initargs=(f,)) as p:
        r = p.map(match, pairs, chunksize=200)
    df = pd.DataFrame(r, columns=["i", "j", "inliers", "flip", "scale", "angle", "tx", "ty"])
    df["a"] = [files[k] for k in df.i]
    df["b"] = [files[k] for k in df.j]
    df.sort_values("inliers", ascending=False).to_csv(OUT, index=False)
    print(df.inliers.describe())
