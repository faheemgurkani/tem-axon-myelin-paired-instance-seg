import sys, os, numpy as np, pandas as pd, tifffile, cv2
from scipy import ndimage as ndi
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D, OV, OUTD = sys.argv[1:4]
O = pd.read_csv(OV); O = O[(O.inliers >= 20) & O.scale.between(0.95, 1.05)]
def ov(img, m):
    rgb = np.stack([img]*3, -1).astype(float)/255
    for sel, c in (((m >= 2) & (m % 2 == 0), [0, .5, 0]), ((m >= 3) & (m % 2 == 1), [0, 0, .5]), ((m == 1), [.5, 0, 0])):
        rgb[sel] = rgb[sel]*.5 + np.array(c)
    return rgb
n = 0
for _, r in O.iterrows():
    ia = tifffile.imread(f"{D}/train_samples/{r.a}"); ma = tifffile.imread(f"{D}/train_instance_labels/{r.a}")
    ib = tifffile.imread(f"{D}/train_samples/{r.b}"); mb = tifffile.imread(f"{D}/train_instance_labels/{r.b}")
    H, W = ia.shape; a = np.radians(r.angle)
    M = np.array([[r.scale*np.cos(a), -r.scale*np.sin(a), r.tx], [r.scale*np.sin(a), r.scale*np.cos(a), r.ty]], np.float32)
    Mi = cv2.invertAffineTransform(M)
    ibw = cv2.warpAffine(ib, Mi, (W, H)); mbw = cv2.warpAffine(mb, Mi, (W, H), flags=cv2.INTER_NEAREST)
    valid = ndi.binary_erosion(cv2.warpAffine(np.ones_like(ib), Mi, (W, H), flags=cv2.INTER_NEAREST).astype(bool), iterations=20)
    fa = np.where(ma >= 2, ma - (ma % 2 == 1), 0)
    for k in np.unique(fa[valid]):
        if k == 0: continue
        reg = fa == k
        if not valid[reg].all(): continue
        v = mbw[reg]
        if (v == 0).mean() > .5 or (v == 1).mean() > .5:
            ys, xs = np.nonzero(reg); cy, cx = int(ys.mean()), int(xs.mean()); h = 450
            y0, x0 = max(cy-h, 0), max(cx-h, 0); y1, x1 = min(cy+h, H), min(cx+h, W)
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].imshow(ia[y0:y1, x0:x1], cmap='gray'); ax[0].set_title(f"{r.a} raw")
            ax[1].imshow(ov(ia[y0:y1, x0:x1], ma[y0:y1, x0:x1])); ax[1].set_title(f"{r.a} label")
            ax[2].imshow(ov(ibw[y0:y1, x0:x1], mbw[y0:y1, x0:x1])); ax[2].set_title(f"{r.b} label (aligned)")
            for x in ax: x.axis('off')
            plt.tight_layout(); plt.savefig(f"{OUTD}/disagree_{n:02d}.png", dpi=45); plt.close(); n += 1
print(n)
