"""Gallery helper: python -I gallery.py <data dir> <spec.csv: file,cy,cx,half,title> <out.png> [ncols]"""
import sys, numpy as np, pandas as pd, tifffile
from skimage.segmentation import find_boundaries
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D, SPEC, OUT = sys.argv[1:4]; nc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
S = pd.read_csv(SPEC); n = len(S); nr = int(np.ceil(n / nc))
fig, ax = plt.subplots(nr, nc * 2, figsize=(nc * 2 * 3, nr * 3))
ax = np.atleast_2d(ax)
cache = {}
for k, r in S.iterrows():
    if r.file not in cache:
        cache = {r.file: (tifffile.imread(f"{D}/train_samples/{r.file}"), tifffile.imread(f"{D}/train_instance_labels/{r.file}"))}
    img, m = cache[r.file]
    H, W = img.shape; h = int(r.half)
    y0, y1, x0, x1 = max(int(r.cy) - h, 0), min(int(r.cy) + h, H), max(int(r.cx) - h, 0), min(int(r.cx) + h, W)
    im, mm = img[y0:y1, x0:x1], m[y0:y1, x0:x1]
    rgb = np.stack([im] * 3, -1).astype(float) / 255
    b = find_boundaries(mm, mode='inner')
    rgb[b & (mm >= 2) & (mm % 2 == 0)] = [0, 1, 0]; rgb[b & (mm >= 3) & (mm % 2 == 1)] = [0, .5, 1]; rgb[b & (mm == 1)] = [1, 0, 0]
    i, j = divmod(k, nc)
    ax[i, 2 * j].imshow(im, cmap='gray', vmin=0, vmax=255); ax[i, 2 * j + 1].imshow(rgb)
    ax[i, 2 * j].set_title(str(r.title)[:40], fontsize=7)
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); plt.savefig(OUT, dpi=50)
