"""List unlabeled ring-like candidates (dark closed bands around a hole) with location and features.
Usage: python -I unl_loc.py <data dir> <out csv>"""
import sys, os, numpy as np, tifffile, pandas as pd
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
D = sys.argv[1]
def f(fn):
    img = tifffile.imread(os.path.join(D, 'train_samples', fn))[::4, ::4].astype(np.float32)
    m = tifffile.imread(os.path.join(D, 'train_instance_labels', fn))[::4, ::4]
    sm = ndi.gaussian_filter(img, 1.5)
    my = (m >= 3) & (m % 2 == 1); bg = m == 0
    thr = (np.median(sm[my]) + np.median(sm[bg])) / 2
    dark = ndi.binary_opening(sm < thr, iterations=2)
    holes = ndi.binary_fill_holes(dark) & ~dark
    L = label(holes); out = []
    for p in regionprops(L):
        if not (1500 / 16 <= p.area <= 1.2e6 / 16) or p.solidity < 0.8: continue
        reg = L == p.label
        vals = m[reg]
        if (vals >= 2).mean() > 0.5 or (vals == 1).mean() > 0.5: continue
        ring = ndi.binary_dilation(reg, iterations=6) & ~reg
        out.append(dict(file=fn, cy=p.centroid[0] * 4, cx=p.centroid[1] * 4, area_px=p.area * 16,
                        d_nm=2 * np.sqrt(p.area * 16 / np.pi) * 2.152, solidity=p.solidity, ecc=p.eccentricity,
                        inner_int=float(img[reg].mean()), ring_int=float(img[ring].mean()),
                        ring_dark_frac=float(dark[ring].mean()), img_my=float(np.median(sm[my])), img_bg=float(np.median(sm[bg]))))
    return out
if __name__ == '__main__':
    fs = sorted(os.listdir(os.path.join(D, 'train_instance_labels')))
    with Pool(8) as p: r = p.map(f, fs)
    pd.DataFrame([x for y in r for x in y]).to_csv(sys.argv[2], index=False)
