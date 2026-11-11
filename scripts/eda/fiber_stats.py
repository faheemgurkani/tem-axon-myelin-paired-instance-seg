"""EDA over challenge train set: per-image and per-fiber tables. Usage: python -I eda.py <data_dir> <out_dir>"""
import sys, os, json
import numpy as np, pandas as pd, tifffile
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.measure import label as cc_label, regionprops, euler_number

PX = 2.152  # nm/px
D, OUT = sys.argv[1], sys.argv[2]
IMG, LAB = os.path.join(D, "train_samples"), os.path.join(D, "train_instance_labels")
B8 = np.ones((3, 3), bool)


def per_image(fn):
    img = tifffile.imread(os.path.join(IMG, fn))
    m = tifffile.imread(os.path.join(LAB, fn))
    H, W = m.shape
    ids = np.unique(m)
    ids_set = set(ids.tolist())
    pos = ids[ids >= 2]
    axons = [i for i in pos if i % 2 == 0]
    myel = [i for i in pos if i % 2 == 1]
    orphan_ax = [i for i in axons if i + 1 not in ids_set]
    orphan_my = [i for i in myel if i - 1 not in ids_set]
    labeled = m >= 2
    art = m == 1
    # intensity
    small = img[::4, ::4].astype(np.float32)
    ms = m[::4, ::4]
    rec = dict(
        file=fn, H=H, W=W, img_dtype=str(img.dtype), n_labels=len(pos), n_axon=len(axons), n_myelin=len(myel),
        n_orphan_axon=len(orphan_ax), n_orphan_myelin=len(orphan_my), max_id=int(ids.max()),
        ids_consecutive=bool(len(axons) and sorted(axons) == list(range(2, 2 * len(axons) + 1, 2))),
        frac_labeled=float(labeled.mean()), frac_artifact=float(art.mean()),
        frac_axon=float(((m >= 2) & (m % 2 == 0)).mean()), frac_myelin=float(((m >= 3) & (m % 2 == 1)).mean()),
        int_mean=float(small.mean()), int_std=float(small.std()), int_p1=float(np.percentile(small, 1)),
        int_p50=float(np.percentile(small, 50)), int_p99=float(np.percentile(small, 99)),
        frac_sat0=float((img == 0).mean()), frac_sat255=float((img == 255).mean()),
        int_myelin=float(small[(ms >= 3) & (ms % 2 == 1)].mean()) if ((ms >= 3) & (ms % 2 == 1)).any() else np.nan,
        int_axon=float(small[(ms >= 2) & (ms % 2 == 0)].mean()) if ((ms >= 2) & (ms % 2 == 0)).any() else np.nan,
        int_bg=float(small[ms == 0].mean()),
    )
    # dark-pixel coverage: fraction of "myelin-dark" pixels (<= image's mean myelin intensity) that are labeled
    if not np.isnan(rec["int_myelin"]):
        dark = small <= rec["int_myelin"]
        rec["frac_dark_unlabeled"] = float((dark & (ms == 0)).sum() / max(dark.sum(), 1))
    # border strips (scale bar / metadata band?) mean intensities
    rec["bottom_band_mean"] = float(img[-120:, :].mean())
    rec["bottom_band_unique"] = int(np.unique(img[-120:, -1200:]).size)
    rec["artifact_components"] = int(cc_label(art[::2, ::2]).max()) if art.any() else 0
    # thumbnail for duplicate detection
    thumb = img[::64, ::64].astype(np.float32)
    np.save(os.path.join(OUT, "thumbs", fn.replace(".tif", ".npy")), thumb)
    # intensity histogram
    rec["hist"] = np.bincount(img.ravel(), minlength=256).tolist()

    fibers = []
    for a in axons:
        my = a + 1
        if my not in ids_set:
            continue
        fib = (m == a) | (m == my)
        sl = ndi.find_objects(fib.astype(np.uint8))[0]
        r0, r1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, H)
        c0, c1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)
        sub = m[r0:r1, c0:c1]
        A = sub == a
        M = sub == my
        F = A | M
        aA, aM = int(A.sum()), int(M.sum())
        aF = aA + aM
        nA = int(cc_label(A, connectivity=2).max())
        nM = int(cc_label(M, connectivity=2).max())
        nF = int(cc_label(F, connectivity=2).max())
        # holes inside axon (unlabeled pixels fully enclosed by axon)
        A_filled = ndi.binary_fill_holes(A)
        axon_holes = int((A_filled & ~A).sum())
        # fiber filled: area enclosed by outer contour
        F_filled = ndi.binary_fill_holes(F)
        fib_holes = F_filled & ~F  # unlabeled inside the fiber (e.g. periaxonal gap / tongue)
        gap_px = int(fib_holes.sum())
        # containment: axon pixels adjacent (8-conn) to anything that is not this axon or this myelin
        ring = ndi.binary_dilation(A, B8) & ~A
        neigh = sub[ring]
        touch_bg = float(np.mean(neigh == 0)) if neigh.size else 0.0
        touch_other = float(np.mean((neigh >= 2) & (neigh != my))) if neigh.size else 0.0
        touch_my = float(np.mean(neigh == my)) if neigh.size else 0.0
        # myelin encloses axon? axon inside filled myelin ring
        M_filled = ndi.binary_fill_holes(M)
        enclosed = float((A & M_filled).sum() / max(aA, 1))
        # Euler number of myelin (1 - holes for single component)
        eul = int(euler_number(M, connectivity=2))
        # outer neighbours of the fiber
        oring = ndi.binary_dilation(F, B8) & ~F
        on = sub[oring]
        touch_fiber = float(np.mean(on >= 2)) if on.size else 0.0
        n_neigh = int(len(set(on[on >= 2].tolist())) // 1)
        border = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W
        pa = regionprops(A.astype(np.uint8))[0] if aA else None
        pf = regionprops(F_filled.astype(np.uint8))[0]
        d_ax = 2 * np.sqrt(aA / np.pi) * PX
        d_fib = 2 * np.sqrt(aF / np.pi) * PX
        d_fib_filled = 2 * np.sqrt(F_filled.sum() / np.pi) * PX
        fibers.append(dict(
            file=fn, axon_id=int(a), area_axon=aA, area_myelin=aM, area_fiber=aF, area_fiber_filled=int(F_filled.sum()),
            n_cc_axon=nA, n_cc_myelin=nM, n_cc_fiber=nF, axon_hole_px=axon_holes, fiber_gap_px=gap_px,
            axon_touch_bg=touch_bg, axon_touch_other=touch_other, axon_touch_own_myelin=touch_my,
            axon_enclosed=enclosed, myelin_euler=eul, fiber_touch_fiber=touch_fiber, n_neighbors=n_neigh,
            border=border, d_axon_nm=d_ax, d_fiber_nm=d_fib, d_fiber_filled_nm=d_fib_filled,
            g=np.sqrt(aA / aF) if aF else np.nan, g_filled=np.sqrt(aA / F_filled.sum()),
            thick_nm=(d_fib - d_ax) / 2,
            axon_ecc=pa.eccentricity if pa else np.nan, axon_solidity=pa.solidity if pa else np.nan,
            fiber_ecc=pf.eccentricity, fiber_solidity=pf.solidity,
            cy=(r0 + r1) / 2, cx=(c0 + c1) / 2,
        ))
    return rec, fibers


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "thumbs"), exist_ok=True)
    files = sorted(os.listdir(LAB))
    with Pool(8) as p:
        res = p.map(per_image, files, chunksize=4)
    imgs = [r for r, _ in res]
    hist = np.array([r.pop("hist") for r in imgs])
    np.save(os.path.join(OUT, "hist.npy"), hist)
    pd.DataFrame(imgs).to_csv(os.path.join(OUT, "images.csv"), index=False)
    pd.DataFrame([f for _, fs in res for f in fs]).to_csv(os.path.join(OUT, "fibers.csv"), index=False)
    print("done", len(imgs))
