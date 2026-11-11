"""Deep EDA: per-fiber label-quality features and per-image TEM image-quality features.

Usage: python -I deep.py <train-dataset dir> <out dir>
"""
import sys, os
import numpy as np, pandas as pd, tifffile
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.measure import regionprops, label as cc_label
from skimage.restoration import estimate_sigma

PX = 2.152
D, OUT = sys.argv[1], sys.argv[2]
OFFS = np.arange(-6, 7)  # signed-distance bins (px) for edge alignment


def edge_profile(grad, mask, region):
    """Mean gradient magnitude as a function of signed distance (px) to the boundary of `mask`, within `region`."""
    din = ndi.distance_transform_edt(mask)
    dout = ndi.distance_transform_edt(~mask)
    sd = np.where(mask, -(din - 0.5), dout - 0.5)  # negative inside
    b = np.rint(sd).astype(int)
    sel = region & (b >= OFFS[0]) & (b <= OFFS[-1])
    s = np.bincount(b[sel] - OFFS[0], grad[sel], minlength=len(OFFS))
    c = np.bincount(b[sel] - OFFS[0], minlength=len(OFFS))
    return s / np.maximum(c, 1)


def image_quality(img):
    f = img.astype(np.float32)
    h = f[::2, ::2]
    q = {}
    # sharpness: variance of Laplacian (full res) and high-frequency energy share
    q["lap_var"] = float(ndi.laplace(f[::1, ::1][:1500, :2000]).var())
    F = np.abs(np.fft.rfft2(h - h.mean())) ** 2
    fy = np.fft.fftfreq(h.shape[0])[:, None]
    fx = np.fft.rfftfreq(h.shape[1])[None, :]
    r = np.hypot(fy, fx)
    q["hf_share"] = float(F[r > 0.25].sum() / F[r > 0.01].sum())
    # periodic spike score (knife chatter / scan lines): max over annulus of (power / local median)
    logF = np.log(F + 1e-6)
    med = ndi.median_filter(logF, size=9)
    ann = (r > 0.02) & (r < 0.45)
    ex = (logF - med)[ann]
    q["fft_spike"] = float(np.percentile(ex, 99.99))
    # row/column banding
    q["row_band"] = float(np.std(ndi.uniform_filter1d(f.mean(1), 3) - ndi.uniform_filter1d(f.mean(1), 101)))
    q["col_band"] = float(np.std(ndi.uniform_filter1d(f.mean(0), 3) - ndi.uniform_filter1d(f.mean(0), 101)))
    # noise (wavelet MAD)
    q["noise_sigma"] = float(estimate_sigma(f[1000:1512, 1500:2012]))
    # illumination non-uniformity: quadratic surface fit on a heavily blurred 1/16 image
    s = ndi.gaussian_filter(f[::16, ::16], 12)
    yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    yy = yy / s.shape[0] - .5
    xx = xx / s.shape[1] - .5
    A = np.stack([np.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy], -1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(A, s.ravel(), rcond=None)
    surf = A @ coef
    q["illum_range_rel"] = float((surf.max() - surf.min()) / surf.mean())
    q["illum_lin_x"] = float(coef[1] / surf.mean())
    q["illum_lin_y"] = float(coef[2] / surf.mean())
    q["illum_quad"] = float((coef[3] + coef[4]) / surf.mean())  # <0 = bright centre (vignetting)
    # JPEG blockiness: diff at 8-px boundaries vs elsewhere
    dx = np.abs(np.diff(f, axis=1))
    cols = np.arange(dx.shape[1])
    q["block8"] = float(dx[:, cols % 8 == 7].mean() / dx[:, cols % 8 != 7].mean())
    # histogram comb (contrast stretch): empty grey levels inside [p0.5,p99.5]
    hist = np.bincount(img.ravel(), minlength=256)
    lo, hi = np.percentile(img, [0.5, 99.5]).astype(int)
    q["hist_empty_bins"] = int((hist[lo:hi + 1] == 0).sum())
    q["hist_span"] = int(hi - lo)
    q["hist_odd_even_ratio"] = float(hist[1::2].sum() / max(hist[0::2].sum(), 1))
    # dark precipitate specks (stain precipitate / dust): small very dark blobs
    thr = np.percentile(f, 0.2)
    sp = cc_label(f[::2, ::2] < thr)
    sizes = np.bincount(sp.ravel())[1:]
    q["dark_specks"] = int(((sizes >= 2) & (sizes <= 30)).sum())
    # corner / burned-in text check: saturated pixels in corners
    c = 200
    corners = [img[:c, :c], img[:c, -c:], img[-c:, :c], img[-c:, -c:]]
    q["corner_sat"] = float(np.mean([((x == 0) | (x == 255)).mean() for x in corners]))
    return q


def per_image(fn):
    img = tifffile.imread(os.path.join(D, "train_samples", fn))
    m = tifffile.imread(os.path.join(D, "train_instance_labels", fn))
    H, W = m.shape
    f = img.astype(np.float32)
    g = ndi.gaussian_gradient_magnitude(f, 1.5)
    q = image_quality(img)
    q["file"] = fn
    bgv = f[m == 0]
    my_all = (m >= 3) & (m % 2 == 1)
    q["img_bg_med"] = float(np.median(bgv))
    q["img_my_med"] = float(np.median(f[my_all])) if my_all.any() else np.nan
    q["fiber_density_per_um2"] = float(len([i for i in np.unique(m) if i >= 2 and i % 2 == 0]) /
                                       ((m != 1).sum() * (PX / 1000) ** 2))
    prof_out, prof_in, rows = [], [], []
    ids = [int(i) for i in np.unique(m) if i >= 2 and i % 2 == 0 and (i + 1) in set(np.unique(m).tolist())]
    objs = ndi.find_objects(m)
    for a in ids:
        sa, sm = objs[a - 1], objs[a]
        if sa is None or sm is None:
            continue
        r0 = max(min(sa[0].start, sm[0].start) - 10, 0)
        r1 = min(max(sa[0].stop, sm[0].stop) + 10, H)
        c0 = max(min(sa[1].start, sm[1].start) - 10, 0)
        c1 = min(max(sa[1].stop, sm[1].stop) + 10, W)
        sub, fi, gi = m[r0:r1, c0:c1], f[r0:r1, c0:c1], g[r0:r1, c0:c1]
        A, M = sub == a, sub == a + 1
        Fb = A | M
        if A.sum() < 50 or M.sum() < 50:
            continue
        # edge alignment: outer boundary evaluated away from other labelled fibers' interiors
        near = ndi.binary_dilation(Fb, iterations=7)
        po = edge_profile(gi, Fb, near & ((sub == 0) | Fb))
        pi = edge_profile(gi, A, ndi.binary_dilation(A, iterations=7) & Fb)
        prof_out.append(po)
        prof_in.append(pi)
        # myelin thickness along outer boundary: distance from outer boundary pixels to axon
        dA = ndi.distance_transform_edt(~A)
        ob = Fb & ~ndi.binary_erosion(Fb)
        t = dA[ob] * PX
        # intensities
        bg_ring = ndi.binary_dilation(Fb, iterations=12) & ~ndi.binary_dilation(Fb, iterations=3) & (sub == 0)
        my_int = fi[M]
        ax_int = fi[A]
        rpa = regionprops(A.astype(np.uint8))[0]
        rpf = regionprops(ndi.binary_fill_holes(Fb).astype(np.uint8))[0]
        rows.append(dict(
            file=fn, axon_id=a,
            out_off=float(OFFS[np.argmax(po)]), in_off=float(OFFS[np.argmax(pi)]),
            out_peak_ratio=float(po.max() / max(np.median(po), 1e-6)),
            thick_med_nm=float(np.median(t)), thick_cv=float(t.std() / max(t.mean(), 1e-6)),
            thick_min_nm=float(np.percentile(t, 2)), thick_max_nm=float(np.percentile(t, 98)),
            my_med=float(np.median(my_int)), ax_med=float(np.median(ax_int)),
            bg_med=float(np.median(fi[bg_ring])) if bg_ring.any() else np.nan,
            my_bright_frac=float((my_int > q["img_bg_med"]).mean()),
            ax_dark_frac=float((ax_int < q["img_my_med"]).mean()),
            g_eq=float(np.sqrt(A.sum() / Fb.sum())),
            g_minor=float(rpa.axis_minor_length / max(rpf.axis_minor_length, 1e-6)),
            g_feret=float(rpa.feret_diameter_max / max(rpf.feret_diameter_max, 1e-6)),
            g_thick=float((2 * np.sqrt(A.sum() / np.pi)) / (2 * np.sqrt(A.sum() / np.pi) + 2 * np.median(t) / PX)),
            ax_minor_nm=float(rpa.axis_minor_length * PX), ax_major_nm=float(rpa.axis_major_length * PX),
            ax_eq_nm=float(2 * np.sqrt(A.sum() / np.pi) * PX),
            fib_ecc=float(rpf.eccentricity), ax_ecc=float(rpa.eccentricity),
            ax_perim_ratio=float(rpa.perimeter / (2 * np.sqrt(np.pi * A.sum()))),  # 1 = circle; high = rough/irregular
            fib_perim_ratio=float(rpf.perimeter / (2 * np.sqrt(np.pi * rpf.area))),
        ))
    po = np.mean(prof_out, 0) if prof_out else np.full(len(OFFS), np.nan)
    pi = np.mean(prof_in, 0) if prof_in else np.full(len(OFFS), np.nan)
    q["img_out_off"] = float(OFFS[np.nanargmax(po)]) if prof_out else np.nan
    q["img_in_off"] = float(OFFS[np.nanargmax(pi)]) if prof_in else np.nan
    for k, o in enumerate(OFFS):
        q[f"po_{o}"] = float(po[k])
        q[f"pi_{o}"] = float(pi[k])
    return q, rows


if __name__ == "__main__":
    files = sorted(os.listdir(os.path.join(D, "train_instance_labels")))
    with Pool(8) as p:
        res = p.map(per_image, files, chunksize=2)
    pd.DataFrame([q for q, _ in res]).to_csv(os.path.join(OUT, "deep_images.csv"), index=False)
    pd.DataFrame([r for _, rs in res for r in rs]).to_csv(os.path.join(OUT, "deep_fibers.csv"), index=False)
    print("done")
