import numpy as np
import pytest
from scipy import ndimage as ndi

from axon_myelin_seg.data.io import DATA_ROOT, load_sample
from axon_myelin_seg.evaluation.diagnostics import diagnose_image
from axon_myelin_seg.evaluation.official import evaluate_image, hd95_nm
from axon_myelin_seg.evaluation.taxonomy import classify_errors

has_data = (DATA_ROOT / "train_samples").exists()


def disk_fiber(m, k, cy, cx, r_in, r_out):
    yy, xx = np.mgrid[: m.shape[0], : m.shape[1]]
    d = np.hypot(yy - cy, xx - cx)
    m[(d >= r_in) & (d < r_out)] = 2 * k + 1
    m[d < r_in] = 2 * k


def toy_gt():
    m = np.zeros((200, 300), np.uint16)
    disk_fiber(m, 1, 50, 50, 15, 25)
    disk_fiber(m, 2, 50, 100, 15, 25)     # touches fiber 1
    disk_fiber(m, 3, 150, 60, 15, 25)
    disk_fiber(m, 4, 150, 200, 15, 25)
    m[0:40, 250:300] = 1                  # label 1 (ignored) region
    return m


def test_gt_vs_gt_is_perfect():
    gt = toy_gt()
    r = evaluate_image(gt, gt)
    assert r["dice_axon"] == 1 and r["dice_myelin"] == 1
    assert r["hd95_axon_nm"] == 0 and r["hd95_myelin_nm"] == 0
    assert r["g_mae"] == 0 and r["recall"] == 1 and r["precision"] == 1
    d = diagnose_image(gt, gt)
    assert d["f1@0.5"] == 1 and d["pq"] == 1 and d["n_merges"] == 0 and d["n_splits"] == 0
    assert d["pairing_correct_rate"] == 1


def test_hd95_known_shift():
    a = np.zeros((100, 100), bool)
    a[20:60, 20:60] = True
    b = np.roll(a, 3, axis=1)
    assert hd95_nm(a, b, pixel_nm=1.0) == pytest.approx(3.0)


def test_prediction_on_label1_is_ignored():
    gt = toy_gt()
    pred = gt.copy()
    pred[pred == 1] = 0
    disk_fiber(pred, 9, 18, 275, 6, 12)   # inside the label-1 box → not FP
    d = diagnose_image(pred, gt)
    assert d["fp@0.5"] == 0
    assert classify_errors(pred, gt)["counts"]["pred_on_label1"] == 1


def test_taxonomy_classes():
    gt = toy_gt()
    pred = np.zeros_like(gt)
    # merge: one predicted fiber covering GT 1 and GT 2
    disk_fiber(pred, 1, 50, 75, 30, 50)
    # split: GT 3 cut into two halves, each its own fiber
    disk_fiber(pred, 3, 150, 60, 15, 25)
    half = np.zeros_like(gt, bool)
    half[:, 60:] = True
    sel = np.isin(pred, [6, 7]) & half
    pred[sel] = pred[sel] + 4                       # fiber 5 (10/11) = right half of GT 3
    # GT 4 matched exactly
    pred[np.isin(gt, [8, 9])] = gt[np.isin(gt, [8, 9])]
    # an unlabelled real fiber in background, with a candidate hole under its axon
    disk_fiber(pred, 7, 100, 260, 10, 18)
    cands = np.zeros(gt.shape, np.int32)
    yy, xx = np.mgrid[:200, :300]
    cands[np.hypot(yy - 100, xx - 260) < 10] = 1
    # a fiber on pure background with no candidate
    disk_fiber(pred, 8, 185, 120, 5, 9)
    r = classify_errors(pred, gt, candidates=cands)
    cls = sorted(r["pred"].values())
    assert "merge" in cls and "unlabelled_real" in cls and "other_bg" in cls
    assert "split_piece" in cls                       # the smaller half; the larger half matches GT 3
    gtc = r["gt"]
    assert gtc[1] == "merged" and gtc[2] == "merged"
    assert 3 not in gtc and 4 not in gtc               # matched
    assert diagnose_image(pred, gt)["n_splits"] == 1 and diagnose_image(pred, gt)["n_merges"] == 1


def test_gap_class():
    gt = np.zeros((200, 200), np.uint16)
    # three touching fibers around a central gap at (100, 100)
    # touching fibers (r_out 23, centres 46 px apart) leave a gap of radius ~3.6 px at the centroid
    for k, (cy, cx) in enumerate([(73.4, 100), (113.3, 77), (113.3, 123)], start=1):
        disk_fiber(gt, k, cy, cx, 10, 23)
    pred = gt.copy()
    yy, xx = np.mgrid[:200, :200]
    pred[(np.hypot(yy - 100, xx - 100) < 5) & (gt == 0)] = 8
    assert (pred == 8).sum() > 10
    ring = ndi.binary_dilation(pred == 8, iterations=3) & (pred != 8)
    pred[ring] = 9                                   # the "ring" is the neighbours' myelin
    r = classify_errors(pred, gt)
    assert "gap" in r["pred"].values()


@pytest.mark.skipif(not has_data, reason="challenge data not present")
def test_one_pixel_axon_erosion_moves_g_by_about_0p005():
    _, gt = load_sample("sample_0001")
    pred = gt.copy()
    ax = (gt >= 2) & (gt % 2 == 0)
    rim = ax & ~ndi.binary_erosion(ax)
    pred[rim] = pred[rim] + 1                        # axon boundary pixels → own myelin
    r = evaluate_image(pred, gt)
    assert 0.002 < r["g_mae"] < 0.01 and r["g_bias"] < 0
