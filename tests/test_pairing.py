import numpy as np

from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances


def fiber(m, k, cy, cx, r_in, r_out):
    yy, xx = np.mgrid[: m.shape[0], : m.shape[1]]
    d = np.hypot(yy - cy, xx - cx)
    m[(d >= r_in) & (d < r_out)] = 2 * k + 1
    m[d < r_in] = 2 * k


def test_two_touching_fibers_are_separated_and_paired():
    gt = np.zeros((100, 160), np.uint16)
    fiber(gt, 1, 50, 50, 12, 25)
    fiber(gt, 2, 50, 100, 12, 25)          # rings touch at x≈75
    out = semantic_to_paired_instances((gt >= 2) & (gt % 2 == 0), (gt >= 3) & (gt % 2 == 1))
    ids = sorted(set(np.unique(out).tolist()) - {0})
    assert ids == [2, 3, 4, 5]
    # each axon sits inside its own ring
    for a in (2, 4):
        assert (out[gt == 2 * (a // 2)] == a).all() or (out[gt == 2 * (3 - a // 2)] == a).all()


def test_ring_without_axon_and_speckle_are_dropped():
    sem = np.zeros((60, 60), np.uint16)
    fiber(sem, 1, 30, 30, 8, 16)
    myelin = (sem >= 3) & (sem % 2 == 1)
    axon = np.zeros_like(myelin)           # no axon at all
    axon[2:4, 2:4] = True                  # 4-px speckle, below min_axon_px
    out = semantic_to_paired_instances(axon, myelin)
    assert out.max() == 0
