import json
from pathlib import Path

import numpy as np
import pytest

from axon_myelin_seg.data.cleaning import clean_training_mask, fiber_defects, orphan_ids
from axon_myelin_seg.data.io import DATA_ROOT, list_samples, load_sample
from axon_myelin_seg.data.splits import check_split

REPO = Path(__file__).resolve().parents[1]
has_data = (DATA_ROOT / "train_samples").exists()


@pytest.mark.skipif(not has_data, reason="challenge data not present")
def test_loader_three_samples():
    names = list_samples()
    assert len(names) == 266
    for n in names[:3]:
        img, mask = load_sample(n)
        assert img.dtype == np.uint8 and mask.dtype == np.uint16 and img.shape == mask.shape == (3008, 4096)


def _ring(h=40, w=40, cy=20, cx=20, r_in=6, r_out=12):
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(yy - cy, xx - cx)
    m = np.zeros((h, w), np.uint16)
    m[(d >= r_in) & (d < r_out)] = 3
    m[d < r_in] = 2
    return m


def test_clean_fiber_is_not_a_defect():
    assert [d["defect"] for d in fiber_defects(_ring())] == [False]


def test_broken_ring_and_orphan_become_ignore():
    m = _ring()
    m[20, 26:33] = 0           # cut the ring: Euler 1 instead of 0 → defect
    m[0:3, 0:3] = 7            # orphan myelin (no axon 6)
    assert orphan_ids(m) == [7]
    out, log = clean_training_mask(m)
    assert log["n_defect_fibers"] == 1 and log["n_orphan_ids"] == 1
    assert set(np.unique(out).tolist()) <= {0, 1}


@pytest.mark.parametrize("fname", ["dev.json", "cv3.json"])
def test_frozen_splits_respect_groups(fname):
    d = json.loads((REPO / "configs" / "splits" / fname).read_text())
    folds = d.get("folds", [d])
    names = sorted(set(folds[0]["train"]) | set(folds[0]["val"]))
    assert len(names) == 266
    for f in folds:
        check_split(f, names)
    if "folds" in d:  # every image is validated exactly once across the 3 folds
        vals = [n for f in folds for n in f["val"]]
        assert sorted(vals) == names
